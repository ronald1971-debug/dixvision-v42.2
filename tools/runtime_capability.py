# Canonical DIX VISION runtime capability resolver — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ()
#
# Authority constraints (pinned by ``tests/test_runtime_capability.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :class:`RuntimeCapabilityMap` is a pure function of
#              the (topology, snapshot) pair: two maps with identical
#              inputs produce byte-identical digests.
#   * No top-level imports of :mod:`subprocess`, :mod:`time`,
#     :mod:`random`, :mod:`asyncio`, :mod:`socket`, :mod:`numpy`,
#     :mod:`torch`, :mod:`requests`.
"""Canonical runtime capability resolver (PR-RT-3).

This is the **resolver** half of the runtime topology authority chain.
PR-RT-1 introduced the declared topology
(:class:`tools.runtime_topology.RuntimeTopology`); PR-RT-2 introduced
the activation registry that pins which declared nodes are
actually-running (:class:`tools.runtime_activation.ActivationSnapshot`).
This module composes those two surfaces into the answer the rest of
the system needs:

    "Who is actually providing capability X right now?"

The answer separates declared from active so detached loops, dormant
buses, and available-but-unwired optimizers surface as ``dormant``
rather than as healthy providers.

Determinism contract (INV-15):

* :meth:`RuntimeCapabilityMap.digest` is a BLAKE2b-128 hex over the
  canonical sorted-key JSON serialization that includes the
  topology digest, the snapshot digest, and the resolver version.
  Two maps with the same (topology, snapshot) pair always produce
  byte-identical digests.
* Capability resolutions sort their declared / active / dormant
  members deterministically; no dict iteration order leaks into
  any output.
* :class:`DependencyGraphResolver` walks the DEPENDS_ON edges of
  the topology in a strictly deterministic depth-first order and
  detects cycles defensively.
* No global mutable state; no clocks; no PRNG; no file-system reads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from tools.runtime_activation import ActivationSnapshot
from tools.runtime_topology import (
    EdgeRelation,
    LifecycleState,
    RuntimeTopology,
)

CAPABILITY_VERSION: Final[str] = "v1.0-RT3"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()

MAX_CAPABILITY_LEN: Final[int] = 128
MAX_DEPENDENCY_CHAIN_LEN: Final[int] = 256


class CapabilityError(ValueError):
    """Raised when a capability resolver input is malformed."""


class CapabilityResolutionError(RuntimeError):
    """Raised when the dependency graph contains a cycle or otherwise
    cannot be resolved deterministically."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityResolution:
    """A frozen breakdown of a single capability resolution.

    ``declared`` is every node that advertises the capability in the
    declared topology. ``active`` is the subset of ``declared`` that
    is actually-running (STARTED / HEALTHY / DEGRADED in the snapshot).
    ``dormant`` is the subset of ``declared`` that is in any other
    state OR is registered but not running.

    The three tuples are mutually consistent: ``set(declared)`` equals
    ``set(active) | set(dormant) | set(unregistered)``. ``unregistered``
    surfaces declared providers that never called
    :meth:`tools.runtime_activation.RuntimeActivationRegistry.register`
    at boot — the most dangerous form of silent drift.
    """

    capability: str
    declared: tuple[str, ...]
    active: tuple[str, ...]
    dormant: tuple[str, ...]
    unregistered: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str):
            raise CapabilityError(f"capability must be a str, got {type(self.capability).__name__}")
        if not self.capability:
            raise CapabilityError("capability must be non-empty")
        if len(self.capability) > MAX_CAPABILITY_LEN:
            raise CapabilityError(
                f"capability length {len(self.capability)} exceeds "
                f"MAX_CAPABILITY_LEN={MAX_CAPABILITY_LEN}"
            )
        for field_name in ("declared", "active", "dormant", "unregistered"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise CapabilityError(f"{field_name} must be a tuple, got {type(value).__name__}")
            for entry in value:
                if not isinstance(entry, str):
                    raise CapabilityError(
                        f"{field_name} entries must be str, got {type(entry).__name__}"
                    )
        declared_set = frozenset(self.declared)
        active_set = frozenset(self.active)
        dormant_set = frozenset(self.dormant)
        unregistered_set = frozenset(self.unregistered)
        if active_set & dormant_set:
            raise CapabilityError(
                "active and dormant must be disjoint, overlap: "
                f"{sorted(active_set & dormant_set)!r}"
            )
        if active_set & unregistered_set:
            raise CapabilityError(
                "active and unregistered must be disjoint, overlap: "
                f"{sorted(active_set & unregistered_set)!r}"
            )
        if dormant_set & unregistered_set:
            raise CapabilityError(
                "dormant and unregistered must be disjoint, overlap: "
                f"{sorted(dormant_set & unregistered_set)!r}"
            )
        combined = active_set | dormant_set | unregistered_set
        if combined != declared_set:
            raise CapabilityError(
                "declared must equal active | dormant | unregistered, "
                f"declared={sorted(declared_set)!r} "
                f"combined={sorted(combined)!r}"
            )
        for field_name in ("declared", "active", "dormant", "unregistered"):
            value = getattr(self, field_name)
            sorted_value = tuple(sorted(value))
            if sorted_value != value:
                object.__setattr__(self, field_name, sorted_value)

    def canonical(self) -> dict[str, object]:
        return {
            "active": list(self.active),
            "capability": self.capability,
            "declared": list(self.declared),
            "dormant": list(self.dormant),
            "unregistered": list(self.unregistered),
        }

    def is_resolved(self) -> bool:
        """Return ``True`` when at least one active provider exists."""

        return len(self.active) > 0

    def is_dormant(self) -> bool:
        """Return ``True`` when the capability is declared but no
        active provider exists."""

        return len(self.declared) > 0 and len(self.active) == 0

    def is_missing(self) -> bool:
        """Return ``True`` when no provider declares the capability."""

        return len(self.declared) == 0

    def has_unregistered_providers(self) -> bool:
        """Return ``True`` when at least one declared provider never
        called ``register`` against the activation registry."""

        return len(self.unregistered) > 0


@dataclass(frozen=True, slots=True)
class DependencyMissingLink:
    """A frozen description of the first missing or dormant ancestor
    along a DEPENDS_ON chain.

    ``start_node_id`` is the node the resolver was asked about;
    ``missing_node_id`` is the first ancestor in the chain that is
    either not active or not registered; ``state`` is the lifecycle
    state of ``missing_node_id`` or ``None`` if it never registered;
    ``chain`` is the deterministic depth-first walk that surfaced it.
    """

    start_node_id: str
    missing_node_id: str
    state: LifecycleState | None
    chain: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("start_node_id", "missing_node_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise CapabilityError(f"{field_name} must be a str, got {type(value).__name__}")
            if not value:
                raise CapabilityError(f"{field_name} must be non-empty")
        if self.state is not None and not isinstance(self.state, LifecycleState):
            raise CapabilityError(
                f"state must be a LifecycleState or None, got {type(self.state).__name__}"
            )
        if not isinstance(self.chain, tuple):
            raise CapabilityError(f"chain must be a tuple, got {type(self.chain).__name__}")
        if not self.chain:
            raise CapabilityError("chain must be non-empty")
        for entry in self.chain:
            if not isinstance(entry, str):
                raise CapabilityError(f"chain entries must be str, got {type(entry).__name__}")
        if self.chain[0] != self.start_node_id:
            raise CapabilityError(
                f"chain[0] must equal start_node_id, got "
                f"chain[0]={self.chain[0]!r} "
                f"start_node_id={self.start_node_id!r}"
            )
        if self.chain[-1] != self.missing_node_id:
            raise CapabilityError(
                "chain[-1] must equal missing_node_id, got "
                f"chain[-1]={self.chain[-1]!r} "
                f"missing_node_id={self.missing_node_id!r}"
            )

    def canonical(self) -> dict[str, object]:
        return {
            "chain": list(self.chain),
            "missing_node_id": self.missing_node_id,
            "start_node_id": self.start_node_id,
            "state": None if self.state is None else self.state.value,
        }


# ---------------------------------------------------------------------------
# RuntimeCapabilityMap
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityMap:
    """Composes a declared topology with an activation snapshot to
    answer capability resolution queries.

    The map is a pure value object: two maps constructed from the
    same (topology, snapshot) pair produce byte-identical digests
    and identical query results.
    """

    topology: RuntimeTopology
    snapshot: ActivationSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.topology, RuntimeTopology):
            raise CapabilityError(
                f"topology must be a RuntimeTopology, got {type(self.topology).__name__}"
            )
        if not isinstance(self.snapshot, ActivationSnapshot):
            raise CapabilityError(
                f"snapshot must be an ActivationSnapshot, got {type(self.snapshot).__name__}"
            )

    def canonical(self) -> dict[str, object]:
        return {
            "snapshot_digest": self.snapshot.digest(),
            "topology_digest": self.topology.digest(),
            "version": CAPABILITY_VERSION,
        }

    def digest(self) -> str:
        """Return a BLAKE2b-128 hex digest over the canonical
        sorted-key JSON serialization."""

        payload = json.dumps(
            self.canonical(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _validate_capability(self, capability: str) -> None:
        if not isinstance(capability, str):
            raise CapabilityError(f"capability must be a str, got {type(capability).__name__}")
        if not capability:
            raise CapabilityError("capability must be non-empty")
        if len(capability) > MAX_CAPABILITY_LEN:
            raise CapabilityError(
                f"capability length {len(capability)} exceeds "
                f"MAX_CAPABILITY_LEN={MAX_CAPABILITY_LEN}"
            )

    def who_provides(self, capability: str) -> tuple[str, ...]:
        """Return the sorted node_ids that *actively* provide the
        capability — the answer to "who provides X right now?".

        Dormant, unregistered, or detached providers are not in the
        result. Use :meth:`resolve` to get the full breakdown when
        you also need to see the silent-drift slice.
        """

        self._validate_capability(capability)
        declared = self.topology.providers_of(capability)
        active = self.snapshot.active_node_ids()
        return tuple(sorted(n.node_id for n in declared if n.node_id in active))

    def dormant_providers(self, capability: str) -> tuple[str, ...]:
        """Return the sorted node_ids that declare the capability
        but are not active (DECLARED / WIRED / DORMANT / STOPPED)."""

        self._validate_capability(capability)
        declared = self.topology.providers_of(capability)
        active = self.snapshot.active_node_ids()
        all_states = {nid for nid, _ in self.snapshot.states}
        return tuple(
            sorted(
                n.node_id for n in declared if n.node_id in all_states and n.node_id not in active
            )
        )

    def unregistered_providers(self, capability: str) -> tuple[str, ...]:
        """Return the sorted node_ids that declare the capability
        but never registered against the activation registry —
        the most dangerous form of silent drift."""

        self._validate_capability(capability)
        declared = self.topology.providers_of(capability)
        all_states = {nid for nid, _ in self.snapshot.states}
        return tuple(sorted(n.node_id for n in declared if n.node_id not in all_states))

    def resolve(self, capability: str) -> CapabilityResolution:
        """Return the full breakdown of a capability resolution."""

        self._validate_capability(capability)
        declared_nodes = self.topology.providers_of(capability)
        declared_ids = tuple(sorted(n.node_id for n in declared_nodes))
        active_set = self.snapshot.active_node_ids()
        all_state_ids = {nid for nid, _ in self.snapshot.states}
        active = tuple(sorted(nid for nid in declared_ids if nid in active_set))
        dormant = tuple(
            sorted(nid for nid in declared_ids if nid in all_state_ids and nid not in active_set)
        )
        unregistered = tuple(sorted(nid for nid in declared_ids if nid not in all_state_ids))
        return CapabilityResolution(
            capability=capability,
            declared=declared_ids,
            active=active,
            dormant=dormant,
            unregistered=unregistered,
        )

    def all_capabilities(self) -> tuple[str, ...]:
        """Return every capability tag declared by any node, sorted
        and deduplicated."""

        capabilities: set[str] = set()
        for node in self.topology.nodes:
            capabilities.update(node.capabilities)
        return tuple(sorted(capabilities))

    def unresolved_capabilities(self) -> tuple[str, ...]:
        """Return every declared capability with no active provider.

        These are the silent-drift capabilities PR-RT-5's
        total-validation invariant will require an explicit
        ``DECLARED_BUT_DORMANT`` ledger admission for.
        """

        return tuple(
            sorted(cap for cap in self.all_capabilities() if len(self.who_provides(cap)) == 0)
        )

    def resolution_count(self) -> int:
        return len(self.all_capabilities())


# ---------------------------------------------------------------------------
# DependencyGraphResolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DependencyGraphResolver:
    """Walks the DEPENDS_ON edges of the declared topology and reports
    the first missing or dormant ancestor.

    The resolver is a pure value object: two resolvers constructed
    from the same (topology, snapshot) pair always return identical
    answers and emit byte-identical digests.
    """

    topology: RuntimeTopology
    snapshot: ActivationSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.topology, RuntimeTopology):
            raise CapabilityError(
                f"topology must be a RuntimeTopology, got {type(self.topology).__name__}"
            )
        if not isinstance(self.snapshot, ActivationSnapshot):
            raise CapabilityError(
                f"snapshot must be an ActivationSnapshot, got {type(self.snapshot).__name__}"
            )

    def canonical(self) -> dict[str, object]:
        return {
            "snapshot_digest": self.snapshot.digest(),
            "topology_digest": self.topology.digest(),
            "version": CAPABILITY_VERSION,
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.canonical(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _validate_node_id(self, node_id: str) -> None:
        if not isinstance(node_id, str):
            raise CapabilityError(f"node_id must be a str, got {type(node_id).__name__}")
        if not node_id:
            raise CapabilityError("node_id must be non-empty")

    def _depends_on_targets(self, node_id: str) -> tuple[str, ...]:
        edges = self.topology.edges_from(node_id)
        return tuple(sorted(e.target_id for e in edges if e.relation is EdgeRelation.DEPENDS_ON))

    def dependency_chain(self, start_node_id: str) -> tuple[str, ...]:
        """Return the deterministic depth-first DEPENDS_ON walk from
        ``start_node_id`` (inclusive).

        Cycles are detected and raised as
        :class:`CapabilityResolutionError` so the walk never loops
        silently. The chain length is bounded by
        ``MAX_DEPENDENCY_CHAIN_LEN``.
        """

        self._validate_node_id(start_node_id)
        if self.topology.find_node(start_node_id) is None:
            raise CapabilityError(f"node_id {start_node_id!r} is not declared in topology")
        chain: list[str] = []
        visited: set[str] = set()

        def _walk(node: str, path: frozenset[str]) -> None:
            if node in path:
                raise CapabilityResolutionError(
                    f"DEPENDS_ON cycle detected at node {node!r} while "
                    f"walking from {start_node_id!r}; chain={chain!r}"
                )
            if node in visited:
                return
            visited.add(node)
            chain.append(node)
            if len(chain) > MAX_DEPENDENCY_CHAIN_LEN:
                raise CapabilityResolutionError(
                    "dependency chain exceeded "
                    f"MAX_DEPENDENCY_CHAIN_LEN={MAX_DEPENDENCY_CHAIN_LEN} "
                    f"while walking from {start_node_id!r}"
                )
            extended = path | {node}
            for target in self._depends_on_targets(node):
                _walk(target, extended)

        _walk(start_node_id, frozenset())
        return tuple(chain)

    def is_reachable(self, node_id: str) -> bool:
        """Return ``True`` when ``node_id`` is active AND every
        DEPENDS_ON ancestor is also active.

        A reachable node has no dormant or unregistered ancestor in
        its declared dependency chain.
        """

        self._validate_node_id(node_id)
        if self.topology.find_node(node_id) is None:
            raise CapabilityError(f"node_id {node_id!r} is not declared in topology")
        active = self.snapshot.active_node_ids()
        chain = self.dependency_chain(node_id)
        return all(member in active for member in chain)

    def missing_link(self, start_node_id: str) -> DependencyMissingLink | None:
        """Return the first DEPENDS_ON ancestor of ``start_node_id``
        that is not active, or ``None`` when the whole chain is
        reachable.

        The walk follows the same deterministic depth-first order as
        :meth:`dependency_chain`; the returned ``chain`` is the prefix
        from ``start_node_id`` to (and including) the missing ancestor.
        """

        self._validate_node_id(start_node_id)
        if self.topology.find_node(start_node_id) is None:
            raise CapabilityError(f"node_id {start_node_id!r} is not declared in topology")
        active = self.snapshot.active_node_ids()
        all_states = {nid for nid, _ in self.snapshot.states}
        chain: list[str] = []
        visited: set[str] = set()
        result: list[DependencyMissingLink] = []

        def _walk(node: str, path: frozenset[str]) -> bool:
            if node in path:
                raise CapabilityResolutionError(
                    f"DEPENDS_ON cycle detected at node {node!r} while "
                    f"walking from {start_node_id!r}; chain={chain!r}"
                )
            if node in visited:
                return False
            visited.add(node)
            chain.append(node)
            if len(chain) > MAX_DEPENDENCY_CHAIN_LEN:
                raise CapabilityResolutionError(
                    "dependency chain exceeded "
                    f"MAX_DEPENDENCY_CHAIN_LEN={MAX_DEPENDENCY_CHAIN_LEN} "
                    f"while walking from {start_node_id!r}"
                )
            if node not in active:
                state = self.snapshot.state_of(node) if node in all_states else None
                result.append(
                    DependencyMissingLink(
                        start_node_id=start_node_id,
                        missing_node_id=node,
                        state=state,
                        chain=tuple(chain),
                    )
                )
                return True
            extended = path | {node}
            for target in self._depends_on_targets(node):
                if _walk(target, extended):
                    return True
            return False

        _walk(start_node_id, frozenset())
        return result[0] if result else None

    def unreachable_nodes(self) -> tuple[str, ...]:
        """Return the sorted node_ids whose DEPENDS_ON chain contains
        any dormant or unregistered ancestor (themselves included)."""

        out: list[str] = []
        for node in self.topology.nodes:
            try:
                if not self.is_reachable(node.node_id):
                    out.append(node.node_id)
            except CapabilityResolutionError:
                out.append(node.node_id)
        return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def build_capability_map(
    topology: RuntimeTopology,
    snapshot: ActivationSnapshot,
) -> RuntimeCapabilityMap:
    """Construct a :class:`RuntimeCapabilityMap` from a topology and
    snapshot. Provided as a small convenience over the constructor."""

    return RuntimeCapabilityMap(topology=topology, snapshot=snapshot)


def build_dependency_resolver(
    topology: RuntimeTopology,
    snapshot: ActivationSnapshot,
) -> DependencyGraphResolver:
    """Construct a :class:`DependencyGraphResolver` from a topology
    and snapshot."""

    return DependencyGraphResolver(topology=topology, snapshot=snapshot)


# ---------------------------------------------------------------------------
# Lazy seam factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityFactory:
    """A lazy-seam factory the rest of the system can swap to
    introduce alternative back-ends (e.g. graph-database-backed
    resolvers) without changing call sites.

    The default factory returns the canonical pure-stdlib resolvers.
    """

    version: str = CAPABILITY_VERSION

    def capability_map(
        self,
        topology: RuntimeTopology,
        snapshot: ActivationSnapshot,
    ) -> RuntimeCapabilityMap:
        return build_capability_map(topology=topology, snapshot=snapshot)

    def dependency_resolver(
        self,
        topology: RuntimeTopology,
        snapshot: ActivationSnapshot,
    ) -> DependencyGraphResolver:
        return build_dependency_resolver(topology=topology, snapshot=snapshot)


def enable_runtime_capability_factory(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeCapabilityFactory:
    """Construct the canonical stdlib :class:`RuntimeCapabilityFactory`.

    ``overrides`` is accepted for forward-compatibility with a future
    alternative backend; the current implementation only knows the
    canonical version and rejects any other key.
    """

    if overrides is None:
        return RuntimeCapabilityFactory()
    if not isinstance(overrides, Mapping):
        raise CapabilityError(f"overrides must be a Mapping, got {type(overrides).__name__}")
    allowed = {"version"}
    unknown = set(overrides) - allowed
    if unknown:
        raise CapabilityError(
            f"unknown override keys: {sorted(unknown)!r}; allowed: {sorted(allowed)!r}"
        )
    version = overrides.get("version", CAPABILITY_VERSION)
    if not isinstance(version, str):
        raise CapabilityError(f"version override must be a str, got {type(version).__name__}")
    return RuntimeCapabilityFactory(version=version)
