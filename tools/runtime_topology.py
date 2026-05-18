# Canonical DIX VISION runtime topology authority — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ()
#
# Authority constraints (pinned by ``tests/test_runtime_topology.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :class:`RuntimeTopology` is a pure function of
#              its :class:`RuntimeNode` and :class:`RuntimeEdge`
#              tuples: three independent constructions produce
#              byte-identical digests.
#   * No top-level imports of :mod:`subprocess`, :mod:`time`,
#     :mod:`random`, :mod:`asyncio`, :mod:`socket`, :mod:`numpy`,
#     :mod:`torch`, :mod:`requests`.
"""Canonical runtime topology authority (PR-RT-1).

This is the **declared** half of the runtime topology authority chain.
Subsequent PR-RT-2 introduces an :class:`ActiveLoopRegistry` and
:class:`EngineActivationRegistry` that pin the *actually-running*
subgraph; PR-RT-3 introduces a :class:`RuntimeCapabilityMap` that
resolves declared capability tags to the first reachable active
provider; PR-RT-4 wires the registries into ``ui/server.py`` at boot;
PR-RT-5 adds a ``tools/total_validation.py`` rule that fails CI when
the declared topology drifts from the active topology without an
explicit ``DECLARED_BUT_DORMANT`` ledger admission.

The motivation is the analysis-of-record finding "runtime activation
is implicit": several engines, loops, and event buses are *declared*
(present in code, importable, instantiable) but never *invoked* on
the hot path. That creates false-positive health reporting and silent
topology drift between the declared and active architectures. This
module is the first load-bearing surface that lets the rest of the
system answer "what is ACTUALLY active right now?" deterministically.

Determinism contract (INV-15):

* :meth:`RuntimeTopology.digest` is a BLAKE2b-128 hex over the
  canonical sorted-key JSON serialization. Two topologies with the
  same nodes and edges (regardless of construction order) produce
  byte-identical digests.
* Node and edge sequences are sorted before serialization; tag
  frozensets are sorted; no dict iteration order leaks into the
  output.
* No global mutable state; no clocks; no PRNG; no file-system reads.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

TOPOLOGY_VERSION: Final[str] = "v1.0-RT1"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()

MAX_NODE_ID_LEN: Final[int] = 128
MAX_CAPABILITY_LEN: Final[int] = 128
MAX_CAPABILITIES_PER_NODE: Final[int] = 64
MAX_NODES_PER_TOPOLOGY: Final[int] = 4096
MAX_EDGES_PER_TOPOLOGY: Final[int] = 16_384
MAX_VERSION_LEN: Final[int] = 32
MAX_NOTE_LEN: Final[int] = 256

NODE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
CAPABILITY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")
VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]+$")


class TopologyError(ValueError):
    """Raised when a :class:`RuntimeTopology` is malformed."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeKind(Enum):
    """The kind of a runtime node — what *role* the node plays.

    These are intentionally coarse; the topology authority only needs
    enough granularity to reason about wiring, not to act as a full
    type registry. Finer typing happens inside each engine's own
    protocol definitions.
    """

    ENGINE = "ENGINE"
    LOOP = "LOOP"
    BUS = "BUS"
    SENSOR = "SENSOR"
    ADAPTER = "ADAPTER"
    REGISTRY = "REGISTRY"
    ROUTE = "ROUTE"
    GATE = "GATE"
    POLICY = "POLICY"
    STORE = "STORE"


class NodeTier(Enum):
    """The tier of a runtime node — used to enforce B-tier authority
    lint at the topology level.

    ``T0`` is the safety-critical kernel; ``T1`` is the engine layer;
    ``T2`` is the cognitive and intelligence depth layer; ``UI`` is
    everything operator-facing.
    """

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    UI = "UI"


class EdgeRelation(Enum):
    """The relation between two nodes in the topology graph.

    These are interpreted by :class:`RuntimeCapabilityMap` (PR-RT-3)
    when resolving capability queries; the topology authority itself
    only stores the typed relation.
    """

    PRODUCES = "PRODUCES"
    CONSUMES = "CONSUMES"
    OWNS = "OWNS"
    GATES = "GATES"
    DEPENDS_ON = "DEPENDS_ON"
    PROJECTS = "PROJECTS"


class LifecycleState(Enum):
    """The lifecycle state of a runtime node.

    The legal transitions (enforced by PR-RT-2) are:

    .. code-block::

        DECLARED -> WIRED   -> STARTED -> HEALTHY -> DEGRADED -> STOPPED
                                                  -> STOPPED

    Any other transition is an audited
    ``RUNTIME_ACTIVATION_VIOLATION`` ledger row in PR-RT-2.
    """

    DECLARED = "DECLARED"
    WIRED = "WIRED"
    STARTED = "STARTED"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"
    DORMANT = "DORMANT"


_LEGAL_TRANSITIONS: Final[Mapping[LifecycleState, frozenset[LifecycleState]]] = {
    LifecycleState.DECLARED: frozenset({LifecycleState.WIRED, LifecycleState.DORMANT}),
    LifecycleState.WIRED: frozenset({LifecycleState.STARTED, LifecycleState.DORMANT}),
    LifecycleState.STARTED: frozenset(
        {LifecycleState.HEALTHY, LifecycleState.DEGRADED, LifecycleState.STOPPED}
    ),
    LifecycleState.HEALTHY: frozenset({LifecycleState.DEGRADED, LifecycleState.STOPPED}),
    LifecycleState.DEGRADED: frozenset({LifecycleState.HEALTHY, LifecycleState.STOPPED}),
    LifecycleState.STOPPED: frozenset({LifecycleState.WIRED}),
    LifecycleState.DORMANT: frozenset({LifecycleState.WIRED}),
}


def is_legal_transition(src: LifecycleState, dst: LifecycleState) -> bool:
    """Return ``True`` iff ``src -> dst`` is a legal lifecycle move.

    The transition table is exposed as a pure function so PR-RT-2 can
    use it inside the activation registry without reaching into
    module-private state.
    """

    if not isinstance(src, LifecycleState):  # pragma: no cover - defensive
        raise TopologyError(f"src is not a LifecycleState: {src!r}")
    if not isinstance(dst, LifecycleState):  # pragma: no cover - defensive
        raise TopologyError(f"dst is not a LifecycleState: {dst!r}")
    return dst in _LEGAL_TRANSITIONS[src]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeNode:
    """One node in the declared runtime topology.

    ``node_id`` is a dotted lowercase identifier (e.g.
    ``"execution_engine.gate"``); ``kind`` selects from
    :class:`NodeKind`; ``tier`` selects from :class:`NodeTier`;
    ``declared_version`` is a short string the registry pins on first
    registration; ``capabilities`` is a frozenset of capability tags
    (e.g. ``"learning.closed_loop"``, ``"execution.gate"``,
    ``"sensor.hazard.policy_drift"``) that PR-RT-3 uses to answer
    capability queries.
    """

    node_id: str
    kind: NodeKind
    tier: NodeTier
    declared_version: str
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str):
            raise TopologyError(f"node_id must be a str, got {type(self.node_id).__name__}")
        if not self.node_id:
            raise TopologyError("node_id must not be empty")
        if len(self.node_id) > MAX_NODE_ID_LEN:
            raise TopologyError(
                f"node_id length {len(self.node_id)} exceeds MAX_NODE_ID_LEN={MAX_NODE_ID_LEN}"
            )
        if not NODE_ID_PATTERN.match(self.node_id):
            raise TopologyError(f"node_id {self.node_id!r} does not match NODE_ID_PATTERN")
        if not isinstance(self.kind, NodeKind):
            raise TopologyError(f"kind must be a NodeKind, got {type(self.kind).__name__}")
        if not isinstance(self.tier, NodeTier):
            raise TopologyError(f"tier must be a NodeTier, got {type(self.tier).__name__}")
        if not isinstance(self.declared_version, str):
            raise TopologyError(
                f"declared_version must be a str, got {type(self.declared_version).__name__}"
            )
        if not self.declared_version:
            raise TopologyError("declared_version must not be empty")
        if len(self.declared_version) > MAX_VERSION_LEN:
            raise TopologyError(
                f"declared_version length {len(self.declared_version)} "
                f"exceeds MAX_VERSION_LEN={MAX_VERSION_LEN}"
            )
        if not VERSION_PATTERN.match(self.declared_version):
            raise TopologyError(
                f"declared_version {self.declared_version!r} does not match VERSION_PATTERN"
            )
        if not isinstance(self.capabilities, frozenset):
            raise TopologyError(
                f"capabilities must be a frozenset, got {type(self.capabilities).__name__}"
            )
        if len(self.capabilities) > MAX_CAPABILITIES_PER_NODE:
            raise TopologyError(
                f"capabilities length {len(self.capabilities)} exceeds "
                f"MAX_CAPABILITIES_PER_NODE={MAX_CAPABILITIES_PER_NODE}"
            )
        for capability in self.capabilities:
            if not isinstance(capability, str):
                raise TopologyError(
                    f"every capability must be a str, got {type(capability).__name__}"
                )
            if not capability:
                raise TopologyError("capability must not be empty")
            if len(capability) > MAX_CAPABILITY_LEN:
                raise TopologyError(
                    f"capability length {len(capability)} exceeds "
                    f"MAX_CAPABILITY_LEN={MAX_CAPABILITY_LEN}"
                )
            if not CAPABILITY_PATTERN.match(capability):
                raise TopologyError(f"capability {capability!r} does not match CAPABILITY_PATTERN")

    def canonical(self) -> dict[str, object]:
        """Return a canonical dict representation for digesting.

        The mapping has sorted keys; the capabilities list is sorted;
        no other ordering decisions leak into the output.
        """

        return {
            "capabilities": sorted(self.capabilities),
            "declared_version": self.declared_version,
            "kind": self.kind.value,
            "node_id": self.node_id,
            "tier": self.tier.value,
        }


@dataclass(frozen=True, slots=True)
class RuntimeEdge:
    """One typed edge between two declared :class:`RuntimeNode` nodes.

    Edges are directional. ``relation`` selects from
    :class:`EdgeRelation`; ``note`` is an optional short human-readable
    string used only for operator-facing rendering and never read by
    the topology authority itself.
    """

    source_id: str
    target_id: str
    relation: EdgeRelation
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str):
            raise TopologyError(f"source_id must be a str, got {type(self.source_id).__name__}")
        if not NODE_ID_PATTERN.match(self.source_id):
            raise TopologyError(f"source_id {self.source_id!r} does not match NODE_ID_PATTERN")
        if not isinstance(self.target_id, str):
            raise TopologyError(f"target_id must be a str, got {type(self.target_id).__name__}")
        if not NODE_ID_PATTERN.match(self.target_id):
            raise TopologyError(f"target_id {self.target_id!r} does not match NODE_ID_PATTERN")
        if self.source_id == self.target_id:
            raise TopologyError(
                f"self-loop is not allowed (source_id == target_id == {self.source_id!r})"
            )
        if not isinstance(self.relation, EdgeRelation):
            raise TopologyError(
                f"relation must be an EdgeRelation, got {type(self.relation).__name__}"
            )
        if not isinstance(self.note, str):
            raise TopologyError(f"note must be a str, got {type(self.note).__name__}")
        if len(self.note) > MAX_NOTE_LEN:
            raise TopologyError(f"note length {len(self.note)} exceeds MAX_NOTE_LEN={MAX_NOTE_LEN}")

    def canonical(self) -> dict[str, object]:
        return {
            "note": self.note,
            "relation": self.relation.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
        }


@dataclass(frozen=True, slots=True)
class RuntimeTopology:
    """The declared runtime topology: a frozen set of nodes and edges
    with a BLAKE2b-128 digest over the canonical sorted-key JSON
    serialization.

    Constructing a :class:`RuntimeTopology` validates:

    * every ``node_id`` is unique
    * every ``RuntimeEdge.source_id`` and ``RuntimeEdge.target_id``
      refers to a node that is present in ``nodes``
    * the (source_id, target_id, relation) triple is unique (parallel
      edges with the same relation are not allowed)
    * caps on node and edge counts hold

    The constructor sorts ``nodes`` by ``node_id`` and ``edges`` by
    ``(source_id, target_id, relation.value)`` and stores them as
    immutable tuples so the digest is reproducible regardless of
    construction order.
    """

    nodes: tuple[RuntimeNode, ...]
    edges: tuple[RuntimeEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple):
            raise TopologyError(f"nodes must be a tuple, got {type(self.nodes).__name__}")
        if not isinstance(self.edges, tuple):
            raise TopologyError(f"edges must be a tuple, got {type(self.edges).__name__}")
        if len(self.nodes) > MAX_NODES_PER_TOPOLOGY:
            raise TopologyError(
                f"node count {len(self.nodes)} exceeds "
                f"MAX_NODES_PER_TOPOLOGY={MAX_NODES_PER_TOPOLOGY}"
            )
        if len(self.edges) > MAX_EDGES_PER_TOPOLOGY:
            raise TopologyError(
                f"edge count {len(self.edges)} exceeds "
                f"MAX_EDGES_PER_TOPOLOGY={MAX_EDGES_PER_TOPOLOGY}"
            )

        seen_node_ids: set[str] = set()
        for node in self.nodes:
            if not isinstance(node, RuntimeNode):
                raise TopologyError(
                    f"every entry of nodes must be a RuntimeNode, got {type(node).__name__}"
                )
            if node.node_id in seen_node_ids:
                raise TopologyError(f"duplicate node_id {node.node_id!r}")
            seen_node_ids.add(node.node_id)

        sorted_nodes = tuple(sorted(self.nodes, key=lambda n: n.node_id))
        if sorted_nodes != self.nodes:
            object.__setattr__(self, "nodes", sorted_nodes)

        seen_edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if not isinstance(edge, RuntimeEdge):
                raise TopologyError(
                    f"every entry of edges must be a RuntimeEdge, got {type(edge).__name__}"
                )
            if edge.source_id not in seen_node_ids:
                raise TopologyError(
                    f"edge source_id {edge.source_id!r} does not match any node_id in nodes"
                )
            if edge.target_id not in seen_node_ids:
                raise TopologyError(
                    f"edge target_id {edge.target_id!r} does not match any node_id in nodes"
                )
            key = (edge.source_id, edge.target_id, edge.relation.value)
            if key in seen_edge_keys:
                raise TopologyError(
                    f"duplicate edge (source_id={edge.source_id!r}, "
                    f"target_id={edge.target_id!r}, "
                    f"relation={edge.relation.value!r})"
                )
            seen_edge_keys.add(key)

        sorted_edges = tuple(
            sorted(
                self.edges,
                key=lambda e: (
                    e.source_id,
                    e.target_id,
                    e.relation.value,
                ),
            )
        )
        if sorted_edges != self.edges:
            object.__setattr__(self, "edges", sorted_edges)

    def canonical(self) -> dict[str, object]:
        return {
            "edges": [edge.canonical() for edge in self.edges],
            "nodes": [node.canonical() for node in self.nodes],
            "version": TOPOLOGY_VERSION,
        }

    def digest(self) -> str:
        """Return a BLAKE2b-128 hex digest over the canonical
        sorted-key JSON serialization.

        Two topologies with the same nodes and edges always produce
        byte-identical digests; this is the INV-15 anchor.
        """

        payload = json.dumps(
            self.canonical(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def find_node(self, node_id: str) -> RuntimeNode | None:
        """Return the node matching ``node_id`` or ``None``.

        Lookup is O(N) by design — topologies are small (< 4096 nodes)
        and we never want to introduce a mutable cache in a frozen
        dataclass.
        """

        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def nodes_by_tier(self, tier: NodeTier) -> tuple[RuntimeNode, ...]:
        if not isinstance(tier, NodeTier):
            raise TopologyError(f"tier must be a NodeTier, got {type(tier).__name__}")
        return tuple(n for n in self.nodes if n.tier is tier)

    def nodes_by_kind(self, kind: NodeKind) -> tuple[RuntimeNode, ...]:
        if not isinstance(kind, NodeKind):
            raise TopologyError(f"kind must be a NodeKind, got {type(kind).__name__}")
        return tuple(n for n in self.nodes if n.kind is kind)

    def edges_from(self, source_id: str) -> tuple[RuntimeEdge, ...]:
        return tuple(e for e in self.edges if e.source_id == source_id)

    def edges_to(self, target_id: str) -> tuple[RuntimeEdge, ...]:
        return tuple(e for e in self.edges if e.target_id == target_id)

    def providers_of(self, capability: str) -> tuple[RuntimeNode, ...]:
        """Return every declared node that advertises ``capability``.

        This is the read-side lookup PR-RT-3's capability map uses to
        resolve "who provides X". The topology authority only stores
        declarations; PR-RT-3 cross-references against the live
        activation registry to filter to actually-active providers.
        """

        if not isinstance(capability, str):
            raise TopologyError(f"capability must be a str, got {type(capability).__name__}")
        return tuple(n for n in self.nodes if capability in n.capabilities)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def build_topology(
    nodes: Iterable[RuntimeNode],
    edges: Iterable[RuntimeEdge],
) -> RuntimeTopology:
    """Build a :class:`RuntimeTopology` from any iterables.

    Provided as a small convenience over the constructor so call sites
    can pass generators or lists without worrying about the tuple
    coercion.
    """

    return RuntimeTopology(
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def empty_topology() -> RuntimeTopology:
    """Return the empty topology — useful as an INV-15 anchor in
    tests."""

    return RuntimeTopology(nodes=(), edges=())


# ---------------------------------------------------------------------------
# Lazy seam factory
# ---------------------------------------------------------------------------


def enable_runtime_topology_factory(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeTopologyFactory:
    """Return a :class:`RuntimeTopologyFactory` configured against the
    declared backend.

    The stdlib backend is the production default; the factory exists
    so PR-RT-2 / PR-RT-3 can compose richer activation backends behind
    the same shape without rewriting call sites. There is no vendor
    backend currently — this is intentionally a stdlib-only authority
    because the topology graph is a pure declarative surface and any
    dependency would just add deserialisation paths without changing
    the contract.
    """

    if overrides is not None and not isinstance(overrides, Mapping):
        raise TopologyError(f"overrides must be a Mapping or None, got {type(overrides).__name__}")
    config = dict(overrides) if overrides else {}
    return RuntimeTopologyFactory(backend="stdlib", config=config)


@dataclass(frozen=True, slots=True)
class RuntimeTopologyFactory:
    """The lazy seam factory.

    The factory is the read-side handle PR-RT-2 will consume; it is
    intentionally minimal so the seam can evolve without churning
    PR-RT-2's API.
    """

    backend: str
    config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TopologyError(f"backend must be a str, got {type(self.backend).__name__}")
        if self.backend not in {"stdlib"}:
            raise TopologyError(f"backend {self.backend!r} is not supported (expected 'stdlib')")

    def build(
        self,
        nodes: Sequence[RuntimeNode],
        edges: Sequence[RuntimeEdge],
    ) -> RuntimeTopology:
        return build_topology(nodes, edges)
