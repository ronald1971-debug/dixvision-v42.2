"""HarnessRuntimeRegistrar — boot wiring for the Runtime Topology Authority.

PR-RT-4 of the chain (PR-RT-1 declared topology / PR-RT-2 active
registry / PR-RT-3 capability resolver). This module makes the
declared topology **load-bearing** at harness boot: every engine,
loop, registry, gate, policy, and feed that ``_State`` builds is
matched against a canonical declaration and registered with the
:class:`tools.runtime_activation.RuntimeActivationRegistry` so the
silent-drift surface (declared-but-dormant nodes) is queryable from
the operator routes that ship in this PR.

The declared topology lives in this module as a single static
``_DECLARED_NODE_SPECS`` constant so a reader can navigate every
canonical node in one place. Each spec pairs a
:class:`tools.runtime_topology.RuntimeNode` with a ``state_attr``
name that ``_State`` is expected to set; the registrar's
:meth:`HarnessRuntimeRegistrar.register_at_boot` walks the spec
list, introspects ``_State`` for each attribute, and registers
the node as ``STARTED`` (attribute is present and non-None) or
``DORMANT`` (attribute is missing or ``None``).

INV-15 byte-identical replay: two registrars built from the same
``_State`` produce byte-identical
:meth:`tools.runtime_activation.ActivationSnapshot.digest`. The
declared topology digest is pinned by the underlying
:class:`tools.runtime_topology.RuntimeTopology` invariants.

The registrar exposes four read surfaces consumed by the
``/api/operator/runtime/{topology,active,dormant,capability/{tag}}``
routes added in this PR:

* :meth:`declared_topology_view` — the static declared topology
  (nodes + edges + digest).
* :meth:`active_view` — the subgraph of declared nodes whose
  ``_State`` backing is currently STARTED / HEALTHY / DEGRADED.
* :meth:`dormant_view` — the subgraph of declared-but-dormant
  nodes (the silent-drift surface PR-RT-5 will require explicit
  admission for).
* :meth:`capability_view` — full
  :class:`tools.runtime_capability.CapabilityResolution`
  projection for the requested capability tag.

The registrar holds no per-call mutable state: ``register_at_boot``
is reentrant (idempotent re-registration with the same lifecycle
state is a no-op in the underlying activation registry), and the
projection methods are pure functions of the current snapshot.

This module is pure-stdlib + ``tools.runtime_*``. No top-level
imports of vendor packages or runtime engine tiers. Authority lint
``B1`` (no engine imports from ``ui.harness``) is preserved by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from tools.runtime_activation import (
    ActivationSnapshot,
    RuntimeActivationRegistry,
)
from tools.runtime_capability import (
    DependencyGraphResolver,
    RuntimeCapabilityMap,
)
from tools.runtime_topology import (
    EdgeRelation,
    LifecycleState,
    NodeKind,
    NodeTier,
    RuntimeEdge,
    RuntimeNode,
    RuntimeTopology,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ui.server import _State


REGISTRAR_VERSION: Final[str] = "v1.0-RT4"


# PR-RT-5 — Total Validation invariant allowlist.
#
# Every entry in :data:`_DECLARED_NODE_SPECS` must either be backed by a
# non-None ``_State`` attribute at boot (i.e. STARTED at the activation
# registry) or be on this allowlist. ``tools/total_validation.py``
# Phase 12 (``topology_drift``) walks the static declared topology
# against ``ui/server.py`` AST and against this allowlist; any node
# that is declared, not statically assigned, and not on this allowlist
# is a silent-runtime-topology-drift violation and downgrades the
# pipeline to FAIL in strict mode.
#
# Adding an entry to this allowlist is an explicit operator decision
# — the entry's reason string is the audit row that explains why the
# declared node is permitted to be dormant. The allowlist starts
# empty: every declared node is currently STARTED at boot under the
# PR-RT-4 wiring.
DECLARED_BUT_DORMANT_ALLOWLIST: Final[frozenset[str]] = frozenset()


@dataclass(frozen=True, slots=True)
class _DeclaredNodeSpec:
    """Static declaration of one canonical runtime node.

    Marries a :class:`RuntimeNode` with the ``_State`` attribute
    that backs it. ``state_attr`` is the attribute name the
    registrar uses to introspect ``_State`` at boot — if
    ``getattr(state, state_attr, None)`` is non-None the node is
    registered as STARTED, otherwise as DORMANT.
    """

    node: RuntimeNode
    state_attr: str


_DECLARED_NODE_SPECS: tuple[_DeclaredNodeSpec, ...] = (
    # --- ENGINES (T1) ----------------------------------------------------
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="intelligence_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset(
                {
                    "intelligence.signal",
                    "intelligence.meta_controller",
                }
            ),
        ),
        state_attr="intelligence",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="execution_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset(
                {
                    "execution.dispatch",
                    "execution.gate",
                }
            ),
        ),
        state_attr="execution",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="system_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset(
                {
                    "system.hazard",
                    "system.health",
                }
            ),
        ),
        state_attr="system",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="governance_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset(
                {
                    "governance.mode_fsm",
                    "governance.strategy_lifecycle",
                }
            ),
        ),
        state_attr="governance",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="learning_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"learning.update_emitter"}),
        ),
        state_attr="learning",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="evolution_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"evolution.mutation_proposer"}),
        ),
        state_attr="evolution",
    ),
    # --- LOOPS (T2) ------------------------------------------------------
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="meta_controller_hot_path",
            kind=NodeKind.LOOP,
            tier=NodeTier.T2,
            declared_version="v1.0",
            capabilities=frozenset(
                {
                    "intelligence.belief_state",
                    "intelligence.pressure_vector",
                }
            ),
        ),
        state_attr="meta_controller_hot_path",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="closed_learning_loop",
            kind=NodeKind.LOOP,
            tier=NodeTier.T2,
            declared_version="v1.0",
            capabilities=frozenset({"learning.closed_loop"}),
        ),
        state_attr="closed_learning_loop",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="structural_evolution_loop",
            kind=NodeKind.LOOP,
            tier=NodeTier.T2,
            declared_version="v1.0",
            capabilities=frozenset({"evolution.structural_loop"}),
        ),
        state_attr="structural_evolution_loop",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="slow_loop_learner",
            kind=NodeKind.LOOP,
            tier=NodeTier.T2,
            declared_version="v1.0",
            capabilities=frozenset({"learning.slow_loop"}),
        ),
        state_attr="slow_loop_learner",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="patch_pipeline",
            kind=NodeKind.LOOP,
            tier=NodeTier.T2,
            declared_version="v1.0",
            capabilities=frozenset({"evolution.patch_pipeline"}),
        ),
        state_attr="patch_pipeline",
    ),
    # --- REGISTRIES (T0) -------------------------------------------------
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="source_registry",
            kind=NodeKind.REGISTRY,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"registry.sources"}),
        ),
        state_attr="source_registry",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="strategy_registry",
            kind=NodeKind.REGISTRY,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"registry.strategies"}),
        ),
        state_attr="strategy_registry",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="plugin_registry",
            kind=NodeKind.REGISTRY,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"registry.plugins"}),
        ),
        state_attr="plugin_registry",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="signal_trust_registry",
            kind=NodeKind.REGISTRY,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"registry.signal_trust"}),
        ),
        state_attr="signal_trust_registry",
    ),
    # --- GATES / POLICIES (T0) -------------------------------------------
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="authority_guard",
            kind=NodeKind.GATE,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"execution.authority_guard"}),
        ),
        state_attr="authority_guard",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="hazard_throttle",
            kind=NodeKind.GATE,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"execution.hazard_throttle"}),
        ),
        state_attr="hazard_throttle",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="decision_signer",
            kind=NodeKind.GATE,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"governance.decision_signer"}),
        ),
        state_attr="decision_signer",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="policy_hash_anchor",
            kind=NodeKind.POLICY,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset({"governance.policy_hash"}),
        ),
        state_attr="policy_hash_anchor",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="development_mode_policy",
            kind=NodeKind.POLICY,
            tier=NodeTier.T0,
            declared_version="v1.0",
            capabilities=frozenset(
                {
                    "policy.development_mode",
                    "policy.trading_allowed",
                }
            ),
        ),
        state_attr="development_mode_policy",
    ),
    # --- ADAPTERS / SINKS (T1) -------------------------------------------
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="feedback_collector",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"learning.feedback_collector"}),
        ),
        state_attr="feedback_collector",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="learning_interface",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"learning.interface"}),
        ),
        state_attr="learning_interface",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="sensor_array",
            kind=NodeKind.SENSOR,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"system.sensor_array"}),
        ),
        state_attr="sensor_array",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="runtime_context_builder",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"system.runtime_context"}),
        ),
        state_attr="runtime_context_builder",
    ),
    # --- FEEDS / ADAPTERS (T1) -------------------------------------------
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="binance_feed",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"feed.market.binance"}),
        ),
        state_attr="binance_feed",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="coindesk_feed",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"feed.news.coindesk"}),
        ),
        state_attr="coindesk_feed",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="pumpfun_feed",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"feed.memecoin.pumpfun"}),
        ),
        state_attr="pumpfun_feed",
    ),
    _DeclaredNodeSpec(
        node=RuntimeNode(
            node_id="raydium_feed",
            kind=NodeKind.ADAPTER,
            tier=NodeTier.T1,
            declared_version="v1.0",
            capabilities=frozenset({"feed.memecoin.raydium"}),
        ),
        state_attr="raydium_feed",
    ),
)


_CANONICAL_EDGES: tuple[RuntimeEdge, ...] = (
    # Hot path: intelligence -> execution -> governance.
    RuntimeEdge(
        source_id="intelligence_engine",
        target_id="execution_engine",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="execution_engine",
        target_id="authority_guard",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    RuntimeEdge(
        source_id="execution_engine",
        target_id="hazard_throttle",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    RuntimeEdge(
        source_id="execution_engine",
        target_id="decision_signer",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    RuntimeEdge(
        source_id="execution_engine",
        target_id="development_mode_policy",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    RuntimeEdge(
        source_id="governance_engine",
        target_id="execution_engine",
        relation=EdgeRelation.GATES,
    ),
    # Meta controller + intelligence depth.
    RuntimeEdge(
        source_id="meta_controller_hot_path",
        target_id="intelligence_engine",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    # Sensor array -> hazard throttle.
    RuntimeEdge(
        source_id="sensor_array",
        target_id="system_engine",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="system_engine",
        target_id="hazard_throttle",
        relation=EdgeRelation.PRODUCES,
    ),
    # Learning loop closure.
    RuntimeEdge(
        source_id="execution_engine",
        target_id="feedback_collector",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="feedback_collector",
        target_id="closed_learning_loop",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="closed_learning_loop",
        target_id="learning_engine",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="closed_learning_loop",
        target_id="strategy_registry",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    RuntimeEdge(
        source_id="learning_interface",
        target_id="learning_engine",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    # Structural evolution chain.
    RuntimeEdge(
        source_id="structural_evolution_loop",
        target_id="evolution_engine",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="evolution_engine",
        target_id="patch_pipeline",
        relation=EdgeRelation.PRODUCES,
    ),
    RuntimeEdge(
        source_id="patch_pipeline",
        target_id="governance_engine",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    # Slow loop learner reads sources.
    RuntimeEdge(
        source_id="slow_loop_learner",
        target_id="source_registry",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    # Governance dependencies.
    RuntimeEdge(
        source_id="governance_engine",
        target_id="strategy_registry",
        relation=EdgeRelation.OWNS,
    ),
    RuntimeEdge(
        source_id="governance_engine",
        target_id="policy_hash_anchor",
        relation=EdgeRelation.OWNS,
    ),
    # System engine consumes signal-trust + sources at boot.
    RuntimeEdge(
        source_id="execution_engine",
        target_id="signal_trust_registry",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    RuntimeEdge(
        source_id="intelligence_engine",
        target_id="source_registry",
        relation=EdgeRelation.DEPENDS_ON,
    ),
    # Plugin registry owned by harness.
    RuntimeEdge(
        source_id="plugin_registry",
        target_id="intelligence_engine",
        relation=EdgeRelation.PROJECTS,
    ),
    # Runtime context builder reads system engine.
    RuntimeEdge(
        source_id="runtime_context_builder",
        target_id="system_engine",
        relation=EdgeRelation.DEPENDS_ON,
    ),
)


def _canonical_topology() -> RuntimeTopology:
    """Build the canonical declared topology from the static specs.

    Pure function; the same module load always returns a topology
    with the same BLAKE2b-128 digest.
    """

    nodes = tuple(spec.node for spec in _DECLARED_NODE_SPECS)
    return RuntimeTopology(nodes=nodes, edges=_CANONICAL_EDGES)


def declared_topology() -> RuntimeTopology:
    """Public accessor for the canonical declared topology.

    Used by tests and by PR-RT-5 total_validation to anchor the
    declared-vs-active invariant.
    """

    return _canonical_topology()


def declared_node_ids() -> tuple[str, ...]:
    """Return every declared node_id, sorted lexicographically."""

    return tuple(sorted(spec.node.node_id for spec in _DECLARED_NODE_SPECS))


def declared_state_attr_for(node_id: str) -> str | None:
    """Return the ``_State`` attribute name backing ``node_id``, or
    ``None`` if ``node_id`` is not declared."""

    for spec in _DECLARED_NODE_SPECS:
        if spec.node.node_id == node_id:
            return spec.state_attr
    return None


def declared_state_attrs() -> tuple[tuple[str, str], ...]:
    """Return every (node_id, state_attr) pair, sorted by node_id.

    Used by :mod:`tools.total_validation` to anchor the declared-vs-
    statically-wired invariant (PR-RT-5). Pure, byte-stable: the
    returned tuple sorts by ``node_id`` so two independent runs of the
    same import return the byte-identical sequence.
    """

    return tuple(
        sorted(
            ((spec.node.node_id, spec.state_attr) for spec in _DECLARED_NODE_SPECS),
            key=lambda pair: pair[0],
        )
    )


@dataclass
class HarnessRuntimeRegistrar:
    """The runtime topology authority bound to a running harness.

    Owns the immutable declared topology, the mutable activation
    registry, and projects them into the four read surfaces the
    operator routes (``/api/operator/runtime/{topology,active,
    dormant,capability/{tag}}``) consume.

    The registrar is constructed empty;
    :meth:`register_at_boot` walks the declared topology, introspects
    the supplied ``_State``, and registers each declared node as
    ``STARTED`` (backed by a non-None attribute) or ``DORMANT``
    (the silent-drift surface).

    Re-running :meth:`register_at_boot` on the same registrar is a
    no-op (the underlying activation registry tolerates idempotent
    re-registration with the same state).
    """

    topology: RuntimeTopology = field(default_factory=_canonical_topology)
    _registry: RuntimeActivationRegistry = field(init=False)

    def __post_init__(self) -> None:
        self._registry = RuntimeActivationRegistry(topology=self.topology)

    # -- public accessors -------------------------------------------------

    @property
    def registry(self) -> RuntimeActivationRegistry:
        """The mutable activation registry. Exposed for tests and for
        downstream PRs that want to drive lifecycle transitions
        (e.g. STARTED -> HEALTHY after a health check)."""

        return self._registry

    def snapshot(self) -> ActivationSnapshot:
        """Project the current registry state as an immutable snapshot."""

        return self._registry.snapshot()

    def capability_map(self) -> RuntimeCapabilityMap:
        """The capability resolver for the current snapshot."""

        return RuntimeCapabilityMap(
            topology=self.topology,
            snapshot=self.snapshot(),
        )

    def dependency_resolver(self) -> DependencyGraphResolver:
        """The DEPENDS_ON walker for the current snapshot."""

        return DependencyGraphResolver(
            topology=self.topology,
            snapshot=self.snapshot(),
        )

    # -- boot wiring ------------------------------------------------------

    def register_at_boot(self, state: _State) -> None:
        """Walk the declared topology and register each node against
        ``_State``.

        For each declared node:

        * If ``getattr(state, spec.state_attr, None)`` is non-None,
          register as :class:`LifecycleState.STARTED` with reason
          ``"boot:started"``.
        * Otherwise, register as :class:`LifecycleState.DORMANT`
          with reason ``"boot:dormant"`` — the silent-drift surface.

        Reentrant: re-registering a node with the same lifecycle
        state is a no-op in the underlying activation registry.
        Re-registering with a *different* state raises
        :class:`tools.runtime_activation.ActivationViolation` (with
        an audited ``RUNTIME_ACTIVATION_VIOLATION`` payload).
        """

        for spec in _DECLARED_NODE_SPECS:
            backing = getattr(state, spec.state_attr, None)
            if backing is None:
                initial_state = LifecycleState.DORMANT
                reason = "boot:dormant"
            else:
                initial_state = LifecycleState.STARTED
                reason = "boot:started"
            self._registry.register(
                node_id=spec.node.node_id,
                initial_state=initial_state,
                reason=reason,
            )

    # -- read projections (operator routes) -------------------------------

    def declared_topology_view(self) -> dict[str, Any]:
        """Canonical projection of the declared topology.

        Returned shape::

            {
              "version": "v1.0-RT4",
              "topology_digest": "<hex>",
              "nodes": [...],
              "edges": [...]
            }
        """

        canonical = self.topology.canonical()
        return {
            "version": REGISTRAR_VERSION,
            "topology_digest": self.topology.digest(),
            "node_count": self.topology.node_count(),
            "edge_count": self.topology.edge_count(),
            "nodes": canonical["nodes"],
            "edges": canonical["edges"],
        }

    def active_view(self) -> dict[str, Any]:
        """The currently-active subgraph projection.

        Returned shape::

            {
              "version": "v1.0-RT4",
              "topology_digest": "<hex>",
              "snapshot_digest": "<hex>",
              "active_node_ids": [...],   # STARTED ∪ HEALTHY ∪ DEGRADED
              "wired_node_ids": [...]
            }
        """

        snap = self.snapshot()
        return {
            "version": REGISTRAR_VERSION,
            "topology_digest": self.topology.digest(),
            "snapshot_digest": snap.digest(),
            "active_node_ids": sorted(snap.active_node_ids()),
            "wired_node_ids": sorted(snap.wired_node_ids()),
        }

    def dormant_view(self) -> dict[str, Any]:
        """The currently-dormant subgraph projection.

        Returned shape::

            {
              "version": "v1.0-RT4",
              "topology_digest": "<hex>",
              "snapshot_digest": "<hex>",
              "dormant_node_ids": [...],    # DECLARED ∪ DORMANT ∪ STOPPED
              "unregistered_node_ids": [...]  # declared but never registered
            }
        """

        snap = self.snapshot()
        declared = set(spec.node.node_id for spec in _DECLARED_NODE_SPECS)
        registered = {nid for nid, _state in snap.states}
        unregistered = declared - registered
        return {
            "version": REGISTRAR_VERSION,
            "topology_digest": self.topology.digest(),
            "snapshot_digest": snap.digest(),
            "dormant_node_ids": sorted(snap.dormant_node_ids()),
            "unregistered_node_ids": sorted(unregistered),
        }

    def capability_view(self, capability: str) -> dict[str, Any]:
        """Resolve a capability against the declared + active topology.

        Returned shape::

            {
              "version": "v1.0-RT4",
              "capability": "<tag>",
              "declared": [...],
              "active": [...],
              "dormant": [...],
              "unregistered": [...],
              "is_resolved": bool,
              "is_dormant": bool,
              "is_missing": bool,
              "has_unregistered_providers": bool
            }

        Raises :class:`tools.runtime_capability.CapabilityError` if
        ``capability`` fails the underlying validation (empty,
        too long, contains illegal characters).
        """

        resolution = self.capability_map().resolve(capability)
        return {
            "version": REGISTRAR_VERSION,
            "capability": resolution.capability,
            "declared": list(resolution.declared),
            "active": list(resolution.active),
            "dormant": list(resolution.dormant),
            "unregistered": list(resolution.unregistered),
            "is_resolved": resolution.is_resolved(),
            "is_dormant": resolution.is_dormant(),
            "is_missing": resolution.is_missing(),
            "has_unregistered_providers": resolution.has_unregistered_providers(),
        }


__all__ = (
    "DECLARED_BUT_DORMANT_ALLOWLIST",
    "HarnessRuntimeRegistrar",
    "REGISTRAR_VERSION",
    "declared_node_ids",
    "declared_state_attr_for",
    "declared_state_attrs",
    "declared_topology",
)
