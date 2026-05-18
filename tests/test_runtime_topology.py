"""Tests for ``tools/runtime_topology.py`` (PR-RT-1).

These tests pin the declared-half of the runtime topology authority
chain. They cover:

* module identity (constants, exports, no banned top-level imports)
* :class:`NodeKind`, :class:`NodeTier`, :class:`EdgeRelation`,
  :class:`LifecycleState` enum surfaces
* :class:`RuntimeNode` validation
* :class:`RuntimeEdge` validation
* :class:`RuntimeTopology` validation + canonicalisation + digest
* :meth:`RuntimeTopology.find_node`, :meth:`nodes_by_tier`,
  :meth:`nodes_by_kind`, :meth:`edges_from`, :meth:`edges_to`,
  :meth:`providers_of` helpers
* :func:`is_legal_transition` lifecycle FSM
* :func:`enable_runtime_topology_factory` lazy seam
* INV-15 byte-identical three-run determinism
* AST guards forbidding top-level vendor / network / engine imports
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from tools.runtime_topology import (
    CAPABILITY_PATTERN,
    MAX_CAPABILITIES_PER_NODE,
    MAX_CAPABILITY_LEN,
    MAX_EDGES_PER_TOPOLOGY,
    MAX_NODE_ID_LEN,
    MAX_NODES_PER_TOPOLOGY,
    MAX_NOTE_LEN,
    MAX_VERSION_LEN,
    NEW_PIP_DEPENDENCIES,
    NODE_ID_PATTERN,
    TOPOLOGY_VERSION,
    VERSION_PATTERN,
    EdgeRelation,
    LifecycleState,
    NodeKind,
    NodeTier,
    RuntimeEdge,
    RuntimeNode,
    RuntimeTopology,
    RuntimeTopologyFactory,
    TopologyError,
    build_topology,
    empty_topology,
    enable_runtime_topology_factory,
    is_legal_transition,
)

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "runtime_topology.py"


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_topology_version_constant() -> None:
    assert TOPOLOGY_VERSION == "v1.0-RT1"


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_caps_have_sane_values() -> None:
    assert MAX_NODE_ID_LEN == 128
    assert MAX_CAPABILITY_LEN == 128
    assert MAX_CAPABILITIES_PER_NODE == 64
    assert MAX_NODES_PER_TOPOLOGY == 4096
    assert MAX_EDGES_PER_TOPOLOGY == 16_384
    assert MAX_VERSION_LEN == 32
    assert MAX_NOTE_LEN == 256


def test_patterns_compile() -> None:
    assert NODE_ID_PATTERN.match("execution_engine.gate")
    assert NODE_ID_PATTERN.match("a")
    assert not NODE_ID_PATTERN.match("Execution")
    assert not NODE_ID_PATTERN.match("1bad")
    assert not NODE_ID_PATTERN.match("foo.")
    assert CAPABILITY_PATTERN.match("learning.closed_loop")
    assert CAPABILITY_PATTERN.match("learning")
    assert not CAPABILITY_PATTERN.match("Learning")
    assert VERSION_PATTERN.match("v1.0-RT1")
    assert not VERSION_PATTERN.match("v1 0")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_node_kind_members() -> None:
    expected = {
        "ENGINE",
        "LOOP",
        "BUS",
        "SENSOR",
        "ADAPTER",
        "REGISTRY",
        "ROUTE",
        "GATE",
        "POLICY",
        "STORE",
    }
    assert {m.value for m in NodeKind} == expected


def test_node_tier_members() -> None:
    assert {m.value for m in NodeTier} == {"T0", "T1", "T2", "UI"}


def test_edge_relation_members() -> None:
    expected = {
        "PRODUCES",
        "CONSUMES",
        "OWNS",
        "GATES",
        "DEPENDS_ON",
        "PROJECTS",
    }
    assert {m.value for m in EdgeRelation} == expected


def test_lifecycle_state_members() -> None:
    expected = {
        "DECLARED",
        "WIRED",
        "STARTED",
        "HEALTHY",
        "DEGRADED",
        "STOPPED",
        "DORMANT",
    }
    assert {m.value for m in LifecycleState} == expected


# ---------------------------------------------------------------------------
# Lifecycle FSM
# ---------------------------------------------------------------------------


def test_legal_transitions_declared_to_wired() -> None:
    assert is_legal_transition(LifecycleState.DECLARED, LifecycleState.WIRED)


def test_legal_transitions_declared_to_dormant() -> None:
    assert is_legal_transition(LifecycleState.DECLARED, LifecycleState.DORMANT)


def test_legal_transitions_wired_to_started() -> None:
    assert is_legal_transition(LifecycleState.WIRED, LifecycleState.STARTED)


def test_legal_transitions_started_to_healthy() -> None:
    assert is_legal_transition(LifecycleState.STARTED, LifecycleState.HEALTHY)


def test_legal_transitions_healthy_to_degraded() -> None:
    assert is_legal_transition(LifecycleState.HEALTHY, LifecycleState.DEGRADED)


def test_legal_transitions_degraded_to_healthy() -> None:
    assert is_legal_transition(LifecycleState.DEGRADED, LifecycleState.HEALTHY)


def test_legal_transitions_started_to_stopped() -> None:
    assert is_legal_transition(LifecycleState.STARTED, LifecycleState.STOPPED)


def test_legal_transitions_stopped_to_wired_only() -> None:
    assert is_legal_transition(LifecycleState.STOPPED, LifecycleState.WIRED)
    assert not is_legal_transition(LifecycleState.STOPPED, LifecycleState.HEALTHY)


def test_legal_transitions_dormant_to_wired_only() -> None:
    assert is_legal_transition(LifecycleState.DORMANT, LifecycleState.WIRED)
    assert not is_legal_transition(LifecycleState.DORMANT, LifecycleState.HEALTHY)


def test_illegal_transition_declared_to_healthy() -> None:
    assert not is_legal_transition(LifecycleState.DECLARED, LifecycleState.HEALTHY)


def test_illegal_transition_healthy_to_declared() -> None:
    assert not is_legal_transition(LifecycleState.HEALTHY, LifecycleState.DECLARED)


# ---------------------------------------------------------------------------
# RuntimeNode validation
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "execution_engine.gate",
    kind: NodeKind = NodeKind.GATE,
    tier: NodeTier = NodeTier.T0,
    version: str = "v1",
    capabilities: frozenset[str] = frozenset({"execution.gate"}),
) -> RuntimeNode:
    return RuntimeNode(
        node_id=node_id,
        kind=kind,
        tier=tier,
        declared_version=version,
        capabilities=capabilities,
    )


def test_runtime_node_is_frozen() -> None:
    node = _make_node()
    with pytest.raises((AttributeError, TypeError)):
        node.node_id = "other"  # type: ignore[misc]


def test_runtime_node_is_slotted() -> None:
    node = _make_node()
    assert not hasattr(node, "__dict__")


def test_runtime_node_accepts_valid_inputs() -> None:
    node = _make_node()
    assert node.node_id == "execution_engine.gate"
    assert node.kind is NodeKind.GATE
    assert node.tier is NodeTier.T0
    assert node.declared_version == "v1"
    assert node.capabilities == frozenset({"execution.gate"})


def test_runtime_node_rejects_non_string_id() -> None:
    with pytest.raises(TopologyError):
        RuntimeNode(
            node_id=42,  # type: ignore[arg-type]
            kind=NodeKind.GATE,
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities=frozenset(),
        )


def test_runtime_node_rejects_empty_id() -> None:
    with pytest.raises(TopologyError):
        _make_node(node_id="")


def test_runtime_node_rejects_oversize_id() -> None:
    with pytest.raises(TopologyError):
        _make_node(node_id="a" * (MAX_NODE_ID_LEN + 1))


def test_runtime_node_rejects_uppercase_id() -> None:
    with pytest.raises(TopologyError):
        _make_node(node_id="Execution.Engine")


def test_runtime_node_rejects_numeric_leading_id() -> None:
    with pytest.raises(TopologyError):
        _make_node(node_id="1engine")


def test_runtime_node_rejects_non_enum_kind() -> None:
    with pytest.raises(TopologyError):
        RuntimeNode(
            node_id="ok",
            kind="GATE",  # type: ignore[arg-type]
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities=frozenset(),
        )


def test_runtime_node_rejects_non_enum_tier() -> None:
    with pytest.raises(TopologyError):
        RuntimeNode(
            node_id="ok",
            kind=NodeKind.GATE,
            tier="T0",  # type: ignore[arg-type]
            declared_version="v1",
            capabilities=frozenset(),
        )


def test_runtime_node_rejects_empty_version() -> None:
    with pytest.raises(TopologyError):
        _make_node(version="")


def test_runtime_node_rejects_oversize_version() -> None:
    with pytest.raises(TopologyError):
        _make_node(version="v" * (MAX_VERSION_LEN + 1))


def test_runtime_node_rejects_invalid_version_chars() -> None:
    with pytest.raises(TopologyError):
        _make_node(version="v 1")


def test_runtime_node_rejects_non_frozenset_capabilities() -> None:
    with pytest.raises(TopologyError):
        RuntimeNode(
            node_id="ok",
            kind=NodeKind.GATE,
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities={"execution.gate"},  # type: ignore[arg-type]
        )


def test_runtime_node_rejects_too_many_capabilities() -> None:
    caps = frozenset(f"cap.{i}" for i in range(MAX_CAPABILITIES_PER_NODE + 1))
    with pytest.raises(TopologyError):
        _make_node(capabilities=caps)


def test_runtime_node_rejects_oversize_capability() -> None:
    with pytest.raises(TopologyError):
        _make_node(capabilities=frozenset({"a" * (MAX_CAPABILITY_LEN + 1)}))


def test_runtime_node_rejects_uppercase_capability() -> None:
    with pytest.raises(TopologyError):
        _make_node(capabilities=frozenset({"Execution"}))


def test_runtime_node_canonical_sorts_capabilities() -> None:
    node = _make_node(
        capabilities=frozenset({"z", "a", "m"}),
    )
    assert node.canonical()["capabilities"] == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# RuntimeEdge validation
# ---------------------------------------------------------------------------


def test_runtime_edge_is_frozen() -> None:
    edge = RuntimeEdge(source_id="a", target_id="b", relation=EdgeRelation.PRODUCES)
    with pytest.raises((AttributeError, TypeError)):
        edge.source_id = "other"  # type: ignore[misc]


def test_runtime_edge_is_slotted() -> None:
    edge = RuntimeEdge(source_id="a", target_id="b", relation=EdgeRelation.PRODUCES)
    assert not hasattr(edge, "__dict__")


def test_runtime_edge_rejects_self_loop() -> None:
    with pytest.raises(TopologyError):
        RuntimeEdge(source_id="a", target_id="a", relation=EdgeRelation.PRODUCES)


def test_runtime_edge_rejects_invalid_source() -> None:
    with pytest.raises(TopologyError):
        RuntimeEdge(
            source_id="Bad",
            target_id="ok",
            relation=EdgeRelation.PRODUCES,
        )


def test_runtime_edge_rejects_non_enum_relation() -> None:
    with pytest.raises(TopologyError):
        RuntimeEdge(
            source_id="a",
            target_id="b",
            relation="PRODUCES",  # type: ignore[arg-type]
        )


def test_runtime_edge_rejects_non_string_note() -> None:
    with pytest.raises(TopologyError):
        RuntimeEdge(
            source_id="a",
            target_id="b",
            relation=EdgeRelation.PRODUCES,
            note=42,  # type: ignore[arg-type]
        )


def test_runtime_edge_rejects_oversize_note() -> None:
    with pytest.raises(TopologyError):
        RuntimeEdge(
            source_id="a",
            target_id="b",
            relation=EdgeRelation.PRODUCES,
            note="x" * (MAX_NOTE_LEN + 1),
        )


# ---------------------------------------------------------------------------
# RuntimeTopology validation
# ---------------------------------------------------------------------------


def _two_node_topology() -> RuntimeTopology:
    a = _make_node(node_id="a", capabilities=frozenset({"x"}))
    b = _make_node(
        node_id="b",
        kind=NodeKind.LOOP,
        tier=NodeTier.T1,
        capabilities=frozenset({"y"}),
    )
    return RuntimeTopology(
        nodes=(a, b),
        edges=(RuntimeEdge(source_id="a", target_id="b", relation=EdgeRelation.PRODUCES),),
    )


def test_runtime_topology_is_frozen() -> None:
    topology = _two_node_topology()
    with pytest.raises((AttributeError, TypeError)):
        topology.nodes = ()  # type: ignore[misc]


def test_runtime_topology_rejects_non_tuple_nodes() -> None:
    with pytest.raises(TopologyError):
        RuntimeTopology(nodes=[_make_node()], edges=())  # type: ignore[arg-type]


def test_runtime_topology_rejects_non_tuple_edges() -> None:
    with pytest.raises(TopologyError):
        RuntimeTopology(
            nodes=(_make_node(),),
            edges=[],  # type: ignore[arg-type]
        )


def test_runtime_topology_rejects_duplicate_node_id() -> None:
    a1 = _make_node(node_id="a", capabilities=frozenset({"x"}))
    a2 = _make_node(node_id="a", capabilities=frozenset({"y"}))
    with pytest.raises(TopologyError):
        RuntimeTopology(nodes=(a1, a2), edges=())


def test_runtime_topology_rejects_dangling_edge_source() -> None:
    with pytest.raises(TopologyError):
        RuntimeTopology(
            nodes=(_make_node(node_id="a", capabilities=frozenset()),),
            edges=(
                RuntimeEdge(
                    source_id="ghost",
                    target_id="a",
                    relation=EdgeRelation.PRODUCES,
                ),
            ),
        )


def test_runtime_topology_rejects_dangling_edge_target() -> None:
    with pytest.raises(TopologyError):
        RuntimeTopology(
            nodes=(_make_node(node_id="a", capabilities=frozenset()),),
            edges=(
                RuntimeEdge(
                    source_id="a",
                    target_id="ghost",
                    relation=EdgeRelation.PRODUCES,
                ),
            ),
        )


def test_runtime_topology_rejects_duplicate_edge() -> None:
    a = _make_node(node_id="a", capabilities=frozenset())
    b = _make_node(node_id="b", kind=NodeKind.LOOP, capabilities=frozenset())
    with pytest.raises(TopologyError):
        RuntimeTopology(
            nodes=(a, b),
            edges=(
                RuntimeEdge(
                    source_id="a",
                    target_id="b",
                    relation=EdgeRelation.PRODUCES,
                ),
                RuntimeEdge(
                    source_id="a",
                    target_id="b",
                    relation=EdgeRelation.PRODUCES,
                ),
            ),
        )


def test_runtime_topology_allows_parallel_edges_with_different_relation() -> None:
    a = _make_node(node_id="a", capabilities=frozenset())
    b = _make_node(node_id="b", kind=NodeKind.LOOP, capabilities=frozenset())
    topology = RuntimeTopology(
        nodes=(a, b),
        edges=(
            RuntimeEdge(
                source_id="a",
                target_id="b",
                relation=EdgeRelation.PRODUCES,
            ),
            RuntimeEdge(
                source_id="a",
                target_id="b",
                relation=EdgeRelation.OWNS,
            ),
        ),
    )
    assert topology.edge_count() == 2


def test_runtime_topology_sorts_nodes_on_construction() -> None:
    a = _make_node(node_id="z_last", capabilities=frozenset())
    b = _make_node(node_id="a_first", kind=NodeKind.LOOP, capabilities=frozenset())
    topology = RuntimeTopology(nodes=(a, b), edges=())
    assert [n.node_id for n in topology.nodes] == ["a_first", "z_last"]


def test_runtime_topology_sorts_edges_on_construction() -> None:
    a = _make_node(node_id="a", capabilities=frozenset())
    b = _make_node(node_id="b", kind=NodeKind.LOOP, capabilities=frozenset())
    c = _make_node(node_id="c", kind=NodeKind.LOOP, capabilities=frozenset())
    topology = RuntimeTopology(
        nodes=(a, b, c),
        edges=(
            RuntimeEdge(
                source_id="b",
                target_id="c",
                relation=EdgeRelation.PRODUCES,
            ),
            RuntimeEdge(
                source_id="a",
                target_id="b",
                relation=EdgeRelation.PRODUCES,
            ),
        ),
    )
    assert [(e.source_id, e.target_id) for e in topology.edges] == [("a", "b"), ("b", "c")]


# ---------------------------------------------------------------------------
# Canonical & digest
# ---------------------------------------------------------------------------


def test_topology_canonical_has_sorted_keys() -> None:
    topology = _two_node_topology()
    canonical = topology.canonical()
    assert list(canonical.keys()) == ["edges", "nodes", "version"]


def test_topology_digest_is_blake2b_128_hex() -> None:
    topology = _two_node_topology()
    digest = topology.digest()
    assert len(digest) == 32
    int(digest, 16)


def test_topology_digest_is_deterministic_across_runs() -> None:
    def fresh() -> str:
        return _two_node_topology().digest()

    assert fresh() == fresh() == fresh()


def test_topology_digest_changes_on_node_addition() -> None:
    base = _two_node_topology()
    extra = _make_node(node_id="c", kind=NodeKind.LOOP, capabilities=frozenset({"z"}))
    enlarged = RuntimeTopology(nodes=(*base.nodes, extra), edges=base.edges)
    assert base.digest() != enlarged.digest()


def test_topology_digest_is_order_independent() -> None:
    a = _make_node(node_id="a", capabilities=frozenset({"x"}))
    b = _make_node(
        node_id="b",
        kind=NodeKind.LOOP,
        tier=NodeTier.T1,
        capabilities=frozenset({"y"}),
    )
    edge_ab = RuntimeEdge(source_id="a", target_id="b", relation=EdgeRelation.PRODUCES)

    forward = RuntimeTopology(nodes=(a, b), edges=(edge_ab,))
    reverse = RuntimeTopology(nodes=(b, a), edges=(edge_ab,))
    assert forward.digest() == reverse.digest()


def test_topology_digest_changes_on_relation_swap() -> None:
    base = _two_node_topology()
    alt_edges = (RuntimeEdge(source_id="a", target_id="b", relation=EdgeRelation.GATES),)
    alt = RuntimeTopology(nodes=base.nodes, edges=alt_edges)
    assert base.digest() != alt.digest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_find_node_returns_match() -> None:
    topology = _two_node_topology()
    assert topology.find_node("a") is not None


def test_find_node_returns_none_for_missing() -> None:
    topology = _two_node_topology()
    assert topology.find_node("ghost") is None


def test_nodes_by_tier_filters() -> None:
    topology = _two_node_topology()
    tier_t0 = topology.nodes_by_tier(NodeTier.T0)
    tier_t1 = topology.nodes_by_tier(NodeTier.T1)
    assert {n.node_id for n in tier_t0} == {"a"}
    assert {n.node_id for n in tier_t1} == {"b"}


def test_nodes_by_kind_filters() -> None:
    topology = _two_node_topology()
    gates = topology.nodes_by_kind(NodeKind.GATE)
    loops = topology.nodes_by_kind(NodeKind.LOOP)
    assert {n.node_id for n in gates} == {"a"}
    assert {n.node_id for n in loops} == {"b"}


def test_edges_from_filters() -> None:
    topology = _two_node_topology()
    assert {e.target_id for e in topology.edges_from("a")} == {"b"}
    assert topology.edges_from("b") == ()


def test_edges_to_filters() -> None:
    topology = _two_node_topology()
    assert {e.source_id for e in topology.edges_to("b")} == {"a"}
    assert topology.edges_to("a") == ()


def test_providers_of_returns_advertising_nodes() -> None:
    topology = _two_node_topology()
    providers = topology.providers_of("x")
    assert {n.node_id for n in providers} == {"a"}


def test_providers_of_returns_empty_when_missing() -> None:
    topology = _two_node_topology()
    assert topology.providers_of("ghost") == ()


def test_providers_of_rejects_non_string() -> None:
    topology = _two_node_topology()
    with pytest.raises(TopologyError):
        topology.providers_of(42)  # type: ignore[arg-type]


def test_node_count_and_edge_count() -> None:
    topology = _two_node_topology()
    assert topology.node_count() == 2
    assert topology.edge_count() == 1


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_build_topology_accepts_iterables() -> None:
    a = _make_node(node_id="a", capabilities=frozenset())
    b = _make_node(node_id="b", kind=NodeKind.LOOP, capabilities=frozenset())
    topology = build_topology(
        nodes=iter([a, b]),
        edges=iter(
            [
                RuntimeEdge(
                    source_id="a",
                    target_id="b",
                    relation=EdgeRelation.PRODUCES,
                )
            ]
        ),
    )
    assert topology.node_count() == 2


def test_empty_topology_is_legal() -> None:
    topology = empty_topology()
    assert topology.node_count() == 0
    assert topology.edge_count() == 0


def test_empty_topology_digest_is_stable() -> None:
    assert empty_topology().digest() == empty_topology().digest()


# ---------------------------------------------------------------------------
# Lazy seam factory
# ---------------------------------------------------------------------------


def test_enable_factory_returns_stdlib_backend() -> None:
    factory = enable_runtime_topology_factory()
    assert isinstance(factory, RuntimeTopologyFactory)
    assert factory.backend == "stdlib"


def test_enable_factory_rejects_non_mapping_overrides() -> None:
    with pytest.raises(TopologyError):
        enable_runtime_topology_factory(overrides=[])  # type: ignore[arg-type]


def test_factory_build_matches_direct_construction() -> None:
    factory = enable_runtime_topology_factory()
    a = _make_node(node_id="a", capabilities=frozenset())
    b = _make_node(node_id="b", kind=NodeKind.LOOP, capabilities=frozenset())
    edge = RuntimeEdge(source_id="a", target_id="b", relation=EdgeRelation.PRODUCES)
    topology = factory.build(nodes=(a, b), edges=(edge,))
    direct = RuntimeTopology(nodes=(a, b), edges=(edge,))
    assert topology.digest() == direct.digest()


def test_factory_rejects_unsupported_backend() -> None:
    with pytest.raises(TopologyError):
        RuntimeTopologyFactory(backend="vendor", config={})


# ---------------------------------------------------------------------------
# INV-15 determinism
# ---------------------------------------------------------------------------


def _build_realistic_topology() -> RuntimeTopology:
    nodes = (
        RuntimeNode(
            node_id="execution_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities=frozenset({"execution.dispatch"}),
        ),
        RuntimeNode(
            node_id="execution_engine.gate",
            kind=NodeKind.GATE,
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities=frozenset({"execution.gate"}),
        ),
        RuntimeNode(
            node_id="governance_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities=frozenset({"governance.policy", "governance.consent"}),
        ),
        RuntimeNode(
            node_id="intelligence_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T1,
            declared_version="v1",
            capabilities=frozenset({"intelligence.signal", "intelligence.meta"}),
        ),
        RuntimeNode(
            node_id="intelligence_engine.closed_learning_loop",
            kind=NodeKind.LOOP,
            tier=NodeTier.T1,
            declared_version="v1",
            capabilities=frozenset({"learning.closed_loop"}),
        ),
        RuntimeNode(
            node_id="intelligence_engine.structural_evolution_loop",
            kind=NodeKind.LOOP,
            tier=NodeTier.T1,
            declared_version="v1",
            capabilities=frozenset({"evolution.structural"}),
        ),
        RuntimeNode(
            node_id="system_engine",
            kind=NodeKind.ENGINE,
            tier=NodeTier.T0,
            declared_version="v1",
            capabilities=frozenset({"system.hazard"}),
        ),
    )
    edges = (
        RuntimeEdge(
            source_id="execution_engine.gate",
            target_id="execution_engine",
            relation=EdgeRelation.GATES,
        ),
        RuntimeEdge(
            source_id="execution_engine",
            target_id="intelligence_engine",
            relation=EdgeRelation.PRODUCES,
        ),
        RuntimeEdge(
            source_id="intelligence_engine",
            target_id="intelligence_engine.closed_learning_loop",
            relation=EdgeRelation.OWNS,
        ),
        RuntimeEdge(
            source_id="intelligence_engine",
            target_id="intelligence_engine.structural_evolution_loop",
            relation=EdgeRelation.OWNS,
        ),
        RuntimeEdge(
            source_id="governance_engine",
            target_id="execution_engine.gate",
            relation=EdgeRelation.GATES,
        ),
        RuntimeEdge(
            source_id="system_engine",
            target_id="governance_engine",
            relation=EdgeRelation.PRODUCES,
        ),
    )
    return RuntimeTopology(nodes=nodes, edges=edges)


def test_inv_15_three_runs_produce_identical_digest() -> None:
    digest_1 = _build_realistic_topology().digest()
    digest_2 = _build_realistic_topology().digest()
    digest_3 = _build_realistic_topology().digest()
    assert digest_1 == digest_2 == digest_3


def test_inv_15_three_runs_produce_identical_canonical() -> None:
    payload_1 = json.dumps(
        _build_realistic_topology().canonical(),
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_2 = json.dumps(
        _build_realistic_topology().canonical(),
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_3 = json.dumps(
        _build_realistic_topology().canonical(),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert payload_1 == payload_2 == payload_3


def test_realistic_topology_resolves_providers() -> None:
    topology = _build_realistic_topology()
    learning_providers = topology.providers_of("learning.closed_loop")
    assert {n.node_id for n in learning_providers} == {"intelligence_engine.closed_learning_loop"}
    governance_providers = topology.providers_of("governance.policy")
    assert {n.node_id for n in governance_providers} == {"governance_engine"}


# ---------------------------------------------------------------------------
# Reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_idempotency() -> None:
    import tools.runtime_topology as mod

    digest_before = _build_realistic_topology().digest()
    importlib.reload(mod)
    digest_after = mod.RuntimeTopology(  # type: ignore[attr-defined]
        nodes=(
            mod.RuntimeNode(  # type: ignore[attr-defined]
                node_id="a",
                kind=mod.NodeKind.GATE,  # type: ignore[attr-defined]
                tier=mod.NodeTier.T0,  # type: ignore[attr-defined]
                declared_version="v1",
                capabilities=frozenset({"x"}),
            ),
        ),
        edges=(),
    ).digest()
    assert digest_before != digest_after  # different topologies => different digests
    # Reload preserves contract identity:
    assert mod.TOPOLOGY_VERSION == "v1.0-RT1"

    # PR-RT-4: reloading ``tools.runtime_topology`` swaps the underlying
    # class objects (e.g. ``RuntimeNode``), but any downstream module that
    # imported those classes before the reload still references the
    # pre-reload objects. Cascade-reload the known downstream modules so
    # later tests in the same pytest run see a consistent class identity
    # and ``isinstance`` checks across the topology / activation /
    # capability / harness-registrar stack keep agreeing.
    import sys

    for _dep_name in (
        "tools.runtime_activation",
        "tools.runtime_capability",
        "ui.harness.runtime_registrar",
    ):
        _dep_mod = sys.modules.get(_dep_name)
        if _dep_mod is not None:
            importlib.reload(_dep_mod)


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_BANNED_TOP_LEVEL_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "asyncio",
        "socket",
        "random",
        "time",
        "numpy",
        "torch",
        "requests",
        "core_engine",
        "execution_engine",
        "governance_engine",
        "intelligence_engine",
        "system_engine",
        "ui",
    }
)


def _top_level_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                out.append(node.module)
    return out


def test_no_banned_top_level_imports() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = _top_level_imports(source)
    for imp in imports:
        root = imp.split(".")[0]
        assert root not in _BANNED_TOP_LEVEL_MODULES, (
            f"banned top-level import {imp!r} (root {root!r}) found in tools/runtime_topology.py"
        )


def test_lazy_seam_is_present() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "def enable_runtime_topology_factory" in source


def test_factory_is_frozen_slotted_dataclass() -> None:
    assert hasattr(RuntimeTopologyFactory, "__slots__")
    factory = enable_runtime_topology_factory()
    with pytest.raises((AttributeError, TypeError)):
        factory.backend = "vendor"  # type: ignore[misc]
