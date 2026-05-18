"""Tests for ``tools/runtime_capability.py`` (PR-RT-3).

These tests pin the **resolver** half of the runtime topology authority
chain. They cover:

* module identity (constants, exports, no banned top-level imports)
* :class:`CapabilityResolution` value object validation
* :class:`DependencyMissingLink` value object validation
* :class:`RuntimeCapabilityMap` filters declared providers to active
* dormant + unregistered providers surface separately
* unresolved_capabilities surfaces silent drift
* :class:`DependencyGraphResolver` walks DEPENDS_ON deterministically
* missing_link returns the first dormant ancestor in the chain
* cycle detection raises CapabilityResolutionError
* INV-15 determinism: three-run identical digest
* lazy-seam factory accepts overrides
* AST guards forbidding top-level vendor / network / engine imports
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.runtime_activation import (
    ActivationSnapshot,
    build_registry,
)
from tools.runtime_capability import (
    CAPABILITY_VERSION,
    MAX_CAPABILITY_LEN,
    MAX_DEPENDENCY_CHAIN_LEN,
    NEW_PIP_DEPENDENCIES,
    CapabilityError,
    CapabilityResolution,
    CapabilityResolutionError,
    DependencyGraphResolver,
    DependencyMissingLink,
    RuntimeCapabilityFactory,
    RuntimeCapabilityMap,
    build_capability_map,
    build_dependency_resolver,
    enable_runtime_capability_factory,
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

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "runtime_capability.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    *,
    kind: NodeKind = NodeKind.ENGINE,
    tier: NodeTier = NodeTier.T1,
    capabilities: frozenset[str] = frozenset(),
) -> RuntimeNode:
    return RuntimeNode(
        node_id=node_id,
        kind=kind,
        tier=tier,
        declared_version="v1.0",
        capabilities=capabilities,
    )


def _edge(source: str, target: str, relation: EdgeRelation) -> RuntimeEdge:
    return RuntimeEdge(
        source_id=source,
        target_id=target,
        relation=relation,
        note="",
    )


def _three_provider_topology() -> RuntimeTopology:
    """Three providers of ``learning.closed_loop`` for filter tests."""

    return RuntimeTopology(
        nodes=(
            _node(
                "learning.closed_loop_a",
                kind=NodeKind.LOOP,
                capabilities=frozenset({"learning.closed_loop"}),
            ),
            _node(
                "learning.closed_loop_b",
                kind=NodeKind.LOOP,
                capabilities=frozenset({"learning.closed_loop"}),
            ),
            _node(
                "learning.closed_loop_c",
                kind=NodeKind.LOOP,
                capabilities=frozenset({"learning.closed_loop"}),
            ),
        ),
        edges=(),
    )


def _depends_topology() -> RuntimeTopology:
    """A → DEPENDS_ON → B → DEPENDS_ON → C dependency chain."""

    return RuntimeTopology(
        nodes=(
            _node("alpha"),
            _node("beta"),
            _node("gamma"),
        ),
        edges=(
            _edge("alpha", "beta", EdgeRelation.DEPENDS_ON),
            _edge("beta", "gamma", EdgeRelation.DEPENDS_ON),
        ),
    )


def _cycle_topology() -> RuntimeTopology:
    """alpha → beta → alpha cycle."""

    return RuntimeTopology(
        nodes=(_node("alpha"), _node("beta")),
        edges=(
            _edge("alpha", "beta", EdgeRelation.DEPENDS_ON),
            _edge("beta", "alpha", EdgeRelation.DEPENDS_ON),
        ),
    )


def _snapshot_with_states(
    topology: RuntimeTopology,
    states: dict[str, LifecycleState],
) -> ActivationSnapshot:
    """Build a registry, drive every node to its target state, and
    return the snapshot."""

    registry = build_registry(topology=topology)
    for node_id, state in states.items():
        registry.register(node_id)
        current = LifecycleState.DECLARED
        legal_path: dict[LifecycleState, tuple[LifecycleState, ...]] = {
            LifecycleState.DECLARED: (),
            LifecycleState.WIRED: (LifecycleState.WIRED,),
            LifecycleState.STARTED: (
                LifecycleState.WIRED,
                LifecycleState.STARTED,
            ),
            LifecycleState.HEALTHY: (
                LifecycleState.WIRED,
                LifecycleState.STARTED,
                LifecycleState.HEALTHY,
            ),
            LifecycleState.DEGRADED: (
                LifecycleState.WIRED,
                LifecycleState.STARTED,
                LifecycleState.DEGRADED,
            ),
            LifecycleState.STOPPED: (
                LifecycleState.WIRED,
                LifecycleState.STARTED,
                LifecycleState.STOPPED,
            ),
            LifecycleState.DORMANT: (LifecycleState.DORMANT,),
        }
        path = legal_path[state]
        for step in path:
            registry.transition(node_id, step)
            current = step
        assert current is state or state is LifecycleState.DECLARED
    return registry.snapshot()


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_capability_version_constant() -> None:
    assert CAPABILITY_VERSION == "v1.0-RT3"


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_max_caps() -> None:
    assert MAX_CAPABILITY_LEN == 128
    assert MAX_DEPENDENCY_CHAIN_LEN == 256


# ---------------------------------------------------------------------------
# CapabilityResolution
# ---------------------------------------------------------------------------


def test_capability_resolution_is_frozen() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=("alpha",),
        active=("alpha",),
        dormant=(),
        unregistered=(),
    )
    with pytest.raises((AttributeError, TypeError)):
        res.capability = "other"  # type: ignore[misc]


def test_capability_resolution_rejects_empty_capability() -> None:
    with pytest.raises(CapabilityError):
        CapabilityResolution(
            capability="",
            declared=(),
            active=(),
            dormant=(),
            unregistered=(),
        )


def test_capability_resolution_rejects_oversize_capability() -> None:
    too_long = "a" * (MAX_CAPABILITY_LEN + 1)
    with pytest.raises(CapabilityError):
        CapabilityResolution(
            capability=too_long,
            declared=(),
            active=(),
            dormant=(),
            unregistered=(),
        )


def test_capability_resolution_rejects_overlap_between_active_and_dormant() -> None:
    with pytest.raises(CapabilityError):
        CapabilityResolution(
            capability="x.y",
            declared=("alpha",),
            active=("alpha",),
            dormant=("alpha",),
            unregistered=(),
        )


def test_capability_resolution_rejects_declared_missing_member() -> None:
    with pytest.raises(CapabilityError):
        CapabilityResolution(
            capability="x.y",
            declared=("alpha",),
            active=(),
            dormant=(),
            unregistered=(),
        )


def test_capability_resolution_sorts_tuples() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=("c", "a", "b"),
        active=("b", "a"),
        dormant=("c",),
        unregistered=(),
    )
    assert res.declared == ("a", "b", "c")
    assert res.active == ("a", "b")
    assert res.dormant == ("c",)


def test_capability_resolution_is_resolved() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=("alpha",),
        active=("alpha",),
        dormant=(),
        unregistered=(),
    )
    assert res.is_resolved() is True
    assert res.is_dormant() is False
    assert res.is_missing() is False


def test_capability_resolution_is_dormant() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=("alpha",),
        active=(),
        dormant=("alpha",),
        unregistered=(),
    )
    assert res.is_resolved() is False
    assert res.is_dormant() is True
    assert res.is_missing() is False


def test_capability_resolution_is_missing() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=(),
        active=(),
        dormant=(),
        unregistered=(),
    )
    assert res.is_resolved() is False
    assert res.is_dormant() is False
    assert res.is_missing() is True


def test_capability_resolution_has_unregistered_providers() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=("alpha",),
        active=(),
        dormant=(),
        unregistered=("alpha",),
    )
    assert res.has_unregistered_providers() is True
    assert res.is_dormant() is True


def test_capability_resolution_canonical_is_sorted() -> None:
    res = CapabilityResolution(
        capability="x.y",
        declared=("a", "b"),
        active=("a",),
        dormant=("b",),
        unregistered=(),
    )
    canonical = res.canonical()
    assert canonical == {
        "active": ["a"],
        "capability": "x.y",
        "declared": ["a", "b"],
        "dormant": ["b"],
        "unregistered": [],
    }


# ---------------------------------------------------------------------------
# DependencyMissingLink
# ---------------------------------------------------------------------------


def test_dependency_missing_link_is_frozen() -> None:
    link = DependencyMissingLink(
        start_node_id="alpha",
        missing_node_id="beta",
        state=LifecycleState.DORMANT,
        chain=("alpha", "beta"),
    )
    with pytest.raises((AttributeError, TypeError)):
        link.start_node_id = "other"  # type: ignore[misc]


def test_dependency_missing_link_rejects_chain_mismatch_first() -> None:
    with pytest.raises(CapabilityError):
        DependencyMissingLink(
            start_node_id="alpha",
            missing_node_id="beta",
            state=None,
            chain=("zeta", "beta"),
        )


def test_dependency_missing_link_rejects_chain_mismatch_last() -> None:
    with pytest.raises(CapabilityError):
        DependencyMissingLink(
            start_node_id="alpha",
            missing_node_id="beta",
            state=None,
            chain=("alpha", "zeta"),
        )


def test_dependency_missing_link_rejects_empty_chain() -> None:
    with pytest.raises(CapabilityError):
        DependencyMissingLink(
            start_node_id="alpha",
            missing_node_id="alpha",
            state=None,
            chain=(),
        )


def test_dependency_missing_link_canonical_none_state() -> None:
    link = DependencyMissingLink(
        start_node_id="alpha",
        missing_node_id="alpha",
        state=None,
        chain=("alpha",),
    )
    canonical = link.canonical()
    assert canonical["state"] is None
    assert canonical["chain"] == ["alpha"]


# ---------------------------------------------------------------------------
# RuntimeCapabilityMap
# ---------------------------------------------------------------------------


def test_capability_map_constructs() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(topology, {})
    cap_map = RuntimeCapabilityMap(topology=topology, snapshot=snapshot)
    assert isinstance(cap_map, RuntimeCapabilityMap)


def test_capability_map_rejects_non_topology() -> None:
    with pytest.raises(CapabilityError):
        RuntimeCapabilityMap(
            topology="not a topology",  # type: ignore[arg-type]
            snapshot=_snapshot_with_states(_three_provider_topology(), {}),
        )


def test_capability_map_rejects_non_snapshot() -> None:
    with pytest.raises(CapabilityError):
        RuntimeCapabilityMap(
            topology=_three_provider_topology(),
            snapshot="not a snapshot",  # type: ignore[arg-type]
        )


def test_capability_map_who_provides_filters_to_active() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "learning.closed_loop_a": LifecycleState.HEALTHY,
            "learning.closed_loop_b": LifecycleState.DORMANT,
            # _c never registered
        },
    )
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.who_provides("learning.closed_loop") == ("learning.closed_loop_a",)


def test_capability_map_who_provides_empty_when_all_dormant() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "learning.closed_loop_a": LifecycleState.DORMANT,
            "learning.closed_loop_b": LifecycleState.DECLARED,
            "learning.closed_loop_c": LifecycleState.STOPPED,
        },
    )
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.who_provides("learning.closed_loop") == ()


def test_capability_map_dormant_providers_surfaces_dormant_only() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "learning.closed_loop_a": LifecycleState.HEALTHY,
            "learning.closed_loop_b": LifecycleState.DORMANT,
            "learning.closed_loop_c": LifecycleState.STOPPED,
        },
    )
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.dormant_providers("learning.closed_loop") == (
        "learning.closed_loop_b",
        "learning.closed_loop_c",
    )


def test_capability_map_unregistered_providers_surfaces_silent_drift() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {"learning.closed_loop_a": LifecycleState.HEALTHY},
    )
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.unregistered_providers("learning.closed_loop") == (
        "learning.closed_loop_b",
        "learning.closed_loop_c",
    )


def test_capability_map_resolve_full_breakdown() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "learning.closed_loop_a": LifecycleState.HEALTHY,
            "learning.closed_loop_b": LifecycleState.DORMANT,
            # _c never registered → unregistered
        },
    )
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    res = cap_map.resolve("learning.closed_loop")
    assert res.capability == "learning.closed_loop"
    assert res.declared == (
        "learning.closed_loop_a",
        "learning.closed_loop_b",
        "learning.closed_loop_c",
    )
    assert res.active == ("learning.closed_loop_a",)
    assert res.dormant == ("learning.closed_loop_b",)
    assert res.unregistered == ("learning.closed_loop_c",)
    assert res.is_resolved() is True
    assert res.is_dormant() is False


def test_capability_map_resolve_unknown_returns_empty() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(topology, {})
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    res = cap_map.resolve("not.declared.anywhere")
    assert res.is_missing() is True
    assert res.declared == ()
    assert res.active == ()
    assert res.dormant == ()
    assert res.unregistered == ()


def test_capability_map_who_provides_rejects_non_string() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(topology, {})
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    with pytest.raises(CapabilityError):
        cap_map.who_provides(123)  # type: ignore[arg-type]


def test_capability_map_who_provides_rejects_empty_capability() -> None:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(topology, {})
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    with pytest.raises(CapabilityError):
        cap_map.who_provides("")


def test_capability_map_all_capabilities_sorted_and_deduplicated() -> None:
    topology = RuntimeTopology(
        nodes=(
            _node(
                "alpha",
                capabilities=frozenset({"b.thing", "a.thing"}),
            ),
            _node(
                "beta",
                capabilities=frozenset({"a.thing", "c.thing"}),
            ),
        ),
        edges=(),
    )
    snapshot = _snapshot_with_states(topology, {})
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.all_capabilities() == ("a.thing", "b.thing", "c.thing")


def test_capability_map_unresolved_capabilities() -> None:
    topology = RuntimeTopology(
        nodes=(
            _node(
                "alpha",
                capabilities=frozenset({"resolved.cap"}),
            ),
            _node(
                "beta",
                capabilities=frozenset({"dormant.cap"}),
            ),
        ),
        edges=(),
    )
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.DORMANT,
        },
    )
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.unresolved_capabilities() == ("dormant.cap",)


def test_capability_map_resolution_count() -> None:
    topology = RuntimeTopology(
        nodes=(
            _node(
                "alpha",
                capabilities=frozenset({"a.thing", "b.thing"}),
            ),
        ),
        edges=(),
    )
    snapshot = _snapshot_with_states(topology, {})
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    assert cap_map.resolution_count() == 2


# ---------------------------------------------------------------------------
# DependencyGraphResolver
# ---------------------------------------------------------------------------


def test_dependency_resolver_constructs() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = DependencyGraphResolver(topology=topology, snapshot=snapshot)
    assert isinstance(resolver, DependencyGraphResolver)


def test_dependency_resolver_rejects_non_topology() -> None:
    snapshot = _snapshot_with_states(_depends_topology(), {})
    with pytest.raises(CapabilityError):
        DependencyGraphResolver(
            topology="not a topology",  # type: ignore[arg-type]
            snapshot=snapshot,
        )


def test_dependency_chain_walks_full_chain() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    assert resolver.dependency_chain("alpha") == ("alpha", "beta", "gamma")


def test_dependency_chain_single_node() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    assert resolver.dependency_chain("gamma") == ("gamma",)


def test_dependency_chain_rejects_unknown_node() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    with pytest.raises(CapabilityError):
        resolver.dependency_chain("nonexistent")


def test_dependency_chain_detects_cycle() -> None:
    topology = _cycle_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    with pytest.raises(CapabilityResolutionError):
        resolver.dependency_chain("alpha")


def test_is_reachable_true_when_all_active() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.HEALTHY,
            "gamma": LifecycleState.HEALTHY,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    assert resolver.is_reachable("alpha") is True


def test_is_reachable_false_when_ancestor_dormant() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.HEALTHY,
            "gamma": LifecycleState.DORMANT,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    assert resolver.is_reachable("alpha") is False


def test_missing_link_returns_none_when_all_active() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.HEALTHY,
            "gamma": LifecycleState.HEALTHY,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    assert resolver.missing_link("alpha") is None


def test_missing_link_finds_dormant_ancestor() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.HEALTHY,
            "gamma": LifecycleState.DORMANT,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    link = resolver.missing_link("alpha")
    assert link is not None
    assert link.start_node_id == "alpha"
    assert link.missing_node_id == "gamma"
    assert link.state is LifecycleState.DORMANT
    assert link.chain == ("alpha", "beta", "gamma")


def test_missing_link_finds_unregistered_ancestor() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.HEALTHY,
            # gamma never registered
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    link = resolver.missing_link("alpha")
    assert link is not None
    assert link.missing_node_id == "gamma"
    assert link.state is None
    assert link.chain == ("alpha", "beta", "gamma")


def test_missing_link_returns_first_break_not_deepest() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.DORMANT,
            "gamma": LifecycleState.DORMANT,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    link = resolver.missing_link("alpha")
    assert link is not None
    assert link.missing_node_id == "beta"


def test_missing_link_starts_at_self_when_self_dormant() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.DORMANT,
            "beta": LifecycleState.HEALTHY,
            "gamma": LifecycleState.HEALTHY,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    link = resolver.missing_link("alpha")
    assert link is not None
    assert link.start_node_id == "alpha"
    assert link.missing_node_id == "alpha"
    assert link.chain == ("alpha",)


def test_missing_link_rejects_unknown_node() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    with pytest.raises(CapabilityError):
        resolver.missing_link("nonexistent")


def test_unreachable_nodes_lists_all_broken() -> None:
    topology = _depends_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "alpha": LifecycleState.HEALTHY,
            "beta": LifecycleState.DORMANT,
            "gamma": LifecycleState.HEALTHY,
        },
    )
    resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
    unreachable = resolver.unreachable_nodes()
    assert "alpha" in unreachable
    assert "beta" in unreachable
    assert "gamma" not in unreachable


# ---------------------------------------------------------------------------
# INV-15 determinism
# ---------------------------------------------------------------------------


def _build_canonical_pair() -> tuple[RuntimeTopology, ActivationSnapshot]:
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {
            "learning.closed_loop_a": LifecycleState.HEALTHY,
            "learning.closed_loop_b": LifecycleState.DORMANT,
        },
    )
    return topology, snapshot


def test_inv_15_capability_map_digest_three_runs_identical() -> None:
    digests: list[str] = []
    for _ in range(3):
        topology, snapshot = _build_canonical_pair()
        cap_map = build_capability_map(topology=topology, snapshot=snapshot)
        digests.append(cap_map.digest())
    assert digests[0] == digests[1] == digests[2]


def test_inv_15_capability_map_digest_is_blake2b_128_hex() -> None:
    topology, snapshot = _build_canonical_pair()
    cap_map = build_capability_map(topology=topology, snapshot=snapshot)
    digest = cap_map.digest()
    assert len(digest) == 32
    int(digest, 16)


def test_inv_15_dependency_resolver_digest_three_runs_identical() -> None:
    digests: list[str] = []
    for _ in range(3):
        topology = _depends_topology()
        snapshot = _snapshot_with_states(topology, {"alpha": LifecycleState.HEALTHY})
        resolver = build_dependency_resolver(topology=topology, snapshot=snapshot)
        digests.append(resolver.digest())
    assert digests[0] == digests[1] == digests[2]


def test_inv_15_capability_map_digest_changes_when_snapshot_changes() -> None:
    topology = _three_provider_topology()
    snap_a = _snapshot_with_states(topology, {"learning.closed_loop_a": LifecycleState.HEALTHY})
    snap_b = _snapshot_with_states(topology, {"learning.closed_loop_b": LifecycleState.HEALTHY})
    map_a = build_capability_map(topology=topology, snapshot=snap_a)
    map_b = build_capability_map(topology=topology, snapshot=snap_b)
    assert map_a.digest() != map_b.digest()


# ---------------------------------------------------------------------------
# Lazy seam factory
# ---------------------------------------------------------------------------


def test_factory_default_returns_canonical_version() -> None:
    factory = enable_runtime_capability_factory()
    assert isinstance(factory, RuntimeCapabilityFactory)
    assert factory.version == CAPABILITY_VERSION


def test_factory_accepts_version_override() -> None:
    factory = enable_runtime_capability_factory({"version": "v1.0-test"})
    assert factory.version == "v1.0-test"


def test_factory_rejects_unknown_override() -> None:
    with pytest.raises(CapabilityError):
        enable_runtime_capability_factory({"unknown": "value"})


def test_factory_rejects_non_mapping_overrides() -> None:
    with pytest.raises(CapabilityError):
        enable_runtime_capability_factory("not a mapping")  # type: ignore[arg-type]


def test_factory_builds_capability_map() -> None:
    factory = enable_runtime_capability_factory()
    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(topology, {})
    cap_map = factory.capability_map(topology=topology, snapshot=snapshot)
    assert isinstance(cap_map, RuntimeCapabilityMap)


def test_factory_builds_dependency_resolver() -> None:
    factory = enable_runtime_capability_factory()
    topology = _depends_topology()
    snapshot = _snapshot_with_states(topology, {})
    resolver = factory.dependency_resolver(topology=topology, snapshot=snapshot)
    assert isinstance(resolver, DependencyGraphResolver)


# ---------------------------------------------------------------------------
# Reload idempotency
# ---------------------------------------------------------------------------


def test_reload_yields_byte_identical_digest() -> None:
    import importlib

    import tools.runtime_capability as cap_module

    topology = _three_provider_topology()
    snapshot = _snapshot_with_states(
        topology,
        {"learning.closed_loop_a": LifecycleState.HEALTHY},
    )
    digest_before = cap_module.build_capability_map(topology=topology, snapshot=snapshot).digest()

    reloaded = importlib.reload(cap_module)
    digest_after = reloaded.build_capability_map(topology=topology, snapshot=snapshot).digest()

    assert digest_before == digest_after


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_BANNED_TOP_LEVEL_MODULES = frozenset(
    {
        "subprocess",
        "time",
        "random",
        "asyncio",
        "socket",
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
            f"banned top-level import {imp!r} (root {root!r}) found in tools/runtime_capability.py"
        )


def test_only_allowed_tools_imports() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = _top_level_imports(source)
    tools_imports = sorted(imp for imp in imports if imp.startswith("tools."))
    assert tools_imports == [
        "tools.runtime_activation",
        "tools.runtime_topology",
    ]
