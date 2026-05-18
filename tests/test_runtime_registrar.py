"""Tests for ``ui/harness/runtime_registrar.py`` (PR-RT-4).

These tests pin the boot-wiring half of the Runtime Topology
Authority chain:

* module identity (constants, exports, banned-import AST guards)
* declared topology invariants (node count, edge count, digest,
  INV-15 byte-identical three-run determinism)
* ``register_at_boot`` with a fully populated ``_State`` stand-in
  (every declared attribute non-None -> every node STARTED)
* ``register_at_boot`` with a sparse ``_State`` stand-in
  (selected attributes None -> selected nodes DORMANT)
* :meth:`HarnessRuntimeRegistrar.declared_topology_view`,
  :meth:`active_view`, :meth:`dormant_view`,
  :meth:`capability_view` projection shapes
* capability resolution against a partially-active boot snapshot
* dependency resolver traversing the declared topology
* operator-route smoke tests through a :class:`TestClient`
* route-registrar canonical inventory updated
* AST guards forbidding top-level vendor / engine imports.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.runtime_activation import (
    RuntimeActivationRegistry,
)
from tools.runtime_capability import (
    DependencyGraphResolver,
    RuntimeCapabilityMap,
)
from tools.runtime_topology import (
    EdgeRelation,
    NodeKind,
    NodeTier,
    RuntimeTopology,
)
from ui.harness.runtime_registrar import (
    REGISTRAR_VERSION,
    HarnessRuntimeRegistrar,
    declared_node_ids,
    declared_state_attr_for,
    declared_topology,
)

MODULE_PATH = Path(__file__).resolve().parents[1] / "ui" / "harness" / "runtime_registrar.py"


# ---------------------------------------------------------------------------
# Module identity / constants
# ---------------------------------------------------------------------------


def test_registrar_version_constant() -> None:
    assert REGISTRAR_VERSION == "v1.0-RT4"


def test_declared_node_ids_is_sorted_and_unique() -> None:
    ids = declared_node_ids()
    assert list(ids) == sorted(ids)
    assert len(ids) == len(set(ids))


def test_declared_node_ids_count() -> None:
    assert len(declared_node_ids()) == 28


def test_declared_state_attr_for_known_node() -> None:
    assert declared_state_attr_for("intelligence_engine") == "intelligence"
    assert declared_state_attr_for("execution_engine") == "execution"
    assert declared_state_attr_for("governance_engine") == "governance"
    assert declared_state_attr_for("closed_learning_loop") == "closed_learning_loop"


def test_declared_state_attr_for_unknown_node() -> None:
    assert declared_state_attr_for("does_not_exist") is None


def test_declared_node_ids_match_declared_topology_nodes() -> None:
    topology = declared_topology()
    topo_ids = sorted(n.node_id for n in topology.nodes)
    assert topo_ids == list(declared_node_ids())


# ---------------------------------------------------------------------------
# Declared topology invariants
# ---------------------------------------------------------------------------


def test_declared_topology_is_runtime_topology() -> None:
    assert isinstance(declared_topology(), RuntimeTopology)


def test_declared_topology_node_count() -> None:
    assert declared_topology().node_count() == 28


def test_declared_topology_edge_count() -> None:
    assert declared_topology().edge_count() == 24


def test_declared_topology_kind_coverage() -> None:
    topology = declared_topology()
    kinds = {n.kind for n in topology.nodes}
    # All canonical engine/loop/registry/gate/policy/adapter/sensor
    # kinds we declare are present.
    assert kinds == {
        NodeKind.ENGINE,
        NodeKind.LOOP,
        NodeKind.REGISTRY,
        NodeKind.GATE,
        NodeKind.POLICY,
        NodeKind.ADAPTER,
        NodeKind.SENSOR,
    }


def test_declared_topology_tier_coverage() -> None:
    topology = declared_topology()
    tiers = {n.tier for n in topology.nodes}
    assert tiers == {NodeTier.T0, NodeTier.T1, NodeTier.T2}


def test_declared_topology_relation_coverage() -> None:
    topology = declared_topology()
    relations = {e.relation for e in topology.edges}
    # The canonical edge set exercises every relation kind the
    # topology contract knows about except OWNS (reserved for
    # future PRs that introduce composite ownership). PR-RT-4
    # explicitly anchors the relations actually used.
    assert relations <= {
        EdgeRelation.PRODUCES,
        EdgeRelation.CONSUMES,
        EdgeRelation.OWNS,
        EdgeRelation.GATES,
        EdgeRelation.DEPENDS_ON,
        EdgeRelation.PROJECTS,
    }
    # At least these four must appear so the dependency graph + the
    # capability resolver have non-trivial structure to walk.
    assert EdgeRelation.PRODUCES in relations
    assert EdgeRelation.DEPENDS_ON in relations
    assert EdgeRelation.PROJECTS in relations


def test_declared_topology_edges_reference_declared_nodes() -> None:
    topology = declared_topology()
    declared_ids = set(declared_node_ids())
    for edge in topology.edges:
        assert edge.source_id in declared_ids, edge
        assert edge.target_id in declared_ids, edge


def test_declared_topology_capabilities_are_nonempty() -> None:
    topology = declared_topology()
    for node in topology.nodes:
        assert node.capabilities, node.node_id


def test_declared_topology_digest_is_stable() -> None:
    """INV-15 byte-identical declared topology digest."""

    d1 = declared_topology().digest()
    d2 = declared_topology().digest()
    d3 = declared_topology().digest()
    assert d1 == d2 == d3


def test_declared_topology_digest_pinned() -> None:
    """INV-15 byte-identical declared topology digest — pin the
    canonical value so any silent change to the declared topology
    surfaces in CI as a digest mismatch (the operator must update
    this test deliberately whenever they extend the topology)."""

    # The digest is BLAKE2b-128 of the canonical RuntimeTopology
    # serialization. Computed by building HarnessRuntimeRegistrar
    # at the time PR-RT-4 was landed; refresh this value when the
    # declared topology is intentionally extended.
    assert declared_topology().digest() == "291212644c5e4431fb11f2a31ae6d8c4"


# ---------------------------------------------------------------------------
# HarnessRuntimeRegistrar construction
# ---------------------------------------------------------------------------


def test_registrar_constructs_empty() -> None:
    reg = HarnessRuntimeRegistrar()
    assert isinstance(reg.topology, RuntimeTopology)
    assert isinstance(reg.registry, RuntimeActivationRegistry)


def test_registrar_topology_matches_declared() -> None:
    reg = HarnessRuntimeRegistrar()
    assert reg.topology.digest() == declared_topology().digest()


def test_registrar_snapshot_is_empty_before_boot() -> None:
    reg = HarnessRuntimeRegistrar()
    snap = reg.snapshot()
    assert snap.active_node_ids() == frozenset()
    assert snap.dormant_node_ids() == frozenset()


def test_registrar_capability_map_is_callable_before_boot() -> None:
    reg = HarnessRuntimeRegistrar()
    cap_map = reg.capability_map()
    assert isinstance(cap_map, RuntimeCapabilityMap)


def test_registrar_dependency_resolver_is_callable_before_boot() -> None:
    reg = HarnessRuntimeRegistrar()
    resolver = reg.dependency_resolver()
    assert isinstance(resolver, DependencyGraphResolver)


# ---------------------------------------------------------------------------
# register_at_boot
# ---------------------------------------------------------------------------


class _FullState:
    """``_State`` stand-in where every declared attribute is non-None."""

    def __init__(self) -> None:
        for node_id in declared_node_ids():
            attr = declared_state_attr_for(node_id)
            assert attr is not None
            setattr(self, attr, object())


class _SparseState:
    """``_State`` stand-in where only a subset of attributes are set.

    Used to drive the dormant / silent-drift surface assertions.
    """

    def __init__(self, present: set[str]) -> None:
        for node_id in declared_node_ids():
            attr = declared_state_attr_for(node_id)
            assert attr is not None
            if node_id in present:
                setattr(self, attr, object())
            else:
                setattr(self, attr, None)


def test_register_at_boot_full_state_all_started() -> None:
    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_FullState())
    snap = reg.snapshot()
    assert snap.active_node_ids() == frozenset(declared_node_ids())
    assert snap.dormant_node_ids() == frozenset()


def test_register_at_boot_empty_state_all_dormant() -> None:
    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_SparseState(present=set()))
    snap = reg.snapshot()
    assert snap.active_node_ids() == frozenset()
    assert snap.dormant_node_ids() == frozenset(declared_node_ids())


def test_register_at_boot_sparse_state_partitions_active_vs_dormant() -> None:
    present = {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
    }
    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_SparseState(present=present))
    snap = reg.snapshot()
    assert snap.active_node_ids() == frozenset(present)
    assert snap.dormant_node_ids() == frozenset(declared_node_ids()) - frozenset(present)


def test_register_at_boot_is_idempotent_for_same_state() -> None:
    reg = HarnessRuntimeRegistrar()
    state = _FullState()
    reg.register_at_boot(state)
    digest1 = reg.snapshot().digest()
    reg.register_at_boot(state)
    digest2 = reg.snapshot().digest()
    assert digest1 == digest2


def test_register_at_boot_treats_missing_attribute_as_dormant() -> None:
    """An attribute that is *missing* from ``_State`` (not just
    None) must still register the node as DORMANT, not raise."""

    class _MissingState:
        intelligence = object()  # the only attribute present

    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_MissingState())
    snap = reg.snapshot()
    assert snap.active_node_ids() == frozenset({"intelligence_engine"})
    assert "execution_engine" in snap.dormant_node_ids()


# ---------------------------------------------------------------------------
# Read projections (operator route shapes)
# ---------------------------------------------------------------------------


def _booted_registrar() -> HarnessRuntimeRegistrar:
    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_FullState())
    return reg


def test_declared_topology_view_shape() -> None:
    view = _booted_registrar().declared_topology_view()
    assert view["version"] == REGISTRAR_VERSION
    assert view["node_count"] == 28
    assert view["edge_count"] == 24
    assert isinstance(view["topology_digest"], str)
    assert len(view["topology_digest"]) == 32
    assert isinstance(view["nodes"], list)
    assert isinstance(view["edges"], list)
    assert len(view["nodes"]) == 28
    assert len(view["edges"]) == 24


def test_declared_topology_view_is_byte_stable() -> None:
    """INV-15: serialising the declared topology view yields the same
    bytes across three independent registrars."""

    def _serialize() -> bytes:
        view = HarnessRuntimeRegistrar().declared_topology_view()
        return json.dumps(view, sort_keys=True, separators=(",", ":")).encode("utf-8")

    s1 = _serialize()
    s2 = _serialize()
    s3 = _serialize()
    assert s1 == s2 == s3
    # And the digest is non-trivial.
    assert hashlib.blake2b(s1, digest_size=16).hexdigest() != "0" * 32


def test_active_view_shape() -> None:
    view = _booted_registrar().active_view()
    assert view["version"] == REGISTRAR_VERSION
    assert isinstance(view["topology_digest"], str)
    assert isinstance(view["snapshot_digest"], str)
    assert isinstance(view["active_node_ids"], list)
    assert view["active_node_ids"] == sorted(declared_node_ids())


def test_active_view_against_sparse_state() -> None:
    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_SparseState(present={"intelligence_engine", "execution_engine"}))
    view = reg.active_view()
    assert view["active_node_ids"] == ["execution_engine", "intelligence_engine"]


def test_dormant_view_shape() -> None:
    view = _booted_registrar().dormant_view()
    assert view["version"] == REGISTRAR_VERSION
    assert isinstance(view["topology_digest"], str)
    assert isinstance(view["snapshot_digest"], str)
    assert isinstance(view["dormant_node_ids"], list)
    assert isinstance(view["unregistered_node_ids"], list)
    # Full boot -> nothing dormant.
    assert view["dormant_node_ids"] == []
    assert view["unregistered_node_ids"] == []


def test_dormant_view_surfaces_silent_drift() -> None:
    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_SparseState(present={"intelligence_engine", "execution_engine"}))
    view = reg.dormant_view()
    dormant = set(view["dormant_node_ids"])
    assert "closed_learning_loop" in dormant
    assert "structural_evolution_loop" in dormant
    assert "intelligence_engine" not in dormant
    assert "execution_engine" not in dormant


def test_capability_view_resolved() -> None:
    view = _booted_registrar().capability_view("intelligence.signal")
    assert view["version"] == REGISTRAR_VERSION
    assert view["capability"] == "intelligence.signal"
    assert view["declared"] == ["intelligence_engine"]
    assert view["active"] == ["intelligence_engine"]
    assert view["dormant"] == []
    assert view["unregistered"] == []
    assert view["is_resolved"] is True
    assert view["is_dormant"] is False
    assert view["is_missing"] is False
    assert view["has_unregistered_providers"] is False


def test_capability_view_dormant_provider() -> None:
    """A capability whose only declared provider is dormant must be
    reported as ``is_dormant=True`` and ``is_resolved=False``."""

    reg = HarnessRuntimeRegistrar()
    reg.register_at_boot(_SparseState(present={"intelligence_engine", "execution_engine"}))
    view = reg.capability_view("learning.closed_loop")
    assert view["declared"] == ["closed_learning_loop"]
    assert view["active"] == []
    assert view["dormant"] == ["closed_learning_loop"]
    assert view["is_resolved"] is False
    assert view["is_dormant"] is True


def test_capability_view_unknown_capability() -> None:
    view = _booted_registrar().capability_view("does.not.exist")
    assert view["declared"] == []
    assert view["active"] == []
    assert view["dormant"] == []
    assert view["is_resolved"] is False
    assert view["is_missing"] is True


def test_capability_view_invalid_capability_raises() -> None:
    from tools.runtime_capability import CapabilityError

    with pytest.raises(CapabilityError):
        _booted_registrar().capability_view("")


# ---------------------------------------------------------------------------
# Snapshot digest INV-15 determinism
# ---------------------------------------------------------------------------


def test_snapshot_digest_byte_identical_three_runs() -> None:
    """INV-15: building three independent registrars with the same
    boot order produces byte-identical activation snapshot digests."""

    def _digest() -> str:
        reg = HarnessRuntimeRegistrar()
        reg.register_at_boot(_FullState())
        return reg.snapshot().digest()

    d1 = _digest()
    d2 = _digest()
    d3 = _digest()
    assert d1 == d2 == d3


def test_snapshot_digest_differs_for_different_topologies() -> None:
    """A registrar booted with all-STARTED vs all-DORMANT must
    produce *different* snapshot digests."""

    full = HarnessRuntimeRegistrar()
    full.register_at_boot(_FullState())

    empty = HarnessRuntimeRegistrar()
    empty.register_at_boot(_SparseState(present=set()))

    assert full.snapshot().digest() != empty.snapshot().digest()


# ---------------------------------------------------------------------------
# Operator routes — TestClient smoke
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _test_client() -> Any:
    """Boot the harness once for route smoke tests.

    The harness boot requires either a persistent ledger path or
    ``DIXVISION_PERMIT_EPHEMERAL_LEDGER=1``; the latter is set for
    test scope only. We import ``ui.server`` lazily so the
    fixture isolates the side effects (the harness boots STATE
    at module import time)."""

    import os

    os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")
    from fastapi.testclient import TestClient

    import ui.server as server  # noqa: WPS433  (lazy import is intentional)

    return TestClient(server.app)


def test_route_topology_returns_200(_test_client: Any) -> None:
    resp = _test_client.get("/api/operator/runtime/topology")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == REGISTRAR_VERSION
    assert body["node_count"] == 28
    assert body["edge_count"] == 24


def test_route_active_returns_200(_test_client: Any) -> None:
    resp = _test_client.get("/api/operator/runtime/active")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == REGISTRAR_VERSION
    assert isinstance(body["active_node_ids"], list)


def test_route_dormant_returns_200(_test_client: Any) -> None:
    resp = _test_client.get("/api/operator/runtime/dormant")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == REGISTRAR_VERSION
    assert isinstance(body["dormant_node_ids"], list)


def test_route_capability_returns_200(_test_client: Any) -> None:
    resp = _test_client.get("/api/operator/runtime/capability/intelligence.signal")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == REGISTRAR_VERSION
    assert body["capability"] == "intelligence.signal"


def test_route_capability_unknown_resolves_missing(_test_client: Any) -> None:
    resp = _test_client.get("/api/operator/runtime/capability/does.not.exist")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_missing"] is True


def test_route_registrar_audit_passes_with_new_routes(
    _test_client: Any,
) -> None:
    """The boot-time route registrar audit (P1.4) must accept the four
    new ``/api/operator/runtime/*`` routes — exercised through the
    same JSON inventory the operator dashboard consumes."""

    resp = _test_client.get("/api/admin/route_inventory")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["unexpected"] == []
    paths_in_operator_domain = {
        (r["method"], r["path"])
        for domain in body["domains"]
        if domain["name"] == "operator"
        for r in domain["routes"]
    }
    assert ("GET", "/api/operator/runtime/topology") in paths_in_operator_domain
    assert ("GET", "/api/operator/runtime/active") in paths_in_operator_domain
    assert ("GET", "/api/operator/runtime/dormant") in paths_in_operator_domain
    assert (
        "GET",
        "/api/operator/runtime/capability/{tag}",
    ) in paths_in_operator_domain


# ---------------------------------------------------------------------------
# Re-import determinism
# ---------------------------------------------------------------------------


def test_module_reimport_preserves_declared_topology_digest() -> None:
    import ui.harness.runtime_registrar as mod

    d_before = mod.declared_topology().digest()
    reloaded = importlib.reload(mod)
    d_after = reloaded.declared_topology().digest()
    assert d_before == d_after


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
        "fastapi",
        "uvicorn",
        "pydantic",
        # Engine tiers — the registrar lives in ui.harness and must
        # never reach into runtime engine code at module load.
        "core_engine",
        "execution_engine",
        "governance_engine",
        "intelligence_engine",
        "system_engine",
        "learning_engine",
        "evolution_engine",
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
            f"banned top-level import {imp!r} (root {root!r}) found "
            f"in ui/harness/runtime_registrar.py"
        )


def test_no_top_level_ui_server_import() -> None:
    """``ui.server`` (the harness god-object) must never be imported
    at module load time — only inside the ``TYPE_CHECKING`` block."""

    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = _top_level_imports(source)
    for imp in imports:
        assert not imp.startswith("ui.server"), (
            f"top-level ui.server import {imp!r} would couple the "
            f"registrar to harness boot order; keep it inside the "
            f"TYPE_CHECKING block."
        )


def test_only_allowed_top_level_imports() -> None:
    """Top-level imports must come from stdlib or from
    ``tools.runtime_*`` — the registrar is a pure projection module
    and must not pick up dashboard / engine / vendor packages."""

    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = _top_level_imports(source)
    allowed_roots = {
        # stdlib (sample — extend as needed)
        "__future__",
        "dataclasses",
        "typing",
        "enum",
        "collections",
        # the only intra-repo dependency allowed at module load
        "tools",
    }
    for imp in imports:
        root = imp.split(".")[0]
        assert root in allowed_roots, (
            f"unexpected top-level import {imp!r} (root {root!r}) "
            f"in ui/harness/runtime_registrar.py"
        )
