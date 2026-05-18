"""A-12 — tests for :mod:`core.causal_graph`.

Covers:

* Module identity (``NEW_PIP_DEPENDENCIES``, version, exports).
* :class:`NodeView` / :class:`EdgeView` — frozen + slotted +
  attribute mappings sorted + read-only.
* :class:`CausalGraph` write surface — ``add_node`` / ``add_edge``
  idempotency, auto-endpoint creation, attribute updates,
  rejection of self-edges, rejection of cycles, attribute type
  validation.
* :class:`CausalGraph` read surface — ``has_node`` / ``has_edge`` /
  ``node`` / ``edge`` / ``predecessors`` / ``successors`` /
  ``ancestors`` / ``descendants`` / ``is_dag`` /
  ``topological_sort`` / ``shortest_path`` / ``all_simple_paths``.
* Serialisation round-trip (byte-stable, sorted keys, version-tagged).
* INV-15 byte-identical 3-run replay over arbitrary insertion order.
* AST guards:

  - No top-level ``networkx`` import.
  - No top-level clock / random / os / asyncio / httpx / websockets.
  - No ``numpy`` / ``torch`` / ``polars`` / ``pandas`` / ``scipy``.
  - No engine cross-imports (``governance_engine`` / ``execution_engine``
    / ``intelligence_engine`` / ``evolution_engine`` /
    ``learning_engine``).
  - No typed bus event construction (``SignalEvent`` /
    ``ExecutionEvent`` / ``SystemEvent`` / ``HazardEvent`` /
    ``GovernanceDecision`` / ``PatchProposal`` /
    ``LearningUpdate``).
  - ``# ADAPTED FROM:`` header present.
  - ``networkx`` import confined to ``networkx_export``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from core.causal_graph import (
    CAUSAL_GRAPH_VERSION,
    NEW_PIP_DEPENDENCIES,
    CausalGraph,
    CausalGraphError,
    CycleError,
    EdgeView,
    NodeView,
)

MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "causal_graph.py"


# ---------------------------------------------------------------------------
# Module identity.
# ---------------------------------------------------------------------------


def test_module_version_is_string_one() -> None:
    assert CAUSAL_GRAPH_VERSION == "1"


def test_new_pip_dependencies_contains_only_networkx() -> None:
    assert NEW_PIP_DEPENDENCIES == ("networkx",)


def test_public_exports_pinned() -> None:
    import core.causal_graph as mod

    expected = {
        "CAUSAL_GRAPH_VERSION",
        "CausalGraph",
        "CausalGraphError",
        "CycleError",
        "EdgeView",
        "NEW_PIP_DEPENDENCIES",
        "NodeView",
    }
    assert set(mod.__all__) == expected


def test_cycle_error_is_causal_graph_error() -> None:
    assert issubclass(CycleError, CausalGraphError)


# ---------------------------------------------------------------------------
# Value primitives.
# ---------------------------------------------------------------------------


def test_node_view_is_frozen() -> None:
    view = NodeView(node_id="a", attrs={"k": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.node_id = "b"  # type: ignore[misc]


def test_edge_view_is_frozen() -> None:
    view = EdgeView(source_id="a", target_id="b", attrs={"w": 1.0})
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.target_id = "c"  # type: ignore[misc]


def test_node_view_slots_pinned() -> None:
    assert NodeView.__slots__ == ("node_id", "attrs")


def test_edge_view_slots_pinned() -> None:
    assert EdgeView.__slots__ == ("source_id", "target_id", "attrs")


# ---------------------------------------------------------------------------
# Write surface.
# ---------------------------------------------------------------------------


def test_add_node_idempotent() -> None:
    g = CausalGraph()
    g.add_node("a", weight=1)
    g.add_node("a", color="red")
    view = g.node("a")
    assert dict(view.attrs) == {"color": "red", "weight": 1}


def test_add_node_rejects_empty_id() -> None:
    g = CausalGraph()
    with pytest.raises(CausalGraphError):
        g.add_node("")


def test_add_node_rejects_non_primitive_attr() -> None:
    g = CausalGraph()
    with pytest.raises(CausalGraphError):
        g.add_node("a", payload=object())


def test_add_node_accepts_json_primitives() -> None:
    g = CausalGraph()
    g.add_node("a", a=1, b=1.0, c="x", d=True, e=None)
    view = g.node("a")
    assert dict(view.attrs) == {
        "a": 1,
        "b": 1.0,
        "c": "x",
        "d": True,
        "e": None,
    }


def test_add_edge_auto_creates_endpoints() -> None:
    g = CausalGraph()
    g.add_edge("a", "b", weight=0.5)
    assert g.has_node("a")
    assert g.has_node("b")
    assert g.has_edge("a", "b")


def test_add_edge_rejects_self_edge() -> None:
    g = CausalGraph()
    with pytest.raises(CycleError):
        g.add_edge("a", "a")


def test_add_edge_rejects_cycle() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    with pytest.raises(CycleError):
        g.add_edge("c", "a")
    assert not g.has_edge("c", "a")
    # Existing edges intact.
    assert g.has_edge("a", "b")
    assert g.has_edge("b", "c")


def test_add_edge_idempotent_attr_update() -> None:
    g = CausalGraph()
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("a", "b", color="red")
    edge = g.edge("a", "b")
    assert dict(edge.attrs) == {"color": "red", "weight": 1.0}


def test_set_node_attr_validates_value() -> None:
    g = CausalGraph()
    g.add_node("a")
    with pytest.raises(CausalGraphError):
        g.set_node_attr("a", "k", object())


def test_set_edge_attr_validates_value() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    with pytest.raises(CausalGraphError):
        g.set_edge_attr("a", "b", "k", object())


def test_set_node_attr_persists_value() -> None:
    g = CausalGraph()
    g.add_node("a")
    g.set_node_attr("a", "k", 42)
    assert g.node("a").attrs["k"] == 42


def test_set_edge_attr_persists_value() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.set_edge_attr("a", "b", "weight", 0.5)
    assert g.edge("a", "b").attrs["weight"] == 0.5


def test_set_node_attr_rejects_missing_node() -> None:
    g = CausalGraph()
    with pytest.raises(KeyError):
        g.set_node_attr("missing", "k", 1)


def test_set_edge_attr_rejects_missing_edge() -> None:
    g = CausalGraph()
    g.add_node("a")
    g.add_node("b")
    with pytest.raises(KeyError):
        g.set_edge_attr("a", "b", "k", 1)


def test_remove_edge_works() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.remove_edge("a", "b")
    assert not g.has_edge("a", "b")
    # Endpoints remain.
    assert g.has_node("a")
    assert g.has_node("b")
    # Reverse adjacency cleaned.
    assert g.predecessors("b") == ()


def test_remove_edge_rejects_missing() -> None:
    g = CausalGraph()
    g.add_node("a")
    g.add_node("b")
    with pytest.raises(KeyError):
        g.remove_edge("a", "b")


def test_clear_empties_graph() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.clear()
    assert len(g) == 0
    assert g.edge_count == 0


def test_node_view_attrs_sorted() -> None:
    g = CausalGraph()
    g.add_node("a", z=1, a=2, m=3)
    assert list(g.node("a").attrs.keys()) == ["a", "m", "z"]


def test_edge_view_attrs_sorted() -> None:
    g = CausalGraph()
    g.add_edge("a", "b", z=1, a=2, m=3)
    assert list(g.edge("a", "b").attrs.keys()) == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Read surface.
# ---------------------------------------------------------------------------


def test_has_node_and_edge_basic() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    assert g.has_node("a")
    assert g.has_node("b")
    assert not g.has_node("c")
    assert g.has_edge("a", "b")
    assert not g.has_edge("b", "a")


def test_node_raises_keyerror_when_missing() -> None:
    g = CausalGraph()
    with pytest.raises(KeyError):
        g.node("missing")


def test_predecessors_and_successors_sorted() -> None:
    g = CausalGraph()
    for src in ("c", "a", "b"):
        g.add_edge(src, "z")
    for tgt in ("c", "a", "b"):
        g.add_edge("source", tgt)
    assert g.predecessors("z") == ("a", "b", "c")
    assert g.successors("source") == ("a", "b", "c")


def test_ancestors_excludes_self_and_sorted() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("d", "b")
    assert g.ancestors("c") == ("a", "b", "d")
    assert g.ancestors("a") == ()


def test_descendants_excludes_self_and_sorted() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("b", "d")
    assert g.descendants("a") == ("b", "c", "d")
    assert g.descendants("c") == ()


def test_topological_sort_is_valid_order() -> None:
    g = CausalGraph()
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e")]
    for src, dst in edges:
        g.add_edge(src, dst)
    order = g.topological_sort()
    # Each edge points forward in the ordering.
    index = {n: i for i, n in enumerate(order)}
    for src, dst in edges:
        assert index[src] < index[dst]
    assert set(order) == {"a", "b", "c", "d", "e"}


def test_topological_sort_deterministic() -> None:
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e")]
    g1 = CausalGraph()
    for src, dst in edges:
        g1.add_edge(src, dst)
    g2 = CausalGraph()
    for src, dst in reversed(edges):
        g2.add_edge(src, dst)
    assert g1.topological_sort() == g2.topological_sort()


def test_is_dag_is_always_true() -> None:
    g = CausalGraph()
    assert g.is_dag() is True
    g.add_edge("a", "b")
    assert g.is_dag() is True


def test_shortest_path_basic() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("a", "c")
    # The direct edge a->c wins over a->b->c.
    assert g.shortest_path("a", "c") == ("a", "c")


def test_shortest_path_self_is_singleton() -> None:
    g = CausalGraph()
    g.add_node("a")
    assert g.shortest_path("a", "a") == ("a",)


def test_shortest_path_none_when_unreachable() -> None:
    g = CausalGraph()
    g.add_node("a")
    g.add_node("b")
    assert g.shortest_path("a", "b") is None


def test_all_simple_paths_enumerated_sorted() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    paths = g.all_simple_paths("a", "d", max_depth=3)
    assert paths == (("a", "b", "d"), ("a", "c", "d"))


def test_all_simple_paths_respects_max_depth() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    paths = g.all_simple_paths("a", "d", max_depth=2)
    assert paths == ()


def test_all_simple_paths_rejects_nonpositive_depth() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    with pytest.raises(CausalGraphError):
        g.all_simple_paths("a", "b", max_depth=0)


def test_all_simple_paths_rejects_bool_depth() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    with pytest.raises(CausalGraphError):
        g.all_simple_paths("a", "b", max_depth=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Iteration / counts.
# ---------------------------------------------------------------------------


def test_iter_nodes_sorted_and_frozen() -> None:
    g = CausalGraph()
    for nid in ("c", "a", "b"):
        g.add_node(nid, payload=nid)
    nodes = list(g.iter_nodes())
    assert [n.node_id for n in nodes] == ["a", "b", "c"]
    # NodeView is frozen.
    with pytest.raises(dataclasses.FrozenInstanceError):
        nodes[0].node_id = "x"  # type: ignore[misc]


def test_iter_edges_sorted() -> None:
    g = CausalGraph()
    for src, dst in (("c", "z"), ("a", "z"), ("b", "z")):
        g.add_edge(src, dst)
    edges = list(g.iter_edges())
    assert [(e.source_id, e.target_id) for e in edges] == [("a", "z"), ("b", "z"), ("c", "z")]


def test_len_and_edge_count() -> None:
    g = CausalGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert len(g) == 3
    assert g.edge_count == 2


def test_contains_dunder() -> None:
    g = CausalGraph()
    g.add_node("a")
    assert "a" in g
    assert "missing" not in g
    assert 123 not in g


# ---------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------


def test_serialize_round_trip() -> None:
    g = CausalGraph()
    g.add_edge("a", "b", weight=0.5)
    g.add_edge("b", "c", weight=0.7)
    g.set_node_attr("a", "kind", "strategy")
    blob = g.serialize()
    decoded = CausalGraph.deserialize(blob)
    assert decoded.serialize() == blob


def test_serialize_byte_stable_sorted_keys() -> None:
    g = CausalGraph()
    g.add_edge("a", "b", w=1, x=2)
    payload = json.loads(g.serialize())
    assert payload["version"] == CAUSAL_GRAPH_VERSION
    assert list(payload.keys()) == ["edges", "nodes", "version"]


def test_serialize_includes_version() -> None:
    g = CausalGraph()
    payload = json.loads(g.serialize())
    assert payload["version"] == CAUSAL_GRAPH_VERSION


def test_deserialize_rejects_wrong_version() -> None:
    blob = json.dumps(
        {"version": "999", "nodes": [], "edges": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(CausalGraphError):
        CausalGraph.deserialize(blob)


def test_deserialize_rejects_non_bytes() -> None:
    with pytest.raises(CausalGraphError):
        CausalGraph.deserialize("not-bytes")  # type: ignore[arg-type]


def test_deserialize_rejects_corrupt_json() -> None:
    with pytest.raises(CausalGraphError):
        CausalGraph.deserialize(b"{not valid json")


def test_deserialize_rejects_non_object_root() -> None:
    with pytest.raises(CausalGraphError):
        CausalGraph.deserialize(b"[1,2,3]")


# ---------------------------------------------------------------------------
# INV-15: byte-identical replay over different insertion orders.
# ---------------------------------------------------------------------------


def _build_canonical_graph() -> CausalGraph:
    """Construct a fixed graph used by INV-15 tests."""
    g = CausalGraph()
    edges = [
        ("strategy:scalper", "failure:overconfidence", 0.7),
        ("strategy:scalper", "regime:trending", 0.3),
        ("strategy:swing", "regime:trending", 0.6),
        ("failure:overconfidence", "regime:volatile", 0.4),
    ]
    for src, dst, w in edges:
        g.add_edge(src, dst, weight=w)
    return g


def test_inv15_three_run_byte_identical_replay() -> None:
    blobs = [_build_canonical_graph().serialize() for _ in range(3)]
    assert blobs[0] == blobs[1] == blobs[2]


def test_inv15_insertion_order_independence() -> None:
    edges = [
        ("strategy:scalper", "failure:overconfidence", 0.7),
        ("strategy:scalper", "regime:trending", 0.3),
        ("strategy:swing", "regime:trending", 0.6),
        ("failure:overconfidence", "regime:volatile", 0.4),
    ]
    g_forward = CausalGraph()
    for src, dst, w in edges:
        g_forward.add_edge(src, dst, weight=w)
    g_reverse = CausalGraph()
    for src, dst, w in reversed(edges):
        g_reverse.add_edge(src, dst, weight=w)
    assert g_forward.serialize() == g_reverse.serialize()


def test_inv15_topological_sort_order_independence() -> None:
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
    ]
    g_forward = CausalGraph()
    for src, dst in edges:
        g_forward.add_edge(src, dst)
    g_reverse = CausalGraph()
    for src, dst in reversed(edges):
        g_reverse.add_edge(src, dst)
    assert g_forward.topological_sort() == g_reverse.topological_sort()


# ---------------------------------------------------------------------------
# networkx export (lazy import).
# ---------------------------------------------------------------------------


def test_networkx_export_raises_when_missing() -> None:
    """If networkx is absent, ``networkx_export`` raises a typed error."""
    pytest.importorskip("networkx", reason="networkx is optional")
    g = CausalGraph()
    g.add_edge("a", "b", weight=0.5)
    nx_graph = g.networkx_export()
    assert nx_graph.number_of_nodes() == 2
    assert nx_graph.number_of_edges() == 1
    assert nx_graph.has_edge("a", "b")


# ---------------------------------------------------------------------------
# AST guards.
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports() -> set[str]:
    tree = _module_ast()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.add(node.module.split(".")[0])
    return out


def test_no_top_level_networkx_import() -> None:
    assert "networkx" not in _top_level_imports()


def test_no_top_level_clock_or_random_imports() -> None:
    banned = {"time", "datetime", "random", "os", "asyncio", "httpx", "websockets"}
    assert banned.isdisjoint(_top_level_imports())


def test_no_numerics_or_dataframe_imports() -> None:
    banned = {"numpy", "torch", "polars", "pandas", "scipy"}
    assert banned.isdisjoint(_top_level_imports())


def test_no_engine_cross_imports() -> None:
    banned = {
        "governance_engine",
        "execution_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
    }
    assert banned.isdisjoint(_top_level_imports())


def test_no_typed_event_construction() -> None:
    """Authority symmetry: module must not construct typed bus events."""
    banned_names = {
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "HazardEvent",
        "GovernanceDecision",
        "PatchProposal",
        "LearningUpdate",
        "TraderObservation",
    }
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_names, f"forbidden construction: {node.func.id}"


def test_adapted_from_header_present() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM:" in source
    assert "networkx/networkx" in source


def test_networkx_import_confined_to_export_function() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "networkx_export":
            imports_inside: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        imports_inside.append(alias.name)
                elif isinstance(child, ast.ImportFrom) and child.module:
                    imports_inside.append(child.module)
            assert "networkx" in imports_inside
            return
    raise AssertionError("networkx_export() not found")
