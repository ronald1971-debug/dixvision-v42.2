"""A-11 — tests for :mod:`state.knowledge_graph` + :mod:`state.knowledge_graph_queries`.

Covers:

* Module identity (``NEW_PIP_DEPENDENCIES``, version, exports).
* :class:`NodeKind` / :class:`NodeRecord` / :class:`EdgeRecord` /
  :class:`CausalChain` primitives — frozen + slotted + structurally
  equal.
* :class:`InMemoryGraphTransport` Cypher evaluator — every template in
  :data:`state.knowledge_graph_queries.ALL_QUERIES` is exercised.
* :class:`KnowledgeGraph` coordinator — merge / fetch / chain
  traversal / iter / serialise round-trip.
* INV-15 byte-identical 3-run replay over the in-memory fallback.
* AST guards:

  - No top-level ``neo4j`` import.
  - No top-level clock / random / os / asyncio / httpx / websockets.
  - No ``numpy`` / ``torch`` / ``polars`` / ``pandas`` / ``scipy``.
  - No engine cross-imports (``governance_engine`` / ``execution_engine``
    / ``intelligence_engine`` / ``evolution_engine``).
  - No typed bus event construction (``SignalEvent`` /
    ``ExecutionEvent`` / ``SystemEvent`` / ``HazardEvent`` /
    ``GovernanceDecision`` / ``PatchProposal``).
  - ``# ADAPTED FROM:`` header present, neo4j import confined to
    factory function.
  - Logic module never inlines a Cypher string — every query reference
    resolves to a constant imported from
    :mod:`state.knowledge_graph_queries`.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib

import pytest

import state.knowledge_graph as kg
import state.knowledge_graph_queries as kgq
from state.knowledge_graph import (
    CAUSED_BY,
    KNOWLEDGE_GRAPH_VERSION,
    NEW_PIP_DEPENDENCIES,
    CausalChain,
    EdgeRecord,
    GraphTransport,
    InMemoryGraphTransport,
    KnowledgeGraph,
    KnowledgeGraphError,
    NodeKind,
    NodeRecord,
    neo4j_driver_factory,
)

_KG_PATH = pathlib.Path(kg.__file__)
_KGQ_PATH = pathlib.Path(kgq.__file__)


# ---------------------------------------------------------------------------
# Module identity.
# ---------------------------------------------------------------------------


def test_new_pip_dependencies():
    assert NEW_PIP_DEPENDENCIES == ("neo4j",)


def test_knowledge_graph_version():
    assert KNOWLEDGE_GRAPH_VERSION == "1"


def test_caused_by_constant():
    assert CAUSED_BY == "CAUSED_BY"


def test_all_exports_present():
    for name in (
        "CAUSED_BY",
        "CausalChain",
        "EdgeRecord",
        "GraphTransport",
        "InMemoryGraphTransport",
        "KNOWLEDGE_GRAPH_VERSION",
        "KnowledgeGraph",
        "KnowledgeGraphError",
        "NEW_PIP_DEPENDENCIES",
        "NodeKind",
        "NodeRecord",
        "neo4j_driver_factory",
    ):
        assert name in kg.__all__


def test_query_surface_pinned():
    assert len(kgq.ALL_QUERIES) == 10
    for q in kgq.ALL_QUERIES:
        assert isinstance(q, str) and q
    # All queries are unique
    assert len(set(kgq.ALL_QUERIES)) == len(kgq.ALL_QUERIES)


# ---------------------------------------------------------------------------
# Value-object primitives.
# ---------------------------------------------------------------------------


def test_node_kind_enum():
    assert NodeKind.STRATEGY.value == "Strategy"
    assert NodeKind.REGIME.value == "Regime"
    assert NodeKind.FAILURE.value == "Failure"


def test_node_record_frozen():
    rec = NodeRecord(
        kind=NodeKind.STRATEGY,
        node_id="s-1",
        props={"version": 1},
        ts_ns=42,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.node_id = "s-2"  # type: ignore[misc]


def test_edge_record_frozen():
    rec = EdgeRecord(source_id="s-1", target_id="r-1", weight=0.5, ts_ns=42)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.weight = 0.9  # type: ignore[misc]


def test_causal_chain_frozen():
    chain = CausalChain(node_ids=("a", "b"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        chain.node_ids = ("a", "c")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InMemoryGraphTransport — direct cypher evaluation.
# ---------------------------------------------------------------------------


def test_inmemory_transport_satisfies_protocol():
    transport = InMemoryGraphTransport()
    assert isinstance(transport, GraphTransport)


def test_inmemory_merge_strategy_idempotent():
    transport = InMemoryGraphTransport()
    rows = transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s-1",
        version=1,
        lifecycle="DRAFT",
        ts_ns=100,
    )
    assert rows == ({"id": "s-1"},)
    # Re-merging is allowed and idempotent.
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s-1",
        version=2,
        lifecycle="APPROVED",
        ts_ns=200,
    )
    nodes = transport.run(kgq.FETCH_ALL_NODES)
    assert len(nodes) == 1
    assert nodes[0]["props"]["version"] == 2
    assert nodes[0]["props"]["lifecycle"] == "APPROVED"


def test_inmemory_relabel_rejected():
    transport = InMemoryGraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="x",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(KnowledgeGraphError):
        transport.run(
            kgq.MERGE_REGIME,
            regime_id="x",
            label="BULL",
            ts_ns=2,
        )


def test_inmemory_unknown_cypher():
    transport = InMemoryGraphTransport()
    with pytest.raises(KnowledgeGraphError):
        transport.run("MATCH (n) RETURN n")


def test_inmemory_edge_requires_existing_endpoints():
    transport = InMemoryGraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(KnowledgeGraphError):
        transport.run(
            kgq.MERGE_CAUSED_BY,
            src_id="s",
            dst_id="missing",
            weight=0.5,
            ts_ns=2,
        )


def test_inmemory_self_edge_rejected():
    transport = InMemoryGraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(KnowledgeGraphError):
        transport.run(
            kgq.MERGE_CAUSED_BY,
            src_id="s",
            dst_id="s",
            weight=0.5,
            ts_ns=2,
        )


def test_inmemory_weight_bounds():
    transport = InMemoryGraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="a",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="b",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(KnowledgeGraphError):
        transport.run(
            kgq.MERGE_CAUSED_BY,
            src_id="a",
            dst_id="b",
            weight=1.5,
            ts_ns=1,
        )


def test_inmemory_missing_required_param():
    transport = InMemoryGraphTransport()
    with pytest.raises(KnowledgeGraphError):
        transport.run(kgq.MERGE_STRATEGY, version=1, lifecycle="DRAFT", ts_ns=1)


def test_coordinator_int_param_rejects_bool():
    g = KnowledgeGraph()
    with pytest.raises(KnowledgeGraphError):
        g.merge_strategy(
            strategy_id="s",
            version=True,  # type: ignore[arg-type]
            lifecycle="DRAFT",
            ts_ns=1,
        )


def test_inmemory_close_is_noop():
    transport = InMemoryGraphTransport()
    transport.close()  # should not raise


# ---------------------------------------------------------------------------
# KnowledgeGraph coordinator surface.
# ---------------------------------------------------------------------------


def _populate(graph: KnowledgeGraph) -> None:
    graph.merge_strategy(
        strategy_id="strat-A",
        version=1,
        lifecycle="APPROVED",
        ts_ns=1_000,
    )
    graph.merge_strategy(
        strategy_id="strat-B",
        version=2,
        lifecycle="APPROVED",
        ts_ns=1_500,
    )
    graph.merge_regime(regime_id="reg-bull", label="BULL", ts_ns=900)
    graph.merge_failure(
        failure_id="fail-1",
        kind="SLIPPAGE",
        severity="HIGH",
        ts_ns=2_000,
    )
    graph.merge_caused_by(
        source_id="strat-A",
        target_id="fail-1",
        weight=0.8,
        ts_ns=2_100,
    )
    graph.merge_caused_by(
        source_id="fail-1",
        target_id="reg-bull",
        weight=0.4,
        ts_ns=2_200,
    )
    graph.merge_caused_by(
        source_id="strat-A",
        target_id="strat-B",
        weight=0.6,
        ts_ns=2_300,
    )


def test_kg_default_transport_is_inmemory():
    g = KnowledgeGraph()
    assert isinstance(g._transport, InMemoryGraphTransport)


def test_kg_rejects_non_transport():
    with pytest.raises(TypeError):
        KnowledgeGraph(transport="not a transport")  # type: ignore[arg-type]


def test_kg_fetch_node_round_trip():
    g = KnowledgeGraph()
    _populate(g)
    node = g.fetch_node(node_id="strat-A")
    assert node is not None
    assert node.kind is NodeKind.STRATEGY
    assert node.node_id == "strat-A"
    assert node.props["version"] == 1
    assert node.ts_ns == 1_000


def test_kg_fetch_missing_node_returns_none():
    g = KnowledgeGraph()
    _populate(g)
    assert g.fetch_node(node_id="nonexistent") is None


def test_kg_fetch_outgoing_deterministic_order():
    g = KnowledgeGraph()
    _populate(g)
    out = g.fetch_outgoing(node_id="strat-A")
    targets = [e.target_id for e in out]
    assert targets == sorted(targets)
    assert "fail-1" in targets
    assert "strat-B" in targets


def test_kg_fetch_incoming_deterministic_order():
    g = KnowledgeGraph()
    _populate(g)
    inc = g.fetch_incoming(node_id="fail-1")
    sources = [e.source_id for e in inc]
    assert sources == sorted(sources)
    assert sources == ["strat-A"]


def test_kg_fetch_causal_chain():
    g = KnowledgeGraph()
    _populate(g)
    chains = g.fetch_causal_chain(source_id="strat-A", max_depth=3)
    chain_ids = [tuple(c.node_ids) for c in chains]
    assert ("strat-A", "fail-1") in chain_ids
    assert ("strat-A", "strat-B") in chain_ids
    assert ("strat-A", "fail-1", "reg-bull") in chain_ids


def test_kg_fetch_causal_chain_max_depth_clamps_length():
    g = KnowledgeGraph()
    _populate(g)
    chains = g.fetch_causal_chain(source_id="strat-A", max_depth=1)
    for c in chains:
        assert len(c.node_ids) <= 2


def test_kg_fetch_causal_chain_invalid_depth():
    g = KnowledgeGraph()
    with pytest.raises(KnowledgeGraphError):
        g.fetch_causal_chain(source_id="strat-A", max_depth=0)


def test_kg_fetch_causal_chain_unknown_source():
    g = KnowledgeGraph()
    chains = g.fetch_causal_chain(source_id="ghost", max_depth=5)
    assert chains == ()


def test_kg_iter_nodes_deterministic():
    g = KnowledgeGraph()
    _populate(g)
    ids = [n.node_id for n in g.iter_nodes()]
    # Sorted by (label, id) — labels are Failure, Regime, Strategy.
    assert ids == ["fail-1", "reg-bull", "strat-A", "strat-B"]


def test_kg_iter_edges_deterministic():
    g = KnowledgeGraph()
    _populate(g)
    pairs = [(e.source_id, e.target_id) for e in g.iter_edges()]
    assert pairs == sorted(pairs)


def test_kg_merge_strategy_validation():
    g = KnowledgeGraph()
    with pytest.raises(KnowledgeGraphError):
        g.merge_strategy(strategy_id="", version=1, lifecycle="DRAFT", ts_ns=1)
    with pytest.raises(KnowledgeGraphError):
        g.merge_strategy(strategy_id="s", version="bad", lifecycle="DRAFT", ts_ns=1)  # type: ignore[arg-type]


def test_kg_merge_caused_by_validation():
    g = KnowledgeGraph()
    g.merge_strategy(strategy_id="s", version=1, lifecycle="DRAFT", ts_ns=1)
    g.merge_strategy(strategy_id="t", version=1, lifecycle="DRAFT", ts_ns=1)
    with pytest.raises(KnowledgeGraphError):
        g.merge_caused_by(source_id="s", target_id="t", weight="bad", ts_ns=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Serialisation / round-trip.
# ---------------------------------------------------------------------------


def test_kg_serialize_round_trip():
    g = KnowledgeGraph()
    _populate(g)
    blob = g.serialize()
    rebuilt = KnowledgeGraph.deserialize(blob)
    assert rebuilt.serialize() == blob


def test_kg_serialize_byte_stable():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _populate(g1)
    _populate(g2)
    assert g1.serialize() == g2.serialize()


def test_kg_deserialize_rejects_bad_blob():
    with pytest.raises(KnowledgeGraphError):
        KnowledgeGraph.deserialize(b"{not json")


def test_kg_deserialize_rejects_wrong_version():
    blob = b'{"version":"999","nodes":[],"edges":[]}'
    with pytest.raises(KnowledgeGraphError):
        KnowledgeGraph.deserialize(blob)


def test_kg_deserialize_rejects_non_bytes():
    with pytest.raises(KnowledgeGraphError):
        KnowledgeGraph.deserialize("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay.
# ---------------------------------------------------------------------------


def test_inv15_three_run_byte_identical():
    def run() -> bytes:
        g = KnowledgeGraph()
        _populate(g)
        return g.serialize()

    a = run()
    b = run()
    c = run()
    assert a == b == c


def test_inv15_node_insertion_order_independence():
    g1 = KnowledgeGraph()
    g1.merge_strategy(strategy_id="b", version=1, lifecycle="DRAFT", ts_ns=2)
    g1.merge_strategy(strategy_id="a", version=1, lifecycle="DRAFT", ts_ns=1)

    g2 = KnowledgeGraph()
    g2.merge_strategy(strategy_id="a", version=1, lifecycle="DRAFT", ts_ns=1)
    g2.merge_strategy(strategy_id="b", version=1, lifecycle="DRAFT", ts_ns=2)

    assert g1.serialize() == g2.serialize()


# ---------------------------------------------------------------------------
# neo4j factory.
# ---------------------------------------------------------------------------


def test_neo4j_factory_validation():
    with pytest.raises(ValueError):
        neo4j_driver_factory(uri="")
    with pytest.raises(ValueError):
        neo4j_driver_factory(user="")
    with pytest.raises(TypeError):
        neo4j_driver_factory(password=123)  # type: ignore[arg-type]


def test_neo4j_factory_raises_when_missing(monkeypatch):
    # The neo4j dep may or may not be installed in CI. If it's installed,
    # we skip; if absent, we verify the lazy-import error path raises
    # KnowledgeGraphError.
    try:
        import neo4j  # noqa: F401, PLC0415

        pytest.skip("neo4j installed; factory error path not exercised here")
    except ImportError:
        with pytest.raises(KnowledgeGraphError):
            neo4j_driver_factory()


# ---------------------------------------------------------------------------
# AST guards.
# ---------------------------------------------------------------------------


def _kg_tree() -> ast.AST:
    return ast.parse(_KG_PATH.read_text(encoding="utf-8"))


def _kgq_tree() -> ast.AST:
    return ast.parse(_KGQ_PATH.read_text(encoding="utf-8"))


def test_ast_adapted_from_header_present():
    text = _KG_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: neo4j/neo4j-python-driver" in text


def test_ast_no_toplevel_neo4j_import():
    tree = _kg_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("neo4j")
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("neo4j")


def test_ast_neo4j_import_confined_to_factory():
    tree = _kg_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("neo4j"):
            # Walk up: find the enclosing FunctionDef.
            parents: list[ast.AST] = []
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    if child is node:
                        parents.append(parent)
            # The import must live inside a function body.
            # We don't have direct parent pointers; instead we inspect
            # the function defs and check whether the import node is
            # in their body.
            in_factory = False
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef) and fn.name == "neo4j_driver_factory":
                    for sub in ast.walk(fn):
                        if sub is node:
                            in_factory = True
                            break
            assert in_factory, "neo4j import must live inside neo4j_driver_factory"


_BANNED_TOPLEVEL = (
    "time",
    "datetime",
    "random",
    "secrets",
    "os",
    "asyncio",
    "httpx",
    "requests",
    "websockets",
    "numpy",
    "torch",
    "polars",
    "pandas",
    "scipy",
    "qdrant_client",
)


def test_ast_no_banned_toplevel_imports():
    for path in (_KG_PATH, _KGQ_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in _BANNED_TOPLEVEL, (
                        f"{path.name}: forbidden top-level import {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in _BANNED_TOPLEVEL, (
                    f"{path.name}: forbidden top-level import-from {node.module}"
                )


_FORBIDDEN_ENGINES = (
    "governance_engine",
    "execution_engine",
    "intelligence_engine",
    "evolution_engine",
    "system_engine",
)


def test_ast_no_engine_cross_imports():
    for path in (_KG_PATH, _KGQ_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in _FORBIDDEN_ENGINES, (
                    f"{path.name}: forbidden engine import {node.module}"
                )


_FORBIDDEN_EVENT_CLASSES = (
    "SignalEvent",
    "ExecutionEvent",
    "SystemEvent",
    "HazardEvent",
    "GovernanceDecision",
    "PatchProposal",
    "LearningUpdate",
    "TraderObservation",
)


def test_ast_no_typed_event_construction():
    tree = _kg_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in _FORBIDDEN_EVENT_CLASSES, (
                f"forbidden event constructor call: {node.func.id}"
            )


def test_ast_no_inline_cypher_in_logic_module():
    """Spec line 1197: every Cypher query lives in
    :mod:`state.knowledge_graph_queries`; the logic module must reference
    them by name only, never via inline string literals."""
    tree = _kg_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if any(kw in text for kw in ("MATCH ", "MERGE ", "CREATE ", "RETURN ")):
                # Allowlist: comments/docstrings that *describe* Cypher
                # patterns conceptually are permitted (we filter by
                # presence of explicit ``$`` parameter markers which
                # only the query templates use).
                if "$" in text:
                    pytest.fail(f"inline cypher detected in state.knowledge_graph: {text[:80]!r}")


def test_ast_query_module_constants_only():
    """Cypher templates are constant strings, not f-strings / format calls."""
    tree = _kgq_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            pytest.fail("f-strings forbidden in knowledge_graph_queries")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"format", "format_map"}:
                pytest.fail(f"runtime string formatting forbidden in queries: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Module-shape sanity.
# ---------------------------------------------------------------------------


def test_kg_signature_surface_is_keyword_only():
    """All mutators / readers are keyword-only — INV-15 / call-site clarity."""
    for name in (
        "merge_strategy",
        "merge_regime",
        "merge_failure",
        "merge_caused_by",
        "fetch_node",
        "fetch_outgoing",
        "fetch_incoming",
        "fetch_causal_chain",
    ):
        sig = inspect.signature(getattr(KnowledgeGraph, name))
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name}.{param_name} must be keyword-only"
            )
