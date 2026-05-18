"""C-22 — tests for :mod:`state.knowledge_store_memgraph` +
:mod:`state.knowledge_store_memgraph_queries`.

Mirrors the A-11 :mod:`tests.test_knowledge_graph` suite. Covers:

* Module identity (``NEW_PIP_DEPENDENCIES``, version, exports).
* :class:`NodeKind` / :class:`NodeRecord` / :class:`EdgeRecord` /
  :class:`CausalChain` primitives — frozen + slotted + structurally
  equal.
* :class:`InMemoryMemgraphTransport` Cypher evaluator — every template
  in :data:`state.knowledge_store_memgraph_queries.ALL_QUERIES` is
  exercised.
* :class:`MemgraphKnowledgeStore` coordinator — merge / fetch / chain
  traversal / iter / serialise round-trip.
* INV-15 byte-identical 3-run replay over the in-memory fallback.
* AST guards:

  - No top-level ``gqlalchemy`` import.
  - No top-level clock / random / os / asyncio / httpx / websockets.
  - No ``numpy`` / ``torch`` / ``polars`` / ``pandas`` / ``scipy``.
  - No engine cross-imports.
  - No typed bus event construction.
  - ``# ADAPTED FROM:`` header present, gqlalchemy import confined to
    factory function.
  - Logic module never inlines a Cypher string — every query reference
    resolves to a constant imported from the queries module.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib

import pytest

import state.knowledge_store_memgraph as kg
import state.knowledge_store_memgraph_queries as kgq
from state.knowledge_store_memgraph import (
    CAUSED_BY,
    MEMGRAPH_STORE_VERSION,
    NEW_PIP_DEPENDENCIES,
    CausalChain,
    EdgeRecord,
    InMemoryMemgraphTransport,
    MemgraphKnowledgeStore,
    MemgraphKnowledgeStoreError,
    MemgraphTransport,
    NodeKind,
    NodeRecord,
    memgraph_client_factory,
)

_KG_PATH = pathlib.Path(kg.__file__)
_KGQ_PATH = pathlib.Path(kgq.__file__)


# ---------------------------------------------------------------------------
# Module identity.
# ---------------------------------------------------------------------------


def test_new_pip_dependencies():
    assert NEW_PIP_DEPENDENCIES == ("gqlalchemy",)


def test_memgraph_store_version():
    assert MEMGRAPH_STORE_VERSION == "1"


def test_caused_by_constant():
    assert CAUSED_BY == "CAUSED_BY"


def test_all_exports_present():
    for name in (
        "CAUSED_BY",
        "CausalChain",
        "EdgeRecord",
        "InMemoryMemgraphTransport",
        "MEMGRAPH_STORE_VERSION",
        "MemgraphKnowledgeStore",
        "MemgraphKnowledgeStoreError",
        "MemgraphTransport",
        "NEW_PIP_DEPENDENCIES",
        "NodeKind",
        "NodeRecord",
        "memgraph_client_factory",
    ):
        assert name in kg.__all__


def test_query_surface_pinned():
    assert len(kgq.ALL_QUERIES) == 10
    for q in kgq.ALL_QUERIES:
        assert isinstance(q, str) and q
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
# InMemoryMemgraphTransport — direct cypher evaluation.
# ---------------------------------------------------------------------------


def test_inmemory_transport_satisfies_protocol():
    transport = InMemoryMemgraphTransport()
    assert isinstance(transport, MemgraphTransport)


def test_inmemory_merge_strategy_idempotent():
    transport = InMemoryMemgraphTransport()
    rows = transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s-1",
        version=1,
        lifecycle="DRAFT",
        ts_ns=100,
    )
    assert rows == ({"id": "s-1"},)
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
    transport = InMemoryMemgraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="x",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(MemgraphKnowledgeStoreError):
        transport.run(
            kgq.MERGE_REGIME,
            regime_id="x",
            label="BULL",
            ts_ns=2,
        )


def test_inmemory_unknown_cypher():
    transport = InMemoryMemgraphTransport()
    with pytest.raises(MemgraphKnowledgeStoreError):
        transport.run("MATCH (n) RETURN n")


def test_inmemory_edge_requires_existing_endpoints():
    transport = InMemoryMemgraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(MemgraphKnowledgeStoreError):
        transport.run(
            kgq.MERGE_CAUSED_BY,
            src_id="s",
            dst_id="missing",
            weight=0.5,
            ts_ns=2,
        )


def test_inmemory_self_edge_rejected():
    transport = InMemoryMemgraphTransport()
    transport.run(
        kgq.MERGE_STRATEGY,
        strategy_id="s",
        version=1,
        lifecycle="DRAFT",
        ts_ns=1,
    )
    with pytest.raises(MemgraphKnowledgeStoreError):
        transport.run(
            kgq.MERGE_CAUSED_BY,
            src_id="s",
            dst_id="s",
            weight=0.5,
            ts_ns=2,
        )


def test_inmemory_weight_bounds():
    transport = InMemoryMemgraphTransport()
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
    with pytest.raises(MemgraphKnowledgeStoreError):
        transport.run(
            kgq.MERGE_CAUSED_BY,
            src_id="a",
            dst_id="b",
            weight=1.5,
            ts_ns=1,
        )


def test_inmemory_missing_required_param():
    transport = InMemoryMemgraphTransport()
    with pytest.raises(MemgraphKnowledgeStoreError):
        transport.run(kgq.MERGE_STRATEGY, version=1, lifecycle="DRAFT", ts_ns=1)


def test_coordinator_int_param_rejects_bool():
    g = MemgraphKnowledgeStore()
    with pytest.raises(MemgraphKnowledgeStoreError):
        g.merge_strategy(
            strategy_id="s",
            version=True,  # type: ignore[arg-type]
            lifecycle="DRAFT",
            ts_ns=1,
        )


def test_inmemory_close_is_noop():
    transport = InMemoryMemgraphTransport()
    transport.close()  # should not raise


# ---------------------------------------------------------------------------
# MemgraphKnowledgeStore coordinator surface.
# ---------------------------------------------------------------------------


def _populate(graph: MemgraphKnowledgeStore) -> None:
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


def test_store_default_transport_is_inmemory():
    g = MemgraphKnowledgeStore()
    assert isinstance(g._transport, InMemoryMemgraphTransport)


def test_store_rejects_non_transport():
    with pytest.raises(TypeError):
        MemgraphKnowledgeStore(transport="not a transport")  # type: ignore[arg-type]


def test_store_fetch_node_round_trip():
    g = MemgraphKnowledgeStore()
    _populate(g)
    node = g.fetch_node(node_id="strat-A")
    assert node is not None
    assert node.kind is NodeKind.STRATEGY
    assert node.node_id == "strat-A"
    assert node.props["version"] == 1
    assert node.ts_ns == 1_000


def test_store_fetch_missing_node_returns_none():
    g = MemgraphKnowledgeStore()
    _populate(g)
    assert g.fetch_node(node_id="nonexistent") is None


def test_store_fetch_outgoing_deterministic_order():
    g = MemgraphKnowledgeStore()
    _populate(g)
    out = g.fetch_outgoing(node_id="strat-A")
    targets = [e.target_id for e in out]
    assert targets == sorted(targets)
    assert "fail-1" in targets
    assert "strat-B" in targets


def test_store_fetch_incoming_deterministic_order():
    g = MemgraphKnowledgeStore()
    _populate(g)
    inc = g.fetch_incoming(node_id="fail-1")
    sources = [e.source_id for e in inc]
    assert sources == sorted(sources)
    assert sources == ["strat-A"]


def test_store_fetch_causal_chain():
    g = MemgraphKnowledgeStore()
    _populate(g)
    chains = g.fetch_causal_chain(source_id="strat-A", max_depth=3)
    chain_ids = [tuple(c.node_ids) for c in chains]
    assert ("strat-A", "fail-1") in chain_ids
    assert ("strat-A", "strat-B") in chain_ids
    assert ("strat-A", "fail-1", "reg-bull") in chain_ids


def test_store_fetch_causal_chain_max_depth_clamps_length():
    g = MemgraphKnowledgeStore()
    _populate(g)
    chains = g.fetch_causal_chain(source_id="strat-A", max_depth=1)
    for c in chains:
        assert len(c.node_ids) <= 2


def test_store_fetch_causal_chain_invalid_depth():
    g = MemgraphKnowledgeStore()
    with pytest.raises(MemgraphKnowledgeStoreError):
        g.fetch_causal_chain(source_id="strat-A", max_depth=0)


def test_store_fetch_causal_chain_unknown_source():
    g = MemgraphKnowledgeStore()
    chains = g.fetch_causal_chain(source_id="ghost", max_depth=5)
    assert chains == ()


def test_store_iter_nodes_deterministic():
    g = MemgraphKnowledgeStore()
    _populate(g)
    ids = [n.node_id for n in g.iter_nodes()]
    assert ids == ["fail-1", "reg-bull", "strat-A", "strat-B"]


def test_store_iter_edges_deterministic():
    g = MemgraphKnowledgeStore()
    _populate(g)
    pairs = [(e.source_id, e.target_id) for e in g.iter_edges()]
    assert pairs == sorted(pairs)


def test_store_merge_strategy_validation():
    g = MemgraphKnowledgeStore()
    with pytest.raises(MemgraphKnowledgeStoreError):
        g.merge_strategy(strategy_id="", version=1, lifecycle="DRAFT", ts_ns=1)
    with pytest.raises(MemgraphKnowledgeStoreError):
        g.merge_strategy(strategy_id="s", version="bad", lifecycle="DRAFT", ts_ns=1)  # type: ignore[arg-type]


def test_store_merge_caused_by_validation():
    g = MemgraphKnowledgeStore()
    g.merge_strategy(strategy_id="s", version=1, lifecycle="DRAFT", ts_ns=1)
    g.merge_strategy(strategy_id="t", version=1, lifecycle="DRAFT", ts_ns=1)
    with pytest.raises(MemgraphKnowledgeStoreError):
        g.merge_caused_by(source_id="s", target_id="t", weight="bad", ts_ns=1)  # type: ignore[arg-type]


def test_store_close_does_not_raise():
    g = MemgraphKnowledgeStore()
    g.close()


# ---------------------------------------------------------------------------
# Serialisation / round-trip.
# ---------------------------------------------------------------------------


def test_store_serialize_round_trip():
    g = MemgraphKnowledgeStore()
    _populate(g)
    blob = g.serialize()
    rebuilt = MemgraphKnowledgeStore.deserialize(blob)
    assert rebuilt.serialize() == blob


def test_store_serialize_byte_stable():
    g1 = MemgraphKnowledgeStore()
    g2 = MemgraphKnowledgeStore()
    _populate(g1)
    _populate(g2)
    assert g1.serialize() == g2.serialize()


def test_store_deserialize_rejects_bad_blob():
    with pytest.raises(MemgraphKnowledgeStoreError):
        MemgraphKnowledgeStore.deserialize(b"{not json")


def test_store_deserialize_rejects_wrong_version():
    blob = b'{"version":"999","nodes":[],"edges":[]}'
    with pytest.raises(MemgraphKnowledgeStoreError):
        MemgraphKnowledgeStore.deserialize(blob)


def test_store_deserialize_rejects_non_bytes():
    with pytest.raises(MemgraphKnowledgeStoreError):
        MemgraphKnowledgeStore.deserialize("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay.
# ---------------------------------------------------------------------------


def test_inv15_three_run_byte_identical():
    def run() -> bytes:
        g = MemgraphKnowledgeStore()
        _populate(g)
        return g.serialize()

    a = run()
    b = run()
    c = run()
    assert a == b == c


def test_inv15_node_insertion_order_independence():
    g1 = MemgraphKnowledgeStore()
    g1.merge_strategy(strategy_id="b", version=1, lifecycle="DRAFT", ts_ns=2)
    g1.merge_strategy(strategy_id="a", version=1, lifecycle="DRAFT", ts_ns=1)

    g2 = MemgraphKnowledgeStore()
    g2.merge_strategy(strategy_id="a", version=1, lifecycle="DRAFT", ts_ns=1)
    g2.merge_strategy(strategy_id="b", version=1, lifecycle="DRAFT", ts_ns=2)

    assert g1.serialize() == g2.serialize()


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def test_memgraph_factory_validation():
    with pytest.raises(ValueError):
        memgraph_client_factory(host="")
    with pytest.raises(ValueError):
        memgraph_client_factory(port=0)
    with pytest.raises(ValueError):
        memgraph_client_factory(port=70000)
    with pytest.raises(TypeError):
        memgraph_client_factory(port=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        memgraph_client_factory(username=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        memgraph_client_factory(password=123)  # type: ignore[arg-type]


def test_memgraph_factory_raises_when_missing():
    try:
        import gqlalchemy  # noqa: F401, PLC0415

        pytest.skip("gqlalchemy installed; factory error path not exercised here")
    except ImportError:
        with pytest.raises(MemgraphKnowledgeStoreError):
            memgraph_client_factory()


# ---------------------------------------------------------------------------
# AST guards.
# ---------------------------------------------------------------------------


def _kg_tree() -> ast.AST:
    return ast.parse(_KG_PATH.read_text(encoding="utf-8"))


def _kgq_tree() -> ast.AST:
    return ast.parse(_KGQ_PATH.read_text(encoding="utf-8"))


def test_ast_adapted_from_header_present():
    text = _KG_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: memgraph/gqlalchemy" in text


def test_ast_no_toplevel_gqlalchemy_import():
    tree = _kg_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("gqlalchemy")
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("gqlalchemy")


def test_ast_gqlalchemy_import_confined_to_factory():
    tree = _kg_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gqlalchemy"):
            in_factory = False
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef) and fn.name == "memgraph_client_factory":
                    for sub in ast.walk(fn):
                        if sub is node:
                            in_factory = True
                            break
            assert in_factory, "gqlalchemy import must live inside memgraph_client_factory"


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
    "neo4j",
    "qdrant_client",
    "weaviate",
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
    tree = _kg_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if any(kw in text for kw in ("MATCH ", "MERGE ", "CREATE ", "RETURN ")):
                if "$" in text:
                    pytest.fail(
                        f"inline cypher detected in state.knowledge_store_memgraph: {text[:80]!r}"
                    )


def test_ast_query_module_constants_only():
    tree = _kgq_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            pytest.fail("f-strings forbidden in knowledge_store_memgraph_queries")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"format", "format_map"}:
                pytest.fail(f"runtime string formatting forbidden in queries: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Module-shape sanity.
# ---------------------------------------------------------------------------


def test_store_signature_surface_is_keyword_only():
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
        sig = inspect.signature(getattr(MemgraphKnowledgeStore, name))
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name}.{param_name} must be keyword-only"
            )
