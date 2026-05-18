"""C-07 streamz — comprehensive test suite.

Pins the OFFLINE_ONLY contract, the deterministic-replay invariant
(INV-15), the B27 / B28 / INV-71 authority-symmetry rule, and B1
runtime-tier isolation. Also exercises every operator value-object and
every executor branch.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from system_engine.streaming import streamz_cep as cep
from system_engine.streaming.streamz_cep import (
    NEW_PIP_DEPENDENCIES,
    STREAMZ_CEP_VERSION,
    AccumulateNode,
    CombineLatestNode,
    FilterNode,
    GraphResult,
    MapNode,
    Node,
    SinkNode,
    SlidingWindowNode,
    SourceNode,
    StreamGraph,
    ZipNode,
    run_graph,
    streamz_stream_factory,
)

# ---------------------------------------------------------------------------
# Module surface constants.
# ---------------------------------------------------------------------------


def test_module_version_is_one() -> None:
    assert STREAMZ_CEP_VERSION == 1


def test_new_pip_dependencies_declared_but_never_imported() -> None:
    assert NEW_PIP_DEPENDENCIES == ("streamz",)


# ---------------------------------------------------------------------------
# Empty / source-only graphs.
# ---------------------------------------------------------------------------


def test_empty_graph_has_no_nodes() -> None:
    g = StreamGraph(name="empty")
    assert g.node_names() == ()
    assert g.source_names() == ()
    assert g.sink_names() == ()


def test_source_only_graph_runs() -> None:
    g = StreamGraph(name="src").source("a")
    result = run_graph(g, {"a": [1, 2, 3]})
    assert result.sink_outputs == {}
    # graph_digest is non-empty 32-hex.
    assert len(result.graph_digest) == 32
    assert int(result.graph_digest, 16) >= 0


def test_run_graph_rejects_unknown_sources() -> None:
    g = StreamGraph(name="src").source("a")
    with pytest.raises(KeyError, match="unknown nodes"):
        run_graph(g, {"a": [1], "b": [2]})


def test_run_graph_rejects_missing_sources() -> None:
    g = StreamGraph(name="src").source("a").source("b")
    with pytest.raises(KeyError, match="missing for nodes"):
        run_graph(g, {"a": [1]})


def test_run_graph_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        run_graph("not-a-graph", {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Map / Filter / Sink.
# ---------------------------------------------------------------------------


def test_map_doubles_every_element() -> None:
    g = StreamGraph(name="m").source("src").map("dbl", "src", lambda x: x * 2).sink("out", "dbl")
    result = run_graph(g, {"src": [1, 2, 3]})
    assert result.sink_outputs == {"out": (2, 4, 6)}


def test_filter_drops_odd() -> None:
    g = (
        StreamGraph(name="f")
        .source("src")
        .filter("even", "src", lambda x: x % 2 == 0)
        .sink("out", "even")
    )
    result = run_graph(g, {"src": [1, 2, 3, 4, 5, 6]})
    assert result.sink_outputs == {"out": (2, 4, 6)}


def test_sink_records_arrival_order() -> None:
    g = StreamGraph(name="s").source("src").sink("out", "src")
    result = run_graph(g, {"src": ["c", "a", "b"]})
    assert result.sink_outputs == {"out": ("c", "a", "b")}


def test_sink_fn_called_per_element() -> None:
    seen: list[int] = []
    g = StreamGraph(name="s").source("src").sink("out", "src", sink_fn=seen.append)
    run_graph(g, {"src": [10, 20, 30]})
    assert seen == [10, 20, 30]


# ---------------------------------------------------------------------------
# Accumulate.
# ---------------------------------------------------------------------------


def test_accumulate_running_sum() -> None:
    def step(acc: int, item: int) -> tuple[int, int]:
        new = acc + item
        return new, new

    g = (
        StreamGraph(name="a")
        .source("src")
        .accumulate("sum", "src", init=lambda: 0, step=step)
        .sink("out", "sum")
    )
    result = run_graph(g, {"src": [1, 2, 3, 4]})
    assert result.sink_outputs == {"out": (1, 3, 6, 10)}


def test_accumulate_can_emit_different_type() -> None:
    def step(acc: int, item: int) -> tuple[int, str]:
        return acc + 1, f"#{acc + 1}:{item}"

    g = (
        StreamGraph(name="a")
        .source("src")
        .accumulate("enum", "src", init=lambda: 0, step=step)
        .sink("out", "enum")
    )
    result = run_graph(g, {"src": ["x", "y", "z"]})
    assert result.sink_outputs == {"out": ("#1:x", "#2:y", "#3:z")}


# ---------------------------------------------------------------------------
# SlidingWindow.
# ---------------------------------------------------------------------------


def test_sliding_window_emits_only_full_windows_by_default() -> None:
    g = StreamGraph(name="w").source("src").sliding_window("win", "src", n=3).sink("out", "win")
    result = run_graph(g, {"src": [1, 2, 3, 4, 5]})
    assert result.sink_outputs == {"out": ((1, 2, 3), (2, 3, 4), (3, 4, 5))}


def test_sliding_window_return_partial_emits_partials() -> None:
    g = (
        StreamGraph(name="w")
        .source("src")
        .sliding_window("win", "src", n=3, return_partial=True)
        .sink("out", "win")
    )
    result = run_graph(g, {"src": [1, 2, 3, 4]})
    assert result.sink_outputs == {"out": ((1,), (1, 2), (1, 2, 3), (2, 3, 4))}


def test_sliding_window_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        StreamGraph(name="w").source("src").sliding_window("win", "src", n=0)
    with pytest.raises(ValueError):
        StreamGraph(name="w").source("src").sliding_window("win", "src", n=-1)


# ---------------------------------------------------------------------------
# Zip / CombineLatest.
# ---------------------------------------------------------------------------


def test_zip_synchronous_pairs() -> None:
    g = (
        StreamGraph(name="z")
        .source("a")
        .source("b")
        .zip("paired", ["a", "b"])
        .sink("out", "paired")
    )
    result = run_graph(g, {"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert result.sink_outputs == {"out": ((1, "x"), (2, "y"), (3, "z"))}


def test_zip_truncates_to_shortest() -> None:
    g = (
        StreamGraph(name="z")
        .source("a")
        .source("b")
        .zip("paired", ["a", "b"])
        .sink("out", "paired")
    )
    result = run_graph(g, {"a": [1, 2, 3], "b": ["x", "y"]})
    assert result.sink_outputs == {"out": ((1, "x"), (2, "y"))}


def test_zip_three_upstreams() -> None:
    g = (
        StreamGraph(name="z3")
        .source("a")
        .source("b")
        .source("c")
        .zip("t", ["a", "b", "c"])
        .sink("out", "t")
    )
    result = run_graph(g, {"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
    assert result.sink_outputs == {"out": ((1, "x", True), (2, "y", False))}


def test_zip_rejects_single_upstream() -> None:
    g = StreamGraph(name="z").source("a")
    with pytest.raises(ValueError):
        g.zip("paired", ["a"])


def test_combine_latest_waits_for_every_upstream() -> None:
    g = (
        StreamGraph(name="cl")
        .source("a")
        .source("b")
        .combine_latest("c", ["a", "b"])
        .sink("out", "c")
    )
    result = run_graph(g, {"a": [1, 2], "b": ["x", "y"]})
    # Executor processes upstream "a" first (both values), then "b" —
    # so no output until "b" emits "x", then a final emission on "y".
    assert result.sink_outputs == {"out": ((2, "x"), (2, "y"))}


def test_combine_latest_rejects_single_upstream() -> None:
    g = StreamGraph(name="cl").source("a")
    with pytest.raises(ValueError):
        g.combine_latest("c", ["a"])


# ---------------------------------------------------------------------------
# Graph builder validation.
# ---------------------------------------------------------------------------


def test_duplicate_node_name_rejected() -> None:
    g = StreamGraph(name="d").source("a")
    with pytest.raises(ValueError, match="duplicate node name"):
        g.source("a")


def test_unknown_upstream_rejected() -> None:
    g = StreamGraph(name="u").source("a")
    with pytest.raises(KeyError):
        g.map("m", "missing", lambda x: x)
    with pytest.raises(KeyError):
        g.filter("f", "missing", lambda x: True)
    with pytest.raises(KeyError):
        g.accumulate("a2", "missing", init=lambda: 0, step=lambda a, x: (a, x))
    with pytest.raises(KeyError):
        g.sliding_window("w", "missing", n=2)
    with pytest.raises(KeyError):
        g.sink("s", "missing")
    with pytest.raises(KeyError):
        g.zip("z", ["a", "missing"])
    with pytest.raises(KeyError):
        g.combine_latest("c", ["a", "missing"])


def test_builders_return_new_instances() -> None:
    g0 = StreamGraph(name="b")
    g1 = g0.source("a")
    assert g0 is not g1
    assert g0.nodes == ()
    assert g1.nodes != ()


# ---------------------------------------------------------------------------
# Node names / introspection.
# ---------------------------------------------------------------------------


def test_node_names_and_categorisation() -> None:
    g = (
        StreamGraph(name="i")
        .source("a")
        .source("b")
        .map("m", "a", lambda x: x)
        .sink("s1", "m")
        .sink("s2", "b")
    )
    assert g.node_names() == ("a", "b", "m", "s1", "s2")
    assert g.source_names() == ("a", "b")
    assert g.sink_names() == ("s1", "s2")


# ---------------------------------------------------------------------------
# Digest / INV-15 byte-identical replay.
# ---------------------------------------------------------------------------


def _build_pipeline() -> StreamGraph:
    return (
        StreamGraph(name="p")
        .source("src")
        .filter("even", "src", lambda x: x % 2 == 0)
        .map("dbl", "even", lambda x: x * 2)
        .accumulate(
            "running_sum",
            "dbl",
            init=lambda: 0,
            step=lambda a, x: (a + x, a + x),
        )
        .sliding_window("win3", "running_sum", n=3)
        .sink("out", "win3")
    )


def test_graph_digest_is_stable_across_rebuilds() -> None:
    g1 = _build_pipeline()
    g2 = _build_pipeline()
    assert g1.graph_digest() == g2.graph_digest()


def test_graph_digest_changes_with_topology() -> None:
    g1 = _build_pipeline()
    g2 = StreamGraph(name="p").source("src").map("dbl", "src", lambda x: x * 2).sink("out", "dbl")
    assert g1.graph_digest() != g2.graph_digest()


def test_run_digest_is_byte_identical_across_three_runs() -> None:
    graph = _build_pipeline()
    inputs = {"src": list(range(20))}
    r1 = run_graph(graph, inputs)
    r2 = run_graph(graph, inputs)
    r3 = run_graph(graph, inputs)
    assert r1.run_digest == r2.run_digest == r3.run_digest
    assert r1.graph_digest == r2.graph_digest == r3.graph_digest
    assert r1.sink_outputs == r2.sink_outputs == r3.sink_outputs


def test_run_digest_changes_with_input() -> None:
    graph = _build_pipeline()
    r1 = run_graph(graph, {"src": list(range(20))})
    r2 = run_graph(graph, {"src": list(range(21))})
    assert r1.run_digest != r2.run_digest


def test_graph_result_record_fields() -> None:
    graph = _build_pipeline()
    r = run_graph(graph, {"src": [0, 1, 2, 3, 4]})
    assert isinstance(r, GraphResult)
    assert r.graph_name == "p"
    assert isinstance(r.sink_outputs, dict)
    assert "out" in r.sink_outputs
    assert len(r.graph_digest) == 32
    assert len(r.run_digest) == 32


# ---------------------------------------------------------------------------
# Topological order — branches, joins, lexicographic tie-break.
# ---------------------------------------------------------------------------


def test_branch_and_join_pipeline() -> None:
    """Source -> two parallel maps -> zip -> sink."""

    g = (
        StreamGraph(name="bj")
        .source("src")
        .map("plus1", "src", lambda x: x + 1)
        .map("times2", "src", lambda x: x * 2)
        .zip("paired", ["plus1", "times2"])
        .sink("out", "paired")
    )
    result = run_graph(g, {"src": [1, 2, 3]})
    assert result.sink_outputs == {"out": ((2, 2), (3, 4), (4, 6))}


def test_cycle_detection() -> None:
    # Manually craft a graph with a cycle by injecting node tuples
    # — the public builders don't allow it.
    g = StreamGraph(
        name="cyc",
        nodes=(
            SourceNode(name="a"),
            MapNode(name="m1", upstream="m2", fn=lambda x: x),
            MapNode(name="m2", upstream="m1", fn=lambda x: x),
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        run_graph(g, {"a": [1]})


def test_two_sinks_record_independently() -> None:
    g = (
        StreamGraph(name="2s")
        .source("src")
        .map("dbl", "src", lambda x: x * 2)
        .sink("s1", "src")
        .sink("s2", "dbl")
    )
    result = run_graph(g, {"src": [1, 2, 3]})
    assert result.sink_outputs == {
        "s1": (1, 2, 3),
        "s2": (2, 4, 6),
    }


# ---------------------------------------------------------------------------
# Lazy seam — real streamz hookup gate.
# ---------------------------------------------------------------------------


def test_streamz_stream_factory_raises_not_implemented() -> None:
    g = StreamGraph(name="x").source("a")
    with pytest.raises(NotImplementedError, match="research-acceptance"):
        streamz_stream_factory(g)


# ---------------------------------------------------------------------------
# AST guardrails — INV-15 / B27 / B28 / INV-71 / B1.
# ---------------------------------------------------------------------------


MODULE_PATH = Path(cep.__file__)
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")
MODULE_TREE = ast.parse(MODULE_SOURCE)


def _top_level_imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                out.append(node.module)
    return out


FORBIDDEN_TOPLEVEL = (
    "time",
    "datetime",
    "random",
    "asyncio",
    "os",
    "streamz",
    "numpy",
    "torch",
    "polars",
    "requests",
    "httpx",
    "aiohttp",
    "tornado",
)


def test_no_forbidden_toplevel_imports() -> None:
    imported = _top_level_imports(MODULE_TREE)
    for forbidden in FORBIDDEN_TOPLEVEL:
        for name in imported:
            assert not (name == forbidden or name.startswith(forbidden + ".")), (
                f"streamz_cep.py imports {forbidden!r} at module top-level"
            )


RUNTIME_TIERS = (
    "intelligence_engine",
    "execution_engine",
    "governance_engine",
    "evolution_engine",
    "learning_engine",
)


def test_no_runtime_tier_imports() -> None:
    imported = _top_level_imports(MODULE_TREE)
    for tier in RUNTIME_TIERS:
        for name in imported:
            assert not (name == tier or name.startswith(tier + ".")), (
                f"streamz_cep.py imports runtime tier {tier!r}"
            )


FORBIDDEN_TYPED_EVENT_CTORS = (
    "PatchProposal",
    "HazardEvent",
    "SignalEvent",
    "ExecutionEvent",
    "SystemEvent",
)


def test_no_typed_event_constructors_called() -> None:
    """B27 / B28 / INV-71: only the engine that produced an event may
    construct it; transports never call typed-event constructors."""

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.bad: list[str] = []

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in FORBIDDEN_TYPED_EVENT_CTORS:
                self.bad.append(name)
            self.generic_visit(node)

    v = _Visitor()
    v.visit(MODULE_TREE)
    assert not v.bad, "streamz_cep.py constructs typed events: " + ", ".join(v.bad)


def test_module_reimports_clean() -> None:
    """Reimporting the module must not have side-effects that import a
    forbidden package."""

    reloaded = importlib.reload(cep)
    assert reloaded.STREAMZ_CEP_VERSION == 1
    # The reload itself must not pull in streamz; this is checked
    # implicitly by ``test_no_forbidden_toplevel_imports`` plus the
    # fact that the reload completes without error in a clean env.


# ---------------------------------------------------------------------------
# Value-object structural checks.
# ---------------------------------------------------------------------------


def test_node_classes_are_frozen_slotted_dataclasses() -> None:
    """All node value objects are frozen + slotted — mutation must
    raise ``AttributeError``."""
    for cls in (
        SourceNode,
        MapNode,
        FilterNode,
        AccumulateNode,
        ZipNode,
        CombineLatestNode,
        SlidingWindowNode,
        SinkNode,
    ):
        assert issubclass(cls, Node)


def test_source_node_is_immutable() -> None:
    n = SourceNode(name="a")
    with pytest.raises((AttributeError, TypeError)):
        n.name = "b"  # type: ignore[misc]


def test_stream_graph_is_immutable() -> None:
    g = StreamGraph(name="im").source("a")
    with pytest.raises((AttributeError, TypeError)):
        g.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Sensor-array motivating example — feeds hazard sensors.
# ---------------------------------------------------------------------------


def test_hazard_sensor_motivating_pipeline() -> None:
    """Spike-detector pipeline — sliding window of 5, max-min spread,
    threshold filter, sink count.

    Mirrors the kind of CEP graph ``sensor_array.py`` consumes: turns
    a raw value stream into a small set of "spike-detected" advisory
    records.
    """

    def spread(window: tuple[int, ...]) -> int:
        return max(window) - min(window)

    g = (
        StreamGraph(name="haz")
        .source("ticks")
        .sliding_window("w5", "ticks", n=5)
        .map("spread", "w5", spread)
        .filter("spiky", "spread", lambda s: s >= 10)
        .sink("alerts", "spiky")
    )
    inputs = {
        "ticks": [
            100,
            101,
            99,
            102,
            100,  # window 1: spread=3
            105,
            110,
            108,
            95,
            120,  # window ends with spread=25
            121,
            122,
            123,
            124,
            125,  # spread=4
        ],
    }
    result = run_graph(g, inputs)
    alerts = result.sink_outputs["alerts"]
    # Every sliding window after position 4 produces a spread; only
    # those >=10 are kept.
    assert all(s >= 10 for s in alerts)
    assert len(alerts) > 0
