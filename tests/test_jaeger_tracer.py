"""I-30 — tests for the canonical jaeger-shape distributed tracer."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from tools.jaeger_tracer import (
    MAX_SERVICE_NAME_LEN,
    MAX_SPAN_NAME_LEN,
    MAX_SPANS_PER_TRACE,
    MAX_TAG_KEY_LEN,
    MAX_TAG_VALUE_LEN,
    MAX_TAGS_PER_SPAN,
    MAX_TRACE_DEPTH,
    NEW_PIP_DEPENDENCIES,
    TRACER_VERSION,
    Clock,
    Span,
    Tracer,
    TraceReport,
    TracerError,
    enable_jaeger_factory,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_tracer_version_is_pinned() -> None:
    assert TRACER_VERSION == "v1.0-I30"


def test_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("jaeger-client",)


def test_max_lengths_are_pinned() -> None:
    assert MAX_SERVICE_NAME_LEN == 64
    assert MAX_SPAN_NAME_LEN == 128
    assert MAX_TAG_KEY_LEN == 64
    assert MAX_TAG_VALUE_LEN == 1024
    assert MAX_TAGS_PER_SPAN == 64
    assert MAX_SPANS_PER_TRACE == 10_000
    assert MAX_TRACE_DEPTH == 64


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_clock_is_monotone() -> None:
    clock = Clock(origin_ns=0, tick_ns=1_000)
    a = clock.now_ns()
    b = clock.now_ns()
    c = clock.now_ns()
    assert (a, b, c) == (0, 1_000, 2_000)


def test_clock_origin_ns_validated() -> None:
    with pytest.raises(TracerError):
        Clock(origin_ns=-1, tick_ns=1)


def test_clock_tick_ns_validated() -> None:
    with pytest.raises(TracerError):
        Clock(origin_ns=0, tick_ns=0)


# ---------------------------------------------------------------------------
# Span value object
# ---------------------------------------------------------------------------


def _span(**kwargs) -> Span:
    defaults = dict(
        span_id=1,
        parent_id=None,
        trace_id=42,
        service_name="dixvision",
        operation_name="op",
        start_ns=100,
        end_ns=200,
        tags={},
    )
    defaults.update(kwargs)
    return Span(**defaults)


def test_span_construction_happy_path() -> None:
    s = _span(tags={"k": "v"})
    assert s.duration_ns == 100
    assert s.tags["k"] == "v"


def test_span_is_frozen_and_slotted() -> None:
    s = _span()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.span_id = 99  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_span_span_id_negative() -> None:
    with pytest.raises(TracerError):
        _span(span_id=-1)


def test_span_parent_id_type() -> None:
    with pytest.raises(TracerError):
        _span(parent_id="x")  # type: ignore[arg-type]


def test_span_parent_id_negative() -> None:
    with pytest.raises(TracerError):
        _span(parent_id=-1)


def test_span_trace_id_validated() -> None:
    with pytest.raises(TracerError):
        _span(trace_id=-1)


def test_span_service_name_validated() -> None:
    with pytest.raises(TracerError):
        _span(service_name="")


def test_span_service_name_length() -> None:
    with pytest.raises(TracerError):
        _span(service_name="x" * 65)


def test_span_operation_name_validated() -> None:
    with pytest.raises(TracerError):
        _span(operation_name="")


def test_span_operation_name_length() -> None:
    with pytest.raises(TracerError):
        _span(operation_name="x" * 129)


def test_span_start_ns_validated() -> None:
    with pytest.raises(TracerError):
        _span(start_ns=-1)


def test_span_end_ns_before_start() -> None:
    with pytest.raises(TracerError):
        _span(start_ns=200, end_ns=100)


def test_span_tags_must_be_mapping() -> None:
    with pytest.raises(TracerError):
        _span(tags=[("k", "v")])  # type: ignore[arg-type]


def test_span_tag_key_validated() -> None:
    with pytest.raises(TracerError):
        _span(tags={"": "v"})


def test_span_tag_key_length() -> None:
    with pytest.raises(TracerError):
        _span(tags={"k" * 65: "v"})


def test_span_tag_value_scalar() -> None:
    with pytest.raises(TracerError):
        _span(tags={"k": [1]})  # type: ignore[dict-item]


def test_span_tag_value_length() -> None:
    with pytest.raises(TracerError):
        _span(tags={"k": "x" * 1025})


def test_span_tags_count_cap() -> None:
    too_many = {f"k{i}": i for i in range(MAX_TAGS_PER_SPAN + 1)}
    with pytest.raises(TracerError):
        _span(tags=too_many)


# ---------------------------------------------------------------------------
# TraceReport
# ---------------------------------------------------------------------------


def _empty_report(backend: str = "stdlib") -> TraceReport:
    return TraceReport(
        service_name="dixvision",
        trace_id=1,
        backend=backend,
        spans=(),
    )


def test_trace_report_construction() -> None:
    rep = _empty_report()
    assert rep.spans == ()


def test_trace_report_frozen_slotted() -> None:
    rep = _empty_report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.backend = "x"  # type: ignore[misc]
    assert not hasattr(rep, "__dict__")


def test_trace_report_service_name_validated() -> None:
    with pytest.raises(TracerError):
        TraceReport(
            service_name="",
            trace_id=1,
            backend="stdlib",
            spans=(),
        )


def test_trace_report_trace_id_validated() -> None:
    with pytest.raises(TracerError):
        TraceReport(
            service_name="x",
            trace_id=-1,
            backend="stdlib",
            spans=(),
        )


def test_trace_report_backend_validated() -> None:
    with pytest.raises(TracerError):
        TraceReport(
            service_name="x",
            trace_id=1,
            backend="bogus",
            spans=(),
        )


def test_trace_report_spans_must_be_tuple() -> None:
    with pytest.raises(TracerError):
        TraceReport(
            service_name="x",
            trace_id=1,
            backend="stdlib",
            spans=[_span()],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Tracer — construction
# ---------------------------------------------------------------------------


def _make_tracer(*, seed: int = 42, trace_id: int = 1) -> Tracer:
    return Tracer(
        service_name="dixvision",
        trace_id=trace_id,
        seed=seed,
        clock=Clock(origin_ns=0, tick_ns=1_000),
    )


def test_tracer_service_name_validated() -> None:
    with pytest.raises(TracerError):
        Tracer(
            service_name="",
            trace_id=1,
            seed=0,
            clock=Clock(),
        )


def test_tracer_service_name_length() -> None:
    with pytest.raises(TracerError):
        Tracer(
            service_name="x" * 65,
            trace_id=1,
            seed=0,
            clock=Clock(),
        )


def test_tracer_trace_id_validated() -> None:
    with pytest.raises(TracerError):
        Tracer(
            service_name="x",
            trace_id=-1,
            seed=0,
            clock=Clock(),
        )


def test_tracer_seed_validated() -> None:
    with pytest.raises(TracerError):
        Tracer(
            service_name="x",
            trace_id=1,
            seed="hello",  # type: ignore[arg-type]
            clock=Clock(),
        )


def test_tracer_clock_validated() -> None:
    with pytest.raises(TracerError):
        Tracer(
            service_name="x",
            trace_id=1,
            seed=0,
            clock=object(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Tracer — happy paths
# ---------------------------------------------------------------------------


def test_tracer_single_span() -> None:
    tracer = _make_tracer()
    sid = tracer.start_span("root")
    tracer.finish_span(sid, tags={"status": "ok"})
    report = tracer.snapshot()
    assert len(report.spans) == 1
    span = report.spans[0]
    assert span.operation_name == "root"
    assert span.parent_id is None
    assert span.tags == {"status": "ok"}


def test_tracer_nested_spans_resolve_parent() -> None:
    tracer = _make_tracer()
    outer = tracer.start_span("outer")
    inner = tracer.start_span("inner")
    tracer.finish_span(inner)
    tracer.finish_span(outer)
    report = tracer.snapshot()
    assert len(report.spans) == 2
    by_name = {s.operation_name: s for s in report.spans}
    assert by_name["inner"].parent_id == by_name["outer"].span_id


def test_tracer_explicit_parent_id_allowed_when_in_flight() -> None:
    tracer = _make_tracer()
    a = tracer.start_span("a")
    b = tracer.start_span("b", parent_id=a)
    tracer.finish_span(b)
    tracer.finish_span(a)
    report = tracer.snapshot()
    children = report.children_of(a)
    assert any(s.operation_name == "b" for s in children)


def test_tracer_root_spans_helper() -> None:
    tracer = _make_tracer()
    a = tracer.start_span("a")
    b = tracer.start_span("b")
    tracer.finish_span(b)
    tracer.finish_span(a)
    report = tracer.snapshot()
    roots = report.root_spans()
    assert len(roots) == 1
    assert roots[0].operation_name == "a"


def test_tracer_find_by_operation() -> None:
    tracer = _make_tracer()
    a = tracer.start_span("step")
    tracer.finish_span(a)
    b = tracer.start_span("step")
    tracer.finish_span(b)
    report = tracer.snapshot()
    matches = report.find("step")
    assert len(matches) == 2


def test_tracer_spans_sorted_by_start_then_id() -> None:
    tracer = _make_tracer()
    a = tracer.start_span("a")
    tracer.finish_span(a)
    b = tracer.start_span("b")
    tracer.finish_span(b)
    report = tracer.snapshot()
    assert report.spans[0].operation_name == "a"
    assert report.spans[1].operation_name == "b"


def test_tracer_span_ids_are_seeded_deterministic() -> None:
    t1 = _make_tracer(seed=42)
    t2 = _make_tracer(seed=42)
    a1 = t1.start_span("a")
    t1.finish_span(a1)
    a2 = t2.start_span("a")
    t2.finish_span(a2)
    assert a1 == a2


def test_tracer_different_seed_different_ids() -> None:
    t1 = _make_tracer(seed=42)
    t2 = _make_tracer(seed=43)
    a1 = t1.start_span("a")
    t1.finish_span(a1)
    a2 = t2.start_span("a")
    t2.finish_span(a2)
    assert a1 != a2


def test_tracer_determinism_three_runs() -> None:
    def run() -> TraceReport:
        tracer = _make_tracer()
        a = tracer.start_span("outer")
        b = tracer.start_span("inner")
        tracer.finish_span(b, tags={"k": "v"})
        tracer.finish_span(a, tags={"status": "ok"})
        return tracer.snapshot()

    r1 = run()
    r2 = run()
    r3 = run()
    assert r1.digest == r2.digest == r3.digest
    assert r1.spans == r2.spans == r3.spans


def test_tracer_finished_tags_sorted_in_storage() -> None:
    tracer = _make_tracer()
    sid = tracer.start_span("x")
    tracer.finish_span(sid, tags={"z": 1, "a": 2, "m": 3})
    report = tracer.snapshot()
    assert list(report.spans[0].tags.keys()) == ["a", "m", "z"]


def test_tracer_tags_optional_on_finish() -> None:
    tracer = _make_tracer()
    sid = tracer.start_span("x")
    span = tracer.finish_span(sid)
    assert span.tags == {}


# ---------------------------------------------------------------------------
# Tracer — error paths
# ---------------------------------------------------------------------------


def test_tracer_start_span_name_empty() -> None:
    tracer = _make_tracer()
    with pytest.raises(TracerError):
        tracer.start_span("")


def test_tracer_start_span_name_too_long() -> None:
    tracer = _make_tracer()
    with pytest.raises(TracerError):
        tracer.start_span("x" * 129)


def test_tracer_start_span_unknown_parent() -> None:
    tracer = _make_tracer()
    with pytest.raises(TracerError):
        tracer.start_span("child", parent_id=12345)


def test_tracer_finish_unknown_span() -> None:
    tracer = _make_tracer()
    with pytest.raises(TracerError):
        tracer.finish_span(99999)


def test_tracer_finish_violates_stack_order() -> None:
    tracer = _make_tracer()
    a = tracer.start_span("a")
    b = tracer.start_span("b")
    with pytest.raises(TracerError):
        tracer.finish_span(a)
    tracer.finish_span(b)
    tracer.finish_span(a)


def test_tracer_finish_tags_type() -> None:
    tracer = _make_tracer()
    sid = tracer.start_span("x")
    with pytest.raises(TracerError):
        tracer.finish_span(sid, tags=[("k", "v")])  # type: ignore[arg-type]


def test_tracer_finish_tag_value_non_scalar() -> None:
    tracer = _make_tracer()
    sid = tracer.start_span("x")
    with pytest.raises(TracerError):
        tracer.finish_span(sid, tags={"k": object()})  # type: ignore[dict-item]


def test_tracer_snapshot_with_in_flight_raises() -> None:
    tracer = _make_tracer()
    tracer.start_span("x")
    with pytest.raises(TracerError):
        tracer.snapshot()


def test_tracer_trace_depth_cap() -> None:
    tracer = _make_tracer()
    opened: list[int] = []
    for i in range(MAX_TRACE_DEPTH):
        opened.append(tracer.start_span(f"d{i}"))
    with pytest.raises(TracerError):
        tracer.start_span("overflow")
    for sid in reversed(opened):
        tracer.finish_span(sid)


# ---------------------------------------------------------------------------
# Lazy seam
# ---------------------------------------------------------------------------


def test_enable_jaeger_factory_without_dep_raises_import_error() -> None:
    try:
        import jaeger_client  # type: ignore[import-not-found]  # noqa: F401

        pytest.skip("jaeger-client installed; nothing to assert")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        enable_jaeger_factory()


def test_enable_jaeger_factory_returns_callable_when_present() -> None:
    try:
        import jaeger_client  # type: ignore[import-not-found]  # noqa: F401
        import opentracing  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("jaeger-client/opentracing not installed")
    factory = enable_jaeger_factory()
    tracer = factory(
        service_name="dixvision",
        trace_id=1,
        seed=42,
        clock=Clock(),
    )
    sid = tracer.start_span("x")
    tracer.finish_span(sid)
    report = tracer.snapshot()
    assert report.backend == "jaeger"


def test_enable_jaeger_factory_rejects_unknown_override_keys() -> None:
    try:
        import jaeger_client  # type: ignore[import-not-found]  # noqa: F401
        import opentracing  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("jaeger-client/opentracing not installed")
    with pytest.raises(TracerError):
        enable_jaeger_factory(overrides={"bogus_key": 1})


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "jaeger_tracer.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_jaeger_import() -> None:
    imports = _top_level_imports(_module_ast())
    assert "jaeger_client" not in imports
    assert "opentracing" not in imports
    assert "thrift" not in imports


def test_no_top_level_subprocess_import() -> None:
    assert "subprocess" not in _top_level_imports(_module_ast())


def test_no_top_level_time_or_random_import() -> None:
    banned = {"time", "random", "datetime", "asyncio"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_network_imports() -> None:
    banned = {"socket", "urllib", "requests", "httpx", "aiohttp"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_top_level_engine_imports() -> None:
    banned_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
        "core.contracts.",
    )
    for name in _top_level_imports(_module_ast()):
        for prefix in banned_prefixes:
            assert not name.startswith(prefix), name


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


def test_jaeger_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name in ("jaeger_client", "opentracing"):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside enable_jaeger_factory"
                    )
                    assert parent.name == "enable_jaeger_factory", (
                        f"{name} imported in {parent.name!r} — must be inside enable_jaeger_factory"
                    )


# ---------------------------------------------------------------------------
# Realistic trace demo
# ---------------------------------------------------------------------------


def test_realistic_signed_execution_trace() -> None:
    tracer = _make_tracer()
    intent = tracer.start_span("intent.build")
    tracer.finish_span(intent, tags={"asset": "BTC"})
    decision = tracer.start_span("decision.evaluate")
    signed = tracer.start_span("decision.signed")
    tracer.finish_span(signed, tags={"algo": "blake2b"})
    dispatch = tracer.start_span("execution.dispatch")
    tracer.finish_span(dispatch, tags={"venue": "binance"})
    tracer.finish_span(decision, tags={"verdict": "BUY"})
    report = tracer.snapshot()
    assert len(report.spans) == 4
    signed_spans = report.find("decision.signed")
    assert len(signed_spans) == 1
    assert signed_spans[0].tags["algo"] == "blake2b"
    dispatch_spans = report.find("execution.dispatch")
    decision_spans = report.find("decision.evaluate")
    assert dispatch_spans[0].parent_id == decision_spans[0].span_id


# ---------------------------------------------------------------------------
# Reload idempotency — runs last
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import tools.jaeger_tracer as mod1

    importlib.reload(mod1)
    import tools.jaeger_tracer as mod2

    assert mod1.TRACER_VERSION == mod2.TRACER_VERSION
    assert mod1.MAX_SPANS_PER_TRACE == mod2.MAX_SPANS_PER_TRACE
