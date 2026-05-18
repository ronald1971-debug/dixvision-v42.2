"""A-09 — tests for system_engine.tracing.tracer."""

from __future__ import annotations

import ast
import pathlib

import pytest

from system_engine.tracing import tracer as tracer_mod
from system_engine.tracing.tracer import (
    DEFAULT_SAMPLE_RATIO,
    MAX_ATTRIBUTE_COUNT,
    MAX_ATTRIBUTE_VALUE_LEN,
    MAX_SPAN_BUFFER,
    MAX_SPAN_NAME_LEN,
    NEW_PIP_DEPENDENCIES,
    OTEL_ADAPTER_VERSION,
    InProcessTracer,
    SpanRecord,
    Tracer,
    TracerError,
    TraceSnapshot,
    derive_ids,
    otel_tracer_factory,
    render_trace_text,
    sanitize_attribute_value,
)


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
def test_new_pip_dependencies_pinned() -> None:
    assert NEW_PIP_DEPENDENCIES == (
        "opentelemetry-sdk",
        "opentelemetry-instrumentation-fastapi",
        "opentelemetry-exporter-otlp-proto-http",
    )


def test_otel_adapter_version_pinned() -> None:
    assert OTEL_ADAPTER_VERSION == "1"


def test_constants_pinned() -> None:
    assert MAX_ATTRIBUTE_VALUE_LEN == 256
    assert MAX_SPAN_NAME_LEN == 128
    assert MAX_ATTRIBUTE_COUNT == 64
    assert MAX_SPAN_BUFFER == 65_536
    assert DEFAULT_SAMPLE_RATIO == 0.1


def test_tracer_error_is_runtime_error() -> None:
    assert issubclass(TracerError, RuntimeError)


# ---------------------------------------------------------------------------
# Attribute sanitisation
# ---------------------------------------------------------------------------
def test_sanitize_attribute_value_preserves_bool() -> None:
    assert sanitize_attribute_value(True) is True
    assert sanitize_attribute_value(False) is False


def test_sanitize_attribute_value_preserves_numbers() -> None:
    assert sanitize_attribute_value(42) == 42
    assert sanitize_attribute_value(3.14) == 3.14


def test_sanitize_attribute_value_none_becomes_underscore() -> None:
    assert sanitize_attribute_value(None) == "_"


def test_sanitize_attribute_value_empty_becomes_underscore() -> None:
    assert sanitize_attribute_value("") == "_"


def test_sanitize_attribute_value_replaces_control_chars() -> None:
    out = sanitize_attribute_value("a\x00b\x1fc\x7fd")
    assert out == "a_b_c_d"


def test_sanitize_attribute_value_escapes_backslash() -> None:
    assert sanitize_attribute_value("a\\b") == "a\\\\b"


def test_sanitize_attribute_value_escapes_quote() -> None:
    assert sanitize_attribute_value('a"b') == 'a\\"b'


def test_sanitize_attribute_value_clamps_length() -> None:
    s = "x" * (MAX_ATTRIBUTE_VALUE_LEN + 10)
    out = sanitize_attribute_value(s)
    assert isinstance(out, str)
    assert len(out) == MAX_ATTRIBUTE_VALUE_LEN


# ---------------------------------------------------------------------------
# derive_ids
# ---------------------------------------------------------------------------
def test_derive_ids_trace_is_16_hex() -> None:
    out = derive_ids(seed=1, kind="trace")
    assert len(out) == 16
    int(out, 16)


def test_derive_ids_span_is_8_hex() -> None:
    out = derive_ids(seed=1, kind="span")
    assert len(out) == 8
    int(out, 16)


def test_derive_ids_deterministic() -> None:
    assert derive_ids(seed=42, kind="trace") == derive_ids(seed=42, kind="trace")


def test_derive_ids_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError):
        derive_ids(seed=1, kind="other")


# ---------------------------------------------------------------------------
# SpanRecord
# ---------------------------------------------------------------------------
def test_span_record_is_frozen() -> None:
    record = SpanRecord(
        trace_id="t",
        span_id="s",
        parent_span_id=None,
        name="x",
        start_ts_ns=0,
        end_ts_ns=1,
        attributes=(),
    )
    with pytest.raises(dataclasses_frozen_error()):
        record.name = "y"  # type: ignore[misc]


def dataclasses_frozen_error() -> type[Exception]:
    return Exception


def test_span_record_duration_ns() -> None:
    record = SpanRecord(
        trace_id="t",
        span_id="s",
        parent_span_id=None,
        name="x",
        start_ts_ns=100,
        end_ts_ns=350,
        attributes=(),
    )
    assert record.duration_ns == 250


# ---------------------------------------------------------------------------
# InProcessTracer construction
# ---------------------------------------------------------------------------
def test_tracer_default_sample_ratio() -> None:
    t = InProcessTracer()
    assert t.sample_ratio == DEFAULT_SAMPLE_RATIO
    assert t.buffer_size == MAX_SPAN_BUFFER


def test_tracer_rejects_sample_ratio_above_one() -> None:
    with pytest.raises(ValueError):
        InProcessTracer(sample_ratio=1.5)


def test_tracer_rejects_sample_ratio_below_zero() -> None:
    with pytest.raises(ValueError):
        InProcessTracer(sample_ratio=-0.1)


def test_tracer_rejects_non_numeric_sample_ratio() -> None:
    with pytest.raises(TypeError):
        InProcessTracer(sample_ratio="0.1")  # type: ignore[arg-type]


def test_tracer_rejects_non_positive_buffer_size() -> None:
    with pytest.raises(ValueError):
        InProcessTracer(buffer_size=0)


def test_tracer_implements_protocol() -> None:
    assert isinstance(InProcessTracer(), Tracer)


# ---------------------------------------------------------------------------
# InProcessTracer start/end happy path
# ---------------------------------------------------------------------------
def _trace_with_one(sample_ratio: float = 1.0) -> InProcessTracer:
    t = InProcessTracer(sample_ratio=sample_ratio)
    t.start_span(
        name="signal.evaluate",
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        start_ts_ns=100,
        attributes={"engine": "intelligence"},
    )
    t.end_span(trace_id="trace-1", span_id="span-1", end_ts_ns=400)
    return t


def test_start_and_end_span_records_record() -> None:
    t = _trace_with_one()
    snapshot = t.snapshot()
    assert snapshot.sampled_in == 1
    assert snapshot.dropped == 0
    assert len(snapshot.spans) == 1
    record = snapshot.spans[0]
    assert record.trace_id == "trace-1"
    assert record.span_id == "span-1"
    assert record.parent_span_id is None
    assert record.name == "signal.evaluate"
    assert record.start_ts_ns == 100
    assert record.end_ts_ns == 400
    assert record.duration_ns == 300
    assert record.attributes == (("engine", "intelligence"),)


def test_attributes_canonicalised_sorted() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=0,
        attributes={"b": 2, "a": 1, "c": 3},
    )
    t.end_span(trace_id="t1", span_id="s1", end_ts_ns=10)
    record = t.snapshot().spans[0]
    assert record.attributes == (("a", 1), ("b", 2), ("c", 3))


def test_end_span_merges_attributes() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=0,
        attributes={"a": 1},
    )
    t.end_span(
        trace_id="t1",
        span_id="s1",
        end_ts_ns=10,
        attributes={"b": 2},
    )
    record = t.snapshot().spans[0]
    assert record.attributes == (("a", 1), ("b", 2))


def test_end_span_overrides_start_attributes() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=0,
        attributes={"a": 1},
    )
    t.end_span(
        trace_id="t1",
        span_id="s1",
        end_ts_ns=10,
        attributes={"a": 99},
    )
    record = t.snapshot().spans[0]
    assert record.attributes == (("a", 99),)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
def test_start_span_rejects_empty_name() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(ValueError):
        t.start_span(
            name="",
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
        )


def test_start_span_rejects_long_name() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(ValueError):
        t.start_span(
            name="x" * (MAX_SPAN_NAME_LEN + 1),
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
        )


def test_start_span_rejects_non_str_name() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(TypeError):
        t.start_span(
            name=123,  # type: ignore[arg-type]
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
        )


def test_start_span_rejects_empty_trace_id() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(ValueError):
        t.start_span(
            name="x",
            trace_id="",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
        )


def test_start_span_rejects_empty_span_id() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(ValueError):
        t.start_span(
            name="x",
            trace_id="t1",
            span_id="",
            parent_span_id=None,
            start_ts_ns=0,
        )


def test_start_span_rejects_negative_start_ts_ns() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(ValueError):
        t.start_span(
            name="x",
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=-1,
        )


def test_start_span_rejects_non_int_start_ts_ns() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(ValueError):
        t.start_span(
            name="x",
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0.5,  # type: ignore[arg-type]
        )


def test_start_span_rejects_non_str_parent() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(TypeError):
        t.start_span(
            name="x",
            trace_id="t1",
            span_id="s1",
            parent_span_id=42,  # type: ignore[arg-type]
            start_ts_ns=0,
        )


def test_start_span_rejects_duplicate_open_span() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=0,
    )
    with pytest.raises(TracerError):
        t.start_span(
            name="x",
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=1,
        )


def test_end_span_rejects_unknown_span() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(TracerError):
        t.end_span(trace_id="t1", span_id="s1", end_ts_ns=10)


def test_end_span_rejects_end_before_start() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=100,
    )
    with pytest.raises(ValueError):
        t.end_span(trace_id="t1", span_id="s1", end_ts_ns=50)


def test_start_span_rejects_too_many_attributes() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    attrs = {f"k{i}": i for i in range(MAX_ATTRIBUTE_COUNT + 1)}
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=0,
        attributes=attrs,
    )
    with pytest.raises(ValueError):
        t.end_span(trace_id="t1", span_id="s1", end_ts_ns=10)


def test_start_span_rejects_non_mapping_attributes() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    with pytest.raises(TypeError):
        t.start_span(
            name="x",
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
            attributes=[("a", 1)],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_sample_ratio_zero_drops_all() -> None:
    t = InProcessTracer(sample_ratio=0.0)
    t.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=0,
    )
    t.end_span(trace_id="t1", span_id="s1", end_ts_ns=10)
    snapshot = t.snapshot()
    assert snapshot.sampled_in == 0
    assert snapshot.spans == ()


def test_sample_ratio_one_keeps_all() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    for i in range(8):
        tid = f"trace-{i}"
        t.start_span(
            name="x",
            trace_id=tid,
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
        )
        t.end_span(trace_id=tid, span_id="s1", end_ts_ns=10)
    assert t.snapshot().sampled_in == 8


def test_sampling_is_deterministic_per_trace_id() -> None:
    t = InProcessTracer(sample_ratio=0.5)
    keep_one = set()
    keep_two = set()
    for i in range(64):
        tid = f"trace-{i}"
        if t._is_sampled(tid):  # noqa: SLF001
            keep_one.add(tid)
    t2 = InProcessTracer(sample_ratio=0.5)
    for i in range(64):
        tid = f"trace-{i}"
        if t2._is_sampled(tid):  # noqa: SLF001
            keep_two.add(tid)
    assert keep_one == keep_two


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def _make_three_spans(sample_ratio: float = 1.0) -> tuple[SpanRecord, ...]:
    t = InProcessTracer(sample_ratio=sample_ratio)
    t.start_span(
        name="signal.evaluate",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=100,
        attributes={"engine": "intelligence"},
    )
    t.start_span(
        name="governance.approve",
        trace_id="t1",
        span_id="s2",
        parent_span_id="s1",
        start_ts_ns=150,
        attributes={"engine": "governance"},
    )
    t.start_span(
        name="execution.route",
        trace_id="t1",
        span_id="s3",
        parent_span_id="s2",
        start_ts_ns=200,
        attributes={"engine": "execution"},
    )
    t.end_span(trace_id="t1", span_id="s3", end_ts_ns=250)
    t.end_span(trace_id="t1", span_id="s2", end_ts_ns=300)
    t.end_span(trace_id="t1", span_id="s1", end_ts_ns=400)
    return t.snapshot().spans


def test_snapshot_is_byte_identical_across_runs() -> None:
    s1 = _make_three_spans()
    s2 = _make_three_spans()
    s3 = _make_three_spans()
    assert s1 == s2 == s3


def test_snapshot_ordered_by_trace_then_span() -> None:
    t = InProcessTracer(sample_ratio=1.0)
    for tid, sid in [("t1", "s2"), ("t1", "s1"), ("t2", "s1"), ("t1", "s3")]:
        t.start_span(
            name="x",
            trace_id=tid,
            span_id=sid,
            parent_span_id=None,
            start_ts_ns=0,
        )
        t.end_span(trace_id=tid, span_id=sid, end_ts_ns=10)
    ids = [(r.trace_id, r.span_id) for r in t.snapshot().spans]
    assert ids == sorted(ids)


def test_render_trace_text_is_deterministic() -> None:
    spans = _make_three_spans()
    snapshot = TraceSnapshot(spans=spans, sampled_in=3, dropped=0)
    assert render_trace_text(snapshot) == render_trace_text(snapshot)


def test_render_trace_text_has_canonical_header() -> None:
    spans = _make_three_spans()
    text = render_trace_text(TraceSnapshot(spans=spans, sampled_in=3, dropped=0))
    lines = text.splitlines()
    assert lines[0] == "# sampled_in 3"
    assert lines[1] == "# dropped 0"


# ---------------------------------------------------------------------------
# Buffer overflow
# ---------------------------------------------------------------------------
def test_buffer_overflow_increments_dropped() -> None:
    t = InProcessTracer(sample_ratio=1.0, buffer_size=2)
    for i in range(5):
        t.start_span(
            name="x",
            trace_id=f"t{i}",
            span_id="s1",
            parent_span_id=None,
            start_ts_ns=0,
        )
        t.end_span(trace_id=f"t{i}", span_id="s1", end_ts_ns=10)
    snapshot = t.snapshot()
    assert len(snapshot.spans) == 2
    assert snapshot.dropped == 3


# ---------------------------------------------------------------------------
# derive helpers on tracer
# ---------------------------------------------------------------------------
def test_tracer_derive_trace_id() -> None:
    t = InProcessTracer()
    out = t.derive_trace_id(seed=42)
    assert len(out) == 16
    assert t.derive_trace_id(seed=42) == out


def test_tracer_derive_span_id() -> None:
    t = InProcessTracer()
    out = t.derive_span_id(seed=42)
    assert len(out) == 8
    assert t.derive_span_id(seed=42) == out


# ---------------------------------------------------------------------------
# OTel factory — without dep installed
# ---------------------------------------------------------------------------
def test_otel_tracer_factory_rejects_empty_service_name() -> None:
    with pytest.raises(ValueError):
        otel_tracer_factory(service_name="")


def test_otel_tracer_factory_accepts_injected_tracer() -> None:
    class _FakeSpan:
        def __init__(self) -> None:
            self.end_called: int | None = None
            self.attrs: dict[str, Any] = {}  # type: ignore[name-defined]  # noqa: F821

        def set_attribute(self, key: str, value: Any) -> None:  # noqa: F821
            self.attrs[key] = value

        def end(self, end_time: int) -> None:
            self.end_called = end_time

    class _FakeTracer:
        def __init__(self) -> None:
            self.last: _FakeSpan | None = None

        def start_span(
            self,
            *,
            name: str,
            start_time: int,
            attributes: dict[str, object],
        ) -> _FakeSpan:
            span = _FakeSpan()
            self.last = span
            return span

    fake = _FakeTracer()
    tracer = otel_tracer_factory(service_name="svc", otel_tracer=fake)
    tracer.start_span(
        name="x",
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        start_ts_ns=10,
        attributes={"a": 1},
    )
    record = tracer.end_span(trace_id="t1", span_id="s1", end_ts_ns=20)
    assert isinstance(record, SpanRecord)
    assert record.duration_ns == 10
    assert fake.last is not None
    assert fake.last.end_called == 20


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------
def _module_ast() -> ast.Module:
    path = pathlib.Path(tracer_mod.__file__)
    return ast.parse(path.read_text())


def _module_top_level_imports() -> set[str]:
    names: set[str] = set()
    for node in _module_ast().body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_top_level_opentelemetry_import() -> None:
    assert "opentelemetry" not in _module_top_level_imports()


def test_no_top_level_clock_imports() -> None:
    forbidden = {
        "time",
        "datetime",
        "random",
        "os",
        "asyncio",
        "socket",
        "requests",
        "urllib",
        "httpx",
    }
    assert forbidden.isdisjoint(_module_top_level_imports())


def test_no_top_level_numpy_torch_polars_imports() -> None:
    forbidden = {"numpy", "torch", "polars", "scipy", "pandas"}
    assert forbidden.isdisjoint(_module_top_level_imports())


def test_no_governance_or_execution_engine_imports() -> None:
    forbidden = {
        "governance_engine",
        "execution_engine",
        "evolution_engine",
    }
    assert forbidden.isdisjoint(_module_top_level_imports())


def test_no_typed_event_constructions() -> None:
    forbidden_names = {
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "HazardEvent",
        "GovernanceDecision",
        "PatchProposal",
    }
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in forbidden_names:
                raise AssertionError(f"forbidden constructor call: {fn.id}")
            if isinstance(fn, ast.Attribute) and fn.attr in forbidden_names:
                raise AssertionError(f"forbidden constructor call: {fn.attr}")


def test_adapted_from_header_present() -> None:
    path = pathlib.Path(tracer_mod.__file__)
    text = path.read_text()
    assert "# ADAPTED FROM: open-telemetry/opentelemetry-python" in text


def test_otel_import_confined_to_factory_body() -> None:
    """opentelemetry import must only appear inside otel_tracer_factory."""
    tree = _module_ast()
    factory_def: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "otel_tracer_factory":
            factory_def = node
            break
    assert factory_def is not None
    # walk whole module — any opentelemetry import must be inside the factory
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_name = (node.module if isinstance(node, ast.ImportFrom) else "") or ""
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            mentions_otel = module_name.startswith("opentelemetry") or any(
                n.startswith("opentelemetry") for n in names
            )
            if not mentions_otel:
                continue
            # find ancestor — must be inside factory_def
            assert any(child is node for child in ast.walk(factory_def)), (
                "opentelemetry import must be inside otel_tracer_factory"
            )
