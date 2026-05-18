"""Tests for A-08 ``system_engine/metrics/exporter.py``."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
)
from core.contracts.governance import DecisionKind, GovernanceDecision
from system_engine.metrics import exporter as exporter_mod
from system_engine.metrics.exporter import (
    DEFAULT_EXECUTION_LATENCY_BUCKETS_NS,
    EXECUTION_LATENCY_NS,
    EXECUTIONS_TOTAL,
    GOVERNANCE_APPROVALS_TOTAL,
    HAZARD_EVENTS_TOTAL,
    HEALTH_STATE,
    MAX_LABEL_VALUE_LEN,
    NEW_PIP_DEPENDENCIES,
    PNL_USD,
    PROMETHEUS_ADAPTER_VERSION,
    InProcessMetricsSink,
    MetricsExporter,
    MetricsExporterError,
    MetricsSink,
    MetricsSnapshotError,
    render_prometheus_text,
    sanitize_label_value,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_pins_prometheus_client() -> None:
    assert NEW_PIP_DEPENDENCIES == ("prometheus-client",)


def test_adapter_version_is_stable() -> None:
    assert PROMETHEUS_ADAPTER_VERSION == "1"


def test_metric_name_constants_match_spec() -> None:
    assert EXECUTIONS_TOTAL == "executions_total"
    assert PNL_USD == "pnl_usd"
    assert EXECUTION_LATENCY_NS == "execution_latency_ns"
    assert HAZARD_EVENTS_TOTAL == "hazard_events_total"
    assert GOVERNANCE_APPROVALS_TOTAL == "governance_approvals_total"
    assert HEALTH_STATE == "health_state"


def test_max_label_value_len_is_finite_positive_int() -> None:
    assert isinstance(MAX_LABEL_VALUE_LEN, int)
    assert MAX_LABEL_VALUE_LEN > 0


def test_default_buckets_are_strictly_increasing_positive() -> None:
    buckets = DEFAULT_EXECUTION_LATENCY_BUCKETS_NS
    assert len(buckets) > 0
    assert all(b > 0 for b in buckets)
    # buckets[1:] is intentionally one shorter than buckets.
    pairs = zip(buckets, buckets[1:])  # noqa: B905
    assert all(b1 < b2 for b1, b2 in pairs)


def test_metrics_exporter_error_is_runtime_error_subclass() -> None:
    assert issubclass(MetricsExporterError, RuntimeError)
    assert issubclass(MetricsSnapshotError, MetricsExporterError)


# ---------------------------------------------------------------------------
# AST guards (INV-15 + tier discipline)
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(exporter_mod.__file__)
_MODULE_TREE = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports() -> list[str]:
    names: list[str] = []
    for node in _MODULE_TREE.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_prometheus_client_import() -> None:
    assert not any(
        name == "prometheus_client" or name.startswith("prometheus_client.")
        for name in _top_level_imports()
    )


def test_no_top_level_clock_random_or_io_imports() -> None:
    forbidden = {
        "asyncio",
        "datetime",
        "numpy",
        "os",
        "polars",
        "random",
        "requests",
        "socket",
        "time",
        "torch",
    }
    assert forbidden.isdisjoint(_top_level_imports())


def test_no_engine_cross_imports() -> None:
    """system_engine.metrics may import core.contracts but NOT engines."""

    forbidden_prefixes = (
        "evolution_engine",
        "execution_engine",
        "governance_engine",
        "intelligence_engine",
    )
    for name in _top_level_imports():
        assert not name.startswith(forbidden_prefixes), name


def test_no_typed_event_construction_in_module() -> None:
    """B27 / B28 / INV-71 — metrics module never builds typed bus events.

    GovernanceDecision is referenced as a *type* for ``isinstance`` checks
    in :meth:`MetricsExporter.record_governance_decision`, but never
    constructed (no ``GovernanceDecision(...)`` call expressions).
    """

    forbidden = {
        "ExecutionEvent",
        "GovernanceDecision",
        "HazardEvent",
        "PatchProposal",
        "SignalEvent",
        "SystemEvent",
    }
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                raise AssertionError(f"forbidden typed-event constructor call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                raise AssertionError(
                    f"forbidden typed-event constructor attribute call: {func.attr}"
                )


def test_prometheus_client_import_only_inside_factory() -> None:
    """``prometheus_client`` is imported only inside the factory body."""

    factory: ast.FunctionDef | None = None
    for node in _MODULE_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "prometheus_metrics_sink_factory":
            factory = node
            break
    assert factory is not None, "factory function not found"

    found_in_factory = False
    for sub in ast.walk(factory):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.name == "prometheus_client":
                    found_in_factory = True
    assert found_in_factory


# ---------------------------------------------------------------------------
# sanitize_label_value
# ---------------------------------------------------------------------------


def test_sanitize_label_value_passes_normal_text() -> None:
    assert sanitize_label_value("BTC-USDT") == "BTC-USDT"
    assert sanitize_label_value("BUY") == "BUY"


def test_sanitize_label_value_replaces_control_chars() -> None:
    assert sanitize_label_value("a\nb") == "a_b"
    assert sanitize_label_value("a\x00b") == "a_b"
    assert sanitize_label_value("a\x7fb") == "a_b"


def test_sanitize_label_value_escapes_backslash_and_quote() -> None:
    assert sanitize_label_value('a"b') == 'a\\"b'
    assert sanitize_label_value("a\\b") == "a\\\\b"


def test_sanitize_label_value_clamps_to_max_len() -> None:
    long_text = "x" * (MAX_LABEL_VALUE_LEN * 2)
    result = sanitize_label_value(long_text)
    assert len(result) == MAX_LABEL_VALUE_LEN


def test_sanitize_label_value_empty_falls_back() -> None:
    assert sanitize_label_value("") == "_"
    assert sanitize_label_value(None) == "_"


def test_sanitize_label_value_coerces_non_string() -> None:
    assert sanitize_label_value(123) == "123"
    assert sanitize_label_value(True) == "True"


# ---------------------------------------------------------------------------
# InProcessMetricsSink
# ---------------------------------------------------------------------------


def _make_sink() -> InProcessMetricsSink:
    return InProcessMetricsSink()


def test_in_process_sink_satisfies_metrics_sink_protocol() -> None:
    sink = _make_sink()
    assert isinstance(sink, MetricsSink)


def test_in_process_sink_rejects_empty_buckets() -> None:
    with pytest.raises(ValueError):
        InProcessMetricsSink(execution_latency_buckets_ns=())


def test_in_process_sink_rejects_non_increasing_buckets() -> None:
    with pytest.raises(ValueError):
        InProcessMetricsSink(execution_latency_buckets_ns=(10.0, 5.0))


def test_in_process_sink_rejects_non_positive_buckets() -> None:
    with pytest.raises(ValueError):
        InProcessMetricsSink(execution_latency_buckets_ns=(0.0, 10.0))


def test_in_process_sink_rejects_non_numeric_buckets() -> None:
    with pytest.raises(TypeError):
        InProcessMetricsSink(
            execution_latency_buckets_ns=("a", "b"),  # type: ignore[arg-type]
        )


def test_inc_executions_accumulates_per_label_set() -> None:
    sink = _make_sink()
    sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    sink.inc_executions(symbol="ETH", side="SELL", status="FILLED")

    snapshot = sink.snapshot()
    counter = snapshot["counters"][EXECUTIONS_TOTAL]
    assert len(counter) == 2
    for key, value in counter.items():
        labels = dict(key)
        if labels["symbol"] == "BTC":
            assert value == 2.0
        else:
            assert value == 1.0


def test_set_pnl_usd_overwrites_per_symbol() -> None:
    sink = _make_sink()
    sink.set_pnl_usd(symbol="BTC", value=100.0)
    sink.set_pnl_usd(symbol="BTC", value=125.0)
    sink.set_pnl_usd(symbol="ETH", value=-50.0)

    snapshot = sink.snapshot()
    pnl = snapshot["gauges"][PNL_USD]
    assert pnl[(("symbol", "BTC"),)] == 125.0
    assert pnl[(("symbol", "ETH"),)] == -50.0


def test_set_pnl_usd_rejects_non_numeric() -> None:
    sink = _make_sink()
    with pytest.raises(TypeError):
        sink.set_pnl_usd(symbol="BTC", value="100")  # type: ignore[arg-type]


def test_set_pnl_usd_rejects_bool() -> None:
    sink = _make_sink()
    with pytest.raises(TypeError):
        sink.set_pnl_usd(symbol="BTC", value=True)  # type: ignore[arg-type]


def test_set_pnl_usd_rejects_nan_and_inf() -> None:
    sink = _make_sink()
    with pytest.raises(ValueError):
        sink.set_pnl_usd(symbol="BTC", value=float("nan"))
    with pytest.raises(ValueError):
        sink.set_pnl_usd(symbol="BTC", value=float("inf"))


def test_observe_execution_latency_ns_buckets_correctly() -> None:
    sink = InProcessMetricsSink(
        execution_latency_buckets_ns=(10.0, 100.0, 1000.0),
    )
    sink.observe_execution_latency_ns(5.0)
    sink.observe_execution_latency_ns(50.0)
    sink.observe_execution_latency_ns(500.0)
    sink.observe_execution_latency_ns(50_000.0)

    snapshot = sink.snapshot()
    hist = snapshot["histograms"][EXECUTION_LATENCY_NS]
    assert hist["count"] == 4
    assert hist["sum"] == pytest.approx(5.0 + 50.0 + 500.0 + 50_000.0)
    # le=10 catches 5
    # le=100 catches 5, 50
    # le=1000 catches 5, 50, 500
    assert hist["bucket_counts"] == (1, 2, 3)


def test_observe_execution_latency_ns_rejects_negative() -> None:
    sink = _make_sink()
    with pytest.raises(ValueError):
        sink.observe_execution_latency_ns(-1.0)


def test_inc_hazard_groups_by_kind_and_severity() -> None:
    sink = _make_sink()
    sink.inc_hazard(kind="HAZ-01", severity="HIGH")
    sink.inc_hazard(kind="HAZ-01", severity="HIGH")
    sink.inc_hazard(kind="HAZ-02", severity="LOW")

    snapshot = sink.snapshot()
    counter = snapshot["counters"][HAZARD_EVENTS_TOTAL]
    assert len(counter) == 2


def test_inc_governance_groups_by_verdict() -> None:
    sink = _make_sink()
    sink.inc_governance(verdict="APPROVED")
    sink.inc_governance(verdict="APPROVED")
    sink.inc_governance(verdict="REJECTED")

    snapshot = sink.snapshot()
    counter = snapshot["counters"][GOVERNANCE_APPROVALS_TOTAL]
    assert counter[(("verdict", "APPROVED"),)] == 2.0
    assert counter[(("verdict", "REJECTED"),)] == 1.0


def test_set_health_state_accepts_unit_interval() -> None:
    sink = _make_sink()
    sink.set_health_state(0.0)
    sink.set_health_state(1.0)
    sink.set_health_state(0.5)


def test_set_health_state_rejects_out_of_range() -> None:
    sink = _make_sink()
    with pytest.raises(ValueError):
        sink.set_health_state(1.5)
    with pytest.raises(ValueError):
        sink.set_health_state(-0.1)


def test_in_process_sink_render_is_deterministic_byte_identical() -> None:
    sink_a = _make_sink()
    sink_b = _make_sink()
    for sink in (sink_a, sink_b):
        sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
        sink.inc_executions(symbol="ETH", side="SELL", status="FILLED")
        sink.set_pnl_usd(symbol="BTC", value=100.5)
        sink.observe_execution_latency_ns(1e6)
        sink.inc_hazard(kind="HAZ-01", severity="HIGH")
        sink.inc_governance(verdict="APPROVED")
        sink.set_health_state(0.75)

    assert sink_a.render() == sink_b.render()


def test_in_process_sink_render_three_run_byte_identical() -> None:
    renders: set[bytes] = set()
    for _ in range(3):
        sink = _make_sink()
        sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
        sink.set_pnl_usd(symbol="BTC", value=42.0)
        sink.observe_execution_latency_ns(5000.0)
        sink.inc_hazard(kind="HAZ-01", severity="HIGH")
        sink.inc_governance(verdict="APPROVED")
        sink.set_health_state(0.5)
        renders.add(sink.render())
    assert len(renders) == 1


def test_in_process_sink_snapshot_is_order_independent() -> None:
    sink_a = _make_sink()
    sink_b = _make_sink()
    sink_a.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    sink_a.inc_executions(symbol="ETH", side="SELL", status="FILLED")
    sink_b.inc_executions(symbol="ETH", side="SELL", status="FILLED")
    sink_b.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    assert sink_a.render() == sink_b.render()


def test_in_process_sink_render_includes_type_lines() -> None:
    sink = _make_sink()
    sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    sink.set_pnl_usd(symbol="BTC", value=10.0)
    sink.observe_execution_latency_ns(100.0)
    sink.inc_hazard(kind="HAZ-01", severity="HIGH")
    sink.inc_governance(verdict="APPROVED")
    sink.set_health_state(0.5)

    text = sink.render().decode("utf-8")
    assert f"# TYPE {EXECUTIONS_TOTAL} counter" in text
    assert f"# TYPE {PNL_USD} gauge" in text
    assert f"# TYPE {EXECUTION_LATENCY_NS} histogram" in text
    assert f"# TYPE {HAZARD_EVENTS_TOTAL} counter" in text
    assert f"# TYPE {GOVERNANCE_APPROVALS_TOTAL} counter" in text
    assert f"# TYPE {HEALTH_STATE} gauge" in text


def test_in_process_sink_render_includes_help_lines() -> None:
    sink = _make_sink()
    sink.set_health_state(0.5)
    text = sink.render().decode("utf-8")
    assert f"# HELP {HEALTH_STATE}" in text


def test_in_process_sink_histogram_includes_plus_inf_bucket() -> None:
    sink = _make_sink()
    sink.observe_execution_latency_ns(100.0)
    text = sink.render().decode("utf-8")
    assert 'le="+Inf"' in text
    assert f"{EXECUTION_LATENCY_NS}_sum" in text
    assert f"{EXECUTION_LATENCY_NS}_count" in text


# ---------------------------------------------------------------------------
# render_prometheus_text (direct)
# ---------------------------------------------------------------------------


def test_render_prometheus_text_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        render_prometheus_text("not a mapping")  # type: ignore[arg-type]


def test_render_prometheus_text_empty_snapshot_is_empty_bytes() -> None:
    assert render_prometheus_text({"counters": {}, "gauges": {}, "histograms": {}}) == b""


def test_render_prometheus_text_returns_bytes() -> None:
    sink = _make_sink()
    sink.set_health_state(0.5)
    result = render_prometheus_text(sink.snapshot())
    assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# MetricsExporter — record_execution / record_hazard / record_governance
# ---------------------------------------------------------------------------


def _exec_event(
    *,
    symbol: str = "BTC",
    side: Side = Side.BUY,
    status: ExecutionStatus = ExecutionStatus.FILLED,
    qty: float = 1.0,
    price: float = 50_000.0,
) -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=0,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        status=status,
    )


def _hazard_event(
    *,
    code: str = "HAZ-01",
    severity: HazardSeverity = HazardSeverity.HIGH,
) -> HazardEvent:
    return HazardEvent(ts_ns=0, code=code, severity=severity, source="test")


def _approval_decision(*, approved: bool = True) -> GovernanceDecision:
    return GovernanceDecision(
        ts_ns=0,
        kind=DecisionKind.MODE_TRANSITION,
        approved=approved,
        summary="t",
    )


def test_exporter_requires_metrics_sink() -> None:
    with pytest.raises(TypeError):
        MetricsExporter(object())  # type: ignore[arg-type]


def test_exporter_exposes_sink_property() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    assert exporter.sink is sink


def test_record_execution_increments_counter() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event())
    snapshot = sink.snapshot()
    counter = snapshot["counters"][EXECUTIONS_TOTAL]
    assert sum(counter.values()) == 1.0
    key = next(iter(counter))
    labels = dict(key)
    assert labels["symbol"] == "BTC"
    assert labels["side"] == "BUY"
    assert labels["status"] == "FILLED"


def test_record_execution_optional_pnl_sets_gauge() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event(), pnl_usd=42.5)
    snapshot = sink.snapshot()
    assert snapshot["gauges"][PNL_USD][(("symbol", "BTC"),)] == 42.5


def test_record_execution_optional_latency_observes_histogram() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event(), latency_ns=5_000.0)
    snapshot = sink.snapshot()
    assert snapshot["histograms"][EXECUTION_LATENCY_NS]["count"] == 1


def test_record_execution_skips_pnl_and_latency_when_none() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event())
    snapshot = sink.snapshot()
    assert snapshot["gauges"][PNL_USD] == {}
    assert snapshot["histograms"][EXECUTION_LATENCY_NS]["count"] == 0


def test_record_execution_rejects_non_execution_event() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    with pytest.raises(TypeError):
        exporter.record_execution(object())  # type: ignore[arg-type]


def test_record_hazard_increments_counter_with_severity() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_hazard(_hazard_event())
    snapshot = sink.snapshot()
    counter = snapshot["counters"][HAZARD_EVENTS_TOTAL]
    assert len(counter) == 1
    key = next(iter(counter))
    labels = dict(key)
    assert labels["kind"] == "HAZ-01"
    assert labels["severity"] == "HIGH"


def test_record_hazard_rejects_non_hazard_event() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    with pytest.raises(TypeError):
        exporter.record_hazard(object())  # type: ignore[arg-type]


def test_record_governance_decision_maps_approved_true() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_governance_decision(_approval_decision(approved=True))
    snapshot = sink.snapshot()
    counter = snapshot["counters"][GOVERNANCE_APPROVALS_TOTAL]
    assert counter[(("verdict", "APPROVED"),)] == 1.0


def test_record_governance_decision_maps_approved_false() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_governance_decision(_approval_decision(approved=False))
    snapshot = sink.snapshot()
    counter = snapshot["counters"][GOVERNANCE_APPROVALS_TOTAL]
    assert counter[(("verdict", "REJECTED"),)] == 1.0


def test_record_governance_decision_rejects_non_decision() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    with pytest.raises(TypeError):
        exporter.record_governance_decision(object())  # type: ignore[arg-type]


def test_set_health_state_writes_through() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.set_health_state(0.7)
    snapshot = sink.snapshot()
    assert snapshot["gauges"][HEALTH_STATE][()] == pytest.approx(0.7)


def test_set_health_state_clamps_via_sink_rejects_out_of_range() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    with pytest.raises(ValueError):
        exporter.set_health_state(2.0)


def test_exporter_render_returns_bytes() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.set_health_state(0.5)
    result = exporter.render()
    assert isinstance(result, bytes)


def test_record_execution_event_kind_is_execution() -> None:
    """Sanity check that we are projecting EVT-02 (execution) events."""

    event = _exec_event()
    assert event.kind == EventKind.EXECUTION


def test_record_hazard_event_kind_is_hazard() -> None:
    event = _hazard_event()
    assert event.kind == EventKind.HAZARD


# ---------------------------------------------------------------------------
# Label projection: events drive labels, never hardcoded strings.
# ---------------------------------------------------------------------------


def test_record_execution_uses_event_symbol() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event(symbol="SOL-USDT"))
    counter = sink.snapshot()["counters"][EXECUTIONS_TOTAL]
    key = next(iter(counter))
    assert dict(key)["symbol"] == "SOL-USDT"


def test_record_execution_uses_event_side_enum_value() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event(side=Side.SELL))
    counter = sink.snapshot()["counters"][EXECUTIONS_TOTAL]
    key = next(iter(counter))
    assert dict(key)["side"] == "SELL"


def test_record_execution_uses_event_status_enum_value() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_execution(_exec_event(status=ExecutionStatus.REJECTED))
    counter = sink.snapshot()["counters"][EXECUTIONS_TOTAL]
    key = next(iter(counter))
    assert dict(key)["status"] == "REJECTED"


def test_record_hazard_uses_event_code_as_kind() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_hazard(_hazard_event(code="HAZ-NEWS-SHOCK"))
    counter = sink.snapshot()["counters"][HAZARD_EVENTS_TOTAL]
    key = next(iter(counter))
    assert dict(key)["kind"] == "HAZ-NEWS-SHOCK"


def test_record_hazard_uses_severity_enum_value() -> None:
    sink = _make_sink()
    exporter = MetricsExporter(sink)
    exporter.record_hazard(_hazard_event(severity=HazardSeverity.LOW))
    counter = sink.snapshot()["counters"][HAZARD_EVENTS_TOTAL]
    key = next(iter(counter))
    assert dict(key)["severity"] == "LOW"


# ---------------------------------------------------------------------------
# Determinism & order independence over the exporter
# ---------------------------------------------------------------------------


def test_exporter_render_three_run_byte_identical() -> None:
    renders: set[bytes] = set()
    for _ in range(3):
        sink = _make_sink()
        exporter = MetricsExporter(sink)
        exporter.record_execution(_exec_event(), pnl_usd=10.0, latency_ns=500.0)
        exporter.record_hazard(_hazard_event())
        exporter.record_governance_decision(_approval_decision(approved=True))
        exporter.record_governance_decision(_approval_decision(approved=False))
        exporter.set_health_state(0.5)
        renders.add(exporter.render())
    assert len(renders) == 1


def test_exporter_render_is_event_order_independent() -> None:
    sink_a = _make_sink()
    sink_b = _make_sink()
    exp_a = MetricsExporter(sink_a)
    exp_b = MetricsExporter(sink_b)

    exp_a.record_execution(_exec_event(symbol="BTC"))
    exp_a.record_execution(_exec_event(symbol="ETH"))
    exp_b.record_execution(_exec_event(symbol="ETH"))
    exp_b.record_execution(_exec_event(symbol="BTC"))

    assert exp_a.render() == exp_b.render()


# ---------------------------------------------------------------------------
# Stress / boundary
# ---------------------------------------------------------------------------


def test_inc_executions_one_thousand_increments() -> None:
    sink = _make_sink()
    for _ in range(1_000):
        sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    snapshot = sink.snapshot()
    counter = snapshot["counters"][EXECUTIONS_TOTAL]
    assert sum(counter.values()) == 1_000.0


def test_observe_latency_zero_is_accepted() -> None:
    sink = _make_sink()
    sink.observe_execution_latency_ns(0.0)
    snapshot = sink.snapshot()
    assert snapshot["histograms"][EXECUTION_LATENCY_NS]["count"] == 1


def test_set_pnl_usd_handles_negative() -> None:
    sink = _make_sink()
    sink.set_pnl_usd(symbol="BTC", value=-1000.5)
    snapshot = sink.snapshot()
    assert snapshot["gauges"][PNL_USD][(("symbol", "BTC"),)] == -1000.5


def test_inc_executions_distinct_label_sets_remain_separate() -> None:
    sink = _make_sink()
    sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    sink.inc_executions(symbol="BTC", side="SELL", status="FILLED")
    sink.inc_executions(symbol="BTC", side="BUY", status="REJECTED")

    snapshot = sink.snapshot()
    counter = snapshot["counters"][EXECUTIONS_TOTAL]
    assert len(counter) == 3
    for value in counter.values():
        assert value == 1.0


# ---------------------------------------------------------------------------
# prometheus_metrics_sink_factory — only runs when prometheus_client is
# importable. We DO NOT want pytest to require the dependency, so the
# factory tests are wrapped with importorskip.
# ---------------------------------------------------------------------------


def _prometheus_or_skip() -> None:
    pytest.importorskip("prometheus_client")


def test_prometheus_sink_factory_returns_metrics_sink() -> None:
    _prometheus_or_skip()
    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    sink = prometheus_metrics_sink_factory()
    assert isinstance(sink, MetricsSink)


def test_prometheus_sink_factory_records_through_official_client() -> None:
    _prometheus_or_skip()
    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    sink = prometheus_metrics_sink_factory()
    sink.inc_executions(symbol="BTC", side="BUY", status="FILLED")
    sink.set_pnl_usd(symbol="BTC", value=42.0)
    sink.observe_execution_latency_ns(123.0)
    sink.inc_hazard(kind="HAZ-01", severity="HIGH")
    sink.inc_governance(verdict="APPROVED")
    sink.set_health_state(0.5)
    output = sink.render()
    text = output.decode("utf-8")
    assert "executions_total" in text
    assert "pnl_usd" in text
    assert "execution_latency_ns" in text
    assert "hazard_events_total" in text
    assert "governance_approvals_total" in text
    assert "health_state" in text


def test_prometheus_sink_factory_rejects_empty_buckets() -> None:
    _prometheus_or_skip()
    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    with pytest.raises(ValueError):
        prometheus_metrics_sink_factory(execution_latency_buckets_ns=())


def test_prometheus_sink_factory_uses_custom_registry() -> None:
    _prometheus_or_skip()
    import prometheus_client

    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    registry = prometheus_client.CollectorRegistry(auto_describe=False)
    sink = prometheus_metrics_sink_factory(registry=registry)
    sink.set_health_state(0.5)
    text = prometheus_client.generate_latest(registry).decode("utf-8")
    assert "health_state" in text


def test_prometheus_sink_rejects_invalid_health_state() -> None:
    _prometheus_or_skip()
    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    sink = prometheus_metrics_sink_factory()
    with pytest.raises(ValueError):
        sink.set_health_state(2.0)


def test_prometheus_sink_rejects_negative_latency() -> None:
    _prometheus_or_skip()
    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    sink = prometheus_metrics_sink_factory()
    with pytest.raises(ValueError):
        sink.observe_execution_latency_ns(-1.0)


def test_prometheus_sink_rejects_nan_pnl() -> None:
    _prometheus_or_skip()
    from system_engine.metrics.exporter import prometheus_metrics_sink_factory

    sink = prometheus_metrics_sink_factory()
    with pytest.raises(ValueError):
        sink.set_pnl_usd(symbol="BTC", value=math.nan)
