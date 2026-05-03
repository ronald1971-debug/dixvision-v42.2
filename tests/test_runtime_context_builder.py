"""P0-4 — RuntimeContext builder tests."""

from __future__ import annotations

import pytest

from core.contracts.risk import RiskSnapshot
from intelligence_engine import (
    DEFAULT_LATENCY_BUDGET_NS,
    RuntimeMonitorView,
    build_runtime_context,
)


def _risk(*, halted: bool = False) -> RiskSnapshot:
    return RiskSnapshot(version=1, ts_ns=1_000, halted=halted)


def _monitor(
    *,
    fail_rate: float = 0.0,
    reject_rate: float = 0.0,
    p99_latency_ns: int = 0,
) -> RuntimeMonitorView:
    return RuntimeMonitorView(
        fail_rate=fail_rate,
        reject_rate=reject_rate,
        p99_latency_ns=p99_latency_ns,
    )


# ---------------------------------------------------------------------------
# Happy path — defaults
# ---------------------------------------------------------------------------


def test_default_inputs_produce_zero_pressure_context() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(),
        elapsed_ns=1_500,
    )
    assert ctx.perf == 0.0
    assert ctx.risk == 0.0
    assert ctx.drift == 0.0
    assert ctx.latency == 0.0
    assert ctx.vol_spike_z == 0.0
    assert ctx.elapsed_ns == 1_500


def test_halted_risk_snapshot_saturates_risk_pressure() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(halted=True),
        runtime_monitor=_monitor(),
        elapsed_ns=1,
    )
    assert ctx.risk == 1.0


def test_default_perf_is_fail_rate_plus_reject_rate_clamped() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(fail_rate=0.05, reject_rate=0.10),
        elapsed_ns=1,
    )
    assert ctx.perf == pytest.approx(0.15)


def test_default_perf_clamps_above_one() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(fail_rate=0.7, reject_rate=0.7),
        elapsed_ns=1,
    )
    assert ctx.perf == 1.0


def test_explicit_perf_pressure_overrides_default() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(fail_rate=0.5, reject_rate=0.5),
        perf_pressure=0.25,
        elapsed_ns=1,
    )
    assert ctx.perf == 0.25


# ---------------------------------------------------------------------------
# Latency normalisation
# ---------------------------------------------------------------------------


def test_latency_normalises_against_default_budget() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(p99_latency_ns=DEFAULT_LATENCY_BUDGET_NS // 2),
        elapsed_ns=1,
    )
    assert ctx.latency == pytest.approx(0.5)


def test_latency_saturates_at_budget() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(p99_latency_ns=DEFAULT_LATENCY_BUDGET_NS * 5),
        elapsed_ns=1,
    )
    assert ctx.latency == 1.0


def test_custom_latency_budget_is_honoured() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(p99_latency_ns=10_000_000),
        latency_budget_ns=20_000_000,
        elapsed_ns=1,
    )
    assert ctx.latency == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Drift / vol-spike / elapsed_ns
# ---------------------------------------------------------------------------


def test_drift_is_clamped_to_unit_interval() -> None:
    ctx_lo = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(),
        drift_deviation=-0.4,
        elapsed_ns=1,
    )
    ctx_hi = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(),
        drift_deviation=4.2,
        elapsed_ns=1,
    )
    assert ctx_lo.drift == 0.0
    assert ctx_hi.drift == 1.0


def test_vol_spike_z_passed_through_unchanged() -> None:
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(),
        vol_spike_z=-2.7,
        elapsed_ns=1,
    )
    assert ctx.vol_spike_z == -2.7


def test_elapsed_ns_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        build_runtime_context(
            risk_snapshot=_risk(),
            runtime_monitor=_monitor(),
            elapsed_ns=-1,
        )


def test_latency_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        build_runtime_context(
            risk_snapshot=_risk(),
            runtime_monitor=_monitor(),
            latency_budget_ns=0,
            elapsed_ns=1,
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_builder_is_deterministic_for_identical_inputs() -> None:
    inputs = dict(
        risk_snapshot=_risk(),
        runtime_monitor=_monitor(fail_rate=0.1, p99_latency_ns=10_000),
        drift_deviation=0.3,
        vol_spike_z=1.5,
        elapsed_ns=4_000,
    )
    a = build_runtime_context(**inputs)  # type: ignore[arg-type]
    b = build_runtime_context(**inputs)  # type: ignore[arg-type]
    assert a == b


def test_runtime_monitor_report_is_structurally_compatible() -> None:
    """RuntimeMonitorReport from execution_engine should slot in directly.

    The builder accepts ``RuntimeMonitorView``; the production
    ``RuntimeMonitorReport`` exposes the same three fields, so we verify
    structural compatibility without importing the execution engine.
    """
    from core.contracts.engine import HealthState
    from execution_engine.protections.runtime_monitor import (
        RuntimeMonitorReport,
        RuntimeMonitorState,
    )

    report = RuntimeMonitorReport(
        state=RuntimeMonitorState.OK,
        health=HealthState.OK,
        submitted=10,
        filled=8,
        rejected=1,
        failed=1,
        fill_rate=0.8,
        reject_rate=0.1,
        fail_rate=0.1,
        p50_latency_ns=1_000,
        p95_latency_ns=2_000,
        p99_latency_ns=50_000_000,
        queue_depth=0,
        detail="ok",
    )
    ctx = build_runtime_context(
        risk_snapshot=_risk(),
        runtime_monitor=report,  # type: ignore[arg-type]
        elapsed_ns=1,
    )
    assert ctx.perf == pytest.approx(0.2)
    assert ctx.latency == pytest.approx(0.5)
