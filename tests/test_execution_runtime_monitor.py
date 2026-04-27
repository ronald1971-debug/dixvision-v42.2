"""Phase 2 — execution runtime monitor unit tests."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState
from core.contracts.events import ExecutionEvent, ExecutionStatus, Side
from execution_engine.protections import (
    RuntimeMonitor,
    RuntimeMonitorState,
)


def _ev(status: ExecutionStatus, ts: int = 1) -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=ts,
        symbol="BTC-USD",
        side=Side.BUY,
        qty=1.0,
        price=100.0,
        status=status,
        venue="paper",
        order_id=f"o-{ts}",
    )


def test_monitor_starts_ok_when_empty():
    mon = RuntimeMonitor()
    rep = mon.report()
    assert rep.state is RuntimeMonitorState.OK
    assert rep.health is HealthState.OK
    assert rep.submitted == 0


def test_monitor_records_filled_increments_filled_only():
    mon = RuntimeMonitor()
    mon.record(_ev(ExecutionStatus.FILLED), latency_ns=50)
    rep = mon.report()
    assert rep.submitted == 1
    assert rep.filled == 1
    assert rep.rejected == 0
    assert rep.failed == 0


def test_monitor_high_reject_rate_marks_fail():
    mon = RuntimeMonitor(fail_reject_rate=0.30)
    for _ in range(7):
        mon.record(_ev(ExecutionStatus.REJECTED))
    for _ in range(3):
        mon.record(_ev(ExecutionStatus.FILLED))
    rep = mon.report()
    assert rep.state is RuntimeMonitorState.FAIL
    assert rep.health is HealthState.FAIL


def test_monitor_moderate_reject_rate_marks_degraded():
    mon = RuntimeMonitor(
        degraded_reject_rate=0.10,
        fail_reject_rate=0.50,
    )
    for _ in range(2):
        mon.record(_ev(ExecutionStatus.REJECTED))
    for _ in range(8):
        mon.record(_ev(ExecutionStatus.FILLED))
    rep = mon.report()
    assert rep.state is RuntimeMonitorState.DEGRADED
    assert rep.health is HealthState.DEGRADED


def test_monitor_high_fail_rate_marks_fail():
    mon = RuntimeMonitor(fail_fail_rate=0.10)
    for _ in range(3):
        mon.record(_ev(ExecutionStatus.FAILED))
    for _ in range(7):
        mon.record(_ev(ExecutionStatus.FILLED))
    rep = mon.report()
    assert rep.state is RuntimeMonitorState.FAIL


def test_monitor_queue_depth_threshold_marks_degraded():
    mon = RuntimeMonitor(max_queue_depth=10)
    mon.set_queue_depth(20)
    mon.record(_ev(ExecutionStatus.FILLED))
    rep = mon.report()
    assert rep.queue_depth == 20
    assert rep.state is RuntimeMonitorState.DEGRADED


def test_monitor_p95_latency_is_nonzero_with_samples():
    mon = RuntimeMonitor()
    for i in range(1, 101):
        mon.record(_ev(ExecutionStatus.FILLED, ts=i), latency_ns=i)
    rep = mon.report()
    assert rep.p50_latency_ns > 0
    assert rep.p95_latency_ns >= rep.p50_latency_ns
    assert rep.p99_latency_ns >= rep.p95_latency_ns


def test_monitor_record_batch_aggregates():
    mon = RuntimeMonitor()
    mon.record_batch(
        [
            _ev(ExecutionStatus.FILLED),
            _ev(ExecutionStatus.REJECTED),
            _ev(ExecutionStatus.FAILED),
        ]
    )
    rep = mon.report()
    assert rep.submitted == 3
    assert rep.filled == 1
    assert rep.rejected == 1
    assert rep.failed == 1


def test_monitor_invalid_latency_rejected():
    mon = RuntimeMonitor()
    with pytest.raises(ValueError):
        mon.record(_ev(ExecutionStatus.FILLED), latency_ns=-1)


def test_monitor_invalid_queue_depth_rejected():
    mon = RuntimeMonitor()
    with pytest.raises(ValueError):
        mon.set_queue_depth(-1)


def test_monitor_invalid_window_rejected():
    with pytest.raises(ValueError):
        RuntimeMonitor(window=0)
    with pytest.raises(ValueError):
        RuntimeMonitor(latency_window=0)


def test_monitor_rates_are_windowed_not_cumulative():
    """Old failures must fall off once they leave the rolling window."""
    mon = RuntimeMonitor(window=10, fail_reject_rate=0.30)

    # Burst of rejects fills the window → FAIL.
    for _ in range(10):
        mon.record(_ev(ExecutionStatus.REJECTED))
    assert mon.report().state is RuntimeMonitorState.FAIL

    # Healthy fills push the rejects out of the window → recovery.
    for _ in range(10):
        mon.record(_ev(ExecutionStatus.FILLED))
    rep = mon.report()
    assert rep.state is RuntimeMonitorState.OK
    assert rep.reject_rate == 0.0
    # Lifetime totals still reflect everything we ever saw.
    assert rep.submitted == 20
    assert rep.rejected == 10
    assert rep.filled == 10


def test_monitor_replay_determinism_same_inputs_same_report():
    def run() -> tuple:
        mon = RuntimeMonitor()
        for i in range(50):
            status = (
                ExecutionStatus.FILLED if i % 2 == 0 else ExecutionStatus.REJECTED
            )
            mon.record(_ev(status, ts=i), latency_ns=10 * (i + 1))
        rep = mon.report()
        return (
            rep.state,
            rep.submitted,
            rep.filled,
            rep.rejected,
            rep.p50_latency_ns,
            rep.p95_latency_ns,
            rep.p99_latency_ns,
        )

    assert run() == run()
