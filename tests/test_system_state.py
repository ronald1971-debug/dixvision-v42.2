"""Tests for SystemState + AnomalyDetector + DriftMonitor (Phase 4)."""

from __future__ import annotations

import pytest

from core.contracts.events import HazardEvent, HazardSeverity
from system_engine.state import (
    AnomalyDetector,
    AnomalyVerdict,
    DriftMonitor,
    DriftReading,
    SystemState,
    SystemStateSnapshot,
)


def _hazard(code: str, ts_ns: int = 1) -> HazardEvent:
    return HazardEvent(
        ts_ns=ts_ns,
        code=code,
        severity=HazardSeverity.HIGH,
        source="test",
    )


# ---------------------------------------------------------------------------
# SystemState
# ---------------------------------------------------------------------------


def test_system_state_empty_snapshot():
    s = SystemState()
    snap = s.snapshot(ts_ns=100)
    assert isinstance(snap, SystemStateSnapshot)
    assert snap.ts_ns == 100
    assert snap.heartbeats == ()
    assert snap.open_hazards == ()
    assert snap.hazard_count == 0


def test_system_state_record_heartbeats_sorted():
    s = SystemState()
    s.record_heartbeat(engine="zeta", ts_ns=50)
    s.record_heartbeat(engine="alpha", ts_ns=10)
    snap = s.snapshot(ts_ns=100)
    assert snap.heartbeats == (("alpha", 10), ("zeta", 50))


def test_system_state_record_hazard_increments_count():
    s = SystemState()
    s.record_hazard(_hazard("HAZ-01"))
    s.record_hazard(_hazard("HAZ-02"))
    snap = s.snapshot(ts_ns=100)
    assert snap.hazard_count == 2
    assert tuple(h.code for h in snap.open_hazards) == ("HAZ-01", "HAZ-02")


def test_system_state_latest_wins_per_code():
    s = SystemState()
    s.record_hazard(_hazard("HAZ-01", ts_ns=1))
    s.record_hazard(_hazard("HAZ-01", ts_ns=2))
    snap = s.snapshot(ts_ns=100)
    assert len(snap.open_hazards) == 1
    assert snap.open_hazards[0].ts_ns == 2
    # but counter is monotonic
    assert snap.hazard_count == 2


def test_system_state_clear_hazard():
    s = SystemState()
    s.record_hazard(_hazard("HAZ-01"))
    s.clear_hazard("HAZ-01")
    snap = s.snapshot(ts_ns=100)
    assert snap.open_hazards == ()
    assert snap.hazard_count == 1  # counter is monotonic; clear does not reset


def test_system_state_clear_unknown_code_noop():
    s = SystemState()
    s.clear_hazard("HAZ-99")  # must not raise


def test_system_state_snapshot_is_immutable():
    s = SystemState()
    s.record_heartbeat(engine="indira", ts_ns=10)
    snap = s.snapshot(ts_ns=100)
    with pytest.raises((AttributeError, TypeError)):
        snap.ts_ns = 999  # type: ignore[misc]


def test_system_state_replay_determinism():
    def run() -> SystemStateSnapshot:
        s = SystemState()
        s.record_heartbeat(engine="indira", ts_ns=10)
        s.record_hazard(_hazard("HAZ-01"))
        s.record_heartbeat(engine="dyon", ts_ns=20)
        s.record_hazard(_hazard("HAZ-02"))
        return s.snapshot(ts_ns=100)

    assert run() == run()


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------


def test_anomaly_below_min_samples_never_anomaly():
    d = AnomalyDetector(metric="pnl", window=8, z_threshold=2.0, min_samples=4)
    for v in (1.0, 2.0, 3.0):
        verdict = d.observe(ts_ns=1, value=v)
        assert verdict.is_anomaly is False
        assert verdict.z_score == 0.0


def test_anomaly_normal_sample_classified_normal():
    d = AnomalyDetector(metric="pnl", window=8, z_threshold=3.0, min_samples=4)
    for v in (1.0, 1.0, 1.0, 1.0):
        d.observe(ts_ns=1, value=v)
    verdict = d.observe(ts_ns=2, value=1.05)
    assert verdict.is_anomaly is False


def test_anomaly_outlier_classified_anomalous():
    d = AnomalyDetector(metric="pnl", window=8, z_threshold=3.0, min_samples=4)
    for v in (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0):
        d.observe(ts_ns=1, value=v)
    # population looks ~constant — but introduce a tiny non-zero variance first
    d.observe(ts_ns=2, value=1.001)
    verdict = d.observe(ts_ns=3, value=100.0)
    assert verdict.is_anomaly is True
    assert verdict.z_score > 3.0


def test_anomaly_zero_variance_no_div_by_zero():
    d = AnomalyDetector(metric="x", window=8, z_threshold=3.0, min_samples=4)
    for _ in range(5):
        d.observe(ts_ns=1, value=42.0)
    verdict = d.observe(ts_ns=2, value=42.0)
    assert verdict.z_score == 0.0
    assert verdict.is_anomaly is False


def test_anomaly_returns_typed_verdict():
    d = AnomalyDetector(metric="lat", window=8, z_threshold=3.0, min_samples=4)
    v = d.observe(ts_ns=10, value=5.0)
    assert isinstance(v, AnomalyVerdict)
    assert v.metric == "lat"
    assert v.ts_ns == 10
    assert v.value == 5.0


def test_anomaly_validates_constructor_args():
    with pytest.raises(ValueError):
        AnomalyDetector(metric="", window=8, z_threshold=3.0, min_samples=4)
    with pytest.raises(ValueError):
        AnomalyDetector(metric="x", window=2, z_threshold=3.0, min_samples=4)
    with pytest.raises(ValueError):
        AnomalyDetector(metric="x", window=8, z_threshold=0.0, min_samples=4)
    with pytest.raises(ValueError):
        AnomalyDetector(metric="x", window=8, z_threshold=3.0, min_samples=2)
    with pytest.raises(ValueError):
        AnomalyDetector(metric="x", window=8, z_threshold=3.0, min_samples=10)


def test_anomaly_replay_determinism():
    def run() -> tuple[AnomalyVerdict, ...]:
        d = AnomalyDetector(metric="x", window=8, z_threshold=3.0, min_samples=4)
        out: list[AnomalyVerdict] = []
        for i, v in enumerate((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0)):
            out.append(d.observe(ts_ns=i, value=v))
        return tuple(out)

    assert run() == run()


def test_anomaly_sample_count_caps_at_window():
    """Regression: PR #32 review BUG_0003.

    Once the deque reaches ``maxlen``, ``sample_count`` must cap at
    ``window`` (the buffer's true size after eviction) instead of
    monotonically reporting ``n + 1``.
    """
    d = AnomalyDetector(metric="x", window=8, z_threshold=3.0, min_samples=4)
    last: AnomalyVerdict | None = None
    for i in range(20):
        last = d.observe(ts_ns=i, value=1.0 + (i * 0.001))
    assert last is not None
    assert last.sample_count == 8


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------


def test_drift_first_sample_seeds_ewma():
    d = DriftMonitor(metric="pnl", alpha=0.1, drift_threshold=0.5)
    r = d.observe(ts_ns=1, value=5.0)
    assert isinstance(r, DriftReading)
    assert r.ewma == 5.0
    assert r.deviation == 0.0
    assert r.is_drifting is False


def test_drift_stable_series_no_drift():
    d = DriftMonitor(metric="pnl", alpha=0.5, drift_threshold=0.5)
    d.observe(ts_ns=1, value=5.0)
    r = d.observe(ts_ns=2, value=5.05)
    assert r.is_drifting is False


def test_drift_large_jump_triggers_drift_flag():
    d = DriftMonitor(metric="pnl", alpha=0.5, drift_threshold=0.25)
    d.observe(ts_ns=1, value=5.0)
    r = d.observe(ts_ns=2, value=10.0)  # 100% deviation
    assert r.is_drifting is True
    assert r.deviation > 0.25


def test_drift_zero_prev_uses_unit_denominator():
    d = DriftMonitor(metric="x", alpha=0.5, drift_threshold=0.1)
    d.observe(ts_ns=1, value=0.0)
    r = d.observe(ts_ns=2, value=0.5)
    # denom collapses to 1.0 → deviation == 0.5
    assert r.deviation == 0.5
    assert r.is_drifting is True


def test_drift_validates_constructor_args():
    with pytest.raises(ValueError):
        DriftMonitor(metric="", alpha=0.1, drift_threshold=0.1)
    with pytest.raises(ValueError):
        DriftMonitor(metric="x", alpha=0.0, drift_threshold=0.1)
    with pytest.raises(ValueError):
        DriftMonitor(metric="x", alpha=1.5, drift_threshold=0.1)
    with pytest.raises(ValueError):
        DriftMonitor(metric="x", alpha=0.1, drift_threshold=0.0)


def test_drift_replay_determinism():
    def run() -> tuple[DriftReading, ...]:
        d = DriftMonitor(metric="x", alpha=0.3, drift_threshold=0.25)
        out: list[DriftReading] = []
        for i, v in enumerate((5.0, 5.5, 6.0, 12.0, 12.5)):
            out.append(d.observe(ts_ns=i, value=v))
        return tuple(out)

    assert run() == run()
