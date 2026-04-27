"""Tests for Phase 4 health monitors (heartbeat / liveness / watchdog)."""

from __future__ import annotations

import pytest

from system_engine.health_monitors import (
    EngineLiveness,
    HeartbeatMonitor,
    LivenessChecker,
    LivenessState,
    Watchdog,
)

# ---------------------------------------------------------------------------
# HeartbeatMonitor
# ---------------------------------------------------------------------------


def test_heartbeat_record_and_last_seen():
    m = HeartbeatMonitor()
    m.record(engine="indira", ts_ns=100)
    assert m.last_seen("indira") == 100
    assert m.last_seen("dyon") is None


def test_heartbeat_last_seen_advances():
    m = HeartbeatMonitor()
    m.record(engine="indira", ts_ns=100)
    m.record(engine="indira", ts_ns=200)
    assert m.last_seen("indira") == 200


def test_heartbeat_rejects_empty_engine():
    m = HeartbeatMonitor()
    with pytest.raises(ValueError):
        m.record(engine="", ts_ns=100)


def test_heartbeat_rejects_non_monotonic_ts():
    m = HeartbeatMonitor()
    m.record(engine="indira", ts_ns=200)
    with pytest.raises(ValueError):
        m.record(engine="indira", ts_ns=100)


def test_heartbeat_snapshot_independent_of_internal_state():
    m = HeartbeatMonitor()
    m.record(engine="indira", ts_ns=100)
    snap = m.snapshot()
    snap["dyon"] = 999  # mutate copy
    assert m.last_seen("dyon") is None


def test_heartbeat_replay_determinism():
    def run() -> dict[str, int]:
        m = HeartbeatMonitor()
        m.record(engine="a", ts_ns=10)
        m.record(engine="b", ts_ns=20)
        m.record(engine="a", ts_ns=30)
        return m.snapshot()

    assert run() == run()


# ---------------------------------------------------------------------------
# LivenessChecker
# ---------------------------------------------------------------------------


def test_liveness_alive_within_threshold():
    c = LivenessChecker(suspect_after_ns=1_000, dead_after_ns=5_000)
    out = c.classify(ts_ns=500, heartbeats={"indira": 0})
    assert len(out) == 1
    assert out[0].state is LivenessState.ALIVE
    assert out[0].age_ns == 500


def test_liveness_suspect_band():
    c = LivenessChecker(suspect_after_ns=1_000, dead_after_ns=5_000)
    out = c.classify(ts_ns=2_000, heartbeats={"indira": 0})
    assert out[0].state is LivenessState.SUSPECT


def test_liveness_dead_band():
    c = LivenessChecker(suspect_after_ns=1_000, dead_after_ns=5_000)
    out = c.classify(ts_ns=10_000, heartbeats={"indira": 0})
    assert out[0].state is LivenessState.DEAD


def test_liveness_unknown_for_explicit_engine_without_heartbeat():
    c = LivenessChecker(suspect_after_ns=1_000, dead_after_ns=5_000)
    out = c.classify(
        ts_ns=100,
        heartbeats={},
        engines=("indira",),
    )
    assert out[0].state is LivenessState.UNKNOWN
    assert out[0].last_seen_ns is None
    assert out[0].age_ns is None


def test_liveness_classify_sorted_output():
    c = LivenessChecker(suspect_after_ns=1_000, dead_after_ns=5_000)
    out = c.classify(
        ts_ns=500,
        heartbeats={"zeta": 0, "alpha": 0, "mu": 0},
    )
    assert tuple(e.engine for e in out) == ("alpha", "mu", "zeta")


def test_liveness_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        LivenessChecker(suspect_after_ns=5_000, dead_after_ns=1_000)


def test_liveness_rejects_zero_threshold():
    with pytest.raises(ValueError):
        LivenessChecker(suspect_after_ns=0, dead_after_ns=1_000)


def test_liveness_replay_determinism():
    def run() -> tuple[EngineLiveness, ...]:
        c = LivenessChecker(suspect_after_ns=1_000, dead_after_ns=5_000)
        return c.classify(
            ts_ns=2_000,
            heartbeats={"indira": 0, "dyon": 1_500, "execution": 1_999},
        )

    assert run() == run()


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


def test_watchdog_no_bump_no_stall():
    w = Watchdog(timeout_ns=1_000)
    assert w.is_stalled(ts_ns=10_000) is False


def test_watchdog_within_budget_not_stalled():
    w = Watchdog(timeout_ns=1_000)
    w.bump(ts_ns=100)
    assert w.is_stalled(ts_ns=500) is False


def test_watchdog_stall_one_shot():
    w = Watchdog(timeout_ns=1_000)
    w.bump(ts_ns=0)
    assert w.is_stalled(ts_ns=2_000) is True
    assert w.is_stalled(ts_ns=3_000) is False  # already armed


def test_watchdog_rearms_after_bump():
    w = Watchdog(timeout_ns=1_000)
    w.bump(ts_ns=0)
    w.is_stalled(ts_ns=2_000)
    w.bump(ts_ns=2_500)
    assert w.is_stalled(ts_ns=4_000) is True


def test_watchdog_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        Watchdog(timeout_ns=0)


def test_watchdog_rejects_non_monotonic_bump():
    w = Watchdog(timeout_ns=1_000)
    w.bump(ts_ns=200)
    with pytest.raises(ValueError):
        w.bump(ts_ns=100)


def test_watchdog_replay_determinism():
    def run() -> tuple[bool, ...]:
        w = Watchdog(timeout_ns=1_000)
        w.bump(ts_ns=0)
        a = w.is_stalled(ts_ns=2_000)
        w.bump(ts_ns=2_500)
        b = w.is_stalled(ts_ns=4_000)
        return (a, b)

    assert run() == run()
