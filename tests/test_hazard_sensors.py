"""Tests for Phase 4 hazard sensors (HAZ-01..12) + SensorArray.

Pure-Python, deterministic — no clocks, no IO. Each sensor is exercised
with at least two real cases plus edge cases plus an INV-15 replay
determinism check.
"""

from __future__ import annotations

import pytest

from core.contracts.events import HazardSeverity
from core.contracts.market import MarketTick
from system_engine.hazard_sensors import (
    ClockDriftSensor,
    ExchangeUnreachableSensor,
    HeartbeatMissedSensor,
    LatencySpikeSensor,
    MarketAnomalySensor,
    MemoryOverflowSensor,
    OrderFloodSensor,
    RiskSnapshotStaleSensor,
    RuntimeBreakerOpenSensor,
    SensorArray,
    StaleDataSensor,
    SystemAnomalySensor,
    WSTimeoutSensor,
)

# ---------------------------------------------------------------------------
# SensorArray
# ---------------------------------------------------------------------------


def test_sensor_array_register_and_iterate():
    arr = SensorArray()
    s1 = WSTimeoutSensor(timeout_ns=1_000)
    s2 = ClockDriftSensor(tolerance_ns=1_000)
    arr.register(s1)
    arr.register(s2)
    assert len(arr) == 2
    assert arr.sensors == (s1, s2)


def test_sensor_array_rejects_duplicate_name():
    arr = SensorArray()
    arr.register(WSTimeoutSensor(timeout_ns=1_000))
    with pytest.raises(ValueError):
        arr.register(WSTimeoutSensor(timeout_ns=1_000))


def test_sensor_array_deregister():
    arr = SensorArray()
    s1 = WSTimeoutSensor(timeout_ns=1_000)
    arr.register(s1)
    arr.deregister("ws_timeout")
    assert len(arr) == 0


def test_sensor_array_rejects_empty_name():
    arr = SensorArray()

    class _Anon:
        name = ""
        code = "X"

    with pytest.raises(ValueError):
        arr.register(_Anon())  # type: ignore[arg-type]


def test_sensor_array_collect_preserves_order():
    arr = SensorArray()
    arr.register(WSTimeoutSensor(timeout_ns=10))
    sensor = arr.sensors[0]
    sensor.on_tick(0)
    a = sensor.observe(100)
    sensor2 = ClockDriftSensor(tolerance_ns=1)
    b = sensor2.observe(ts_ns=200, reference_ns=0, sample_ns=10_000)
    out = arr.collect([a, b])
    assert out == a + b
    assert tuple(e.code for e in out) == ("HAZ-01", "HAZ-03")


# ---------------------------------------------------------------------------
# HAZ-01 — WSTimeoutSensor
# ---------------------------------------------------------------------------


def test_ws_timeout_no_tick_no_event():
    s = WSTimeoutSensor(timeout_ns=1_000)
    assert s.observe(10_000) == ()


def test_ws_timeout_within_tolerance_silent():
    s = WSTimeoutSensor(timeout_ns=1_000)
    s.on_tick(0)
    assert s.observe(500) == ()


def test_ws_timeout_emits_once_per_episode():
    s = WSTimeoutSensor(timeout_ns=1_000)
    s.on_tick(0)
    out1 = s.observe(2_000)
    out2 = s.observe(3_000)
    assert len(out1) == 1
    assert out1[0].code == "HAZ-01"
    assert out1[0].severity is HazardSeverity.HIGH
    assert out2 == ()


def test_ws_timeout_rearms_after_next_tick():
    s = WSTimeoutSensor(timeout_ns=1_000)
    s.on_tick(0)
    out1 = s.observe(2_000)
    assert len(out1) == 1
    s.on_tick(3_000)  # fresh tick disarms
    assert s.observe(3_500) == ()  # 500ns gap < 1000ns timeout
    out2 = s.observe(5_000)  # 2000ns gap > 1000ns → second episode
    assert len(out2) == 1


def test_ws_timeout_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        WSTimeoutSensor(timeout_ns=0)


def test_ws_timeout_replay_determinism():
    def run() -> tuple:
        s = WSTimeoutSensor(timeout_ns=1_000)
        s.on_tick(0)
        a = s.observe(2_000)
        s.on_tick(2_500)
        b = s.observe(4_000)
        return tuple((e.ts_ns, e.code, e.severity) for e in (*a, *b))

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-02 — ExchangeUnreachableSensor
# ---------------------------------------------------------------------------


def test_exchange_unreachable_clean_state():
    s = ExchangeUnreachableSensor(threshold=2)
    s.record_attempt("BINANCE", ok=True)
    assert s.observe(10) == ()


def test_exchange_unreachable_threshold_breach():
    s = ExchangeUnreachableSensor(threshold=3)
    s.record_attempt("BINANCE", ok=False)
    s.record_attempt("BINANCE", ok=False)
    assert s.observe(10) == ()  # below threshold
    s.record_attempt("BINANCE", ok=False)
    out = s.observe(20)
    assert len(out) == 1
    assert out[0].meta["venue"] == "BINANCE"


def test_exchange_unreachable_resets_on_success():
    s = ExchangeUnreachableSensor(threshold=2)
    s.record_attempt("V", ok=False)
    s.record_attempt("V", ok=False)
    s.observe(10)
    s.record_attempt("V", ok=True)
    s.record_attempt("V", ok=False)
    s.record_attempt("V", ok=False)
    out = s.observe(20)
    assert len(out) == 1


def test_exchange_unreachable_one_shot_per_episode():
    s = ExchangeUnreachableSensor(threshold=1)
    s.record_attempt("V", ok=False)
    out1 = s.observe(10)
    out2 = s.observe(20)
    assert len(out1) == 1
    assert out2 == ()


def test_exchange_unreachable_rejects_zero_threshold():
    with pytest.raises(ValueError):
        ExchangeUnreachableSensor(threshold=0)


def test_exchange_unreachable_replay_determinism():
    def run() -> tuple:
        s = ExchangeUnreachableSensor(threshold=2)
        s.record_attempt("A", ok=False)
        s.record_attempt("B", ok=False)
        s.record_attempt("A", ok=False)
        out = s.observe(100)
        return tuple((e.code, e.meta["venue"]) for e in out)

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-03 — ClockDriftSensor
# ---------------------------------------------------------------------------


def test_clock_drift_within_tolerance_silent():
    s = ClockDriftSensor(tolerance_ns=100)
    assert s.observe(ts_ns=1, reference_ns=1_000, sample_ns=1_050) == ()


def test_clock_drift_emits_above_tolerance():
    s = ClockDriftSensor(tolerance_ns=100)
    # drift=200 (> 100, <= 400) → HIGH severity
    out = s.observe(ts_ns=1, reference_ns=1_000, sample_ns=1_200)
    assert len(out) == 1
    assert out[0].severity is HazardSeverity.HIGH


def test_clock_drift_critical_when_4x():
    s = ClockDriftSensor(tolerance_ns=100)
    out = s.observe(ts_ns=1, reference_ns=0, sample_ns=500)
    assert out[0].severity is HazardSeverity.CRITICAL


def test_clock_drift_one_shot_then_disarms_when_back():
    s = ClockDriftSensor(tolerance_ns=100)
    s.observe(ts_ns=1, reference_ns=0, sample_ns=500)
    out2 = s.observe(ts_ns=2, reference_ns=0, sample_ns=600)
    assert out2 == ()
    s.observe(ts_ns=3, reference_ns=0, sample_ns=10)  # back to in-tolerance
    out3 = s.observe(ts_ns=4, reference_ns=0, sample_ns=600)
    assert len(out3) == 1


def test_clock_drift_rejects_invalid_tolerance():
    with pytest.raises(ValueError):
        ClockDriftSensor(tolerance_ns=0)


def test_clock_drift_replay_determinism():
    def run() -> tuple:
        s = ClockDriftSensor(tolerance_ns=100)
        a = s.observe(ts_ns=1, reference_ns=0, sample_ns=500)
        b = s.observe(ts_ns=2, reference_ns=0, sample_ns=550)
        return tuple((e.ts_ns, e.code, e.severity) for e in (*a, *b))

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-04 — StaleDataSensor
# ---------------------------------------------------------------------------


def test_stale_data_no_symbol_no_event():
    s = StaleDataSensor(max_gap_ns=1_000)
    assert s.observe(10_000) == ()


def test_stale_data_emits_per_symbol():
    s = StaleDataSensor(max_gap_ns=1_000)
    s.on_tick(symbol="BTCUSD", ts_ns=0)
    s.on_tick(symbol="ETHUSD", ts_ns=0)
    out = s.observe(2_000)
    assert len(out) == 2
    syms = {e.meta["symbol"] for e in out}
    assert syms == {"BTCUSD", "ETHUSD"}


def test_stale_data_one_shot_per_symbol():
    s = StaleDataSensor(max_gap_ns=1_000)
    s.on_tick(symbol="X", ts_ns=0)
    out1 = s.observe(2_000)
    out2 = s.observe(3_000)
    assert len(out1) == 1
    assert out2 == ()


def test_stale_data_rearms_after_fresh_tick():
    s = StaleDataSensor(max_gap_ns=1_000)
    s.on_tick(symbol="X", ts_ns=0)
    s.observe(2_000)
    s.on_tick(symbol="X", ts_ns=2_500)
    out = s.observe(4_000)
    assert len(out) == 1


def test_stale_data_replay_determinism():
    def run() -> tuple:
        s = StaleDataSensor(max_gap_ns=1_000)
        s.on_tick(symbol="A", ts_ns=0)
        s.on_tick(symbol="B", ts_ns=10)
        out = s.observe(5_000)
        return tuple(sorted(e.meta["symbol"] for e in out))

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-05 — MemoryOverflowSensor
# ---------------------------------------------------------------------------


def test_memory_overflow_below_warn_silent():
    s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
    assert s.observe(ts_ns=1, rss_bytes=500) == ()


def test_memory_overflow_warn_band():
    s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
    out = s.observe(ts_ns=1, rss_bytes=1_500)
    assert len(out) == 1
    assert out[0].severity is HazardSeverity.MEDIUM


def test_memory_overflow_critical_band():
    s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
    out = s.observe(ts_ns=1, rss_bytes=3_000)
    assert out[0].severity is HazardSeverity.CRITICAL


def test_memory_overflow_one_shot_critical():
    s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
    s.observe(ts_ns=1, rss_bytes=3_000)
    assert s.observe(ts_ns=2, rss_bytes=3_500) == ()


def test_memory_overflow_disarms_below_warn():
    s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
    s.observe(ts_ns=1, rss_bytes=3_000)
    s.observe(ts_ns=2, rss_bytes=500)  # disarm
    out = s.observe(ts_ns=3, rss_bytes=3_000)
    assert len(out) == 1


def test_memory_overflow_rejects_critical_below_warn():
    with pytest.raises(ValueError):
        MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=500)


def test_memory_overflow_replay_determinism():
    def run() -> tuple:
        s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
        a = s.observe(ts_ns=1, rss_bytes=1_500)
        b = s.observe(ts_ns=2, rss_bytes=3_000)
        return tuple((e.ts_ns, e.severity) for e in (*a, *b))

    assert run() == run()


def test_memory_overflow_critical_warn_critical_re_emits():
    """Regression: PR #32 review BUG_0002.

    Critical → warn → critical must rearm the critical band on the warn
    transit. Previously ``_armed_critical`` stayed True and the second
    CRITICAL was silently swallowed.
    """
    s = MemoryOverflowSensor(warn_bytes=1_000, critical_bytes=2_000)
    first = s.observe(ts_ns=1, rss_bytes=3_000)
    assert len(first) == 1
    assert first[0].severity is HazardSeverity.CRITICAL

    warn = s.observe(ts_ns=2, rss_bytes=1_500)
    assert len(warn) == 1
    assert warn[0].severity is HazardSeverity.MEDIUM

    second = s.observe(ts_ns=3, rss_bytes=3_000)
    assert len(second) == 1
    assert second[0].severity is HazardSeverity.CRITICAL


# ---------------------------------------------------------------------------
# HAZ-06 — LatencySpikeSensor
# ---------------------------------------------------------------------------


def test_latency_spike_window_not_full_silent():
    s = LatencySpikeSensor(budget_ns=10, window=4, breach_quota=2)
    s.record_sample(100)
    assert s.observe(1) == ()


def test_latency_spike_emits_when_quota_breached():
    s = LatencySpikeSensor(budget_ns=10, window=4, breach_quota=2)
    for v in (5, 100, 5, 100):
        s.record_sample(v)
    out = s.observe(1)
    assert len(out) == 1


def test_latency_spike_silent_when_below_quota():
    s = LatencySpikeSensor(budget_ns=10, window=4, breach_quota=3)
    for v in (5, 100, 5, 100):
        s.record_sample(v)
    assert s.observe(1) == ()


def test_latency_spike_one_shot_then_disarms():
    s = LatencySpikeSensor(budget_ns=10, window=4, breach_quota=2)
    for v in (100, 100, 100, 100):
        s.record_sample(v)
    out1 = s.observe(1)
    out2 = s.observe(2)
    assert len(out1) == 1
    assert out2 == ()
    # drop window below quota
    for v in (5, 5, 5, 5):
        s.record_sample(v)
    s.observe(3)
    for v in (100, 100, 100, 100):
        s.record_sample(v)
    out3 = s.observe(4)
    assert len(out3) == 1


def test_latency_spike_validates_quota():
    with pytest.raises(ValueError):
        LatencySpikeSensor(budget_ns=10, window=4, breach_quota=5)


def test_latency_spike_replay_determinism():
    def run() -> tuple:
        s = LatencySpikeSensor(budget_ns=10, window=4, breach_quota=2)
        for v in (5, 100, 5, 100):
            s.record_sample(v)
        out = s.observe(1)
        return tuple((e.ts_ns, e.code) for e in out)

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-07 — HeartbeatMissedSensor
# ---------------------------------------------------------------------------


def test_heartbeat_missed_no_engine_no_event():
    s = HeartbeatMissedSensor(timeout_ns=1_000)
    assert s.observe(10_000) == ()


def test_heartbeat_missed_emits_per_engine():
    s = HeartbeatMissedSensor(timeout_ns=1_000)
    s.on_heartbeat(engine="indira", ts_ns=0)
    s.on_heartbeat(engine="dyon", ts_ns=0)
    out = s.observe(2_000)
    assert len(out) == 2


def test_heartbeat_missed_rearms_after_new_beat():
    s = HeartbeatMissedSensor(timeout_ns=1_000)
    s.on_heartbeat(engine="dyon", ts_ns=0)
    s.observe(2_000)
    s.on_heartbeat(engine="dyon", ts_ns=2_500)
    out = s.observe(4_000)
    assert len(out) == 1


def test_heartbeat_missed_replay_determinism():
    def run() -> tuple:
        s = HeartbeatMissedSensor(timeout_ns=1_000)
        s.on_heartbeat(engine="a", ts_ns=0)
        s.on_heartbeat(engine="b", ts_ns=0)
        out = s.observe(5_000)
        return tuple(sorted(e.meta["engine"] for e in out))

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-08 — RiskSnapshotStaleSensor
# ---------------------------------------------------------------------------


def test_risk_snapshot_first_sample_silent():
    s = RiskSnapshotStaleSensor(max_age_ns=1_000)
    assert s.observe(ts_ns=10, version_id=1) == ()


def test_risk_snapshot_stale_emits_when_frozen():
    s = RiskSnapshotStaleSensor(max_age_ns=1_000)
    s.observe(ts_ns=0, version_id=1)
    out = s.observe(ts_ns=2_000, version_id=1)
    assert len(out) == 1


def test_risk_snapshot_resets_on_version_advance():
    s = RiskSnapshotStaleSensor(max_age_ns=1_000)
    s.observe(ts_ns=0, version_id=1)
    s.observe(ts_ns=2_000, version_id=1)  # emits
    s.observe(ts_ns=2_500, version_id=2)  # advance
    out = s.observe(ts_ns=3_000, version_id=2)  # 500ns < 1000, silent
    assert out == ()


def test_risk_snapshot_one_shot_per_episode():
    s = RiskSnapshotStaleSensor(max_age_ns=1_000)
    s.observe(ts_ns=0, version_id=1)
    out1 = s.observe(ts_ns=2_000, version_id=1)
    out2 = s.observe(ts_ns=3_000, version_id=1)
    assert len(out1) == 1
    assert out2 == ()


def test_risk_snapshot_replay_determinism():
    def run() -> tuple:
        s = RiskSnapshotStaleSensor(max_age_ns=1_000)
        s.observe(ts_ns=0, version_id=1)
        out = s.observe(ts_ns=2_000, version_id=1)
        return tuple((e.ts_ns, e.code) for e in out)

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-09 — OrderFloodSensor
# ---------------------------------------------------------------------------


def test_order_flood_under_cap_silent():
    s = OrderFloodSensor(window_ns=1_000, max_orders=5)
    for ts in range(3):
        s.record_order(ts)
    assert s.observe(10) == ()


def test_order_flood_over_cap_emits():
    s = OrderFloodSensor(window_ns=1_000, max_orders=2)
    for ts in (0, 1, 2, 3):
        s.record_order(ts)
    out = s.observe(4)
    assert len(out) == 1
    assert int(out[0].meta["orders"]) > 2


def test_order_flood_evicts_old_orders():
    s = OrderFloodSensor(window_ns=100, max_orders=2)
    for ts in (0, 10, 20):
        s.record_order(ts)
    # at ts=200, the cutoff is 100 → all evicted
    assert s.observe(200) == ()


def test_order_flood_one_shot():
    s = OrderFloodSensor(window_ns=1_000, max_orders=1)
    s.record_order(0)
    s.record_order(1)
    s.record_order(2)
    out1 = s.observe(3)
    out2 = s.observe(4)
    assert len(out1) == 1
    assert out2 == ()


def test_order_flood_rejects_invalid():
    with pytest.raises(ValueError):
        OrderFloodSensor(window_ns=0, max_orders=1)
    with pytest.raises(ValueError):
        OrderFloodSensor(window_ns=10, max_orders=0)


def test_order_flood_replay_determinism():
    def run() -> tuple:
        s = OrderFloodSensor(window_ns=1_000, max_orders=2)
        for ts in (0, 1, 2, 3):
            s.record_order(ts)
        out = s.observe(4)
        return tuple((e.ts_ns, e.meta["orders"]) for e in out)

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-10 — RuntimeBreakerOpenSensor
# ---------------------------------------------------------------------------


def test_runtime_breaker_emits_on_open():
    s = RuntimeBreakerOpenSensor()
    out = s.report_open(scope="MEMECOIN", ts_ns=10)
    assert len(out) == 1
    assert out[0].severity is HazardSeverity.CRITICAL
    assert out[0].meta["scope"] == "MEMECOIN"


def test_runtime_breaker_idempotent_for_same_scope():
    s = RuntimeBreakerOpenSensor()
    s.report_open(scope="MEMECOIN", ts_ns=10)
    out = s.report_open(scope="MEMECOIN", ts_ns=20)
    assert out == ()


def test_runtime_breaker_separate_scopes_emit_separately():
    s = RuntimeBreakerOpenSensor()
    a = s.report_open(scope="MEMECOIN", ts_ns=10)
    b = s.report_open(scope="NORMAL", ts_ns=20)
    assert len(a) == 1
    assert len(b) == 1


def test_runtime_breaker_close_then_reopen():
    s = RuntimeBreakerOpenSensor()
    s.report_open(scope="X", ts_ns=10)
    s.report_closed(scope="X")
    out = s.report_open(scope="X", ts_ns=20)
    assert len(out) == 1


def test_runtime_breaker_observe_is_noop():
    s = RuntimeBreakerOpenSensor()
    s.report_open(scope="X", ts_ns=10)
    assert s.observe(100) == ()


def test_runtime_breaker_replay_determinism():
    def run() -> tuple:
        s = RuntimeBreakerOpenSensor()
        a = s.report_open(scope="A", ts_ns=10)
        b = s.report_open(scope="B", ts_ns=20)
        return tuple((e.meta["scope"], e.ts_ns) for e in (*a, *b))

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-11 — MarketAnomalySensor
# ---------------------------------------------------------------------------


def _tick(symbol: str, ts_ns: int, bid: float, ask: float, last: float) -> MarketTick:
    return MarketTick(ts_ns=ts_ns, symbol=symbol, bid=bid, ask=ask, last=last)


def test_market_anomaly_normal_tick_silent():
    s = MarketAnomalySensor(max_spread_bps=50.0, max_jump_bps=200.0)
    assert s.on_tick(_tick("BTC", 1, 100.0, 100.05, 100.02)) == ()


def test_market_anomaly_spread_blowout():
    s = MarketAnomalySensor(max_spread_bps=10.0, max_jump_bps=200.0)
    out = s.on_tick(_tick("BTC", 1, 100.0, 110.0, 105.0))
    assert any(e.meta["kind"] == "spread" for e in out)


def test_market_anomaly_price_jump():
    s = MarketAnomalySensor(max_spread_bps=10_000.0, max_jump_bps=100.0)
    s.on_tick(_tick("BTC", 1, 100.0, 100.01, 100.005))
    out = s.on_tick(_tick("BTC", 2, 105.0, 105.01, 105.005))
    assert any(e.meta["kind"] == "jump" for e in out)


def test_market_anomaly_per_symbol_state():
    s = MarketAnomalySensor(max_spread_bps=10_000.0, max_jump_bps=100.0)
    s.on_tick(_tick("BTC", 1, 100.0, 100.01, 100.005))
    s.on_tick(_tick("ETH", 1, 50.0, 50.01, 50.005))
    # BTC jump
    out_btc = s.on_tick(_tick("BTC", 2, 105.0, 105.01, 105.005))
    # ETH stable
    out_eth = s.on_tick(_tick("ETH", 2, 50.0, 50.01, 50.005))
    assert any(e.meta["kind"] == "jump" for e in out_btc)
    assert out_eth == ()


def test_market_anomaly_replay_determinism():
    def run() -> tuple:
        s = MarketAnomalySensor(max_spread_bps=10.0, max_jump_bps=100.0)
        a = s.on_tick(_tick("BTC", 1, 100.0, 110.0, 105.0))
        return tuple((e.ts_ns, e.meta["kind"]) for e in a)

    assert run() == run()


# ---------------------------------------------------------------------------
# HAZ-12 — SystemAnomalySensor
# ---------------------------------------------------------------------------


def test_system_anomaly_clean_silent():
    s = SystemAnomalySensor(max_cpu_pct=90.0, max_open_fds=4096)
    assert s.observe(ts_ns=1, cpu_pct=10.0, open_fds=100) == ()


def test_system_anomaly_cpu_breach():
    s = SystemAnomalySensor(max_cpu_pct=80.0, max_open_fds=4096)
    out = s.observe(ts_ns=1, cpu_pct=95.0, open_fds=100)
    assert len(out) == 1
    assert out[0].meta["resource"] == "cpu"


def test_system_anomaly_fd_breach():
    s = SystemAnomalySensor(max_cpu_pct=99.9, max_open_fds=10)
    out = s.observe(ts_ns=1, cpu_pct=10.0, open_fds=20)
    assert len(out) == 1
    assert out[0].meta["resource"] == "fds"


def test_system_anomaly_both_breach():
    s = SystemAnomalySensor(max_cpu_pct=10.0, max_open_fds=10)
    out = s.observe(ts_ns=1, cpu_pct=99.0, open_fds=99)
    kinds = {e.meta["resource"] for e in out}
    assert kinds == {"cpu", "fds"}


def test_system_anomaly_one_shot_until_recovery():
    s = SystemAnomalySensor(max_cpu_pct=80.0, max_open_fds=4096)
    s.observe(ts_ns=1, cpu_pct=95.0, open_fds=100)
    assert s.observe(ts_ns=2, cpu_pct=99.0, open_fds=100) == ()
    s.observe(ts_ns=3, cpu_pct=10.0, open_fds=100)  # disarm
    out = s.observe(ts_ns=4, cpu_pct=95.0, open_fds=100)
    assert len(out) == 1


def test_system_anomaly_rejects_invalid_cpu():
    with pytest.raises(ValueError):
        SystemAnomalySensor(max_cpu_pct=0.0, max_open_fds=10)
    with pytest.raises(ValueError):
        SystemAnomalySensor(max_cpu_pct=120.0, max_open_fds=10)


def test_system_anomaly_replay_determinism():
    def run() -> tuple:
        s = SystemAnomalySensor(max_cpu_pct=80.0, max_open_fds=10)
        a = s.observe(ts_ns=1, cpu_pct=95.0, open_fds=20)
        return tuple((e.ts_ns, e.meta["resource"]) for e in a)

    assert run() == run()
