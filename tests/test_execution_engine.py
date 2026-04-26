"""Phase E1 — ExecutionEngine end-to-end + paper broker tests."""

from __future__ import annotations

import time

import pytest

from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
)
from core.contracts.market import MarketTick
from execution_engine.adapters import BrokerAdapter, PaperBroker
from execution_engine.engine import ExecutionEngine
from intelligence_engine.engine import IntelligenceEngine

# ---------------------------------------------------------------------------
# PaperBroker unit tests
# ---------------------------------------------------------------------------


def test_paper_broker_implements_protocol():
    assert isinstance(PaperBroker(), BrokerAdapter)


def test_paper_broker_buy_fills_at_mark_when_no_slippage():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=10, symbol="EURUSD", side=Side.BUY, confidence=0.7)
    out = broker.submit(sig, mark_price=1.10)
    assert isinstance(out, ExecutionEvent)
    assert out.status is ExecutionStatus.FILLED
    assert out.side is Side.BUY
    assert out.price == pytest.approx(1.10)
    assert out.qty == pytest.approx(1.0)
    assert out.venue == "paper"
    assert out.order_id.startswith("PAPER-")


def test_paper_broker_sell_applies_slippage():
    broker = PaperBroker(slippage_bps=10.0)  # 0.10 %
    sig = SignalEvent(ts_ns=11, symbol="BTCUSDT", side=Side.SELL, confidence=0.8)
    out = broker.submit(sig, mark_price=50_000.0)
    assert out.status is ExecutionStatus.FILLED
    assert out.side is Side.SELL
    assert out.price == pytest.approx(50_000.0 - 50.0)


def test_paper_broker_hold_rejects():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=12, symbol="X", side=Side.HOLD, confidence=0.0)
    out = broker.submit(sig, mark_price=100.0)
    assert out.status is ExecutionStatus.REJECTED
    assert out.qty == 0.0


def test_paper_broker_non_positive_mark_fails():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=13, symbol="X", side=Side.BUY, confidence=0.5)
    out = broker.submit(sig, mark_price=0.0)
    assert out.status is ExecutionStatus.FAILED
    assert out.order_id == ""


def test_paper_broker_meta_qty_overrides_default():
    broker = PaperBroker(default_qty=1.0)
    sig = SignalEvent(
        ts_ns=14,
        symbol="X",
        side=Side.BUY,
        confidence=0.5,
        meta={"qty": "2.5"},
    )
    out = broker.submit(sig, mark_price=10.0)
    assert out.qty == pytest.approx(2.5)


def test_paper_broker_order_ids_are_monotonic():
    broker = PaperBroker()
    sig = SignalEvent(ts_ns=15, symbol="X", side=Side.BUY, confidence=0.5)
    a = broker.submit(sig, mark_price=10.0)
    b = broker.submit(sig, mark_price=10.0)
    assert a.order_id < b.order_id


def test_paper_broker_rejects_negative_slippage():
    with pytest.raises(ValueError):
        PaperBroker(slippage_bps=-1.0)


def test_paper_broker_rejects_non_positive_default_qty():
    with pytest.raises(ValueError):
        PaperBroker(default_qty=0.0)


# ---------------------------------------------------------------------------
# ExecutionEngine integration tests
# ---------------------------------------------------------------------------


def test_execution_engine_no_mark_returns_failed_event():
    engine = ExecutionEngine()
    sig = SignalEvent(ts_ns=20, symbol="EURUSD", side=Side.BUY, confidence=0.7)
    out = engine.process(sig)
    assert len(out) == 1
    evt = out[0]
    assert isinstance(evt, ExecutionEvent)
    assert evt.status is ExecutionStatus.FAILED
    assert evt.kind is EventKind.EXECUTION


def test_execution_engine_with_mark_fills_buy():
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=21, symbol="EURUSD", bid=1.0998, ask=1.1002, last=1.1)
    )
    sig = SignalEvent(ts_ns=22, symbol="EURUSD", side=Side.BUY, confidence=0.7)
    out = engine.process(sig)
    assert len(out) == 1
    evt = out[0]
    assert isinstance(evt, ExecutionEvent)
    assert evt.status is ExecutionStatus.FILLED
    assert evt.price == pytest.approx(1.1)


def test_execution_engine_ignores_non_signal_events():
    engine = ExecutionEngine()
    haz = HazardEvent(
        ts_ns=23, code="API_DOWN", severity=HazardSeverity.HIGH, source="system"
    )
    assert engine.process(haz) == ()


def test_execution_engine_reset_mark_with_zero_last_is_ignored():
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=24, symbol="X", bid=1.0, ask=1.01, last=1.0)
    )
    engine.on_market(
        MarketTick(ts_ns=25, symbol="X", bid=1.0, ask=1.01, last=0.0)
    )
    sig = SignalEvent(ts_ns=26, symbol="X", side=Side.BUY, confidence=0.5)
    out = engine.process(sig)
    assert out[0].status is ExecutionStatus.FILLED


def test_execution_engine_default_adapter_is_paper():
    engine = ExecutionEngine()
    assert engine.adapter.name == "paper"


# ---------------------------------------------------------------------------
# End-to-end: Intelligence -> Execution
# ---------------------------------------------------------------------------


def test_e2e_intelligence_to_execution_round_trip():
    intelligence = IntelligenceEngine()
    execution = ExecutionEngine()

    execution.on_market(
        MarketTick(ts_ns=100, symbol="BTCUSDT", bid=49_990.0, ask=50_010.0, last=50_000.0)
    )

    sig = SignalEvent(
        ts_ns=101,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.8,
        plugin_chain=("phase_e1_test",),
    )

    intelligence_out = intelligence.process(sig)
    assert len(intelligence_out) == 1

    execution_events: list = []
    for evt in intelligence_out:
        execution_events.extend(execution.process(evt))

    assert len(execution_events) == 1
    exec_evt = execution_events[0]
    assert isinstance(exec_evt, ExecutionEvent)
    assert exec_evt.status is ExecutionStatus.FILLED
    assert exec_evt.side is Side.BUY
    assert exec_evt.price == pytest.approx(50_000.0)
    assert exec_evt.symbol == "BTCUSDT"


def test_e2e_replay_is_deterministic():
    """INV-15 / TEST-01 — same input sequence -> bit-identical output."""

    def run() -> list:
        intelligence = IntelligenceEngine()
        execution = ExecutionEngine(adapter=PaperBroker(slippage_bps=2.5))
        execution.on_market(
            MarketTick(
                ts_ns=200, symbol="EURUSD", bid=1.0998, ask=1.1002, last=1.1
            )
        )
        signals = [
            SignalEvent(ts_ns=200 + i, symbol="EURUSD", side=Side.BUY, confidence=0.6)
            for i in range(5)
        ]
        out: list = []
        for s in signals:
            for ev in intelligence.process(s):
                out.extend(execution.process(ev))
        return out

    a = run()
    b = run()
    assert a == b
    assert len(a) == 5


# ---------------------------------------------------------------------------
# PERF-01..02 latency SLO (TEST-06).
# Build plan target: p50 < 1 ms / p99 < 5 ms in CI.
# We assert a generous CI envelope (p50 < 2 ms / p99 < 10 ms) so flaky CI
# doesn't fail on cold caches; the typical machine is well below target.
# ---------------------------------------------------------------------------


def test_execution_engine_latency_slo():
    engine = ExecutionEngine()
    engine.on_market(
        MarketTick(ts_ns=300, symbol="X", bid=99.5, ask=100.5, last=100.0)
    )
    sig = SignalEvent(ts_ns=300, symbol="X", side=Side.BUY, confidence=0.7)

    # warm-up
    for _ in range(200):
        engine.process(sig)

    n = 5_000
    samples = [0] * n
    for i in range(n):
        t0 = time.perf_counter_ns()
        engine.process(sig)
        samples[i] = time.perf_counter_ns() - t0

    samples.sort()
    p50_ms = samples[n // 2] / 1_000_000.0
    p99_ms = samples[int(n * 0.99)] / 1_000_000.0

    assert p50_ms < 2.0, f"p50 too high: {p50_ms:.3f} ms"
    assert p99_ms < 10.0, f"p99 too high: {p99_ms:.3f} ms"
