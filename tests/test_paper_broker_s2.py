"""Paper-S2 — PaperBroker upgrade unit tests.

Covers the five new fidelity features added to
:class:`execution_engine.adapters.paper.PaperBroker`:

1. Deterministic latency model (no clocks, no PRNG).
2. Maker / taker fee model.
3. Virtual balance ledger (cash + per-symbol position).
4. Partial fills via ``signal.meta['max_fill_qty']`` cap.
5. Bounded fill-tracking ring.

These tests sit alongside (not replacing) the existing v1 PaperBroker
suite in :mod:`tests.test_execution_engine`, which proves that all
v1 callers (``PaperBroker()``, ``slippage_bps``, ``default_qty``)
continue to work unchanged.
"""

from __future__ import annotations

import pytest

from core.contracts.events import ExecutionStatus, Side, SignalEvent
from execution_engine.adapters.paper import PaperBroker

# ---------------------------------------------------------------------------
# 1. Deterministic latency model
# ---------------------------------------------------------------------------


def _signal(ts_ns: int, side: Side = Side.BUY) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="BTCUSDT",
        side=side,
        confidence=0.7,
    )


def test_latency_zero_by_default():
    broker = PaperBroker()
    out = broker.submit(_signal(100), mark_price=50_000.0)
    assert out.ts_ns == 100
    assert out.meta["latency_ns"] == "0"


def test_latency_constant_base_no_jitter():
    broker = PaperBroker(latency_ns_base=5_000)
    out = broker.submit(_signal(100), mark_price=50_000.0)
    assert out.ts_ns == 100 + 5_000
    assert out.meta["latency_ns"] == "5000"


def test_latency_jitter_is_deterministic():
    """Same inputs → same latency; different counters → spread within window."""
    broker_a = PaperBroker(latency_ns_jitter=1_000)
    broker_b = PaperBroker(latency_ns_jitter=1_000)
    out_a = broker_a.submit(_signal(123), mark_price=100.0)
    out_b = broker_b.submit(_signal(123), mark_price=100.0)
    assert out_a.ts_ns == out_b.ts_ns
    # Bounded by the configured jitter window (inclusive).
    delta = out_a.ts_ns - 123
    assert 0 <= delta <= 1_000


def test_latency_jitter_spreads_across_counters():
    broker = PaperBroker(latency_ns_jitter=1_000_000)
    deltas = []
    for i in range(1, 11):
        out = broker.submit(_signal(1_000 * i), mark_price=100.0)
        deltas.append(out.ts_ns - 1_000 * i)
    # Not constant — the jitter formula spreads results.
    assert len(set(deltas)) > 1
    for d in deltas:
        assert 0 <= d <= 1_000_000


def test_latency_negative_rejects_at_construction():
    with pytest.raises(ValueError):
        PaperBroker(latency_ns_base=-1)
    with pytest.raises(ValueError):
        PaperBroker(latency_ns_jitter=-1)


# ---------------------------------------------------------------------------
# 2. Maker / taker fee model
# ---------------------------------------------------------------------------


def test_fee_zero_by_default():
    broker = PaperBroker()
    out = broker.submit(_signal(1), mark_price=100.0)
    assert float(out.meta["fee_usd"]) == pytest.approx(0.0)
    assert float(out.meta["notional_usd"]) == pytest.approx(100.0)


def test_taker_fee_charged_against_notional_buy():
    broker = PaperBroker(taker_fee_bps=10.0)  # 10 bps = 0.10%
    out = broker.submit(_signal(1), mark_price=100.0)
    # fee = notional * 0.001 = 100 * 0.001 = 0.10
    assert float(out.meta["fee_usd"]) == pytest.approx(0.10)
    # BUY: cash -= notional + fee → cash = -100.10
    assert broker.cash_balance() == pytest.approx(-100.10)


def test_taker_fee_charged_against_notional_sell():
    broker = PaperBroker(taker_fee_bps=10.0, initial_cash=1_000.0)
    out = broker.submit(_signal(1, Side.SELL), mark_price=100.0)
    # SELL: cash += notional - fee = 100 - 0.10 = 99.90
    assert broker.cash_balance() == pytest.approx(1_000.0 + 99.90)
    assert float(out.meta["fee_usd"]) == pytest.approx(0.10)


def test_negative_fees_rejected_at_construction():
    with pytest.raises(ValueError):
        PaperBroker(taker_fee_bps=-1.0)
    with pytest.raises(ValueError):
        PaperBroker(maker_fee_bps=-1.0)


# ---------------------------------------------------------------------------
# 3. Virtual balance ledger
# ---------------------------------------------------------------------------


def test_initial_cash_starts_at_zero_unless_configured():
    assert PaperBroker().cash_balance() == pytest.approx(0.0)
    assert PaperBroker(initial_cash=10_000.0).cash_balance() == pytest.approx(10_000.0)
    assert PaperBroker().initial_cash() == pytest.approx(0.0)


def test_buy_decrements_cash_and_grows_position():
    broker = PaperBroker(initial_cash=10_000.0)
    sig = SignalEvent(
        ts_ns=1, symbol="BTCUSDT", side=Side.BUY,
        confidence=0.8, meta={"qty": "0.5"},
    )
    out = broker.submit(sig, mark_price=200.0)
    assert out.status is ExecutionStatus.FILLED
    # cash -= 200 * 0.5 = 100 → 9_900
    assert broker.cash_balance() == pytest.approx(9_900.0)
    assert broker.position("BTCUSDT") == pytest.approx(0.5)


def test_sell_increments_cash_and_shrinks_position():
    broker = PaperBroker(initial_cash=10_000.0)
    # Build an inventory first.
    broker.submit(
        SignalEvent(
            ts_ns=1, symbol="BTCUSDT", side=Side.BUY,
            confidence=0.8, meta={"qty": "1.0"},
        ),
        mark_price=200.0,
    )
    assert broker.position("BTCUSDT") == pytest.approx(1.0)
    out = broker.submit(
        SignalEvent(
            ts_ns=2, symbol="BTCUSDT", side=Side.SELL,
            confidence=0.8, meta={"qty": "0.4"},
        ),
        mark_price=210.0,
    )
    # Realised: cash from sell = 210 * 0.4 = 84 → 9_800 + 84 = 9_884
    assert broker.cash_balance() == pytest.approx(9_884.0)
    assert broker.position("BTCUSDT") == pytest.approx(0.6)
    assert out.status is ExecutionStatus.FILLED


def test_hold_does_not_move_ledger():
    broker = PaperBroker(initial_cash=10_000.0)
    out = broker.submit(_signal(1, Side.HOLD), mark_price=100.0)
    assert out.status is ExecutionStatus.REJECTED
    assert broker.cash_balance() == pytest.approx(10_000.0)
    assert broker.position("BTCUSDT") == pytest.approx(0.0)


def test_failed_does_not_move_ledger():
    broker = PaperBroker(initial_cash=10_000.0)
    out = broker.submit(_signal(1), mark_price=0.0)
    assert out.status is ExecutionStatus.FAILED
    assert broker.cash_balance() == pytest.approx(10_000.0)
    assert broker.position("BTCUSDT") == pytest.approx(0.0)


def test_positions_returns_only_nonzero():
    broker = PaperBroker(initial_cash=10_000.0)
    broker.submit(
        SignalEvent(ts_ns=1, symbol="A", side=Side.BUY, confidence=0.7,
                    meta={"qty": "1.0"}),
        mark_price=10.0,
    )
    broker.submit(
        SignalEvent(ts_ns=2, symbol="B", side=Side.BUY, confidence=0.7,
                    meta={"qty": "2.0"}),
        mark_price=20.0,
    )
    broker.submit(
        SignalEvent(ts_ns=3, symbol="A", side=Side.SELL, confidence=0.7,
                    meta={"qty": "1.0"}),
        mark_price=10.0,
    )
    snap = broker.positions()
    assert "A" not in snap  # netted out
    assert snap["B"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. Partial fills
# ---------------------------------------------------------------------------


def test_partial_fill_when_max_fill_qty_below_default():
    broker = PaperBroker(default_qty=10.0)
    sig = SignalEvent(
        ts_ns=1, symbol="X", side=Side.BUY, confidence=0.7,
        meta={"max_fill_qty": "3.0"},
    )
    out = broker.submit(sig, mark_price=100.0)
    assert out.status is ExecutionStatus.PARTIALLY_FILLED
    assert out.qty == pytest.approx(3.0)
    assert out.meta["requested_qty"] == "10"
    assert out.meta["filled_qty"] == "3"
    assert out.meta["remaining_qty"] == "7"


def test_partial_fill_uses_qty_meta_as_requested():
    broker = PaperBroker(default_qty=1.0)
    sig = SignalEvent(
        ts_ns=1, symbol="X", side=Side.BUY, confidence=0.7,
        meta={"qty": "10.0", "max_fill_qty": "4.0"},
    )
    out = broker.submit(sig, mark_price=50.0)
    assert out.status is ExecutionStatus.PARTIALLY_FILLED
    assert out.qty == pytest.approx(4.0)
    assert out.meta["requested_qty"] == "10"
    assert out.meta["filled_qty"] == "4"


def test_no_partial_when_cap_above_requested():
    broker = PaperBroker(default_qty=2.0)
    sig = SignalEvent(
        ts_ns=1, symbol="X", side=Side.BUY, confidence=0.7,
        meta={"max_fill_qty": "100.0"},
    )
    out = broker.submit(sig, mark_price=10.0)
    assert out.status is ExecutionStatus.FILLED
    assert "requested_qty" not in out.meta


def test_zero_max_fill_qty_rejects_without_ledger_change():
    broker = PaperBroker(initial_cash=1_000.0)
    sig = SignalEvent(
        ts_ns=1, symbol="X", side=Side.BUY, confidence=0.7,
        meta={"max_fill_qty": "0.0"},
    )
    out = broker.submit(sig, mark_price=100.0)
    assert out.status is ExecutionStatus.REJECTED
    assert broker.cash_balance() == pytest.approx(1_000.0)
    assert broker.position("X") == pytest.approx(0.0)


def test_invalid_max_fill_qty_falls_back_to_full_fill():
    broker = PaperBroker(default_qty=2.0)
    sig = SignalEvent(
        ts_ns=1, symbol="X", side=Side.BUY, confidence=0.7,
        meta={"max_fill_qty": "not-a-number"},
    )
    out = broker.submit(sig, mark_price=10.0)
    assert out.status is ExecutionStatus.FILLED
    assert out.qty == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 5. Fill ring
# ---------------------------------------------------------------------------


def test_fill_ring_retains_recent_fills():
    broker = PaperBroker()
    for i in range(5):
        broker.submit(_signal(i), mark_price=100.0)
    fills = broker.recent_fills()
    assert len(fills) == 5
    # Ordered oldest → newest.
    assert [f.ts_ns for f in fills] == [0, 1, 2, 3, 4]


def test_fill_ring_n_returns_last_n():
    broker = PaperBroker()
    for i in range(10):
        broker.submit(_signal(i), mark_price=100.0)
    last3 = broker.recent_fills(n=3)
    assert [f.ts_ns for f in last3] == [7, 8, 9]


def test_fill_ring_bounded_by_size():
    broker = PaperBroker(fill_ring_size=3)
    for i in range(10):
        broker.submit(_signal(i), mark_price=100.0)
    fills = broker.recent_fills()
    assert len(fills) == 3
    assert [f.ts_ns for f in fills] == [7, 8, 9]


def test_fill_ring_disabled_when_size_zero():
    broker = PaperBroker(fill_ring_size=0)
    broker.submit(_signal(1), mark_price=100.0)
    broker.submit(_signal(2), mark_price=100.0)
    assert broker.recent_fills() == []
    assert broker.recent_fills(n=10) == []


def test_fill_ring_excludes_rejected_and_failed():
    broker = PaperBroker()
    broker.submit(_signal(1, Side.HOLD), mark_price=100.0)  # REJECTED
    broker.submit(_signal(2), mark_price=0.0)               # FAILED
    broker.submit(_signal(3), mark_price=100.0)             # FILLED
    fills = broker.recent_fills()
    assert len(fills) == 1
    assert fills[0].status is ExecutionStatus.FILLED


def test_fill_ring_includes_partial_fills():
    broker = PaperBroker(default_qty=10.0)
    sig = SignalEvent(
        ts_ns=1, symbol="X", side=Side.BUY, confidence=0.7,
        meta={"max_fill_qty": "3.0"},
    )
    broker.submit(sig, mark_price=100.0)
    fills = broker.recent_fills()
    assert len(fills) == 1
    assert fills[0].status is ExecutionStatus.PARTIALLY_FILLED


def test_negative_fill_ring_size_rejected_at_construction():
    with pytest.raises(ValueError):
        PaperBroker(fill_ring_size=-1)


# ---------------------------------------------------------------------------
# Cross-cutting determinism
# ---------------------------------------------------------------------------


def test_two_brokers_same_inputs_produce_identical_outputs():
    """INV-15 — replay determinism across the new fields."""
    cfg = dict(
        slippage_bps=2.5,
        taker_fee_bps=5.0,
        latency_ns_base=1_000,
        latency_ns_jitter=10_000,
        initial_cash=10_000.0,
        fill_ring_size=64,
    )
    a = PaperBroker(**cfg)
    b = PaperBroker(**cfg)
    sigs = [
        SignalEvent(ts_ns=10 * i, symbol="BTCUSDT",
                    side=Side.BUY if i % 2 else Side.SELL,
                    confidence=0.5 + 0.05 * i,
                    meta={"qty": str(0.1 * (i + 1))})
        for i in range(20)
    ]
    out_a = [a.submit(s, mark_price=50_000.0 + s.ts_ns) for s in sigs]
    out_b = [b.submit(s, mark_price=50_000.0 + s.ts_ns) for s in sigs]
    for ea, eb in zip(out_a, out_b, strict=True):
        assert ea == eb
    assert a.cash_balance() == pytest.approx(b.cash_balance())
    assert a.positions() == b.positions()
