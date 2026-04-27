"""Tests for the Phase 3 LearningInterface (Indira → Learning bridge)."""

from __future__ import annotations

import pytest

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from intelligence_engine.learning_interface import (
    FeedbackRecord,
    LearningInterface,
)


def _signal(side: Side = Side.BUY, plugin: str = "p1") -> SignalEvent:
    return SignalEvent(
        ts_ns=10,
        symbol="BTCUSDT",
        side=side,
        confidence=0.8,
        plugin_chain=(plugin,),
        meta={},
    )


def _exec(
    status: ExecutionStatus = ExecutionStatus.FILLED,
    qty: float = 1.0,
    price: float = 100.0,
    symbol: str = "BTCUSDT",
) -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=20,
        symbol=symbol,
        side=Side.BUY,
        qty=qty,
        price=price,
        status=status,
        venue="paper",
        order_id="o1",
        meta={},
    )


def test_learning_interface_records_filled_buy_pnl():
    li = LearningInterface()
    rec = li.record(
        signal=_signal(Side.BUY),
        execution=_exec(qty=2.0, price=100.0),
        mark_price=101.0,
    )
    assert isinstance(rec, FeedbackRecord)
    assert rec.is_realised()
    assert rec.executed_qty == 2.0
    assert rec.executed_price == 100.0
    # BUY PnL = qty * (mark - entry) = 2 * 1 = 2
    assert rec.realised_pnl == pytest.approx(2.0)


def test_learning_interface_records_filled_sell_pnl():
    li = LearningInterface()
    rec = li.record(
        signal=_signal(Side.SELL),
        execution=_exec(qty=2.0, price=100.0),
        mark_price=99.0,
    )
    # SELL PnL = qty * (entry - mark) = 2 * 1 = 2
    assert rec.realised_pnl == pytest.approx(2.0)


def test_learning_interface_rejected_no_pnl_no_fill():
    li = LearningInterface()
    rec = li.record(
        signal=_signal(),
        execution=_exec(status=ExecutionStatus.REJECTED),
        mark_price=101.0,
    )
    assert not rec.is_realised()
    assert rec.executed_qty == 0.0
    assert rec.executed_price == 0.0
    assert rec.realised_pnl == 0.0


def test_learning_interface_partial_fill_is_realised():
    li = LearningInterface()
    rec = li.record(
        signal=_signal(),
        execution=_exec(status=ExecutionStatus.PARTIALLY_FILLED, qty=0.5),
        mark_price=100.5,
    )
    assert rec.is_realised()
    assert rec.executed_qty == 0.5


def test_learning_interface_pnl_zero_without_mark():
    li = LearningInterface()
    rec = li.record(
        signal=_signal(),
        execution=_exec(),
        mark_price=None,
    )
    assert rec.realised_pnl == 0.0


def test_learning_interface_symbol_mismatch_raises():
    li = LearningInterface()
    with pytest.raises(ValueError):
        li.record(
            signal=_signal(),
            execution=_exec(symbol="ETHUSDT"),
            mark_price=100.0,
        )


def test_learning_interface_strategy_id_from_plugin_chain():
    li = LearningInterface()
    rec = li.record(
        signal=SignalEvent(
            ts_ns=10,
            symbol="X",
            side=Side.BUY,
            confidence=0.5,
            plugin_chain=("first", "second"),
        ),
        execution=ExecutionEvent(
            ts_ns=20,
            symbol="X",
            side=Side.BUY,
            qty=1.0,
            price=10.0,
            status=ExecutionStatus.FILLED,
        ),
    )
    assert rec.strategy_id == "first"


def test_learning_interface_drain_clears_buffer():
    li = LearningInterface()
    li.record(signal=_signal(), execution=_exec())
    li.record(signal=_signal(), execution=_exec())
    assert len(li) == 2
    drained = li.drain()
    assert len(drained) == 2
    assert len(li) == 0
    assert li.drain() == ()


def test_learning_interface_record_many():
    li = LearningInterface()
    pairs = [(_signal(), _exec()), (_signal(), _exec())]
    out = li.record_many(pairs, mark_price=101.0)
    assert len(out) == 2
    assert all(r.realised_pnl == pytest.approx(1.0) for r in out)


def test_learning_interface_replay_determinism():
    def run() -> tuple:
        li = LearningInterface()
        for i in range(5):
            li.record(
                signal=SignalEvent(
                    ts_ns=10 + i,
                    symbol="X",
                    side=Side.BUY,
                    confidence=0.5,
                    plugin_chain=("p",),
                ),
                execution=ExecutionEvent(
                    ts_ns=20 + i,
                    symbol="X",
                    side=Side.BUY,
                    qty=1.0,
                    price=10.0 + i * 0.1,
                    status=ExecutionStatus.FILLED,
                ),
                mark_price=11.0,
            )
        return tuple(
            (r.ts_ns, r.symbol, r.executed_price, round(r.realised_pnl, 9))
            for r in li.drain()
        )

    assert run() == run()
