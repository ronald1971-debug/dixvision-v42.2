"""Phase 2 — fast-execute hot-path unit tests."""

from __future__ import annotations

import pytest

from core.contracts.events import ExecutionStatus, Side, SignalEvent
from execution_engine.hot_path import (
    FastExecutor,
    HotPathOutcome,
    RiskSnapshot,
)


def _signal(
    *,
    ts_ns: int = 1_000_000_000,
    symbol: str = "BTC-USD",
    side: Side = Side.BUY,
    confidence: float = 0.9,
    qty: str | None = None,
) -> SignalEvent:
    meta = {"qty": qty} if qty is not None else {}
    return SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        confidence=confidence,
        meta=meta,
    )


def _snapshot(**overrides) -> RiskSnapshot:
    base = {
        "version": 1,
        "ts_ns": 1_000_000_000,
        "max_position_qty": 5.0,
        "max_signal_confidence": 0.5,
    }
    base.update(overrides)
    return RiskSnapshot(**base)


def test_approved_path_emits_filled_eligible_event():
    fx = FastExecutor()
    decision = fx.execute(
        signal=_signal(),
        snapshot=_snapshot(),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.APPROVED
    assert decision.event.status is ExecutionStatus.APPROVED
    assert decision.event.qty == 1.0
    assert decision.event.price == 50_000.0
    assert decision.event.venue == "hot_path"
    assert decision.event.meta["risk_version"] == "1"


def test_rejected_when_risk_stale():
    fx = FastExecutor(max_staleness_ns=1_000_000)  # 1 ms
    decision = fx.execute(
        signal=_signal(ts_ns=2_000_000_000),
        snapshot=_snapshot(ts_ns=1_000_000_000),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_RISK_STALE
    assert decision.event.status is ExecutionStatus.REJECTED


def test_rejected_when_no_mark():
    fx = FastExecutor()
    decision = fx.execute(
        signal=_signal(),
        snapshot=_snapshot(),
        mark_price=0.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_NO_MARK


def test_rejected_when_halted():
    fx = FastExecutor()
    decision = fx.execute(
        signal=_signal(),
        snapshot=_snapshot(halted=True),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
    assert decision.event.meta["reason"] == "halted"


def test_rejected_when_below_confidence_floor():
    fx = FastExecutor()
    decision = fx.execute(
        signal=_signal(confidence=0.1),
        snapshot=_snapshot(max_signal_confidence=0.5),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_LOW_CONFIDENCE


def test_rejected_when_hold_side():
    fx = FastExecutor()
    decision = fx.execute(
        signal=_signal(side=Side.HOLD),
        snapshot=_snapshot(),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_HOLD


def test_rejected_when_qty_above_cap():
    fx = FastExecutor()
    decision = fx.execute(
        signal=_signal(qty="100.0"),
        snapshot=_snapshot(max_position_qty=10.0),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_LIMIT
    assert decision.event.meta["reason"] == "qty_above_cap"


def test_symbol_cap_overrides_global_cap():
    fx = FastExecutor()
    snap = _snapshot(
        max_position_qty=100.0,
        symbol_caps={"BTC-USD": 1.0},
    )
    decision = fx.execute(
        signal=_signal(qty="2.0"),
        snapshot=snap,
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.REJECTED_LIMIT


def test_default_qty_used_when_meta_missing():
    fx = FastExecutor(default_qty=2.5)
    decision = fx.execute(
        signal=_signal(),
        snapshot=_snapshot(max_position_qty=10.0),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.APPROVED
    assert decision.event.qty == 2.5


def test_invalid_meta_qty_falls_back_to_default():
    fx = FastExecutor(default_qty=1.5)
    decision = fx.execute(
        signal=_signal(qty="not-a-number"),
        snapshot=_snapshot(max_position_qty=10.0),
        mark_price=50_000.0,
    )
    assert decision.outcome is HotPathOutcome.APPROVED
    assert decision.event.qty == 1.5


def test_order_id_increments_monotonically():
    fx = FastExecutor()
    a = fx.execute(
        signal=_signal(),
        snapshot=_snapshot(),
        mark_price=50_000.0,
    )
    b = fx.execute(
        signal=_signal(),
        snapshot=_snapshot(),
        mark_price=50_000.0,
    )
    assert a.event.order_id == "HP-00000001"
    assert b.event.order_id == "HP-00000002"


def test_replay_determinism_same_inputs_same_event():
    def run() -> tuple:
        fx = FastExecutor()
        out = []
        for i in range(3):
            d = fx.execute(
                signal=_signal(ts_ns=1_000_000_000 + i),
                snapshot=_snapshot(),
                mark_price=50_000.0,
            )
            out.append(d.event)
        return tuple(out)

    assert run() == run()


def test_invalid_constructor_args_rejected():
    with pytest.raises(ValueError):
        FastExecutor(max_staleness_ns=0)
    with pytest.raises(ValueError):
        FastExecutor(default_qty=0.0)
