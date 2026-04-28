"""Phase 6.T1a — :class:`BeliefState` projection tests."""

from __future__ import annotations

import dataclasses

import pytest

from core.coherence.belief_state import (
    BELIEF_STATE_VERSION,
    Regime,
    derive_belief_state,
)
from core.contracts.events import (
    EventKind,
    Side,
    SignalEvent,
    SystemEventKind,
)


def _sig(side: Side, *, conf: float = 0.7, sym: str = "EURUSD") -> SignalEvent:
    return SignalEvent(ts_ns=1, symbol=sym, side=side, confidence=conf)


# ---------------------------------------------------------------------------
# Frozen / immutable / replay-determinism
# ---------------------------------------------------------------------------


def test_belief_state_is_frozen_dataclass() -> None:
    bs = derive_belief_state(ts_ns=1, signals=[_sig(Side.BUY)])
    assert dataclasses.is_dataclass(bs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bs.regime = Regime.RANGE  # type: ignore[misc]


def test_belief_state_replay_determinism() -> None:
    """Same inputs → identical output (INV-15)."""
    sigs = [_sig(Side.BUY, conf=0.6), _sig(Side.SELL, conf=0.7)]
    a = derive_belief_state(ts_ns=42, signals=sigs)
    b = derive_belief_state(ts_ns=42, signals=sigs)
    assert a == b


# ---------------------------------------------------------------------------
# Empty + boundary cases
# ---------------------------------------------------------------------------


def test_belief_state_empty_window_is_unknown() -> None:
    bs = derive_belief_state(ts_ns=1, signals=[])
    assert bs.regime is Regime.UNKNOWN
    assert bs.regime_confidence == 0.0
    assert bs.consensus_side is Side.HOLD
    assert bs.signal_count == 0
    assert bs.avg_confidence == 0.0
    assert bs.symbols == ()
    assert bs.version == BELIEF_STATE_VERSION


def test_belief_state_only_hold_signals_is_strong_range() -> None:
    bs = derive_belief_state(
        ts_ns=1,
        signals=[_sig(Side.HOLD), _sig(Side.HOLD), _sig(Side.HOLD)],
    )
    assert bs.regime is Regime.RANGE
    assert bs.regime_confidence == 1.0
    assert bs.consensus_side is Side.HOLD


# ---------------------------------------------------------------------------
# Regime classification rules
# ---------------------------------------------------------------------------


def test_belief_state_strong_buy_consensus_is_trend_up() -> None:
    sigs = [_sig(Side.BUY) for _ in range(8)] + [_sig(Side.SELL)] * 1
    bs = derive_belief_state(ts_ns=1, signals=sigs)
    assert bs.regime is Regime.TREND_UP
    assert bs.consensus_side is Side.BUY
    assert bs.regime_confidence >= 0.8


def test_belief_state_strong_sell_consensus_is_trend_down() -> None:
    sigs = [_sig(Side.SELL) for _ in range(9)] + [_sig(Side.BUY)] * 1
    bs = derive_belief_state(ts_ns=1, signals=sigs)
    assert bs.regime is Regime.TREND_DOWN
    assert bs.consensus_side is Side.SELL


def test_belief_state_balanced_buy_sell_is_range() -> None:
    sigs = [_sig(Side.BUY) for _ in range(5)] + [_sig(Side.SELL) for _ in range(5)]
    bs = derive_belief_state(ts_ns=1, signals=sigs)
    assert bs.regime is Regime.RANGE
    assert bs.consensus_side is Side.HOLD  # tie → HOLD


def test_belief_state_vol_spike_overrides_consensus() -> None:
    sigs = [_sig(Side.BUY) for _ in range(10)]
    bs = derive_belief_state(ts_ns=1, signals=sigs, vol_spike_z=4.0)
    assert bs.regime is Regime.VOL_SPIKE
    assert 0.0 <= bs.regime_confidence <= 1.0


def test_belief_state_vol_spike_confidence_capped_at_one() -> None:
    bs = derive_belief_state(
        ts_ns=1,
        signals=[_sig(Side.BUY)],
        vol_spike_z=999.0,
    )
    assert bs.regime is Regime.VOL_SPIKE
    assert bs.regime_confidence == 1.0


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def test_belief_state_avg_confidence_and_symbols() -> None:
    sigs = [
        _sig(Side.BUY, conf=0.4, sym="BTCUSDT"),
        _sig(Side.BUY, conf=0.6, sym="BTCUSDT"),
        _sig(Side.SELL, conf=0.8, sym="EURUSD"),
    ]
    bs = derive_belief_state(ts_ns=1, signals=sigs)
    assert bs.signal_count == 3
    assert bs.avg_confidence == pytest.approx((0.4 + 0.6 + 0.8) / 3)
    assert bs.symbols == ("BTCUSDT", "EURUSD")  # sorted


# ---------------------------------------------------------------------------
# Snapshot SystemEvent (INV-53 calibration hook)
# ---------------------------------------------------------------------------


def test_belief_state_snapshot_event_shape() -> None:
    bs = derive_belief_state(
        ts_ns=12345,
        signals=[_sig(Side.BUY, conf=0.9)],
    )
    ev = bs.to_event()
    assert ev.kind is EventKind.SYSTEM
    assert ev.sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ev.ts_ns == 12345
    assert ev.source == "core.coherence.belief_state"
    assert ev.payload["regime"] == bs.regime.value
    assert ev.payload["consensus_side"] == bs.consensus_side.value
    assert ev.payload["version"] == BELIEF_STATE_VERSION
