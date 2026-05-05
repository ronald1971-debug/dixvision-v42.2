"""Tests for AGT-01 scalper agent (INV-54)."""

from __future__ import annotations

import pytest

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.agents import ScalperAgent
from intelligence_engine.agents._base import AgentBase


def _tick(ts: int, bid: float, ask: float) -> MarketTick:
    return MarketTick(ts_ns=ts, symbol="BTC-USD", bid=bid, ask=ask, last=0.5 * (bid + ask))


def _signal(ts: int, side: Side, conf: float = 0.8) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts,
        symbol="BTC-USD",
        side=side,
        confidence=conf,
        plugin_chain=("test",),
        meta={"signal_id": f"sig-{ts}"},
        produced_by_engine="intelligence_engine",
    )


def test_scalper_satisfies_introspection_protocol() -> None:
    a = ScalperAgent()
    assert isinstance(a, AgentIntrospection)


def test_state_snapshot_keys_match_registry() -> None:
    a = ScalperAgent()
    snap = a.state_snapshot()
    allowlist = AgentBase._load_allowed_state_keys("AGT-01-scalper")
    assert allowlist, "allowlist must be configured for AGT-01-scalper"
    assert set(snap.keys()).issubset(allowlist)


def test_state_snapshot_is_pure() -> None:
    a = ScalperAgent()
    s1 = dict(a.state_snapshot())
    s2 = dict(a.state_snapshot())
    assert s1 == s2


def test_state_snapshot_values_are_strings() -> None:
    a = ScalperAgent()
    snap = a.state_snapshot()
    for key, value in snap.items():
        assert isinstance(value, str), f"{key} -> {value!r}"


def test_warmup_emits_hold() -> None:
    a = ScalperAgent(mid_window_size=4)
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.5, 101.5))
    trace = a.decide(_signal(2, Side.BUY))
    assert trace.direction == "HOLD"
    assert "momentum_neutral" in trace.rationale_tags


def test_aligned_buy_passes() -> None:
    a = ScalperAgent(mid_window_size=4, momentum_threshold_bps=0.5, min_confidence=0.0)
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.2, 101.2))
    a.observe_tick(_tick(2, 99.4, 101.4))
    a.observe_tick(_tick(3, 99.6, 101.6))
    trace = a.decide(_signal(4, Side.BUY, conf=0.8))
    assert trace.direction == "BUY"
    assert pytest.approx(trace.confidence) == 0.8
    assert "momentum_up" in trace.rationale_tags
    assert "mid_drift_buy" in trace.rationale_tags


def test_counter_signal_downgraded_to_hold() -> None:
    a = ScalperAgent(mid_window_size=4, momentum_threshold_bps=0.5, min_confidence=0.0)
    for i in range(4):
        a.observe_tick(_tick(i, 99.0 + i * 0.2, 101.0 + i * 0.2))
    trace = a.decide(_signal(4, Side.SELL))
    assert trace.direction == "HOLD"


def test_aligned_sell_passes() -> None:
    a = ScalperAgent(mid_window_size=4, momentum_threshold_bps=0.5, min_confidence=0.0)
    for i in range(4):
        a.observe_tick(_tick(i, 99.6 - i * 0.2, 101.6 - i * 0.2))
    trace = a.decide(_signal(4, Side.SELL, conf=0.7))
    assert trace.direction == "SELL"
    assert "momentum_down" in trace.rationale_tags


def test_low_confidence_signal_downgraded() -> None:
    a = ScalperAgent(mid_window_size=4, momentum_threshold_bps=0.5, min_confidence=0.5)
    for i in range(4):
        a.observe_tick(_tick(i, 99.0 + i * 0.2, 101.0 + i * 0.2))
    trace = a.decide(_signal(4, Side.BUY, conf=0.1))
    assert trace.direction == "HOLD"
    assert "confidence_below_floor" in trace.rationale_tags


def test_recent_decisions_is_bounded_ring() -> None:
    a = ScalperAgent(ring_capacity=3)
    for i in range(5):
        a.decide(_signal(i, Side.HOLD))
    recent = a.recent_decisions(10)
    assert len(recent) == 3
    assert all(isinstance(t, AgentDecisionTrace) for t in recent)
    assert [t.ts_ns for t in recent] == [2, 3, 4]


def test_recent_decisions_n_zero() -> None:
    a = ScalperAgent()
    a.decide(_signal(0, Side.HOLD))
    assert a.recent_decisions(0) == ()
    assert a.recent_decisions(-1) == ()


def test_recent_decisions_n_partial() -> None:
    a = ScalperAgent(ring_capacity=10)
    for i in range(5):
        a.decide(_signal(i, Side.HOLD))
    recent = a.recent_decisions(2)
    assert len(recent) == 2
    assert [t.ts_ns for t in recent] == [3, 4]


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        ScalperAgent(mid_window_size=1)
    with pytest.raises(ValueError):
        ScalperAgent(momentum_threshold_bps=-0.1)
    with pytest.raises(ValueError):
        ScalperAgent(min_confidence=2.0)
    with pytest.raises(ValueError):
        ScalperAgent(ring_capacity=0)


def test_invalid_tick_skipped() -> None:
    a = ScalperAgent(mid_window_size=2)
    a.observe_tick(MarketTick(ts_ns=0, symbol="X", bid=0.0, ask=1.0, last=1.0))
    a.observe_tick(MarketTick(ts_ns=1, symbol="X", bid=2.0, ask=1.0, last=1.0))  # crossed
    trace = a.decide(_signal(2, Side.BUY))
    assert trace.direction == "HOLD"
