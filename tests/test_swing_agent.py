"""Tests for AGT-02 swing agent (INV-54)."""

from __future__ import annotations

import pytest

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.agents import SwingAgent
from intelligence_engine.agents._base import AgentBase


def _tick(ts: int, bid: float, ask: float) -> MarketTick:
    return MarketTick(
        ts_ns=ts, symbol="BTC-USD", bid=bid, ask=ask, last=0.5 * (bid + ask)
    )


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


def test_swing_satisfies_introspection_protocol() -> None:
    a = SwingAgent()
    assert isinstance(a, AgentIntrospection)


def test_state_snapshot_keys_match_registry() -> None:
    a = SwingAgent()
    snap = a.state_snapshot()
    allowlist = AgentBase._load_allowed_state_keys("AGT-02-swing")
    assert allowlist, "allowlist must be configured for AGT-02-swing"
    assert set(snap.keys()).issubset(allowlist)


def test_state_snapshot_is_pure() -> None:
    a = SwingAgent()
    s1 = dict(a.state_snapshot())
    s2 = dict(a.state_snapshot())
    assert s1 == s2


def test_state_snapshot_values_are_strings() -> None:
    a = SwingAgent()
    snap = a.state_snapshot()
    for key, value in snap.items():
        assert isinstance(value, str), f"{key} -> {value!r}"


def test_warmup_emits_hold() -> None:
    a = SwingAgent(fast_window_size=2, slow_window_size=4)
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.5, 101.5))
    trace = a.decide(_signal(2, Side.BUY))
    assert trace.direction == "HOLD"
    assert "momentum_neutral" in trace.rationale_tags


def test_aligned_buy_passes_on_positive_crossover() -> None:
    a = SwingAgent(
        fast_window_size=2,
        slow_window_size=4,
        crossover_threshold_bps=1.0,
        min_confidence=0.0,
    )
    # Slow window includes earlier flat ticks; fast window catches recent rally.
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.0, 101.0))
    a.observe_tick(_tick(2, 99.5, 101.5))
    a.observe_tick(_tick(3, 100.0, 102.0))
    trace = a.decide(_signal(4, Side.BUY, conf=0.8))
    assert trace.direction == "BUY"
    assert pytest.approx(trace.confidence) == 0.8
    assert "momentum_up" in trace.rationale_tags
    assert "ma_crossover_buy" in trace.rationale_tags


def test_counter_signal_downgraded_to_hold() -> None:
    a = SwingAgent(
        fast_window_size=2,
        slow_window_size=4,
        crossover_threshold_bps=1.0,
        min_confidence=0.0,
    )
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.0, 101.0))
    a.observe_tick(_tick(2, 99.5, 101.5))
    a.observe_tick(_tick(3, 100.0, 102.0))
    trace = a.decide(_signal(4, Side.SELL))
    assert trace.direction == "HOLD"


def test_aligned_sell_passes_on_negative_crossover() -> None:
    a = SwingAgent(
        fast_window_size=2,
        slow_window_size=4,
        crossover_threshold_bps=1.0,
        min_confidence=0.0,
    )
    # Slow window includes earlier high ticks; fast window catches recent drop.
    a.observe_tick(_tick(0, 100.0, 102.0))
    a.observe_tick(_tick(1, 100.0, 102.0))
    a.observe_tick(_tick(2, 99.5, 101.5))
    a.observe_tick(_tick(3, 99.0, 101.0))
    trace = a.decide(_signal(4, Side.SELL, conf=0.7))
    assert trace.direction == "SELL"
    assert "momentum_down" in trace.rationale_tags
    assert "ma_crossover_sell" in trace.rationale_tags


def test_low_confidence_signal_downgraded() -> None:
    a = SwingAgent(
        fast_window_size=2,
        slow_window_size=4,
        crossover_threshold_bps=1.0,
        min_confidence=0.5,
    )
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.0, 101.0))
    a.observe_tick(_tick(2, 99.5, 101.5))
    a.observe_tick(_tick(3, 100.0, 102.0))
    trace = a.decide(_signal(4, Side.BUY, conf=0.1))
    assert trace.direction == "HOLD"
    assert "confidence_below_floor" in trace.rationale_tags


def test_neutral_crossover_holds() -> None:
    a = SwingAgent(
        fast_window_size=2,
        slow_window_size=4,
        crossover_threshold_bps=10.0,
        min_confidence=0.0,
    )
    # Almost flat: spread within threshold band.
    for i in range(4):
        a.observe_tick(_tick(i, 100.0, 102.0))
    trace = a.decide(_signal(4, Side.BUY))
    assert trace.direction == "HOLD"
    assert "momentum_neutral" in trace.rationale_tags


def test_recent_decisions_is_bounded_ring() -> None:
    a = SwingAgent(ring_capacity=3)
    for i in range(5):
        a.decide(_signal(i, Side.HOLD))
    recent = a.recent_decisions(10)
    assert len(recent) == 3
    assert all(isinstance(t, AgentDecisionTrace) for t in recent)
    assert [t.ts_ns for t in recent] == [2, 3, 4]


def test_recent_decisions_n_zero() -> None:
    a = SwingAgent()
    a.decide(_signal(0, Side.HOLD))
    assert a.recent_decisions(0) == ()
    assert a.recent_decisions(-1) == ()


def test_recent_decisions_n_partial() -> None:
    a = SwingAgent(ring_capacity=10)
    for i in range(5):
        a.decide(_signal(i, Side.HOLD))
    recent = a.recent_decisions(2)
    assert len(recent) == 2
    assert [t.ts_ns for t in recent] == [3, 4]


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        SwingAgent(fast_window_size=1)
    with pytest.raises(ValueError):
        SwingAgent(fast_window_size=8, slow_window_size=8)
    with pytest.raises(ValueError):
        SwingAgent(fast_window_size=8, slow_window_size=4)
    with pytest.raises(ValueError):
        SwingAgent(crossover_threshold_bps=-0.1)
    with pytest.raises(ValueError):
        SwingAgent(min_confidence=2.0)
    with pytest.raises(ValueError):
        SwingAgent(ring_capacity=0)


def test_invalid_tick_skipped() -> None:
    a = SwingAgent(fast_window_size=2, slow_window_size=4)
    a.observe_tick(MarketTick(ts_ns=0, symbol="X", bid=0.0, ask=1.0, last=1.0))
    a.observe_tick(MarketTick(ts_ns=1, symbol="X", bid=2.0, ask=1.0, last=1.0))
    trace = a.decide(_signal(2, Side.BUY))
    assert trace.direction == "HOLD"


def test_rationale_tags_in_registry_allowlist() -> None:
    """Every rationale tag emitted by the swing agent must be in
    registry/agent_rationale_tags.yaml.
    """
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    doc = yaml.safe_load(
        (repo / "registry" / "agent_rationale_tags.yaml").read_text(
            encoding="utf-8"
        )
    )
    allowed = set(doc.get("tags", []))
    used = {
        "momentum_up",
        "momentum_down",
        "momentum_neutral",
        "ma_crossover_buy",
        "ma_crossover_sell",
        "book_invalid",
        "confidence_below_floor",
    }
    assert used.issubset(allowed), used - allowed
