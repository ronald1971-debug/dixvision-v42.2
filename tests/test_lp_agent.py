"""Tests for AGT-04 LP / mean-reversion agent (INV-54)."""

from __future__ import annotations

import pytest

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.agents import LiquidityProviderAgent
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


def test_lp_satisfies_introspection_protocol() -> None:
    a = LiquidityProviderAgent()
    assert isinstance(a, AgentIntrospection)


def test_state_snapshot_keys_match_registry() -> None:
    a = LiquidityProviderAgent()
    snap = a.state_snapshot()
    allowlist = AgentBase._load_allowed_state_keys("AGT-04-lp")
    assert allowlist, "allowlist must be configured for AGT-04-lp"
    assert set(snap.keys()).issubset(allowlist)


def test_state_snapshot_is_pure() -> None:
    a = LiquidityProviderAgent()
    s1 = dict(a.state_snapshot())
    s2 = dict(a.state_snapshot())
    assert s1 == s2


def test_state_snapshot_values_are_strings() -> None:
    a = LiquidityProviderAgent()
    snap = a.state_snapshot()
    for key, value in snap.items():
        assert isinstance(value, str), f"{key} -> {value!r}"


def test_warmup_emits_hold() -> None:
    a = LiquidityProviderAgent(fair_value_window=4)
    a.observe_tick(_tick(0, 99.0, 101.0))
    a.observe_tick(_tick(1, 99.0, 101.0))
    trace = a.decide(_signal(2, Side.BUY))
    assert trace.direction == "HOLD"
    assert "mean_reversion_neutral" in trace.rationale_tags


def test_dip_below_band_buy_signal_passes() -> None:
    a = LiquidityProviderAgent(
        fair_value_window=4,
        band_bps=10.0,
        min_confidence=0.0,
    )
    # Stable around 100; latest tick dips well below the band.
    a.observe_tick(_tick(0, 99.5, 100.5))  # mid 100.0
    a.observe_tick(_tick(1, 99.5, 100.5))  # mid 100.0
    a.observe_tick(_tick(2, 99.5, 100.5))  # mid 100.0
    a.observe_tick(_tick(3, 99.0, 99.4))   # mid 99.2 → ~-50 bps from fair ~99.8
    trace = a.decide(_signal(4, Side.BUY, conf=0.8))
    assert trace.direction == "BUY"
    assert "lp_quote_buy" in trace.rationale_tags


def test_pop_above_band_sell_signal_passes() -> None:
    a = LiquidityProviderAgent(
        fair_value_window=4,
        band_bps=10.0,
        min_confidence=0.0,
    )
    a.observe_tick(_tick(0, 99.5, 100.5))
    a.observe_tick(_tick(1, 99.5, 100.5))
    a.observe_tick(_tick(2, 99.5, 100.5))
    a.observe_tick(_tick(3, 100.6, 101.0))  # mid 100.8 → ~+80 bps
    trace = a.decide(_signal(4, Side.SELL, conf=0.7))
    assert trace.direction == "SELL"
    assert "lp_quote_sell" in trace.rationale_tags


def test_inside_band_holds() -> None:
    a = LiquidityProviderAgent(
        fair_value_window=4,
        band_bps=200.0,  # wide band
        min_confidence=0.0,
    )
    a.observe_tick(_tick(0, 99.5, 100.5))
    a.observe_tick(_tick(1, 99.5, 100.5))
    a.observe_tick(_tick(2, 99.6, 100.6))
    a.observe_tick(_tick(3, 99.7, 100.7))
    trace = a.decide(_signal(4, Side.BUY))
    assert trace.direction == "HOLD"
    assert "mean_reversion_neutral" in trace.rationale_tags


def test_counter_signal_downgraded_on_dip() -> None:
    """Below-band BUY stance should reject incoming SELL signal."""
    a = LiquidityProviderAgent(
        fair_value_window=4,
        band_bps=10.0,
        min_confidence=0.0,
    )
    a.observe_tick(_tick(0, 99.5, 100.5))
    a.observe_tick(_tick(1, 99.5, 100.5))
    a.observe_tick(_tick(2, 99.5, 100.5))
    a.observe_tick(_tick(3, 99.0, 99.4))  # below band
    trace = a.decide(_signal(4, Side.SELL))
    assert trace.direction == "HOLD"


def test_low_confidence_signal_downgraded() -> None:
    a = LiquidityProviderAgent(
        fair_value_window=4,
        band_bps=10.0,
        min_confidence=0.5,
    )
    a.observe_tick(_tick(0, 99.5, 100.5))
    a.observe_tick(_tick(1, 99.5, 100.5))
    a.observe_tick(_tick(2, 99.5, 100.5))
    a.observe_tick(_tick(3, 99.0, 99.4))  # below band
    trace = a.decide(_signal(4, Side.BUY, conf=0.1))
    assert trace.direction == "HOLD"
    assert "confidence_below_floor" in trace.rationale_tags


def test_recent_decisions_is_bounded_ring() -> None:
    a = LiquidityProviderAgent(ring_capacity=3)
    for i in range(5):
        a.decide(_signal(i, Side.HOLD))
    recent = a.recent_decisions(10)
    assert len(recent) == 3
    assert all(isinstance(t, AgentDecisionTrace) for t in recent)
    assert [t.ts_ns for t in recent] == [2, 3, 4]


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        LiquidityProviderAgent(fair_value_window=1)
    with pytest.raises(ValueError):
        LiquidityProviderAgent(band_bps=-0.1)
    with pytest.raises(ValueError):
        LiquidityProviderAgent(min_confidence=2.0)
    with pytest.raises(ValueError):
        LiquidityProviderAgent(ring_capacity=0)


def test_invalid_tick_skipped() -> None:
    a = LiquidityProviderAgent(fair_value_window=2)
    a.observe_tick(MarketTick(ts_ns=0, symbol="X", bid=0.0, ask=1.0, last=1.0))
    a.observe_tick(MarketTick(ts_ns=1, symbol="X", bid=2.0, ask=1.0, last=1.0))
    trace = a.decide(_signal(2, Side.BUY))
    assert trace.direction == "HOLD"


def test_rationale_tags_in_registry_allowlist() -> None:
    """Every rationale tag emitted by the LP agent must be in
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
        "lp_quote_buy",
        "lp_quote_sell",
        "mean_reversion_neutral",
        "book_invalid",
        "confidence_below_floor",
    }
    assert used.issubset(allowed), used - allowed
