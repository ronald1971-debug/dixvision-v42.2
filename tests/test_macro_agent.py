"""Tests for AGT-03 macro agent (INV-54)."""

from __future__ import annotations

import pytest

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from core.contracts.macro_regime import MacroRegime
from intelligence_engine.agents import MacroAgent
from intelligence_engine.agents._base import AgentBase


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


def test_macro_satisfies_introspection_protocol() -> None:
    a = MacroAgent()
    assert isinstance(a, AgentIntrospection)


def test_state_snapshot_keys_match_registry() -> None:
    a = MacroAgent()
    snap = a.state_snapshot()
    allowlist = AgentBase._load_allowed_state_keys("AGT-03-macro")
    assert allowlist, "allowlist must be configured for AGT-03-macro"
    assert set(snap.keys()).issubset(allowlist)


def test_state_snapshot_is_pure() -> None:
    a = MacroAgent()
    s1 = dict(a.state_snapshot())
    s2 = dict(a.state_snapshot())
    assert s1 == s2


def test_state_snapshot_values_are_strings() -> None:
    a = MacroAgent()
    snap = a.state_snapshot()
    for key, value in snap.items():
        assert isinstance(value, str), f"{key} -> {value!r}"


def test_unknown_regime_holds() -> None:
    a = MacroAgent()
    trace = a.decide(_signal(1, Side.BUY))
    assert trace.direction == "HOLD"
    assert "regime_unknown" in trace.rationale_tags
    assert trace.confidence == 0.0


def test_crisis_regime_holds_all() -> None:
    a = MacroAgent()
    a.observe_regime(MacroRegime.CRISIS)
    for side in (Side.BUY, Side.SELL, Side.HOLD):
        trace = a.decide(_signal(1, side))
        assert trace.direction == "HOLD"
        assert "regime_crisis" in trace.rationale_tags


def test_risk_on_passes_buy_blocks_sell() -> None:
    a = MacroAgent(min_confidence=0.0)
    a.observe_regime(MacroRegime.RISK_ON)
    buy = a.decide(_signal(1, Side.BUY, conf=0.8))
    sell = a.decide(_signal(2, Side.SELL, conf=0.8))
    assert buy.direction == "BUY"
    assert pytest.approx(buy.confidence) == 0.8
    assert "regime_risk_on" in buy.rationale_tags
    assert "macro_aligned_buy" in buy.rationale_tags
    assert sell.direction == "HOLD"


def test_risk_off_passes_sell_blocks_buy() -> None:
    a = MacroAgent(min_confidence=0.0)
    a.observe_regime(MacroRegime.RISK_OFF)
    sell = a.decide(_signal(1, Side.SELL, conf=0.7))
    buy = a.decide(_signal(2, Side.BUY, conf=0.7))
    assert sell.direction == "SELL"
    assert pytest.approx(sell.confidence) == 0.7
    assert "regime_risk_off" in sell.rationale_tags
    assert "macro_aligned_sell" in sell.rationale_tags
    assert buy.direction == "HOLD"


def test_neutral_regime_scales_confidence() -> None:
    a = MacroAgent(min_confidence=0.0, neutral_confidence_scale=0.5)
    a.observe_regime(MacroRegime.NEUTRAL)
    buy = a.decide(_signal(1, Side.BUY, conf=0.8))
    sell = a.decide(_signal(2, Side.SELL, conf=0.6))
    assert buy.direction == "BUY"
    assert pytest.approx(buy.confidence) == 0.4
    assert sell.direction == "SELL"
    assert pytest.approx(sell.confidence) == 0.3
    assert "regime_neutral" in buy.rationale_tags


def test_regime_changes_take_effect_on_next_decide() -> None:
    a = MacroAgent(min_confidence=0.0)
    a.observe_regime(MacroRegime.RISK_OFF)
    t1 = a.decide(_signal(1, Side.BUY))
    assert t1.direction == "HOLD"
    a.observe_regime(MacroRegime.RISK_ON)
    t2 = a.decide(_signal(2, Side.BUY, conf=0.6))
    assert t2.direction == "BUY"


def test_low_confidence_signal_downgraded() -> None:
    a = MacroAgent(min_confidence=0.5)
    a.observe_regime(MacroRegime.RISK_ON)
    trace = a.decide(_signal(1, Side.BUY, conf=0.1))
    assert trace.direction == "HOLD"
    assert "confidence_below_floor" in trace.rationale_tags


def test_neutral_low_confidence_after_scaling_downgraded() -> None:
    # 0.4 BUY scaled to 0.2 under NEUTRAL — below floor of 0.3.
    a = MacroAgent(min_confidence=0.3, neutral_confidence_scale=0.5)
    a.observe_regime(MacroRegime.NEUTRAL)
    trace = a.decide(_signal(1, Side.BUY, conf=0.4))
    assert trace.direction == "HOLD"
    assert "confidence_below_floor" in trace.rationale_tags


def test_recent_decisions_is_bounded_ring() -> None:
    a = MacroAgent(ring_capacity=3)
    for i in range(5):
        a.decide(_signal(i, Side.HOLD))
    recent = a.recent_decisions(10)
    assert len(recent) == 3
    assert all(isinstance(t, AgentDecisionTrace) for t in recent)
    assert [t.ts_ns for t in recent] == [2, 3, 4]


def test_recent_decisions_n_zero() -> None:
    a = MacroAgent()
    a.decide(_signal(0, Side.HOLD))
    assert a.recent_decisions(0) == ()
    assert a.recent_decisions(-1) == ()


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        MacroAgent(neutral_confidence_scale=-0.1)
    with pytest.raises(ValueError):
        MacroAgent(neutral_confidence_scale=1.5)
    with pytest.raises(ValueError):
        MacroAgent(min_confidence=2.0)
    with pytest.raises(ValueError):
        MacroAgent(ring_capacity=0)


def test_rationale_tags_in_registry_allowlist() -> None:
    """Every rationale tag emitted by the macro agent must be in
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
        "regime_risk_on",
        "regime_risk_off",
        "regime_neutral",
        "regime_crisis",
        "regime_unknown",
        "macro_aligned_buy",
        "macro_aligned_sell",
        "confidence_below_floor",
    }
    assert used.issubset(allowed), used - allowed
