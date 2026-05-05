"""Tests for AGT-05 adversarial / fade-the-crowd agent (INV-54)."""

from __future__ import annotations

import pytest

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from intelligence_engine.agents import AdversarialAgent
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


def test_adversarial_satisfies_introspection_protocol() -> None:
    a = AdversarialAgent()
    assert isinstance(a, AgentIntrospection)


def test_state_snapshot_keys_match_registry() -> None:
    a = AdversarialAgent()
    snap = a.state_snapshot()
    allowlist = AgentBase._load_allowed_state_keys("AGT-05-adversarial")
    assert allowlist, "allowlist must be configured for AGT-05-adversarial"
    assert set(snap.keys()).issubset(allowlist)


def test_state_snapshot_is_pure() -> None:
    a = AdversarialAgent()
    s1 = dict(a.state_snapshot())
    s2 = dict(a.state_snapshot())
    assert s1 == s2


def test_state_snapshot_values_are_strings() -> None:
    a = AdversarialAgent()
    snap = a.state_snapshot()
    for key, value in snap.items():
        assert isinstance(value, str), f"{key} -> {value!r}"


def test_strong_buy_is_faded_to_sell() -> None:
    a = AdversarialAgent(
        fade_threshold=0.5, fade_confidence_scale=0.5, min_confidence=0.0
    )
    trace = a.decide(_signal(1, Side.BUY, conf=0.8))
    assert trace.direction == "SELL"
    assert pytest.approx(trace.confidence) == 0.4
    assert "adversarial_fade_buy" in trace.rationale_tags


def test_strong_sell_is_faded_to_buy() -> None:
    a = AdversarialAgent(
        fade_threshold=0.5, fade_confidence_scale=0.5, min_confidence=0.0
    )
    trace = a.decide(_signal(1, Side.SELL, conf=0.9))
    assert trace.direction == "BUY"
    assert pytest.approx(trace.confidence) == 0.45
    assert "adversarial_fade_sell" in trace.rationale_tags


def test_below_threshold_holds() -> None:
    a = AdversarialAgent(fade_threshold=0.7)
    trace = a.decide(_signal(1, Side.BUY, conf=0.6))
    assert trace.direction == "HOLD"
    assert "adversarial_below_threshold" in trace.rationale_tags
    assert trace.confidence == 0.0


def test_hold_side_not_faded() -> None:
    a = AdversarialAgent(fade_threshold=0.0)
    trace = a.decide(_signal(1, Side.HOLD, conf=0.99))
    assert trace.direction == "HOLD"
    assert "adversarial_below_threshold" in trace.rationale_tags


def test_threshold_boundary_is_inclusive() -> None:
    a = AdversarialAgent(
        fade_threshold=0.5, fade_confidence_scale=1.0, min_confidence=0.0
    )
    trace = a.decide(_signal(1, Side.BUY, conf=0.5))
    assert trace.direction == "SELL"
    assert "adversarial_fade_buy" in trace.rationale_tags


def test_low_confidence_after_scaling_downgraded() -> None:
    """0.6 BUY scaled by 0.1 = 0.06 — below floor of 0.1 → HOLD."""
    a = AdversarialAgent(
        fade_threshold=0.5, fade_confidence_scale=0.1, min_confidence=0.1
    )
    trace = a.decide(_signal(1, Side.BUY, conf=0.6))
    assert trace.direction == "HOLD"
    assert "confidence_below_floor" in trace.rationale_tags


def test_recent_decisions_is_bounded_ring() -> None:
    a = AdversarialAgent(ring_capacity=3)
    for i in range(5):
        a.decide(_signal(i, Side.BUY, conf=0.8))
    recent = a.recent_decisions(10)
    assert len(recent) == 3
    assert all(isinstance(t, AgentDecisionTrace) for t in recent)
    assert [t.ts_ns for t in recent] == [2, 3, 4]


def test_recent_decisions_n_zero() -> None:
    a = AdversarialAgent()
    a.decide(_signal(0, Side.BUY, conf=0.9))
    assert a.recent_decisions(0) == ()
    assert a.recent_decisions(-1) == ()


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        AdversarialAgent(fade_threshold=-0.1)
    with pytest.raises(ValueError):
        AdversarialAgent(fade_threshold=1.5)
    with pytest.raises(ValueError):
        AdversarialAgent(fade_confidence_scale=-0.1)
    with pytest.raises(ValueError):
        AdversarialAgent(fade_confidence_scale=2.0)
    with pytest.raises(ValueError):
        AdversarialAgent(min_confidence=2.0)
    with pytest.raises(ValueError):
        AdversarialAgent(ring_capacity=0)


def test_rationale_tags_in_registry_allowlist() -> None:
    """Every rationale tag emitted by the adversarial agent must be in
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
        "adversarial_fade_buy",
        "adversarial_fade_sell",
        "adversarial_below_threshold",
        "confidence_below_floor",
    }
    assert used.issubset(allowed), used - allowed
