"""Tests for the closed-loop weight adjuster (BEHAVIOR-P2).

Covers:

* WeightAdjustmentConfig validation
* WeightBinding validation
* Pearson correlation (well-defined and undefined cases)
* propose_weight_updates: under-sample, zero variance, perfectly
  correlated, anticorrelated, below-floor, ceiling-saturation,
  floor-saturation, max-nudge clip, multi-binding ordering,
  determinism (replay), and integration with the existing
  UpdateEmitter (LearningUpdate → SystemEvent).
"""

from __future__ import annotations

import math

import pytest

from core.contracts.events import SystemEventKind
from learning_engine.lanes.reward_shaping import RewardBreakdown
from learning_engine.lanes.weight_adjuster import (
    WEIGHT_ADJUSTER_VERSION,
    WeightAdjustment,
    WeightAdjustmentConfig,
    WeightBinding,
    _pearson,  # type: ignore[attr-defined]
    propose_weight_updates,
)
from learning_engine.update_emitter import UpdateEmitter


def _cfg(**overrides) -> WeightAdjustmentConfig:
    base = {
        "learning_rate": 0.1,
        "max_nudge_per_step": 0.05,
        "min_weight": 0.0,
        "max_weight": 1.0,
        "min_samples": 4,
        "correlation_floor": 0.1,
    }
    base.update(overrides)
    return WeightAdjustmentConfig(**base)


def _binding(**overrides) -> WeightBinding:
    base = {
        "parameter": "consensus_weight",
        "component_name": "confidence_consensus",
        "current_value": 0.5,
        "strategy_id": "indira",
    }
    base.update(overrides)
    return WeightBinding(**base)


def _breakdown(
    *,
    ts_ns: int,
    component_value: float,
    shaped_reward: float,
    component_name: str = "confidence_consensus",
) -> RewardBreakdown:
    return RewardBreakdown(
        ts_ns=ts_ns,
        raw_pnl=shaped_reward,
        components=((component_name, component_value),),
        shaped_reward=shaped_reward,
        shaping_version="test",
    )


# ---------------------------------------------------------------------------
# Config + binding validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.1}, "learning_rate"),
        ({"learning_rate": math.inf}, "learning_rate"),
        ({"max_nudge_per_step": 0.0}, "max_nudge_per_step"),
        ({"max_nudge_per_step": -0.1}, "max_nudge_per_step"),
        ({"min_weight": -0.1}, "min_weight"),
        ({"min_weight": math.nan}, "min_weight"),
        ({"min_weight": 0.5, "max_weight": 0.5}, "max_weight"),
        ({"min_weight": 0.5, "max_weight": 0.4}, "max_weight"),
        ({"min_samples": 1}, "min_samples"),
        ({"min_samples": 0}, "min_samples"),
        ({"correlation_floor": -0.1}, "correlation_floor"),
        ({"correlation_floor": 1.5}, "correlation_floor"),
    ],
)
def test_config_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kwargs)


def test_config_default_version_pinned():
    assert _cfg().version == WEIGHT_ADJUSTER_VERSION


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"parameter": ""}, "parameter"),
        ({"component_name": ""}, "component_name"),
        ({"strategy_id": ""}, "strategy_id"),
        ({"current_value": math.nan}, "current_value"),
        ({"current_value": math.inf}, "current_value"),
    ],
)
def test_binding_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _binding(**kwargs)


# ---------------------------------------------------------------------------
# Pearson edge cases
# ---------------------------------------------------------------------------


def test_pearson_perfect_positive():
    assert _pearson([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(
        1.0, abs=1e-12
    )


def test_pearson_perfect_negative():
    assert _pearson([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(
        -1.0, abs=1e-12
    )


def test_pearson_zero_variance_returns_none():
    assert _pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert _pearson([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]) is None


def test_pearson_under_two_samples_returns_none():
    assert _pearson([], []) is None
    assert _pearson([1.0], [1.0]) is None


def test_pearson_clamped_into_unit_interval():
    # Identical sequences should give exactly +1.0 (or as close as
    # float arithmetic allows). The clamp guards against any tiny
    # rounding above 1.
    r = _pearson([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert r is not None
    assert -1.0 <= r <= 1.0
    assert r == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# propose_weight_updates — primary scoring behavior
# ---------------------------------------------------------------------------


def _perfectly_positive_window():
    return [
        _breakdown(ts_ns=1_000, component_value=0.1, shaped_reward=0.1),
        _breakdown(ts_ns=2_000, component_value=0.2, shaped_reward=0.2),
        _breakdown(ts_ns=3_000, component_value=0.3, shaped_reward=0.3),
        _breakdown(ts_ns=4_000, component_value=0.4, shaped_reward=0.4),
    ]


def test_perfect_positive_correlation_proposes_upward_nudge():
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=_perfectly_positive_window(),
        bindings=(_binding(),),
        config=_cfg(),
    )
    assert len(updates) == 1
    assert len(diagnostics) == 1
    diag = diagnostics[0]
    assert diag.proposed is True
    assert diag.correlation == pytest.approx(1.0, abs=1e-12)
    # learning_rate=0.1, |r|=1.0 -> raw_nudge=0.1, clipped to 0.05
    assert diag.raw_nudge == pytest.approx(0.1, abs=1e-12)
    assert diag.clipped_nudge == pytest.approx(0.05, abs=1e-12)
    assert diag.new_value == pytest.approx(0.55, abs=1e-12)
    assert diag.sample_count == 4


def test_perfect_negative_correlation_proposes_downward_nudge():
    breakdowns = [
        _breakdown(ts_ns=1_000, component_value=0.1, shaped_reward=0.4),
        _breakdown(ts_ns=2_000, component_value=0.2, shaped_reward=0.3),
        _breakdown(ts_ns=3_000, component_value=0.3, shaped_reward=0.2),
        _breakdown(ts_ns=4_000, component_value=0.4, shaped_reward=0.1),
    ]
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=(_binding(),),
        config=_cfg(),
    )
    assert len(updates) == 1
    diag = diagnostics[0]
    assert diag.correlation == pytest.approx(-1.0, abs=1e-12)
    assert diag.clipped_nudge == pytest.approx(-0.05, abs=1e-12)
    assert diag.new_value == pytest.approx(0.45, abs=1e-12)


def test_under_sampled_window_returns_no_update():
    breakdowns = [
        _breakdown(ts_ns=1_000, component_value=0.1, shaped_reward=0.1),
        _breakdown(ts_ns=2_000, component_value=0.2, shaped_reward=0.2),
    ]  # 2 samples, min_samples=4
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=(_binding(),),
        config=_cfg(),
    )
    assert updates == ()
    assert len(diagnostics) == 1
    assert diagnostics[0].proposed is False
    assert diagnostics[0].correlation is None
    assert diagnostics[0].sample_count == 2


def test_zero_variance_component_returns_no_update():
    # All component values equal -> Pearson undefined -> no update.
    breakdowns = [
        _breakdown(ts_ns=t, component_value=0.5, shaped_reward=float(t))
        for t in (1_000, 2_000, 3_000, 4_000)
    ]
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=(_binding(),),
        config=_cfg(),
    )
    assert updates == ()
    assert diagnostics[0].proposed is False
    assert diagnostics[0].correlation is None


def test_below_correlation_floor_returns_no_update():
    # Hand-crafted: small but non-zero positive Pearson.
    breakdowns = [
        _breakdown(ts_ns=1_000, component_value=0.1, shaped_reward=1.0),
        _breakdown(ts_ns=2_000, component_value=0.2, shaped_reward=0.9),
        _breakdown(ts_ns=3_000, component_value=0.3, shaped_reward=1.1),
        _breakdown(ts_ns=4_000, component_value=0.4, shaped_reward=1.05),
    ]
    cfg = _cfg(correlation_floor=0.99)
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=(_binding(),),
        config=cfg,
    )
    assert updates == ()
    assert diagnostics[0].proposed is False
    assert diagnostics[0].correlation is not None
    assert abs(diagnostics[0].correlation) < 0.99


# ---------------------------------------------------------------------------
# Bound enforcement (SAFE-65)
# ---------------------------------------------------------------------------


def test_max_nudge_per_step_clips_oversized_raw_step():
    cfg = _cfg(learning_rate=10.0, max_nudge_per_step=0.05)
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=_perfectly_positive_window(),
        bindings=(_binding(current_value=0.5),),
        config=cfg,
    )
    diag = diagnostics[0]
    assert diag.raw_nudge == pytest.approx(10.0, abs=1e-12)
    assert diag.clipped_nudge == pytest.approx(0.05, abs=1e-12)
    assert diag.new_value == pytest.approx(0.55, abs=1e-12)
    assert len(updates) == 1


def test_post_clip_bounds_pin_to_max_weight():
    # Already at the ceiling; positive nudge is bounded to no-op.
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=_perfectly_positive_window(),
        bindings=(_binding(current_value=1.0),),
        config=_cfg(),
    )
    assert updates == ()
    diag = diagnostics[0]
    assert diag.proposed is False
    assert diag.new_value == pytest.approx(1.0, abs=1e-12)
    # Diagnostic still records the math.
    assert diag.correlation == pytest.approx(1.0, abs=1e-12)
    assert diag.clipped_nudge > 0


def test_post_clip_bounds_pin_to_min_weight():
    # Already at the floor; negative nudge is bounded to no-op.
    breakdowns = [
        _breakdown(ts_ns=1_000, component_value=0.1, shaped_reward=0.4),
        _breakdown(ts_ns=2_000, component_value=0.2, shaped_reward=0.3),
        _breakdown(ts_ns=3_000, component_value=0.3, shaped_reward=0.2),
        _breakdown(ts_ns=4_000, component_value=0.4, shaped_reward=0.1),
    ]
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=(_binding(current_value=0.0),),
        config=_cfg(),
    )
    assert updates == ()
    assert diagnostics[0].proposed is False
    assert diagnostics[0].new_value == pytest.approx(0.0, abs=1e-12)


def test_current_value_outside_envelope_rejected():
    # Pre-condition: caller's runtime weight must already be inside
    # [min_weight, max_weight]. Otherwise the loop has been bypassed.
    with pytest.raises(ValueError, match="envelope"):
        propose_weight_updates(
            ts_ns=10**9,
            breakdowns=_perfectly_positive_window(),
            bindings=(_binding(current_value=1.5),),
            config=_cfg(),
        )


def test_negative_ts_rejected():
    with pytest.raises(ValueError, match="ts_ns"):
        propose_weight_updates(
            ts_ns=-1,
            breakdowns=_perfectly_positive_window(),
            bindings=(_binding(),),
            config=_cfg(),
        )


# ---------------------------------------------------------------------------
# Multi-binding behavior
# ---------------------------------------------------------------------------


def test_multi_binding_preserves_input_order():
    breakdowns = [
        RewardBreakdown(
            ts_ns=t,
            raw_pnl=float(t),
            components=(
                ("confidence_consensus", float(t) / 10_000.0),
                # Anti-correlated against shaped_reward.
                ("confidence_strength", 1.0 - float(t) / 10_000.0),
            ),
            shaped_reward=float(t),
            shaping_version="test",
        )
        for t in (1_000, 2_000, 3_000, 4_000)
    ]
    bindings = (
        _binding(parameter="consensus_weight", component_name="confidence_consensus"),
        _binding(parameter="strength_weight", component_name="confidence_strength"),
    )
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=bindings,
        config=_cfg(),
    )
    assert tuple(d.parameter for d in diagnostics) == (
        "consensus_weight",
        "strength_weight",
    )
    assert tuple(u.parameter for u in updates) == (
        "consensus_weight",
        "strength_weight",
    )
    assert diagnostics[0].correlation == pytest.approx(1.0, abs=1e-12)
    assert diagnostics[1].correlation == pytest.approx(-1.0, abs=1e-12)
    assert updates[0].new_value == "0.550000"
    assert updates[1].new_value == "0.450000"


def test_missing_component_skipped_per_breakdown():
    # First two rows lack the named component. The two remaining rows
    # are still under min_samples=4, so no update is proposed.
    breakdowns = [
        RewardBreakdown(
            ts_ns=1_000,
            raw_pnl=0.1,
            components=(("other", 1.0),),
            shaped_reward=0.1,
            shaping_version="test",
        ),
        RewardBreakdown(
            ts_ns=2_000,
            raw_pnl=0.2,
            components=(("other", 2.0),),
            shaped_reward=0.2,
            shaping_version="test",
        ),
        _breakdown(ts_ns=3_000, component_value=0.3, shaped_reward=0.3),
        _breakdown(ts_ns=4_000, component_value=0.4, shaped_reward=0.4),
    ]
    updates, diagnostics = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=breakdowns,
        bindings=(_binding(),),
        config=_cfg(),
    )
    assert updates == ()
    assert diagnostics[0].sample_count == 2
    assert diagnostics[0].proposed is False


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_replay_determinism():
    breakdowns = _perfectly_positive_window()
    bindings = (_binding(),)
    cfg = _cfg()
    a = propose_weight_updates(
        ts_ns=10**9, breakdowns=breakdowns, bindings=bindings, config=cfg
    )
    b = propose_weight_updates(
        ts_ns=10**9, breakdowns=breakdowns, bindings=bindings, config=cfg
    )
    assert a == b


# ---------------------------------------------------------------------------
# UpdateEmitter integration (LearningUpdate → SystemEvent)
# ---------------------------------------------------------------------------


def test_proposed_updates_flow_through_update_emitter():
    updates, _ = propose_weight_updates(
        ts_ns=10**9,
        breakdowns=_perfectly_positive_window(),
        bindings=(_binding(),),
        config=_cfg(),
    )
    emitter = UpdateEmitter(source="learning")
    events = emitter.emit_many(updates)
    assert len(events) == 1
    ev = events[0]
    assert ev.sub_kind == SystemEventKind.UPDATE_PROPOSED
    assert ev.source == "learning"
    assert ev.payload["parameter"] == "consensus_weight"
    assert ev.payload["old_value"] == "0.500000"
    assert ev.payload["new_value"] == "0.550000"
    assert ev.payload["strategy_id"] == "indira"
    # Adjuster diagnostics ride on meta for offline calibration.
    assert ev.meta["adjuster_version"] == WEIGHT_ADJUSTER_VERSION
    assert ev.meta["component"] == "confidence_consensus"
    assert ev.meta["sample_count"] == "4"
    assert float(ev.meta["correlation"]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Frozen-dataclass guarantees
# ---------------------------------------------------------------------------


def test_weight_adjustment_is_frozen():
    diag = WeightAdjustment(
        parameter="consensus_weight",
        component_name="confidence_consensus",
        sample_count=4,
        correlation=1.0,
        raw_nudge=0.1,
        clipped_nudge=0.05,
        old_value=0.5,
        new_value=0.55,
        proposed=True,
    )
    with pytest.raises((AttributeError, TypeError)):
        diag.new_value = 999  # type: ignore[misc]


def test_config_is_frozen():
    cfg = _cfg()
    with pytest.raises((AttributeError, TypeError)):
        cfg.learning_rate = 999  # type: ignore[misc]


def test_binding_is_frozen():
    binding = _binding()
    with pytest.raises((AttributeError, TypeError)):
        binding.current_value = 999  # type: ignore[misc]
