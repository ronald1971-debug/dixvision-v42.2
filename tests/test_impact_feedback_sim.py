"""Tests for SIM-10 impact_feedback (H4.3)."""

from __future__ import annotations

import math

import pytest

from core.contracts.simulation import RealityScenario
from simulation.impact_feedback import (
    ImpactFeedback,
    ImpactFeedbackConfig,
)


def _scenario(**meta_overrides: object) -> RealityScenario:
    meta: dict[str, object] = {
        "reference_price": 100.0,
        "order_size_usd": 100_000.0,
        "liquidity_depth_usd": 1_000_000.0,
        "side": "buy",
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="IF-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = ImpactFeedback().step(seed=42, scenario=s)
    b = ImpactFeedback().step(seed=42, scenario=s)
    assert a == b


def test_no_jitter_matches_sqrt_law() -> None:
    cfg = ImpactFeedbackConfig(impact_jitter=0.0, impact_coef=0.1)
    out = ImpactFeedback(cfg).step(seed=0, scenario=_scenario())
    # ratio = 0.1, sqrt = ~0.3162, slippage = 0.0316
    expected_slippage = 0.1 * math.sqrt(0.1)
    expected_cost = 100_000.0 * expected_slippage
    assert out.pnl_usd == pytest.approx(-expected_cost)
    assert out.terminal_drawdown_usd == pytest.approx(expected_cost)


def test_buy_rule_fired() -> None:
    out = ImpactFeedback().step(seed=0, scenario=_scenario(side="buy"))
    assert out.rule_fired == "buy_impact"


def test_sell_rule_fired() -> None:
    out = ImpactFeedback().step(seed=0, scenario=_scenario(side="sell"))
    assert out.rule_fired == "sell_impact"


def test_pnl_is_always_non_positive() -> None:
    s = _scenario()
    runner = ImpactFeedback()
    for seed in range(50):
        out = runner.step(seed=seed, scenario=s)
        assert out.pnl_usd <= 0.0
        assert out.terminal_drawdown_usd >= 0.0


def test_larger_size_means_larger_cost_under_clean_config() -> None:
    cfg = ImpactFeedbackConfig(impact_jitter=0.0)
    s_small = _scenario(order_size_usd=10_000.0)
    s_big = _scenario(order_size_usd=400_000.0)
    out_s = ImpactFeedback(cfg).step(seed=0, scenario=s_small)
    out_b = ImpactFeedback(cfg).step(seed=0, scenario=s_big)
    assert out_b.terminal_drawdown_usd > out_s.terminal_drawdown_usd


def test_thicker_depth_means_smaller_cost_under_clean_config() -> None:
    cfg = ImpactFeedbackConfig(impact_jitter=0.0)
    s_thin = _scenario(liquidity_depth_usd=200_000.0)
    s_thick = _scenario(liquidity_depth_usd=10_000_000.0)
    out_t = ImpactFeedback(cfg).step(seed=0, scenario=s_thin)
    out_k = ImpactFeedback(cfg).step(seed=0, scenario=s_thick)
    assert out_t.terminal_drawdown_usd > out_k.terminal_drawdown_usd


def test_max_ratio_caps_slippage() -> None:
    cfg = ImpactFeedbackConfig(
        impact_jitter=0.0, impact_coef=0.5, max_ratio=4.0
    )
    s = _scenario(
        order_size_usd=100_000_000.0, liquidity_depth_usd=1_000_000.0
    )
    out = ImpactFeedback(cfg).step(seed=0, scenario=s)
    # ratio capped at 4.0; slippage = 0.5 * sqrt(4) = 1.0; cost = size * 1.0
    assert out.terminal_drawdown_usd == pytest.approx(100_000_000.0)


def test_outcome_fields_round_trip() -> None:
    s = _scenario()
    out = ImpactFeedback().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99
    assert out.fills_count == 1


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        ImpactFeedback().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        ImpactFeedback().step(seed=0, scenario=_scenario(side="long"))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"reference_price": 100.0},
    )
    with pytest.raises(ValueError):
        ImpactFeedback().step(seed=0, scenario=bad)


def test_zero_or_negative_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        ImpactFeedback().step(seed=0, scenario=_scenario(order_size_usd=0.0))
    with pytest.raises(ValueError):
        ImpactFeedback().step(
            seed=0, scenario=_scenario(liquidity_depth_usd=-1.0)
        )
    with pytest.raises(ValueError):
        ImpactFeedback().step(seed=0, scenario=_scenario(reference_price=0.0))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ImpactFeedbackConfig(impact_coef=0.0)
    with pytest.raises(ValueError):
        ImpactFeedbackConfig(impact_coef=1.5)
    with pytest.raises(ValueError):
        ImpactFeedbackConfig(impact_jitter=-0.1)
    with pytest.raises(ValueError):
        ImpactFeedbackConfig(max_ratio=0.0)
    with pytest.raises(ValueError):
        ImpactFeedbackConfig(max_ratio=200.0)


def test_seed_independence_from_meta() -> None:
    s1 = _scenario()
    s2 = _scenario(reference_price=200.0)
    a = ImpactFeedback().step(seed=7, scenario=s1)
    b = ImpactFeedback().step(seed=7, scenario=s2)
    # Same seed and scenario_id -> same slippage fraction; cost scales
    # only with order_size_usd (which is unchanged here).
    assert a.terminal_drawdown_usd == pytest.approx(b.terminal_drawdown_usd)


def test_distribution_over_seeds_varies() -> None:
    s = _scenario()
    runner = ImpactFeedback()
    costs = {
        runner.step(seed=seed, scenario=s).terminal_drawdown_usd
        for seed in range(50)
    }
    # With non-zero impact_jitter, costs vary across seeds.
    assert len(costs) > 1


def test_buy_and_sell_have_same_cost_magnitude() -> None:
    cfg = ImpactFeedbackConfig(impact_jitter=0.0)
    out_buy = ImpactFeedback(cfg).step(seed=0, scenario=_scenario(side="buy"))
    out_sell = ImpactFeedback(cfg).step(seed=0, scenario=_scenario(side="sell"))
    assert out_buy.pnl_usd == pytest.approx(out_sell.pnl_usd)
    assert (
        out_buy.terminal_drawdown_usd
        == pytest.approx(out_sell.terminal_drawdown_usd)
    )


def test_sell_with_slippage_clamped_to_one_does_not_crash() -> None:
    """Regression: sell side at full slippage hits avg_fill = 0.0.

    When slippage clamps to 1.0 (size >= max_ratio * depth at high
    impact_coef), the sell branch computes ``avg_fill = ref * (1 -
    slippage) = 0.0``. The internal invariant must allow this — the
    price went to zero, which is catastrophic but valid — only a
    strictly negative avg_fill would indicate a logic bug.
    """
    cfg = ImpactFeedbackConfig(
        impact_coef=0.5, max_ratio=4.0, impact_jitter=0.0
    )
    s = _scenario(
        order_size_usd=100_000_000.0,
        liquidity_depth_usd=1_000_000.0,
        side="sell",
    )
    out = ImpactFeedback(cfg).step(seed=0, scenario=s)
    assert out.terminal_drawdown_usd == pytest.approx(100_000_000.0)
    assert out.pnl_usd == pytest.approx(-100_000_000.0)
    assert out.rule_fired == "sell_impact"
