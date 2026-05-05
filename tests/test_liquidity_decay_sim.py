"""Tests for SIM-11 liquidity_decay (H4.4)."""

from __future__ import annotations

import pytest

from core.contracts.simulation import RealityScenario
from simulation.liquidity_decay import (
    LiquidityDecay,
    LiquidityDecayConfig,
)


def _scenario(**meta_overrides: object) -> RealityScenario:
    meta: dict[str, object] = {
        "reference_price": 100.0,
        "order_size_usd": 100_000.0,
        "initial_depth_usd": 1_000_000.0,
        "decay_rate": 0.1,
        "num_slices": 10,
        "side": "buy",
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="LD-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = LiquidityDecay().step(seed=42, scenario=s)
    b = LiquidityDecay().step(seed=42, scenario=s)
    assert a == b


def test_no_decay_no_jitter_matches_linear_law() -> None:
    cfg = LiquidityDecayConfig(depth_jitter=0.0)
    s = _scenario(decay_rate=0.0)
    out = LiquidityDecay(cfg).step(seed=0, scenario=s)
    # 10 slices of 10k each, depth constant at 1M, slippage = 10k/1M = 0.01
    # cost = 10 * 10k * 0.01 = 1000
    assert out.pnl_usd == pytest.approx(-1_000.0)
    assert out.terminal_drawdown_usd == pytest.approx(1_000.0)


def test_decay_increases_cost_vs_no_decay() -> None:
    cfg = LiquidityDecayConfig(depth_jitter=0.0)
    no_decay = LiquidityDecay(cfg).step(
        seed=0, scenario=_scenario(decay_rate=0.0)
    )
    with_decay = LiquidityDecay(cfg).step(
        seed=0, scenario=_scenario(decay_rate=0.3)
    )
    assert with_decay.terminal_drawdown_usd > no_decay.terminal_drawdown_usd


def test_more_slices_smooths_cost() -> None:
    # With no decay and zero jitter, total cost depends on slice count
    # via the linear-slippage law: cost_per_slice = slice_size^2 / depth.
    # Cutting slices in half cuts each slippage in half AND each slice
    # size in half, so total cost halves.
    cfg = LiquidityDecayConfig(depth_jitter=0.0)
    s_few = _scenario(decay_rate=0.0, num_slices=10)
    s_many = _scenario(decay_rate=0.0, num_slices=20)
    out_f = LiquidityDecay(cfg).step(seed=0, scenario=s_few)
    out_m = LiquidityDecay(cfg).step(seed=0, scenario=s_many)
    assert out_m.terminal_drawdown_usd < out_f.terminal_drawdown_usd


def test_buy_rule_fired() -> None:
    out = LiquidityDecay().step(seed=0, scenario=_scenario(side="buy"))
    assert out.rule_fired == "buy_decay"


def test_sell_rule_fired() -> None:
    out = LiquidityDecay().step(seed=0, scenario=_scenario(side="sell"))
    assert out.rule_fired == "sell_decay"


def test_pnl_always_non_positive() -> None:
    s = _scenario()
    runner = LiquidityDecay()
    for seed in range(50):
        out = runner.step(seed=seed, scenario=s)
        assert out.pnl_usd <= 0.0
        assert out.terminal_drawdown_usd >= 0.0


def test_fills_count_matches_slices() -> None:
    out = LiquidityDecay().step(seed=0, scenario=_scenario(num_slices=7))
    assert out.fills_count == 7


def test_outcome_round_trip() -> None:
    s = _scenario()
    out = LiquidityDecay().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(side="long"))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"reference_price": 100.0},
    )
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=bad)


def test_invalid_decay_rate_rejected() -> None:
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(decay_rate=1.0))
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(decay_rate=-0.1))


def test_invalid_num_slices_rejected() -> None:
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(num_slices=0))
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(num_slices=-1))
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(num_slices=1.5))


def test_invalid_size_or_depth_rejected() -> None:
    with pytest.raises(ValueError):
        LiquidityDecay().step(seed=0, scenario=_scenario(order_size_usd=0.0))
    with pytest.raises(ValueError):
        LiquidityDecay().step(
            seed=0, scenario=_scenario(initial_depth_usd=-1.0)
        )


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        LiquidityDecayConfig(depth_jitter=-0.1)
    with pytest.raises(ValueError):
        LiquidityDecayConfig(min_depth_usd=0.0)
    with pytest.raises(ValueError):
        LiquidityDecayConfig(max_slices=0)


def test_max_slices_cap_enforced() -> None:
    cfg = LiquidityDecayConfig(max_slices=5)
    with pytest.raises(ValueError):
        LiquidityDecay(cfg).step(seed=0, scenario=_scenario(num_slices=10))


def test_distribution_over_seeds_varies() -> None:
    s = _scenario()
    runner = LiquidityDecay()
    costs = {
        runner.step(seed=seed, scenario=s).terminal_drawdown_usd
        for seed in range(50)
    }
    assert len(costs) > 1


def test_min_depth_floor_prevents_explosion() -> None:
    cfg = LiquidityDecayConfig(depth_jitter=0.0, min_depth_usd=1_000.0)
    s = _scenario(decay_rate=0.99, initial_depth_usd=1_000_000.0)
    out = LiquidityDecay(cfg).step(seed=0, scenario=s)
    # Cost is bounded above by order_size_usd (slippage clamped to [0, 1]).
    assert out.terminal_drawdown_usd <= 100_000.0
