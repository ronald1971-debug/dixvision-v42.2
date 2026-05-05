"""Tests for SIM-14 partial_fill_chaos (H4.7)."""

from __future__ import annotations

import pytest

from core.contracts.simulation import RealityScenario
from simulation.partial_fill_chaos import (
    PartialFillChaos,
    PartialFillChaosConfig,
)


def _scenario(**meta_overrides: object) -> RealityScenario:
    meta: dict[str, object] = {
        "entry_price": 100.0,
        "order_size_usd": 50_000.0,
        "side": "buy",
        "num_attempts": 5,
        "fill_ratio_mean": 0.5,
        "fill_ratio_std": 0.1,
        "adverse_drift_per_attempt": 0.001,
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="PFC-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = PartialFillChaos().step(seed=42, scenario=s)
    b = PartialFillChaos().step(seed=42, scenario=s)
    assert a == b


def test_full_fill_no_drift_no_cost() -> None:
    s = _scenario(
        fill_ratio_mean=1.0,
        fill_ratio_std=0.0,
        adverse_drift_per_attempt=0.0,
    )
    out = PartialFillChaos().step(seed=0, scenario=s)
    assert out.pnl_usd == 0.0
    assert out.terminal_drawdown_usd == 0.0
    assert out.rule_fired == "fully_filled"
    assert out.fills_count == 1


def test_full_fill_with_drift_costs_zero_first_attempt() -> None:
    # First attempt at idx=0 has adverse=0; if it fully fills,
    # cost is 0 even with drift_per_attempt > 0.
    s = _scenario(
        fill_ratio_mean=1.0,
        fill_ratio_std=0.0,
        adverse_drift_per_attempt=0.01,
    )
    out = PartialFillChaos().step(seed=0, scenario=s)
    assert out.pnl_usd == 0.0
    assert out.rule_fired == "fully_filled"


def test_partial_fills_accumulate_drift_cost() -> None:
    # 50% fill per attempt, 0.001 drift per attempt.
    # remaining: 50000 -> 25000 -> 12500 -> 6250 -> 3125 -> 1562.5
    # fill_usd:        25000   12500   6250   3125   1562.5
    # adverse:         0       0.001   0.002  0.003  0.004
    # cost:            0       12.5    12.5   9.375  6.25
    # total cost = 40.625
    s = _scenario(
        fill_ratio_mean=0.5,
        fill_ratio_std=0.0,
        adverse_drift_per_attempt=0.001,
        num_attempts=5,
    )
    out = PartialFillChaos().step(seed=0, scenario=s)
    assert out.pnl_usd == pytest.approx(-40.625)
    assert out.terminal_drawdown_usd == pytest.approx(40.625)
    assert out.fills_count == 5
    assert out.rule_fired == "incomplete_fill"


def test_pnl_always_non_positive() -> None:
    s = _scenario()
    runner = PartialFillChaos()
    for seed in range(50):
        out = runner.step(seed=seed, scenario=s)
        assert out.pnl_usd <= 0.0
        assert out.terminal_drawdown_usd >= 0.0


def test_zero_drift_zero_cost_regardless_of_chaos() -> None:
    s = _scenario(adverse_drift_per_attempt=0.0)
    runner = PartialFillChaos()
    for seed in range(20):
        out = runner.step(seed=seed, scenario=s)
        assert out.pnl_usd == 0.0


def test_more_attempts_yield_more_complete_fill() -> None:
    # 50% fill rate: residual after N attempts is 50000 * 0.5^N.
    # Need 50000 * 0.5^N <= 1e-6 -> N >= 36.  Use 50 to be safe.
    runner = PartialFillChaos()
    s_few = _scenario(
        fill_ratio_mean=0.5,
        fill_ratio_std=0.0,
        adverse_drift_per_attempt=0.0,
        num_attempts=2,
    )
    s_many = _scenario(
        fill_ratio_mean=0.5,
        fill_ratio_std=0.0,
        adverse_drift_per_attempt=0.0,
        num_attempts=50,
    )
    out_few = runner.step(seed=0, scenario=s_few)
    out_many = runner.step(seed=0, scenario=s_many)
    assert out_few.rule_fired == "incomplete_fill"
    assert out_many.rule_fired == "fully_filled"


def test_outcome_round_trip() -> None:
    s = _scenario()
    out = PartialFillChaos().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_attempts_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(num_attempts=0))
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(num_attempts=-3))
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(num_attempts="five"))


def test_invalid_fill_ratio_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(fill_ratio_mean=1.5))
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(fill_ratio_std=-0.1))
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(fill_ratio_std=0.6))


def test_invalid_drift_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaos().step(
            seed=0, scenario=_scenario(adverse_drift_per_attempt=1.5)
        )
    with pytest.raises(ValueError):
        PartialFillChaos().step(
            seed=0, scenario=_scenario(adverse_drift_per_attempt=-0.1)
        )


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=_scenario(order_size_usd=-1.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        PartialFillChaos().step(seed=0, scenario=bad)


def test_num_attempts_above_cap_rejected() -> None:
    cfg = PartialFillChaosConfig(max_attempts=10)
    with pytest.raises(ValueError):
        PartialFillChaos(cfg).step(seed=0, scenario=_scenario(num_attempts=20))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        PartialFillChaosConfig(max_attempts=0)
    with pytest.raises(ValueError):
        PartialFillChaosConfig(max_attempts=-1)
    with pytest.raises(ValueError):
        PartialFillChaosConfig(residual_epsilon_usd=0.0)
    with pytest.raises(ValueError):
        PartialFillChaosConfig(residual_epsilon_usd=-1.0)


def test_distribution_over_seeds_varies() -> None:
    s = _scenario(fill_ratio_std=0.2, adverse_drift_per_attempt=0.001)
    runner = PartialFillChaos()
    pnls = {
        runner.step(seed=seed, scenario=s).pnl_usd for seed in range(50)
    }
    assert len(pnls) > 1


def test_buy_and_sell_yield_same_cost_structure() -> None:
    # Drift is applied symmetrically against side in this model;
    # cost magnitude is identical for buy vs sell at same seed.
    s_buy = _scenario(side="buy")
    s_sell = _scenario(side="sell")
    out_buy = PartialFillChaos().step(seed=42, scenario=s_buy)
    out_sell = PartialFillChaos().step(seed=42, scenario=s_sell)
    assert out_buy.pnl_usd == pytest.approx(out_sell.pnl_usd)
    assert out_buy.fills_count == out_sell.fills_count
