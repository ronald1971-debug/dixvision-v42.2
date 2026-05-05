"""Tests for SIM-13 latency_jitter (H4.6)."""

from __future__ import annotations

import pytest

from core.contracts.simulation import RealityScenario
from simulation.latency_jitter import (
    LatencyJitter,
    LatencyJitterConfig,
)


def _scenario(**meta_overrides: object) -> RealityScenario:
    meta: dict[str, object] = {
        "entry_price": 100.0,
        "order_size_usd": 50_000.0,
        "side": "buy",
        "expected_latency_ms": 50.0,
        "jitter_std_ms": 10.0,
        "price_drift_per_ms": 0.0001,
        "price_volatility": 0.001,
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="LJ-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = LatencyJitter().step(seed=42, scenario=s)
    b = LatencyJitter().step(seed=42, scenario=s)
    assert a == b


def test_no_drift_no_volatility_pnl_zero() -> None:
    s = _scenario(price_drift_per_ms=0.0, price_volatility=0.0)
    out = LatencyJitter().step(seed=0, scenario=s)
    assert out.pnl_usd == 0.0
    assert out.terminal_drawdown_usd == 0.0


def test_buy_with_positive_drift_loses_money() -> None:
    s = _scenario(
        side="buy",
        price_drift_per_ms=0.001,
        price_volatility=0.0,
        jitter_std_ms=0.0,
    )
    # drift = 0.001 * 50ms = 0.05; buy pnl = -50000 * 0.05 = -2500.
    out = LatencyJitter().step(seed=0, scenario=s)
    assert out.pnl_usd == pytest.approx(-2_500.0)
    assert out.terminal_drawdown_usd == pytest.approx(2_500.0)
    assert out.rule_fired == "buy_jitter"


def test_sell_with_positive_drift_makes_money() -> None:
    s = _scenario(
        side="sell",
        price_drift_per_ms=0.001,
        price_volatility=0.0,
        jitter_std_ms=0.0,
    )
    out = LatencyJitter().step(seed=0, scenario=s)
    assert out.pnl_usd == pytest.approx(2_500.0)
    assert out.terminal_drawdown_usd == 0.0
    assert out.rule_fired == "sell_jitter"


def test_buy_and_sell_pnl_are_opposite_sign() -> None:
    s_buy = _scenario(side="buy")
    s_sell = _scenario(side="sell")
    out_buy = LatencyJitter().step(seed=42, scenario=s_buy)
    out_sell = LatencyJitter().step(seed=42, scenario=s_sell)
    # Same scenario_id, same seed → same RNG draws → opposite signs.
    assert out_buy.pnl_usd == pytest.approx(-out_sell.pnl_usd)


def test_high_jitter_increases_pnl_variance() -> None:
    s_low = _scenario(jitter_std_ms=0.0, price_volatility=0.001)
    s_high = _scenario(jitter_std_ms=20.0, price_volatility=0.001)
    runner = LatencyJitter()
    pnls_low = [
        runner.step(seed=seed, scenario=s_low).pnl_usd for seed in range(50)
    ]
    pnls_high = [
        runner.step(seed=seed, scenario=s_high).pnl_usd for seed in range(50)
    ]
    var_low = sum((p - sum(pnls_low) / 50) ** 2 for p in pnls_low) / 50
    var_high = sum((p - sum(pnls_high) / 50) ** 2 for p in pnls_high) / 50
    assert var_high > var_low


def test_outcome_round_trip() -> None:
    s = _scenario()
    out = LatencyJitter().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99
    assert out.fills_count == 1


def test_drawdown_non_negative() -> None:
    s = _scenario(jitter_std_ms=20.0, price_volatility=0.005)
    runner = LatencyJitter()
    for seed in range(100):
        out = runner.step(seed=seed, scenario=s)
        assert out.terminal_drawdown_usd >= 0.0


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_latency_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(expected_latency_ms=0.0))
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(jitter_std_ms=-1.0))


def test_invalid_drift_or_volatility_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(price_drift_per_ms=1.5))
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(price_volatility=-0.1))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=_scenario(order_size_usd=-1.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        LatencyJitter().step(seed=0, scenario=bad)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyJitterConfig(max_latency_ms=0.0)
    with pytest.raises(ValueError):
        LatencyJitterConfig(max_latency_ms=-1.0)


def test_max_latency_cap_bounds_drift() -> None:
    cfg = LatencyJitterConfig(max_latency_ms=10.0)
    s = _scenario(
        expected_latency_ms=10_000.0,
        jitter_std_ms=0.0,
        price_drift_per_ms=0.001,
        price_volatility=0.0,
    )
    # Without cap drift would be 10*1000 -> clamped at 1.0.
    # With cap 10ms drift = 0.001 * 10 = 0.01.
    out = LatencyJitter(cfg).step(seed=0, scenario=s)
    assert out.pnl_usd == pytest.approx(-500.0)


def test_drift_clamp_to_unit_interval() -> None:
    s = _scenario(
        price_drift_per_ms=0.5,
        price_volatility=0.0,
        jitter_std_ms=0.0,
        expected_latency_ms=100.0,
    )
    # base_drift = 50.0, clamped to 1.0; pnl = -size * 1.0.
    out = LatencyJitter().step(seed=0, scenario=s)
    assert out.pnl_usd == pytest.approx(-50_000.0)


def test_distribution_over_seeds_varies() -> None:
    s = _scenario(jitter_std_ms=10.0, price_volatility=0.001)
    runner = LatencyJitter()
    pnls = {
        runner.step(seed=seed, scenario=s).pnl_usd for seed in range(50)
    }
    assert len(pnls) > 1
