"""Tests for SIM-22 oracle_lag — H4.15 of the canonical-rebuild walk."""

from __future__ import annotations

import math
from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.oracle_lag import OracleLag, OracleLagConfig


def _scenario(
    *,
    scenario_id: str = "S",
    extra: dict[str, Any] | None = None,
    **overrides: Any,
) -> RealityScenario:
    meta: dict[str, Any] = {
        "entry_price": 100.0,
        "order_size_usd": 10_000.0,
        "side": "buy",
        "num_steps": 100,
        "oracle_lag_steps": 10,
        "per_step_drift": 0.0,
        "per_step_std": 0.005,
        "oracle_noise_bps": 5.0,
    }
    meta.update(overrides)
    if extra is not None:
        meta.update(extra)
    return RealityScenario(
        scenario_id=scenario_id,
        ts_ns=1,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    sim = OracleLag()
    scenario = _scenario()
    a = sim.step(seed=7, scenario=scenario)
    b = sim.step(seed=7, scenario=scenario)
    assert a == b


def test_outcome_round_trip() -> None:
    sim = OracleLag()
    out = sim.step(seed=1, scenario=_scenario())
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "S"
    assert out.seed == 1


def test_zero_lag_yields_zero_blindspot_modulo_noise() -> None:
    """With oracle_lag_steps=0 and oracle_noise_bps=0 the oracle is
    perfectly fresh, so perceived pnl == actual pnl and the
    blindspot is zero."""
    out = OracleLag().step(
        seed=0,
        scenario=_scenario(oracle_lag_steps=0, oracle_noise_bps=0.0),
    )
    assert out.terminal_drawdown_usd == 0.0
    assert out.fills_count == 0
    assert out.rule_fired == "fresh"


def test_fills_count_encodes_lag_depth() -> None:
    """fills_count is the SIM-22 lag-depth projection."""
    for lag in (0, 5, 25, 60):
        out = OracleLag().step(
            seed=0,
            scenario=_scenario(num_steps=100, oracle_lag_steps=lag),
        )
        assert out.fills_count == lag


def test_rule_fired_thresholds() -> None:
    """rule_fired classifies lag-ratio into 4 buckets."""
    sim = OracleLag()
    cases = [
        (4, "fresh"),       # 4/100 = 0.04 < 0.05
        (5, "slight_lag"),  # 5/100 = 0.05 (>= 0.05, < 0.20)
        (19, "slight_lag"),
        (20, "moderate_lag"),  # 20/100 = 0.20
        (49, "moderate_lag"),
        (50, "severe_lag"),    # 50/100 = 0.50
        (100, "severe_lag"),
    ]
    for lag, expected in cases:
        out = sim.step(
            seed=0,
            scenario=_scenario(num_steps=100, oracle_lag_steps=lag),
        )
        assert out.rule_fired == expected, (lag, expected, out.rule_fired)


def test_pnl_sign_by_side_buy_gain_equals_sell_loss() -> None:
    """At the same seed, buy pnl and sell pnl must be exact negatives:
    they share the underlying true_path."""
    buy = OracleLag().step(
        seed=42,
        scenario=_scenario(side="buy", per_step_drift=0.001, oracle_noise_bps=0.0),
    )
    sell = OracleLag().step(
        seed=42,
        scenario=_scenario(
            scenario_id="S",
            side="sell",
            per_step_drift=0.001,
            oracle_noise_bps=0.0,
        ),
    )
    assert buy.pnl_usd == pytest.approx(-sell.pnl_usd, abs=1e-9)


def test_blindspot_grows_with_lag() -> None:
    """Larger lag → more disagreement between true terminal and
    oracle terminal → larger blindspot in expectation."""
    sim = OracleLag()
    blindspots: dict[int, list[float]] = {0: [], 50: []}
    for seed in range(40):
        for lag in (0, 50):
            out = sim.step(
                seed=seed,
                scenario=_scenario(
                    scenario_id=f"sweep_{seed}",
                    num_steps=100,
                    oracle_lag_steps=lag,
                    per_step_std=0.01,
                    oracle_noise_bps=0.0,
                ),
            )
            blindspots[lag].append(out.terminal_drawdown_usd)
    avg_low = sum(blindspots[0]) / len(blindspots[0])
    avg_high = sum(blindspots[50]) / len(blindspots[50])
    assert avg_high > avg_low * 5.0, (avg_low, avg_high)


def test_blindspot_non_negative() -> None:
    sim = OracleLag()
    for seed in range(30):
        out = sim.step(
            seed=seed,
            scenario=_scenario(scenario_id=f"nn_{seed}"),
        )
        assert out.terminal_drawdown_usd >= 0.0


def test_distribution_over_seeds_varies() -> None:
    sim = OracleLag()
    pnls = [
        sim.step(
            seed=seed,
            scenario=_scenario(scenario_id=f"d_{seed}", per_step_std=0.01),
        ).pnl_usd
        for seed in range(40)
    ]
    spread = max(pnls) - min(pnls)
    assert spread > 50.0, spread


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError, match="seed"):
        OracleLag().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError, match="side"):
        OracleLag().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_steps_rejected() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        OracleLag().step(seed=0, scenario=_scenario(num_steps=0))
    with pytest.raises(ValueError, match="num_steps"):
        OracleLag().step(seed=0, scenario=_scenario(num_steps=-5))


def test_num_steps_above_cap_rejected() -> None:
    sim = OracleLag(OracleLagConfig(max_steps=10))
    with pytest.raises(ValueError, match="exceeds max_steps"):
        sim.step(seed=0, scenario=_scenario(num_steps=11))


def test_oracle_lag_steps_above_num_steps_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed num_steps"):
        OracleLag().step(
            seed=0,
            scenario=_scenario(num_steps=10, oracle_lag_steps=11),
        )


def test_invalid_drift_or_std_rejected() -> None:
    with pytest.raises(ValueError, match="per_step_drift"):
        OracleLag().step(seed=0, scenario=_scenario(per_step_drift=0.1))
    with pytest.raises(ValueError, match="per_step_std"):
        OracleLag().step(seed=0, scenario=_scenario(per_step_std=0.5))
    with pytest.raises(ValueError, match="per_step_std"):
        OracleLag().step(seed=0, scenario=_scenario(per_step_std=-0.001))


def test_invalid_noise_rejected() -> None:
    with pytest.raises(ValueError, match="oracle_noise_bps"):
        OracleLag().step(seed=0, scenario=_scenario(oracle_noise_bps=-1.0))
    with pytest.raises(ValueError, match="oracle_noise_bps"):
        OracleLag().step(seed=0, scenario=_scenario(oracle_noise_bps=200.0))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError, match="entry_price"):
        OracleLag().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError, match="entry_price"):
        OracleLag().step(seed=0, scenario=_scenario(entry_price=-1.0))
    with pytest.raises(ValueError, match="order_size_usd"):
        OracleLag().step(seed=0, scenario=_scenario(order_size_usd=0.0))


def test_missing_meta_keys_rejected() -> None:
    base = _scenario()
    for k in (
        "entry_price",
        "order_size_usd",
        "side",
        "num_steps",
        "oracle_lag_steps",
        "per_step_drift",
        "per_step_std",
        "oracle_noise_bps",
    ):
        meta = dict(base.meta)
        meta.pop(k)
        bad = RealityScenario(
            scenario_id="S", ts_ns=1, initial_state_hash="h", meta=meta
        )
        with pytest.raises(ValueError, match=k):
            OracleLag().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "entry_price",
        "order_size_usd",
        "per_step_drift",
        "per_step_std",
        "oracle_noise_bps",
    ):
        with pytest.raises(ValueError):
            OracleLag().step(seed=0, scenario=_scenario(extra={key: nan}))


def test_infinity_inputs_rejected() -> None:
    inf = math.inf
    for key in (
        "entry_price",
        "order_size_usd",
        "per_step_drift",
        "per_step_std",
        "oracle_noise_bps",
    ):
        with pytest.raises(ValueError):
            OracleLag().step(seed=0, scenario=_scenario(extra={key: inf}))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        OracleLagConfig(max_steps=0)
    with pytest.raises(ValueError):
        OracleLagConfig(max_steps=-1)
    with pytest.raises(ValueError):
        OracleLagConfig(max_steps=2_000_000)


def test_overflow_walk_raises_value_error() -> None:
    """Inf-overflow guard from PR #268 / SIM-20: a custom config +
    drift+std combination can compound past float64 range. Rather
    than silently emit NaN / inf in pnl_usd we fail fast."""
    sim = OracleLag(OracleLagConfig(max_steps=200_000))
    with pytest.raises(ValueError, match="non-finite"):
        sim.step(
            seed=0,
            scenario=_scenario(
                num_steps=200_000,
                oracle_lag_steps=0,
                per_step_drift=0.005,
                per_step_std=0.0,
                oracle_noise_bps=0.0,
            ),
        )
