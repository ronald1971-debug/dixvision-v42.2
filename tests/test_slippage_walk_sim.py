"""Tests for SIM-15 slippage_walk."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.slippage_walk import SlippageWalk, SlippageWalkConfig


def _scenario(
    *,
    scenario_id: str = "S",
    entry_price: float = 100.0,
    order_size_usd: float = 50_000.0,
    side: str = "buy",
    num_legs: int = 20,
    per_leg_drift_mean: float = 0.0,
    per_leg_drift_std: float = 0.001,
    extra: dict[str, Any] | None = None,
) -> RealityScenario:
    meta: dict[str, Any] = {
        "entry_price": entry_price,
        "order_size_usd": order_size_usd,
        "side": side,
        "num_legs": num_legs,
        "per_leg_drift_mean": per_leg_drift_mean,
        "per_leg_drift_std": per_leg_drift_std,
    }
    if extra:
        meta.update(extra)
    return RealityScenario(
        scenario_id=scenario_id,
        ts_ns=42,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = SlippageWalk().step(seed=7, scenario=s)
    b = SlippageWalk().step(seed=7, scenario=s)
    assert a == b


def test_zero_drift_zero_std_pnl_is_zero() -> None:
    s = _scenario(per_leg_drift_mean=0.0, per_leg_drift_std=0.0)
    out = SlippageWalk().step(seed=1, scenario=s)
    assert out.pnl_usd == 0.0
    assert out.terminal_drawdown_usd == 0.0
    assert out.rule_fired == "buy_walk"
    assert out.fills_count == s.meta["num_legs"]


def test_positive_mean_drift_buy_loses_sell_wins() -> None:
    s_buy = _scenario(side="buy", per_leg_drift_mean=0.005, per_leg_drift_std=0.0)
    s_sell = _scenario(side="sell", per_leg_drift_mean=0.005, per_leg_drift_std=0.0)
    buy = SlippageWalk().step(seed=0, scenario=s_buy)
    sell = SlippageWalk().step(seed=0, scenario=s_sell)
    assert buy.pnl_usd < 0.0
    assert sell.pnl_usd > 0.0
    # std=0 so the walk is fully deterministic; same magnitude.
    assert buy.pnl_usd == pytest.approx(-sell.pnl_usd)


def test_negative_mean_drift_buy_wins_sell_loses() -> None:
    s_buy = _scenario(side="buy", per_leg_drift_mean=-0.005, per_leg_drift_std=0.0)
    s_sell = _scenario(side="sell", per_leg_drift_mean=-0.005, per_leg_drift_std=0.0)
    buy = SlippageWalk().step(seed=0, scenario=s_buy)
    sell = SlippageWalk().step(seed=0, scenario=s_sell)
    assert buy.pnl_usd > 0.0
    assert sell.pnl_usd < 0.0


def test_zero_std_compounding_is_geometric() -> None:
    # With std=0 and mean=m, price after N legs = entry * (1+m)^N.
    # delta_sum = leg_size * sum_{i=1..N}((1+m)^i - 1).
    s = _scenario(
        entry_price=100.0,
        order_size_usd=10_000.0,
        side="buy",
        num_legs=4,
        per_leg_drift_mean=0.01,
        per_leg_drift_std=0.0,
    )
    out = SlippageWalk().step(seed=0, scenario=s)
    leg = 10_000.0 / 4
    expected_delta = sum(leg * ((1.01 ** i) - 1.0) for i in range(1, 5))
    assert out.pnl_usd == pytest.approx(-expected_delta)


def test_drawdown_is_negative_pnl_clamped_at_zero() -> None:
    # Adverse for buy => positive drawdown matching abs(pnl).
    bad = _scenario(side="buy", per_leg_drift_mean=0.005, per_leg_drift_std=0.0)
    out = SlippageWalk().step(seed=0, scenario=bad)
    assert out.pnl_usd < 0.0
    assert out.terminal_drawdown_usd == pytest.approx(-out.pnl_usd)
    # Favourable for sell => zero drawdown.
    good = _scenario(side="sell", per_leg_drift_mean=0.005, per_leg_drift_std=0.0)
    out2 = SlippageWalk().step(seed=0, scenario=good)
    assert out2.pnl_usd > 0.0
    assert out2.terminal_drawdown_usd == 0.0


def test_pnl_symmetric_across_sides_at_same_seed() -> None:
    s_buy = _scenario(side="buy", per_leg_drift_std=0.005)
    s_sell = _scenario(side="sell", per_leg_drift_std=0.005)
    buy = SlippageWalk().step(seed=11, scenario=s_buy)
    sell = SlippageWalk().step(seed=11, scenario=s_sell)
    assert buy.pnl_usd == pytest.approx(-sell.pnl_usd)


def test_more_legs_more_dispersion() -> None:
    runner = SlippageWalk()
    pnls_short = [
        runner.step(seed=k, scenario=_scenario(num_legs=4, per_leg_drift_std=0.01)).pnl_usd
        for k in range(60)
    ]
    pnls_long = [
        runner.step(seed=k, scenario=_scenario(num_legs=64, per_leg_drift_std=0.01)).pnl_usd
        for k in range(60)
    ]
    var_short = sum(p * p for p in pnls_short) / len(pnls_short)
    var_long = sum(p * p for p in pnls_long) / len(pnls_long)
    assert var_long > var_short


def test_outcome_round_trip() -> None:
    s = _scenario(scenario_id="round-trip-15")
    out = SlippageWalk().step(seed=3, scenario=s)
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "round-trip-15"
    assert out.seed == 3
    assert out.fills_count == s.meta["num_legs"]


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_legs_rejected() -> None:
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(num_legs=0))
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(num_legs=-3))
    with pytest.raises(ValueError):
        SlippageWalk().step(
            seed=0,
            scenario=_scenario(extra={"num_legs": 3.5}),  # type: ignore[arg-type]
        )


def test_num_legs_above_cap_rejected() -> None:
    cfg = SlippageWalkConfig(max_legs=10)
    with pytest.raises(ValueError):
        SlippageWalk(cfg).step(seed=0, scenario=_scenario(num_legs=20))


def test_invalid_drift_mean_or_std_rejected() -> None:
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(per_leg_drift_mean=0.5))
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(per_leg_drift_mean=-0.5))
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(per_leg_drift_std=-0.01))
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(per_leg_drift_std=1.5))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=_scenario(order_size_usd=-1.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        SlippageWalk().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "entry_price",
        "order_size_usd",
        "per_leg_drift_mean",
        "per_leg_drift_std",
    ):
        with pytest.raises(ValueError):
            SlippageWalk().step(seed=0, scenario=_scenario(extra={key: nan}))


def test_infinity_inputs_rejected() -> None:
    # Regression for Devin Review BUG_pr-review-job-318da2deb_0001:
    # `not v > 0.0` lets +inf through (inf > 0 is True), and
    # `inf - inf = nan` then silently propagates NaN through the
    # geometric walk. Both `entry_price` and `order_size_usd` use
    # the positive-float validator and must reject +inf explicitly.
    inf = float("inf")
    for key in ("entry_price", "order_size_usd"):
        with pytest.raises(ValueError):
            SlippageWalk().step(seed=0, scenario=_scenario(extra={key: inf}))
        with pytest.raises(ValueError):
            SlippageWalk().step(seed=0, scenario=_scenario(extra={key: -inf}))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        SlippageWalkConfig(max_legs=0)
    with pytest.raises(ValueError):
        SlippageWalkConfig(max_legs=-5)


def test_distribution_over_seeds_varies() -> None:
    s = _scenario(per_leg_drift_mean=0.0, per_leg_drift_std=0.01)
    runner = SlippageWalk()
    pnls = {
        runner.step(seed=k, scenario=s).pnl_usd for k in range(50)
    }
    # With random walk and 50 seeds, must produce >> 1 distinct outcomes.
    assert len(pnls) >= 25
