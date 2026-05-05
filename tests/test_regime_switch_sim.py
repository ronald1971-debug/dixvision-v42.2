"""Tests for SIM-17 regime_switch_sim — H4.10 of the canonical-rebuild walk."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.regime_switch_sim import (
    RegimeSwitchSim,
    RegimeSwitchSimConfig,
)


def _scenario(
    *,
    scenario_id: str = "S",
    extra: dict[str, Any] | None = None,
    **overrides: Any,
) -> RealityScenario:
    meta = {
        "entry_price": 100.0,
        "order_size_usd": 10_000.0,
        "side": "buy",
        "num_steps": 40,
        "switch_probability": 0.05,
        "regime_a_drift": 0.001,
        "regime_a_std": 0.01,
        "regime_b_drift": -0.001,
        "regime_b_std": 0.02,
        "starting_regime": "A",
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
    sim = RegimeSwitchSim()
    a = sim.step(seed=99, scenario=_scenario())
    b = sim.step(seed=99, scenario=_scenario())
    assert a == b


def test_outcome_round_trip() -> None:
    out = RegimeSwitchSim().step(seed=7, scenario=_scenario())
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "S"
    assert out.seed == 7
    assert out.fills_count == 40


def test_zero_switch_probability_stays_in_starting_regime() -> None:
    out_a = RegimeSwitchSim().step(
        seed=1,
        scenario=_scenario(switch_probability=0.0, starting_regime="A"),
    )
    out_b = RegimeSwitchSim().step(
        seed=1,
        scenario=_scenario(switch_probability=0.0, starting_regime="B"),
    )
    assert out_a.rule_fired == "stable_a"
    assert out_b.rule_fired == "stable_b"


def test_full_switch_probability_flips_every_step() -> None:
    out = RegimeSwitchSim().step(
        seed=3,
        scenario=_scenario(switch_probability=1.0, num_steps=40),
    )
    assert out.rule_fired == "switching_many"


def test_pnl_sign_by_side_matches_walk_direction() -> None:
    # Push the walk strongly upward via positive drift in both regimes.
    sc = _scenario(
        regime_a_drift=0.005,
        regime_a_std=0.0,
        regime_b_drift=0.005,
        regime_b_std=0.0,
        switch_probability=0.0,
    )
    buy = RegimeSwitchSim().step(seed=11, scenario=sc)
    sell_meta = dict(sc.meta)
    sell_meta["side"] = "sell"
    sc_sell = RealityScenario(
        scenario_id=sc.scenario_id,
        ts_ns=sc.ts_ns,
        initial_state_hash=sc.initial_state_hash,
        meta=sell_meta,
    )
    sell = RegimeSwitchSim().step(seed=11, scenario=sc_sell)
    assert buy.pnl_usd > 0.0
    assert sell.pnl_usd < 0.0
    assert pytest.approx(buy.pnl_usd, rel=1e-9) == -sell.pnl_usd


def test_drawdown_non_negative_and_matches_negative_pnl() -> None:
    out = RegimeSwitchSim().step(seed=5, scenario=_scenario())
    assert out.terminal_drawdown_usd >= 0.0
    if out.pnl_usd < 0.0:
        assert pytest.approx(out.terminal_drawdown_usd, rel=1e-9) == -out.pnl_usd
    else:
        assert out.terminal_drawdown_usd == 0.0


def test_fills_count_equals_num_steps() -> None:
    for n in (1, 10, 100):
        out = RegimeSwitchSim().step(
            seed=0,
            scenario=_scenario(num_steps=n),
        )
        assert out.fills_count == n


def test_rule_fired_categories_present_across_seeds() -> None:
    sim = RegimeSwitchSim()
    rules: set[str] = set()
    for seed in range(80):
        out = sim.step(seed=seed, scenario=_scenario(switch_probability=0.1))
        rules.add(out.rule_fired)
    # With 40 steps at p=0.1 we should see a mix of trajectories.
    assert "switching_few" in rules or "switching_many" in rules


def test_distribution_over_seeds_varies() -> None:
    sim = RegimeSwitchSim()
    pnls = {
        sim.step(seed=s, scenario=_scenario()).pnl_usd
        for s in range(50)
    }
    assert len(pnls) >= 5


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_starting_regime_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(
            seed=0,
            scenario=_scenario(starting_regime="C"),
        )


def test_invalid_num_steps_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(num_steps=0))
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(num_steps=-5))
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(num_steps="5"))


def test_num_steps_above_cap_rejected() -> None:
    sim = RegimeSwitchSim(RegimeSwitchSimConfig(max_steps=10))
    with pytest.raises(ValueError):
        sim.step(seed=0, scenario=_scenario(num_steps=11))


def test_invalid_switch_probability_rejected() -> None:
    for bad in (-0.01, 1.01, "abc"):
        with pytest.raises(ValueError):
            RegimeSwitchSim().step(
                seed=0,
                scenario=_scenario(switch_probability=bad),
            )


def test_invalid_drift_or_std_rejected() -> None:
    for key, bad in (
        ("regime_a_drift", -0.06),
        ("regime_a_drift", 0.06),
        ("regime_a_std", -0.001),
        ("regime_a_std", 0.51),
        ("regime_b_drift", -0.06),
        ("regime_b_std", 0.51),
    ):
        with pytest.raises(ValueError):
            RegimeSwitchSim().step(seed=0, scenario=_scenario(**{key: bad}))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(entry_price=-1.0))
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=_scenario(order_size_usd=0.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        RegimeSwitchSim().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "entry_price",
        "order_size_usd",
        "switch_probability",
        "regime_a_drift",
        "regime_a_std",
        "regime_b_drift",
        "regime_b_std",
    ):
        with pytest.raises(ValueError):
            RegimeSwitchSim().step(
                seed=0,
                scenario=_scenario(extra={key: nan}),
            )


def test_infinity_inputs_rejected() -> None:
    # PR #263 review lesson: positive-float validators must reject +inf
    # because `inf > 0.0` is True, and inf propagates NaN through the walk.
    inf = float("inf")
    for key in ("entry_price", "order_size_usd"):
        with pytest.raises(ValueError):
            RegimeSwitchSim().step(
                seed=0,
                scenario=_scenario(extra={key: inf}),
            )
        with pytest.raises(ValueError):
            RegimeSwitchSim().step(
                seed=0,
                scenario=_scenario(extra={key: -inf}),
            )


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        RegimeSwitchSimConfig(max_steps=0)
    with pytest.raises(ValueError):
        RegimeSwitchSimConfig(max_steps=-1)
    with pytest.raises(ValueError):
        RegimeSwitchSimConfig(max_steps=10_000_000)
