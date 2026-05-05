"""Tests for SIM-19 drawdown_walk — H4.12 of the canonical-rebuild walk."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.drawdown_walk import DrawdownWalk, DrawdownWalkConfig


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
        "num_steps": 50,
        "per_step_drift": 0.0,
        "per_step_std": 0.01,
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
    sim = DrawdownWalk()
    a = sim.step(seed=42, scenario=_scenario())
    b = sim.step(seed=42, scenario=_scenario())
    assert a == b


def test_outcome_round_trip() -> None:
    out = DrawdownWalk().step(seed=7, scenario=_scenario())
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "S"
    assert out.seed == 7
    assert out.fills_count == 50


def test_zero_volatility_yields_shallow_drawdown() -> None:
    out = DrawdownWalk().step(
        seed=1,
        scenario=_scenario(per_step_std=0.0, per_step_drift=0.0),
    )
    assert out.terminal_drawdown_usd == 0.0
    assert out.rule_fired == "shallow"


def test_terminal_drawdown_can_exceed_terminal_underwater() -> None:
    """Key SIM-19 invariant: max running DD ≥ terminal underwater.

    A profitable walk that dips first must report a non-zero
    terminal_drawdown_usd reflecting the dip — distinct from
    other SIM modules where terminal_drawdown_usd == max(0,-pnl).
    """
    sim = DrawdownWalk()
    found_profitable_with_dd = False
    for seed in range(50):
        out = sim.step(seed=seed, scenario=_scenario())
        if out.pnl_usd > 0.0 and out.terminal_drawdown_usd > 0.0:
            found_profitable_with_dd = True
            break
    assert found_profitable_with_dd


def test_pnl_sign_by_side_buy_gain_equals_sell_loss() -> None:
    sc = _scenario()
    buy = DrawdownWalk().step(seed=11, scenario=sc)
    sell_meta = dict(sc.meta)
    sell_meta["side"] = "sell"
    sc_sell = RealityScenario(
        scenario_id=sc.scenario_id,
        ts_ns=sc.ts_ns,
        initial_state_hash=sc.initial_state_hash,
        meta=sell_meta,
    )
    sell = DrawdownWalk().step(seed=11, scenario=sc_sell)
    assert pytest.approx(buy.pnl_usd, rel=1e-9) == -sell.pnl_usd


def test_drawdown_non_negative() -> None:
    sim = DrawdownWalk()
    for seed in range(30):
        out = sim.step(seed=seed, scenario=_scenario())
        assert out.terminal_drawdown_usd >= 0.0


def test_fills_count_equals_num_steps() -> None:
    for n in (1, 10, 100):
        out = DrawdownWalk().step(seed=0, scenario=_scenario(num_steps=n))
        assert out.fills_count == n


def test_rule_fired_diversity_across_seeds() -> None:
    """Across many seeds at moderate vol we see multiple categories."""
    sim = DrawdownWalk()
    rules: set[str] = set()
    for seed in range(80):
        out = sim.step(
            seed=seed,
            scenario=_scenario(per_step_std=0.02, num_steps=100),
        )
        rules.add(out.rule_fired)
    assert len(rules) >= 2
    assert rules <= {"shallow", "moderate", "deep", "catastrophic"}


def test_higher_volatility_increases_drawdown_depth() -> None:
    sim = DrawdownWalk()
    low_vol_dds = [
        sim.step(
            seed=s, scenario=_scenario(per_step_std=0.001)
        ).terminal_drawdown_usd
        for s in range(40)
    ]
    high_vol_dds = [
        sim.step(
            seed=s, scenario=_scenario(per_step_std=0.05)
        ).terminal_drawdown_usd
        for s in range(40)
    ]
    assert max(high_vol_dds) > max(low_vol_dds) * 5.0


def test_distribution_over_seeds_varies() -> None:
    sim = DrawdownWalk()
    pnls = {sim.step(seed=s, scenario=_scenario()).pnl_usd for s in range(50)}
    assert len(pnls) >= 5


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_steps_rejected() -> None:
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=_scenario(num_steps=0))
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=_scenario(num_steps="5"))


def test_num_steps_above_cap_rejected() -> None:
    sim = DrawdownWalk(DrawdownWalkConfig(max_steps=10))
    with pytest.raises(ValueError):
        sim.step(seed=0, scenario=_scenario(num_steps=11))


def test_invalid_drift_or_std_rejected() -> None:
    for key, bad in (
        ("per_step_drift", -0.011),
        ("per_step_drift", 0.011),
        ("per_step_std", -0.001),
        ("per_step_std", 0.11),
    ):
        with pytest.raises(ValueError):
            DrawdownWalk().step(seed=0, scenario=_scenario(**{key: bad}))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=_scenario(entry_price=-1.0))
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=_scenario(order_size_usd=0.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        DrawdownWalk().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "entry_price",
        "order_size_usd",
        "per_step_drift",
        "per_step_std",
    ):
        with pytest.raises(ValueError):
            DrawdownWalk().step(seed=0, scenario=_scenario(extra={key: nan}))


def test_infinity_inputs_rejected() -> None:
    inf = float("inf")
    for key in ("entry_price", "order_size_usd"):
        with pytest.raises(ValueError):
            DrawdownWalk().step(seed=0, scenario=_scenario(extra={key: inf}))
        with pytest.raises(ValueError):
            DrawdownWalk().step(seed=0, scenario=_scenario(extra={key: -inf}))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        DrawdownWalkConfig(max_steps=0)
    with pytest.raises(ValueError):
        DrawdownWalkConfig(max_steps=-1)
    with pytest.raises(ValueError):
        DrawdownWalkConfig(max_steps=10_000_000)
