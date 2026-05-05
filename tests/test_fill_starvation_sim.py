"""Tests for SIM-20 fill_starvation — H4.13 of the canonical-rebuild walk."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.fill_starvation import FillStarvation, FillStarvationConfig


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
        "per_step_fill_probability": 0.1,
        "per_step_fill_fraction": 0.1,
        "per_step_drift": 0.0,
        "per_step_std": 0.005,
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
    sim = FillStarvation()
    a = sim.step(seed=42, scenario=_scenario())
    b = sim.step(seed=42, scenario=_scenario())
    assert a == b


def test_outcome_round_trip() -> None:
    out = FillStarvation().step(seed=7, scenario=_scenario())
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "S"
    assert out.seed == 7


def test_zero_fill_probability_yields_starved() -> None:
    out = FillStarvation().step(
        seed=1,
        scenario=_scenario(per_step_fill_probability=0.0),
    )
    assert out.rule_fired == "starved"
    assert out.fills_count == 0
    assert out.terminal_drawdown_usd == pytest.approx(10_000.0, rel=1e-9)
    assert out.pnl_usd == 0.0


def test_full_fill_probability_and_fraction_yields_full_fill() -> None:
    out = FillStarvation().step(
        seed=1,
        scenario=_scenario(
            per_step_fill_probability=1.0,
            per_step_fill_fraction=1.0,
        ),
    )
    assert out.rule_fired == "full_fill"
    assert out.fills_count == 1
    assert out.terminal_drawdown_usd == pytest.approx(0.0, abs=1e-9)


def test_partial_fill_intermediate_probability() -> None:
    sim = FillStarvation()
    rules: set[str] = set()
    for seed in range(80):
        out = sim.step(
            seed=seed,
            scenario=_scenario(
                per_step_fill_probability=0.3,
                per_step_fill_fraction=0.2,
            ),
        )
        rules.add(out.rule_fired)
    assert "partial_fill" in rules


def test_pnl_only_on_filled_portion() -> None:
    """Unfilled notional contributes 0 to pnl, all to drawdown."""
    sim = FillStarvation()
    out = sim.step(
        seed=1,
        scenario=_scenario(
            per_step_fill_probability=0.0,
            per_step_drift=0.005,
            per_step_std=0.0,
        ),
    )
    assert out.pnl_usd == 0.0
    assert out.terminal_drawdown_usd == pytest.approx(10_000.0, rel=1e-9)


def test_pnl_sign_by_side_buy_gain_equals_sell_loss() -> None:
    sc = _scenario(per_step_fill_probability=1.0, per_step_fill_fraction=1.0)
    buy = FillStarvation().step(seed=11, scenario=sc)
    sell_meta = dict(sc.meta)
    sell_meta["side"] = "sell"
    sc_sell = RealityScenario(
        scenario_id=sc.scenario_id,
        ts_ns=sc.ts_ns,
        initial_state_hash=sc.initial_state_hash,
        meta=sell_meta,
    )
    sell = FillStarvation().step(seed=11, scenario=sc_sell)
    assert pytest.approx(buy.pnl_usd, rel=1e-9) == -sell.pnl_usd


def test_drawdown_non_negative() -> None:
    sim = FillStarvation()
    for seed in range(30):
        out = sim.step(seed=seed, scenario=_scenario())
        assert out.terminal_drawdown_usd >= 0.0


def test_fills_count_non_negative_and_bounded() -> None:
    sim = FillStarvation()
    for seed in range(20):
        out = sim.step(seed=seed, scenario=_scenario(num_steps=50))
        assert 0 <= out.fills_count <= 50


def test_higher_fill_probability_increases_filled_fraction() -> None:
    sim = FillStarvation()
    low_dds = [
        sim.step(
            seed=s,
            scenario=_scenario(
                per_step_fill_probability=0.05,
                per_step_fill_fraction=0.5,
            ),
        ).terminal_drawdown_usd
        for s in range(40)
    ]
    high_dds = [
        sim.step(
            seed=s,
            scenario=_scenario(
                per_step_fill_probability=0.95,
                per_step_fill_fraction=0.5,
            ),
        ).terminal_drawdown_usd
        for s in range(40)
    ]
    assert sum(high_dds) < sum(low_dds)


def test_distribution_over_seeds_varies() -> None:
    sim = FillStarvation()
    pnls = {sim.step(seed=s, scenario=_scenario()).pnl_usd for s in range(50)}
    assert len(pnls) >= 5


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        FillStarvation().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_steps_rejected() -> None:
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=_scenario(num_steps=0))
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=_scenario(num_steps="5"))


def test_num_steps_above_cap_rejected() -> None:
    sim = FillStarvation(FillStarvationConfig(max_steps=10))
    with pytest.raises(ValueError):
        sim.step(seed=0, scenario=_scenario(num_steps=11))


def test_invalid_probability_or_fraction_rejected() -> None:
    for key, bad in (
        ("per_step_fill_probability", -0.01),
        ("per_step_fill_probability", 1.01),
        ("per_step_fill_fraction", -0.01),
        ("per_step_fill_fraction", 1.01),
        ("per_step_drift", -0.006),
        ("per_step_drift", 0.006),
        ("per_step_std", -0.001),
        ("per_step_std", 0.06),
    ):
        with pytest.raises(ValueError):
            FillStarvation().step(seed=0, scenario=_scenario(**{key: bad}))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=_scenario(entry_price=-1.0))
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=_scenario(order_size_usd=0.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        FillStarvation().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "entry_price",
        "order_size_usd",
        "per_step_fill_probability",
        "per_step_fill_fraction",
        "per_step_drift",
        "per_step_std",
    ):
        with pytest.raises(ValueError):
            FillStarvation().step(
                seed=0, scenario=_scenario(extra={key: nan})
            )


def test_infinity_inputs_rejected() -> None:
    inf = float("inf")
    for key in ("entry_price", "order_size_usd"):
        with pytest.raises(ValueError):
            FillStarvation().step(
                seed=0, scenario=_scenario(extra={key: inf})
            )
        with pytest.raises(ValueError):
            FillStarvation().step(
                seed=0, scenario=_scenario(extra={key: -inf})
            )


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        FillStarvationConfig(max_steps=0)
    with pytest.raises(ValueError):
        FillStarvationConfig(max_steps=-1)
    with pytest.raises(ValueError):
        FillStarvationConfig(max_steps=10_000_000)


def test_zero_fill_fraction_does_not_inflate_fills_count() -> None:
    """Devin Review BUG_0002 regression: with per_step_fill_fraction=0
    and per_step_fill_probability>0, every step would have passed the
    probability check and produced a zero-notional 'fill', incrementing
    fills_count without changing filled_usd. The guard rejects those."""
    out = FillStarvation().step(
        seed=1,
        scenario=_scenario(
            num_steps=200,
            per_step_fill_probability=1.0,
            per_step_fill_fraction=0.0,
        ),
    )
    assert out.fills_count == 0
    assert out.rule_fired == "starved"
    assert out.terminal_drawdown_usd == pytest.approx(10_000.0)


def test_overflow_walk_raises_value_error() -> None:
    """Devin Review BUG_0001 regression: with default max_steps the
    walk cannot overflow, but a custom config + drift+std combination
    can compound past float64 range. Rather than silently emit NaN /
    inf in pnl_usd we fail fast with a clear ValueError."""
    sim = FillStarvation(FillStarvationConfig(max_steps=200_000))
    with pytest.raises(ValueError, match="non-finite mid"):
        sim.step(
            seed=0,
            scenario=_scenario(
                num_steps=200_000,
                per_step_drift=0.005,
                per_step_std=0.0,
                per_step_fill_probability=0.0,
                per_step_fill_fraction=0.0,
            ),
        )
