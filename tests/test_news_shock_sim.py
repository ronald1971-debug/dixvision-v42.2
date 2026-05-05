"""Tests for SIM-18 news_shock_sim — H4.11 of the canonical-rebuild walk."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.news_shock_sim import NewsShockSim, NewsShockSimConfig


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
        "shock_probability_per_step": 0.05,
        "shock_magnitude_bps": 100.0,
        "shock_bullish_probability": 0.5,
        "baseline_drift": 0.0,
        "baseline_std": 0.005,
        "aftershock_decay": 0.5,
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
    sim = NewsShockSim()
    a = sim.step(seed=42, scenario=_scenario())
    b = sim.step(seed=42, scenario=_scenario())
    assert a == b


def test_outcome_round_trip() -> None:
    out = NewsShockSim().step(seed=7, scenario=_scenario())
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "S"
    assert out.seed == 7
    # fills_count is the latency-to-shock; bounded by num_steps.
    assert 0 <= out.fills_count <= 50


def test_zero_shock_probability_yields_no_shock() -> None:
    out = NewsShockSim().step(
        seed=3,
        scenario=_scenario(shock_probability_per_step=0.0),
    )
    assert out.rule_fired == "no_shock"


def test_full_shock_probability_fires_immediately() -> None:
    # At p=1.0 the shock fires at step 0.
    out = NewsShockSim().step(
        seed=1,
        scenario=_scenario(
            shock_probability_per_step=1.0,
            shock_bullish_probability=1.0,
            baseline_std=0.0,
            shock_magnitude_bps=500.0,
            aftershock_decay=5.0,
        ),
    )
    assert out.rule_fired == "buy_shock"


def test_bullish_probability_zero_yields_sell_shock() -> None:
    out = NewsShockSim().step(
        seed=1,
        scenario=_scenario(
            shock_probability_per_step=1.0,
            shock_bullish_probability=0.0,
        ),
    )
    assert out.rule_fired == "sell_shock"


def test_pnl_sign_by_side_buy_gain_equals_sell_loss() -> None:
    sc = _scenario()
    buy = NewsShockSim().step(seed=11, scenario=sc)
    sell_meta = dict(sc.meta)
    sell_meta["side"] = "sell"
    sc_sell = RealityScenario(
        scenario_id=sc.scenario_id,
        ts_ns=sc.ts_ns,
        initial_state_hash=sc.initial_state_hash,
        meta=sell_meta,
    )
    sell = NewsShockSim().step(seed=11, scenario=sc_sell)
    assert pytest.approx(buy.pnl_usd, rel=1e-9) == -sell.pnl_usd


def test_drawdown_non_negative_and_matches_negative_pnl() -> None:
    out = NewsShockSim().step(seed=5, scenario=_scenario())
    assert out.terminal_drawdown_usd >= 0.0
    if out.pnl_usd < 0.0:
        assert pytest.approx(out.terminal_drawdown_usd, rel=1e-9) == -out.pnl_usd
    else:
        assert out.terminal_drawdown_usd == 0.0


def test_fills_count_encodes_latency_to_shock() -> None:
    """fills_count = step at which shock fired, or num_steps if no shock.

    The docstring on simulation/news_shock_sim.py explicitly
    overloads fills_count to carry latency-to-shock semantic;
    this test pins that contract.
    """
    sim = NewsShockSim()

    # No shock: fills_count == num_steps for any n.
    for n in (1, 10, 100):
        out = sim.step(
            seed=0,
            scenario=_scenario(
                num_steps=n, shock_probability_per_step=0.0
            ),
        )
        assert out.rule_fired == "no_shock"
        assert out.fills_count == n

    # Guaranteed shock at p=1.0 fires at step 0.
    out = sim.step(
        seed=0,
        scenario=_scenario(
            num_steps=50,
            shock_probability_per_step=1.0,
            shock_bullish_probability=1.0,
        ),
    )
    assert out.rule_fired == "buy_shock"
    assert out.fills_count == 0

    # 0 < p < 1: latency must be a valid step index when fired.
    for seed in range(40):
        out = sim.step(
            seed=seed,
            scenario=_scenario(
                num_steps=50, shock_probability_per_step=0.5
            ),
        )
        if out.rule_fired == "no_shock":
            assert out.fills_count == 50
        else:
            assert 0 <= out.fills_count < 50


def test_rule_fired_diversity_across_seeds() -> None:
    sim = NewsShockSim()
    rules: set[str] = set()
    for seed in range(80):
        out = sim.step(seed=seed, scenario=_scenario())
        rules.add(out.rule_fired)
    # With 50 steps at p=0.05 we should see all three categories.
    assert "no_shock" in rules
    assert "buy_shock" in rules or "sell_shock" in rules


def test_distribution_over_seeds_varies() -> None:
    sim = NewsShockSim()
    pnls = {sim.step(seed=s, scenario=_scenario()).pnl_usd for s in range(50)}
    assert len(pnls) >= 5


def test_higher_shock_magnitude_increases_pnl_dispersion() -> None:
    sim = NewsShockSim()
    pnls_small = [
        sim.step(
            seed=s,
            scenario=_scenario(
                shock_probability_per_step=1.0,
                shock_magnitude_bps=10.0,
            ),
        ).pnl_usd
        for s in range(40)
    ]
    pnls_big = [
        sim.step(
            seed=s,
            scenario=_scenario(
                shock_probability_per_step=1.0,
                shock_magnitude_bps=2000.0,
            ),
        ).pnl_usd
        for s in range(40)
    ]
    spread_small = max(pnls_small) - min(pnls_small)
    spread_big = max(pnls_big) - min(pnls_big)
    assert spread_big > spread_small * 5.0


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_steps_rejected() -> None:
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=_scenario(num_steps=0))
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=_scenario(num_steps="5"))


def test_num_steps_above_cap_rejected() -> None:
    sim = NewsShockSim(NewsShockSimConfig(max_steps=10))
    with pytest.raises(ValueError):
        sim.step(seed=0, scenario=_scenario(num_steps=11))


def test_invalid_probability_or_magnitude_rejected() -> None:
    for key, bad in (
        ("shock_probability_per_step", -0.01),
        ("shock_probability_per_step", 1.01),
        ("shock_magnitude_bps", -1.0),
        ("shock_magnitude_bps", 10_001.0),
        ("shock_bullish_probability", -0.01),
        ("shock_bullish_probability", 1.01),
        ("aftershock_decay", -0.001),
        ("aftershock_decay", 5.01),
    ):
        with pytest.raises(ValueError):
            NewsShockSim().step(seed=0, scenario=_scenario(**{key: bad}))


def test_invalid_baseline_rejected() -> None:
    for key, bad in (
        ("baseline_drift", -0.006),
        ("baseline_drift", 0.006),
        ("baseline_std", -0.001),
        ("baseline_std", 0.11),
    ):
        with pytest.raises(ValueError):
            NewsShockSim().step(seed=0, scenario=_scenario(**{key: bad}))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=_scenario(entry_price=-1.0))
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=_scenario(order_size_usd=0.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        NewsShockSim().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "entry_price",
        "order_size_usd",
        "shock_probability_per_step",
        "shock_magnitude_bps",
        "shock_bullish_probability",
        "baseline_drift",
        "baseline_std",
        "aftershock_decay",
    ):
        with pytest.raises(ValueError):
            NewsShockSim().step(seed=0, scenario=_scenario(extra={key: nan}))


def test_infinity_inputs_rejected() -> None:
    inf = float("inf")
    for key in ("entry_price", "order_size_usd"):
        with pytest.raises(ValueError):
            NewsShockSim().step(seed=0, scenario=_scenario(extra={key: inf}))
        with pytest.raises(ValueError):
            NewsShockSim().step(seed=0, scenario=_scenario(extra={key: -inf}))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        NewsShockSimConfig(max_steps=0)
    with pytest.raises(ValueError):
        NewsShockSimConfig(max_steps=-1)
    with pytest.raises(ValueError):
        NewsShockSimConfig(max_steps=10_000_000)


def test_aftershock_std_never_drops_below_baseline() -> None:
    """Devin Review BUG_0001 regression: previously the aftershock std
    overwrote (rather than augmented) baseline_std and decayed toward
    zero, freezing the post-shock walk. With high decay + non-zero
    baseline, dispersion across seeds must remain non-trivial."""
    sim = NewsShockSim()
    pnls = []
    for seed in range(20):
        out = sim.step(
            seed=seed,
            scenario=_scenario(
                scenario_id=f"floor_{seed}",
                num_steps=200,
                # Force shock at step 0.
                shock_probability_per_step=1.0,
                shock_magnitude_bps=100.0,
                shock_bullish_probability=1.0,
                baseline_drift=0.0,
                baseline_std=0.005,
                # High decay → aftershock contribution near-zero by
                # step ~10. The walk must still diffuse via
                # baseline_std for the rest.
                aftershock_decay=5.0,
            ),
        )
        pnls.append(out.pnl_usd)
    # Without the fix, std collapses → near-deterministic pnl across
    # seeds. With the fix, baseline_std keeps the walk diffusing so
    # dispersion is meaningfully non-zero.
    spread = max(pnls) - min(pnls)
    assert spread > 50.0, f"expected spread>50, got {spread}"
