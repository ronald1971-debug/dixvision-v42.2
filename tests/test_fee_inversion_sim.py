"""Tests for SIM-21 fee_inversion — H4.14 of the canonical-rebuild walk."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.fee_inversion import FeeInversion, FeeInversionConfig


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
        "per_step_drift": 0.0,
        "per_step_std": 0.005,
        "taker_fee_bps": 10.0,
        "funding_rate_bps_per_step": 1.0,
        "exit_slippage_bps": 5.0,
        "breakeven_band_bps": 5.0,
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
    sim = FeeInversion()
    a = sim.step(seed=42, scenario=_scenario())
    b = sim.step(seed=42, scenario=_scenario())
    assert a == b


def test_outcome_round_trip() -> None:
    out = FeeInversion().step(seed=7, scenario=_scenario())
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "S"
    assert out.seed == 7
    assert out.fills_count == 100
    assert out.terminal_drawdown_usd >= 0.0


def test_zero_costs_yields_pure_price_pnl() -> None:
    """With every cost zero'd out, net_pnl == gross_pnl."""
    out = FeeInversion().step(
        seed=5,
        scenario=_scenario(
            taker_fee_bps=0.0,
            funding_rate_bps_per_step=0.0,
            exit_slippage_bps=0.0,
            breakeven_band_bps=0.0,
        ),
    )
    assert out.terminal_drawdown_usd == 0.0
    assert out.rule_fired in ("profitable", "straight_loss", "breakeven")


def test_inverted_rule_fires_when_gross_positive_but_fees_eat_it() -> None:
    """Carve out: tiny upward drift gives a small gross gain, then
    fat fees + funding burn flips it to net negative ⇒ inverted."""
    out = FeeInversion().step(
        seed=1,
        scenario=_scenario(
            per_step_drift=0.0001,
            per_step_std=0.0,
            num_steps=10,
            taker_fee_bps=200.0,
            funding_rate_bps_per_step=100.0,
            exit_slippage_bps=500.0,
            breakeven_band_bps=0.0,
            side="buy",
        ),
    )
    assert out.rule_fired == "inverted"
    assert out.pnl_usd < 0.0
    assert out.terminal_drawdown_usd > 0.0


def test_profitable_rule_fires_when_gain_clears_fees() -> None:
    """Strong upward drift, modest fees ⇒ profitable."""
    out = FeeInversion().step(
        seed=1,
        scenario=_scenario(
            per_step_drift=0.005,
            per_step_std=0.0,
            num_steps=200,
            taker_fee_bps=5.0,
            funding_rate_bps_per_step=0.0,
            exit_slippage_bps=2.0,
            breakeven_band_bps=5.0,
            side="buy",
        ),
    )
    assert out.rule_fired == "profitable"
    assert out.pnl_usd > 0.0


def test_straight_loss_rule_fires_when_gross_negative() -> None:
    """Strong downward drift on a long ⇒ gross < 0 ⇒ straight_loss."""
    out = FeeInversion().step(
        seed=1,
        scenario=_scenario(
            per_step_drift=-0.005,
            per_step_std=0.0,
            num_steps=200,
            taker_fee_bps=0.0,
            funding_rate_bps_per_step=0.0,
            exit_slippage_bps=0.0,
            breakeven_band_bps=5.0,
            side="buy",
        ),
    )
    assert out.rule_fired == "straight_loss"
    assert out.pnl_usd < 0.0


def test_breakeven_band_classifies_small_pnl_as_breakeven() -> None:
    """A wide breakeven_band catches any near-zero net pnl."""
    out = FeeInversion().step(
        seed=1,
        scenario=_scenario(
            per_step_drift=0.0,
            per_step_std=0.0,
            num_steps=1,
            taker_fee_bps=0.0,
            funding_rate_bps_per_step=0.0,
            exit_slippage_bps=0.0,
            breakeven_band_bps=100.0,
        ),
    )
    assert out.rule_fired == "breakeven"
    assert abs(out.pnl_usd) <= 10_000.0 * 100.0 / 10_000.0


def test_short_funding_sign_is_inverted_relative_to_long() -> None:
    """Positive funding rate ⇒ longs pay, shorts receive."""
    sc_long = _scenario(
        per_step_drift=0.0,
        per_step_std=0.0,
        num_steps=100,
        taker_fee_bps=0.0,
        funding_rate_bps_per_step=10.0,
        exit_slippage_bps=0.0,
        breakeven_band_bps=0.0,
        side="buy",
    )
    sc_short = _scenario(
        per_step_drift=0.0,
        per_step_std=0.0,
        num_steps=100,
        taker_fee_bps=0.0,
        funding_rate_bps_per_step=10.0,
        exit_slippage_bps=0.0,
        breakeven_band_bps=0.0,
        side="sell",
    )
    long_out = FeeInversion().step(seed=0, scenario=sc_long)
    short_out = FeeInversion().step(seed=0, scenario=sc_short)
    # Long pays 10bps * 100 steps * 10_000 size = 1000 USD funding.
    assert pytest.approx(long_out.pnl_usd, abs=1e-6) == -1_000.0
    # Short receives the same amount.
    assert pytest.approx(short_out.pnl_usd, abs=1e-6) == 1_000.0


def test_burn_equals_gross_minus_net() -> None:
    """terminal_drawdown_usd is the USD of cost drag, not a price DD."""
    out = FeeInversion().step(
        seed=4,
        scenario=_scenario(
            per_step_drift=0.001,
            per_step_std=0.0,
            num_steps=50,
            taker_fee_bps=20.0,
            funding_rate_bps_per_step=2.0,
            exit_slippage_bps=10.0,
            breakeven_band_bps=0.0,
            side="buy",
        ),
    )
    # Recompute the gross from the deterministic walk.
    final_price = 100.0 * (1.001 ** 50)
    gross = 10_000.0 * (final_price - 100.0) / 100.0
    expected_burn = max(0.0, gross - out.pnl_usd)
    assert pytest.approx(out.terminal_drawdown_usd, rel=1e-9) == expected_burn


def test_higher_fees_increase_burn_monotonically() -> None:
    sim = FeeInversion()
    burns: list[float] = []
    for fee_bps in (1.0, 10.0, 50.0, 100.0):
        out = sim.step(
            seed=1,
            scenario=_scenario(
                per_step_drift=0.001,
                per_step_std=0.0,
                taker_fee_bps=fee_bps,
                funding_rate_bps_per_step=0.0,
                exit_slippage_bps=0.0,
                breakeven_band_bps=0.0,
            ),
        )
        burns.append(out.terminal_drawdown_usd)
    # Burn must be monotonically non-decreasing in fee.
    for i in range(1, len(burns)):
        assert burns[i] >= burns[i - 1]


def test_rule_fired_diversity_across_seeds() -> None:
    sim = FeeInversion()
    rules: set[str] = set()
    for seed in range(80):
        out = sim.step(seed=seed, scenario=_scenario())
        rules.add(out.rule_fired)
    # With drift=0 + non-trivial std + non-trivial fees we expect
    # multiple categories to fire across 80 seeds.
    assert len(rules) >= 2


def test_distribution_over_seeds_varies() -> None:
    sim = FeeInversion()
    pnls = {sim.step(seed=s, scenario=_scenario()).pnl_usd for s in range(50)}
    assert len(pnls) >= 5


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        FeeInversion().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
        FeeInversion().step(seed=0, scenario=_scenario(side="long"))


def test_num_steps_lower_bound_enforced() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        FeeInversion().step(seed=0, scenario=_scenario(num_steps=0))


def test_num_steps_upper_bound_enforced() -> None:
    sim = FeeInversion(FeeInversionConfig(max_steps=10))
    with pytest.raises(ValueError, match="exceeds max_steps"):
        sim.step(seed=0, scenario=_scenario(num_steps=11))


def test_taker_fee_bps_bounds_enforced() -> None:
    with pytest.raises(ValueError, match=r"taker_fee_bps"):
        FeeInversion().step(seed=0, scenario=_scenario(taker_fee_bps=-1.0))
    with pytest.raises(ValueError, match=r"taker_fee_bps"):
        FeeInversion().step(seed=0, scenario=_scenario(taker_fee_bps=300.0))


def test_funding_rate_bounds_enforced() -> None:
    with pytest.raises(ValueError, match=r"funding_rate_bps_per_step"):
        FeeInversion().step(
            seed=0, scenario=_scenario(funding_rate_bps_per_step=-200.0)
        )
    with pytest.raises(ValueError, match=r"funding_rate_bps_per_step"):
        FeeInversion().step(
            seed=0, scenario=_scenario(funding_rate_bps_per_step=200.0)
        )


def test_exit_slippage_bps_bounds_enforced() -> None:
    with pytest.raises(ValueError, match=r"exit_slippage_bps"):
        FeeInversion().step(
            seed=0, scenario=_scenario(exit_slippage_bps=-1.0)
        )
    with pytest.raises(ValueError, match=r"exit_slippage_bps"):
        FeeInversion().step(
            seed=0, scenario=_scenario(exit_slippage_bps=600.0)
        )


def test_entry_price_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match=r"entry_price"):
        FeeInversion().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError, match=r"entry_price"):
        FeeInversion().step(
            seed=0, scenario=_scenario(entry_price=float("nan"))
        )
    with pytest.raises(ValueError, match=r"entry_price"):
        FeeInversion().step(
            seed=0, scenario=_scenario(entry_price=float("inf"))
        )


def test_size_usd_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match=r"order_size_usd"):
        FeeInversion().step(seed=0, scenario=_scenario(order_size_usd=0.0))
    with pytest.raises(ValueError, match=r"order_size_usd"):
        FeeInversion().step(
            seed=0, scenario=_scenario(order_size_usd=float("nan"))
        )


def test_per_step_drift_nan_rejected() -> None:
    with pytest.raises(ValueError, match=r"per_step_drift.*finite"):
        FeeInversion().step(
            seed=0, scenario=_scenario(per_step_drift=float("nan"))
        )


def test_missing_meta_key_rejected() -> None:
    sc = _scenario()
    bad_meta = dict(sc.meta)
    del bad_meta["taker_fee_bps"]
    bad = RealityScenario(
        scenario_id=sc.scenario_id,
        ts_ns=sc.ts_ns,
        initial_state_hash=sc.initial_state_hash,
        meta=bad_meta,
    )
    with pytest.raises(ValueError, match=r"taker_fee_bps"):
        FeeInversion().step(seed=0, scenario=bad)


def test_max_steps_config_bounds() -> None:
    with pytest.raises(ValueError, match=r"max_steps"):
        FeeInversionConfig(max_steps=0)
    with pytest.raises(ValueError, match=r"max_steps"):
        FeeInversionConfig(max_steps=2_000_000)
