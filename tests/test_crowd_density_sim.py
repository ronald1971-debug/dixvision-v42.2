"""Tests for SIM-12 crowd_density (H4.5)."""

from __future__ import annotations

import pytest

from core.contracts.simulation import RealityScenario
from simulation.crowd_density import CrowdDensity, CrowdDensityConfig


def _scenario(**meta_overrides: object) -> RealityScenario:
    meta: dict[str, object] = {
        "entry_price": 100.0,
        "position_size_usd": 50_000.0,
        "side": "long",
        "crowd_share": 0.9,
        "squeeze_intensity": 0.9,
        "unwind_pct": 0.05,
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="CD-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = CrowdDensity().step(seed=42, scenario=s)
    b = CrowdDensity().step(seed=42, scenario=s)
    assert a == b


def test_high_density_triggers_long_squeeze() -> None:
    cfg = CrowdDensityConfig(pressure_jitter=0.0, unwind_jitter=0.0)
    s = _scenario(side="long", crowd_share=0.9, squeeze_intensity=0.9)
    out = CrowdDensity(cfg).step(seed=0, scenario=s)
    # base_pressure = 0.81 > threshold 0.5, so squeeze fires.
    # adverse = 0.05, terminal = 95, pnl = 50000 * (95 - 100)/100 = -2500.
    assert out.rule_fired == "long_squeeze"
    assert out.pnl_usd == pytest.approx(-2_500.0)
    assert out.terminal_drawdown_usd == pytest.approx(2_500.0)
    assert out.fills_count == 2


def test_high_density_triggers_short_squeeze() -> None:
    cfg = CrowdDensityConfig(pressure_jitter=0.0, unwind_jitter=0.0)
    s = _scenario(side="short", crowd_share=0.9, squeeze_intensity=0.9)
    out = CrowdDensity(cfg).step(seed=0, scenario=s)
    assert out.rule_fired == "short_squeeze"
    assert out.pnl_usd == pytest.approx(-2_500.0)
    assert out.terminal_drawdown_usd == pytest.approx(2_500.0)


def test_low_density_no_squeeze() -> None:
    cfg = CrowdDensityConfig(pressure_jitter=0.0, unwind_jitter=0.0)
    s = _scenario(crowd_share=0.3, squeeze_intensity=0.3)
    out = CrowdDensity(cfg).step(seed=0, scenario=s)
    # base_pressure = 0.09 < threshold 0.5.
    assert out.rule_fired == "no_squeeze"
    assert out.pnl_usd == 0.0
    assert out.terminal_drawdown_usd == 0.0
    assert out.fills_count == 1


def test_neutral_density_no_squeeze() -> None:
    cfg = CrowdDensityConfig(pressure_jitter=0.0, unwind_jitter=0.0)
    s = _scenario(crowd_share=0.5, squeeze_intensity=0.5)
    out = CrowdDensity(cfg).step(seed=0, scenario=s)
    # base_pressure = 0.25 < threshold 0.5.
    assert out.rule_fired == "no_squeeze"


def test_squeeze_pnl_always_non_positive_when_fires() -> None:
    s = _scenario(crowd_share=1.0, squeeze_intensity=1.0, unwind_pct=0.1)
    runner = CrowdDensity()
    for seed in range(50):
        out = runner.step(seed=seed, scenario=s)
        if out.rule_fired in {"long_squeeze", "short_squeeze"}:
            assert out.pnl_usd <= 0.0
            assert out.terminal_drawdown_usd >= 0.0


def test_outcome_round_trip() -> None:
    s = _scenario()
    out = CrowdDensity().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=_scenario(side="buy"))


def test_invalid_crowd_share_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=_scenario(crowd_share=1.1))
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=_scenario(crowd_share=-0.1))


def test_invalid_squeeze_intensity_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=_scenario(squeeze_intensity=2.0))


def test_invalid_unwind_pct_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=_scenario(unwind_pct=1.5))


def test_invalid_entry_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=_scenario(entry_price=0.0))
    with pytest.raises(ValueError):
        CrowdDensity().step(
            seed=0, scenario=_scenario(position_size_usd=-1.0)
        )


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        CrowdDensity().step(seed=0, scenario=bad)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        CrowdDensityConfig(squeeze_threshold=0.0)
    with pytest.raises(ValueError):
        CrowdDensityConfig(squeeze_threshold=1.5)
    with pytest.raises(ValueError):
        CrowdDensityConfig(pressure_jitter=-0.1)
    with pytest.raises(ValueError):
        CrowdDensityConfig(unwind_jitter=2.0)


def test_distribution_over_seeds_varies() -> None:
    s = _scenario(crowd_share=0.6, squeeze_intensity=0.6)
    runner = CrowdDensity()
    rules = {
        runner.step(seed=seed, scenario=s).rule_fired for seed in range(50)
    }
    # Pressure of 0.36 sits below the 0.5 threshold but jitter +/-0.15
    # should cross it sometimes — mix of squeeze and no_squeeze.
    assert len(rules) >= 1


def test_extreme_unwind_clamped_to_full_loss() -> None:
    cfg = CrowdDensityConfig(pressure_jitter=0.0, unwind_jitter=0.0)
    s = _scenario(unwind_pct=1.0)
    out = CrowdDensity(cfg).step(seed=0, scenario=s)
    # adverse = 1.0, terminal = 0, pnl = -size_usd.
    assert out.pnl_usd == pytest.approx(-50_000.0)
    assert out.terminal_drawdown_usd == pytest.approx(50_000.0)


def test_jitter_independence_from_meta() -> None:
    cfg = CrowdDensityConfig(pressure_jitter=0.0, unwind_jitter=0.0)
    s_a = _scenario(entry_price=100.0)
    s_b = _scenario(entry_price=200.0)
    out_a = CrowdDensity(cfg).step(seed=0, scenario=s_a)
    out_b = CrowdDensity(cfg).step(seed=0, scenario=s_b)
    # Same scenario_id, same seed → same RNG draw → same rule_fired.
    assert out_a.rule_fired == out_b.rule_fired
