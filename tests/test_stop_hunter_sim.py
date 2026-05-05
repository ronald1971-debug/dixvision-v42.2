"""Tests for SIM-08 stop_hunter (H4.1)."""

from __future__ import annotations

import pytest

from core.contracts.simulation import RealityScenario
from simulation.stop_hunter import StopHunter, StopHunterConfig


def _scenario(**meta_overrides: float) -> RealityScenario:
    meta: dict[str, float] = {
        "entry_price": 100.0,
        "position_size_usd": 10_000.0,
        "stop_price": 95.0,
        "cluster_thickness_usd": 500_000.0,
        "hunt_intensity": 0.7,
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="SCN-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = StopHunter().step(seed=42, scenario=s)
    b = StopHunter().step(seed=42, scenario=s)
    assert a == b


def test_different_seeds_can_diverge_but_stay_bounded() -> None:
    s = _scenario()
    out0 = StopHunter().step(seed=0, scenario=s)
    out1 = StopHunter().step(seed=1, scenario=s)
    # Both must be either hunt or respect; PnL is at most -loss_at_wick.
    max_loss = 10_000.0 * (100.0 - (95.0 - 0.5 * 5.0)) / 100.0
    assert -max_loss - 1e-9 <= out0.pnl_usd <= 0.0
    assert -max_loss - 1e-9 <= out1.pnl_usd <= 0.0


def test_no_intensity_means_stop_respected() -> None:
    out = StopHunter().step(seed=7, scenario=_scenario(hunt_intensity=0.0))
    assert out.rule_fired == "stop_respected"
    # Loss is exactly entry-stop * size / entry = 5/100 * 10_000 = 500
    assert out.pnl_usd == pytest.approx(-500.0)
    assert out.terminal_drawdown_usd == pytest.approx(500.0)
    assert out.fills_count == 1


def test_full_intensity_triggers_hunt() -> None:
    # intensity 1.0 + any cluster_pull >= 0 means triggered >= 1.0 > 0.5
    out = StopHunter().step(seed=7, scenario=_scenario(hunt_intensity=1.0))
    assert out.rule_fired == "stop_hunt_triggered"
    # Wick price = 95 - 0.5*5 = 92.5; loss = 10_000 * 7.5 / 100 = 750
    assert out.pnl_usd == pytest.approx(-750.0)
    assert out.terminal_drawdown_usd == pytest.approx(750.0)
    assert out.fills_count == 2


def test_overshoot_factor_increases_loss() -> None:
    light = StopHunter(StopHunterConfig(overshoot_factor=0.0))
    heavy = StopHunter(StopHunterConfig(overshoot_factor=1.0))
    s = _scenario(hunt_intensity=1.0)
    out_l = light.step(seed=0, scenario=s)
    out_h = heavy.step(seed=0, scenario=s)
    # overshoot 0 still triggers but wick == stop -> loss == 500
    assert out_l.pnl_usd == pytest.approx(-500.0)
    # overshoot 1 -> wick = 95 - 5 = 90; loss = 1000
    assert out_h.pnl_usd == pytest.approx(-1000.0)


def test_outcome_scenario_and_seed_round_trip() -> None:
    s = _scenario()
    out = StopHunter().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        StopHunter().step(seed=-1, scenario=_scenario())


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        StopHunter().step(seed=0, scenario=bad)


def test_non_numeric_meta_rejected() -> None:
    bad = _scenario()
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={**bad.meta, "entry_price": "high"},
    )
    with pytest.raises(ValueError):
        StopHunter().step(seed=0, scenario=bad)


def test_stop_above_entry_rejected() -> None:
    with pytest.raises(ValueError):
        StopHunter().step(
            seed=0,
            scenario=_scenario(stop_price=110.0),
        )


def test_intensity_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        StopHunter().step(
            seed=0,
            scenario=_scenario(hunt_intensity=1.5),
        )
    with pytest.raises(ValueError):
        StopHunter().step(
            seed=0,
            scenario=_scenario(hunt_intensity=-0.1),
        )


def test_negative_thickness_rejected() -> None:
    with pytest.raises(ValueError):
        StopHunter().step(
            seed=0,
            scenario=_scenario(cluster_thickness_usd=-1.0),
        )


def test_zero_thickness_still_classifies_correctly() -> None:
    out = StopHunter().step(
        seed=0,
        scenario=_scenario(
            cluster_thickness_usd=0.0,
            hunt_intensity=1.0,
        ),
    )
    # Even with zero thickness, intensity 1.0 still triggers
    # (triggered = 1.0 * (1 + ~0) >= 0.5).
    assert out.rule_fired == "stop_hunt_triggered"


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        StopHunterConfig(trigger_threshold=-0.1)
    with pytest.raises(ValueError):
        StopHunterConfig(overshoot_factor=1.5)
    with pytest.raises(ValueError):
        StopHunterConfig(cluster_jitter=-0.5)


def test_distribution_over_seeds_has_expected_fingerprint() -> None:
    s = _scenario(hunt_intensity=0.4, cluster_thickness_usd=200_000.0)
    runner = StopHunter()
    triggers = sum(
        1
        for seed in range(50)
        if runner.step(seed=seed, scenario=s).rule_fired
        == "stop_hunt_triggered"
    )
    # intensity 0.4 with thickness 200k -> normalised_thickness 0.2 plus
    # jitter in [-0.2, 0.2] gives cluster_pull in [0, 0.4], so triggered
    # is in [0.4, 0.56] which straddles the 0.5 threshold and produces
    # a deterministic mix across the 50 seeds.
    assert 0 < triggers < 50


def test_scales_with_position_size() -> None:
    s_small = _scenario(position_size_usd=1_000.0, hunt_intensity=1.0)
    s_big = _scenario(position_size_usd=100_000.0, hunt_intensity=1.0)
    out_small = StopHunter().step(seed=0, scenario=s_small)
    out_big = StopHunter().step(seed=0, scenario=s_big)
    # Both trigger; bigger position scales pnl exactly 100x.
    assert out_small.rule_fired == "stop_hunt_triggered"
    assert out_big.rule_fired == "stop_hunt_triggered"
    assert out_big.pnl_usd == pytest.approx(out_small.pnl_usd * 100.0)
