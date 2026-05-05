"""Tests for SIM-09 flash_crash_synth (H4.2)."""

from __future__ import annotations

import pytest

from core.contracts.simulation import RealityScenario
from simulation.flash_crash_synth import FlashCrashConfig, FlashCrashSynth


def _scenario(**meta_overrides: object) -> RealityScenario:
    meta: dict[str, object] = {
        "entry_price": 100.0,
        "position_size_usd": 10_000.0,
        "side": "long",
        "max_drop_pct": 0.15,
        "recovery_pct": 0.5,
    }
    meta.update(meta_overrides)
    return RealityScenario(
        scenario_id="FC-1",
        ts_ns=1_000,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = FlashCrashSynth().step(seed=42, scenario=s)
    b = FlashCrashSynth().step(seed=42, scenario=s)
    assert a == b


def test_long_flash_crash_full_recovery_zero_pnl() -> None:
    out = FlashCrashSynth(
        FlashCrashConfig(drop_jitter=0.0, recovery_jitter=0.0)
    ).step(seed=0, scenario=_scenario(recovery_pct=1.0))
    # Full recovery means terminal == entry: pnl == 0.
    assert out.pnl_usd == pytest.approx(0.0)
    # But drawdown is still the trough distance: 15% * 10_000 = 1500.
    assert out.terminal_drawdown_usd == pytest.approx(1500.0)


def test_long_flash_crash_no_recovery_max_loss() -> None:
    out = FlashCrashSynth(
        FlashCrashConfig(drop_jitter=0.0, recovery_jitter=0.0)
    ).step(seed=0, scenario=_scenario(recovery_pct=0.0))
    # No recovery: terminal == trough; pnl == -drawdown.
    assert out.pnl_usd == pytest.approx(-1500.0)
    assert out.terminal_drawdown_usd == pytest.approx(1500.0)


def test_short_flash_spike_full_recovery_zero_pnl() -> None:
    out = FlashCrashSynth(
        FlashCrashConfig(drop_jitter=0.0, recovery_jitter=0.0)
    ).step(
        seed=0,
        scenario=_scenario(side="short", recovery_pct=1.0),
    )
    assert out.pnl_usd == pytest.approx(0.0)
    assert out.terminal_drawdown_usd == pytest.approx(1500.0)
    assert out.rule_fired == "short_flash_spike"


def test_short_flash_spike_no_recovery_max_loss() -> None:
    out = FlashCrashSynth(
        FlashCrashConfig(drop_jitter=0.0, recovery_jitter=0.0)
    ).step(
        seed=0,
        scenario=_scenario(side="short", recovery_pct=0.0),
    )
    # Short with no recovery: terminal = spike, loss = drawdown.
    assert out.pnl_usd == pytest.approx(-1500.0)
    assert out.terminal_drawdown_usd == pytest.approx(1500.0)


def test_outcome_fields_round_trip() -> None:
    s = _scenario()
    out = FlashCrashSynth().step(seed=99, scenario=s)
    assert out.scenario_id == s.scenario_id
    assert out.seed == 99
    assert out.fills_count == 2


def test_long_rule_fired() -> None:
    out = FlashCrashSynth().step(seed=0, scenario=_scenario(side="long"))
    assert out.rule_fired == "long_flash_crash"


def test_drawdown_is_non_negative_for_all_seeds() -> None:
    s = _scenario()
    runner = FlashCrashSynth()
    for seed in range(50):
        out = runner.step(seed=seed, scenario=s)
        assert out.terminal_drawdown_usd >= 0.0


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=0, scenario=_scenario(side="flat"))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"entry_price": 100.0},
    )
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=0, scenario=bad)


def test_max_drop_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=0, scenario=_scenario(max_drop_pct=0.0))
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=0, scenario=_scenario(max_drop_pct=1.5))


def test_recovery_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=0, scenario=_scenario(recovery_pct=-0.1))
    with pytest.raises(ValueError):
        FlashCrashSynth().step(seed=0, scenario=_scenario(recovery_pct=1.5))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        FlashCrashConfig(drop_jitter=-0.1)
    with pytest.raises(ValueError):
        FlashCrashConfig(recovery_jitter=1.5)


def test_long_loss_scales_with_position_size() -> None:
    s_small = _scenario(position_size_usd=1_000.0, recovery_pct=0.0)
    s_big = _scenario(position_size_usd=100_000.0, recovery_pct=0.0)
    cfg = FlashCrashConfig(drop_jitter=0.0, recovery_jitter=0.0)
    out_s = FlashCrashSynth(cfg).step(seed=0, scenario=s_small)
    out_b = FlashCrashSynth(cfg).step(seed=0, scenario=s_big)
    assert out_b.pnl_usd == pytest.approx(out_s.pnl_usd * 100.0)
    assert out_b.terminal_drawdown_usd == pytest.approx(
        out_s.terminal_drawdown_usd * 100.0
    )


def test_drop_jitter_perturbs_trough_only() -> None:
    cfg_jitter = FlashCrashConfig(drop_jitter=0.5, recovery_jitter=0.0)
    cfg_clean = FlashCrashConfig(drop_jitter=0.0, recovery_jitter=0.0)
    s = _scenario(recovery_pct=0.0)
    out_j = FlashCrashSynth(cfg_jitter).step(seed=0, scenario=s)
    out_c = FlashCrashSynth(cfg_clean).step(seed=0, scenario=s)
    # With jitter and seed=0, drop differs from clean baseline.
    assert out_j.terminal_drawdown_usd != out_c.terminal_drawdown_usd


def test_seeded_random_is_independent_of_scenario_meta() -> None:
    s1 = _scenario()
    s2 = _scenario(position_size_usd=20_000.0)
    a = FlashCrashSynth().step(seed=7, scenario=s1)
    b = FlashCrashSynth().step(seed=7, scenario=s2)
    # Same seed and scenario_id -> same drop/recovery factor; the
    # only difference is the linear position_size scaling.
    assert b.pnl_usd == pytest.approx(a.pnl_usd * 2.0)


def test_extreme_drop_clamped_to_one() -> None:
    # Even with max_drop=1.0 and positive drop_jitter, realised drop
    # must clamp to 1.0 so trough is non-negative.
    out = FlashCrashSynth(FlashCrashConfig(drop_jitter=0.5)).step(
        seed=0,
        scenario=_scenario(max_drop_pct=1.0, recovery_pct=0.0),
    )
    # Trough is at most entry * (1 - 1.0) = 0; loss <= -size_usd.
    assert out.pnl_usd >= -10_000.0 - 1e-9
    assert out.terminal_drawdown_usd <= 10_000.0 + 1e-9
