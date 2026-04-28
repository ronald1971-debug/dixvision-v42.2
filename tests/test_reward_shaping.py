"""Tier 1.5 — reward shaping (H5 + J3 per-component breakdown) tests."""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts.events import SystemEventKind
from learning_engine.lanes.reward_shaping import (
    KNOWN_SIZING_RATIONALES,
    REWARD_SHAPING_VERSION,
    SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR,
    SIZING_RATIONALE_KELLY_CAPPED,
    SIZING_RATIONALE_PRIMARY,
    SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER,
    RewardBreakdown,
    RewardShapingConfig,
    breakdown_components_dict,
    compute_reward_breakdown,
    load_reward_shaping_config,
)


def _config(**overrides: float) -> RewardShapingConfig:
    base = dict(
        pnl_weight=1.0,
        slippage_penalty_per_bps=0.05,
        latency_penalty_per_us=0.001,
        confidence_consensus_weight=0.05,
        confidence_strength_weight=0.05,
        confidence_coverage_weight=0.02,
        sizing_kelly_cap_penalty=0.0,
        sizing_floor_penalty=0.0,
        fallback_penalty=0.5,
    )
    base.update(overrides)
    return RewardShapingConfig(**base)  # type: ignore[arg-type]


def _call(**overrides: object) -> RewardBreakdown:
    base: dict[str, object] = dict(
        ts_ns=100,
        raw_pnl=10.0,
        slippage_bps=0.0,
        latency_ns=0,
        confidence_consensus=1.0,
        confidence_strength=1.0,
        confidence_coverage=1.0,
        sizing_rationale=SIZING_RATIONALE_PRIMARY,
        fallback=False,
        config=_config(),
    )
    base.update(overrides)
    return compute_reward_breakdown(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_negative_weights() -> None:
    for key in (
        "pnl_weight",
        "slippage_penalty_per_bps",
        "latency_penalty_per_us",
        "confidence_consensus_weight",
        "confidence_strength_weight",
        "confidence_coverage_weight",
        "sizing_kelly_cap_penalty",
        "sizing_floor_penalty",
        "fallback_penalty",
    ):
        with pytest.raises(ValueError, match=key):
            _config(**{key: -0.1})  # type: ignore[arg-type]


def test_config_default_version_matches_module_constant() -> None:
    cfg = _config()
    assert cfg.version == REWARD_SHAPING_VERSION


def test_load_config_round_trip(tmp_path) -> None:
    p = tmp_path / "reward_shaping.yaml"
    p.write_text(
        "version: v3.3-J3\n"
        "pnl_weight: 1.0\n"
        "slippage_penalty_per_bps: 0.05\n"
        "latency_penalty_per_us: 0.001\n"
        "confidence_consensus_weight: 0.05\n"
        "confidence_strength_weight: 0.05\n"
        "confidence_coverage_weight: 0.02\n"
        "sizing_kelly_cap_penalty: 0.0\n"
        "sizing_floor_penalty: 0.0\n"
        "fallback_penalty: 0.5\n"
    )
    cfg = load_reward_shaping_config(p)
    assert cfg.pnl_weight == pytest.approx(1.0)
    assert cfg.fallback_penalty == pytest.approx(0.5)
    assert cfg.version == "v3.3-J3"


def test_load_config_rejects_missing_keys(tmp_path) -> None:
    p = tmp_path / "reward_shaping.yaml"
    p.write_text("pnl_weight: 1.0\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_reward_shaping_config(p)


def test_load_config_rejects_unknown_keys(tmp_path) -> None:
    p = tmp_path / "reward_shaping.yaml"
    p.write_text(
        "pnl_weight: 1.0\n"
        "slippage_penalty_per_bps: 0.05\n"
        "latency_penalty_per_us: 0.001\n"
        "confidence_consensus_weight: 0.05\n"
        "confidence_strength_weight: 0.05\n"
        "confidence_coverage_weight: 0.02\n"
        "sizing_kelly_cap_penalty: 0.0\n"
        "sizing_floor_penalty: 0.0\n"
        "fallback_penalty: 0.5\n"
        "extra: 99\n"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_reward_shaping_config(p)


def test_registry_yaml_loads_with_default_loader() -> None:
    """Pin the on-disk registry/reward_shaping.yaml against the loader."""
    cfg = load_reward_shaping_config("registry/reward_shaping.yaml")
    assert cfg.version == REWARD_SHAPING_VERSION


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_breakdown_is_frozen() -> None:
    out = _call()
    assert dataclasses.is_dataclass(out)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.raw_pnl = 0.0  # type: ignore[misc]


def test_breakdown_components_are_tuple_of_tuples() -> None:
    """J3 requires the breakdown be hashable + replay-deterministic."""
    out = _call()
    assert isinstance(out.components, tuple)
    for entry in out.components:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        assert isinstance(entry[0], str)
        assert isinstance(entry[1], float)


def test_known_sizing_rationales_match_position_sizer() -> None:
    """Pin the alignment with intelligence_engine.meta_controller.allocation.

    The reward-shaping module duplicates the rationale tokens to
    avoid an L2-violating import; this guards against drift.
    """
    from intelligence_engine.meta_controller.allocation import position_sizer

    sizer_tokens = {
        position_sizer._RATIONALE_PRIMARY,
        position_sizer._RATIONALE_BELOW_FLOOR,
        position_sizer._RATIONALE_REGIME_ZERO,
        position_sizer._RATIONALE_KELLY_CAPPED,
    }
    assert sizer_tokens == set(KNOWN_SIZING_RATIONALES)


# ---------------------------------------------------------------------------
# INV-47 invertibility
# ---------------------------------------------------------------------------


def test_raw_pnl_preserved_unchanged() -> None:
    """INV-47: shaping must not lose the raw PnL."""
    out = _call(raw_pnl=42.5)
    assert out.raw_pnl == 42.5


def test_raw_pnl_preserved_when_pnl_weight_zero() -> None:
    """Even if pnl_weight==0 (no contribution), raw_pnl is retained."""
    out = _call(raw_pnl=42.5, config=_config(pnl_weight=0.0))
    assert out.raw_pnl == 42.5
    components = breakdown_components_dict(out)
    assert components["pnl"] == 0.0


# ---------------------------------------------------------------------------
# J3 — shaped reward equals sum of components
# ---------------------------------------------------------------------------


def test_shaped_reward_is_sum_of_components() -> None:
    out = _call(
        raw_pnl=5.0,
        slippage_bps=2.0,
        latency_ns=1_000,
        confidence_consensus=0.8,
        confidence_strength=0.7,
        confidence_coverage=0.5,
        sizing_rationale=SIZING_RATIONALE_KELLY_CAPPED,
        fallback=True,
        config=_config(sizing_kelly_cap_penalty=0.1),
    )
    expected = sum(v for _, v in out.components)
    assert out.shaped_reward == pytest.approx(expected)


def test_pnl_weight_applies_correctly() -> None:
    out = _call(raw_pnl=10.0, config=_config(pnl_weight=0.5))
    components = breakdown_components_dict(out)
    assert components["pnl"] == pytest.approx(5.0)


def test_confidence_components_apply_with_weights() -> None:
    out = _call(
        confidence_consensus=0.8,
        confidence_strength=0.6,
        confidence_coverage=0.4,
        config=_config(
            confidence_consensus_weight=0.1,
            confidence_strength_weight=0.2,
            confidence_coverage_weight=0.3,
        ),
    )
    components = breakdown_components_dict(out)
    assert components["confidence_consensus"] == pytest.approx(0.08)
    assert components["confidence_strength"] == pytest.approx(0.12)
    assert components["confidence_coverage"] == pytest.approx(0.12)


def test_slippage_penalty_uses_absolute_value() -> None:
    """Negative slippage_bps still penalises (we paid a bad price)."""
    out_pos = _call(slippage_bps=4.0, config=_config(slippage_penalty_per_bps=0.1))
    out_neg = _call(slippage_bps=-4.0, config=_config(slippage_penalty_per_bps=0.1))
    components_pos = breakdown_components_dict(out_pos)
    components_neg = breakdown_components_dict(out_neg)
    assert components_pos["slippage_penalty"] == pytest.approx(-0.4)
    assert components_neg["slippage_penalty"] == pytest.approx(-0.4)


def test_latency_penalty_per_microsecond() -> None:
    out = _call(latency_ns=10_000, config=_config(latency_penalty_per_us=0.5))
    components = breakdown_components_dict(out)
    # 10_000 ns = 10 µs, ×0.5 = 5.0 penalty
    assert components["latency_penalty"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Sizing-rationale gates
# ---------------------------------------------------------------------------


def test_kelly_cap_penalty_only_on_kelly_capped() -> None:
    cfg = _config(sizing_kelly_cap_penalty=0.3, sizing_floor_penalty=0.4)
    out_kc = _call(sizing_rationale=SIZING_RATIONALE_KELLY_CAPPED, config=cfg)
    out_pri = _call(sizing_rationale=SIZING_RATIONALE_PRIMARY, config=cfg)
    out_below = _call(
        sizing_rationale=SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR, config=cfg
    )
    out_zero = _call(
        sizing_rationale=SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER, config=cfg
    )
    assert dict(out_kc.components).get("sizing_kelly_cap_penalty") == pytest.approx(
        -0.3
    )
    assert "sizing_kelly_cap_penalty" not in dict(out_pri.components)
    assert "sizing_kelly_cap_penalty" not in dict(out_below.components)
    assert "sizing_kelly_cap_penalty" not in dict(out_zero.components)


def test_floor_penalty_only_on_confidence_below_floor() -> None:
    cfg = _config(sizing_floor_penalty=0.4)
    out = _call(
        sizing_rationale=SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR, config=cfg
    )
    assert dict(out.components).get("sizing_floor_penalty") == pytest.approx(-0.4)
    out_zero = _call(
        sizing_rationale=SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER, config=cfg
    )
    assert "sizing_floor_penalty" not in dict(out_zero.components)


def test_regime_zero_multiplier_is_not_penalised() -> None:
    """A zero-size outcome under regime_zero_multiplier is *correct* —
    it must not draw any sizing penalty."""
    cfg = _config(sizing_kelly_cap_penalty=99.0, sizing_floor_penalty=99.0)
    out = _call(
        sizing_rationale=SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER, config=cfg
    )
    components = dict(out.components)
    assert "sizing_kelly_cap_penalty" not in components
    assert "sizing_floor_penalty" not in components


def test_fallback_penalty_only_when_fallback_true() -> None:
    cfg = _config(fallback_penalty=1.5)
    out_on = _call(fallback=True, config=cfg)
    out_off = _call(fallback=False, config=cfg)
    assert dict(out_on.components).get("fallback_penalty") == pytest.approx(-1.5)
    assert "fallback_penalty" not in dict(out_off.components)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_sizing_rationale_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sizing_rationale"):
        _call(sizing_rationale="bogus")


def test_negative_latency_rejected() -> None:
    with pytest.raises(ValueError, match="latency_ns"):
        _call(latency_ns=-1)


@pytest.mark.parametrize(
    "field",
    ["confidence_consensus", "confidence_strength", "confidence_coverage"],
)
def test_out_of_range_confidence_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _call(**{field: 1.5})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SystemEvent projection
# ---------------------------------------------------------------------------


def test_to_event_projects_breakdown() -> None:
    out = _call(
        raw_pnl=5.0,
        sizing_rationale=SIZING_RATIONALE_KELLY_CAPPED,
        config=_config(sizing_kelly_cap_penalty=0.1),
    )
    ev = out.to_event()
    assert ev.sub_kind is SystemEventKind.REWARD_BREAKDOWN
    assert ev.ts_ns == 100
    assert ev.payload["raw_pnl"] == "5.000000"
    assert ev.payload["shaping_version"] == REWARD_SHAPING_VERSION
    assert "c.pnl" in ev.payload
    assert "c.sizing_kelly_cap_penalty" in ev.payload


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_compute_is_replay_deterministic() -> None:
    a = _call(raw_pnl=1.23, slippage_bps=2.0, latency_ns=500)
    b = _call(raw_pnl=1.23, slippage_bps=2.0, latency_ns=500)
    assert a == b
    assert hash(a) == hash(b)


def test_components_are_ordered_consistently() -> None:
    """J3 — component order is part of the contract (dashboards rely
    on it). The first three components are always pnl + the three
    confidence components, in that order."""
    out = _call(sizing_rationale=SIZING_RATIONALE_KELLY_CAPPED, fallback=True)
    names = [n for n, _ in out.components]
    assert names[0] == "pnl"
    assert names[1] == "confidence_consensus"
    assert names[2] == "confidence_strength"
    assert names[3] == "confidence_coverage"
