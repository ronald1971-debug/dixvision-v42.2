"""Phase 6.T1b — Position Sizer tests.

Covers:

* Frozen records + range validation.
* Composite formula across all four rationales:
  ``primary`` / ``confidence_below_floor`` / ``regime_zero_multiplier``
  / ``kelly_capped``.
* ``regime → multiplier`` mapping (TREND / RANGE / VOL_SPIKE / UNKNOWN).
* Replay-determinism (INV-15).
* Registry YAML loader + bad-input rejection.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.coherence.belief_state import Regime
from core.coherence.performance_pressure import PressureVector
from intelligence_engine.meta_controller.allocation import (
    POSITION_SIZER_VERSION,
    PositionSizerConfig,
    SizingComponents,
    compute_position_size,
    load_position_sizer_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pressure(*, risk: float = 0.0) -> PressureVector:
    return PressureVector(
        ts_ns=1,
        perf=0.0,
        risk=risk,
        drift=0.0,
        latency=0.0,
        uncertainty=0.1,
        safety_modifier=1.0,
        cross_signal_entropy=0.0,
        signal_count=0,
    )


def _config(
    *,
    base: float = 1.0,
    cap: float = 0.25,
    trend: float = 1.0,
    range_: float = 0.5,
    vol_spike: float = 0.0,
    floor: float = 0.2,
    risk_damping: float = 0.5,
) -> PositionSizerConfig:
    return PositionSizerConfig(
        base_fraction=base,
        kelly_cap=cap,
        trend_multiplier=trend,
        range_multiplier=range_,
        vol_spike_multiplier=vol_spike,
        confidence_floor=floor,
        risk_damping=risk_damping,
    )


# ---------------------------------------------------------------------------
# Records — frozen + range validation
# ---------------------------------------------------------------------------


def test_components_are_frozen() -> None:
    c = SizingComponents(
        confidence_factor=0.5,
        regime_factor=1.0,
        risk_factor=0.8,
        pre_cap_size=0.4,
        final_size=0.25,
        rationale="primary",
    )
    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.final_size = 0.0  # type: ignore[misc]


def test_components_validate_ranges() -> None:
    with pytest.raises(ValueError, match="final_size"):
        SizingComponents(
            confidence_factor=0.5,
            regime_factor=1.0,
            risk_factor=0.8,
            pre_cap_size=0.4,
            final_size=1.5,
            rationale="primary",
        )
    with pytest.raises(ValueError, match="regime_factor"):
        SizingComponents(
            confidence_factor=0.5,
            regime_factor=-0.1,
            risk_factor=0.8,
            pre_cap_size=0.4,
            final_size=0.25,
            rationale="primary",
        )


def test_config_validates_ranges() -> None:
    with pytest.raises(ValueError, match="kelly_cap"):
        PositionSizerConfig(
            base_fraction=1.0,
            kelly_cap=1.5,
            trend_multiplier=1.0,
            range_multiplier=0.5,
            vol_spike_multiplier=0.0,
            confidence_floor=0.2,
            risk_damping=0.5,
        )
    with pytest.raises(ValueError, match="trend_multiplier"):
        PositionSizerConfig(
            base_fraction=1.0,
            kelly_cap=0.25,
            trend_multiplier=-0.5,
            range_multiplier=0.5,
            vol_spike_multiplier=0.0,
            confidence_floor=0.2,
            risk_damping=0.5,
        )
    with pytest.raises(ValueError, match="risk_damping"):
        PositionSizerConfig(
            base_fraction=1.0,
            kelly_cap=0.25,
            trend_multiplier=1.0,
            range_multiplier=0.5,
            vol_spike_multiplier=0.0,
            confidence_floor=0.2,
            risk_damping=1.5,
        )


# ---------------------------------------------------------------------------
# multiplier_for
# ---------------------------------------------------------------------------


def test_multiplier_for_regime() -> None:
    cfg = _config()
    assert cfg.multiplier_for(Regime.TREND_UP) == 1.0
    assert cfg.multiplier_for(Regime.TREND_DOWN) == 1.0
    assert cfg.multiplier_for(Regime.RANGE) == 0.5
    assert cfg.multiplier_for(Regime.VOL_SPIKE) == 0.0
    assert cfg.multiplier_for(Regime.UNKNOWN) == 0.0


# ---------------------------------------------------------------------------
# Rationale paths
# ---------------------------------------------------------------------------


def test_below_floor_returns_zero() -> None:
    out = compute_position_size(
        confidence=0.1,
        regime=Regime.TREND_UP,
        pressure=_pressure(),
        config=_config(floor=0.2),
    )
    assert out.rationale == "confidence_below_floor"
    assert out.final_size == 0.0
    assert out.confidence_factor == 0.0


def test_unknown_regime_returns_zero() -> None:
    out = compute_position_size(
        confidence=0.9,
        regime=Regime.UNKNOWN,
        pressure=_pressure(),
        config=_config(),
    )
    assert out.rationale == "regime_zero_multiplier"
    assert out.regime_factor == 0.0
    assert out.final_size == 0.0
    # confidence factor still recorded for audit
    assert out.confidence_factor == pytest.approx(0.9)


def test_vol_spike_returns_zero_by_default() -> None:
    out = compute_position_size(
        confidence=0.9,
        regime=Regime.VOL_SPIKE,
        pressure=_pressure(),
        config=_config(vol_spike=0.0),
    )
    assert out.rationale == "regime_zero_multiplier"
    assert out.final_size == 0.0


def test_primary_path_trend_up_no_risk() -> None:
    out = compute_position_size(
        confidence=0.5,
        regime=Regime.TREND_UP,
        pressure=_pressure(risk=0.0),
        config=_config(base=1.0, cap=0.5, trend=1.0, risk_damping=0.5),
    )
    # confidence_factor = 0.5, regime = 1.0, risk = 1.0
    # pre_cap = 0.5; cap = 0.5 -> not capped
    assert out.rationale == "primary"
    assert out.final_size == pytest.approx(0.5)


def test_kelly_cap_fires_when_pre_cap_exceeds() -> None:
    out = compute_position_size(
        confidence=1.0,
        regime=Regime.TREND_UP,
        pressure=_pressure(risk=0.0),
        config=_config(base=1.0, cap=0.25, trend=1.0, risk_damping=0.5),
    )
    # pre_cap = 1.0; capped to 0.25
    assert out.rationale == "kelly_capped"
    assert out.pre_cap_size == pytest.approx(1.0)
    assert out.final_size == pytest.approx(0.25)


def test_range_regime_uses_range_multiplier() -> None:
    out = compute_position_size(
        confidence=0.8,
        regime=Regime.RANGE,
        pressure=_pressure(risk=0.0),
        config=_config(range_=0.5, cap=1.0, risk_damping=0.5),
    )
    # 0.8 · 0.5 · 1 = 0.4
    assert out.regime_factor == pytest.approx(0.5)
    assert out.final_size == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Risk damping
# ---------------------------------------------------------------------------


def test_risk_damping_reduces_size() -> None:
    out_low = compute_position_size(
        confidence=0.8,
        regime=Regime.TREND_UP,
        pressure=_pressure(risk=0.0),
        config=_config(cap=1.0, risk_damping=0.5),
    )
    out_high = compute_position_size(
        confidence=0.8,
        regime=Regime.TREND_UP,
        pressure=_pressure(risk=1.0),
        config=_config(cap=1.0, risk_damping=0.5),
    )
    assert out_low.risk_factor == pytest.approx(1.0)
    assert out_high.risk_factor == pytest.approx(0.5)
    assert out_high.final_size < out_low.final_size


def test_risk_damping_zero_disables() -> None:
    out = compute_position_size(
        confidence=0.8,
        regime=Regime.TREND_UP,
        pressure=_pressure(risk=1.0),
        config=_config(cap=1.0, risk_damping=0.0),
    )
    assert out.risk_factor == pytest.approx(1.0)


def test_pressure_risk_clamped() -> None:
    """Out-of-range pressure.risk values still produce valid output."""
    p = PressureVector(
        ts_ns=1,
        perf=0.0,
        risk=2.5,  # out of nominal range
        drift=0.0,
        latency=0.0,
        uncertainty=0.1,
        safety_modifier=1.0,
        cross_signal_entropy=0.0,
        signal_count=0,
    )
    out = compute_position_size(
        confidence=0.8,
        regime=Regime.TREND_UP,
        pressure=p,
        config=_config(cap=1.0, risk_damping=1.0),
    )
    # risk clamped to 1 -> risk_factor = 0
    assert out.risk_factor == pytest.approx(0.0)
    assert out.final_size == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_compute_is_replay_deterministic() -> None:
    kwargs = dict(
        confidence=0.7,
        regime=Regime.TREND_UP,
        pressure=_pressure(risk=0.3),
        config=_config(),
    )
    a = compute_position_size(**kwargs)  # type: ignore[arg-type]
    b = compute_position_size(**kwargs)  # type: ignore[arg-type]
    assert a == b


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def test_load_registry_yaml_round_trip() -> None:
    cfg = load_position_sizer_config(
        REPO_ROOT / "registry" / "position_sizer.yaml"
    )
    assert cfg.base_fraction == pytest.approx(1.0)
    assert cfg.kelly_cap == pytest.approx(0.25)
    assert cfg.trend_multiplier == pytest.approx(1.0)
    assert cfg.range_multiplier == pytest.approx(0.5)
    assert cfg.vol_spike_multiplier == pytest.approx(0.0)
    assert cfg.confidence_floor == pytest.approx(0.2)
    assert cfg.risk_damping == pytest.approx(0.5)
    assert cfg.version == POSITION_SIZER_VERSION


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("base_fraction: 1.0\nkelly_cap: 0.25\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_position_sizer_config(p)


def test_load_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "base_fraction: 1.0\n"
        "kelly_cap: 0.25\n"
        "trend_multiplier: 1.0\n"
        "range_multiplier: 0.5\n"
        "vol_spike_multiplier: 0.0\n"
        "confidence_floor: 0.2\n"
        "risk_damping: 0.5\n"
        "rogue_key: nope\n"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_position_sizer_config(p)


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_position_sizer_config(p)
