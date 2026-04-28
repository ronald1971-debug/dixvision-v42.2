"""Phase 6.T1a — :class:`PressureVector` projection tests.

Exercises:

* Frozen / immutable / replay-determinism.
* Cross-signal entropy (INV-50): 5 BUY + 5 SELL → high uncertainty
  even when individual confidences are high.
* Continuous safety modifier (v3.1 H2 / INV-31): monotone-decreasing
  in uncertainty.
* Snapshot ``SystemEvent`` shape (INV-53 calibration hook).
* Config validation (alpha + beta <= 1, ranges).
* Registry YAML load — ``registry/pressure.yaml`` is the source of
  truth and parses to a valid :class:`PressureConfig`.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from core.coherence.performance_pressure import (
    PRESSURE_VECTOR_VERSION,
    PressureConfig,
    PressureVector,
    derive_pressure_vector,
    load_pressure_config,
)
from core.contracts.events import (
    EventKind,
    Side,
    SignalEvent,
    SystemEventKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sig(side: Side, *, conf: float = 0.9) -> SignalEvent:
    return SignalEvent(ts_ns=1, symbol="EURUSD", side=side, confidence=conf)


def _config(
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    high_water: float = 0.6,
    modifier: float = 1.5,
) -> PressureConfig:
    return PressureConfig(
        alpha=alpha,
        beta=beta,
        entropy_high_water=high_water,
        entropy_high_water_modifier=modifier,
    )


def _derive(
    signals: list[SignalEvent],
    *,
    config: PressureConfig | None = None,
    perf: float = 0.0,
    risk: float = 0.0,
    drift: float = 0.0,
    latency: float = 0.0,
) -> PressureVector:
    return derive_pressure_vector(
        ts_ns=42,
        signals=signals,
        perf=perf,
        risk=risk,
        drift=drift,
        latency=latency,
        config=config or _config(),
    )


# ---------------------------------------------------------------------------
# Frozen / replay-determinism
# ---------------------------------------------------------------------------


def test_pressure_vector_is_frozen_dataclass() -> None:
    pv = _derive([_sig(Side.BUY)])
    assert dataclasses.is_dataclass(pv)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pv.uncertainty = 0.5  # type: ignore[misc]


def test_pressure_vector_replay_determinism() -> None:
    sigs = [_sig(Side.BUY, conf=0.5), _sig(Side.SELL, conf=0.7)]
    a = _derive(sigs)
    b = _derive(sigs)
    assert a == b


# ---------------------------------------------------------------------------
# Cross-signal entropy (INV-50)
# ---------------------------------------------------------------------------


def test_uncertainty_low_when_signals_agree_at_high_confidence() -> None:
    sigs = [_sig(Side.BUY, conf=0.95) for _ in range(10)]
    pv = _derive(sigs)
    # Raw uncertainty is small (1 - 0.95 = 0.05), entropy is 0 → composite low.
    assert pv.cross_signal_entropy == 0.0
    assert pv.uncertainty < 0.1


def test_uncertainty_high_when_signals_disagree_at_high_confidence() -> None:
    """5 BUY + 5 SELL at 0.95 conf → high uncertainty (the INV-50 case)."""
    sigs = [_sig(Side.BUY, conf=0.95) for _ in range(5)] + [
        _sig(Side.SELL, conf=0.95) for _ in range(5)
    ]
    pv = _derive(sigs)
    # Two-side uniform split: H = log2(2) / log2(3) ≈ 0.6309.
    assert pv.cross_signal_entropy == pytest.approx(math.log2(2) / math.log2(3))
    # With alpha=beta=0.5, raw_unc ≈ 0.05 → composite uncertainty
    # ≈ 0.5 * 0.05 + 0.5 * 0.6309 ≈ 0.34. The headline assertion is
    # that it is materially larger than the agreement case.
    assert pv.uncertainty > 0.3


def test_uncertainty_max_when_uniform_three_way_split() -> None:
    """Uniform across BUY/SELL/HOLD → entropy = 1.0 (normalised)."""
    sigs = [_sig(Side.BUY)] * 3 + [_sig(Side.SELL)] * 3 + [_sig(Side.HOLD)] * 3
    pv = _derive(sigs)
    assert pv.cross_signal_entropy == pytest.approx(1.0)


def test_uncertainty_zero_for_empty_window() -> None:
    pv = _derive([])
    assert pv.uncertainty == 0.0
    assert pv.cross_signal_entropy == 0.0
    assert pv.signal_count == 0


# ---------------------------------------------------------------------------
# Continuous safety modifier (v3.1 H2 / INV-31)
# ---------------------------------------------------------------------------


def test_safety_modifier_is_one_below_high_water() -> None:
    sigs = [_sig(Side.BUY, conf=0.95) for _ in range(10)]
    pv = _derive(sigs)
    assert pv.safety_modifier == pytest.approx(1.0)


def test_safety_modifier_compresses_above_high_water() -> None:
    """Force max disagreement → safety_modifier strictly below 1."""
    sigs = [_sig(Side.BUY, conf=0.0)] * 3 + [_sig(Side.SELL, conf=0.0)] * 3 + [
        _sig(Side.HOLD, conf=0.0)
    ] * 3
    pv = _derive(sigs)
    assert pv.uncertainty == pytest.approx(1.0)
    # over = 1.0 - 0.6 = 0.4 ; modifier 1.5 → 1 - 0.6 = 0.4
    assert pv.safety_modifier == pytest.approx(0.4)


def test_safety_modifier_monotone_decreasing_in_uncertainty() -> None:
    """Increasing disagreement → never-increasing safety modifier."""
    series = [
        [_sig(Side.BUY, conf=0.95)] * 10,  # full agreement
        [_sig(Side.BUY, conf=0.95)] * 8 + [_sig(Side.SELL, conf=0.95)] * 2,
        [_sig(Side.BUY, conf=0.95)] * 5 + [_sig(Side.SELL, conf=0.95)] * 5,
        [_sig(Side.BUY, conf=0.0)] * 5 + [_sig(Side.SELL, conf=0.0)] * 5,
    ]
    mods = [_derive(s).safety_modifier for s in series]
    for prev, curr in zip(mods[:-1], mods[1:], strict=True):
        assert curr <= prev + 1e-9


def test_safety_modifier_clamped_to_zero() -> None:
    """Aggressive modifier slope must not produce negative damping."""
    cfg = _config(high_water=0.0, modifier=10.0)
    sigs = [_sig(Side.BUY, conf=0.0)] * 5 + [_sig(Side.SELL, conf=0.0)] * 5
    pv = _derive(sigs, config=cfg)
    assert pv.safety_modifier == 0.0


# ---------------------------------------------------------------------------
# Domain clamping
# ---------------------------------------------------------------------------


def test_pressure_inputs_clamped_to_unit_interval() -> None:
    pv = _derive(
        [_sig(Side.BUY)],
        perf=2.5,
        risk=-0.5,
        drift=10.0,
        latency=-99.0,
    )
    assert pv.perf == 1.0
    assert pv.risk == 0.0
    assert pv.drift == 1.0
    assert pv.latency == 0.0


# ---------------------------------------------------------------------------
# Snapshot SystemEvent (INV-53 calibration hook)
# ---------------------------------------------------------------------------


def test_pressure_snapshot_event_shape() -> None:
    pv = _derive([_sig(Side.BUY, conf=0.6)], perf=0.1, risk=0.2)
    ev = pv.to_event()
    assert ev.kind is EventKind.SYSTEM
    assert ev.sub_kind is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT
    assert ev.ts_ns == 42
    assert ev.source == "core.coherence.performance_pressure"
    assert ev.payload["version"] == PRESSURE_VECTOR_VERSION
    assert "uncertainty" in ev.payload
    assert "safety_modifier" in ev.payload
    assert "cross_signal_entropy" in ev.payload


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_alpha_plus_beta_above_one() -> None:
    with pytest.raises(ValueError, match="alpha \\+ beta"):
        PressureConfig(
            alpha=0.7,
            beta=0.7,
            entropy_high_water=0.6,
            entropy_high_water_modifier=1.0,
        )


def test_config_rejects_out_of_range_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PressureConfig(
            alpha=1.5,
            beta=0.0,
            entropy_high_water=0.6,
            entropy_high_water_modifier=1.0,
        )


def test_config_rejects_negative_modifier() -> None:
    with pytest.raises(ValueError, match="entropy_high_water_modifier"):
        PressureConfig(
            alpha=0.5,
            beta=0.5,
            entropy_high_water=0.6,
            entropy_high_water_modifier=-0.1,
        )


def test_load_pressure_config_from_registry() -> None:
    cfg = load_pressure_config(REPO_ROOT / "registry" / "pressure.yaml")
    assert cfg.alpha + cfg.beta <= 1.0 + 1e-9
    assert 0.0 <= cfg.entropy_high_water <= 1.0
    assert cfg.entropy_high_water_modifier >= 0.0
    assert cfg.version == PRESSURE_VECTOR_VERSION
