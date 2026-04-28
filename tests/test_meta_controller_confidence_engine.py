"""Phase 6.T1b — Confidence Engine tests.

Covers:

* :func:`resolve_proposed_side` consensus / tie / empty cases.
* :func:`compute_confidence` per-component breakdown (J3-aligned).
* Replay-determinism (INV-15).
* Config validation + YAML loader.
* Composite stays inside ``[0, 1]`` under degenerate inputs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.contracts.events import Side, SignalEvent
from intelligence_engine.meta_controller.evaluation import (
    CONFIDENCE_ENGINE_VERSION,
    ConfidenceComponents,
    ConfidenceEngineConfig,
    compute_confidence,
    load_confidence_engine_config,
    resolve_proposed_side,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _signal(side: Side, confidence: float, ts_ns: int = 1) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="X",
        side=side,
        confidence=confidence,
    )


def _config(
    *,
    consensus: float = 0.5,
    strength: float = 0.3,
    coverage: float = 0.2,
    saturation: int = 8,
) -> ConfidenceEngineConfig:
    return ConfidenceEngineConfig(
        consensus_weight=consensus,
        strength_weight=strength,
        coverage_weight=coverage,
        saturation_count=saturation,
    )


# ---------------------------------------------------------------------------
# Records — frozen + range validation
# ---------------------------------------------------------------------------


def test_components_are_frozen() -> None:
    c = ConfidenceComponents(
        consensus=0.5,
        strength=0.5,
        coverage=0.5,
        composite=0.5,
        signal_count=4,
    )
    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.composite = 0.0  # type: ignore[misc]


def test_components_validate_ranges() -> None:
    with pytest.raises(ValueError, match="composite"):
        ConfidenceComponents(
            consensus=0.5,
            strength=0.5,
            coverage=0.5,
            composite=1.5,
            signal_count=4,
        )
    with pytest.raises(ValueError, match="signal_count"):
        ConfidenceComponents(
            consensus=0.5,
            strength=0.5,
            coverage=0.5,
            composite=0.5,
            signal_count=-1,
        )


def test_config_validates_weights() -> None:
    with pytest.raises(ValueError, match="consensus_weight"):
        ConfidenceEngineConfig(
            consensus_weight=-0.1,
            strength_weight=0.3,
            coverage_weight=0.2,
            saturation_count=8,
        )


def test_config_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError, match="at least one weight"):
        ConfidenceEngineConfig(
            consensus_weight=0.0,
            strength_weight=0.0,
            coverage_weight=0.0,
            saturation_count=8,
        )


def test_config_rejects_zero_saturation() -> None:
    with pytest.raises(ValueError, match="saturation_count"):
        ConfidenceEngineConfig(
            consensus_weight=0.5,
            strength_weight=0.3,
            coverage_weight=0.2,
            saturation_count=0,
        )


# ---------------------------------------------------------------------------
# resolve_proposed_side
# ---------------------------------------------------------------------------


def test_proposed_side_empty_is_hold() -> None:
    assert resolve_proposed_side([]) is Side.HOLD


def test_proposed_side_strict_majority_buy() -> None:
    sigs = [
        _signal(Side.BUY, 0.6),
        _signal(Side.BUY, 0.7),
        _signal(Side.SELL, 0.8),
    ]
    assert resolve_proposed_side(sigs) is Side.BUY


def test_proposed_side_strict_majority_sell() -> None:
    sigs = [
        _signal(Side.BUY, 0.9),
        _signal(Side.SELL, 0.4),
        _signal(Side.SELL, 0.4),
    ]
    assert resolve_proposed_side(sigs) is Side.SELL


def test_proposed_side_tie_is_hold() -> None:
    sigs = [
        _signal(Side.BUY, 0.9),
        _signal(Side.SELL, 0.9),
    ]
    assert resolve_proposed_side(sigs) is Side.HOLD


def test_proposed_side_hold_signals_abstain() -> None:
    sigs = [
        _signal(Side.HOLD, 0.5),
        _signal(Side.HOLD, 0.5),
        _signal(Side.BUY, 0.6),
    ]
    assert resolve_proposed_side(sigs) is Side.BUY


# ---------------------------------------------------------------------------
# compute_confidence — empty / tie / consensus
# ---------------------------------------------------------------------------


def test_compute_confidence_empty_is_zero() -> None:
    out = compute_confidence([], _config())
    assert out.composite == 0.0
    assert out.signal_count == 0
    assert out.consensus == 0.0
    assert out.strength == 0.0
    assert out.coverage == 0.0


def test_compute_confidence_tie_is_zero_composite() -> None:
    sigs = [
        _signal(Side.BUY, 0.9),
        _signal(Side.SELL, 0.9),
    ]
    out = compute_confidence(sigs, _config())
    assert out.composite == 0.0
    assert out.signal_count == 2
    assert out.consensus == 0.0


def test_compute_confidence_unanimous_buy_high_score() -> None:
    sigs = [
        _signal(Side.BUY, 0.9),
        _signal(Side.BUY, 0.9),
        _signal(Side.BUY, 0.9),
        _signal(Side.BUY, 0.9),
    ]
    cfg = _config(saturation=4)
    out = compute_confidence(sigs, cfg)
    # consensus = 4/4 = 1; strength = 0.9; coverage = 4/4 = 1.
    # composite = 0.5*1 + 0.3*0.9 + 0.2*1 = 0.97
    assert out.consensus == pytest.approx(1.0)
    assert out.strength == pytest.approx(0.9)
    assert out.coverage == pytest.approx(1.0)
    assert out.composite == pytest.approx(0.97)
    assert out.signal_count == 4


def test_compute_confidence_clamps_input_signal_confidence() -> None:
    """Out-of-range raw confidence values are clamped to [0, 1] for the
    strength term — defensive, doesn't reject the signal outright."""
    sigs = [
        _signal(Side.BUY, 1.5),
        _signal(Side.BUY, -0.2),
    ]
    out = compute_confidence(sigs, _config(saturation=2))
    # clamped to 1.0 and 0.0 -> mean 0.5
    assert out.strength == pytest.approx(0.5)


def test_compute_confidence_coverage_caps_at_one() -> None:
    sigs = [_signal(Side.BUY, 0.5) for _ in range(20)]
    cfg = _config(saturation=4)
    out = compute_confidence(sigs, cfg)
    assert out.coverage == pytest.approx(1.0)
    # consensus = 1, strength = 0.5, coverage = 1
    # composite = 0.5 + 0.15 + 0.2 = 0.85
    assert out.composite == pytest.approx(0.85)


def test_compute_confidence_low_strength_drags_composite() -> None:
    sigs = [
        _signal(Side.BUY, 0.1),
        _signal(Side.BUY, 0.1),
        _signal(Side.BUY, 0.1),
    ]
    out = compute_confidence(sigs, _config(saturation=8))
    # consensus = 1, strength = 0.1, coverage = 3/8 = 0.375
    # composite = 0.5*1 + 0.3*0.1 + 0.2*0.375 = 0.605
    assert out.composite == pytest.approx(0.605)


def test_compute_confidence_excludes_minority_from_strength() -> None:
    """Strength is averaged over the consensus side only — a strong
    minority signal must not lift the composite."""
    sigs = [
        _signal(Side.BUY, 0.1),
        _signal(Side.BUY, 0.1),
        _signal(Side.SELL, 0.99),  # ignored for strength
    ]
    out = compute_confidence(sigs, _config(saturation=4))
    assert out.strength == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_compute_confidence_is_replay_deterministic() -> None:
    sigs = [
        _signal(Side.BUY, 0.6, ts_ns=1),
        _signal(Side.BUY, 0.7, ts_ns=2),
        _signal(Side.SELL, 0.4, ts_ns=3),
    ]
    cfg = _config()
    a = compute_confidence(sigs, cfg)
    b = compute_confidence(sigs, cfg)
    assert a == b


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def test_load_registry_yaml_round_trip() -> None:
    cfg = load_confidence_engine_config(REPO_ROOT / "registry" / "confidence.yaml")
    assert cfg.consensus_weight == pytest.approx(0.5)
    assert cfg.strength_weight == pytest.approx(0.3)
    assert cfg.coverage_weight == pytest.approx(0.2)
    assert cfg.saturation_count == 8
    assert cfg.version == CONFIDENCE_ENGINE_VERSION


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("consensus_weight: 1.0\nstrength_weight: 1.0\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_confidence_engine_config(p)


def test_load_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "consensus_weight: 0.5\n"
        "strength_weight: 0.3\n"
        "coverage_weight: 0.2\n"
        "saturation_count: 8\n"
        "rogue_field: nope\n"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_confidence_engine_config(p)


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_confidence_engine_config(p)
