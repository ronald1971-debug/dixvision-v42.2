"""Unit tests for ``intelligence_engine.macro.regime_engine``.

Pure-function tests: every classification is asserted against a known
input, then re-run to verify replay determinism (INV-15). The engine's
config is loaded from the canonical registry YAML so the YAML schema is
covered too.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts.macro_regime import (
    MacroRegime,
    MacroRegimeReading,
    MacroSnapshot,
)
from intelligence_engine.macro import (
    MacroRegimeEngine,
    MacroRegimeEngineConfig,
    load_macro_regime_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(
    *,
    ts_ns: int = 1_700_000_000_000_000_000,
    vol_index: float = 18.0,
    breadth: float = 0.10,
    credit_spread_bps: float = 400.0,
    dollar_strength: float = 0.0,
    return_correlation: float = 0.40,
) -> MacroSnapshot:
    return MacroSnapshot(
        ts_ns=ts_ns,
        vol_index=vol_index,
        breadth=breadth,
        credit_spread_bps=credit_spread_bps,
        dollar_strength=dollar_strength,
        return_correlation=return_correlation,
    )


def _engine() -> MacroRegimeEngine:
    return MacroRegimeEngine(load_macro_regime_config())


# ---------------------------------------------------------------------------
# Snapshot contract
# ---------------------------------------------------------------------------


def test_snapshot_rejects_invalid_ranges() -> None:
    base = _snap()
    with pytest.raises(ValueError):
        dataclasses.replace(base, vol_index=-1.0)
    with pytest.raises(ValueError):
        dataclasses.replace(base, vol_index=150.0)
    with pytest.raises(ValueError):
        dataclasses.replace(base, breadth=2.0)
    with pytest.raises(ValueError):
        dataclasses.replace(base, credit_spread_bps=-1.0)
    with pytest.raises(ValueError):
        dataclasses.replace(base, return_correlation=1.5)
    with pytest.raises(ValueError):
        dataclasses.replace(base, dollar_strength=5.0)
    with pytest.raises(ValueError):
        dataclasses.replace(base, ts_ns=0)


def test_reading_rejects_invalid_confidence_or_rule() -> None:
    with pytest.raises(ValueError):
        MacroRegimeReading(
            regime=MacroRegime.NEUTRAL,
            confidence=1.5,
            rule_fired="x",
            snapshot_ts_ns=1,
        )
    with pytest.raises(ValueError):
        MacroRegimeReading(
            regime=MacroRegime.NEUTRAL,
            confidence=0.5,
            rule_fired="",
            snapshot_ts_ns=1,
        )
    with pytest.raises(ValueError):
        MacroRegimeReading(
            regime=MacroRegime.NEUTRAL,
            confidence=0.5,
            rule_fired="x",
            snapshot_ts_ns=0,
        )


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------


def test_load_macro_regime_config_from_registry() -> None:
    cfg = load_macro_regime_config()
    assert isinstance(cfg, MacroRegimeEngineConfig)
    assert cfg.vol_crisis > cfg.vol_risk_off > cfg.vol_risk_on > 0.0


def test_config_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        MacroRegimeEngineConfig(
            vol_crisis=10.0,
            correlation_crisis=0.85,
            vol_risk_off=20.0,  # > vol_crisis -> reject
            breadth_risk_off=-0.2,
            credit_risk_off_bps=500.0,
            vol_risk_on=5.0,
            breadth_risk_on=0.2,
            credit_risk_on_bps=300.0,
            confidence_floor=0.4,
            confidence_ceiling=0.9,
        )


# ---------------------------------------------------------------------------
# Classification — first-match rule order
# ---------------------------------------------------------------------------


def test_crisis_fires_on_extreme_vol() -> None:
    eng = _engine()
    r = eng.classify(_snap(vol_index=55.0, return_correlation=0.5))
    assert r.regime is MacroRegime.CRISIS
    assert r.rule_fired == "crisis_vol"
    assert 0.0 < r.confidence <= eng.config.confidence_ceiling


def test_crisis_fires_on_extreme_correlation() -> None:
    eng = _engine()
    r = eng.classify(_snap(vol_index=20.0, return_correlation=0.95))
    assert r.regime is MacroRegime.CRISIS
    assert r.rule_fired == "crisis_correlation"


def test_crisis_label_uses_normalised_dominant_driver() -> None:
    """When both CRISIS dimensions trip, rule_fired must reflect the
    dominant driver on a common scale.

    Regression: comparing raw vol_excess (range 0–60) against raw
    corr_excess (range 0–0.15) almost always picked vol even when
    correlation was the overwhelmingly larger violation. The label must
    be derived from the normalised confidences instead.
    """

    eng = _engine()

    # vol just barely over crisis (40.0), correlation deeply over crisis (0.85).
    # Raw excess: vol=0.5 vs corr=0.149. Normalised: vol≈0.017 of span,
    # corr≈0.99 of span. Correlation is the dominant driver.
    r_corr = eng.classify(_snap(vol_index=40.5, return_correlation=0.999))
    assert r_corr.regime is MacroRegime.CRISIS
    assert r_corr.rule_fired == "crisis_correlation"

    # Mirror case: vol deeply over crisis, correlation just barely over.
    # Vol must dominate the label.
    r_vol = eng.classify(_snap(vol_index=70.0, return_correlation=0.86))
    assert r_vol.regime is MacroRegime.CRISIS
    assert r_vol.rule_fired == "crisis_vol"


def test_risk_off_fires_on_elevated_vol_alone() -> None:
    eng = _engine()
    r = eng.classify(
        _snap(vol_index=30.0, breadth=0.05, credit_spread_bps=400.0)
    )
    assert r.regime is MacroRegime.RISK_OFF
    assert r.rule_fired.startswith("risk_off_")


def test_risk_off_fires_on_negative_breadth_alone() -> None:
    eng = _engine()
    r = eng.classify(
        _snap(vol_index=15.0, breadth=-0.30, credit_spread_bps=300.0)
    )
    assert r.regime is MacroRegime.RISK_OFF


def test_risk_off_fires_on_wide_credit_alone() -> None:
    eng = _engine()
    r = eng.classify(
        _snap(vol_index=15.0, breadth=0.10, credit_spread_bps=600.0)
    )
    assert r.regime is MacroRegime.RISK_OFF


def test_risk_off_confidence_scales_with_dimensions() -> None:
    eng = _engine()
    one = eng.classify(
        _snap(vol_index=30.0, breadth=0.05, credit_spread_bps=300.0)
    )
    three = eng.classify(
        _snap(vol_index=30.0, breadth=-0.30, credit_spread_bps=600.0)
    )
    assert one.regime is three.regime is MacroRegime.RISK_OFF
    assert three.confidence > one.confidence
    assert one.rule_fired == "risk_off_1_of_3"
    assert three.rule_fired == "risk_off_3_of_3"


def test_risk_on_requires_all_three_dimensions() -> None:
    eng = _engine()
    r = eng.classify(
        _snap(vol_index=12.0, breadth=0.40, credit_spread_bps=300.0)
    )
    assert r.regime is MacroRegime.RISK_ON
    assert r.rule_fired == "risk_on_all_dimensions"


def test_risk_on_does_not_fire_when_one_dimension_misaligned() -> None:
    eng = _engine()
    # vol clean + credit clean but breadth too tepid (<0.20 threshold)
    r = eng.classify(
        _snap(vol_index=12.0, breadth=0.05, credit_spread_bps=300.0)
    )
    assert r.regime is MacroRegime.NEUTRAL


def test_neutral_fallback_when_nothing_fires() -> None:
    eng = _engine()
    r = eng.classify(
        _snap(vol_index=20.0, breadth=0.10, credit_spread_bps=400.0)
    )
    assert r.regime is MacroRegime.NEUTRAL
    assert r.rule_fired == "neutral_fallback"
    assert r.confidence == eng.config.confidence_floor


# ---------------------------------------------------------------------------
# Determinism / reading provenance
# ---------------------------------------------------------------------------


def test_classify_is_pure_and_deterministic() -> None:
    eng = _engine()
    snap = _snap(vol_index=30.0, breadth=-0.30, credit_spread_bps=600.0)
    a = eng.classify(snap)
    b = eng.classify(snap)
    assert a == b


def test_reading_carries_snapshot_ts() -> None:
    eng = _engine()
    snap = _snap(ts_ns=1_700_000_000_123_456_789)
    r = eng.classify(snap)
    assert r.snapshot_ts_ns == snap.ts_ns


def test_classify_does_not_mutate_engine_state() -> None:
    eng = _engine()
    snap_a = _snap(vol_index=55.0)  # CRISIS
    snap_b = _snap(vol_index=18.0, breadth=0.10)  # NEUTRAL
    eng.classify(snap_a)
    r2 = eng.classify(snap_b)
    # If classify mutated state, snap_b would be tainted by snap_a.
    assert r2.regime is MacroRegime.NEUTRAL
