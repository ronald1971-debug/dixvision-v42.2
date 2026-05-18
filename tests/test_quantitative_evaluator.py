"""P0-B — unit tests for the quantitative promotion-gate evaluator."""

from __future__ import annotations

import pytest

from governance_engine.gates.quantitative_evaluator import (
    DEFAULT_QUANTITATIVE_THRESHOLDS,
    REJECTION_CODE_DRAWDOWN_EXCEEDS_CEILING,
    REJECTION_CODE_INSUFFICIENT_SAMPLES,
    REJECTION_CODE_IS_OOS_DIVERGENCE,
    REJECTION_CODE_SHARPE_BELOW_FLOOR,
    QuantitativeEvaluator,
    QuantitativeMetrics,
    QuantitativeThresholds,
    QuantitativeVerdictKind,
)


def _passing_metrics(**overrides: object) -> QuantitativeMetrics:
    defaults: dict[str, object] = {
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.03,
        "samples": 250,
        "is_score": 0.10,
        "oos_score": 0.10,
        "is_std": 0.05,
    }
    defaults.update(overrides)
    return QuantitativeMetrics(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Defaults + threshold validation
# ---------------------------------------------------------------------------


def test_default_thresholds_match_user_directive() -> None:
    """User-supplied defaults: Sharpe>=1.0, DD<=5%, samples>=200, |IS-OOS|<=0.5σ."""

    t = DEFAULT_QUANTITATIVE_THRESHOLDS
    assert t.sharpe_ratio_min == 1.0
    assert t.max_drawdown_max == 0.05
    assert t.samples_min == 200
    assert t.is_oos_divergence_max_sigma == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sharpe_ratio_min": -0.1},
        {"max_drawdown_max": -0.01},
        {"max_drawdown_max": 1.1},
        {"samples_min": -1},
        {"is_oos_divergence_max_sigma": -0.1},
    ],
)
def test_thresholds_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        QuantitativeThresholds(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": -1},
        {"max_drawdown": -0.01},
        {"is_std": -0.01},
    ],
)
def test_metrics_reject_invalid_values(kwargs: dict[str, object]) -> None:
    base = {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.0,
        "samples": 200,
        "is_score": 0.0,
        "oos_score": 0.0,
        "is_std": 0.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        QuantitativeMetrics(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Passing path
# ---------------------------------------------------------------------------


def test_evaluator_approves_when_all_thresholds_met() -> None:
    ev = QuantitativeEvaluator()
    v = ev.evaluate(_passing_metrics())
    assert v.kind is QuantitativeVerdictKind.APPROVED
    assert v.passed is True
    assert v.rejection_codes == ()
    assert "sharpe=" in v.detail


def test_evaluator_approves_at_exact_thresholds() -> None:
    """Boundary: equal-to-threshold values are inclusive on the passing side."""

    ev = QuantitativeEvaluator()
    v = ev.evaluate(
        _passing_metrics(
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            samples=200,
        )
    )
    assert v.passed is True
    assert v.kind is QuantitativeVerdictKind.APPROVED


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_insufficient_samples_returns_insufficient_data_kind() -> None:
    ev = QuantitativeEvaluator()
    v = ev.evaluate(_passing_metrics(samples=10))
    assert v.kind is QuantitativeVerdictKind.INSUFFICIENT_DATA
    assert v.passed is False
    assert v.rejection_codes == (REJECTION_CODE_INSUFFICIENT_SAMPLES,)


def test_sharpe_below_floor_rejected() -> None:
    ev = QuantitativeEvaluator()
    v = ev.evaluate(_passing_metrics(sharpe_ratio=0.5))
    assert v.kind is QuantitativeVerdictKind.REJECTED
    assert REJECTION_CODE_SHARPE_BELOW_FLOOR in v.rejection_codes


def test_drawdown_above_ceiling_rejected() -> None:
    ev = QuantitativeEvaluator()
    v = ev.evaluate(_passing_metrics(max_drawdown=0.10))
    assert v.kind is QuantitativeVerdictKind.REJECTED
    assert REJECTION_CODE_DRAWDOWN_EXCEEDS_CEILING in v.rejection_codes


def test_is_oos_divergence_rejected_in_sigma_units() -> None:
    # |0.20 - 0.10| / 0.05 == 2σ, > 0.5σ.
    ev = QuantitativeEvaluator()
    v = ev.evaluate(_passing_metrics(is_score=0.20, oos_score=0.10, is_std=0.05))
    assert v.kind is QuantitativeVerdictKind.REJECTED
    assert REJECTION_CODE_IS_OOS_DIVERGENCE in v.rejection_codes


def test_is_oos_divergence_when_is_std_zero_uses_absolute() -> None:
    ev = QuantitativeEvaluator()
    # is_std == 0 → absolute divergence is compared; 0.6 > 0.5.
    v = ev.evaluate(_passing_metrics(is_score=0.6, oos_score=0.0, is_std=0.0))
    assert v.kind is QuantitativeVerdictKind.REJECTED
    assert REJECTION_CODE_IS_OOS_DIVERGENCE in v.rejection_codes


def test_multiple_rejection_codes_are_sorted_and_unique() -> None:
    ev = QuantitativeEvaluator()
    v = ev.evaluate(
        _passing_metrics(
            sharpe_ratio=0.2,
            max_drawdown=0.10,
            is_score=0.20,
            oos_score=0.10,
            is_std=0.05,
        )
    )
    assert v.kind is QuantitativeVerdictKind.REJECTED
    assert list(v.rejection_codes) == sorted(v.rejection_codes)
    assert len(set(v.rejection_codes)) == len(v.rejection_codes)
    assert REJECTION_CODE_SHARPE_BELOW_FLOOR in v.rejection_codes
    assert REJECTION_CODE_DRAWDOWN_EXCEEDS_CEILING in v.rejection_codes
    assert REJECTION_CODE_IS_OOS_DIVERGENCE in v.rejection_codes


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_evaluator_is_pure_replay_byte_identical() -> None:
    ev = QuantitativeEvaluator()
    m = _passing_metrics(sharpe_ratio=0.5, max_drawdown=0.08)
    v1 = ev.evaluate(m)
    v2 = ev.evaluate(m)
    v3 = ev.evaluate(m)
    assert v1 == v2 == v3


def test_evaluator_thresholds_property_exposes_config() -> None:
    custom = QuantitativeThresholds(
        sharpe_ratio_min=2.0,
        max_drawdown_max=0.02,
        samples_min=500,
        is_oos_divergence_max_sigma=0.25,
    )
    ev = QuantitativeEvaluator(thresholds=custom)
    assert ev.thresholds is custom
