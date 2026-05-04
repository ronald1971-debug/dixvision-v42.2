"""Unit tests for sensory.neuromorphic.contracts."""

from __future__ import annotations

import math

import pytest

from sensory.neuromorphic.contracts import (
    POLARITY_LONG,
    POLARITY_NEUTRAL,
    POLARITY_SHORT,
    AnomalyPulse,
    PulseSignal,
    RiskPulse,
)

# ----- PulseSignal --------------------------------------------------- #


def _ok_pulse(**overrides: object) -> PulseSignal:
    kwargs: dict[str, object] = {
        "ts_ns": 1,
        "source": "BINANCE",
        "symbol": "BTCUSDT",
        "polarity": POLARITY_LONG,
        "intensity": 0.5,
        "sample_count": 4,
    }
    kwargs.update(overrides)
    return PulseSignal(**kwargs)  # type: ignore[arg-type]


def test_pulse_minimal() -> None:
    p = _ok_pulse()
    assert p.polarity == POLARITY_LONG
    assert p.intensity == 0.5
    assert p.sample_count == 4
    assert dict(p.evidence) == {}


def test_pulse_frozen() -> None:
    p = _ok_pulse()
    with pytest.raises(AttributeError):
        p.intensity = 0.9  # type: ignore[misc]


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("symbol", ""),
        ("polarity", "BULLISH"),
        ("intensity", -0.01),
        ("intensity", 1.01),
        ("sample_count", 0),
        ("sample_count", -1),
    ],
)
def test_pulse_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok_pulse(**{field: value})


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")]
)
def test_pulse_intensity_rejects_nan_and_infinity(bad: float) -> None:
    """intensity must be finite — INV-15.

    Both ``not (0.0 <= x <= 1.0)`` and ``math.isfinite`` reject NaN
    and Inf, so the validator catches both modes.
    """
    with pytest.raises(ValueError, match="intensity"):
        _ok_pulse(intensity=bad)


@pytest.mark.parametrize(
    "polarity",
    [POLARITY_LONG, POLARITY_SHORT, POLARITY_NEUTRAL],
)
def test_pulse_accepts_canonical_polarities(polarity: str) -> None:
    p = _ok_pulse(polarity=polarity)
    assert p.polarity == polarity


# ----- AnomalyPulse -------------------------------------------------- #


def _ok_anomaly(**overrides: object) -> AnomalyPulse:
    kwargs: dict[str, object] = {
        "ts_ns": 2,
        "source": "system.fast_risk_cache.lag_ns",
        "anomaly_kind": "LATENCY_SPIKE",
        "z_score": 2.5,
        "severity": 0.8,
        "window_size": 32,
    }
    kwargs.update(overrides)
    return AnomalyPulse(**kwargs)  # type: ignore[arg-type]


def test_anomaly_minimal() -> None:
    a = _ok_anomaly()
    assert a.z_score == 2.5
    assert a.severity == 0.8
    assert a.window_size == 32


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("anomaly_kind", ""),
        ("severity", -0.01),
        ("severity", 1.01),
        ("window_size", 0),
        ("window_size", 1),
    ],
)
def test_anomaly_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok_anomaly(**{field: value})


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")]
)
def test_anomaly_z_score_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="z_score"):
        _ok_anomaly(z_score=bad)


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")]
)
def test_anomaly_severity_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="severity"):
        _ok_anomaly(severity=bad)


def test_anomaly_z_score_can_be_negative() -> None:
    """A below-mean sample is a valid anomaly direction."""
    a = _ok_anomaly(z_score=-2.5)
    assert a.z_score == -2.5
    assert math.isfinite(a.z_score)


# ----- RiskPulse ----------------------------------------------------- #


def _ok_risk(**overrides: object) -> RiskPulse:
    kwargs: dict[str, object] = {
        "ts_ns": 3,
        "source": "governance.decision_audit",
        "risk_kind": "REJECT_RATE",
        "risk_score": 0.25,
        "sample_count": 100,
    }
    kwargs.update(overrides)
    return RiskPulse(**kwargs)  # type: ignore[arg-type]


def test_risk_minimal() -> None:
    r = _ok_risk()
    assert r.risk_score == 0.25
    assert r.sample_count == 100


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("risk_kind", ""),
        ("risk_score", -0.01),
        ("risk_score", 1.01),
        ("sample_count", 0),
        ("sample_count", -1),
    ],
)
def test_risk_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok_risk(**{field: value})


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")]
)
def test_risk_score_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="risk_score"):
        _ok_risk(risk_score=bad)
