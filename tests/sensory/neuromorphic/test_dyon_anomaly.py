"""Unit tests for sensory.neuromorphic.dyon_anomaly (NEUR-02)."""

from __future__ import annotations

import math

import pytest

from sensory.neuromorphic.dyon_anomaly import detect_anomaly


def test_anomaly_stationary_window_zero_severity() -> None:
    a = detect_anomaly(
        ts_ns=1,
        source="metric.x",
        anomaly_kind="OUTLIER",
        window=[5.0, 5.0, 5.0, 5.0],
    )
    assert a.z_score == 0.0
    assert a.severity == 0.0
    assert a.window_size == 4


def test_anomaly_3_sigma_saturates() -> None:
    """At |z| >= saturation_z (default 3.0) severity reaches 1.0.

    For ``(n-1)`` baseline samples plus one outlier, the population
    z-score of the outlier converges to ``sqrt(n-1)`` regardless of
    the magnitudes (it's a ratio of deviations). To exceed ``z=3``
    the window therefore needs ``n >= 10``.
    """
    big_window = [1.0] * 11 + [100.0]
    a = detect_anomaly(
        ts_ns=2,
        source="metric.x",
        anomaly_kind="OUTLIER",
        window=big_window,
    )
    # z ≈ sqrt(11) ≈ 3.32 > 3.0 → severity saturates.
    assert a.severity == 1.0
    assert a.z_score > 3.0
    assert math.isfinite(a.z_score)


def test_anomaly_below_mean_negative_z() -> None:
    a = detect_anomaly(
        ts_ns=3,
        source="metric.x",
        anomaly_kind="DROP",
        window=[10.0, 10.0, 10.0, 5.0],
    )
    # mu = 8.75, latest = 5 → z negative.
    assert a.z_score < 0
    assert a.severity > 0


def test_anomaly_clip_uses_saturation_z() -> None:
    """Loosening saturation_z lowers severity for the same window."""
    window = [1.0] * 11 + [100.0]
    tight = detect_anomaly(
        ts_ns=4,
        source="metric.x",
        anomaly_kind="OUTLIER",
        window=window,
        saturation_z=3.0,
    )
    loose = detect_anomaly(
        ts_ns=4,
        source="metric.x",
        anomaly_kind="OUTLIER",
        window=window,
        saturation_z=10.0,
    )
    assert tight.severity == 1.0
    assert loose.severity < 1.0
    # Same z_score regardless of saturation_z (it's only a clip
    # parameter on severity).
    assert tight.z_score == pytest.approx(loose.z_score)


def test_anomaly_window_too_small() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        detect_anomaly(
            ts_ns=5,
            source="metric.x",
            anomaly_kind="OUTLIER",
            window=[1.0],
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_anomaly_window_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        detect_anomaly(
            ts_ns=6,
            source="metric.x",
            anomaly_kind="OUTLIER",
            window=[1.0, bad, 1.0],
        )


@pytest.mark.parametrize(
    "bad",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_anomaly_rejects_bad_saturation_z(bad: float) -> None:
    with pytest.raises(ValueError, match="saturation_z"):
        detect_anomaly(
            ts_ns=7,
            source="metric.x",
            anomaly_kind="OUTLIER",
            window=[1.0, 2.0, 3.0],
            saturation_z=bad,
        )


def test_anomaly_is_deterministic_inv15() -> None:
    window = [1.0, 2.0, 1.0, 2.0, 50.0]
    a1 = detect_anomaly(
        ts_ns=8,
        source="metric.x",
        anomaly_kind="OUTLIER",
        window=window,
    )
    a2 = detect_anomaly(
        ts_ns=8,
        source="metric.x",
        anomaly_kind="OUTLIER",
        window=window,
    )
    assert a1 == a2
