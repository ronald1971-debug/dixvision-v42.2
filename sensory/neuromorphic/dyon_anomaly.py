"""NEUR-02 — Dyon anomaly perception.

Pure function over a fixed-size window of numeric metric samples.
Emits an :class:`AnomalyPulse` describing how far the most recent
sample deviates from the window's mean.

Authority:
    No engine imports, no I/O, no clock. The sensor NEVER emits a
    HazardEvent — Dyon is the sole party that decides whether
    ``severity`` warrants escalation.

Algorithm (v1, intentionally simple):

* Require ``len(window) >= 2`` (sample variance needs at least two
  observations).
* Compute the population mean ``mu`` and stddev ``sigma`` over the
  window.
* If ``sigma == 0`` → ``z_score = 0`` and ``severity = 0`` (a
  perfectly stationary window has no notion of "anomaly"; the
  caller should rely on AbsoluteThreshold sensors for that case).
* Otherwise ``z = (latest - mu) / sigma``. ``severity = clip(|z| /
  saturation_z, 0, 1)`` so that ``severity = 1`` at ``|z| =
  saturation_z`` and saturates beyond.

The default ``saturation_z`` of ``3.0`` matches the conventional
3-sigma anomaly threshold; callers can tighten or loosen it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from sensory.neuromorphic.contracts import AnomalyPulse


def detect_anomaly(
    *,
    ts_ns: int,
    source: str,
    anomaly_kind: str,
    window: Sequence[float],
    saturation_z: float = 3.0,
    evidence: Mapping[str, str] | None = None,
) -> AnomalyPulse:
    """Detect an anomaly in the trailing metric window.

    Args:
        ts_ns: Caller-supplied sample timestamp.
        source: Stable identifier of the metric stream.
        anomaly_kind: Categorical label.
        window: Trailing numeric samples (oldest first). The most
            recent sample is ``window[-1]``. Must contain at least
            2 finite values; +Inf, -Inf, and NaN are rejected.
        saturation_z: ``|z|`` value at which severity reaches 1.0.
            Must be finite and ``> 0``. Default is ``3.0`` (the
            classic 3-sigma anomaly threshold).
        evidence: Optional structural metadata.

    Returns:
        :class:`AnomalyPulse`. ``severity == 0`` means the window
        is perfectly stationary or the latest sample sits exactly
        on the mean.

    Raises:
        ValueError: If ``window`` has fewer than 2 samples, contains
            a non-finite value, or ``saturation_z`` is non-positive
            / non-finite.
    """
    if len(window) < 2:
        raise ValueError(
            "detect_anomaly.window must contain at least 2 samples"
        )
    if not math.isfinite(saturation_z) or saturation_z <= 0.0:
        raise ValueError(
            "detect_anomaly.saturation_z must be finite and > 0"
        )

    for value in window:
        if not math.isfinite(value):
            raise ValueError(
                "detect_anomaly.window samples must all be finite"
            )

    n = len(window)
    mu = sum(window) / n
    # Population stddev (1/n divisor) keeps the formula stable at
    # n=2 and avoids the n/(n-1) correction debate; the caller's
    # ``saturation_z`` choice already absorbs the constant factor.
    variance = sum((x - mu) ** 2 for x in window) / n
    sigma = math.sqrt(variance)
    latest = window[-1]

    if sigma == 0.0:
        z_score = 0.0
        severity = 0.0
    else:
        z_score = (latest - mu) / sigma
        severity = abs(z_score) / saturation_z
        # Defensive clip — ``saturation_z > 0`` already so the
        # ratio is non-negative.
        severity = max(0.0, min(1.0, severity))

    return AnomalyPulse(
        ts_ns=ts_ns,
        source=source,
        anomaly_kind=anomaly_kind,
        z_score=z_score,
        severity=severity,
        window_size=n,
        evidence=dict(evidence) if evidence else {},
    )


__all__ = ["detect_anomaly"]
