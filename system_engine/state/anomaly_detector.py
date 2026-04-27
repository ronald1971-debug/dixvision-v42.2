"""Rolling z-score anomaly detector.

Pure-Python, deterministic. Caller feeds samples; the detector
maintains a fixed-size window and reports whether each new sample is
out-of-distribution at ``z_threshold``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnomalyVerdict:
    ts_ns: int
    metric: str
    value: float
    is_anomaly: bool
    z_score: float
    sample_count: int


class AnomalyDetector:
    """CORE-20."""

    name: str = "anomaly_detector"
    spec_id: str = "CORE-20"

    __slots__ = ("_metric", "_window", "_z_threshold", "_min_samples", "_buf")

    def __init__(
        self,
        *,
        metric: str,
        window: int = 64,
        z_threshold: float = 3.0,
        min_samples: int = 8,
    ) -> None:
        if not metric:
            raise ValueError("metric name must be non-empty")
        if window < 4:
            raise ValueError("window must be >= 4")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if min_samples < 4 or min_samples > window:
            raise ValueError("min_samples must be in [4, window]")
        self._metric = metric
        self._window = window
        self._z_threshold = z_threshold
        self._min_samples = min_samples
        self._buf: deque[float] = deque(maxlen=window)

    def observe(self, *, ts_ns: int, value: float) -> AnomalyVerdict:
        n = len(self._buf)
        if n < self._min_samples:
            self._buf.append(value)
            return AnomalyVerdict(
                ts_ns=ts_ns,
                metric=self._metric,
                value=value,
                is_anomaly=False,
                z_score=0.0,
                sample_count=n + 1,
            )
        mean = sum(self._buf) / n
        var = sum((x - mean) ** 2 for x in self._buf) / n
        std = math.sqrt(var)
        z = 0.0 if std == 0.0 else (value - mean) / std
        is_anomaly = abs(z) > self._z_threshold
        self._buf.append(value)
        return AnomalyVerdict(
            ts_ns=ts_ns,
            metric=self._metric,
            value=value,
            is_anomaly=is_anomaly,
            z_score=z,
            sample_count=n + 1,
        )


__all__ = ["AnomalyDetector", "AnomalyVerdict"]
