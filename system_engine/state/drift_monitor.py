"""Exponentially-weighted-moving-mean drift monitor.

Tracks a single metric's drift relative to a reference EWMA. Flags
when the deviation exceeds ``drift_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DriftReading:
    ts_ns: int
    metric: str
    value: float
    ewma: float
    deviation: float
    is_drifting: bool


class DriftMonitor:
    """CORE-18."""

    name: str = "drift_monitor"
    spec_id: str = "CORE-18"

    __slots__ = ("_metric", "_alpha", "_drift_threshold", "_ewma", "_seen")

    def __init__(
        self,
        *,
        metric: str,
        alpha: float = 0.1,
        drift_threshold: float = 0.25,
    ) -> None:
        if not metric:
            raise ValueError("metric name must be non-empty")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if drift_threshold <= 0.0:
            raise ValueError("drift_threshold must be positive")
        self._metric = metric
        self._alpha = alpha
        self._drift_threshold = drift_threshold
        self._ewma = 0.0
        self._seen = False

    def observe(self, *, ts_ns: int, value: float) -> DriftReading:
        if not self._seen:
            self._ewma = value
            self._seen = True
            return DriftReading(
                ts_ns=ts_ns,
                metric=self._metric,
                value=value,
                ewma=self._ewma,
                deviation=0.0,
                is_drifting=False,
            )
        prev = self._ewma
        self._ewma = self._alpha * value + (1.0 - self._alpha) * prev
        denom = abs(prev) if abs(prev) > 1e-12 else 1.0
        deviation = abs(value - prev) / denom
        return DriftReading(
            ts_ns=ts_ns,
            metric=self._metric,
            value=value,
            ewma=self._ewma,
            deviation=deviation,
            is_drifting=deviation > self._drift_threshold,
        )


__all__ = ["DriftMonitor", "DriftReading"]
