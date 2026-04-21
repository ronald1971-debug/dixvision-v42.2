"""
system_monitor/anomaly_models.py
Statistical anomaly detection primitives (rolling mean + σ outlier).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt


@dataclass
class AnomalyWindow:
    maxlen: int = 128
    z_threshold: float = 3.0
    _values: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._values = deque(maxlen=self.maxlen)

    def add(self, v: float) -> None:
        self._values.append(float(v))

    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def stddev(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean()
        var = sum((x - m) ** 2 for x in self._values) / (len(self._values) - 1)
        return sqrt(var)

    def is_outlier(self, v: float) -> bool:
        if len(self._values) < 5:
            return False
        sd = self.stddev()
        if sd <= 0:
            return False
        return abs(v - self.mean()) / sd >= self.z_threshold


def is_anomalous(window: AnomalyWindow, v: float) -> bool:
    return window.is_outlier(v)
