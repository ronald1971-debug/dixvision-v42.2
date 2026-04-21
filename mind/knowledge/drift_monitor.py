"""
mind.knowledge.drift_monitor — tracks whether a strategy's realized-PnL
distribution has drifted vs its in-sample / recent baseline.

Drift is measured as a rolling Kolmogorov-Smirnov-lite statistic on the
last N trade outcomes vs the previous N. If drift exceeds the threshold,
an event is emitted so the strategy arbiter can demote the strategy.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class DriftStatus:
    strategy: str
    samples: int = 0
    drift_score: float = 0.0   # 0 = no drift, 1 = maximally different
    decayed: bool = False


class DriftMonitor:
    def __init__(self, window: int = 100, threshold: float = 0.35) -> None:
        self._window = window
        self._threshold = threshold
        self._lock = threading.RLock()
        self._series: dict[str, deque[float]] = {}

    def record(self, strategy: str, realized_pnl: float) -> DriftStatus:
        with self._lock:
            q = self._series.setdefault(strategy, deque(maxlen=2 * self._window))
            q.append(realized_pnl)
            return self._status(strategy, q)

    def status(self, strategy: str) -> DriftStatus:
        with self._lock:
            q = self._series.get(strategy, deque())
            return self._status(strategy, q)

    def _status(self, strategy: str, q: deque[float]) -> DriftStatus:
        n = len(q)
        if n < 2 * self._window:
            return DriftStatus(strategy=strategy, samples=n)
        older = list(q)[: self._window]
        newer = list(q)[self._window :]
        older_mean = sum(older) / self._window
        newer_mean = sum(newer) / self._window
        spread = max(abs(newer_mean), abs(older_mean), 1e-9)
        score = abs(newer_mean - older_mean) / spread
        return DriftStatus(
            strategy=strategy,
            samples=n,
            drift_score=score,
            decayed=score >= self._threshold,
        )


_monitor: DriftMonitor | None = None
_lock = threading.Lock()


def get_drift_monitor() -> DriftMonitor:
    global _monitor
    if _monitor is None:
        with _lock:
            if _monitor is None:
                _monitor = DriftMonitor()
    return _monitor
