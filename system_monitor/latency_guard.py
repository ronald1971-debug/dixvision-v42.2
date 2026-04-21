"""system_monitor.latency_guard \u2014 p99 hot-path latency watchdog.

A rolling deque of fast_execute latencies (microseconds). If p99 crosses
the threshold, emits SYSTEM/LATENCY_GUARD_TRIPPED and the governance
layer flips into SAFE_MODE (INDIRA pauses new trades, cancels resting).

No DB, no network. Deque bounded at 4096 samples.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from state.ledger.writer import get_writer

DEFAULT_P99_BUDGET_US = 5000.0                                   # 5 ms
DEFAULT_WINDOW = 4096


@dataclass
class LatencySnapshot:
    n: int
    p50_us: float
    p95_us: float
    p99_us: float
    budget_us: float
    tripped: bool

    def as_dict(self) -> dict:
        return {
            "n": self.n, "p50_us": self.p50_us, "p95_us": self.p95_us,
            "p99_us": self.p99_us, "budget_us": self.budget_us,
            "tripped": self.tripped,
        }


def _pct(xs: list, p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = int(round((len(s) - 1) * p))
    return float(s[idx])


class LatencyGuard:
    def __init__(self, *, budget_us: float = DEFAULT_P99_BUDGET_US,
                 window: int = DEFAULT_WINDOW) -> None:
        self._budget = float(budget_us)
        self._window = int(window)
        self._samples: deque[float] = deque(maxlen=self._window)
        self._tripped = False
        self._lock = threading.RLock()

    def observe(self, microseconds: float) -> None:
        with self._lock:
            self._samples.append(float(microseconds))
            if len(self._samples) < 100:
                return
            p99 = _pct(list(self._samples), 0.99)
            if p99 > self._budget and not self._tripped:
                self._tripped = True
                get_writer().write("SYSTEM", "LATENCY_GUARD_TRIPPED",
                                   "GOVERNANCE",
                                   {"p99_us": round(p99, 1),
                                    "budget_us": self._budget,
                                    "n_samples": len(self._samples)})

    def snapshot(self) -> LatencySnapshot:
        with self._lock:
            xs = list(self._samples)
            return LatencySnapshot(
                n=len(xs),
                p50_us=round(_pct(xs, 0.5), 1),
                p95_us=round(_pct(xs, 0.95), 1),
                p99_us=round(_pct(xs, 0.99), 1),
                budget_us=self._budget,
                tripped=self._tripped,
            )

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._tripped = False


_singleton: LatencyGuard | None = None
_lock = threading.Lock()


def get_latency_guard() -> LatencyGuard:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = LatencyGuard()
    return _singleton


__all__ = ["LatencyGuard", "LatencySnapshot", "get_latency_guard"]
