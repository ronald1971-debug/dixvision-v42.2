"""
system/metrics.py
DIX VISION v42.2 — Prometheus Metrics

Tracks: trade_latency_ms, hazard_detection_time_ms,
        governance_decision_time_ms, circuit_breaker_triggers.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsSink:
    """In-memory metrics sink (Prometheus exporter added in Phase 7)."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0, labels: dict = None) -> None:
        key = f"{name}:{labels}" if labels else name
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._histograms[name].append(value_ms)
            if len(self._histograms[name]) > 10_000:
                self._histograms[name] = self._histograms[name][-5_000:]

    def p99(self, name: str) -> float:
        with self._lock:
            vals = sorted(self._histograms.get(name, [0.0]))
            if not vals:
                return 0.0
            idx = int(len(vals) * 0.99)
            return vals[min(idx, len(vals)-1)]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "p99": {k: self.p99(k) for k in self._histograms},
            }

class LatencyTimer:
    """Context manager for measuring latency."""
    def __init__(self, sink: MetricsSink, metric_name: str) -> None:
        self._sink = sink
        self._name = metric_name
        self._start = 0.0

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._sink.observe(self._name, elapsed_ms)

_metrics: MetricsSink | None = None
_lock = threading.Lock()

def get_metrics() -> MetricsSink:
    global _metrics
    if _metrics is None:
        with _lock:
            if _metrics is None:
                _metrics = MetricsSink()
    return _metrics
