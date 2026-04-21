"""
system_monitor/telemetry_ingest.py
Ingest system + exchange telemetry into the hazard detector + metrics.
"""
from __future__ import annotations

import threading

from execution.hazard.detector import get_hazard_detector
from system.metrics import get_metrics


class TelemetryIngest:
    def __init__(self) -> None:
        self._detector = get_hazard_detector()
        self._metrics = get_metrics()

    def feed_tick(self, feed_name: str) -> None:
        self._detector.record_feed_tick(feed_name)
        self._metrics.increment("telemetry.feed_tick", {"feed": feed_name})

    def execution_latency(self, component: str, latency_ms: float) -> None:
        self._detector.record_latency(component, latency_ms)
        self._metrics.observe(f"telemetry.latency.{component}", latency_ms)

    def exchange_ping(self, exchange: str, rtt_ms: float) -> None:
        self._metrics.observe(f"telemetry.exchange.{exchange}.rtt_ms", rtt_ms)


_ti: TelemetryIngest | None = None
_lock = threading.Lock()


def get_telemetry_ingest() -> TelemetryIngest:
    global _ti
    if _ti is None:
        with _lock:
            if _ti is None:
                _ti = TelemetryIngest()
    return _ti
