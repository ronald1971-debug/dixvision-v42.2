"""
execution/hazard/detector.py
DIX VISION v42.2 — Dyon Anomaly Detector

Monitors system metrics and emits SYSTEM_HAZARD_EVENTs.
Runs in background thread. NEVER touches trading execution.
"""
from __future__ import annotations

import threading
import time

from execution.hazard.event_emitter import get_hazard_emitter
from system.config import get as get_config
from system.health_monitor import get_health_monitor


class HazardDetector:
    """Background hazard detection. Dyon's sensory system."""

    def __init__(self) -> None:
        self._emitter = get_hazard_emitter("dyon.detector")
        self._health = get_health_monitor()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_feed_ts: dict = {}
        self._check_interval = float(get_config("hazard.check_interval_seconds", 1.0))
        self._feed_silence_threshold = float(
            get_config("hazard.feed_silence_threshold_seconds", 5.0)
        )
        self._latency_spike_ms = float(
            get_config("hazard.latency_spike_threshold_ms", 100.0)
        )

    def start(self) -> None:
        # ``Thread`` objects cannot be restarted; re-create on every
        # ``start()`` so graceful restart (stop → start) works.
        self._running = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="HazardDetector"
            )
            self._thread.start()

    def stop(self) -> None:
        self._running = False

    def record_feed_tick(self, feed_name: str) -> None:
        """Call when market data arrives — updates liveness."""
        self._last_feed_ts[feed_name] = time.monotonic()

    def record_latency(self, component: str, latency_ms: float) -> None:
        """Call after any execution step to check for spikes."""
        if latency_ms > self._latency_spike_ms:
            self._emitter.latency_spike(component, latency_ms, self._latency_spike_ms)

    def _loop(self) -> None:
        while self._running:
            try:
                self._check_feed_silence()
                self._check_system_health()
            except Exception:
                pass
            time.sleep(self._check_interval)

    def _check_feed_silence(self) -> None:
        now = time.monotonic()
        for feed, last_ts in list(self._last_feed_ts.items()):
            silence = now - last_ts
            if silence > self._feed_silence_threshold:
                self._emitter.feed_silence(feed, silence)

    def _check_system_health(self) -> None:
        status = self._health.get_status()
        for component, healthy in status.items():
            if not healthy:
                self._emitter.system_degradation(
                    f"component={component} status=DEGRADED"
                )

_detector: HazardDetector | None = None
_lock = threading.Lock()

def get_hazard_detector() -> HazardDetector:
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                _detector = HazardDetector()
    return _detector
