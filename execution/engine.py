"""
execution/engine.py
DIX VISION v42.2 — Dyon System Maintenance Engine

Dyon's domain: system health, maintenance, hazard detection.
CANNOT execute trades. CANNOT call exchange trading APIs.
Emits SYSTEM_HAZARD_EVENTs via async bus only.
"""
from __future__ import annotations

import threading
import time

from execution.hazard.detector import get_hazard_detector
from execution.hazard.event_emitter import get_hazard_emitter
from system.audit_logger import get_audit_logger
from system.health_monitor import get_health_monitor
from system.logger import get_logger
from system.state import get_state_manager


class DyonEngine:
    """
    System maintenance and observation engine.

    Can:  monitor health, emit hazards, manage patches, restart services
    Cannot: execute trades, call exchange APIs, override governance
    """
    def __init__(self) -> None:
        self._state = get_state_manager()
        self._log = get_logger("dyon")
        self._health = get_health_monitor()
        self._audit = get_audit_logger()
        self._detector = get_hazard_detector()
        self._emitter = get_hazard_emitter("dyon.engine")
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._detector.start()
        self._running = True
        # ``Thread`` objects cannot be restarted; re-create on every
        # ``start()`` so graceful restart (stop → start) works.
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="Dyon-Engine"
            )
            self._thread.start()
        self._log.info("Dyon system engine started")
        self._audit.log("SYSTEM", "dyon.engine", {"event": "ENGINE_START"})

    def stop(self) -> None:
        self._running = False
        self._detector.stop()

    def _loop(self) -> None:
        while self._running:
            try:
                self._health_check()
                self._state.heartbeat()  # Dyon updates its own domain heartbeat
            except Exception as e:
                self._log.error("Dyon loop error", error=str(e))
            time.sleep(1.0)

    def _health_check(self) -> None:
        state = self._state.get()
        is_ok = state.health >= 0.5 and state.mode not in {"EMERGENCY_HALT"}
        self._health.report("dyon", is_ok)

    def record_feed_tick(self, feed_name: str) -> None:
        """Inform hazard detector that a feed is alive."""
        self._detector.record_feed_tick(feed_name)

    def record_execution_latency(self, component: str, ms: float) -> None:
        """Pass execution latency to hazard detector for spike detection."""
        self._detector.record_latency(component, ms)

_dyon: DyonEngine | None = None
_lock = threading.Lock()

def get_dyon_engine() -> DyonEngine:
    global _dyon
    if _dyon is None:
        with _lock:
            if _dyon is None:
                _dyon = DyonEngine()
    return _dyon
