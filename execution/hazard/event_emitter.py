"""
execution/hazard/event_emitter.py
DIX VISION v42.2 — SYSTEM_HAZARD_EVENT Emitter

Dyon ONLY calls this to report problems.
No execution authority. Sense + report + escalate only.
"""
from __future__ import annotations

from typing import Any

from execution.hazard.async_bus import HazardEvent, HazardSeverity, HazardType, get_hazard_bus
from state.ledger.event_store import append_event


class HazardEmitter:
    """
    Dyon's only communication channel for system problems.
    Every hazard is:
      1. Emitted to async bus (non-blocking)
      2. Written to ledger (async, separate thread)
    """
    def __init__(self, source: str = "dyon") -> None:
        self.source = source
        self._bus = get_hazard_bus()

    def emit(self, hazard_type: HazardType, severity: HazardSeverity,
             details: dict[str, Any] = None) -> HazardEvent:
        event = HazardEvent(
            hazard_type=hazard_type,
            severity=severity,
            source=self.source,
            details=details or {},
        )
        # Non-blocking emit to governance
        self._bus.emit(event)
        # Write to ledger (async, does not block caller)
        try:
            append_event("HAZARD", hazard_type.value, self.source, {
                "severity": severity.value,
                "details": details or {},
                "timestamp_utc": event.timestamp_utc,
            })
        except Exception:
            pass  # ledger failure never blocks detection
        return event

    # ── Convenience methods for common hazard types ───────────────────────────

    def feed_silence(self, source: str, silence_seconds: float) -> HazardEvent:
        return self.emit(HazardType.FEED_SILENCE, HazardSeverity.HIGH,
                         {"source": source, "silence_seconds": silence_seconds})

    def exchange_timeout(self, exchange: str, timeout_ms: float) -> HazardEvent:
        sev = HazardSeverity.CRITICAL if timeout_ms > 10_000 else HazardSeverity.HIGH
        return self.emit(HazardType.EXCHANGE_TIMEOUT, sev,
                         {"exchange": exchange, "timeout_ms": timeout_ms})

    def latency_spike(self, component: str, latency_ms: float,
                      threshold_ms: float) -> HazardEvent:
        return self.emit(HazardType.EXECUTION_LATENCY_SPIKE, HazardSeverity.MEDIUM,
                         {"component": component, "latency_ms": latency_ms,
                          "threshold_ms": threshold_ms})

    def api_failure(self, exchange: str, error: str) -> HazardEvent:
        return self.emit(HazardType.API_CONNECTIVITY_FAILURE, HazardSeverity.HIGH,
                         {"exchange": exchange, "error": error})

    def system_degradation(self, details: str) -> HazardEvent:
        return self.emit(HazardType.SYSTEM_DEGRADATION, HazardSeverity.HIGH,
                         {"details": details})


_emitter: HazardEmitter | None = None
_lock = __import__("threading").Lock()

def get_hazard_emitter(source: str = "dyon") -> HazardEmitter:
    global _emitter
    if _emitter is None:
        with _lock:
            if _emitter is None:
                _emitter = HazardEmitter(source)
    return _emitter
