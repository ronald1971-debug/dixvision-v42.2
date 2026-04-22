"""
execution/monitoring/neuromorphic_detector.py
DIX VISION v42.2 — Dyon-side neuromorphic sensor (observe + emit only).

Phase 0 stub: event-emitting scaffolding. No LSM yet — Phase 3 replaces
the rule-based anomaly detector here with a Liquid State Machine trained
offline to detect silent / temporal system degradation.

Axioms N1..N8 (immutable_core/neuromorphic_axioms.lean) apply:
  - observes system telemetry,
  - emits SYSTEM_ANOMALY_EVENT → Dyon translates to SYSTEM_HAZARD_EVENT,
  - NEVER restarts services, applies patches, or changes governance mode.

authority_lint rule C2 forbids this file from importing any of:
  governance.kernel, mind.fast_execute, execution.engine,
  security.operator, system.fast_risk_cache (mutators),
  core.registry (register).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from state.ledger.event_store import append_event

ANOMALY_TYPES = (
    "LATENCY_DRIFT",
    "SILENT_DATA_FAILURE",
    "MEMORY_PRESSURE_GRADIENT",
    "EVENT_RHYTHM_BREAK",
)


@dataclass
class SystemAnomalyEvent:
    """Wire format for SYSTEM_ANOMALY_EVENT — consumed by Dyon which
    translates it to SYSTEM_HAZARD_EVENT via severity_classifier."""

    type: str                           # one of ANOMALY_TYPES
    severity: float                     # 0.0..1.0
    component: str                      # subsystem name
    window_seconds: float
    timestamp_utc: str
    details: dict[str, Any] = field(default_factory=dict)


class NeuromorphicDetector:
    """Temporal anomaly sensor for Dyon.

    Phase 0: rule-based threshold + trend detector over telemetry windows.
    Phase 3: LSM backend with frozen ONNX weights.
    """

    name = "neuromorphic_detector"
    heartbeat_interval: float = 1.0     # seconds — N5 dead-man

    def __init__(self) -> None:
        self._last_emission = time.monotonic()
        self._last_tick_seen = time.monotonic()

    # ── Detection entrypoints ────────────────────────────────────────
    def on_telemetry(self, sample: dict[str, Any]) -> SystemAnomalyEvent | None:
        """Accept one telemetry sample, maybe emit an anomaly event.

        Expected keys:
          - component: str
          - cpu_pct: float  0..100
          - mem_pct: float  0..100
          - event_rate_hz: float    observed vs baseline
          - latency_ms_p99: float   rolling window
          - last_tick_gap_ms: float
        """
        component = str(sample.get("component", "unknown"))

        latency = float(sample.get("latency_ms_p99", 0.0))
        if latency > 500.0:
            return self._emit("LATENCY_DRIFT",
                              severity=min(latency / 2000.0, 1.0),
                              component=component,
                              window_seconds=60.0,
                              details={"latency_ms_p99": latency})

        mem = float(sample.get("mem_pct", 0.0))
        if mem > 90.0:
            return self._emit("MEMORY_PRESSURE_GRADIENT",
                              severity=min((mem - 80.0) / 20.0, 1.0),
                              component=component,
                              window_seconds=60.0,
                              details={"mem_pct": mem})

        gap = float(sample.get("last_tick_gap_ms", 0.0))
        if gap > 5_000.0:
            return self._emit("EVENT_RHYTHM_BREAK",
                              severity=min(gap / 30_000.0, 1.0),
                              component=component,
                              window_seconds=gap / 1000.0,
                              details={"tick_gap_ms": gap})

        rate = float(sample.get("event_rate_hz", -1.0))
        expected = float(sample.get("expected_rate_hz", -1.0))
        if 0 < expected and rate >= 0 and rate < expected * 0.2:
            return self._emit("SILENT_DATA_FAILURE",
                              severity=1.0 - (rate / max(expected, 1e-9)),
                              component=component,
                              window_seconds=60.0,
                              details={"event_rate_hz": rate,
                                       "expected_rate_hz": expected})

        self._last_tick_seen = time.monotonic()
        return None

    # ── Self-monitoring (N5 dead-man) ────────────────────────────────
    def check_self(self) -> bool:
        """Detector must prove it's alive.

        Returns False if no sample has been processed recently; system
        dead-man reads this and fails closed — axiom N5.
        """
        silence = time.monotonic() - self._last_tick_seen
        return silence < (self.heartbeat_interval * 3)

    # ── internals ────────────────────────────────────────────────────
    def _emit(self, kind: str, *, severity: float, component: str,
              window_seconds: float,
              details: dict[str, Any]) -> SystemAnomalyEvent:
        from system.time_source import now
        ts = now().utc_time.isoformat()
        event = SystemAnomalyEvent(
            type=kind, severity=severity, component=component,
            window_seconds=window_seconds, timestamp_utc=ts, details=details,
        )
        self._last_emission = time.monotonic()
        self._last_tick_seen = time.monotonic()
        try:
            append_event("NEUROMORPHIC", kind, self.name, {
                "severity": severity, "component": component,
                "window_seconds": window_seconds, "details": details,
            })
        except Exception:
            pass   # ledger failure never blocks detection
        return event


_detector: NeuromorphicDetector | None = None


def get_neuromorphic_detector() -> NeuromorphicDetector:
    global _detector
    if _detector is None:
        _detector = NeuromorphicDetector()
    return _detector
