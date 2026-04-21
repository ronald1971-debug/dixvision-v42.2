"""
system_monitor/checks/clock_sync_check.py
Detects monotonic/walltime drift between successive samples.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ClockSyncResult:
    ok: bool
    drift_ms: float
    detail: str


def check_clock_sync(max_drift_ms: float = 100.0) -> ClockSyncResult:
    t0_wall = time.time()
    t0_mono = time.monotonic()
    time.sleep(0.01)
    t1_wall = time.time()
    t1_mono = time.monotonic()
    drift_ms = abs((t1_wall - t0_wall) - (t1_mono - t0_mono)) * 1000.0
    return ClockSyncResult(drift_ms <= max_drift_ms, drift_ms,
                            f"drift={drift_ms:.3f}ms threshold={max_drift_ms:.1f}ms")
