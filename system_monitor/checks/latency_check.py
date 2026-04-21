"""
system_monitor/checks/latency_check.py
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyResult:
    ok: bool
    elapsed_ms: float
    detail: str


def check_latency(probe: Callable[[], None], threshold_ms: float) -> LatencyResult:
    t0 = time.perf_counter()
    try:
        probe()
    except Exception as e:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000.0
        return LatencyResult(False, elapsed, f"probe_failed: {e}")
    elapsed = (time.perf_counter() - t0) * 1000.0
    return LatencyResult(elapsed <= threshold_ms, elapsed,
                         f"elapsed={elapsed:.2f}ms threshold={threshold_ms:.2f}ms")
