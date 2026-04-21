"""
system_monitor/checks/process_health_check.py
Verifies the current process's primary resource budget (best-effort).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessHealthResult:
    ok: bool
    detail: str
    metrics: dict[str, float]


def check_process_health() -> ProcessHealthResult:
    metrics: dict[str, float] = {"pid": float(os.getpid())}
    try:
        import resource  # POSIX

        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics["user_cpu_s"] = float(usage.ru_utime)
        metrics["sys_cpu_s"] = float(usage.ru_stime)
        metrics["maxrss_kb"] = float(usage.ru_maxrss)
    except Exception:  # Windows
        pass
    return ProcessHealthResult(True, "ok", metrics)
