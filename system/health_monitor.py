"""
system/health_monitor.py
DIX VISION v42.2 — Health Monitor (Reactive via Telemetry)
"""
from __future__ import annotations

import threading


class HealthMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checks: dict[str, bool] = {}

    def report(self, component: str, healthy: bool) -> None:
        with self._lock:
            self._checks[component] = healthy

    def is_healthy(self, component: str = None) -> bool:
        with self._lock:
            if component:
                return self._checks.get(component, True)
            return all(self._checks.values())

    def get_status(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._checks)

    def print_status(self) -> None:
        status = self.get_status()
        print("\n=== SYSTEM HEALTH ===")
        for k, v in status.items():
            icon = "✅" if v else "❌"
            print(f"  {icon} {k}: {'OK' if v else 'DEGRADED'}")
        print("====================\n")

print_dashboard = None  # assigned after instance creation

_monitor: HealthMonitor | None = None
_lock = threading.Lock()

def get_health_monitor() -> HealthMonitor:
    global _monitor, print_dashboard
    if _monitor is None:
        with _lock:
            if _monitor is None:
                _monitor = HealthMonitor()
                print_dashboard = _monitor.print_status
    return _monitor
