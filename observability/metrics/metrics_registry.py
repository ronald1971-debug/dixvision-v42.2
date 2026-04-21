"""
observability/metrics/metrics_registry.py
Thin facade over system.metrics with an observability-focused namespace.
"""
from __future__ import annotations

import threading
from typing import Any

from system.metrics import get_metrics


class MetricsRegistry:
    def __init__(self) -> None:
        self._m = get_metrics()

    def inc(self, name: str, labels: dict[str, str] | None = None) -> None:
        self._m.increment(name, 1.0, labels or None)

    def observe(self, name: str, value: float) -> None:
        self._m.observe(name, value)

    def snapshot(self) -> dict[str, Any]:
        dump = getattr(self._m, "snapshot", None)
        if callable(dump):
            return dict(dump())
        return {}


_reg: MetricsRegistry | None = None
_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    global _reg
    if _reg is None:
        with _lock:
            if _reg is None:
                _reg = MetricsRegistry()
    return _reg
