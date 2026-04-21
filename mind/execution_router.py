"""
mind/execution_router.py
Maps an Indira ExecutionEvent to a concrete exchange adapter.

Strategy: pick the highest-priority adapter whose ``supports(asset)`` returns
True. Routing happens on the fast path but every adapter call is non-blocking.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mind.engine import ExecutionEvent

AdapterFn = Callable[[ExecutionEvent], dict[str, Any]]


@dataclass
class AdapterRegistration:
    name: str
    priority: int
    supports: Callable[[str], bool]
    submit: AdapterFn


@dataclass
class RoutingResult:
    adapter: str
    response: dict[str, Any]


class ExecutionRouter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._adapters: list[AdapterRegistration] = []

    def register(self, adapter: AdapterRegistration) -> None:
        with self._lock:
            self._adapters.append(adapter)
            self._adapters.sort(key=lambda a: -a.priority)

    def clear(self) -> None:
        with self._lock:
            self._adapters.clear()

    def route(self, event: ExecutionEvent) -> RoutingResult | None:
        if event.event_type != "TRADE_EXECUTION" or not event.allowed:
            return None
        with self._lock:
            adapters = list(self._adapters)
        for a in adapters:
            try:
                if a.supports(event.asset):
                    resp = a.submit(event)
                    return RoutingResult(adapter=a.name, response=resp)
            except Exception:
                continue
        return None


_router: ExecutionRouter | None = None
_lock = threading.Lock()


def get_execution_router() -> ExecutionRouter:
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = ExecutionRouter()
    return _router
