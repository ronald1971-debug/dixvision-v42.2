"""
mind/sources/market_streams.py
Market data stream registry. Sources register a pull function; the engine
polls them and feeds normalized MarketTick objects downstream.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from .source_types import MarketTick

PullFn = Callable[[], list[MarketTick]]


class MarketStreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, PullFn] = {}

    def register(self, name: str, pull: PullFn) -> None:
        with self._lock:
            self._sources[name] = pull

    def unregister(self, name: str) -> None:
        with self._lock:
            self._sources.pop(name, None)

    def pull_all(self) -> list[MarketTick]:
        out: list[MarketTick] = []
        with self._lock:
            sources = list(self._sources.items())
        for name, fn in sources:
            try:
                ticks = fn()
                if ticks:
                    out.extend(ticks)
            except Exception:
                continue
        return out


_registry: MarketStreamRegistry | None = None
_lock = threading.Lock()


def get_market_streams() -> MarketStreamRegistry:
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = MarketStreamRegistry()
    return _registry
