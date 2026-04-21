"""
execution/adapter_router.py
Router that resolves a trade intent to a concrete exchange adapter.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from execution.adapters.base import BaseAdapter


@dataclass
class AdapterEntry:
    name: str
    adapter: BaseAdapter
    priority: int = 0


class AdapterRouter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: list[AdapterEntry] = []

    def register(self, name: str, adapter: BaseAdapter, priority: int = 0) -> None:
        with self._lock:
            self._entries.append(AdapterEntry(name, adapter, priority))
            self._entries.sort(key=lambda e: -e.priority)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._entries = [e for e in self._entries if e.name != name]

    def route(self, asset: str) -> BaseAdapter | None:
        with self._lock:
            entries = list(self._entries)
        for e in entries:
            supports = getattr(e.adapter, "supports", None)
            if supports is None or supports(asset):
                return e.adapter
        return None

    def registered(self) -> list[str]:
        with self._lock:
            return [e.name for e in self._entries]

    def get_by_name(self, name: str) -> BaseAdapter | None:
        with self._lock:
            for e in self._entries:
                if e.name == name:
                    return e.adapter
        return None

    def entries(self) -> list[AdapterEntry]:
        with self._lock:
            return list(self._entries)


_router: AdapterRouter | None = None
_lock = threading.Lock()


def get_adapter_router() -> AdapterRouter:
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = AdapterRouter()
    return _router
