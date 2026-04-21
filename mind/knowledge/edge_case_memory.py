"""
mind.knowledge.edge_case_memory — remembers trade contexts that hit edge
cases (rejected by exchange, unusual slippage, partial fill, etc.) so the
strategy arbiter can learn to avoid them.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EdgeCase:
    tag: str
    asset: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = ""


class EdgeCaseMemory:
    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = capacity
        self._lock = threading.RLock()
        self._items: list[EdgeCase] = []

    def remember(self, case: EdgeCase) -> None:
        with self._lock:
            self._items.append(case)
            if len(self._items) > self._capacity:
                self._items = self._items[-self._capacity :]

    def recent(self, n: int = 50) -> list[EdgeCase]:
        with self._lock:
            return list(self._items[-n:])

    def by_tag(self, tag: str) -> list[EdgeCase]:
        with self._lock:
            return [c for c in self._items if c.tag == tag]


_mem: EdgeCaseMemory | None = None
_lock = threading.Lock()


def get_edge_case_memory() -> EdgeCaseMemory:
    global _mem
    if _mem is None:
        with _lock:
            if _mem is None:
                _mem = EdgeCaseMemory()
    return _mem
