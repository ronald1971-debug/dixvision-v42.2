"""
mind/sources/onchain_streams.py
Registry for on-chain data sources (blocks, mempool, DEX pools). Producers
return normalized OnchainEvent records.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OnchainEvent:
    chain: str
    kind: str       # e.g. "block", "tx", "swap", "mint"
    payload: dict[str, Any] = field(default_factory=dict)


OnchainFn = Callable[[], list[OnchainEvent]]


class OnchainStreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: dict[str, OnchainFn] = {}

    def register(self, name: str, fn: OnchainFn) -> None:
        with self._lock:
            self._sources[name] = fn

    def pull_all(self) -> list[OnchainEvent]:
        out: list[OnchainEvent] = []
        with self._lock:
            srcs = list(self._sources.items())
        for _, fn in srcs:
            try:
                out.extend(fn())
            except Exception:
                continue
        return out


_registry: OnchainStreamRegistry | None = None


def get_onchain_streams() -> OnchainStreamRegistry:
    global _registry
    if _registry is None:
        _registry = OnchainStreamRegistry()
    return _registry
