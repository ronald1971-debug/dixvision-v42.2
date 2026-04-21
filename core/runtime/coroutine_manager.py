"""
core/runtime/coroutine_manager.py
Tracks long-running background coroutines so shutdown can cancel them cleanly.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable

from .async_runtime import get_async_runtime


class CoroutineManager:
    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()

    def spawn(self, name: str, coro: Awaitable) -> asyncio.Future:
        fut = get_async_runtime().submit(coro)  # type: ignore[arg-type]
        with self._lock:
            self._futures[name] = fut
        return fut

    def cancel(self, name: str) -> bool:
        with self._lock:
            fut = self._futures.pop(name, None)
        if fut is None:
            return False
        fut.cancel()
        return True

    def cancel_all(self) -> None:
        with self._lock:
            items = list(self._futures.items())
            self._futures.clear()
        for _, fut in items:
            fut.cancel()

    def names(self) -> list[str]:
        with self._lock:
            return list(self._futures.keys())


_cm: CoroutineManager | None = None
_lock = threading.Lock()


def get_coroutine_manager() -> CoroutineManager:
    global _cm
    if _cm is None:
        with _lock:
            if _cm is None:
                _cm = CoroutineManager()
    return _cm
