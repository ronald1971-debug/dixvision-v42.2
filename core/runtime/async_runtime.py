"""
core/runtime/async_runtime.py
Owns a single asyncio event loop running on a dedicated thread. Used by the
hazard bus, async ledger writes, and other non-hot-path async workloads.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import Any


class AsyncRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="DIX-AsyncRuntime"
            )
            self._thread.start()

    def _run(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Awaitable[Any]) -> asyncio.Future[Any]:
        self.start()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    def stop(self) -> None:
        with self._lock:
            if not self._started or self._loop is None:
                return
            loop = self._loop
            self._started = False
        loop.call_soon_threadsafe(loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)


_rt: AsyncRuntime | None = None
_lock = threading.Lock()


def get_async_runtime() -> AsyncRuntime:
    global _rt
    if _rt is None:
        with _lock:
            if _rt is None:
                _rt = AsyncRuntime()
    return _rt
