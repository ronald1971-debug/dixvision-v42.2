"""Background runner for :class:`RaydiumPoolPoller` (D2).

Mirrors :class:`ui.feeds.runner.FeedRunner` /
:class:`ui.feeds.news_runner.CoinDeskRSSFeedRunner`:

* one daemon ``threading.Thread`` and one ``asyncio`` loop,
* one :class:`RaydiumPoolPoller` consumed forever inside the loop,
* sync ``start`` / ``stop`` / ``status`` API for the FastAPI sync
  handlers in ``ui/server.py``.

Determinism (INV-15): caller supplies ``clock_ns``; the runner itself
never reads a wall clock. The snapshot sink is caller-supplied so the
harness keeps the bus-publish responsibility (HARDEN-03 producer split).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from core.contracts.launches import PoolSnapshot
from ui.feeds.raydium_pools import (
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_RETRY_DELAY_MAX_S,
    DEFAULT_RETRY_DELAY_S,
    RAYDIUM_PAIRS_URL,
    ClientFactory,
    RaydiumPoolPoller,
    RaydiumPoolStatus,
)

LOG = logging.getLogger(__name__)


class RaydiumPoolFeedRunner:
    """Owns one asyncio loop + one Raydium pool poller, controlled from sync code."""

    def __init__(
        self,
        sink: Callable[[PoolSnapshot], None],
        *,
        clock_ns: Callable[[], int],
        client_factory: ClientFactory | None = None,
        url: str = RAYDIUM_PAIRS_URL,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
        retry_delay_max_s: float = DEFAULT_RETRY_DELAY_MAX_S,
    ) -> None:
        self._sink = sink
        self._clock_ns = clock_ns
        self._client_factory = client_factory
        self._url = url
        self._poll_interval_s = poll_interval_s
        self._retry_delay_s = retry_delay_s
        self._retry_delay_max_s = retry_delay_max_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._poller: RaydiumPoolPoller | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _provisional_status(self, *, running: bool) -> RaydiumPoolStatus:
        return RaydiumPoolStatus(
            running=running,
            url=self._url,
            last_poll_ts_ns=None,
            snapshots_emitted=0,
            errors=0,
        )

    def status(self) -> RaydiumPoolStatus:
        with self._lock:
            poller = self._poller
        if poller is None:
            return self._provisional_status(running=False)
        return poller.status()

    def start(self) -> RaydiumPoolStatus:
        ready = threading.Event()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._poller is not None:
                    return self._poller.status()
                return self._provisional_status(running=True)

            def _thread_main() -> None:
                loop = asyncio.new_event_loop()
                try:
                    poller = RaydiumPoolPoller(
                        self._sink,
                        clock_ns=self._clock_ns,
                        client_factory=self._client_factory,
                        url=self._url,
                        poll_interval_s=self._poll_interval_s,
                        retry_delay_s=self._retry_delay_s,
                        retry_delay_max_s=self._retry_delay_max_s,
                    )
                    with self._lock:
                        self._loop = loop
                        self._poller = poller
                    ready.set()
                    loop.run_until_complete(poller.run())
                except Exception:  # noqa: BLE001
                    LOG.exception("raydium runner: thread crashed")
                    ready.set()
                finally:
                    try:
                        loop.close()
                    finally:
                        with self._lock:
                            self._loop = None
                            self._poller = None

            thread = threading.Thread(
                target=_thread_main,
                name="raydium-pool-feed-runner",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        ready.wait(timeout=5.0)
        return self.status()

    def stop(self) -> RaydiumPoolStatus:
        with self._lock:
            poller = self._poller
            loop = self._loop
            thread = self._thread
        if poller is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(poller.stop)
            except RuntimeError:
                pass
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            if self._thread is thread and (
                thread is None or not thread.is_alive()
            ):
                self._thread = None
        return self.status()


__all__ = ["RaydiumPoolFeedRunner"]
