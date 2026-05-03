"""Background runner for :class:`PumpFunLaunchPump` (D2).

Mirrors :class:`ui.feeds.runner.FeedRunner` /
:class:`ui.feeds.news_runner.CoinDeskRSSFeedRunner`:

* one daemon ``threading.Thread`` and one ``asyncio`` loop,
* one :class:`PumpFunLaunchPump` consumed forever inside the loop,
* sync ``start`` / ``stop`` / ``status`` API for the FastAPI sync
  handlers in ``ui/server.py``.

Determinism (INV-15): caller supplies ``clock_ns``; the runner itself
never reads a wall clock. The launch sink is caller-supplied so the
harness keeps the bus-publish responsibility (HARDEN-03 producer split).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from core.contracts.launches import LaunchEvent
from ui.feeds.pumpfun_ws import (
    DEFAULT_RECONNECT_DELAY_MAX_S,
    DEFAULT_RECONNECT_DELAY_S,
    PUMPPORTAL_WS_URL,
    PumpFunLaunchPump,
    PumpFunStatus,
    WSConnect,
)

LOG = logging.getLogger(__name__)


class PumpFunFeedRunner:
    """Owns one asyncio loop + one Pump.fun pump, controlled from sync code."""

    def __init__(
        self,
        sink: Callable[[LaunchEvent], None],
        *,
        clock_ns: Callable[[], int],
        connect: WSConnect | None = None,
        url: str = PUMPPORTAL_WS_URL,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
    ) -> None:
        self._sink = sink
        self._clock_ns = clock_ns
        self._connect = connect
        self._url = url
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pump: PumpFunLaunchPump | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _provisional_status(self, *, running: bool) -> PumpFunStatus:
        return PumpFunStatus(
            running=running,
            url=self._url,
            last_launch_ts_ns=None,
            launches_received=0,
            errors=0,
        )

    def status(self) -> PumpFunStatus:
        with self._lock:
            pump = self._pump
        if pump is None:
            return self._provisional_status(running=False)
        return pump.status()

    def start(self) -> PumpFunStatus:
        ready = threading.Event()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._pump is not None:
                    return self._pump.status()
                return self._provisional_status(running=True)

            def _thread_main() -> None:
                loop = asyncio.new_event_loop()
                try:
                    pump = PumpFunLaunchPump(
                        self._sink,
                        clock_ns=self._clock_ns,
                        connect=self._connect,
                        url=self._url,
                        reconnect_delay_s=self._reconnect_delay_s,
                        reconnect_delay_max_s=(
                            self._reconnect_delay_max_s
                        ),
                    )
                    with self._lock:
                        self._loop = loop
                        self._pump = pump
                    ready.set()
                    loop.run_until_complete(pump.run())
                except Exception:  # noqa: BLE001
                    LOG.exception("pumpfun runner: thread crashed")
                    ready.set()
                finally:
                    try:
                        loop.close()
                    finally:
                        with self._lock:
                            self._loop = None
                            self._pump = None

            thread = threading.Thread(
                target=_thread_main,
                name="pumpfun-feed-runner",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        ready.wait(timeout=5.0)
        return self.status()

    def stop(self) -> PumpFunStatus:
        with self._lock:
            pump = self._pump
            loop = self._loop
            thread = self._thread
        if pump is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(pump.stop)
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


__all__ = ["PumpFunFeedRunner"]
