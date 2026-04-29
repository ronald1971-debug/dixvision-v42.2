"""Thread-safe wrapper that runs a :class:`BinancePublicWSPump` in a
background asyncio loop, so the FastAPI sync handlers can ``start()`` /
``stop()`` / ``status()`` it without blocking on a coroutine.

Each :class:`FeedRunner` owns:

* one ``threading.Thread`` (daemon — dies with the harness),
* one ``asyncio.AbstractEventLoop`` running on that thread,
* one :class:`BinancePublicWSPump` that runs forever inside the loop.

The runner is *not* a singleton; ``ui/server.py`` instantiates one at
process start and exposes it via ``POST /api/feeds/binance/start``,
``POST /api/feeds/binance/stop``, ``GET /api/feeds/binance/status``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Sequence

from core.contracts.market import MarketTick
from ui.feeds.binance_public_ws import (
    DEFAULT_SYMBOLS,
    BinancePublicWSPump,
    FeedStatus,
    WSConnect,
)

LOG = logging.getLogger(__name__)


class FeedRunner:
    """Owns one asyncio loop + one Binance pump, controlled from sync code."""

    def __init__(
        self,
        sink: Callable[[MarketTick], None],
        *,
        clock_ns: Callable[[], int],
        symbols: Sequence[str] = DEFAULT_SYMBOLS,
        connect: WSConnect | None = None,
    ) -> None:
        self._sink = sink
        self._clock_ns = clock_ns
        self._symbols: tuple[str, ...] = tuple(symbols)
        self._connect = connect
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pump: BinancePublicWSPump | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def status(self) -> FeedStatus:
        with self._lock:
            pump = self._pump
        if pump is None:
            from ui.feeds.binance_public_ws import (
                make_combined_stream_url,
            )
            return FeedStatus(
                running=False,
                symbols=tuple(s.upper() for s in self._symbols),
                url=make_combined_stream_url(self._symbols),
                last_tick_ts_ns=None,
                ticks_received=0,
                errors=0,
            )
        return pump.status()

    def start(
        self, *, symbols: Sequence[str] | None = None
    ) -> FeedStatus:
        """Spawn the background thread + asyncio loop + pump.

        Idempotent: a no-op (returns current status) if the runner is
        already running. ``symbols``, when provided, replaces the
        configured symbol list for *this run only* — the runner's own
        ``self._symbols`` is left untouched so a subsequent
        argument-less ``start()`` reverts to the original config.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._pump is not None:
                    return self._pump.status()
            symbol_set = tuple(symbols) if symbols else self._symbols
            ready = threading.Event()

            def _thread_main() -> None:
                loop = asyncio.new_event_loop()
                try:
                    pump = BinancePublicWSPump(
                        symbol_set,
                        self._sink,
                        clock_ns=self._clock_ns,
                        connect=self._connect,
                    )
                    with self._lock:
                        self._loop = loop
                        self._pump = pump
                    ready.set()
                    loop.run_until_complete(pump.run())
                except Exception:  # noqa: BLE001
                    LOG.exception("binance_public_ws runner crashed")
                    ready.set()
                finally:
                    try:
                        loop.close()
                    finally:
                        with self._lock:
                            self._loop = None
                            self._pump = None

            t = threading.Thread(
                target=_thread_main,
                name="binance-public-ws-pump",
                daemon=True,
            )
            self._thread = t
            t.start()
        # Wait until pump exists so the caller's status() is accurate.
        ready.wait(timeout=5.0)
        return self.status()

    def stop(self, *, timeout: float = 5.0) -> FeedStatus:
        """Signal the pump to exit, join the thread, and return status.

        Idempotent: a no-op (returns current status) if not running.
        """
        with self._lock:
            loop = self._loop
            pump = self._pump
            t = self._thread
        if t is None or not t.is_alive():
            return self.status()
        if loop is not None and pump is not None:
            try:
                loop.call_soon_threadsafe(pump.stop)
            except RuntimeError:
                # Loop already closed.
                pass
        t.join(timeout=timeout)
        with self._lock:
            if self._thread is t and not t.is_alive():
                self._thread = None
        return self.status()
