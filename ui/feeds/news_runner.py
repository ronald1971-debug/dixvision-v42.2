"""Background runner for :class:`CoinDeskRSSPump` + :class:`NewsFanout` (P0-5).

P0-5 from ``PHASE6_action_plan.md``. Wave-news-fusion shipped the three
leaves (PR #118 projection, PR #119 shock sensor, PR #120 fanout) but
no caller in the live process ran them: ``ui/server.py`` instantiated
:class:`FeedRunner` for Binance market data only, leaving the news
pipeline orphaned. This runner closes the loop.

Mirrors the design of :class:`ui.feeds.runner.FeedRunner`:

* Owns one daemon ``threading.Thread`` and one ``asyncio`` loop.
* Holds one :class:`CoinDeskRSSPump` whose sink is a
  :class:`NewsFanout` instance composed at construction time.
* Provides sync ``start`` / ``stop`` / ``status`` API the FastAPI sync
  handlers can call.

Determinism / authority:

* INV-15 — caller supplies ``clock_ns`` so the runner itself never
  reads a wall clock. Both signal and hazard sinks are caller-supplied
  so the harness keeps the bus-publish responsibility.
* HARDEN-03 producer split — preserved by :class:`NewsFanout`; the
  runner only forwards.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from core.coherence.belief_state import BeliefState
from core.contracts.events import HazardEvent, SignalEvent
from system_engine.hazard_sensors import NewsShockSensor
from ui.feeds.coindesk_rss import (
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_RECONNECT_DELAY_MAX_S,
    DEFAULT_RECONNECT_DELAY_S,
    CoinDeskRSSPump,
    HTTPFetch,
    NewsFeedStatus,
)
from ui.feeds.news_fanout import NewsFanout

LOG = logging.getLogger(__name__)


class CoinDeskRSSFeedRunner:
    """Owns one asyncio loop + one CoinDesk RSS pump, controlled from sync code."""

    def __init__(
        self,
        signal_sink: Callable[[SignalEvent], None],
        hazard_sink: Callable[[HazardEvent], None],
        *,
        clock_ns: Callable[[], int],
        sensor: NewsShockSensor | None = None,
        current_belief: Callable[[], BeliefState | None] | None = None,
        fetch: HTTPFetch | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
        url: str | None = None,
    ) -> None:
        self._signal_sink = signal_sink
        self._hazard_sink = hazard_sink
        self._clock_ns = clock_ns
        self._sensor = sensor if sensor is not None else NewsShockSensor()
        self._current_belief = current_belief
        self._fetch = fetch
        self._poll_interval_s = poll_interval_s
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._url = url
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pump: CoinDeskRSSPump | None = None
        self._fanout: NewsFanout | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _provisional_status(self, *, running: bool) -> NewsFeedStatus:
        return NewsFeedStatus(
            running=running,
            source="SRC-NEWS-COINDESK-001",
            url=self._url or "",
            last_poll_ts_ns=None,
            last_item_ts_ns=None,
            items_received=0,
            polls=0,
            errors=0,
        )

    def status(self) -> NewsFeedStatus:
        with self._lock:
            pump = self._pump
        if pump is None:
            return self._provisional_status(running=False)
        return pump.status()

    def _build_fanout(self) -> NewsFanout:
        return NewsFanout(
            signal_sink=self._signal_sink,
            hazard_sink=self._hazard_sink,
            sensor=self._sensor,
            current_belief=self._current_belief,
        )

    def start(self) -> NewsFeedStatus:
        """Spawn the background thread + asyncio loop + pump.

        Idempotent: a no-op (returns the current status) when the
        runner is already running. Mirrors :class:`FeedRunner.start`'s
        narrow-window protection (PR #68 BUG_0001) so a concurrent
        ``start()`` race does not orphan a daemon thread.
        """
        ready = threading.Event()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._pump is not None:
                    return self._pump.status()
                return self._provisional_status(running=True)

            def _thread_main() -> None:
                loop = asyncio.new_event_loop()
                try:
                    fanout = self._build_fanout()
                    pump = CoinDeskRSSPump(
                        sink=fanout,
                        clock_ns=self._clock_ns,
                        fetch=self._fetch,
                        poll_interval_s=self._poll_interval_s,
                        reconnect_delay_s=self._reconnect_delay_s,
                        reconnect_delay_max_s=self._reconnect_delay_max_s,
                        url=self._url,
                        # Pin the SCVS source tag so live status() (after
                        # pump construction) and provisional status()
                        # (before / after the worker thread exits) agree
                        # on ``source="SRC-NEWS-COINDESK-001"``. The
                        # pump's default is ``SOURCE_TAG = "COINDESK"``.
                        source="SRC-NEWS-COINDESK-001",
                    )
                    with self._lock:
                        self._loop = loop
                        self._pump = pump
                        self._fanout = fanout
                    ready.set()
                    loop.run_until_complete(pump.run())
                except Exception:  # noqa: BLE001
                    LOG.exception("coindesk_rss runner: thread crashed")
                    ready.set()
                finally:
                    try:
                        loop.close()
                    finally:
                        with self._lock:
                            self._loop = None
                            self._pump = None
                            self._fanout = None

            thread = threading.Thread(
                target=_thread_main,
                name="coindesk-rss-feed-runner",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        # Wait until pump exists outside the lock so the worker thread
        # can acquire it to publish ``self._pump`` (mirrors
        # ``FeedRunner.start``).
        ready.wait(timeout=5.0)
        return self.status()

    def stop(self) -> NewsFeedStatus:
        """Signal the run loop to exit and wait for thread shutdown."""
        with self._lock:
            pump = self._pump
            loop = self._loop
            thread = self._thread
        if pump is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(pump.stop)
            except RuntimeError:
                # TOCTOU: the worker thread may have closed the loop
                # between the lock-protected snapshot above and this
                # call. Treat as already-stopped (mirrors
                # ``FeedRunner.stop`` PR #68 narrow-window handling).
                pass
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            # Only clear when the join actually drained this thread; if
            # join timed out the thread is still alive and a subsequent
            # ``start()`` must NOT bypass the idempotency guard and
            # spawn a second pump (mirrors ``FeedRunner.stop``
            # ``ui/feeds/runner.py:168-170``).
            if self._thread is thread and (
                thread is None or not thread.is_alive()
            ):
                self._thread = None
        return self._provisional_status(running=False)


__all__ = ["CoinDeskRSSFeedRunner"]
