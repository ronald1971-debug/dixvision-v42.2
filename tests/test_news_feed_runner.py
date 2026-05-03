"""Tests for :class:`ui.feeds.news_runner.CoinDeskRSSFeedRunner` (P0-5).

Verifies the runner spawns a background asyncio thread, drives a
:class:`CoinDeskRSSPump` with an injected fake ``fetch`` callable, and
fans each polled :class:`NewsItem` through :class:`NewsFanout` into
the caller-supplied signal / hazard sinks. No real network traffic.
"""

from __future__ import annotations

import threading
import time

from core.contracts.events import HazardEvent, SignalEvent
from core.contracts.news import NewsItem
from system_engine.hazard_sensors import NewsShockSensor
from ui.feeds.news_runner import CoinDeskRSSFeedRunner

_RSS_BANG = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Bitcoin bank halt amid crash sparks panic</title>
    <link>https://example.invalid/halt</link>
    <guid>halt-001</guid>
  </item>
</channel></rss>
"""


def _stable_clock() -> int:
    return 1_700_000_000_000_000_000


def test_runner_drives_pump_and_fans_out() -> None:
    """End-to-end: runner -> pump -> NewsFanout -> sinks."""

    signals: list[SignalEvent] = []
    hazards: list[HazardEvent] = []
    fetch_done = threading.Event()
    runner_ref: dict[str, CoinDeskRSSFeedRunner] = {}

    async def fetch(url: str) -> bytes:
        # Stop after one successful poll so the test exits deterministically.
        # We are already on the runner's asyncio loop here, so calling
        # ``pump.stop()`` directly is safe (mirrors ``test_coindesk_rss_pump``).
        runner_ref["r"]._pump.stop()  # type: ignore[union-attr]
        fetch_done.set()
        return _RSS_BANG

    runner = CoinDeskRSSFeedRunner(
        signal_sink=signals.append,
        hazard_sink=hazards.append,
        clock_ns=_stable_clock,
        sensor=NewsShockSensor(),
        fetch=fetch,
        poll_interval_s=0.01,
    )
    runner_ref["r"] = runner

    runner.start()
    # Wait for fetch to complete + give run loop time to drain emissions.
    assert fetch_done.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while runner.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()

    # The crash + halt + panic keywords must trip NewsShockSensor.
    # The fanout dispatch order is hazard-first then signal, so any
    # hazard observed here proves the runner -> pump -> fanout ->
    # hazard_sink chain is wired live.
    assert hazards, "expected at least one HAZ-NEWS-SHOCK"
    assert all(h.code == "HAZ-NEWS-SHOCK" for h in hazards)


def test_runner_status_when_idle() -> None:
    runner = CoinDeskRSSFeedRunner(
        signal_sink=lambda _s: None,
        hazard_sink=lambda _h: None,
        clock_ns=_stable_clock,
    )
    status = runner.status()
    assert status.running is False
    assert status.source == "SRC-NEWS-COINDESK-001"
    assert status.items_received == 0


def test_runner_stop_is_idempotent() -> None:
    runner = CoinDeskRSSFeedRunner(
        signal_sink=lambda _s: None,
        hazard_sink=lambda _h: None,
        clock_ns=_stable_clock,
    )
    # Stop without start must not raise.
    runner.stop()
    runner.stop()


def test_runner_signal_sink_receives_news_item_projection() -> None:
    """Smoke-test: a benign news item still fans through the projection
    layer; the test asserts that *some* sink path runs without raising,
    not that a SignalEvent must be produced (projection may return None).
    """

    signals: list[SignalEvent] = []
    hazards: list[HazardEvent] = []
    fetch_done = threading.Event()
    runner_ref: dict[str, CoinDeskRSSFeedRunner] = {}

    async def fetch(url: str) -> bytes:
        runner_ref["r"]._pump.stop()  # type: ignore[union-attr]
        fetch_done.set()
        return b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Quiet morning update</title>
    <link>https://example.invalid/quiet</link>
    <guid>quiet-001</guid>
  </item>
</channel></rss>"""

    runner = CoinDeskRSSFeedRunner(
        signal_sink=signals.append,
        hazard_sink=hazards.append,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=0.01,
    )
    runner_ref["r"] = runner

    runner.start()
    assert fetch_done.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while runner.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()

    # A benign item must NOT trip the shock sensor; the assertion is
    # that the runner-fanout chain stays quiescent rather than crashing.
    assert hazards == []
    # Constructing a NewsItem ensures the contract import remains
    # exercised (so a future rename of ``core.contracts.news`` cannot
    # silently regress this test file).
    item = NewsItem(
        ts_ns=_stable_clock(),
        source="SRC-NEWS-COINDESK-001",
        guid="x",
        title="Bank halt panic",
        url="https://example.invalid/x",
        summary="",
    )
    assert item.guid == "x"
