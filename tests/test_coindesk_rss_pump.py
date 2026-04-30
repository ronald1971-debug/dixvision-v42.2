"""Tests for :class:`ui.feeds.coindesk_rss.CoinDeskRSSPump`.

These tests inject a fake ``fetch`` callable so no real network
traffic is generated. The pump is exercised end-to-end by driving
its asyncio loop via :func:`asyncio.run` with a hard timeout, the
same pattern used by ``tests/test_binance_public_ws.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.contracts.news import NewsItem
from ui.feeds.coindesk_rss import (
    CoinDeskRSSPump,
    make_coindesk_rss_url,
    parse_rss_feed,
)

_RSS_DOC_A = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Item A1</title>
    <link>https://example.invalid/a1</link>
    <guid>a1</guid>
  </item>
  <item>
    <title>Item A2</title>
    <link>https://example.invalid/a2</link>
    <guid>a2</guid>
  </item>
</channel></rss>
"""

_RSS_DOC_B = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Item A1</title>
    <link>https://example.invalid/a1</link>
    <guid>a1</guid>
  </item>
  <item>
    <title>Item B3</title>
    <link>https://example.invalid/b3</link>
    <guid>b3</guid>
  </item>
</channel></rss>
"""


def _stable_clock() -> int:
    return 1_700_000_000_000_000_000


def _run(coro: Any, timeout: float = 2.0) -> None:
    """Run an awaitable with a hard timeout — never block the test suite."""
    asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def test_pump_emits_items_from_one_poll() -> None:
    received: list[NewsItem] = []
    pump_ref: dict[str, CoinDeskRSSPump] = {}

    async def fetch(url: str) -> bytes:
        assert url == make_coindesk_rss_url()
        pump_ref["p"].stop()
        return _RSS_DOC_A

    pump = CoinDeskRSSPump(
        sink=received.append,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=0.01,
    )
    pump_ref["p"] = pump
    _run(pump.run())

    assert [item.guid for item in received] == ["a1", "a2"]
    status = pump.status()
    assert status.items_received == 2
    assert status.polls == 1
    assert status.errors == 0
    assert status.last_poll_ts_ns == _stable_clock()
    assert status.last_item_ts_ns == _stable_clock()
    assert status.running is False  # set False on exit


def test_pump_dedupes_across_polls() -> None:
    received: list[NewsItem] = []
    payloads = [_RSS_DOC_A, _RSS_DOC_B]
    pump_ref: dict[str, CoinDeskRSSPump] = {}

    async def fetch(url: str) -> bytes:
        if not payloads:
            pump_ref["p"].stop()
            return b""
        return payloads.pop(0)

    pump = CoinDeskRSSPump(
        sink=received.append,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=0.001,
    )
    pump_ref["p"] = pump
    _run(pump.run())

    guids = [item.guid for item in received]
    # a1 appears in both docs but is emitted exactly once.
    assert guids.count("a1") == 1
    assert "a2" in guids
    assert "b3" in guids


def test_pump_counts_fetch_failures_and_recovers() -> None:
    received: list[NewsItem] = []
    attempts = {"n": 0}
    pump_ref: dict[str, CoinDeskRSSPump] = {}

    async def fetch(url: str) -> bytes:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("boom")
        pump_ref["p"].stop()
        return _RSS_DOC_A

    pump = CoinDeskRSSPump(
        sink=received.append,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=0.001,
        reconnect_delay_s=0.001,
        reconnect_delay_max_s=0.002,
    )
    pump_ref["p"] = pump
    _run(pump.run())

    assert attempts["n"] == 2
    assert pump.status().errors == 1
    assert [item.guid for item in received] == ["a1", "a2"]


def test_pump_swallows_sink_exceptions() -> None:
    """A faulty sink must never poison the pump loop."""
    seen: list[str] = []
    pump_ref: dict[str, CoinDeskRSSPump] = {}

    def bad_sink(item: NewsItem) -> None:
        seen.append(item.guid)
        if len(seen) == 1:
            raise RuntimeError("sink boom")

    async def fetch(url: str) -> bytes:
        pump_ref["p"].stop()
        return _RSS_DOC_A

    pump = CoinDeskRSSPump(
        sink=bad_sink,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=0.001,
    )
    pump_ref["p"] = pump
    _run(pump.run())

    # Both items reached the sink even though the first raised.
    assert seen == ["a1", "a2"]
    # The faulty call was counted as an error.
    assert pump.status().errors >= 1
    # Only the successful sink calls increment items_received.
    assert pump.status().items_received == 1


def test_pump_retries_item_on_transient_sink_failure() -> None:
    """An item must NOT be dropped permanently when the sink raises.

    Locks in the fix for the regression where ``_seen_guids`` was added
    *before* the sink call, so a transient sink failure (downstream
    restart, momentary DB outage, etc.) caused the item to be silently
    skipped on every subsequent poll.
    """
    delivered: list[str] = []
    fail_first_call = {"n": 0}
    pump_ref: dict[str, CoinDeskRSSPump] = {}

    def flaky_sink(item: NewsItem) -> None:
        if item.guid == "a1" and fail_first_call["n"] == 0:
            fail_first_call["n"] = 1
            raise RuntimeError("transient sink outage")
        delivered.append(item.guid)

    poll = {"n": 0}

    async def fetch(url: str) -> bytes:
        poll["n"] += 1
        if poll["n"] >= 2:
            pump_ref["p"].stop()
        return _RSS_DOC_A

    pump = CoinDeskRSSPump(
        sink=flaky_sink,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=0.001,
    )
    pump_ref["p"] = pump
    _run(pump.run())

    # Both items eventually delivered exactly once each.
    assert sorted(delivered) == ["a1", "a2"]
    # a1 was retried once (failure counted) and finally landed.
    assert pump.status().items_received == 2
    assert pump.status().errors >= 1


def test_pump_status_before_first_poll() -> None:
    pump = CoinDeskRSSPump(
        sink=lambda _item: None,
        clock_ns=_stable_clock,
    )
    status = pump.status()
    assert status.running is False
    assert status.polls == 0
    assert status.items_received == 0
    assert status.errors == 0
    assert status.last_poll_ts_ns is None
    assert status.last_item_ts_ns is None
    assert status.source == "COINDESK"
    assert status.url == make_coindesk_rss_url()


def test_pump_stop_event_breaks_sleep_window() -> None:
    """Calling stop() during the inter-poll sleep must wake the loop."""
    received: list[NewsItem] = []
    pump_ref: dict[str, CoinDeskRSSPump] = {}

    async def fetch(url: str) -> bytes:
        return _RSS_DOC_A

    pump = CoinDeskRSSPump(
        sink=received.append,
        clock_ns=_stable_clock,
        fetch=fetch,
        poll_interval_s=10.0,  # long, but stop() will wake it
    )
    pump_ref["p"] = pump

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.05)
        pump.stop()
        await task

    _run(driver())
    assert pump.status().polls >= 1


def test_pump_rejects_zero_poll_interval() -> None:
    with pytest.raises(ValueError, match="poll_interval_s"):
        CoinDeskRSSPump(
            sink=lambda _: None,
            clock_ns=_stable_clock,
            poll_interval_s=0,
        )


def test_pump_rejects_inverted_reconnect_window() -> None:
    with pytest.raises(ValueError, match="reconnect_delay_max_s"):
        CoinDeskRSSPump(
            sink=lambda _: None,
            clock_ns=_stable_clock,
            reconnect_delay_s=10.0,
            reconnect_delay_max_s=1.0,
        )


def test_pump_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        CoinDeskRSSPump(
            sink=lambda _: None,
            clock_ns=_stable_clock,
            source="",
        )


def test_parse_rss_feed_is_pure_across_replays() -> None:
    """Replay determinism: parser output is byte-stable for a given input."""
    a = parse_rss_feed(_RSS_DOC_A, ts_ns=42)
    b = parse_rss_feed(_RSS_DOC_A, ts_ns=42)
    assert a == b
