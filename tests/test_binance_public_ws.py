"""Tests for the read-only Binance public WS adapter (SRC-MARKET-BINANCE-001).

Covers three layers without touching the network:

* Pure URL builder (:func:`make_combined_stream_url`).
* Pure 24hrTicker → :class:`MarketTick` projection
  (:func:`parse_24hr_ticker`).
* Pump loop with an injected fake connection
  (:class:`BinancePublicWSPump`).
* :class:`FeedRunner` start/stop lifecycle on a background thread.

INV-15 is exercised explicitly: feeding the same payload twice with the
same ``ts_ns`` produces two byte-identical :class:`MarketTick`
instances. The pump never reads ``time.time`` directly — every
timestamp comes from the injected ``clock_ns`` callable, so test runs
are deterministic.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterable
from typing import Any

import pytest

from core.contracts.market import MarketTick
from ui.feeds.binance_public_ws import (
    BINANCE_PUBLIC_WS_BASE,
    DEFAULT_SYMBOLS,
    VENUE_TAG,
    BinancePublicWSPump,
    FeedStatus,
    make_combined_stream_url,
    parse_24hr_ticker,
)
from ui.feeds.runner import FeedRunner

# ---------------------------------------------------------------------------
# Pure URL builder
# ---------------------------------------------------------------------------


def test_make_combined_stream_url_basic() -> None:
    url = make_combined_stream_url(["BTCUSDT", "ETHUSDT"])
    assert url == (
        f"{BINANCE_PUBLIC_WS_BASE}/stream"
        "?streams=btcusdt@ticker/ethusdt@ticker"
    )


def test_make_combined_stream_url_lowercases_and_trims() -> None:
    url = make_combined_stream_url(["  BtCuSdT  "])
    assert url.endswith("?streams=btcusdt@ticker")


def test_make_combined_stream_url_rejects_empty() -> None:
    with pytest.raises(ValueError):
        make_combined_stream_url([])


@pytest.mark.parametrize(
    "bad",
    ["", "btc-usdt", "btc usdt", "btc/usdt", "btc.usdt"],
)
def test_make_combined_stream_url_rejects_invalid_symbol(bad: str) -> None:
    with pytest.raises(ValueError):
        make_combined_stream_url([bad])


def test_make_combined_stream_url_supports_alt_stream() -> None:
    url = make_combined_stream_url(["btcusdt"], stream="trade")
    assert url.endswith("?streams=btcusdt@trade")


# ---------------------------------------------------------------------------
# Pure 24hrTicker projection
# ---------------------------------------------------------------------------


_VALID_FRAME: dict[str, Any] = {
    "e": "24hrTicker",
    "E": 1_700_000_000_123,  # event time (ms) — intentionally ignored
    "s": "BTCUSDT",
    "b": "65000.10",  # best bid
    "a": "65000.20",  # best ask
    "c": "65000.15",  # last trade price
    "v": "1234.5",  # 24h base-asset volume
}


def test_parse_24hr_ticker_basic() -> None:
    tick = parse_24hr_ticker(_VALID_FRAME, ts_ns=42)
    assert tick == MarketTick(
        ts_ns=42,
        symbol="BTCUSDT",
        bid=65000.10,
        ask=65000.20,
        last=65000.15,
        volume=1234.5,
        venue=VENUE_TAG,
    )


def test_parse_24hr_ticker_combined_stream_wrapper() -> None:
    # Combined-stream framing wraps the payload in {"stream": ..., "data": ...}.
    wrapped = {"stream": "btcusdt@ticker", "data": _VALID_FRAME}
    tick = parse_24hr_ticker(wrapped, ts_ns=42)
    assert tick is not None and tick.symbol == "BTCUSDT"


def test_parse_24hr_ticker_uses_caller_ts_ns_inv15() -> None:
    """INV-15: ``ts_ns`` is supplied by the caller; replays are deterministic."""
    a = parse_24hr_ticker(_VALID_FRAME, ts_ns=100)
    b = parse_24hr_ticker(_VALID_FRAME, ts_ns=100)
    assert a == b
    c = parse_24hr_ticker(_VALID_FRAME, ts_ns=200)
    assert c is not None and c.ts_ns == 200


@pytest.mark.parametrize(
    "frame",
    [
        {"e": "kline", "s": "BTCUSDT"},  # wrong event type
        {"s": "BTCUSDT", "b": "1", "a": "1", "c": "1"},  # missing 'e'
        {},  # empty
        "not a mapping",  # wrong type
        None,
    ],
)
def test_parse_24hr_ticker_returns_none_for_non_data(frame: Any) -> None:
    assert parse_24hr_ticker(frame, ts_ns=0) is None


@pytest.mark.parametrize(
    "broken",
    [
        {**_VALID_FRAME, "b": "not-a-number"},
        {**_VALID_FRAME, "a": None},
        {**_VALID_FRAME, "c": ""},
    ],
)
def test_parse_24hr_ticker_returns_none_for_broken_numbers(
    broken: dict[str, Any],
) -> None:
    assert parse_24hr_ticker(broken, ts_ns=0) is None


@pytest.mark.parametrize(
    "broken",
    [
        {**_VALID_FRAME, "b": "0"},  # zero bid -> sentinel for invalid
        {**_VALID_FRAME, "a": "-1"},
        {**_VALID_FRAME, "c": "0"},
        {**_VALID_FRAME, "s": ""},
    ],
)
def test_parse_24hr_ticker_returns_none_for_invalid_values(
    broken: dict[str, Any],
) -> None:
    assert parse_24hr_ticker(broken, ts_ns=0) is None


def test_parse_24hr_ticker_default_volume_when_missing_or_bad() -> None:
    no_volume = {k: v for k, v in _VALID_FRAME.items() if k != "v"}
    tick = parse_24hr_ticker(no_volume, ts_ns=1)
    assert tick is not None and tick.volume == 0.0
    bad = {**_VALID_FRAME, "v": "garbage"}
    tick2 = parse_24hr_ticker(bad, ts_ns=1)
    assert tick2 is not None and tick2.volume == 0.0


# ---------------------------------------------------------------------------
# Pump loop (with injected fake connection)
# ---------------------------------------------------------------------------


class _FakeConn:
    """Async-iterable + close stub matching ``BinancePublicWSPump``'s
    minimal connection contract.

    Yields all queued frames in order, then signals end-of-stream
    (returning from the loop) which triggers a reconnect on the pump
    side. Tests stop the pump explicitly to break that loop.
    """

    def __init__(self, frames: Iterable[Any]) -> None:
        self._frames = list(frames)
        self.closed = False

    def __aiter__(self) -> _FakeConn:
        return self

    async def __anext__(self) -> str:
        if not self._frames:
            raise StopAsyncIteration
        frame = self._frames.pop(0)
        if isinstance(frame, str):
            return frame
        return json.dumps(frame)

    async def close(self) -> None:
        self.closed = True


def _run(coro: Any, timeout: float = 2.0) -> None:
    """Run an awaitable with a hard timeout — never block the test suite."""
    asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def test_pump_emits_tick_for_each_data_frame() -> None:
    received: list[MarketTick] = []
    initial_frames = [
        {"stream": "btcusdt@ticker", "data": _VALID_FRAME},
        {"stream": "btcusdt@ticker", "data": _VALID_FRAME},
        # Subscription ack — must be silently ignored.
        {"result": None, "id": 1},
    ]
    conn_calls: list[str] = []

    async def fake_connect(url: str) -> _FakeConn:
        conn_calls.append(url)
        # First connection emits the data + ack; subsequent reconnects
        # idle silently so ``len(received)`` is deterministic.
        if len(conn_calls) == 1:
            return _FakeConn(initial_frames)
        return _FakeConn([])

    pump = BinancePublicWSPump(
        ["btcusdt"],
        sink=received.append,
        clock_ns=lambda: 7,
        connect=fake_connect,
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.01,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        # Give the loop one chance to drain frames + start reconnect wait.
        await asyncio.sleep(0.05)
        pump.stop()
        await task

    _run(driver())
    assert len(received) == 2
    assert received[0].symbol == "BTCUSDT"
    assert received[0].ts_ns == 7
    assert received[0].venue == VENUE_TAG
    assert conn_calls and conn_calls[0].startswith(BINANCE_PUBLIC_WS_BASE)
    status = pump.status()
    assert status.ticks_received == 2
    assert status.last_tick_ts_ns == 7
    assert status.errors == 0


def test_pump_counts_json_decode_errors() -> None:
    received: list[MarketTick] = []
    calls = {"n": 0}

    async def fake_connect(url: str) -> _FakeConn:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeConn(["{not json}", json.dumps(_VALID_FRAME)])
        return _FakeConn([])

    pump = BinancePublicWSPump(
        ["btcusdt"],
        sink=received.append,
        clock_ns=lambda: 1,
        connect=fake_connect,
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.01,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.05)
        pump.stop()
        await task

    _run(driver())
    assert len(received) == 1
    assert pump.status().errors >= 1


def test_pump_reconnects_after_connection_failure() -> None:
    received: list[MarketTick] = []
    attempts = {"n": 0}

    async def fake_connect(url: str) -> _FakeConn:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("simulated drop")
        if attempts["n"] == 2:
            return _FakeConn([_VALID_FRAME])
        return _FakeConn([])

    pump = BinancePublicWSPump(
        ["btcusdt"],
        sink=received.append,
        clock_ns=lambda: 99,
        connect=fake_connect,
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.01,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.1)
        pump.stop()
        await task

    _run(driver())
    assert attempts["n"] >= 2
    assert len(received) == 1
    status = pump.status()
    assert status.errors >= 1
    assert status.ticks_received == 1


def test_pump_rejects_empty_symbols() -> None:
    with pytest.raises(ValueError):
        BinancePublicWSPump([], sink=lambda _t: None, clock_ns=lambda: 0)


def test_pump_rejects_bad_backoff() -> None:
    with pytest.raises(ValueError):
        BinancePublicWSPump(
            ["btcusdt"],
            sink=lambda _t: None,
            clock_ns=lambda: 0,
            reconnect_delay_s=0,
        )
    with pytest.raises(ValueError):
        BinancePublicWSPump(
            ["btcusdt"],
            sink=lambda _t: None,
            clock_ns=lambda: 0,
            reconnect_delay_s=10,
            reconnect_delay_max_s=1,
        )


def test_pump_status_initial_state() -> None:
    pump = BinancePublicWSPump(
        ["btcusdt"],
        sink=lambda _t: None,
        clock_ns=lambda: 0,
    )
    s = pump.status()
    assert isinstance(s, FeedStatus)
    assert s.running is False
    assert s.symbols == ("BTCUSDT",)
    assert s.ticks_received == 0
    assert s.errors == 0
    assert s.last_tick_ts_ns is None


# ---------------------------------------------------------------------------
# FeedRunner background-thread lifecycle
# ---------------------------------------------------------------------------


def test_runner_status_when_idle_uses_configured_default_symbols() -> None:
    runner = FeedRunner(sink=lambda _t: None, clock_ns=lambda: 0)
    s = runner.status()
    assert s.running is False
    assert s.symbols == tuple(sym.upper() for sym in DEFAULT_SYMBOLS)


def test_runner_start_then_stop_idempotent() -> None:
    received: list[MarketTick] = []

    # Stream that idles forever (one frame, then end-of-stream + reconnect
    # blocked by tiny backoff so the pump survives until stop()).
    def frames_factory() -> Iterable[Any]:
        return iter([])

    async def fake_connect(url: str) -> _FakeConn:
        return _FakeConn(frames_factory())

    counter = {"v": 0}
    lock = threading.Lock()

    def clock() -> int:
        with lock:
            counter["v"] += 1
            return counter["v"]

    runner = FeedRunner(
        sink=received.append,
        clock_ns=clock,
        symbols=("btcusdt",),
        connect=fake_connect,
    )
    s1 = runner.start()
    assert runner.is_running is True
    # Second start while running is a no-op.
    s2 = runner.start()
    assert s1.symbols == s2.symbols
    # Stop joins the thread.
    runner.stop(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while runner.is_running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert runner.is_running is False
    # Stop again is a no-op.
    runner.stop()


def test_runner_start_with_symbol_override() -> None:
    captured: list[str] = []

    async def fake_connect(url: str) -> _FakeConn:
        captured.append(url)
        return _FakeConn([])

    runner = FeedRunner(
        sink=lambda _t: None,
        clock_ns=lambda: 1,
        symbols=("btcusdt",),
        connect=fake_connect,
    )
    runner.start(symbols=("solusdt", "dogeusdt"))
    deadline = time.monotonic() + 2.0
    while not captured and time.monotonic() < deadline:
        time.sleep(0.05)
    runner.stop(timeout=2.0)
    assert captured, "fake_connect was never called"
    assert "solusdt@ticker" in captured[0]
    assert "dogeusdt@ticker" in captured[0]
