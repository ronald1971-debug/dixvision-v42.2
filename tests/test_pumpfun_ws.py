"""D2 — PumpFunLaunchPump unit tests.

Uses an in-memory fake WebSocket connection so the test never opens a
real socket. INV-15: ``ts_ns`` is supplied via an injected
``clock_ns`` so output is byte-identical between runs. Mirrors the
plain-``asyncio.run`` pattern used by ``tests/test_binance_public_ws.py``
to avoid a hard dependency on ``pytest-asyncio``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import pytest

from core.contracts.launches import LaunchEvent
from ui.feeds.pumpfun_ws import (
    SUBSCRIBE_NEW_TOKEN_FRAME,
    PumpFunLaunchPump,
    parse_new_token,
)

# ---------------------------------------------------------------------------
# parse_new_token
# ---------------------------------------------------------------------------


def test_parse_create_frame() -> None:
    frame = {
        "txType": "create",
        "mint": "MintA111",
        "symbol": "MOON",
        "name": "Moon Coin",
        "traderPublicKey": "DevDev",
        "marketCapSol": 12.5,
        "vSolInBondingCurve": 30.0,
    }
    ev = parse_new_token(frame, ts_ns=42)
    assert isinstance(ev, LaunchEvent)
    assert ev.ts_ns == 42
    assert ev.chain == "solana"
    assert ev.venue == "PUMPFUN"
    assert ev.mint == "MintA111"
    assert ev.symbol == "MOON"
    assert ev.name == "Moon Coin"
    assert ev.creator == "DevDev"
    assert ev.market_cap_usd == pytest.approx(12.5)
    assert ev.liquidity_usd == pytest.approx(30.0)


def test_parse_returns_none_on_subscribe_ack() -> None:
    assert parse_new_token({"message": "Subscribed"}, ts_ns=1) is None


def test_parse_returns_none_on_non_mapping() -> None:
    assert parse_new_token("hello", ts_ns=1) is None
    assert parse_new_token(None, ts_ns=1) is None
    assert parse_new_token(["a", "b"], ts_ns=1) is None


def test_parse_returns_none_without_mint() -> None:
    assert parse_new_token({"txType": "create"}, ts_ns=1) is None
    assert parse_new_token({"txType": "create", "mint": ""}, ts_ns=1) is None


def test_parse_other_tx_types_skipped() -> None:
    assert parse_new_token({"txType": "buy", "mint": "X"}, ts_ns=1) is None
    assert parse_new_token({"txType": "sell", "mint": "X"}, ts_ns=1) is None


def test_parse_handles_string_numerics() -> None:
    ev = parse_new_token(
        {
            "txType": "create",
            "mint": "Mint",
            "marketCapSol": "12.5",
            "vSolInBondingCurve": "bad",
        },
        ts_ns=1,
    )
    assert ev is not None
    assert ev.market_cap_usd == pytest.approx(12.5)
    assert ev.liquidity_usd == 0.0


# ---------------------------------------------------------------------------
# PumpFunLaunchPump
# ---------------------------------------------------------------------------


class _FakeConn:
    """Async-iterable + send/close stub matching the pump's connection
    Protocol. Mirrors the pattern in test_binance_public_ws.py."""

    def __init__(self, frames: Iterable[Any]) -> None:
        self._frames = list(frames)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

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
    """Run an awaitable with a hard timeout."""
    asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def test_pump_emits_launch_then_stops() -> None:
    received: list[LaunchEvent] = []
    calls = {"n": 0}

    initial_frames = [
        {"message": "Subscribed"},
        {
            "txType": "create",
            "mint": "MintA",
            "symbol": "MOON",
            "marketCapSol": 5.0,
        },
    ]
    captured: dict[str, _FakeConn] = {}

    async def fake_connect(_url: str) -> _FakeConn:
        calls["n"] += 1
        if calls["n"] == 1:
            conn = _FakeConn(initial_frames)
            captured["first"] = conn
            return conn
        return _FakeConn([])

    counter = {"n": 1000}

    def clock() -> int:
        counter["n"] += 1
        return counter["n"]

    pump = PumpFunLaunchPump(
        received.append,
        clock_ns=clock,
        connect=fake_connect,  # type: ignore[arg-type]
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.05)
        pump.stop()
        await task

    _run(driver())

    assert captured["first"].sent == [SUBSCRIBE_NEW_TOKEN_FRAME]
    assert len(received) == 1
    ev = received[0]
    assert ev.mint == "MintA"
    assert ev.symbol == "MOON"
    s = pump.status()
    assert s.launches_received == 1
    assert s.last_launch_ts_ns is not None


def test_pump_skips_malformed_json_increments_error() -> None:
    received: list[LaunchEvent] = []
    calls = {"n": 0}

    async def fake_connect(_url: str) -> _FakeConn:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeConn(
                [
                    "not-json{{{",
                    {"txType": "create", "mint": "M"},
                ]
            )
        return _FakeConn([])

    pump = PumpFunLaunchPump(
        received.append,
        clock_ns=lambda: 1,
        connect=fake_connect,  # type: ignore[arg-type]
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.05)
        pump.stop()
        await task

    _run(driver())

    assert len(received) == 1
    s = pump.status()
    assert s.errors >= 1
    assert s.launches_received == 1


def test_pump_reconnects_on_connect_error() -> None:
    received: list[LaunchEvent] = []
    attempts = {"n": 0}

    async def fake_connect(_url: str) -> _FakeConn:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("first attempt fails")
        if attempts["n"] == 2:
            return _FakeConn(
                [{"txType": "create", "mint": "M2"}]
            )
        return _FakeConn([])

    pump = PumpFunLaunchPump(
        received.append,
        clock_ns=lambda: 7,
        connect=fake_connect,  # type: ignore[arg-type]
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.15)
        pump.stop()
        await task

    _run(driver())

    assert attempts["n"] >= 2
    assert any(ev.mint == "M2" for ev in received)
    assert pump.status().errors >= 1


def test_pump_swallows_sink_exception() -> None:
    """Sink exceptions must not kill the pump (INV-69 nonce continuity)."""

    sink_calls = {"n": 0}

    def angry_sink(_ev: LaunchEvent) -> None:
        sink_calls["n"] += 1
        raise RuntimeError("sink boom")

    async def fake_connect(_url: str) -> _FakeConn:
        return _FakeConn(
            [{"txType": "create", "mint": "BoomMint"}]
        )

    pump = PumpFunLaunchPump(
        angry_sink,
        clock_ns=lambda: 1,
        connect=fake_connect,  # type: ignore[arg-type]
        reconnect_delay_s=0.01,
        reconnect_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(pump.run())
        await asyncio.sleep(0.05)
        pump.stop()
        await task

    _run(driver())
    assert sink_calls["n"] >= 1
    assert pump.status().errors >= 1


def test_pump_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        PumpFunLaunchPump(
            lambda _e: None,
            clock_ns=lambda: 0,
            url="",
        )
    with pytest.raises(ValueError):
        PumpFunLaunchPump(
            lambda _e: None,
            clock_ns=lambda: 0,
            reconnect_delay_s=0,
        )
    with pytest.raises(ValueError):
        PumpFunLaunchPump(
            lambda _e: None,
            clock_ns=lambda: 0,
            reconnect_delay_s=10.0,
            reconnect_delay_max_s=1.0,
        )
