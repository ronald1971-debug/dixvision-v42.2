"""D2 — RaydiumPoolPoller unit tests.

Uses ``httpx.MockTransport`` so the test never opens a real socket.
INV-15: ``ts_ns`` is supplied via an injected ``clock_ns`` so output
is byte-identical between runs. Mirrors the plain-``asyncio.run``
pattern used by ``tests/test_binance_public_ws.py`` to avoid a hard
dependency on ``pytest-asyncio``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from core.contracts.launches import PoolSnapshot
from ui.feeds.raydium_pools import (
    RAYDIUM_PAIRS_URL,
    RaydiumPoolPoller,
    parse_pair,
    parse_pairs,
)

# ---------------------------------------------------------------------------
# parse_pair / parse_pairs
# ---------------------------------------------------------------------------


def test_parse_pair_full_row() -> None:
    row = {
        "ammId": "PoolA",
        "baseMint": "BaseMintAAA",
        "quoteMint": "QuoteMintBBB",
        "name": "SOL/USDC",
        "price": "150.5",
        "liquidity": 1_000_000.0,
        "volume24h": 250_000.0,
    }
    snap = parse_pair(row, ts_ns=11)
    assert isinstance(snap, PoolSnapshot)
    assert snap.ts_ns == 11
    assert snap.chain == "solana"
    assert snap.venue == "RAYDIUM"
    assert snap.pool_id == "PoolA"
    assert snap.base_mint == "BaseMintAAA"
    assert snap.quote_mint == "QuoteMintBBB"
    assert snap.base_symbol == "SOL"
    assert snap.quote_symbol == "USDC"
    assert snap.price == pytest.approx(150.5)
    assert snap.liquidity_usd == pytest.approx(1_000_000.0)
    assert snap.volume_24h_usd == pytest.approx(250_000.0)


def test_parse_pair_missing_pool_id() -> None:
    assert parse_pair({"baseMint": "X"}, ts_ns=1) is None
    assert parse_pair({"ammId": ""}, ts_ns=1) is None
    assert parse_pair("not-a-mapping", ts_ns=1) is None


def test_parse_pair_handles_bad_numerics() -> None:
    row = {
        "ammId": "P",
        "price": "nope",
        "liquidity": None,
        "volume24h": "12.5",
    }
    snap = parse_pair(row, ts_ns=1)
    assert snap is not None
    assert snap.price == 0.0
    assert snap.liquidity_usd == 0.0
    assert snap.volume_24h_usd == pytest.approx(12.5)


def test_parse_pairs_skips_invalid_rows() -> None:
    rows = [
        {"ammId": "Good", "name": "A/B"},
        {"ammId": ""},
        "not-a-mapping",
        {"ammId": "Good2", "baseSymbol": "SOL", "quoteSymbol": "USDC"},
    ]
    out = parse_pairs(rows, ts_ns=1)
    assert len(out) == 2
    assert {s.pool_id for s in out} == {"Good", "Good2"}


# ---------------------------------------------------------------------------
# RaydiumPoolPoller
# ---------------------------------------------------------------------------


def _mock_factory(handler):
    def factory() -> httpx.AsyncClient:
        transport = httpx.MockTransport(handler)
        return httpx.AsyncClient(transport=transport)

    return factory


def _run(coro: Any, timeout: float = 2.0) -> None:
    """Run an awaitable with a hard timeout."""
    asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def test_poll_emits_snapshots_then_stops() -> None:
    payload = [
        {"ammId": "P1", "name": "SOL/USDC", "price": "150"},
        {"ammId": "P2", "name": "BTC/USDC", "price": "60000"},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert str(req.url) == RAYDIUM_PAIRS_URL
        return httpx.Response(200, json=payload)

    received: list[PoolSnapshot] = []
    poller = RaydiumPoolPoller(
        received.append,
        clock_ns=lambda: 99,
        client_factory=_mock_factory(handler),
        poll_interval_s=10.0,
        retry_delay_s=0.01,
        retry_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(poller.run())
        for _ in range(50):
            if poller.status().snapshots_emitted >= 2:
                break
            await asyncio.sleep(0.01)
        poller.stop()
        await task

    _run(driver())

    assert {s.pool_id for s in received} == {"P1", "P2"}
    s = poller.status()
    assert s.snapshots_emitted == 2
    assert s.errors == 0
    assert s.last_poll_ts_ns == 99


def test_poll_handles_5xx_increments_errors() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    received: list[PoolSnapshot] = []
    poller = RaydiumPoolPoller(
        received.append,
        clock_ns=lambda: 1,
        client_factory=_mock_factory(handler),
        poll_interval_s=10.0,
        retry_delay_s=0.01,
        retry_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(poller.run())
        await asyncio.sleep(0.1)
        poller.stop()
        await task

    _run(driver())
    assert received == []
    assert poller.status().errors >= 1


def test_poll_handles_non_list_payload() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"oops": "not-a-list"})

    received: list[PoolSnapshot] = []
    poller = RaydiumPoolPoller(
        received.append,
        clock_ns=lambda: 1,
        client_factory=_mock_factory(handler),
        poll_interval_s=10.0,
        retry_delay_s=0.01,
        retry_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(poller.run())
        await asyncio.sleep(0.1)
        poller.stop()
        await task

    _run(driver())
    assert received == []
    assert poller.status().errors >= 1


def test_poll_handles_non_json_body() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    received: list[PoolSnapshot] = []
    poller = RaydiumPoolPoller(
        received.append,
        clock_ns=lambda: 1,
        client_factory=_mock_factory(handler),
        poll_interval_s=10.0,
        retry_delay_s=0.01,
        retry_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(poller.run())
        await asyncio.sleep(0.1)
        poller.stop()
        await task

    _run(driver())
    assert received == []
    assert poller.status().errors >= 1


def test_poll_handles_network_error_then_recovers() -> None:
    state = {"n": 0}
    payload = [{"ammId": "P1"}]

    def handler(_req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("simulated dns failure")
        return httpx.Response(200, json=payload)

    received: list[PoolSnapshot] = []
    poller = RaydiumPoolPoller(
        received.append,
        clock_ns=lambda: 1,
        client_factory=_mock_factory(handler),
        poll_interval_s=10.0,
        retry_delay_s=0.01,
        retry_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(poller.run())
        for _ in range(100):
            if poller.status().snapshots_emitted >= 1:
                break
            await asyncio.sleep(0.01)
        poller.stop()
        await task

    _run(driver(), timeout=5.0)
    assert any(s.pool_id == "P1" for s in received)
    s = poller.status()
    assert s.errors >= 1
    assert s.snapshots_emitted == 1


def test_poll_swallows_sink_exception() -> None:
    payload = [{"ammId": "P1"}, {"ammId": "P2"}]

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    sink_calls = {"n": 0}

    def angry_sink(_snap: PoolSnapshot) -> None:
        sink_calls["n"] += 1
        raise RuntimeError("sink boom")

    poller = RaydiumPoolPoller(
        angry_sink,
        clock_ns=lambda: 1,
        client_factory=_mock_factory(handler),
        poll_interval_s=10.0,
        retry_delay_s=0.01,
        retry_delay_max_s=0.02,
    )

    async def driver() -> None:
        task = asyncio.create_task(poller.run())
        for _ in range(100):
            if sink_calls["n"] >= 2:
                break
            await asyncio.sleep(0.01)
        poller.stop()
        await task

    _run(driver())
    # Two snapshots attempted, both raised; loop survives.
    assert sink_calls["n"] >= 2
    assert poller.status().errors >= 2


def test_poller_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        RaydiumPoolPoller(
            lambda _s: None, clock_ns=lambda: 0, url=""
        )
    with pytest.raises(ValueError):
        RaydiumPoolPoller(
            lambda _s: None, clock_ns=lambda: 0, poll_interval_s=0
        )
    with pytest.raises(ValueError):
        RaydiumPoolPoller(
            lambda _s: None, clock_ns=lambda: 0, retry_delay_s=0
        )
    with pytest.raises(ValueError):
        RaydiumPoolPoller(
            lambda _s: None,
            clock_ns=lambda: 0,
            retry_delay_s=10.0,
            retry_delay_max_s=1.0,
        )
