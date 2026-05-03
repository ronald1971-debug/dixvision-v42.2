"""D1 — HummingbotGatewayClient + HummingbotAdapter unit tests.

Uses ``httpx.MockTransport`` to simulate the Hummingbot Gateway HTTP
surface so the test never opens a real socket.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.contracts.events import ExecutionStatus, Side, SignalEvent
from execution_engine.adapters._hummingbot_gateway import (
    GatewayError,
    GatewayTradeRequest,
    HummingbotGatewayClient,
)
from execution_engine.adapters._live_base import AdapterState
from execution_engine.adapters.hummingbot import HummingbotAdapter


def _signal(side: Side = Side.BUY, symbol: str = "ETH-USDC") -> SignalEvent:
    return SignalEvent(
        ts_ns=1,
        symbol=symbol,
        side=side,
        confidence=0.9,
        meta={},
    )


# ---------------------------------------------------------------------------
# HummingbotGatewayClient
# ---------------------------------------------------------------------------


def _mk_client(
    handler: Any, *, base_url: str = "http://gw.local"
) -> HummingbotGatewayClient:
    transport = httpx.MockTransport(handler)
    return HummingbotGatewayClient(base_url=base_url, transport=transport)


def test_healthcheck_ok():
    def h(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/"
        return httpx.Response(200, json={"status": "ok"})

    with _mk_client(h) as c:
        assert c.healthcheck() is True


def test_healthcheck_non_ok_status():
    def h(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "starting"})

    with _mk_client(h) as c:
        assert c.healthcheck() is False


def test_healthcheck_500():
    def h(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _mk_client(h) as c:
        assert c.healthcheck() is False


def test_healthcheck_transport_error():
    def h(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with _mk_client(h) as c:
        assert c.healthcheck() is False


def test_amm_price_returns_float():
    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/amm/price"
        body = json.loads(req.content)
        assert body["chain"] == "ethereum"
        assert body["base"] == "ETH"
        return httpx.Response(200, json={"price": "3500.5"})

    with _mk_client(h) as c:
        p = c.amm_price(
            GatewayTradeRequest(
                connector="uniswap",
                chain="ethereum",
                network="mainnet",
                base="ETH",
                quote="USDC",
                side="BUY",
                amount=1.0,
                limit_price=None,
                client_order_id="x",
            )
        )
        assert p == pytest.approx(3500.5)


def test_amm_price_malformed_raises():
    def h(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"oops": True})

    with _mk_client(h) as c, pytest.raises(GatewayError):
        c.amm_price(
            GatewayTradeRequest(
                connector="uniswap",
                chain="ethereum",
                network="mainnet",
                base="ETH",
                quote="USDC",
                side="BUY",
                amount=1.0,
                limit_price=None,
                client_order_id="x",
            )
        )


def test_amm_trade_accepted():
    seen: dict[str, Any] = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            201,
            json={
                "accepted": True,
                "txHash": "0xabc",
                "price": "3501.0",
                "amount": "0.5",
            },
        )

    with _mk_client(h) as c:
        resp = c.amm_trade(
            GatewayTradeRequest(
                connector="uniswap",
                chain="ethereum",
                network="mainnet",
                base="ETH",
                quote="USDC",
                side="BUY",
                amount=0.5,
                limit_price=None,
                client_order_id="dix-1",
            ),
            address="0xCAFE",
        )
    assert seen["path"] == "/amm/trade"
    assert seen["body"]["address"] == "0xCAFE"
    assert seen["body"]["clientOrderId"] == "dix-1"
    assert seen["body"]["nonce"] == 1
    assert resp.accepted
    assert resp.venue_order_id == "0xabc"
    assert resp.fill_price == pytest.approx(3501.0)
    assert resp.fill_qty == pytest.approx(0.5)


def test_amm_trade_rejected_status():
    def h(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "rejected", "reason": "insufficient_balance"},
        )

    with _mk_client(h) as c:
        resp = c.amm_trade(
            GatewayTradeRequest(
                connector="uniswap",
                chain="ethereum",
                network="mainnet",
                base="ETH",
                quote="USDC",
                side="BUY",
                amount=0.5,
                limit_price=None,
                client_order_id="dix-1",
            ),
            address="0xCAFE",
        )
    assert resp.accepted is False


def test_clob_order_routes_to_clob_endpoint():
    seen: dict[str, Any] = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "status": "submitted",
                "orderId": "BIN-42",
                "price": "65000",
                "amount": "0.001",
            },
        )

    with _mk_client(h) as c:
        resp = c.clob_order(
            GatewayTradeRequest(
                connector="binance",
                chain="",
                network="",
                base="BTC",
                quote="USDT",
                side="BUY",
                amount=0.001,
                limit_price=None,
                client_order_id="dix-2",
            ),
            address="API_KEY_ID",
        )
    assert seen["path"] == "/clob/orders"
    assert seen["body"]["market"] == "BTC-USDT"
    assert seen["body"]["orderType"] == "MARKET"
    assert resp.accepted
    assert resp.venue_order_id == "BIN-42"


def test_nonce_increments_per_call():
    nonces: list[int] = []

    def h(req: httpx.Request) -> httpx.Response:
        nonces.append(json.loads(req.content)["nonce"])
        return httpx.Response(200, json={"status": "ok", "orderId": "x"})

    with _mk_client(h) as c:
        for _ in range(3):
            c.clob_order(
                GatewayTradeRequest(
                    connector="binance",
                    chain="",
                    network="",
                    base="BTC",
                    quote="USDT",
                    side="BUY",
                    amount=0.001,
                    limit_price=None,
                    client_order_id="x",
                ),
                address="acct",
            )
    assert nonces == [1, 2, 3]


# ---------------------------------------------------------------------------
# HummingbotAdapter
# ---------------------------------------------------------------------------


def test_adapter_disconnected_when_no_creds():
    a = HummingbotAdapter(connector="binance")
    a.connect()
    s = a.status()
    assert s.state is AdapterState.DISCONNECTED
    assert "missing" in s.detail.lower()


def test_adapter_rejects_when_disconnected():
    a = HummingbotAdapter(connector="binance")
    ev = a.submit(_signal(), 100.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.meta["reason"] == "adapter_not_ready"


def test_adapter_connect_healthy(monkeypatch: pytest.MonkeyPatch):
    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    a = HummingbotAdapter(
        connector="binance",
        wallet_address="acct",
        gateway_url="http://gw.local",
    )
    # Inject mock transport into the freshly created client
    real_init = HummingbotGatewayClient.__init__

    def patched_init(self: Any, **kw: Any) -> None:
        kw["transport"] = httpx.MockTransport(h)
        real_init(self, **kw)

    monkeypatch.setattr(HummingbotGatewayClient, "__init__", patched_init)
    a.connect()
    assert a.status().state is AdapterState.READY


def test_adapter_submit_fills_via_amm(monkeypatch: pytest.MonkeyPatch):
    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/amm/trade":
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "txHash": "0xfeed",
                    "price": "3500",
                    "amount": "0.0285",
                },
            )
        return httpx.Response(404)

    a = HummingbotAdapter(
        connector="uniswap",
        chain="ethereum",
        network="mainnet",
        wallet_address="0xCAFE",
        gateway_url="http://gw.local",
        default_size_quote=100.0,
    )
    real_init = HummingbotGatewayClient.__init__

    def patched_init(self: Any, **kw: Any) -> None:
        kw["transport"] = httpx.MockTransport(h)
        real_init(self, **kw)

    monkeypatch.setattr(HummingbotGatewayClient, "__init__", patched_init)
    a.connect()
    ev = a.submit(_signal(symbol="ETH-USDC"), 3500.0)
    assert ev.status is ExecutionStatus.FILLED
    assert ev.venue == "hummingbot:uniswap"
    assert ev.order_id == "0xfeed"
    assert ev.meta["execution_kind"] == "amm"


def test_adapter_submit_clob_path_for_cex(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, str] = {}

    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/clob/orders":
            seen["path"] = req.url.path
            return httpx.Response(
                200,
                json={
                    "status": "submitted",
                    "orderId": "BIN-1",
                    "price": "65000",
                    "amount": "0.0015",
                },
            )
        return httpx.Response(404)

    a = HummingbotAdapter(
        connector="binance",
        wallet_address="acct",
        gateway_url="http://gw.local",
    )
    real_init = HummingbotGatewayClient.__init__

    def patched_init(self: Any, **kw: Any) -> None:
        kw["transport"] = httpx.MockTransport(h)
        real_init(self, **kw)

    monkeypatch.setattr(HummingbotGatewayClient, "__init__", patched_init)
    a.connect()
    ev = a.submit(_signal(symbol="BTC-USDT"), 65000.0)
    assert ev.status is ExecutionStatus.FILLED
    assert seen["path"] == "/clob/orders"
    assert ev.meta["execution_kind"] == "clob"


def test_adapter_handles_gateway_error(monkeypatch: pytest.MonkeyPatch):
    def h(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503, text="service unavailable")

    a = HummingbotAdapter(
        connector="binance",
        wallet_address="acct",
        gateway_url="http://gw.local",
    )
    real_init = HummingbotGatewayClient.__init__

    def patched_init(self: Any, **kw: Any) -> None:
        kw["transport"] = httpx.MockTransport(h)
        real_init(self, **kw)

    monkeypatch.setattr(HummingbotGatewayClient, "__init__", patched_init)
    a.connect()
    ev = a.submit(_signal(symbol="BTC-USDT"), 65000.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert "503" in ev.meta["reason"]
