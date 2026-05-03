"""Unit tests for UniswapXQuoteClient (D3) — uses httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from execution_engine.adapters._uniswapx_quote import (
    QuoteRequest,
    UniswapXError,
    UniswapXQuoteClient,
)


def _quote_payload() -> dict[str, object]:
    return {
        "order": {
            "reactor": "0x" + "a" * 40,
            "swapper": "0x" + "b" * 40,
            "outputs": [
                {
                    "token": "0x" + "d" * 40,
                    "startAmount": "900000",
                    "endAmount": "890000",
                    "recipient": "0x" + "b" * 40,
                }
            ],
            "input": {
                "token": "0x" + "c" * 40,
                "startAmount": "1000000",
                "endAmount": "1000000",
            },
        }
    }


def _client(handler: httpx.MockTransport) -> UniswapXQuoteClient:
    return UniswapXQuoteClient(
        base_url="https://api.test.local",
        timeout_s=1.0,
        transport=handler,
    )


def test_healthcheck_ok() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(200, json={"ok": True})

    with _client(httpx.MockTransport(handler)) as c:
        assert c.healthcheck() is True
    assert seen == ["/v2/health"]


def test_healthcheck_returns_false_on_5xx() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"err": "down"})

    with _client(httpx.MockTransport(handler)) as c:
        assert c.healthcheck() is False


def test_quote_parses_canonical_payload() -> None:
    seen_bodies: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(req.content.decode()))
        return httpx.Response(200, json=_quote_payload())

    with _client(httpx.MockTransport(handler)) as c:
        resp = c.quote(
            QuoteRequest(
                chain_id=1,
                token_in="0x" + "c" * 40,
                token_out="0x" + "d" * 40,
                amount_in=1_000_000,
                swapper="0x" + "b" * 40,
                slippage_bps=50,
            )
        )
    assert resp.amount_out_quote == 900_000
    assert resp.amount_out_min == 890_000
    assert resp.order_payload["reactor"] == "0x" + "a" * 40
    body = seen_bodies[0]
    assert body["chainId"] == 1
    assert body["amount"] == "1000000"
    assert body["type"] == "EXACT_INPUT"
    assert body["slippageTolerance"] == 0.5


def test_quote_raises_on_5xx() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _client(httpx.MockTransport(handler)) as c, pytest.raises(
        UniswapXError, match="HTTP 500"
    ):
        c.quote(
            QuoteRequest(
                chain_id=1,
                token_in="0x" + "c" * 40,
                token_out="0x" + "d" * 40,
                amount_in=1,
                swapper="0x" + "b" * 40,
                slippage_bps=10,
            )
        )


def test_quote_raises_on_missing_order_field() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unrelated": "payload"})

    with _client(httpx.MockTransport(handler)) as c, pytest.raises(
        UniswapXError, match="missing 'order'"
    ):
        c.quote(
            QuoteRequest(
                chain_id=1,
                token_in="0x" + "c" * 40,
                token_out="0x" + "d" * 40,
                amount_in=1,
                swapper="0x" + "b" * 40,
                slippage_bps=10,
            )
        )


def test_submit_order_accepts_2xx_with_hash() -> None:
    bodies: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content.decode()))
        return httpx.Response(200, json={"hash": "0xabc123"})

    with _client(httpx.MockTransport(handler)) as c:
        resp = c.submit_order(
            order_payload={"encodedOrder": "0xdeadbeef"},
            signature="0x" + "1" * 130,
            chain_id=1,
        )
    assert resp.accepted is True
    assert resp.order_hash == "0xabc123"
    assert bodies[0]["encodedOrder"] == "0xdeadbeef"
    assert bodies[0]["signature"] == "0x" + "1" * 130
    assert bodies[0]["chainId"] == 1


def test_submit_order_returns_rejected_on_4xx() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid sig"})

    with _client(httpx.MockTransport(handler)) as c:
        resp = c.submit_order(
            order_payload={"encodedOrder": "0xdeadbeef"},
            signature="0x" + "1" * 130,
            chain_id=1,
        )
    assert resp.accepted is False
    assert resp.order_hash == ""


def test_submit_order_returns_rejected_when_2xx_but_no_hash() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "queued"})

    with _client(httpx.MockTransport(handler)) as c:
        resp = c.submit_order(
            order_payload={"encodedOrder": "0xdeadbeef"},
            signature="0x" + "1" * 130,
            chain_id=1,
        )
    assert resp.accepted is False
    assert resp.order_hash == ""


def test_init_validates_inputs() -> None:
    with pytest.raises(ValueError):
        UniswapXQuoteClient(base_url="")
    with pytest.raises(ValueError):
        UniswapXQuoteClient(base_url="https://x", timeout_s=0)
