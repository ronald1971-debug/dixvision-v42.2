"""End-to-end (mocked) tests for UniswapXAdapter (D3)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from eth_account import Account

from core.contracts.events import (
    ExecutionStatus,
    Side,
    SignalEvent,
)
from execution_engine.adapters._live_base import AdapterState
from execution_engine.adapters._uniswapx_quote import (
    UniswapXQuoteClient,
)
from execution_engine.adapters.uniswapx import UniswapXAdapter

_PRIVATE_KEY = "0x" + "11" * 32
_SIGNER = Account.from_key(_PRIVATE_KEY).address


def _quote_payload(**overrides: object) -> dict[str, object]:
    order: dict[str, object] = {
        "chainId": 1,
        "reactor": "0x" + "a" * 40,
        "swapper": _SIGNER,
        "nonce": "424242",
        "deadline": 1_700_000_120,
        "decayStartTime": 1_700_000_000,
        "decayEndTime": 1_700_000_060,
        "exclusiveFiller": "0x" + "0" * 40,
        "exclusivityOverrideBps": "0",
        "outputs": [
            {
                "token": "0x" + "d" * 40,
                "startAmount": "900000",
                "endAmount": "890000",
                "recipient": _SIGNER,
            }
        ],
        "input": {
            "token": "0x" + "c" * 40,
            "startAmount": "1000000",
            "endAmount": "1000000",
        },
        "encodedOrder": "0xdeadbeef",
    }
    order.update(overrides)
    return {"order": order}


def _signal(meta: dict[str, str] | None = None) -> SignalEvent:
    if meta is None:
        meta = {
            "token_in": "0x" + "c" * 40,
            "token_out": "0x" + "d" * 40,
            "amount_in": "1000000",
            "slippage_bps": "30",
        }
    return SignalEvent(
        ts_ns=1_700_000_000_000_000_000,
        symbol="USDC/WETH",
        side=Side.BUY,
        confidence=0.85,
        meta=meta,
        produced_by_engine="intelligence_engine",
    )


def _write_key(tmp_path: Path) -> str:
    p = tmp_path / "key.hex"
    p.write_text(_PRIVATE_KEY)
    return str(p)


def _adapter(
    *,
    handler: httpx.MockTransport,
    private_key_path: str,
    time_s: int = 1_700_000_000,
    nonce_seed: int = 7,
) -> UniswapXAdapter:
    client = UniswapXQuoteClient(
        base_url="https://api.test.local",
        timeout_s=1.0,
        transport=handler,
    )
    counter = {"n": nonce_seed}

    def nonce() -> int:
        counter["n"] += 1
        return counter["n"]

    return UniswapXAdapter(
        chain_id=1,
        rpc_url="https://rpc.test.local",
        private_key_path=private_key_path,
        api_url="https://api.test.local",
        client=client,
        time_unix_s_provider=lambda: time_s,
        nonce_provider=nonce,
    )


def test_connect_without_credentials_stays_disconnected() -> None:
    a = UniswapXAdapter()
    a.connect()
    s = a.status()
    assert s.state is AdapterState.DISCONNECTED
    assert "missing credentials" in s.detail


def test_connect_with_credentials_reaches_ready(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v2/health"
        return httpx.Response(200, json={"ok": True})

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    s = a.status()
    assert s.state is AdapterState.READY
    assert _SIGNER in s.detail


def test_connect_marks_degraded_on_unhealthy(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"err": "down"})

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    assert a.status().state is AdapterState.DEGRADED


def test_connect_rejects_malformed_private_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.hex"
    p.write_text("not_a_hex_key")
    a = UniswapXAdapter(
        rpc_url="https://rpc.test.local",
        private_key_path=str(p),
        api_url="https://api.test.local",
    )
    a.connect()
    s = a.status()
    assert s.state is AdapterState.DISCONNECTED
    assert "private_key_load_failed" in s.detail


def test_submit_when_disconnected_rejects() -> None:
    a = UniswapXAdapter()
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.qty == 0.0
    assert ev.meta["reason"] == "adapter_not_ready"


def test_submit_full_flow_quote_sign_submit(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if path == "/v2/quote":
            return httpx.Response(200, json=_quote_payload())
        if path == "/v2/order":
            captured.append(json.loads(req.content.decode()))
            return httpx.Response(200, json={"hash": "0xabc123"})
        return httpx.Response(404, json={"err": "not_found"})

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    assert a.status().state is AdapterState.READY

    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.SUBMITTED, ev.meta
    assert ev.order_id == "0xabc123"
    assert ev.meta["signer"] == _SIGNER
    assert ev.meta["amount_in_request"] == "1000000"
    assert ev.meta["amount_in_signed"] == "1000000"
    assert ev.meta["nonce"] == "424242"
    assert ev.meta["deadline"] == "1700000120"
    assert ev.meta["amount_out_quote"] == "900000"
    assert ev.meta["amount_out_min"] == "890000"
    assert ev.meta["execution_kind"] == "uniswapx_intent"

    assert len(captured) == 1
    body = captured[0]
    assert body["chainId"] == 1
    assert body["encodedOrder"] == "0xdeadbeef"
    sig = body["signature"]
    assert isinstance(sig, str) and sig.startswith("0x")
    assert len(sig) == 132


def test_submit_rejects_when_quote_endpoint_5xx(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(500, text="boom")

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert "HTTP 500" in ev.meta["reason"]


def test_submit_rejects_when_order_endpoint_4xx(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if path == "/v2/quote":
            return httpx.Response(200, json=_quote_payload())
        return httpx.Response(400, json={"error": "bad signature"})

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert "venue_rejected" in ev.meta["reason"]


def test_submit_rejects_when_meta_missing_tokens(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        pytest.fail("quote/order should not be hit")
        return httpx.Response(500)

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    ev = a.submit(_signal(meta={}), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.meta["reason"] == "missing_token_in_or_token_out"


def test_submit_falls_back_to_size_usd(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if path == "/v2/quote":
            captured.append(json.loads(req.content.decode()))
            return httpx.Response(200, json=_quote_payload())
        if path == "/v2/order":
            return httpx.Response(200, json={"hash": "0xfeed"})
        return httpx.Response(404)

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    sig = _signal(
        meta={
            "token_in": "0x" + "c" * 40,
            "token_out": "0x" + "d" * 40,
            "size_usd": "250",
        }
    )
    ev = a.submit(sig, mark_price=1.0)
    assert ev.status is ExecutionStatus.SUBMITTED, ev.meta
    body = captured[0]
    # 250 USD → 250 * 1e6 base units (6-decimal stablecoin assumption).
    assert body["amount"] == "250000000"


def test_submit_rejects_when_quote_input_token_mismatches(
    tmp_path: Path,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/v2/quote":
            payload = _quote_payload()
            order = payload["order"]
            assert isinstance(order, dict)
            order["input"] = {
                "token": "0x" + "e" * 40,
                "startAmount": "1000000",
                "endAmount": "1000000",
            }
            return httpx.Response(200, json=payload)
        pytest.fail("/v2/order must not be hit on parse-mismatch reject")
        return httpx.Response(500)

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.meta["reason"] == "quote_input_token_mismatch"


def test_submit_rejects_when_quote_output_token_mismatches(
    tmp_path: Path,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/v2/quote":
            payload = _quote_payload()
            order = payload["order"]
            assert isinstance(order, dict)
            order["outputs"] = [
                {
                    "token": "0x" + "f" * 40,
                    "startAmount": "900000",
                    "endAmount": "890000",
                    "recipient": _SIGNER,
                }
            ]
            return httpx.Response(200, json=payload)
        pytest.fail("/v2/order must not be hit on parse-mismatch reject")
        return httpx.Response(500)

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.meta["reason"] == "quote_output_token_mismatch"


def test_submit_rejects_when_quote_payload_missing_nonce(
    tmp_path: Path,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/v2/quote":
            payload = _quote_payload()
            order = payload["order"]
            assert isinstance(order, dict)
            order.pop("nonce")
            return httpx.Response(200, json=payload)
        pytest.fail("/v2/order must not be hit on parse-fail reject")
        return httpx.Response(500)

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.REJECTED
    assert ev.meta["reason"].startswith("quote_parse_failed")


def test_submit_signature_matches_server_encoded_order_params(
    tmp_path: Path,
) -> None:
    """The signed typed-data must use the SERVER's nonce / deadline /
    decay times — not values pulled from the local clock — so the
    on-chain Reactor can recover the same digest from ``encodedOrder``.
    """
    captured: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v2/health":
            return httpx.Response(200, json={"ok": True})
        if path == "/v2/quote":
            return httpx.Response(200, json=_quote_payload())
        if path == "/v2/order":
            captured.append(json.loads(req.content.decode()))
            return httpx.Response(200, json={"hash": "0xfeedface"})
        return httpx.Response(404)

    # Local clock + nonce providers return values that DIFFER from the
    # quote payload. If the adapter still signed locally-computed data
    # the assertions below would fail.
    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
        time_s=9_999_999_999,
        nonce_seed=7_777_777,
    )
    a.connect()
    ev = a.submit(_signal(), mark_price=1.0)
    assert ev.status is ExecutionStatus.SUBMITTED, ev.meta
    # The server's nonce / deadline (424242 / 1_700_000_120) wins over
    # the local providers' (7_777_777 / 9_999_999_999 + …).
    assert ev.meta["nonce"] == "424242"
    assert ev.meta["deadline"] == "1700000120"
    body = captured[0]
    assert body["encodedOrder"] == "0xdeadbeef"


def test_disconnect_clears_signer(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    a = _adapter(
        handler=httpx.MockTransport(handler),
        private_key_path=_write_key(tmp_path),
    )
    a.connect()
    assert a.signer_address == _SIGNER
    a.disconnect()
    assert a.signer_address is None
    assert a.status().state is AdapterState.DISCONNECTED
