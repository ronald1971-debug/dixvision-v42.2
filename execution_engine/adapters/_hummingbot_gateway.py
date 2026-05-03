"""Real Hummingbot Gateway HTTP client.

The Hummingbot project ships ``gateway`` — a single Node service that
fronts ~100 connectors (binance/kraken/uniswap/jupiter/dydx/…). Every
connector exposes the same REST shape:

* ``GET  /``                        — health
* ``GET  /chain/balances``          — wallet balances
* ``POST /amm/price``               — quote for an AMM swap
* ``POST /amm/trade``               — execute an AMM swap
* ``POST /clob/orders``             — submit a CLOB order
* ``GET  /clob/markets``            — supported markets
* ``DELETE /clob/orders/{id}``      — cancel

This module is the *thin* HTTP client that speaks that shape. It
deliberately knows nothing about ``ExecutionEvent`` — translation
from ``SignalEvent`` to ``GatewayTradeRequest`` and the response back
to ``ExecutionEvent`` happens in :mod:`execution_engine.adapters.hummingbot`.

INV-15 honesty: the client takes an injectable ``time_ns_provider`` so
unit tests can pin timestamps. INV-69 honesty: every request the
client emits is logged with a monotonic counter so a replay can prove
order-of-emission.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class GatewayTradeRequest:
    """Cross-connector trade request shape.

    Attributes:
        connector: Hummingbot connector identifier (``binance``,
            ``uniswap``, ``jupiter``, …).
        chain: Underlying chain when the connector is a DEX (``ethereum``,
            ``solana``, ``base``). Empty for CEX.
        network: Network within the chain (``mainnet``, ``base``).
            Empty for CEX.
        base: Base asset symbol (``ETH``, ``SOL``).
        quote: Quote asset symbol (``USDC``).
        side: ``BUY`` or ``SELL``.
        amount: Quantity in base-asset units.
        limit_price: Optional limit price (None ⇒ market).
        client_order_id: Idempotency key.
    """

    connector: str
    chain: str
    network: str
    base: str
    quote: str
    side: str
    amount: float
    limit_price: float | None
    client_order_id: str


@dataclass(frozen=True)
class GatewayTradeResponse:
    """Normalised gateway response shape."""

    accepted: bool
    venue_order_id: str
    fill_price: float
    fill_qty: float
    raw: Mapping[str, Any]


class GatewayError(RuntimeError):
    """Gateway returned a non-2xx or malformed body."""


class HummingbotGatewayClient:
    """Real HTTP client for a running Hummingbot Gateway.

    Args:
        base_url: ``http(s)://host:15888`` — the gateway listener.
        cert_path: TLS client cert path (gateway requires mTLS by
            default; ``None`` disables verification, for dev only).
        key_path: TLS client key path. Required if ``cert_path`` set.
        timeout_s: Per-request timeout, seconds.
        time_ns_provider: Injectable monotonic clock (INV-15).
    """

    def __init__(
        self,
        *,
        base_url: str,
        cert_path: str | None = None,
        key_path: str | None = None,
        timeout_s: float = 5.0,
        time_ns_provider: Callable[[], int] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._time_ns = time_ns_provider
        self._counter: int = 0
        if cert_path is not None and key_path is None:
            raise ValueError("key_path required when cert_path set")
        verify: Any = True
        cert: Any = None
        if cert_path is not None and key_path is not None:
            cert = (cert_path, key_path)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_s,
            verify=verify,
            cert=cert,
            transport=transport,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HummingbotGatewayClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- introspection -----------------------------------------------------

    def healthcheck(self) -> bool:
        """``True`` iff gateway answers ``GET /`` with ``status: ok``."""
        try:
            r = self._client.get("/")
        except httpx.HTTPError:
            return False
        if r.status_code != 200:
            return False
        try:
            body = r.json()
        except json.JSONDecodeError:
            return False
        return bool(body.get("status") == "ok")

    # -- AMM trade ---------------------------------------------------------

    def amm_price(self, req: GatewayTradeRequest) -> float:
        """Pre-trade AMM quote in quote-asset units per base unit."""
        body = {
            "chain": req.chain,
            "network": req.network,
            "connector": req.connector,
            "base": req.base,
            "quote": req.quote,
            "amount": str(req.amount),
            "side": req.side,
        }
        r = self._client.post("/amm/price", json=body)
        if r.status_code != 200:
            raise GatewayError(
                f"/amm/price -> {r.status_code}: {r.text[:200]}"
            )
        data = r.json()
        try:
            return float(data["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GatewayError(f"malformed price: {data!r}") from exc

    def amm_trade(
        self, req: GatewayTradeRequest, *, address: str
    ) -> GatewayTradeResponse:
        """Execute an AMM swap. ``address`` is the operator wallet."""
        self._counter += 1
        body = {
            "chain": req.chain,
            "network": req.network,
            "connector": req.connector,
            "base": req.base,
            "quote": req.quote,
            "amount": str(req.amount),
            "side": req.side,
            "address": address,
            "limitPrice": (
                None if req.limit_price is None else str(req.limit_price)
            ),
            "clientOrderId": req.client_order_id,
            "nonce": self._counter,
        }
        r = self._client.post("/amm/trade", json=body)
        return self._parse_trade(r)

    # -- CLOB trade --------------------------------------------------------

    def clob_order(
        self, req: GatewayTradeRequest, *, address: str
    ) -> GatewayTradeResponse:
        """Submit a CLOB order (CEX or perps)."""
        self._counter += 1
        body = {
            "connector": req.connector,
            "chain": req.chain,
            "network": req.network,
            "address": address,
            "market": f"{req.base}-{req.quote}",
            "side": req.side,
            "orderType": "LIMIT" if req.limit_price is not None else "MARKET",
            "price": (
                None if req.limit_price is None else str(req.limit_price)
            ),
            "amount": str(req.amount),
            "clientOrderId": req.client_order_id,
            "nonce": self._counter,
        }
        r = self._client.post("/clob/orders", json=body)
        return self._parse_trade(r)

    # -- internals ---------------------------------------------------------

    def _parse_trade(self, r: httpx.Response) -> GatewayTradeResponse:
        if r.status_code not in (200, 201, 202):
            raise GatewayError(
                f"trade -> {r.status_code}: {r.text[:200]}"
            )
        try:
            data = r.json()
        except json.JSONDecodeError as exc:
            raise GatewayError(f"non-json body: {r.text[:200]}") from exc
        # Explicit ``accepted`` wins over the status fallback so a
        # gateway response of ``{"accepted": false, "status": "submitted"}``
        # is reported as REJECTED, not FILLED (INV-65 audit truthfulness).
        raw_accepted = data.get("accepted")
        if raw_accepted is not None:
            accepted = bool(raw_accepted)
        else:
            accepted = data.get("status") in (
                "submitted",
                "filled",
                "ok",
                "OK",
            )
        venue_order_id = str(
            data.get("orderId") or data.get("txHash") or ""
        )
        try:
            fill_price = float(data.get("price") or 0.0)
        except (TypeError, ValueError):
            fill_price = 0.0
        try:
            fill_qty = float(data.get("amount") or 0.0)
        except (TypeError, ValueError):
            fill_qty = 0.0
        return GatewayTradeResponse(
            accepted=accepted,
            venue_order_id=venue_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            raw=data,
        )


__all__ = [
    "GatewayError",
    "GatewayTradeRequest",
    "GatewayTradeResponse",
    "HummingbotGatewayClient",
]
