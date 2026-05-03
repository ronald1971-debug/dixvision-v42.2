"""UniswapX REST quote + order client (D3).

UniswapX exposes two HTTP endpoints the operator-side adapter cares
about:

* ``POST /v2/quote`` — quote an exact-input swap. The response carries
  the unsigned ``ExclusiveDutchOrder`` payload (reactor, nonces, decay
  schedule, input/output legs) that the operator signs locally.
* ``POST /v2/order`` — submit a signed order. The response carries
  the order hash + initial routing decision.

The default endpoint (``https://api.uniswap.org``) is the same one the
official Uniswap web app talks to. The client takes an optional
``transport`` so unit tests inject ``httpx.MockTransport`` and assert
on the wire format without ever touching the network.

INV-15 honesty: the client itself does not call any system clock; the
adapter passes ``deadline_unix_s`` etc. explicitly so two replays are
byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

#: Default UniswapX REST base URL (matches the public Uniswap web app).
DEFAULT_API_URL = "https://api.uniswap.org"


@dataclass(frozen=True)
class QuoteRequest:
    """Inputs for ``POST /v2/quote``.

    Attributes:
        chain_id: EVM chain id (1 = Ethereum, 8453 = Base, …).
        token_in: Input token contract address.
        token_out: Output token contract address.
        amount_in: Exact input amount, in token base units.
        swapper: Wallet address signing the intent.
        slippage_bps: Acceptable slippage, basis points (50 = 0.50%).
    """

    chain_id: int
    token_in: str
    token_out: str
    amount_in: int
    swapper: str
    slippage_bps: int


@dataclass(frozen=True)
class QuoteResponse:
    """Output of ``POST /v2/quote``.

    Attributes:
        amount_out_quote: Quoted output amount (start of decay) in
            base units.
        amount_out_min: Minimum acceptable output amount (end of
            decay) in base units.
        order_payload: The unsigned order dict returned by the
            backend; the adapter passes this verbatim to
            :func:`build_exclusive_dutch_order_typed_data`.
        raw: Full response body for diagnostics.
    """

    amount_out_quote: int
    amount_out_min: int
    order_payload: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class OrderSubmitResponse:
    """Output of ``POST /v2/order``.

    Attributes:
        accepted: ``True`` iff the backend acknowledged the order.
        order_hash: Server-side order hash (empty string on reject).
        raw: Full response body for diagnostics.
    """

    accepted: bool
    order_hash: str
    raw: Mapping[str, Any]


class UniswapXError(RuntimeError):
    """UniswapX backend returned a non-2xx or malformed payload."""


class UniswapXQuoteClient:
    """Thin synchronous client for the UniswapX REST surface.

    Args:
        base_url: REST root (defaults to :data:`DEFAULT_API_URL`).
        timeout_s: Per-request timeout, seconds.
        transport: Optional ``httpx.BaseTransport`` for tests.
        api_key: Optional API key sent as ``x-api-key`` header.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_API_URL,
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
        api_key: str | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        headers: dict[str, str] = {
            "content-type": "application/json",
            "accept": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            transport=transport,
            headers=headers,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> UniswapXQuoteClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- introspection -----------------------------------------------------

    def healthcheck(self) -> bool:
        """``True`` iff the backend answers ``GET /v2/health`` 2xx."""
        try:
            r = self._client.get("/v2/health")
        except httpx.HTTPError:
            return False
        return 200 <= r.status_code < 300

    # -- quote -------------------------------------------------------------

    def quote(self, req: QuoteRequest) -> QuoteResponse:
        """``POST /v2/quote`` — fetch an unsigned ExclusiveDutchOrder.

        Raises:
            UniswapXError: On HTTP error or unexpected payload shape.
        """
        body = {
            "chainId": req.chain_id,
            "tokenIn": req.token_in,
            "tokenOut": req.token_out,
            "amount": str(req.amount_in),
            "type": "EXACT_INPUT",
            "swapper": req.swapper,
            "slippageTolerance": req.slippage_bps / 100.0,
        }
        try:
            r = self._client.post("/v2/quote", json=body)
        except httpx.HTTPError as exc:
            raise UniswapXError(f"quote transport error: {exc!s}") from exc
        if not (200 <= r.status_code < 300):
            raise UniswapXError(
                f"quote HTTP {r.status_code}: {r.text[:200]}"
            )
        try:
            payload = r.json()
        except ValueError as exc:
            raise UniswapXError(
                f"quote: malformed JSON: {exc!s}"
            ) from exc
        order = payload.get("order") or payload.get("quote") or {}
        if not isinstance(order, Mapping) or not order:
            raise UniswapXError(
                f"quote: missing 'order' field: {payload!r}"[:200]
            )
        amount_out_quote = _to_int(
            order.get("outputs", [{}])[0].get("startAmount")
            if isinstance(order.get("outputs"), list)
            and order["outputs"]
            else None
        )
        amount_out_min = _to_int(
            order.get("outputs", [{}])[0].get("endAmount")
            if isinstance(order.get("outputs"), list)
            and order["outputs"]
            else None
        )
        return QuoteResponse(
            amount_out_quote=amount_out_quote,
            amount_out_min=amount_out_min,
            order_payload=order,
            raw=payload,
        )

    # -- order submission --------------------------------------------------

    def submit_order(
        self,
        *,
        order_payload: Mapping[str, Any],
        signature: str,
        chain_id: int,
    ) -> OrderSubmitResponse:
        """``POST /v2/order`` — submit a signed UniswapX order.

        The backend convention: ``{order, signature, chainId}`` envelope.
        """
        body = {
            "encodedOrder": order_payload.get("encodedOrder")
            or order_payload,
            "signature": signature,
            "chainId": chain_id,
        }
        try:
            r = self._client.post("/v2/order", json=body)
        except httpx.HTTPError as exc:
            raise UniswapXError(
                f"order transport error: {exc!s}"
            ) from exc
        try:
            payload = r.json() if r.content else {}
        except ValueError as exc:
            raise UniswapXError(
                f"order: malformed JSON: {exc!s}"
            ) from exc
        if not (200 <= r.status_code < 300):
            return OrderSubmitResponse(
                accepted=False,
                order_hash="",
                raw=payload
                or {"status_code": r.status_code, "text": r.text[:200]},
            )
        order_hash = str(
            payload.get("hash")
            or payload.get("orderHash")
            or payload.get("id")
            or ""
        )
        return OrderSubmitResponse(
            accepted=bool(order_hash),
            order_hash=order_hash,
            raw=payload,
        )


def _to_int(raw: Any) -> int:
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return 0


__all__ = [
    "DEFAULT_API_URL",
    "OrderSubmitResponse",
    "QuoteRequest",
    "QuoteResponse",
    "UniswapXError",
    "UniswapXQuoteClient",
]
