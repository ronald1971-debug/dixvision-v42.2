"""Hummingbot Gateway-backed adapter (D1 — real wiring).

Wraps :class:`HummingbotGatewayClient` so an approved
``SignalEvent`` becomes a real HTTP call against a running Hummingbot
Gateway. Hummingbot's gateway covers ~100 connectors (binance, kraken,
gate, bybit, dydx_v4, uniswap, jupiter, …) so a single adapter
suffices for the whole CEX + DEX execution surface.

Wiring story (matches AdapterStatus FSM in :mod:`_live_base`):

* Construct with ``gateway_url=None`` → adapter stays
  ``DISCONNECTED`` and rejects ``submit()`` with a structured
  ``meta``. This is the in-process default for unit tests + the
  operator dashboard while credentials are still being provisioned.
* ``connect()`` issues ``GET /`` against the gateway. If healthy →
  ``READY``; otherwise ``DEGRADED`` with the failure reason.
* ``submit()`` translates the signal into a
  ``GatewayTradeRequest`` and calls either ``amm_trade`` (DEX
  connectors) or ``clob_order`` (CEX connectors). The adapter
  classifies by the ``connector`` name prefix; operators can override
  with ``signal.meta["execution_kind"] = "amm" | "clob"``.

This module never reads ``os.environ``. Credentials are passed in
explicitly so a malformed env doesn't silently route a real fill to
the wrong wallet (INV-65: per-decision audit truthfulness).
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    SignalEvent,
)
from execution_engine.adapters._hummingbot_gateway import (
    GatewayError,
    GatewayTradeRequest,
    HummingbotGatewayClient,
)
from execution_engine.adapters._live_base import (
    AdapterState,
    LiveAdapterBase,
)

# Connector identifiers Hummingbot tags as DEX-AMM. Anything not in this
# tuple goes through the CLOB path (binance / kraken / dydx / etc.).
_AMM_PREFIXES: tuple[str, ...] = (
    "uniswap",
    "jupiter",
    "pancakeswap",
    "raydium",
    "sushiswap",
    "trader_joe",
)


class HummingbotAdapter(LiveAdapterBase):
    """Real Hummingbot Gateway adapter.

    Args:
        connector: Hummingbot connector identifier
            (e.g. ``"binance_paper"``, ``"binance"``, ``"uniswap"``,
            ``"jupiter"``, ``"dydx_v4"``).
        chain: Chain for DEX connectors (``"ethereum"``, ``"solana"``).
            Empty string for CEX.
        network: Network within the chain
            (``"mainnet"``, ``"base"``, ``"arbitrum"``). Empty for CEX.
        wallet_address: Operator wallet address (``0x…`` for EVM,
            base58 for Solana). Required when not ``None``; if
            ``None``, adapter stays in scaffold mode.
        gateway_url: HTTP base URL of the gateway (``http://host:15888``).
            ``None`` keeps scaffold mode.
        cert_path / key_path: mTLS material for the gateway. Optional.
        default_size_quote: Default trade size in quote-asset units when
            ``signal.meta["size"]`` is not set.
    """

    def __init__(
        self,
        *,
        connector: str = "paper",
        chain: str = "",
        network: str = "",
        wallet_address: str | None = None,
        gateway_url: str | None = None,
        cert_path: str | None = None,
        key_path: str | None = None,
        default_size_quote: float = 100.0,
    ) -> None:
        super().__init__(
            name=f"hummingbot:{connector}",
            venue=f"hummingbot:{connector}",
        )
        if default_size_quote <= 0:
            raise ValueError("default_size_quote must be > 0")
        self._connector = connector
        self._chain = chain
        self._network = network
        self._wallet = wallet_address
        self._gateway_url = gateway_url
        self._cert_path = cert_path
        self._key_path = key_path
        self._default_size_quote = default_size_quote
        self._client: HummingbotGatewayClient | None = None

    @property
    def connector(self) -> str:
        return self._connector

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._gateway_url is None or self._wallet is None:
            self._state = AdapterState.DISCONNECTED
            missing: list[str] = []
            if self._gateway_url is None:
                missing.append("gateway_url")
            if self._wallet is None:
                missing.append("wallet_address")
            self._detail = (
                "missing: " + ", ".join(missing) + " — scaffold mode"
            )
            return
        if self._client is None:
            self._client = HummingbotGatewayClient(
                base_url=self._gateway_url,
                cert_path=self._cert_path,
                key_path=self._key_path,
            )
        self._state = AdapterState.CONNECTING
        self._detail = "calling /healthz"
        ok = False
        try:
            ok = self._client.healthcheck()
        except Exception as exc:  # noqa: BLE001 - surface in detail
            self._state = AdapterState.DEGRADED
            self._detail = f"healthz raised: {exc}"
            return
        if ok:
            self._state = AdapterState.READY
            self._detail = f"connected to {self._gateway_url}"
            self._last_heartbeat_ns = time.time_ns()
        else:
            self._state = AdapterState.DEGRADED
            self._detail = "gateway /healthz did not return status=ok"

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        super().disconnect()

    # ------------------------------------------------------------------
    # BrokerAdapter
    # ------------------------------------------------------------------

    def _submit_live(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        if self._client is None or self._wallet is None:
            return self._reject(signal, mark_price)

        base, quote = self._split_symbol(signal.symbol)
        size = self._size_from_signal(signal, mark_price)
        if size <= 0:
            return self._reject_with(
                signal, mark_price, "non_positive_size"
            )

        req = GatewayTradeRequest(
            connector=self._connector,
            chain=self._chain,
            network=self._network,
            base=base,
            quote=quote,
            side=signal.side.value,
            amount=size,
            limit_price=None,
            client_order_id=self._build_coid(signal),
        )

        try:
            if self._is_amm():
                resp = self._client.amm_trade(req, address=self._wallet)
            else:
                resp = self._client.clob_order(req, address=self._wallet)
        except GatewayError as exc:
            return self._reject_with(signal, mark_price, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._reject_with(
                signal, mark_price, f"transport_error: {exc!s}"
            )

        if not resp.accepted:
            return self._reject_with(
                signal,
                mark_price,
                f"venue_rejected: {resp.raw!r}"[:200],
            )

        meta: Mapping[str, str] = {
            "venue_order_id": resp.venue_order_id,
            "connector": self._connector,
            "execution_kind": "amm" if self._is_amm() else "clob",
        }
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=resp.fill_qty if resp.fill_qty > 0 else size,
            price=resp.fill_price if resp.fill_price > 0 else mark_price,
            status=ExecutionStatus.FILLED,
            venue=self.venue,
            order_id=resp.venue_order_id,
            meta=meta,
            produced_by_engine="execution_engine",
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _is_amm(self) -> bool:
        c = self._connector.lower()
        return any(c.startswith(p) for p in _AMM_PREFIXES)

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        for sep in ("-", "/"):
            if sep in symbol:
                base, _, quote = symbol.partition(sep)
                return base.upper(), quote.upper()
        # Fallback: assume USDC-quoted SPL/ERC20
        return symbol.upper(), "USDC"

    def _size_from_signal(
        self, signal: SignalEvent, mark_price: float
    ) -> float:
        meta_size = signal.meta.get("size") if signal.meta else None
        if meta_size is not None:
            try:
                return float(meta_size)
            except (TypeError, ValueError):
                pass
        if mark_price <= 0:
            return 0.0
        return self._default_size_quote / mark_price

    @staticmethod
    def _build_coid(signal: SignalEvent) -> str:
        return f"dix-{signal.ts_ns}-{signal.symbol.replace('/', '_')}"

    def _reject_with(
        self,
        signal: SignalEvent,
        mark_price: float,
        reason: str,
    ) -> ExecutionEvent:
        meta: Mapping[str, str] = {
            "reason": reason,
            "connector": self._connector,
            "adapter_state": self._state.value,
        }
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=0.0,
            price=mark_price,
            status=ExecutionStatus.REJECTED,
            venue=self.venue,
            order_id="",
            meta=meta,
            produced_by_engine="execution_engine",
        )


__all__ = ["HummingbotAdapter"]
