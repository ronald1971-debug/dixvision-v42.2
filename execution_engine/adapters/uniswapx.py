"""UniswapX intent-based DEX adapter (D-track real wiring).

UniswapX (and CowSwap / Across) replace the classic AMM "swap a fixed
token for at-least N output" with a *signed intent* the operator
broadcasts to a fillers network. The fillers compete to fill the
intent within the operator's slippage band; the winning filler pays
gas and takes a small spread.

This adapter therefore differs structurally from a CEX adapter: there
is no per-order ``buy`` / ``sell`` HTTP call — instead we
1. Quote an exact-input swap via UniswapX ``POST /v2/quote`` (returns
   an unsigned ``ExclusiveDutchOrder`` payload).
2. Build the canonical EIP-712 typed-data dict.
3. Sign with the operator's EVM private key (Permit2 witness).
4. Submit the signed envelope via ``POST /v2/order``.

The operator never broadcasts an on-chain tx themselves; that is the
filler's job.

Operational honesty (INV-56 Triad Lock):
* If credentials are missing the adapter stays ``DISCONNECTED`` and
  ``submit()`` returns ``REJECTED`` with structured ``meta`` rather
  than silently dropping or pretending to fill.
* If the quote endpoint or the signer fail we degrade and return
  ``REJECTED``, never a fake fill.

INV-15 honesty: this adapter takes injectable ``time_unix_s_provider``
and ``nonce_provider`` callables so replay tests are byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from execution_engine.adapters._live_base import (
    AdapterState,
    LiveAdapterBase,
)
from execution_engine.adapters._uniswapx_quote import (
    DEFAULT_API_URL,
    QuoteRequest,
    UniswapXError,
    UniswapXQuoteClient,
)
from execution_engine.adapters._uniswapx_signer import (
    DutchInput,
    DutchOutput,
    ExclusiveDutchOrderIntent,
    build_exclusive_dutch_order_typed_data,
    sign_typed_data,
)
from system.time_source import wall_ns

#: Default decay window — the auction over which the price slides
#: from ``startAmount`` to ``endAmount``. UniswapX V2 uses 30s by
#: default for retail-sized orders.
DEFAULT_DECAY_WINDOW_S = 30

#: Default deadline padding — extra seconds beyond ``decayEndTime``
#: during which a filler may still settle.
DEFAULT_DEADLINE_PADDING_S = 60

#: Default UniswapX V2 reactor address on Ethereum mainnet
#: (``ExclusiveDutchOrderReactor`` deployment).
#: Reference: https://docs.uniswap.org/contracts/uniswapx/deployments
DEFAULT_REACTOR_ADDRESS = "0x6000da47483062A0D734Ba3dc7576Ce6A0B645C4"


class UniswapXAdapter(LiveAdapterBase):
    """Signs UniswapX intents and broadcasts them to the fillers network.

    Args:
        chain_id: EVM chain id (1 = mainnet, 8453 = base, 42161 = arb).
            Default is mainnet.
        rpc_url: EVM RPC endpoint. Currently only used as a presence
            check (the adapter signs locally, the filler broadcasts);
            recorded in ``status().detail`` for operator visibility.
        private_key_path: Filesystem path to a file containing the
            ``0x``-prefixed hex private key. ``None`` keeps scaffold mode.
        api_url: UniswapX backend base URL (orders + quotes). Defaults
            to ``DEFAULT_API_URL``.
        api_key: Optional ``x-api-key`` for the UniswapX backend.
        max_slippage_bps: Slippage tolerance encoded into the intent.
            Operator-overridable per-order via ``signal.meta``.
        reactor_address: ExclusiveDutchOrderReactor for ``chain_id``.
        default_swap_size_usd: Notional USD size used when
            ``signal.meta`` does not pin an explicit ``size``.
        decay_window_s: Auction window between start and end of the
            Dutch decay (seconds).
        deadline_padding_s: Extra deadline beyond decay end (seconds).
        time_unix_s_provider: Injectable wall clock returning the
            current unix epoch in seconds. Defaults to
            ``wall_ns() // 1_000_000_000``.
        nonce_provider: Injectable monotonic nonce source. Defaults to
            a per-instance counter starting at ``wall_ns()``.
        client: Pre-built :class:`UniswapXQuoteClient` (tests).
    """

    def __init__(
        self,
        *,
        chain_id: int = 1,
        rpc_url: str | None = None,
        private_key_path: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        max_slippage_bps: int = 50,
        reactor_address: str | None = None,
        default_swap_size_usd: float = 100.0,
        decay_window_s: int = DEFAULT_DECAY_WINDOW_S,
        deadline_padding_s: int = DEFAULT_DEADLINE_PADDING_S,
        time_unix_s_provider: Callable[[], int] | None = None,
        nonce_provider: Callable[[], int] | None = None,
        client: UniswapXQuoteClient | None = None,
    ) -> None:
        super().__init__(
            name=f"uniswapx:chain_{chain_id}",
            venue=f"uniswapx:chain_{chain_id}",
        )
        if chain_id <= 0:
            raise ValueError("chain_id must be > 0")
        if max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be >= 0")
        if decay_window_s <= 0:
            raise ValueError("decay_window_s must be > 0")
        if deadline_padding_s < 0:
            raise ValueError("deadline_padding_s must be >= 0")
        self._chain_id = chain_id
        self._rpc_url = rpc_url
        self._private_key_path = private_key_path
        self._api_url = api_url or DEFAULT_API_URL
        self._api_key = api_key
        self._max_slippage_bps = max_slippage_bps
        self._reactor = reactor_address or DEFAULT_REACTOR_ADDRESS
        self._default_size_usd = default_swap_size_usd
        self._decay_window_s = decay_window_s
        self._deadline_padding_s = deadline_padding_s
        self._time_unix_s = (
            time_unix_s_provider
            if time_unix_s_provider is not None
            else (lambda: wall_ns() // 1_000_000_000)
        )
        self._counter: int = 0
        self._nonce_provider = (
            nonce_provider
            if nonce_provider is not None
            else self._default_nonce
        )
        self._client = client
        self._private_key: str | None = None
        self._signer_address: str | None = None

    # -- introspection -----------------------------------------------------

    @property
    def chain_id(self) -> int:
        return self._chain_id

    @property
    def max_slippage_bps(self) -> int:
        return self._max_slippage_bps

    @property
    def signer_address(self) -> str | None:
        return self._signer_address

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        missing: list[str] = []
        if self._rpc_url is None:
            missing.append("DIX_EVM_RPC_URL")
        if self._private_key_path is None:
            missing.append("DIX_EVM_PRIVATE_KEY_PATH")
        if missing:
            self._state = AdapterState.DISCONNECTED
            self._detail = (
                "missing credentials: "
                + ", ".join(missing)
                + " — scaffold mode active"
            )
            return
        try:
            self._private_key = self._load_private_key(
                self._private_key_path
            )
        except (OSError, ValueError) as exc:
            self._state = AdapterState.DISCONNECTED
            self._detail = f"private_key_load_failed: {exc!s}"
            return
        # Recover signer address so the operator dashboard can show
        # which wallet the adapter will sign as.
        try:
            from eth_account import Account  # local import keeps test paths clean

            self._signer_address = Account.from_key(
                self._private_key
            ).address
        except Exception as exc:  # noqa: BLE001
            self._state = AdapterState.DISCONNECTED
            self._detail = f"signer_init_failed: {exc!s}"
            return
        if self._client is None:
            self._client = UniswapXQuoteClient(
                base_url=self._api_url,
                api_key=self._api_key,
            )
        self._state = AdapterState.CONNECTING
        self._detail = "calling /v2/health"
        ok = False
        try:
            ok = self._client.healthcheck()
        except Exception as exc:  # noqa: BLE001
            self._state = AdapterState.DEGRADED
            self._detail = f"healthz raised: {exc!s}"
            return
        if ok:
            self._state = AdapterState.READY
            self._detail = (
                f"signer={self._signer_address} api={self._api_url}"
            )
            self._last_heartbeat_ns = wall_ns()
        else:
            self._state = AdapterState.DEGRADED
            self._detail = "/v2/health did not return 2xx"

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._private_key = None
        self._signer_address = None
        super().disconnect()

    # -- submit ------------------------------------------------------------

    def _submit_live(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        if self._client is None or self._private_key is None:
            return self._reject_with(
                signal, mark_price, "client_or_signer_missing"
            )
        token_in = signal.meta.get("token_in") if signal.meta else None
        token_out = signal.meta.get("token_out") if signal.meta else None
        if not token_in or not token_out:
            return self._reject_with(
                signal,
                mark_price,
                "missing_token_in_or_token_out",
            )
        amount_in = self._amount_in_from_signal(signal, mark_price)
        if amount_in <= 0:
            return self._reject_with(
                signal, mark_price, "non_positive_amount_in"
            )
        slippage_bps = self._slippage_bps_from_signal(signal)

        # 1. Quote ----------------------------------------------------------
        try:
            quote = self._client.quote(
                QuoteRequest(
                    chain_id=self._chain_id,
                    token_in=token_in,
                    token_out=token_out,
                    amount_in=amount_in,
                    swapper=self._signer_address or "",
                    slippage_bps=slippage_bps,
                )
            )
        except UniswapXError as exc:
            return self._reject_with(signal, mark_price, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._reject_with(
                signal, mark_price, f"quote_transport_error: {exc!s}"
            )

        # 2. Build EIP-712 typed-data --------------------------------------
        now_s = self._time_unix_s()
        decay_start = now_s
        decay_end = now_s + self._decay_window_s
        deadline = decay_end + self._deadline_padding_s
        outputs = self._extract_outputs(quote.order_payload, signal)
        if not outputs:
            return self._reject_with(
                signal, mark_price, "quote_missing_outputs"
            )
        intent = ExclusiveDutchOrderIntent(
            chain_id=self._chain_id,
            reactor=str(
                quote.order_payload.get("reactor") or self._reactor
            ),
            swapper=self._signer_address or "",
            nonce=self._nonce_provider(),
            deadline_unix_s=deadline,
            decay_start_time_unix_s=decay_start,
            decay_end_time_unix_s=decay_end,
            exclusive_filler="0x" + "0" * 40,
            exclusivity_override_bps=0,
            input=DutchInput(
                token=token_in,
                start_amount=amount_in,
                end_amount=amount_in,
            ),
            outputs=outputs,
        )
        typed_data = build_exclusive_dutch_order_typed_data(intent)

        # 3. Sign ----------------------------------------------------------
        try:
            signed = sign_typed_data(
                private_key=self._private_key,
                typed_data=typed_data,
            )
        except Exception as exc:  # noqa: BLE001
            return self._reject_with(
                signal, mark_price, f"sign_failed: {exc!s}"
            )
        if signed.signer_address != self._signer_address:
            return self._reject_with(
                signal,
                mark_price,
                "signer_address_mismatch",
            )

        # 4. Submit --------------------------------------------------------
        try:
            resp = self._client.submit_order(
                order_payload=quote.order_payload,
                signature=signed.signature,
                chain_id=self._chain_id,
            )
        except UniswapXError as exc:
            return self._reject_with(signal, mark_price, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._reject_with(
                signal, mark_price, f"order_transport_error: {exc!s}"
            )
        if not resp.accepted:
            return self._reject_with(
                signal,
                mark_price,
                f"venue_rejected: {resp.raw!r}"[:200],
            )

        meta: Mapping[str, str] = {
            "order_hash": resp.order_hash,
            "signer": self._signer_address or "",
            "chain_id": str(self._chain_id),
            "execution_kind": "uniswapx_intent",
            "amount_in": str(amount_in),
            "amount_out_quote": str(quote.amount_out_quote),
            "amount_out_min": str(quote.amount_out_min),
        }
        # UniswapX intents are SUBMITTED at sign time, not FILLED — the
        # filler network settles asynchronously. Returning SUBMITTED
        # keeps the audit trail honest (INV-65: the executor must not
        # fabricate fills).
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=float(amount_in),
            price=mark_price,
            status=ExecutionStatus.SUBMITTED,
            venue=self.venue,
            order_id=resp.order_hash,
            meta=meta,
            produced_by_engine="execution_engine",
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _load_private_key(path: str | None) -> str:
        if not path:
            raise ValueError("private_key_path required")
        raw = Path(path).read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError("private key file empty")
        if not raw.startswith("0x"):
            raw = "0x" + raw
        if len(raw) != 66:
            raise ValueError(
                f"invalid private key length: {len(raw)} (want 66)"
            )
        return raw

    def _default_nonce(self) -> int:
        # UniswapX uses a monotonic uint256 nonce; mixing wall_ns()
        # with a per-instance counter avoids collisions on rapid
        # back-to-back orders even within the same nanosecond.
        self._counter += 1
        return wall_ns() + self._counter

    def _amount_in_from_signal(
        self, signal: SignalEvent, mark_price: float
    ) -> int:
        meta_amt = signal.meta.get("amount_in") if signal.meta else None
        if meta_amt is not None:
            try:
                return int(meta_amt)
            except (TypeError, ValueError):
                return 0
        # Fall back to USD-notional sizing — the operator can pin
        # ``signal.meta["size_usd"]`` to override the adapter default.
        size_usd_raw = (
            signal.meta.get("size_usd") if signal.meta else None
        )
        try:
            size_usd = (
                float(size_usd_raw)
                if size_usd_raw is not None
                else self._default_size_usd
            )
        except (TypeError, ValueError):
            size_usd = self._default_size_usd
        if mark_price <= 0:
            return 0
        # Assume 6-decimal stablecoin input (USDC/USDT) — the
        # commonest UniswapX swap shape. Operators with non-stablecoin
        # inputs must pin ``meta["amount_in"]`` directly.
        return int(size_usd * 1_000_000)

    def _slippage_bps_from_signal(self, signal: SignalEvent) -> int:
        if not signal.meta:
            return self._max_slippage_bps
        raw = signal.meta.get("slippage_bps")
        if raw is None:
            return self._max_slippage_bps
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return self._max_slippage_bps
        return max(0, min(v, self._max_slippage_bps))

    def _extract_outputs(
        self,
        order_payload: Mapping[str, object],
        signal: SignalEvent,
    ) -> tuple[DutchOutput, ...]:
        outputs_raw = order_payload.get("outputs")
        if not isinstance(outputs_raw, list) or not outputs_raw:
            return ()
        out: list[DutchOutput] = []
        recipient = self._signer_address or ""
        for row in outputs_raw:
            if not isinstance(row, Mapping):
                continue
            token = str(row.get("token") or "")
            start = _to_int(row.get("startAmount"))
            end = _to_int(row.get("endAmount"))
            row_recipient = str(row.get("recipient") or recipient)
            if not token or start <= 0:
                continue
            out.append(
                DutchOutput(
                    token=token,
                    start_amount=start,
                    end_amount=end if end > 0 else start,
                    recipient=row_recipient,
                )
            )
        # Force the SELL side to mean "swap input -> output", BUY to
        # mean the same with reversed legs at the call-site (left to
        # the caller's signal construction). We do not reorder here;
        # the quote endpoint already encoded the correct direction.
        del signal  # unused — direction was applied to the quote
        return tuple(out)

    def _reject_with(
        self,
        signal: SignalEvent,
        mark_price: float,
        reason: str,
    ) -> ExecutionEvent:
        meta: Mapping[str, str] = {
            "reason": reason,
            "chain_id": str(self._chain_id),
            "adapter_state": self._state.value,
            "signer": self._signer_address or "",
        }
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side if isinstance(signal.side, Side) else Side.BUY,
            qty=0.0,
            price=mark_price,
            status=ExecutionStatus.REJECTED,
            venue=self.venue,
            order_id="",
            meta=meta,
            produced_by_engine="execution_engine",
        )


def _to_int(raw: object) -> int:
    if raw is None:
        return 0
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return int(float(raw))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0


__all__ = [
    "DEFAULT_DEADLINE_PADDING_S",
    "DEFAULT_DECAY_WINDOW_S",
    "DEFAULT_REACTOR_ADDRESS",
    "UniswapXAdapter",
]
