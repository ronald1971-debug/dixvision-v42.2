# ADAPTED FROM: ccxt/python/ccxt/binance.py
# ADAPTED FROM: ccxt/python/ccxt/base/exchange.py
# ADAPTED FROM: ccxt/python/ccxt/base/errors.py
"""Binance spot adapter (S-01) — `BrokerAdapter` over `ccxt.binance`.

This is the first canonical-tier adaptation from the master canonical
plan (S-01 ``ccxt`` → ``execution_engine/adapters/binance.py``). It
takes the battle-tested error hierarchy and request/response shape that
``ccxt`` ships and re-projects them through the DIX
:class:`BrokerAdapter` Protocol so an approved
:class:`SignalEvent` becomes a real Binance spot REST call.

What survives from upstream
---------------------------
* The full ``ccxt`` error hierarchy is honoured — anything raised under
  ``ccxt.BaseError`` is caught here, classified by exception class, and
  surfaced as :attr:`ExecutionStatus.FAILED` with
  ``meta["ccxt_error_class"]`` + ``meta["ccxt_error"]``. The classes
  themselves are never simplified or collapsed.
* The ``create_order`` request shape (``symbol``, ``type``, ``side``,
  ``amount``, ``price``, ``params``) is preserved verbatim.
* The Binance response normalisation lives in ``_normalise_order``
  and reads the same ``id`` / ``status`` / ``filled`` / ``price`` /
  ``average`` / ``cost`` keys that ``ccxt.base.exchange.Exchange``'s
  ``parse_order`` populates.

What is rewritten behind DIX contracts
--------------------------------------
* ``ccxt`` is *lazy-imported* inside :meth:`connect` — no top-level
  ``import ccxt`` and no top-level ``import ccxt.async_support``. The
  adapter module imports cleanly even when ``ccxt`` is not installed,
  so unit tests + the operator dashboard never need the real package.
* No ``time.sleep``, no ``asyncio.sleep``, no daemon thread, no
  internal retry loop. Every :meth:`submit` is a single synchronous
  call with no implicit clocks (INV-15 / T1 / B-CLOCK).
* All time stamping uses ``signal.ts_ns`` from the inbound
  :class:`SignalEvent` plus a deterministic per-adapter monotonic
  counter — ``ccxt``'s internal ``self.milliseconds()`` is never read
  through to the DIX ledger.
* Errors never escape :meth:`_submit_live` — every ``ccxt`` exception
  becomes an :class:`ExecutionEvent` with ``status=FAILED`` so the
  audit ledger never receives an exception trace and downstream
  consumers can replay deterministically (TEST-01).
* Until both ``api_key`` and ``api_secret`` are wired, the adapter
  stays in :attr:`AdapterState.DISCONNECTED` and rejects every
  ``submit()`` with a structured ``meta`` — it never silently fakes a
  fill (INV-56 Triad Lock: an Executor that fakes fills is a hard
  authority breach).

This module never reads ``os.environ``. Credentials are passed in
explicitly so a malformed env doesn't silently route a real fill to
the wrong account (INV-65 per-decision audit truthfulness).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

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

# pip dependency flag (the adapter only imports ccxt at connect()-time,
# so the module itself is importable without the package installed).
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("ccxt",)


# Mapping from ccxt's normalised ``order["status"]`` strings to the DIX
# :class:`ExecutionStatus` enum. ccxt itself emits the lower-case
# tokens ``"open"``, ``"closed"``, ``"canceled"``, ``"expired"``,
# ``"rejected"`` — ``parse_order`` in ``ccxt/base/exchange.py``.
_CCXT_STATUS: Mapping[str, ExecutionStatus] = {
    "open": ExecutionStatus.SUBMITTED,
    "closed": ExecutionStatus.FILLED,
    "canceled": ExecutionStatus.CANCELLED,
    "cancelled": ExecutionStatus.CANCELLED,
    "expired": ExecutionStatus.CANCELLED,
    "rejected": ExecutionStatus.REJECTED,
    "failed": ExecutionStatus.FAILED,
}


class BinanceAdapter(LiveAdapterBase):
    """Real Binance spot adapter wrapping ``ccxt.binance``.

    Until :meth:`connect` is called with both ``api_key`` and
    ``api_secret`` populated the adapter stays in scaffold mode and
    every :meth:`submit` returns ``REJECTED`` with
    ``meta["reason"] = "adapter_not_ready"``.

    Args:
        api_key: Binance API key. ``None`` keeps scaffold mode.
        api_secret: Binance API secret. ``None`` keeps scaffold mode.
        sandbox: When ``True`` the underlying ``ccxt`` client is put in
            test-net mode (``set_sandbox_mode(True)``) so no real funds
            move. Defaults to ``True``: requesting *production* mode
            requires an explicit ``sandbox=False`` from the caller.
        default_qty: Fallback quantity (in base-asset units) used when
            ``signal.meta["qty"]`` is not set.
        default_order_type: One of ``"market"`` or ``"limit"``. Used
            when ``signal.meta["order_type"]`` is not set.
        exchange: Pre-built ``ccxt.Exchange`` instance. Tests pass a
            fake here to avoid the real network and pip dependency.
        exchange_factory: Callable returning a fresh ``ccxt`` exchange
            instance; only used when ``exchange`` is ``None``. Defaults
            to a lazy-import factory that calls
            ``ccxt.binance({...})``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        sandbox: bool = True,
        default_qty: float = 0.0,
        default_order_type: str = "market",
        exchange: Any | None = None,
        exchange_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        super().__init__(name="binance_spot", venue="binance:spot")
        if default_qty < 0.0:
            raise ValueError("default_qty must be >= 0")
        if default_order_type not in ("market", "limit"):
            raise ValueError(
                "default_order_type must be 'market' or 'limit'"
            )
        self._api_key = api_key
        self._api_secret = api_secret
        self._sandbox = bool(sandbox)
        self._default_qty = float(default_qty)
        self._default_order_type = default_order_type
        self._exchange_factory = exchange_factory
        self._exchange: Any | None = exchange
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Build (or accept the injected) ``ccxt`` exchange and arm.

        - If both ``api_key`` and ``api_secret`` are populated *or* an
          ``exchange`` was injected at construction time, the adapter
          flips to :attr:`AdapterState.READY`.
        - Otherwise the adapter stays in :attr:`AdapterState.DISCONNECTED`
          with a structured detail string and the operator dashboard
          can show it as a scaffold.

        ``ccxt`` is imported here, never at module load.
        """
        if self._exchange is not None:
            self._state = AdapterState.READY
            self._detail = "exchange injected at construction"
            return
        missing: list[str] = []
        if self._api_key is None:
            missing.append("api_key")
        if self._api_secret is None:
            missing.append("api_secret")
        if missing:
            self._state = AdapterState.DISCONNECTED
            self._detail = "missing: " + ", ".join(missing) + " — scaffold mode"
            return
        try:
            factory = self._exchange_factory or _default_ccxt_factory
            self._exchange = factory(
                {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "enableRateLimit": True,
                }
            )
            if self._sandbox and hasattr(self._exchange, "set_sandbox_mode"):
                self._exchange.set_sandbox_mode(True)
        except Exception as exc:  # noqa: BLE001  classify in detail
            self._state = AdapterState.DEGRADED
            self._detail = f"ccxt construction failed: {exc.__class__.__name__}"
            return
        self._state = AdapterState.READY
        self._detail = (
            "sandbox" if self._sandbox else "production"
        ) + " — credentials loaded"

    def disconnect(self) -> None:
        super().disconnect()
        self._exchange = None

    # ------------------------------------------------------------------
    # BrokerAdapter
    # ------------------------------------------------------------------

    def _submit_live(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        if self._exchange is None:
            return self._reject(signal, mark_price)
        if mark_price <= 0.0:
            return self._fail(
                signal,
                mark_price,
                reason="non-positive mark_price",
                ccxt_error_class="",
                ccxt_error="",
            )
        if signal.side is Side.HOLD:
            return ExecutionEvent(
                ts_ns=signal.ts_ns,
                symbol=signal.symbol,
                side=Side.HOLD,
                qty=0.0,
                price=mark_price,
                status=ExecutionStatus.REJECTED,
                venue=self.venue,
                order_id="",
                meta={"reason": "HOLD signal"},
                produced_by_engine="execution_engine",
            )

        qty = self._qty_for(signal)
        if qty <= 0.0:
            return ExecutionEvent(
                ts_ns=signal.ts_ns,
                symbol=signal.symbol,
                side=signal.side,
                qty=0.0,
                price=mark_price,
                status=ExecutionStatus.REJECTED,
                venue=self.venue,
                order_id="",
                meta={"reason": "non-positive qty"},
                produced_by_engine="execution_engine",
            )

        order_type = self._order_type_for(signal)
        ccxt_side = "buy" if signal.side is Side.BUY else "sell"
        price_arg: float | None = (
            float(signal.meta.get("limit_price", mark_price))
            if order_type == "limit"
            else None
        )

        try:
            raw = self._exchange.create_order(
                symbol=signal.symbol,
                type=order_type,
                side=ccxt_side,
                amount=qty,
                price=price_arg,
                params={},
            )
        except Exception as exc:  # noqa: BLE001  ccxt taxonomy below
            return self._fail(
                signal,
                mark_price,
                reason="ccxt_error",
                ccxt_error_class=exc.__class__.__name__,
                ccxt_error=str(exc),
            )

        return self._normalise_order(
            signal=signal,
            mark_price=mark_price,
            raw=raw,
            requested_qty=qty,
            order_type=order_type,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _qty_for(self, signal: SignalEvent) -> float:
        raw = signal.meta.get("qty")
        if raw is None:
            return self._default_qty
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return 0.0
        # Reject NaN/-inf via the same `not (>= 0)` IEEE-754 pattern used
        # in PortfolioAllocator (PR #234 follow-up).
        if not (v >= 0.0):
            return 0.0
        return v

    def _order_type_for(self, signal: SignalEvent) -> str:
        raw = str(signal.meta.get("order_type", self._default_order_type))
        if raw not in ("market", "limit"):
            return self._default_order_type
        return raw

    def _normalise_order(
        self,
        *,
        signal: SignalEvent,
        mark_price: float,
        raw: Mapping[str, Any] | None,
        requested_qty: float,
        order_type: str,
    ) -> ExecutionEvent:
        """Project ccxt's ``parse_order`` shape onto :class:`ExecutionEvent`.

        ccxt normalises every venue's response into a dict containing
        ``id``, ``status``, ``filled``, ``amount``, ``price``,
        ``average``, ``cost``. The DIX shape pulls the deterministic
        subset needed for the audit ledger.
        """
        self._counter += 1
        if not isinstance(raw, Mapping):
            return self._fail(
                signal,
                mark_price,
                reason="ccxt response not a mapping",
                ccxt_error_class="",
                ccxt_error="",
            )

        ccxt_status = str(raw.get("status", "")).lower()
        status = _CCXT_STATUS.get(ccxt_status, ExecutionStatus.FAILED)

        # Filled qty: ccxt populates ``filled`` for partial / closed
        # orders. Fall back to ``amount`` for closed orders that omit
        # the explicit ``filled`` field.
        filled = _safe_float(raw.get("filled"))
        if filled is None and status is ExecutionStatus.FILLED:
            filled = _safe_float(raw.get("amount"))
        if filled is None:
            filled = 0.0

        if (
            status is ExecutionStatus.FILLED
            and filled > 0.0
            and filled + 1e-12 < requested_qty
        ):
            status = ExecutionStatus.PARTIALLY_FILLED

        # Average fill price preferred; fall back to limit price; then
        # to mark.
        avg = _safe_float(raw.get("average"))
        if avg is None or avg <= 0.0:
            avg = _safe_float(raw.get("price"))
        if avg is None or avg <= 0.0:
            avg = mark_price

        order_id = str(raw.get("id", "") or "")
        cost = _safe_float(raw.get("cost"))

        meta: dict[str, str] = {
            "adapter": self.name,
            "ccxt_status": ccxt_status,
            "order_type": order_type,
            "requested_qty": f"{requested_qty:.10g}",
            "filled_qty": f"{filled:.10g}",
            "fill_seq": str(self._counter),
            "sandbox": "1" if self._sandbox else "0",
        }
        if cost is not None:
            meta["notional_usd"] = f"{cost:.10g}"

        if status is ExecutionStatus.PARTIALLY_FILLED:
            meta["remaining_qty"] = f"{max(requested_qty - filled, 0.0):.10g}"

        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=filled,
            price=avg,
            status=status,
            venue=self.venue,
            order_id=order_id,
            meta=meta,
            produced_by_engine="execution_engine",
        )

    def _fail(
        self,
        signal: SignalEvent,
        mark_price: float,
        *,
        reason: str,
        ccxt_error_class: str,
        ccxt_error: str,
    ) -> ExecutionEvent:
        meta: dict[str, str] = {
            "adapter": self.name,
            "reason": reason,
        }
        if ccxt_error_class:
            meta["ccxt_error_class"] = ccxt_error_class
        if ccxt_error:
            # Cap the embedded message so a chatty venue cannot bloat
            # the ledger row past a sensible bound.
            meta["ccxt_error"] = ccxt_error[:512]
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=0.0,
            price=mark_price,
            status=ExecutionStatus.FAILED,
            venue=self.venue,
            order_id="",
            meta=meta,
            produced_by_engine="execution_engine",
        )


def _safe_float(value: Any) -> float | None:
    """ccxt's ``safe_float`` analogue — returns ``None`` for missing /
    unparseable values so the caller can pick a sane fallback."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN/+inf via the IEEE-754 trick — they are never sane
    # ledger values for a fill quantity or price.
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _default_ccxt_factory(config: Mapping[str, Any]) -> Any:
    """Lazy-import factory used by :meth:`BinanceAdapter.connect`.

    Importing ``ccxt`` at module load would force every consumer of
    the adapter (including unit tests + the operator dashboard) to
    install the dependency. By keeping the import here we make the
    adapter module importable even without ``ccxt`` installed; the
    error only surfaces at :meth:`BinanceAdapter.connect` time, when
    the operator has explicitly opted in to live mode.
    """
    import ccxt  # noqa: PLC0415  lazy by design

    return ccxt.binance(dict(config))


__all__ = ["BinanceAdapter", "NEW_PIP_DEPENDENCIES"]
