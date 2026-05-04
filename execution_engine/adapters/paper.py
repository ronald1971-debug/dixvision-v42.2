"""Deterministic paper broker (Phase E1 + Paper-S2 upgrade).

The paper broker is the v1 default for ``execution_engine`` and the
reference implementation of :class:`BrokerAdapter`. All fills are
deterministic functions of ``(signal, mark_price)`` and a monotonic
counter; no clocks, no randomness, no external IO. This keeps the
Phase E1 latency SLO test (PERF-01..02) and the deterministic-replay
invariant (INV-15 / TEST-01) honest.

Paper-S2 adds higher-fidelity simulation while preserving every v1
guarantee:

* **Deterministic latency model.** ``ts_ns`` on the returned
  :class:`ExecutionEvent` is bumped by a deterministic delay derived
  from ``(signal.ts_ns, counter)`` via Knuth's multiplicative hash —
  no wall-clock reads, no PRNG.
* **Maker / taker fee model.** Fees are charged in basis points
  against fill notional and surfaced via ``meta['fee_usd']``; the
  default is zero so existing callers keep their idealised pricing.
* **Virtual balance ledger.** Each fill mutates a per-symbol position
  and a single cash balance, exposed via :meth:`cash_balance` and
  :meth:`position`. ``REJECTED`` / ``FAILED`` outputs do **not** move
  the ledger.
* **Partial fills.** When ``signal.meta['max_fill_qty']`` caps the
  requested quantity below the configured default, the broker emits
  ``ExecutionStatus.PARTIALLY_FILLED`` with the actual filled qty,
  the requested qty, and the remaining qty in ``meta``.
* **Fill tracking ring.** A bounded :class:`collections.deque` retains
  the last *N* successful fills (``FILLED`` and ``PARTIALLY_FILLED``)
  for replay / debugging, exposed via :meth:`recent_fills`.

All v1 callers (``PaperBroker()``, ``PaperBroker(slippage_bps=...)``,
``PaperBroker(default_qty=...)``) continue to work unchanged because
every new feature is keyword-only and defaults to a no-op.
"""

from __future__ import annotations

from collections import deque

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)

# Knuth's multiplicative hash (32-bit) — used to spread the
# deterministic latency value across the configured jitter window
# without needing a PRNG.
_KNUTH_32 = 2654435761
_U32_MASK = 0xFFFFFFFF


class PaperBroker:
    """Reference broker — fills at ``mark_price`` plus deterministic slippage.

    Args:
        slippage_bps: Linear slippage in basis points applied on the same
            side as the order (positive for ``BUY``, negative for ``SELL``).
            Default ``0.0`` for an idealised reference.
        default_qty: Default fill quantity when the signal does not carry
            one in ``meta`` (Phase E1: ``SignalEvent`` has no ``qty`` field;
            sizing is configured here).
        taker_fee_bps: Fee charged against fill notional in basis points
            for taker fills. Default ``0.0`` (idealised).
        maker_fee_bps: Fee charged for maker fills. Default ``0.0``. The
            current ``submit`` path treats every fill as a taker because
            slippage is applied on the same side as the order; the field
            is reserved for future post-only / quoting modes.
        latency_ns_base: Deterministic constant latency (in nanoseconds)
            added to ``signal.ts_ns`` before stamping the
            :class:`ExecutionEvent`. Default ``0``.
        latency_ns_jitter: Width of the deterministic jitter window
            (inclusive). The actual jitter is
            ``hash(ts_ns, counter) % (latency_ns_jitter + 1)``. Default
            ``0`` (no jitter).
        initial_cash: Starting virtual cash balance.
        fill_ring_size: Capacity of the recent-fill ring. Set ``0`` to
            disable retention.
    """

    name: str = "paper"

    def __init__(
        self,
        slippage_bps: float = 0.0,
        default_qty: float = 1.0,
        *,
        taker_fee_bps: float = 0.0,
        maker_fee_bps: float = 0.0,
        latency_ns_base: int = 0,
        latency_ns_jitter: int = 0,
        initial_cash: float = 0.0,
        fill_ring_size: int = 256,
    ) -> None:
        if slippage_bps < 0.0:
            raise ValueError("slippage_bps must be >= 0")
        if default_qty <= 0.0:
            raise ValueError("default_qty must be > 0")
        if taker_fee_bps < 0.0:
            raise ValueError("taker_fee_bps must be >= 0")
        if maker_fee_bps < 0.0:
            raise ValueError("maker_fee_bps must be >= 0")
        if latency_ns_base < 0:
            raise ValueError("latency_ns_base must be >= 0")
        if latency_ns_jitter < 0:
            raise ValueError("latency_ns_jitter must be >= 0")
        if fill_ring_size < 0:
            raise ValueError("fill_ring_size must be >= 0")
        self._slippage_bps = slippage_bps
        self._default_qty = default_qty
        self._taker_fee_bps = taker_fee_bps
        self._maker_fee_bps = maker_fee_bps
        self._latency_ns_base = int(latency_ns_base)
        self._latency_ns_jitter = int(latency_ns_jitter)
        self._counter: int = 0
        self._cash: float = float(initial_cash)
        self._initial_cash: float = float(initial_cash)
        self._positions: dict[str, float] = {}
        self._fill_ring_size: int = int(fill_ring_size)
        # ``deque(maxlen=0)`` would silently drop everything; we instead
        # gate the append on ``_fill_ring_size > 0`` so size==0 means
        # "no ring at all" rather than a no-op append on every fill.
        self._fills: deque[ExecutionEvent] = deque(
            maxlen=fill_ring_size if fill_ring_size > 0 else 1
        )

    # -- ledger accessors ---------------------------------------------------

    def cash_balance(self) -> float:
        """Current virtual cash balance."""
        return self._cash

    def initial_cash(self) -> float:
        """Initial virtual cash balance, useful for P&L calculations."""
        return self._initial_cash

    def position(self, symbol: str) -> float:
        """Net position (signed quantity) for ``symbol``."""
        return self._positions.get(symbol, 0.0)

    def positions(self) -> dict[str, float]:
        """Snapshot copy of all non-trivial positions."""
        return {sym: qty for sym, qty in self._positions.items() if qty != 0.0}

    def recent_fills(self, n: int | None = None) -> list[ExecutionEvent]:
        """Return up to the last *n* successful fills (or all retained)."""
        if n is None:
            return list(self._fills)
        if n <= 0:
            return []
        return list(self._fills)[-n:]

    # -- main entrypoint ----------------------------------------------------

    def submit(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        if mark_price <= 0.0:
            self._counter += 1
            return ExecutionEvent(
                ts_ns=signal.ts_ns,
                symbol=signal.symbol,
                side=signal.side,
                qty=0.0,
                price=0.0,
                status=ExecutionStatus.FAILED,
                venue=self.name,
                order_id="",
                meta={"reason": "non-positive mark_price"},
                produced_by_engine="execution_engine",
            )

        self._counter += 1
        order_id = f"PAPER-{self._counter:08d}"

        if signal.side is Side.HOLD:
            return ExecutionEvent(
                ts_ns=signal.ts_ns,
                symbol=signal.symbol,
                side=Side.HOLD,
                qty=0.0,
                price=mark_price,
                status=ExecutionStatus.REJECTED,
                venue=self.name,
                order_id=order_id,
                meta={"reason": "HOLD signal"},
                produced_by_engine="execution_engine",
            )

        slip = mark_price * (self._slippage_bps / 10_000.0)
        if signal.side is Side.BUY:
            fill_price = mark_price + slip
        else:  # SELL
            fill_price = mark_price - slip

        requested_qty, fill_qty = self._qty_for(signal)
        partial = fill_qty < requested_qty
        if fill_qty <= 0.0:
            # Cap explicitly drove the fill to zero — emit a REJECTED
            # row so the audit ledger can see the cap fired without
            # touching the ledger.
            return ExecutionEvent(
                ts_ns=signal.ts_ns,
                symbol=signal.symbol,
                side=signal.side,
                qty=0.0,
                price=fill_price,
                status=ExecutionStatus.REJECTED,
                venue=self.name,
                order_id=order_id,
                meta={"reason": "max_fill_qty cap is zero"},
                produced_by_engine="execution_engine",
            )

        notional = fill_price * fill_qty
        fee = notional * (self._taker_fee_bps / 10_000.0)

        # Mutate ledger.
        if signal.side is Side.BUY:
            self._cash -= notional + fee
            self._positions[signal.symbol] = (
                self._positions.get(signal.symbol, 0.0) + fill_qty
            )
        else:  # SELL
            self._cash += notional - fee
            self._positions[signal.symbol] = (
                self._positions.get(signal.symbol, 0.0) - fill_qty
            )

        # Deterministic latency stamp.
        latency_ns = self._latency_for(signal.ts_ns)
        ts_ns_filled = signal.ts_ns + latency_ns

        meta: dict[str, str] = {
            "adapter": self.name,
            "fee_bps": f"{self._taker_fee_bps:.10g}",
            "fee_usd": f"{fee:.10g}",
            "notional_usd": f"{notional:.10g}",
            "cash_after": f"{self._cash:.10g}",
            "position_after": (
                f"{self._positions.get(signal.symbol, 0.0):.10g}"
            ),
            "latency_ns": str(latency_ns),
        }
        if partial:
            meta["requested_qty"] = f"{requested_qty:.10g}"
            meta["filled_qty"] = f"{fill_qty:.10g}"
            meta["remaining_qty"] = f"{requested_qty - fill_qty:.10g}"

        status = (
            ExecutionStatus.PARTIALLY_FILLED
            if partial
            else ExecutionStatus.FILLED
        )

        evt = ExecutionEvent(
            ts_ns=ts_ns_filled,
            symbol=signal.symbol,
            side=signal.side,
            qty=fill_qty,
            price=fill_price,
            status=status,
            venue=self.name,
            order_id=order_id,
            meta=meta,
            produced_by_engine="execution_engine",
        )

        if self._fill_ring_size > 0:
            self._fills.append(evt)
        return evt

    # -- internals ----------------------------------------------------------

    def _latency_for(self, signal_ts_ns: int) -> int:
        if self._latency_ns_jitter == 0:
            return self._latency_ns_base
        # Knuth multiplicative hash on (ts_ns, counter) — reproducible,
        # no PRNG, well-spread in the bottom bits.
        mix = (signal_ts_ns ^ (self._counter * _KNUTH_32)) & _U32_MASK
        return self._latency_ns_base + (mix % (self._latency_ns_jitter + 1))

    def _qty_for(self, signal: SignalEvent) -> tuple[float, float]:
        """Return ``(requested_qty, fill_qty)`` so callers can detect partials."""
        requested = self._default_qty
        meta_qty = signal.meta.get("qty")
        if meta_qty is not None:
            try:
                value = float(meta_qty)
                if value > 0.0:
                    requested = value
            except (TypeError, ValueError):
                pass

        cap_raw = signal.meta.get("max_fill_qty")
        if cap_raw is not None:
            try:
                cap = float(cap_raw)
            except (TypeError, ValueError):
                return requested, requested
            if cap < 0.0:
                return requested, requested
            if cap < requested:
                return requested, cap
        return requested, requested


__all__ = ["PaperBroker"]
