"""Deterministic paper broker (Phase E1).

The paper broker is the v1 default for ``execution_engine`` and the
reference implementation of :class:`BrokerAdapter`. All fills are
deterministic functions of ``(signal, mark_price)`` and a monotonic
counter; no clocks, no randomness, no external IO. This keeps the
Phase E1 latency SLO test (PERF-01..02) and the deterministic-replay
invariant (INV-15 / TEST-01) honest.
"""

from __future__ import annotations

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)


class PaperBroker:
    """Reference broker — fills at ``mark_price`` plus deterministic slippage.

    Args:
        slippage_bps: Linear slippage in basis points applied on the same
            side as the order (positive for ``BUY``, negative for ``SELL``).
            Default ``0.0`` for an idealised reference.
        default_qty: Default fill quantity when the signal does not carry
            one in ``meta`` (Phase E1: ``SignalEvent`` has no ``qty`` field;
            sizing is configured here).
    """

    name: str = "paper"

    def __init__(
        self,
        slippage_bps: float = 0.0,
        default_qty: float = 1.0,
    ) -> None:
        if slippage_bps < 0.0:
            raise ValueError("slippage_bps must be >= 0")
        if default_qty <= 0.0:
            raise ValueError("default_qty must be > 0")
        self._slippage_bps = slippage_bps
        self._default_qty = default_qty
        self._counter: int = 0

    def submit(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        if mark_price <= 0.0:
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
            )

        slip = mark_price * (self._slippage_bps / 10_000.0)
        if signal.side is Side.BUY:
            fill_price = mark_price + slip
        else:  # SELL
            fill_price = mark_price - slip

        qty = self._qty_for(signal)
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            price=fill_price,
            status=ExecutionStatus.FILLED,
            venue=self.name,
            order_id=order_id,
            meta={"adapter": self.name},
        )

    # -- internals ----------------------------------------------------------

    def _qty_for(self, signal: SignalEvent) -> float:
        meta_qty = signal.meta.get("qty")
        if meta_qty is None:
            return self._default_qty
        try:
            value = float(meta_qty)
        except (TypeError, ValueError):
            return self._default_qty
        if value <= 0.0:
            return self._default_qty
        return value


__all__ = ["PaperBroker"]
