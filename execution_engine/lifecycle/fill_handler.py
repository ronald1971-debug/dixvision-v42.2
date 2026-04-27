"""Fill bookkeeping — Phase 2 / v2-C.

Applies one or more :class:`FillEvent` rows against the
:class:`OrderStateMachine` and maintains per-order aggregates
(``filled_qty``, ``avg_price``, ``last_fill_ts_ns``). Pure-python,
IO-free, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from execution_engine.lifecycle.order_state_machine import (
    OrderState,
    OrderStateMachine,
)


@dataclass(frozen=True, slots=True)
class FillEvent:
    """A single fill from a venue.

    ``order_id`` matches an order opened in :class:`OrderStateMachine`.
    ``qty`` is positive (the side comes from the originating order).
    """

    ts_ns: int
    order_id: str
    qty: float
    price: float


@dataclass(slots=True)
class OrderFillState:
    """Aggregate fill bookkeeping for one order."""

    order_id: str
    target_qty: float
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fills: list[FillEvent] = field(default_factory=list)
    last_fill_ts_ns: int = 0

    def remaining(self) -> float:
        return max(0.0, self.target_qty - self.filled_qty)

    def is_filled(self) -> bool:
        # Treat float epsilon as filled to avoid spurious PARTIAL states.
        return self.filled_qty + 1e-12 >= self.target_qty


class FillHandler:
    """Deterministic fill aggregator + state-machine driver.

    Args:
        fsm: Owns the state writes — ``FillHandler`` only ever calls
            ``fsm.transition`` with legal edges.

    Invariants:

    * ``apply`` is idempotent on duplicate ``(order_id, ts_ns, qty,
      price)`` tuples — duplicates are rejected to keep replay
      deterministic.
    * Once ``filled_qty >= target_qty``, further fills raise.
    """

    name: str = "fill_handler"
    spec_id: str = "EXEC-LC-02"

    def __init__(self, fsm: OrderStateMachine) -> None:
        self._fsm = fsm
        self._book: dict[str, OrderFillState] = {}
        self._seen: set[tuple[str, int, float, float]] = set()

    # -- queries -----------------------------------------------------------

    def state(self, order_id: str) -> OrderFillState | None:
        return self._book.get(order_id)

    # -- mutations ---------------------------------------------------------

    def register(
        self,
        *,
        order_id: str,
        target_qty: float,
        ts_ns: int,
    ) -> OrderFillState:
        """Register a fresh order's target qty and move it to PENDING."""
        if target_qty <= 0.0:
            raise ValueError("target_qty must be > 0")
        if order_id in self._book:
            raise ValueError(f"already registered: {order_id}")
        # Drive the FSM first; if it raises (unknown order, illegal edge),
        # the book entry must not exist — otherwise a retry would fail
        # with "already registered" and apply() would operate against an
        # order whose state is still NEW.
        self._fsm.transition(
            order_id=order_id,
            new_state=OrderState.PENDING,
            ts_ns=ts_ns,
            reason="submitted",
        )
        state = OrderFillState(order_id=order_id, target_qty=target_qty)
        self._book[order_id] = state
        return state

    def apply(self, fill: FillEvent) -> OrderFillState:
        """Record a fill and drive the FSM to the appropriate state."""
        if fill.qty <= 0.0:
            raise ValueError("fill qty must be > 0")
        if fill.price <= 0.0:
            raise ValueError("fill price must be > 0")
        state = self._book.get(fill.order_id)
        if state is None:
            raise KeyError(f"unknown order: {fill.order_id}")

        key = (fill.order_id, fill.ts_ns, fill.qty, fill.price)
        if key in self._seen:
            return state
        self._seen.add(key)

        if state.is_filled():
            raise ValueError(f"order already filled: {fill.order_id}")

        # Volume-weighted average price — pure arithmetic, no IO.
        new_filled = state.filled_qty + fill.qty
        if new_filled > state.target_qty + 1e-9:
            raise ValueError(
                f"overfill: {new_filled} > target {state.target_qty}"
            )
        notional = state.avg_price * state.filled_qty + fill.price * fill.qty
        state.filled_qty = new_filled
        state.avg_price = notional / new_filled if new_filled > 0.0 else 0.0
        state.fills.append(fill)
        state.last_fill_ts_ns = fill.ts_ns

        if state.is_filled():
            target_state = OrderState.FILLED
            reason = "fully_filled"
        else:
            target_state = OrderState.PARTIALLY_FILLED
            reason = "partial_fill"
        self._fsm.transition(
            order_id=fill.order_id,
            new_state=target_state,
            ts_ns=fill.ts_ns,
            reason=reason,
        )
        return state


__all__ = ["FillEvent", "FillHandler", "OrderFillState"]
