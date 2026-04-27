"""Order State Machine — Phase 2 / v2-C.

A deterministic finite-state machine that owns the lifecycle of a single
order from creation to terminal state. Every legal edge is declared in
:data:`LEGAL_ORDER_TRANSITIONS`; every illegal edge raises
:class:`StateTransitionError`.

States::

    NEW
      └─► PENDING
            ├─► PARTIALLY_FILLED ──► FILLED ──► CLOSED
            ├─► FILLED ───────────────────────► CLOSED
            ├─► CANCELLED ───────────────────► CLOSED
            └─► ERROR ────────────────────────► CLOSED
    (CANCELLED / ERROR may also be reached from PARTIALLY_FILLED.)

Determinism contract (INV-15):

* Every transition records ``(ts_ns, prev, new, reason)``; the history
  list is the canonical replay log for one order.
* No clocks, no randomness, no IO — callers pass ``ts_ns`` from the
  source event (``ExecutionEvent.ts_ns``).
* Same input transitions produce identical history rows in identical
  order across processes (TEST-01).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class OrderState(StrEnum):
    """All states an order can occupy."""

    NEW = "NEW"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


# Legal forward edges — anything outside this set raises.
LEGAL_ORDER_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.NEW: frozenset({OrderState.PENDING, OrderState.ERROR}),
    OrderState.PENDING: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.ERROR,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,  # repeated partial fills
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.ERROR,
        }
    ),
    OrderState.FILLED: frozenset({OrderState.CLOSED}),
    OrderState.CANCELLED: frozenset({OrderState.CLOSED}),
    OrderState.ERROR: frozenset({OrderState.CLOSED}),
    OrderState.CLOSED: frozenset(),  # terminal
}


TERMINAL_STATES: frozenset[OrderState] = frozenset({OrderState.CLOSED})


class StateTransitionError(ValueError):
    """Raised on an illegal :class:`OrderState` edge."""


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One row in an order's transition history (INV-15 replay log)."""

    ts_ns: int
    prev: OrderState
    new: OrderState
    reason: str


@dataclass(slots=True)
class OrderRecord:
    """The state of one order under management.

    Mutating methods are confined to :class:`OrderStateMachine`; callers
    treat the record as a read-only view.
    """

    order_id: str
    state: OrderState = OrderState.NEW
    history: list[TransitionRecord] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class OrderStateMachine:
    """Sole writer of :class:`OrderRecord.state`.

    Args:
        records: Optional pre-existing record book (for replay).
    """

    name: str = "order_state_machine"
    spec_id: str = "EXEC-LC-01"

    def __init__(self, records: dict[str, OrderRecord] | None = None) -> None:
        self._records: dict[str, OrderRecord] = dict(records or {})

    # -- queries -----------------------------------------------------------

    def get(self, order_id: str) -> OrderRecord | None:
        return self._records.get(order_id)

    def all(self) -> tuple[OrderRecord, ...]:
        return tuple(self._records.values())

    def __contains__(self, order_id: object) -> bool:
        return isinstance(order_id, str) and order_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    # -- mutations ---------------------------------------------------------

    def open(
        self,
        *,
        order_id: str,
        ts_ns: int,
        reason: str = "open",
    ) -> OrderRecord:
        """Register a new order in :attr:`OrderState.NEW`.

        Raises:
            ValueError: if ``order_id`` already exists.
        """
        if not order_id:
            raise ValueError("order_id required")
        if order_id in self._records:
            raise ValueError(f"order_id already exists: {order_id}")
        record = OrderRecord(order_id=order_id, state=OrderState.NEW)
        record.history.append(
            TransitionRecord(
                ts_ns=ts_ns,
                prev=OrderState.NEW,
                new=OrderState.NEW,
                reason=reason,
            )
        )
        self._records[order_id] = record
        return record

    def transition(
        self,
        *,
        order_id: str,
        new_state: OrderState,
        ts_ns: int,
        reason: str,
    ) -> OrderRecord:
        """Drive an order to ``new_state``.

        Raises:
            KeyError: unknown ``order_id``.
            StateTransitionError: edge not in
                :data:`LEGAL_ORDER_TRANSITIONS`.
        """
        record = self._records.get(order_id)
        if record is None:
            raise KeyError(f"unknown order_id: {order_id}")
        prev = record.state
        legal = LEGAL_ORDER_TRANSITIONS[prev]
        if new_state not in legal:
            raise StateTransitionError(
                f"illegal edge: {prev.name} -> {new_state.name} for {order_id}"
            )
        record.state = new_state
        record.history.append(
            TransitionRecord(
                ts_ns=ts_ns,
                prev=prev,
                new=new_state,
                reason=reason,
            )
        )
        return record


__all__ = [
    "LEGAL_ORDER_TRANSITIONS",
    "OrderRecord",
    "OrderState",
    "OrderStateMachine",
    "StateTransitionError",
    "TERMINAL_STATES",
    "TransitionRecord",
]
