"""FeedbackCollector — EXEC-09.

Bridges the execution domain to the closed loop (Build Compiler Spec §8):

    ExecutionEvent (terminal) → TradeOutcome → Dyon / Learning

Pure, deterministic, IO-free. Caller supplies ``ts_ns``; no system clocks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Final

from core.contracts.events import ExecutionStatus
from core.contracts.learning import TradeOutcome

_TERMINAL_STATUSES: Final[frozenset[ExecutionStatus]] = frozenset(
    {
        ExecutionStatus.FILLED,
        ExecutionStatus.PARTIALLY_FILLED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.REJECTED,
        ExecutionStatus.FAILED,
    }
)


class FeedbackCollector:
    """Buffers execution outcomes for the closed-loop consumer."""

    name: str = "feedback_collector"
    spec_id: str = "EXEC-09"

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue: deque[TradeOutcome] = deque()

    def record(
        self,
        *,
        ts_ns: int,
        strategy_id: str,
        symbol: str,
        qty: float,
        pnl: float,
        status: ExecutionStatus,
        venue: str = "",
        order_id: str = "",
        meta: Mapping[str, str] | None = None,
    ) -> TradeOutcome | None:
        """Append a terminal outcome. Non-terminal statuses are dropped."""
        if status not in _TERMINAL_STATUSES:
            return None
        if not strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not symbol:
            raise ValueError("symbol must be non-empty")
        outcome = TradeOutcome(
            ts_ns=ts_ns,
            strategy_id=strategy_id,
            symbol=symbol,
            qty=qty,
            pnl=pnl,
            status=status,
            venue=venue,
            order_id=order_id,
            meta=dict(meta or {}),
        )
        self._queue.append(outcome)
        return outcome

    def drain(self) -> tuple[TradeOutcome, ...]:
        """Return + clear all buffered outcomes in insertion order."""
        out = tuple(self._queue)
        self._queue.clear()
        return out

    def peek(self) -> tuple[TradeOutcome, ...]:
        """Return buffered outcomes without clearing."""
        return tuple(self._queue)

    def extend(self, outcomes: Iterable[TradeOutcome]) -> None:
        """Append already-built outcomes (used by replay tooling)."""
        for o in outcomes:
            self._queue.append(o)

    def __len__(self) -> int:
        return len(self._queue)


__all__ = ["FeedbackCollector"]
