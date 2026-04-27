"""Learning interface — Phase 3.

The intelligence engine's *outbound* contract to the learning engine.
Indira does **not** import learning_engine (INV-08 / INV-11); it only
emits ``FeedbackRecord`` rows that the learning engine consumes off the
bus or via a periodic pull.

Each :class:`FeedbackRecord` ties a ``SignalEvent`` to its eventual
``ExecutionEvent`` outcome plus a realised PnL hint, so the learning
engine has everything needed to score the strategy without having to
re-derive cross-engine state.

The interface is pure-Python, IO-free, and clock-free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """One learning sample.

    Attributes:
        ts_ns: Time of the execution outcome.
        strategy_id: First plugin in ``signal.plugin_chain``; ``""``
            when the signal had no chain.
        symbol: Symbol the signal targeted.
        side: Side the signal proposed.
        signal_confidence: Original confidence on ``signal``.
        execution_status: Final state of the execution (FILLED,
            REJECTED, PARTIALLY_FILLED, …).
        executed_qty: Filled quantity (``0`` for rejects).
        executed_price: Average fill price (``0`` for rejects).
        realised_pnl: Realised PnL from ``mark_price`` − ``executed_price``
            on a long, sign-flipped on a short. ``0`` when not filled or
            no mark provided.
    """

    ts_ns: int
    strategy_id: str
    symbol: str
    side: Side
    signal_confidence: float
    execution_status: ExecutionStatus
    executed_qty: float
    executed_price: float
    realised_pnl: float

    def is_realised(self) -> bool:
        return self.execution_status in (
            ExecutionStatus.FILLED,
            ExecutionStatus.PARTIALLY_FILLED,
        )


class LearningInterface:
    """Builds :class:`FeedbackRecord` rows from signal/execution pairs."""

    name: str = "learning_interface"
    spec_id: str = "IND-LI-01"

    def __init__(self) -> None:
        self._buffer: list[FeedbackRecord] = []

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def drain(self) -> tuple[FeedbackRecord, ...]:
        out = tuple(self._buffer)
        self._buffer.clear()
        return out

    # -- mutations ---------------------------------------------------------

    def record(
        self,
        *,
        signal: SignalEvent,
        execution: ExecutionEvent,
        mark_price: float | None = None,
    ) -> FeedbackRecord:
        if execution.symbol != signal.symbol:
            raise ValueError(
                "signal.symbol/execution.symbol mismatch — feedback "
                "must reference the same instrument"
            )
        strategy_id = signal.plugin_chain[0] if signal.plugin_chain else ""

        if execution.status in (
            ExecutionStatus.FILLED,
            ExecutionStatus.PARTIALLY_FILLED,
        ):
            executed_qty = execution.qty
            executed_price = execution.price
        else:
            executed_qty = 0.0
            executed_price = 0.0

        pnl = self._pnl(
            side=signal.side,
            qty=executed_qty,
            entry_price=executed_price,
            mark_price=mark_price,
        )

        record = FeedbackRecord(
            ts_ns=execution.ts_ns,
            strategy_id=strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            signal_confidence=signal.confidence,
            execution_status=execution.status,
            executed_qty=executed_qty,
            executed_price=executed_price,
            realised_pnl=pnl,
        )
        self._buffer.append(record)
        return record

    def record_many(
        self,
        pairs: Iterable[tuple[SignalEvent, ExecutionEvent]],
        *,
        mark_price: float | None = None,
    ) -> tuple[FeedbackRecord, ...]:
        return tuple(
            self.record(
                signal=s, execution=e, mark_price=mark_price
            )
            for s, e in pairs
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _pnl(
        *,
        side: Side,
        qty: float,
        entry_price: float,
        mark_price: float | None,
    ) -> float:
        if mark_price is None or qty == 0.0 or entry_price == 0.0:
            return 0.0
        if side is Side.BUY:
            return qty * (mark_price - entry_price)
        if side is Side.SELL:
            return qty * (entry_price - mark_price)
        return 0.0


__all__ = ["FeedbackRecord", "LearningInterface"]
