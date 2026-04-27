"""Stop-loss / take-profit manager — Phase 2 / v2-C.

Tracks per-order :class:`Bracket` levels and reports which bracket — if
any — was triggered by a given mark price. Pure-python, IO-free,
deterministic.

Conventions:

* Long position (``side==BUY``): SL is below entry, TP is above.
* Short position (``side==SELL``): SL is above entry, TP is below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.contracts.events import Side


class BracketTrigger(StrEnum):
    NONE = "NONE"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass(frozen=True, slots=True)
class Bracket:
    """SL/TP brackets attached to one order.

    ``stop_loss`` and ``take_profit`` are *price* levels; ``None``
    disables that side of the bracket.
    """

    order_id: str
    side: Side
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass(frozen=True, slots=True)
class BracketEvaluation:
    order_id: str
    trigger: BracketTrigger
    mark: float


class SLTPManager:
    """Per-order bracket book.

    Brackets are added once per order; subsequent updates require an
    explicit :meth:`detach` first to keep the audit trail clean.
    """

    name: str = "sl_tp_manager"
    spec_id: str = "EXEC-LC-05"

    def __init__(self) -> None:
        self._brackets: dict[str, Bracket] = {}

    # -- queries -----------------------------------------------------------

    def get(self, order_id: str) -> Bracket | None:
        return self._brackets.get(order_id)

    def __len__(self) -> int:
        return len(self._brackets)

    # -- mutations ---------------------------------------------------------

    def attach(self, bracket: Bracket) -> None:
        if bracket.order_id in self._brackets:
            raise ValueError(f"bracket already attached: {bracket.order_id}")
        self._validate(bracket)
        self._brackets[bracket.order_id] = bracket

    def detach(self, order_id: str) -> None:
        self._brackets.pop(order_id, None)

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        *,
        order_id: str,
        mark: float,
    ) -> BracketEvaluation:
        if mark <= 0.0:
            raise ValueError("mark must be > 0")
        bracket = self._brackets.get(order_id)
        if bracket is None:
            return BracketEvaluation(
                order_id=order_id,
                trigger=BracketTrigger.NONE,
                mark=mark,
            )
        trigger = self._trigger_for(bracket, mark)
        return BracketEvaluation(
            order_id=order_id,
            trigger=trigger,
            mark=mark,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _validate(bracket: Bracket) -> None:
        if bracket.entry_price <= 0.0:
            raise ValueError("entry_price must be > 0")
        if bracket.side is Side.HOLD:
            raise ValueError("brackets require BUY or SELL")
        if bracket.side is Side.BUY:
            if bracket.stop_loss is not None and bracket.stop_loss >= bracket.entry_price:
                raise ValueError("BUY stop_loss must be below entry")
            if bracket.take_profit is not None and bracket.take_profit <= bracket.entry_price:
                raise ValueError("BUY take_profit must be above entry")
        else:  # SELL
            if bracket.stop_loss is not None and bracket.stop_loss <= bracket.entry_price:
                raise ValueError("SELL stop_loss must be above entry")
            if bracket.take_profit is not None and bracket.take_profit >= bracket.entry_price:
                raise ValueError("SELL take_profit must be below entry")

    @staticmethod
    def _trigger_for(bracket: Bracket, mark: float) -> BracketTrigger:
        if bracket.side is Side.BUY:
            if bracket.stop_loss is not None and mark <= bracket.stop_loss:
                return BracketTrigger.STOP_LOSS
            if bracket.take_profit is not None and mark >= bracket.take_profit:
                return BracketTrigger.TAKE_PROFIT
            return BracketTrigger.NONE
        # SELL
        if bracket.stop_loss is not None and mark >= bracket.stop_loss:
            return BracketTrigger.STOP_LOSS
        if bracket.take_profit is not None and mark <= bracket.take_profit:
            return BracketTrigger.TAKE_PROFIT
        return BracketTrigger.NONE


__all__ = ["Bracket", "BracketEvaluation", "BracketTrigger", "SLTPManager"]
