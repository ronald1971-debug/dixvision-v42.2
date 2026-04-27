"""Partial-fill resolution — Phase 2 / v2-C.

Decides whether an order with partial fills should be left open, cancelled,
or marked filled given a venue's reported state. Pure function: no IO.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from execution_engine.lifecycle.fill_handler import OrderFillState


class PartialFillResolution(StrEnum):
    LEAVE_OPEN = "LEAVE_OPEN"
    CANCEL_REMAINDER = "CANCEL_REMAINDER"
    MARK_FILLED = "MARK_FILLED"


@dataclass(frozen=True, slots=True)
class ResolutionContext:
    """Inputs to :meth:`PartialFillResolver.resolve`.

    ``min_fill_ratio`` is the threshold above which a partially-filled
    order is treated as effectively filled (e.g. ``0.99`` means 99% +
    of the target).
    """

    min_fill_ratio: float = 0.99
    cancel_after_ratio: float = 0.0


class PartialFillResolver:
    """Deterministic resolver for partial fills.

    Defaults align with paper-broker behaviour; per-domain overrides
    (memecoin: cancel any tail; copy-trading: leave open) belong in the
    caller's :class:`ResolutionContext`.
    """

    name: str = "partial_fill_resolver"
    spec_id: str = "EXEC-LC-03"

    def __init__(self, context: ResolutionContext | None = None) -> None:
        self._ctx = context or ResolutionContext()

    def resolve(
        self,
        state: OrderFillState,
        *,
        venue_says_done: bool = False,
    ) -> PartialFillResolution:
        if state.target_qty <= 0.0:
            return PartialFillResolution.MARK_FILLED
        ratio = state.filled_qty / state.target_qty
        if state.is_filled() or ratio >= self._ctx.min_fill_ratio:
            return PartialFillResolution.MARK_FILLED
        if venue_says_done:
            return PartialFillResolution.CANCEL_REMAINDER
        if (
            self._ctx.cancel_after_ratio > 0.0
            and ratio >= self._ctx.cancel_after_ratio
        ):
            return PartialFillResolution.CANCEL_REMAINDER
        return PartialFillResolution.LEAVE_OPEN


__all__ = ["PartialFillResolution", "PartialFillResolver", "ResolutionContext"]
