"""Rollback planner — deterministic plan for unwinding an APPROVED patch.

The planner does not perform any IO. It produces an ordered tuple of
revert steps that the caller (governance) can apply in sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RollbackStep:
    order: int
    target: str
    action: str
    detail: str = ""


class RollbackPlanner:
    """GOV-G18-S6."""

    name: str = "rollback_planner"
    spec_id: str = "GOV-G18-S6"

    __slots__ = ()

    def plan(
        self,
        *,
        patch_id: str,
        touchpoints: Sequence[str],
    ) -> tuple[RollbackStep, ...]:
        if not patch_id:
            raise ValueError("patch_id must be non-empty")
        # Reverse order so newer touchpoints unwind first.
        ordered = list(touchpoints)[::-1]
        steps = tuple(
            RollbackStep(
                order=i,
                target=tp,
                action="revert",
                detail=f"revert {tp} for patch {patch_id}",
            )
            for i, tp in enumerate(ordered)
        )
        return steps


__all__ = ["RollbackPlanner", "RollbackStep"]
