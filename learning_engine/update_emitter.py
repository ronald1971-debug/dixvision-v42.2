"""UpdateEmitter — Learning → Governance bridge (→ GOV-G18).

Materialises non-structural parameter updates as ``SystemEvent`` records
with ``sub_kind=UPDATE_PROPOSED``. Emission is deterministic: same input,
same output. No clocks, no IO, no randomness.

HARDEN-04 / INV-70: an optional :class:`LearningEvolutionFreezePolicy`
gates :meth:`emit`. When the policy is frozen (default in every mode
except ``LIVE`` with an explicit operator override), :meth:`emit`
raises :class:`LearningEvolutionFrozenError` instead of producing a
``SystemEvent``. Existing offline tests that construct an emitter
without a policy continue to behave deterministically.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.contracts.events import SystemEvent, SystemEventKind
from core.contracts.learning import LearningUpdate
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
    assert_unfrozen,
)


class UpdateEmitter:
    """Translates :class:`LearningUpdate` into bus events."""

    name: str = "update_emitter"
    spec_id: str = "GOV-G18"

    __slots__ = ("_source", "_freeze")

    def __init__(
        self,
        *,
        source: str = "learning",
        freeze: LearningEvolutionFreezePolicy | None = None,
    ) -> None:
        if not source:
            raise ValueError("source must be non-empty")
        self._source = source
        self._freeze = freeze

    def emit(self, update: LearningUpdate) -> SystemEvent:
        assert_unfrozen(self._freeze, action="emit_update")
        payload: dict[str, str] = {
            "strategy_id": update.strategy_id,
            "parameter": update.parameter,
            "old_value": update.old_value,
            "new_value": update.new_value,
            "reason": update.reason,
        }
        return SystemEvent(
            ts_ns=update.ts_ns,
            sub_kind=SystemEventKind.UPDATE_PROPOSED,
            source=self._source,
            payload=payload,
            meta=dict(update.meta),
        )

    def emit_many(
        self, updates: tuple[LearningUpdate, ...]
    ) -> tuple[SystemEvent, ...]:
        return tuple(self.emit(u) for u in updates)

    @staticmethod
    def propose(
        *,
        ts_ns: int,
        strategy_id: str,
        parameter: str,
        old_value: str,
        new_value: str,
        reason: str,
        meta: Mapping[str, str] | None = None,
    ) -> LearningUpdate:
        """Builder helper that validates required fields."""
        if not strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not parameter:
            raise ValueError("parameter must be non-empty")
        if not reason:
            raise ValueError("reason must be non-empty")
        return LearningUpdate(
            ts_ns=ts_ns,
            strategy_id=strategy_id,
            parameter=parameter,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            meta=dict(meta or {}),
        )


__all__ = ["UpdateEmitter"]
