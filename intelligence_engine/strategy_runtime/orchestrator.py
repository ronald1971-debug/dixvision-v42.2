"""Strategy orchestrator — Phase 3 / v2-B.

Activates strategies based on their lifecycle state, the current market
regime, and explicit governance approval. Pure dispatch: no IO, no
clocks. The orchestrator does **not** mutate ``StrategyStateMachine``
itself — only Governance writes lifecycle state. The orchestrator's
output is a *read-only* view: "of all strategies registered, which are
eligible to fire right now?"
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from intelligence_engine.strategy_runtime.regime_detector import MarketRegime
from intelligence_engine.strategy_runtime.state_machine import (
    StrategyState,
    StrategyStateMachine,
)


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    """One registered strategy in the orchestrator's view."""

    strategy_id: str
    allowed_regimes: frozenset[MarketRegime]
    min_state: StrategyState = StrategyState.CANARY

    def is_eligible(
        self, *, state: StrategyState, regime: MarketRegime
    ) -> bool:
        if state in (StrategyState.RETIRED, StrategyState.FAILED):
            return False
        if not _state_at_least(state, self.min_state):
            return False
        if not self.allowed_regimes:
            return True
        return regime in self.allowed_regimes


# Total ordering used for ``min_state`` comparisons.
# Strategy-level SHADOW was demolished by SHADOW-DEMOLITION-02; rank
# values stay stable so callers persisting ``min_state`` ordinals from
# previous releases still round-trip.
_STATE_RANK: dict[StrategyState, int] = {
    StrategyState.PROPOSED: 0,
    StrategyState.CANARY: 2,
    StrategyState.LIVE: 3,
    StrategyState.RETIRED: -1,
    StrategyState.FAILED: -1,
}


def _state_at_least(state: StrategyState, floor: StrategyState) -> bool:
    return _STATE_RANK[state] >= _STATE_RANK[floor]


class StrategyOrchestrator:
    """Read-only resolver of "which strategies fire under this regime?".

    Args:
        fsm: The lifecycle FSM (reader-only access here).
    """

    name: str = "strategy_orchestrator"
    spec_id: str = "IND-ORC-01"

    def __init__(self, fsm: StrategyStateMachine) -> None:
        self._fsm = fsm
        self._records: dict[str, StrategyRecord] = {}

    # -- registration ------------------------------------------------------

    def register(
        self,
        *,
        strategy_id: str,
        allowed_regimes: Iterable[MarketRegime] = (),
        min_state: StrategyState = StrategyState.CANARY,
    ) -> StrategyRecord:
        if not strategy_id:
            raise ValueError("strategy_id required")
        if strategy_id in self._records:
            raise ValueError(f"already registered: {strategy_id}")
        record = StrategyRecord(
            strategy_id=strategy_id,
            allowed_regimes=frozenset(allowed_regimes),
            min_state=min_state,
        )
        self._records[strategy_id] = record
        return record

    def deregister(self, strategy_id: str) -> None:
        self._records.pop(strategy_id, None)

    # -- queries -----------------------------------------------------------

    def get(self, strategy_id: str) -> StrategyRecord | None:
        return self._records.get(strategy_id)

    def eligible(self, regime: MarketRegime) -> tuple[str, ...]:
        out: list[str] = []
        for sid, rec in self._records.items():
            fsm_record = self._fsm.get(sid)
            if fsm_record is None:
                continue
            if rec.is_eligible(state=fsm_record.state, regime=regime):
                out.append(sid)
        return tuple(out)


__all__ = ["StrategyOrchestrator", "StrategyRecord"]


# A small helper for tests / debugging that don't want to expose the
# private rank dict directly.
@dataclass(frozen=True, slots=True)
class _StrategyView:  # pragma: no cover - reflection helper
    strategy_id: str
    state: StrategyState
    record: StrategyRecord = field(repr=False)
