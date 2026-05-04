"""Strategy Lifecycle Panel — Phase 6 IMMUTABLE WIDGET 4 (DASH-SLP-01).

Renders the strategy lifecycle FSM (PROPOSED → CANARY → LIVE →
RETIRED, plus FAILED from anywhere; strategy-level SHADOW was
demolished by SHADOW-DEMOLITION-02) by reading the canonical
:class:`StrategyStateMachine` (IND-SLM-01).

This widget is *purely* a read projection. Lifecycle transitions are
made elsewhere — by the Indira orchestrator under Governance approval.
The dashboard only displays the current state per strategy and its
recorded transition history. (INV-08, INV-37)
"""

from __future__ import annotations

from dataclasses import dataclass

from intelligence_engine.strategy_runtime.state_machine import (
    StrategyRecord,
    StrategyState,
    StrategyStateMachine,
)

# All non-terminal + terminal states in the canonical FSM order, used
# by the UI to lay out columns deterministically.
LIFECYCLE_COLUMNS: tuple[StrategyState, ...] = (
    StrategyState.PROPOSED,
    StrategyState.CANARY,
    StrategyState.LIVE,
    StrategyState.RETIRED,
    StrategyState.FAILED,
)


@dataclass(frozen=True, slots=True)
class StrategyTransitionRow:
    """One row of transition history for a strategy."""

    ts_ns: int
    prev: str
    new: str
    reason: str


@dataclass(frozen=True, slots=True)
class StrategyRow:
    """Renderable read-projection row per strategy."""

    strategy_id: str
    state: str
    is_terminal: bool
    history: tuple[StrategyTransitionRow, ...]


class StrategyLifecyclePanel:
    """DASH-SLP-01 — Strategy Lifecycle Panel widget backend."""

    name: str = "strategy_lifecycle_panel"
    spec_id: str = "DASH-SLP-01"

    def __init__(self, *, fsm: StrategyStateMachine) -> None:
        self._fsm = fsm

    def by_state(self) -> dict[str, tuple[StrategyRow, ...]]:
        """Return rows grouped by state, in canonical FSM column order."""
        out: dict[str, tuple[StrategyRow, ...]] = {}
        for state in LIFECYCLE_COLUMNS:
            records = self._fsm.all_in(state)
            out[state.value] = tuple(self._row_for(r) for r in records)
        return out

    def all_rows(self) -> tuple[StrategyRow, ...]:
        """Flat list of all known strategies, FSM-ordered then by id."""
        rows: list[StrategyRow] = []
        for state in LIFECYCLE_COLUMNS:
            for record in self._fsm.all_in(state):
                rows.append(self._row_for(record))
        return tuple(rows)

    @staticmethod
    def _row_for(record: StrategyRecord) -> StrategyRow:
        return StrategyRow(
            strategy_id=record.strategy_id,
            state=record.state.value,
            is_terminal=record.is_terminal(),
            history=tuple(
                StrategyTransitionRow(
                    ts_ns=h.ts_ns,
                    prev=h.prev.value,
                    new=h.new.value,
                    reason=h.reason,
                )
                for h in record.history
            ),
        )
