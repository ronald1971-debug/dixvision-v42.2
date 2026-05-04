"""Strategy lifecycle FSM — Phase 3 (NEW REQUIRED ADDITION).

A separate, narrower lifecycle than ``PluginLifecycle`` (DISABLED /
ACTIVE): strategies move through promotion gates that mirror the
system-wide mode FSM (``SAFE → PAPER → CANARY → LIVE``) because every
promotion must be auditable and rollback-capable.

States::

    PROPOSED → CANARY → LIVE → RETIRED
        ↘──────────────── FAILED
                  ↘────── FAILED
                              ↘── FAILED
        ↘──────────────── RETIRED  (early withdrawal pre-LIVE)

Strategy-level ``SHADOW`` was demolished by SHADOW-DEMOLITION-02. The
signals-on-execution-off observation tier is now supplied at the
global layer via system mode ``PAPER``.

Terminal: ``RETIRED``, ``FAILED``.

The FSM is the **sole writer** of ``StrategyRecord.state``. Every
transition is appended to :attr:`StrategyRecord.history` for replay
determinism (INV-15) and so Governance can audit promotions.

This module is pure-Python, IO-free, and clock-free. The ``ts_ns`` is
caller-supplied so replay produces identical history.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class StrategyState(StrEnum):
    """All states a strategy can occupy.

    Strategy-level ``SHADOW`` was demolished by SHADOW-DEMOLITION-02:
    the lifecycle is now ``PROPOSED → CANARY → LIVE → RETIRED`` (plus
    ``FAILED`` from anywhere). The signals-on-execution-off observation
    tier no longer exists at the strategy layer — a strategy that has
    cleared ``PROPOSED`` is either firing into the conflict resolver
    (``CANARY``/``LIVE``) or it is not (``RETIRED``/``FAILED``). System
    mode (``PAPER``) supplies the equivalent observe-only behaviour at
    the global layer.
    """

    PROPOSED = "PROPOSED"
    CANARY = "CANARY"
    LIVE = "LIVE"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


# Legal forward edges. Anything outside this set raises
# :class:`StrategyLifecycleError`. ``RETIRED`` and ``FAILED`` are
# terminal — no further transitions.
LEGAL_STRATEGY_TRANSITIONS: dict[StrategyState, frozenset[StrategyState]] = {
    StrategyState.PROPOSED: frozenset(
        {StrategyState.CANARY, StrategyState.RETIRED, StrategyState.FAILED}
    ),
    StrategyState.CANARY: frozenset(
        {
            StrategyState.LIVE,
            StrategyState.PROPOSED,  # rollback for re-evaluation
            StrategyState.RETIRED,
            StrategyState.FAILED,
        }
    ),
    StrategyState.LIVE: frozenset(
        {
            StrategyState.CANARY,  # canary rollback under hazard
            StrategyState.RETIRED,
            StrategyState.FAILED,
        }
    ),
    StrategyState.RETIRED: frozenset(),  # terminal
    StrategyState.FAILED: frozenset(),  # terminal
}


class StrategyLifecycleError(ValueError):
    """Raised when a transition is not in :data:`LEGAL_STRATEGY_TRANSITIONS`."""


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One audit row in the strategy lifecycle history."""

    ts_ns: int
    prev: StrategyState
    new: StrategyState
    reason: str


@dataclass(slots=True)
class StrategyRecord:
    """Per-strategy lifecycle bookkeeping."""

    strategy_id: str
    state: StrategyState
    history: list[TransitionRecord] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.state in (StrategyState.RETIRED, StrategyState.FAILED)


class StrategyStateMachine:
    """Sole writer of :attr:`StrategyRecord.state`.

    Invariants:

    * No edge outside :data:`LEGAL_STRATEGY_TRANSITIONS` is ever taken.
    * Every transition is appended to ``record.history`` exactly once.
    * Terminal states never become non-terminal again.
    """

    name: str = "strategy_state_machine"
    spec_id: str = "IND-SLM-01"

    def __init__(self) -> None:
        self._records: dict[str, StrategyRecord] = {}

    # -- queries -----------------------------------------------------------

    def get(self, strategy_id: str) -> StrategyRecord | None:
        return self._records.get(strategy_id)

    def all_in(self, state: StrategyState) -> tuple[StrategyRecord, ...]:
        """All records currently in ``state``, ordered by registration."""
        return tuple(r for r in self._records.values() if r.state is state)

    # -- mutations ---------------------------------------------------------

    def propose(self, *, strategy_id: str, ts_ns: int) -> StrategyRecord:
        """Register a new strategy in the ``PROPOSED`` state."""
        if not strategy_id:
            raise ValueError("strategy_id required")
        if strategy_id in self._records:
            raise ValueError(f"already proposed: {strategy_id}")
        record = StrategyRecord(
            strategy_id=strategy_id,
            state=StrategyState.PROPOSED,
            history=[
                TransitionRecord(
                    ts_ns=ts_ns,
                    prev=StrategyState.PROPOSED,
                    new=StrategyState.PROPOSED,
                    reason="propose",
                )
            ],
        )
        self._records[strategy_id] = record
        return record

    def transition(
        self,
        *,
        strategy_id: str,
        new_state: StrategyState,
        ts_ns: int,
        reason: str,
    ) -> StrategyRecord:
        record = self._records.get(strategy_id)
        if record is None:
            raise KeyError(f"unknown strategy: {strategy_id}")
        legal = LEGAL_STRATEGY_TRANSITIONS[record.state]
        if new_state not in legal:
            raise StrategyLifecycleError(
                f"illegal transition for {strategy_id}: "
                f"{record.state} → {new_state}"
            )
        record.history.append(
            TransitionRecord(
                ts_ns=ts_ns,
                prev=record.state,
                new=new_state,
                reason=reason,
            )
        )
        record.state = new_state
        return record

    def transition_many(
        self,
        *,
        strategy_ids: Iterable[str],
        new_state: StrategyState,
        ts_ns: int,
        reason: str,
    ) -> tuple[StrategyRecord, ...]:
        return tuple(
            self.transition(
                strategy_id=sid,
                new_state=new_state,
                ts_ns=ts_ns,
                reason=reason,
            )
            for sid in strategy_ids
        )


__all__ = [
    "LEGAL_STRATEGY_TRANSITIONS",
    "StrategyLifecycleError",
    "StrategyRecord",
    "StrategyState",
    "StrategyStateMachine",
    "TransitionRecord",
]
