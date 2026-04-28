"""Phase 5 closed-loop contracts (Learning + Evolution + Governance bridge).

These dataclasses move the closed loop forward (Build Compiler Spec §8):

    Execution → Dyon → Learning → Evolution → Governance approval → Deployment

Like ``core.contracts.events``, every type here is a frozen, slotted, hashable
record carrying ``ts_ns`` and structural metadata only — no callables, no IO,
no clocks. Equality is structural so replay parity (INV-15) is preserved.

INV-08: only typed records cross domain boundaries.
INV-11: no direct cross-engine method calls; only these records flow.
INV-15: all fields are deterministic primitives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.contracts.events import ExecutionStatus


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    """Execution-domain outcome forwarded to Dyon / Learning.

    Emitted by ``execution_engine.protections.feedback`` (EXEC-09) once an
    order reaches a terminal status.
    """

    ts_ns: int
    strategy_id: str
    symbol: str
    qty: float
    pnl: float
    status: ExecutionStatus
    venue: str = ""
    order_id: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategyStats:
    """Rolling performance summary per strategy.

    Emitted by ``learning_engine.lanes.patch_outcome_feedback`` (DYN-L02).
    """

    ts_ns: int
    strategy_id: str
    n_trades: int
    n_wins: int
    n_losses: int
    total_pnl: float
    mean_pnl: float
    win_rate: float


@dataclass(frozen=True, slots=True)
class LearningUpdate:
    """Parameter-level mutation proposal from Learning → Governance.

    Emitted by ``learning_engine.update_emitter`` (→ GOV-G18). This is a
    *parameter* update — non-structural — and is materialised as a
    ``SystemEvent(sub_kind=UPDATE_PROPOSED)`` on the bus.
    """

    ts_ns: int
    strategy_id: str
    parameter: str
    old_value: str
    new_value: str
    reason: str
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PatchProposal:
    """Structural mutation proposal from Evolution → Governance approval.

    Emitted by ``evolution_engine.intelligence_loops.mutation_proposer``
    (DYN-L01). Drives a row through ``evolution_engine.patch_pipeline``.
    """

    ts_ns: int
    patch_id: str
    source: str
    target_strategy: str
    touchpoints: tuple[str, ...]
    rationale: str
    meta: Mapping[str, str] = field(default_factory=dict)


__all__ = [
    "LearningUpdate",
    "PatchProposal",
    "StrategyStats",
    "TradeOutcome",
]
