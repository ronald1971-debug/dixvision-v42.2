"""evolution_engine.loops — structural evolution loop drivers (P0-A).

Structural loop: caller-supplied :class:`StrategyStats` source →
:class:`MutationProposer` → :class:`PatchPipelineOrchestrator` under the
live :class:`LearningEvolutionFreezePolicy` supplied by the runtime.

When frozen, the loop drains the stats source and discards everything
without invoking the proposer or the orchestrator. When unfrozen, each
emitted :class:`PatchProposal` is driven through the full FSM
(PROPOSED → SANDBOX → STATIC_ANALYSIS → BACKTEST → SHADOW → CANARY →
APPROVED/REJECTED) and the run summary is returned to the caller.

The loop is the **single freeze-policy enforcement point** for this
chain. The inner :class:`MutationProposer` is constructed with
``freeze=None`` because the loop already gates every invocation.
"""

from evolution_engine.loops.structural_loop import (
    StructuralEvolutionLoop,
    StructuralLoopTickResult,
)

__all__ = [
    "StructuralEvolutionLoop",
    "StructuralLoopTickResult",
]
