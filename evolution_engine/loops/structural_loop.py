"""StructuralEvolutionLoop — P0-A structural-loop driver.

Wires the caller-supplied :class:`StrategyStats` source →
:class:`MutationProposer` → :class:`PatchPipelineOrchestrator` together
under the live :class:`LearningEvolutionFreezePolicy` supplied by the
runtime context.

Lifecycle on every :meth:`tick`:

1. Snapshot the live freeze policy via ``policy_supplier()``.
2. Drain the strategy stats source via ``stats_supplier()`` (always,
   even when frozen — the source must not back up indefinitely).
3. If frozen: return :class:`StructuralLoopTickResult` with
   ``frozen=True`` and ``drained_stats`` recorded. No proposals,
   no orchestrator runs, no FSM transitions.
4. If unfrozen: for each :class:`StrategyStats` row, call
   ``proposer.evaluate(stats)`` to obtain a tuple of
   :class:`PatchProposal`. For each proposal, build a
   :class:`StageEvidence` via the caller-supplied ``evidence_builder``
   and drive it through the full FSM via
   ``orchestrator.run(proposal=..., evidence=..., ts_ns=...)``.

The loop is the **single freeze-policy enforcement point** for this
chain. The inner :class:`MutationProposer` is constructed with
``freeze=None`` because the loop already gates every invocation.

Pure / deterministic: same inputs → same :class:`StructuralLoopTickResult`.
No clocks, no PRNG, no IO at this seam.

B27 / B28 / INV-71 authority symmetry: the loop lives in
``evolution_engine.*`` so it IS permitted to host the typed-event
construction sites (``PatchProposal`` via the proposer, ``SystemEvent``
via the orchestrator). Pinned by an AST test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.contracts.events import SystemEvent
from core.contracts.learning import PatchProposal, StrategyStats
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)
from evolution_engine.intelligence_loops.mutation_proposer import (
    MutationProposer,
)
from evolution_engine.patch_pipeline.orchestrator import (
    PatchPipelineOrchestrator,
    PatchPipelineRun,
    StageEvidence,
)

#: Type alias: zero-arg supplier returning the live freeze policy. Same
#: contract as :data:`learning_engine.loops.closed_loop.FreezePolicySupplier`.
FreezePolicySupplier = Callable[[], LearningEvolutionFreezePolicy]

#: Type alias: zero-arg supplier yielding the strategy stats batch to
#: drain this tick. The runtime closure typically pulls from a deque or
#: an in-memory rolling-stats projection; tests inject a fixed tuple.
StatsSupplier = Callable[[], tuple[StrategyStats, ...]]

#: Type alias: per-proposal evidence builder. Pure function. Production
#: callers wire a sandbox runner here; tests inject a static
#: :class:`StageEvidence` factory.
EvidenceBuilder = Callable[[PatchProposal], StageEvidence]

#: Type alias: per-proposal ``ts_ns`` derivation. Pure function so a
#: single caller-supplied ``ts_ns`` plus the proposal index can produce
#: deterministic, monotonically-increasing timestamps for every run.
RunTimestampDerivation = Callable[[int, int], int]


def _default_run_ts(base_ts_ns: int, proposal_index: int) -> int:
    """Spread per-proposal runs by 10ms so stage offsets never collide."""
    return base_ts_ns + (proposal_index * 10_000_000)


@dataclass(frozen=True, slots=True)
class StructuralLoopTickResult:
    """Frozen summary of one :meth:`StructuralEvolutionLoop.tick` call.

    Attributes:
        ts_ns: Caller-supplied tick timestamp.
        frozen: ``True`` iff the live freeze policy refused the tick.
        drained_stats: Tuple of :class:`StrategyStats` drained this tick
            (always populated — the supplier is invoked regardless of
            freeze state).
        proposals: Tuple of :class:`PatchProposal` emitted by the
            proposer this tick. Empty when frozen.
        runs: Tuple of :class:`PatchPipelineRun` summaries from
            orchestrator runs, one per proposal. Empty when frozen.
        emitted_events: Tuple of all :class:`SystemEvent` rows emitted
            across every run, in run order. Empty when frozen.
        policy_mode_name: Snapshot of ``policy.mode.name`` at tick time.
        operator_override: Snapshot of ``policy.operator_override``.
    """

    ts_ns: int
    frozen: bool
    drained_stats: tuple[StrategyStats, ...]
    proposals: tuple[PatchProposal, ...]
    runs: tuple[PatchPipelineRun, ...]
    emitted_events: tuple[SystemEvent, ...]
    policy_mode_name: str
    operator_override: bool


class StructuralEvolutionLoop:
    """Deterministic structural-loop driver (P0-A).

    Args:
        proposer: Bounded :class:`MutationProposer`. Must be constructed
            with ``freeze=None`` — the loop owns the gate.
        orchestrator: :class:`PatchPipelineOrchestrator` that drives
            each proposal through the full FSM.
        policy_supplier: Zero-arg callable returning the live
            :class:`LearningEvolutionFreezePolicy`. Invoked exactly
            once per :meth:`tick`.
        stats_supplier: Zero-arg callable returning the
            :class:`StrategyStats` batch to drain this tick.
        evidence_builder: Pure function mapping a :class:`PatchProposal`
            → the :class:`StageEvidence` for its orchestrator run.
        approve_reason: Reason string passed to the bridge on terminal
            CANARY→APPROVED transitions. Defaults to ``"canary_clean"``
            mirroring :meth:`PatchPipelineOrchestrator.run`.
        ts_ns_for_run: Pure function ``(base_ts_ns, proposal_index)`` →
            ``ts_ns`` for the orchestrator run. Defaults to a 10ms
            spread so per-stage offsets never collide between proposals
            on the same tick.

    Raises:
        ValueError: if ``proposer`` was constructed with a non-``None``
            freeze policy.
    """

    name: str = "structural_evolution_loop"
    spec_id: str = "P0-A"

    __slots__ = (
        "_proposer",
        "_orchestrator",
        "_policy_supplier",
        "_stats_supplier",
        "_evidence_builder",
        "_approve_reason",
        "_ts_ns_for_run",
    )

    def __init__(
        self,
        *,
        proposer: MutationProposer,
        orchestrator: PatchPipelineOrchestrator,
        policy_supplier: FreezePolicySupplier,
        stats_supplier: StatsSupplier,
        evidence_builder: EvidenceBuilder,
        approve_reason: str = "canary_clean",
        ts_ns_for_run: RunTimestampDerivation = _default_run_ts,
    ) -> None:
        if getattr(proposer, "_freeze", None) is not None:
            raise ValueError(
                "StructuralEvolutionLoop requires proposer.freeze=None "
                "(the loop is the single freeze gate)"
            )
        if not approve_reason:
            raise ValueError("approve_reason must be non-empty")
        self._proposer = proposer
        self._orchestrator = orchestrator
        self._policy_supplier = policy_supplier
        self._stats_supplier = stats_supplier
        self._evidence_builder = evidence_builder
        self._approve_reason = approve_reason
        self._ts_ns_for_run = ts_ns_for_run

    def tick(self, *, ts_ns: int) -> StructuralLoopTickResult:
        """Drive one structural-loop tick.

        Args:
            ts_ns: Base tick timestamp. Forwarded — via
                ``ts_ns_for_run`` — to every orchestrator run.
        """

        policy = self._policy_supplier()
        stats_batch = tuple(self._stats_supplier())
        if policy.is_frozen():
            return StructuralLoopTickResult(
                ts_ns=ts_ns,
                frozen=True,
                drained_stats=stats_batch,
                proposals=(),
                runs=(),
                emitted_events=(),
                policy_mode_name=policy.mode.name,
                operator_override=policy.operator_override,
            )
        all_proposals: list[PatchProposal] = []
        all_runs: list[PatchPipelineRun] = []
        all_events: list[SystemEvent] = []
        proposal_index = 0
        for stats in stats_batch:
            proposals = self._proposer.evaluate(stats)
            for proposal in proposals:
                evidence = self._evidence_builder(proposal)
                run_ts = self._ts_ns_for_run(ts_ns, proposal_index)
                run = self._orchestrator.run(
                    proposal=proposal,
                    evidence=evidence,
                    ts_ns=run_ts,
                    approve_reason=self._approve_reason,
                )
                all_proposals.append(proposal)
                all_runs.append(run)
                all_events.extend(run.events)
                proposal_index += 1
        return StructuralLoopTickResult(
            ts_ns=ts_ns,
            frozen=False,
            drained_stats=stats_batch,
            proposals=tuple(all_proposals),
            runs=tuple(all_runs),
            emitted_events=tuple(all_events),
            policy_mode_name=policy.mode.name,
            operator_override=policy.operator_override,
        )


__all__ = [
    "EvidenceBuilder",
    "FreezePolicySupplier",
    "RunTimestampDerivation",
    "StatsSupplier",
    "StructuralEvolutionLoop",
    "StructuralLoopTickResult",
]
