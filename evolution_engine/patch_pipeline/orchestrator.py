"""Patch pipeline orchestrator (BEHAVIOR-P5 / INV-66).

Wires the five Phase 4 patch-pipeline stages together so a single
:class:`PatchProposal` can be driven end-to-end through the canonical
FSM exactly the way Build Compiler Spec §1.1 / §8 prescribe::

    Evolution.MutationProposer
        → Governance.PatchApprovalBridge.receive_proposal()
        → SandboxStage         (PROPOSED → SANDBOX)
        → StaticAnalysisStage  (SANDBOX → STATIC_ANALYSIS)
        → BacktestStage        (STATIC_ANALYSIS → BACKTEST)
        → ShadowStage          (BACKTEST → SHADOW)
        → CanaryStage          (SHADOW → CANARY)
        → Governance.PatchApprovalBridge.approve()
                                (CANARY → APPROVED)

* Pure / deterministic. No clocks, no PRNG, no IO. The caller supplies
  every ``ts_ns`` and the stage evidence (touchpoints, findings, summary,
  shadow samples, canary stats).
* Each stage failure short-circuits to ``REJECTED`` with the failing
  stage's verdict as the rejection reason — Governance still drives the
  terminal transition (SAFE-69).
* Every transition emits a canonical ``SystemEvent`` row via
  :mod:`evolution_engine.patch_pipeline.events`. The dashboard's
  Strategy-Lifecycle widget (DASH-SLP-01) and the Indira / Dyon chat
  widgets read these to render reviewable cards.

Discipline (B1 / L2 / L3):
* Imports only ``core.contracts``, the offline pipeline stages, and
  ``governance_engine.services.patch_pipeline_bridge``. The bridge
  exposes a Protocol-driven runtime authority surface; the orchestrator
  holds the bridge instance but never reaches into governance internals.
* The orchestrator lives in ``evolution_engine`` (offline) per Build
  Compiler Spec §2 — it owns the *staging* of structural mutations.
  Governance still owns every state transition; the orchestrator merely
  funnels stage evidence and calls bridge methods in order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from core.contracts.events import SystemEvent
from core.contracts.learning import PatchProposal
from core.contracts.patch import (
    PatchApprovalBridgeProtocol,
    PatchApprovalDecision,
    PatchRecord,
    PatchStage,
    StageVerdict,
)
from evolution_engine.patch_pipeline.backtest import (
    BacktestStage,
    BacktestSummary,
)
from evolution_engine.patch_pipeline.canary import CanaryStage
from evolution_engine.patch_pipeline.events import (
    decision_as_system_event,
    proposal_as_system_event,
    verdict_as_system_event,
)
from evolution_engine.patch_pipeline.sandbox import SandboxStage
from evolution_engine.patch_pipeline.shadow import ShadowStage
from evolution_engine.patch_pipeline.static_analysis import (
    StaticAnalysisFinding,
    StaticAnalysisStage,
)

# Stable per-stage timestamp offsets so a single caller-supplied base
# ``ts_ns`` produces deterministic, monotonically-increasing event
# timestamps for every transition. Replays are byte-identical (INV-15).
_STAGE_TS_OFFSETS: dict[PatchStage, int] = {
    PatchStage.PROPOSED: 0,
    PatchStage.SANDBOX: 1,
    PatchStage.STATIC_ANALYSIS: 2,
    PatchStage.BACKTEST: 3,
    PatchStage.SHADOW: 4,
    PatchStage.CANARY: 5,
    PatchStage.APPROVED: 6,
    PatchStage.REJECTED: 6,
    PatchStage.ROLLED_BACK: 6,
}


# ---------------------------------------------------------------------------
# Stage evidence payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageEvidence:
    """Caller-supplied evidence for every gate the orchestrator runs.

    The orchestrator never *fabricates* gate inputs — they always come
    from the caller (Evolution → MutationProposer pipeline + a sandbox
    runner). This keeps the orchestrator pure and replay-deterministic.
    """

    sandbox_touchpoints: tuple[str, ...]
    static_findings: tuple[StaticAnalysisFinding, ...] = ()
    backtest_summary: BacktestSummary | None = None
    shadow_samples: int = 0
    shadow_matches: int = 0
    canary_orders: int = 0
    canary_rejects: int = 0
    canary_realised_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatchPipelineRun:
    """Frozen summary of a single orchestrator run.

    Attributes:
        record: Final :class:`PatchRecord` after the run terminated.
        decision: Governance's terminal decision (always populated —
            APPROVED on full pass, REJECTED on first stage fail).
        events: Tuple of canonical :class:`SystemEvent` rows emitted in
            order during the run. Replay-deterministic.
        stage_verdicts: Tuple of the per-stage :class:`StageVerdict`
            objects produced by the run, in execution order.
    """

    record: PatchRecord
    decision: PatchApprovalDecision
    events: tuple[SystemEvent, ...]
    stage_verdicts: tuple[StageVerdict, ...] = field(default=())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PatchPipelineOrchestrator:
    """Deterministic driver: PatchProposal → APPROVED or REJECTED.

    Spec ID: GOV-G18 (composition layer).

    The orchestrator is a *thin* glue layer: every gate is owned by an
    existing pipeline stage class and every state transition is owned by
    :class:`PatchApprovalBridge`. The orchestrator's only job is to call
    them in the canonical order and emit ledger events at every step so
    the offline calibrator and the operator dashboard have a complete,
    deterministic audit trail.
    """

    name: str = "patch_pipeline_orchestrator"
    spec_id: str = "GOV-G18"

    __slots__ = (
        "_bridge",
        "_sandbox",
        "_static",
        "_backtest",
        "_shadow",
        "_canary",
    )

    def __init__(
        self,
        *,
        bridge: PatchApprovalBridgeProtocol,
        sandbox: SandboxStage | None = None,
        static_analysis: StaticAnalysisStage | None = None,
        backtest: BacktestStage | None = None,
        shadow: ShadowStage | None = None,
        canary: CanaryStage | None = None,
    ) -> None:
        self._bridge: PatchApprovalBridgeProtocol = bridge
        self._sandbox = sandbox or SandboxStage()
        self._static = static_analysis or StaticAnalysisStage()
        self._backtest = backtest or BacktestStage()
        self._shadow = shadow or ShadowStage()
        self._canary = canary or CanaryStage()

    @property
    def bridge(self) -> PatchApprovalBridgeProtocol:
        return self._bridge

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        proposal: PatchProposal,
        evidence: StageEvidence,
        ts_ns: int,
        approve_reason: str = "canary_clean",
    ) -> PatchPipelineRun:
        """Drive *proposal* through the full pipeline FSM.

        Returns a :class:`PatchPipelineRun` summarising the terminal
        record + decision + every event the run emitted, in order.

        Raises:
            ValueError: if ``ts_ns`` is negative or ``approve_reason``
                is empty.
        """
        if ts_ns < 0:
            raise ValueError("run: ts_ns must be non-negative")
        if not approve_reason:
            raise ValueError("run: approve_reason must be non-empty")

        events: list[SystemEvent] = []
        verdicts: list[StageVerdict] = []

        # Stage 0 — register the proposal with Governance.
        record = self._bridge.receive_proposal(proposal)
        events.append(proposal_as_system_event(proposal))

        # Stage 1 — sandbox isolation.
        sandbox_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.SANDBOX]
        _, v_sandbox = self._sandbox.evaluate(
            ts_ns=sandbox_ts,
            touchpoints=evidence.sandbox_touchpoints,
        )
        verdicts.append(v_sandbox)
        if not v_sandbox.passed:
            return self._reject(
                record=record,
                events=events,
                verdicts=verdicts,
                failing=v_sandbox,
                ts_ns=ts_ns,
            )
        record = self._bridge.advance(
            patch_id=proposal.patch_id,
            new_stage=PatchStage.SANDBOX,
            ts_ns=sandbox_ts,
            verdict=v_sandbox,
        )
        events.append(
            verdict_as_system_event(
                patch_id=proposal.patch_id, verdict=v_sandbox
            )
        )

        # Stage 2 — static analysis.
        static_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.STATIC_ANALYSIS]
        v_static = self._static.evaluate(
            ts_ns=static_ts,
            findings=evidence.static_findings,
        )
        verdicts.append(v_static)
        if not v_static.passed:
            return self._reject(
                record=record,
                events=events,
                verdicts=verdicts,
                failing=v_static,
                ts_ns=ts_ns,
            )
        record = self._bridge.advance(
            patch_id=proposal.patch_id,
            new_stage=PatchStage.STATIC_ANALYSIS,
            ts_ns=static_ts,
            verdict=v_static,
        )
        events.append(
            verdict_as_system_event(
                patch_id=proposal.patch_id, verdict=v_static
            )
        )

        # Stage 3 — backtest.
        backtest_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.BACKTEST]
        if evidence.backtest_summary is None:
            v_backtest = StageVerdict(
                ts_ns=backtest_ts,
                stage=PatchStage.BACKTEST,
                passed=False,
                detail="no backtest summary",
                meta={"missing": "true"},
            )
        else:
            v_backtest = self._backtest.evaluate(
                ts_ns=backtest_ts,
                summary=evidence.backtest_summary,
            )
        verdicts.append(v_backtest)
        if not v_backtest.passed:
            return self._reject(
                record=record,
                events=events,
                verdicts=verdicts,
                failing=v_backtest,
                ts_ns=ts_ns,
            )
        record = self._bridge.advance(
            patch_id=proposal.patch_id,
            new_stage=PatchStage.BACKTEST,
            ts_ns=backtest_ts,
            verdict=v_backtest,
        )
        events.append(
            verdict_as_system_event(
                patch_id=proposal.patch_id, verdict=v_backtest
            )
        )

        # Stage 4 — shadow.
        shadow_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.SHADOW]
        _, v_shadow = self._shadow.evaluate(
            ts_ns=shadow_ts,
            samples=evidence.shadow_samples,
            matches=evidence.shadow_matches,
        )
        verdicts.append(v_shadow)
        if not v_shadow.passed:
            return self._reject(
                record=record,
                events=events,
                verdicts=verdicts,
                failing=v_shadow,
                ts_ns=ts_ns,
            )
        record = self._bridge.advance(
            patch_id=proposal.patch_id,
            new_stage=PatchStage.SHADOW,
            ts_ns=shadow_ts,
            verdict=v_shadow,
        )
        events.append(
            verdict_as_system_event(
                patch_id=proposal.patch_id, verdict=v_shadow
            )
        )

        # Stage 5 — canary.
        canary_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.CANARY]
        _, v_canary = self._canary.evaluate(
            ts_ns=canary_ts,
            orders=evidence.canary_orders,
            rejects=evidence.canary_rejects,
            realised_pnl=evidence.canary_realised_pnl,
        )
        verdicts.append(v_canary)
        if not v_canary.passed:
            return self._reject(
                record=record,
                events=events,
                verdicts=verdicts,
                failing=v_canary,
                ts_ns=ts_ns,
            )
        record = self._bridge.advance(
            patch_id=proposal.patch_id,
            new_stage=PatchStage.CANARY,
            ts_ns=canary_ts,
            verdict=v_canary,
        )
        events.append(
            verdict_as_system_event(
                patch_id=proposal.patch_id, verdict=v_canary
            )
        )

        # Terminal — Governance drives CANARY → APPROVED.
        approve_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.APPROVED]
        decision = self._bridge.approve(
            patch_id=proposal.patch_id,
            ts_ns=approve_ts,
            reason=approve_reason,
        )
        events.append(decision_as_system_event(decision))
        record = self._bridge.pipeline.get(proposal.patch_id)
        return PatchPipelineRun(
            record=record,
            decision=decision,
            events=tuple(events),
            stage_verdicts=tuple(verdicts),
        )

    # ------------------------------------------------------------------
    def _reject(
        self,
        *,
        record: PatchRecord,
        events: list[SystemEvent],
        verdicts: Sequence[StageVerdict],
        failing: StageVerdict,
        ts_ns: int,
    ) -> PatchPipelineRun:
        """Short-circuit the run with a Governance-driven REJECTED."""
        events.append(
            verdict_as_system_event(
                patch_id=record.patch_id, verdict=failing
            )
        )
        reject_ts = ts_ns + _STAGE_TS_OFFSETS[PatchStage.REJECTED]
        reason = f"{failing.stage.value.lower()}_failed:{failing.detail}"
        decision = self._bridge.reject(
            patch_id=record.patch_id,
            ts_ns=reject_ts,
            reason=reason,
        )
        events.append(decision_as_system_event(decision))
        record = self._bridge.pipeline.get(record.patch_id)
        return PatchPipelineRun(
            record=record,
            decision=decision,
            events=tuple(events),
            stage_verdicts=tuple(verdicts),
        )


__all__ = [
    "PatchPipelineOrchestrator",
    "PatchPipelineRun",
    "StageEvidence",
]
