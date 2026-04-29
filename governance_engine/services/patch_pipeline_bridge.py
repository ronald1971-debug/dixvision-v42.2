"""PatchApprovalBridge — GOV-G18.

The single authority bridge between Evolution's :class:`PatchProposal`
records and the patch pipeline FSM. Closes the loop:

    Evolution → Governance approval bridge → PatchPipeline → Deployment

Per Build Compiler Spec §1.1, only Governance is allowed to drive patch
state transitions to APPROVED / REJECTED / ROLLED_BACK. This bridge is the
sole entrypoint that enforces that rule.

Deterministic. No clocks, no IO. Caller supplies all ``ts_ns``.
"""

from __future__ import annotations

from core.contracts.learning import PatchProposal
from core.contracts.patch import (
    PatchApprovalDecision,
    PatchPipelineError,
    PatchPipelineProtocol,
    PatchRecord,
    PatchStage,
    StageVerdict,
)


class PatchApprovalBridge:
    """Deterministic bridge: PatchProposal → PatchPipeline transitions."""

    name: str = "patch_approval_bridge"
    spec_id: str = "GOV-G18"

    __slots__ = ("_pipeline", "_decisions")

    def __init__(self, *, pipeline: PatchPipelineProtocol) -> None:
        # The concrete PatchPipeline lives in evolution_engine (offline).
        # Governance (runtime) drives transitions via the contract
        # Protocol so authority_lint L3 stays clean.
        self._pipeline: PatchPipelineProtocol = pipeline
        self._decisions: list[PatchApprovalDecision] = []

    @property
    def pipeline(self) -> PatchPipelineProtocol:
        return self._pipeline

    @property
    def decisions(self) -> tuple[PatchApprovalDecision, ...]:
        return tuple(self._decisions)

    # ------------------------------------------------------------------
    def receive_proposal(self, proposal: PatchProposal) -> PatchRecord:
        """Register an Evolution proposal as a new ``PatchRecord``."""
        if not proposal.patch_id:
            raise ValueError("proposal.patch_id must be non-empty")
        return self._pipeline.propose(
            patch_id=proposal.patch_id, ts_ns=proposal.ts_ns
        )

    def advance(
        self,
        *,
        patch_id: str,
        new_stage: PatchStage,
        ts_ns: int,
        verdict: StageVerdict | None = None,
        reason: str = "",
    ) -> PatchRecord:
        """Move the patch through one pipeline stage with optional verdict."""
        if verdict is not None:
            self._pipeline.record_verdict(patch_id=patch_id, verdict=verdict)
        return self._pipeline.transition(
            patch_id=patch_id,
            new_stage=new_stage,
            ts_ns=ts_ns,
            reason=reason or new_stage.value,
        )

    def approve(
        self,
        *,
        patch_id: str,
        ts_ns: int,
        reason: str = "approved",
    ) -> PatchApprovalDecision:
        """Final approval: only legal from CANARY → APPROVED."""
        rec = self._pipeline.get(patch_id)
        if rec.stage is not PatchStage.CANARY:
            raise PatchPipelineError(
                f"approve requires CANARY stage; "
                f"patch {patch_id!r} is in {rec.stage.value}"
            )
        rec = self._pipeline.transition(
            patch_id=patch_id,
            new_stage=PatchStage.APPROVED,
            ts_ns=ts_ns,
            reason=reason,
        )
        decision = PatchApprovalDecision(
            ts_ns=ts_ns,
            patch_id=patch_id,
            decision="APPROVED",
            reason=reason,
            final_stage=rec.stage,
            meta={},
        )
        self._decisions.append(decision)
        return decision

    def reject(
        self,
        *,
        patch_id: str,
        ts_ns: int,
        reason: str,
    ) -> PatchApprovalDecision:
        """Reject the patch from any non-terminal stage."""
        if not reason:
            raise ValueError("reason must be non-empty")
        rec = self._pipeline.transition(
            patch_id=patch_id,
            new_stage=PatchStage.REJECTED,
            ts_ns=ts_ns,
            reason=reason,
        )
        decision = PatchApprovalDecision(
            ts_ns=ts_ns,
            patch_id=patch_id,
            decision="REJECTED",
            reason=reason,
            final_stage=rec.stage,
            meta={},
        )
        self._decisions.append(decision)
        return decision

    def rollback(
        self,
        *,
        patch_id: str,
        ts_ns: int,
        reason: str,
    ) -> PatchApprovalDecision:
        """Roll back an APPROVED or CANARY patch."""
        if not reason:
            raise ValueError("reason must be non-empty")
        rec = self._pipeline.transition(
            patch_id=patch_id,
            new_stage=PatchStage.ROLLED_BACK,
            ts_ns=ts_ns,
            reason=reason,
        )
        decision = PatchApprovalDecision(
            ts_ns=ts_ns,
            patch_id=patch_id,
            decision="ROLLED_BACK",
            reason=reason,
            final_stage=rec.stage,
            meta={},
        )
        self._decisions.append(decision)
        return decision


__all__ = ["PatchApprovalBridge", "PatchApprovalDecision"]
