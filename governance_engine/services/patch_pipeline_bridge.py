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
from governance_engine.gates import (
    DEFAULT_QUANTITATIVE_THRESHOLDS,
    QuantitativeEvaluator,
    QuantitativeMetrics,
    RuleGraphPatchEvaluator,
    build_patch_facts,
)


class PatchApprovalBridge:
    """Deterministic bridge: PatchProposal → PatchPipeline transitions."""

    name: str = "patch_approval_bridge"
    spec_id: str = "GOV-G18"

    __slots__ = (
        "_pipeline",
        "_decisions",
        "_quantitative_evaluator",
        "_rulegraph_evaluator",
    )

    def __init__(
        self,
        *,
        pipeline: PatchPipelineProtocol,
        quantitative_evaluator: QuantitativeEvaluator | None = None,
        rulegraph_evaluator: RuleGraphPatchEvaluator | None = None,
    ) -> None:
        # The concrete PatchPipeline lives in evolution_engine (offline).
        # Governance (runtime) drives transitions via the contract
        # Protocol so authority_lint L3 stays clean.
        self._pipeline: PatchPipelineProtocol = pipeline
        self._decisions: list[PatchApprovalDecision] = []
        # P0-B promotion gates. Both are optional so existing callers
        # (and pre-P0-B tests) continue to work; supplying either one
        # enables the corresponding pre-approval check on the
        # CANARY → APPROVED edge.
        self._quantitative_evaluator = quantitative_evaluator
        self._rulegraph_evaluator = rulegraph_evaluator

    @property
    def pipeline(self) -> PatchPipelineProtocol:
        return self._pipeline

    @property
    def decisions(self) -> tuple[PatchApprovalDecision, ...]:
        return tuple(self._decisions)

    @property
    def quantitative_evaluator(self) -> QuantitativeEvaluator | None:
        return self._quantitative_evaluator

    @property
    def rulegraph_evaluator(self) -> RuleGraphPatchEvaluator | None:
        return self._rulegraph_evaluator

    # ------------------------------------------------------------------
    def receive_proposal(self, proposal: PatchProposal) -> PatchRecord:
        """Register an Evolution proposal as a new ``PatchRecord``."""
        if not proposal.patch_id:
            raise ValueError("proposal.patch_id must be non-empty")
        return self._pipeline.propose(patch_id=proposal.patch_id, ts_ns=proposal.ts_ns)

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
        proposal: PatchProposal | None = None,
        metrics: QuantitativeMetrics | None = None,
    ) -> PatchApprovalDecision:
        """Final approval: only legal from CANARY → APPROVED.

        When either a :class:`QuantitativeEvaluator` or a
        :class:`RuleGraphPatchEvaluator` is configured on the bridge,
        the corresponding gate runs on the ``CANARY → APPROVED`` edge
        *before* any operator-supplied verdict is honoured. A failing
        gate transitions the patch to ``REJECTED`` and returns a
        ``REJECTED`` :class:`PatchApprovalDecision` whose ``meta``
        carries the structured rejection codes.
        """

        rec = self._pipeline.get(patch_id)
        if rec.stage is not PatchStage.CANARY:
            raise PatchPipelineError(
                f"approve requires CANARY stage; patch {patch_id!r} is in {rec.stage.value}"
            )

        gate_meta = self._run_promotion_gates(
            patch_id=patch_id,
            proposal=proposal,
            metrics=metrics,
        )
        if gate_meta is not None:
            # Gate failure — reject the patch and surface the codes
            # on the decision's ``meta`` mapping.
            gate_reason = gate_meta.get("gate_detail", "promotion gate rejected")
            rec = self._pipeline.transition(
                patch_id=patch_id,
                new_stage=PatchStage.REJECTED,
                ts_ns=ts_ns,
                reason=gate_reason,
            )
            decision = PatchApprovalDecision(
                ts_ns=ts_ns,
                patch_id=patch_id,
                decision="REJECTED",
                reason=gate_reason,
                final_stage=rec.stage,
                meta=gate_meta,
            )
            self._decisions.append(decision)
            return decision

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

    # ------------------------------------------------------------------
    def _run_promotion_gates(
        self,
        *,
        patch_id: str,
        proposal: PatchProposal | None,
        metrics: QuantitativeMetrics | None,
    ) -> dict[str, str] | None:
        """Run configured promotion gates; return rejection meta or None.

        Returns ``None`` when no gate is configured *or* every
        configured gate passes. Returns a sorted-key ``dict`` of
        rejection metadata when at least one gate blocks promotion.
        Pure / deterministic — INV-15.
        """

        quant = self._quantitative_evaluator
        rulegraph = self._rulegraph_evaluator
        if quant is None and rulegraph is None:
            return None

        if metrics is None:
            raise ValueError(
                f"approve requires metrics when promotion gates are configured; patch {patch_id!r}"
            )

        rejection_codes: list[str] = []
        details: list[str] = []

        if quant is not None:
            qv = quant.evaluate(metrics)
            if not qv.passed:
                rejection_codes.extend(qv.rejection_codes)
                details.append(f"quantitative={qv.kind.value} ({qv.detail})")

        if rulegraph is not None:
            if proposal is None:
                raise ValueError(
                    f"approve requires proposal when rulegraph_evaluator "
                    f"is configured; patch {patch_id!r}"
                )
            thresholds = quant.thresholds if quant is not None else DEFAULT_QUANTITATIVE_THRESHOLDS
            facts = build_patch_facts(
                proposal=proposal,
                metrics=metrics,
                sharpe_ratio_min=thresholds.sharpe_ratio_min,
                max_drawdown_max=thresholds.max_drawdown_max,
                samples_min=thresholds.samples_min,
                is_oos_divergence_max_sigma=(thresholds.is_oos_divergence_max_sigma),
            )
            rgv = rulegraph.evaluate(facts)
            if not rgv.passed:
                rejection_codes.extend(rgv.blocking_rule_ids)
                details.append(f"rulegraph={rgv.kind.value} ({rgv.detail})")

        if not rejection_codes:
            return None

        # Sorted, dedup'd, byte-stable rejection-code projection.
        sorted_codes = sorted(set(rejection_codes))
        return {
            "gate_detail": "; ".join(details),
            "gate_rejection_codes": ",".join(sorted_codes),
        }

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
