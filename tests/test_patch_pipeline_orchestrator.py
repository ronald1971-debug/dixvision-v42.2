"""Tests for the BEHAVIOR-P5 patch pipeline orchestrator + ledger surface."""

from __future__ import annotations

import pytest

from core.contracts.events import EventKind, SystemEvent, SystemEventKind
from core.contracts.learning import PatchProposal
from core.contracts.patch import (
    PatchApprovalDecision,
    PatchPipelineError,
    PatchStage,
    StageVerdict,
)
from evolution_engine.patch_pipeline.backtest import BacktestSummary
from evolution_engine.patch_pipeline.events import (
    PATCH_EVENT_SOURCE_DECISION,
    PATCH_EVENT_SOURCE_PROPOSAL,
    PATCH_EVENT_SOURCE_VERDICT,
    decision_as_system_event,
    decision_from_system_event,
    proposal_as_system_event,
    proposal_from_system_event,
    verdict_as_system_event,
    verdict_from_system_event,
)
from evolution_engine.patch_pipeline.orchestrator import (
    PatchPipelineOrchestrator,
    StageEvidence,
)
from evolution_engine.patch_pipeline.pipeline import PatchPipeline
from evolution_engine.patch_pipeline.static_analysis import (
    FindingSeverity,
    StaticAnalysisFinding,
)
from governance_engine.services.patch_pipeline_bridge import (
    PatchApprovalBridge,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _proposal(patch_id: str = "p1", ts_ns: int = 1000) -> PatchProposal:
    return PatchProposal(
        ts_ns=ts_ns,
        patch_id=patch_id,
        source="evolution.mutation_proposer",
        target_strategy="strategy_alpha",
        touchpoints=("indira_engine.weights",),
        rationale="win_rate_below_floor",
        meta={"reason": "win_rate_below_floor", "value": "0.42"},
    )


def _clean_evidence() -> StageEvidence:
    return StageEvidence(
        sandbox_touchpoints=("indira_engine.weights",),
        static_findings=(
            StaticAnalysisFinding(
                rule="ruff/E501",
                severity=FindingSeverity.INFO,
                location="indira_engine/weights.py:42",
                detail="line too long",
            ),
        ),
        backtest_summary=BacktestSummary(
            runs=10, pnl=12.5, sharpe=1.5, max_drawdown=0.05
        ),
        shadow_samples=120,
        shadow_matches=119,
        canary_orders=20,
        canary_rejects=1,
        canary_realised_pnl=0.5,
    )


def _make_orchestrator() -> tuple[
    PatchPipelineOrchestrator, PatchApprovalBridge
]:
    pipeline = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pipeline)
    return PatchPipelineOrchestrator(bridge=bridge), bridge


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_run_happy_path_drives_proposal_to_approved():
    orch, bridge = _make_orchestrator()
    proposal = _proposal()
    run = orch.run(proposal=proposal, evidence=_clean_evidence(), ts_ns=1000)

    assert run.decision.decision == "APPROVED"
    assert run.record.stage is PatchStage.APPROVED
    assert tuple(v.stage for v in run.stage_verdicts) == (
        PatchStage.SANDBOX,
        PatchStage.STATIC_ANALYSIS,
        PatchStage.BACKTEST,
        PatchStage.SHADOW,
        PatchStage.CANARY,
    )
    assert all(v.passed for v in run.stage_verdicts)
    assert tuple(e.sub_kind for e in run.events) == (
        SystemEventKind.PATCH_PROPOSED,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_DECISION,
    )
    assert all(e.kind is EventKind.SYSTEM for e in run.events)
    assert [e.ts_ns for e in run.events] == sorted(e.ts_ns for e in run.events)
    assert bridge.decisions[-1] is run.decision


# ---------------------------------------------------------------------------
# per-stage rejection
# ---------------------------------------------------------------------------


def test_sandbox_failure_short_circuits_to_rejected():
    orch, _ = _make_orchestrator()
    evidence = _clean_evidence()
    bad = StageEvidence(
        sandbox_touchpoints=evidence.sandbox_touchpoints + ("subprocess.run",),
        static_findings=evidence.static_findings,
        backtest_summary=evidence.backtest_summary,
        shadow_samples=evidence.shadow_samples,
        shadow_matches=evidence.shadow_matches,
        canary_orders=evidence.canary_orders,
        canary_rejects=evidence.canary_rejects,
        canary_realised_pnl=evidence.canary_realised_pnl,
    )
    run = orch.run(proposal=_proposal(), evidence=bad, ts_ns=2000)

    assert run.decision.decision == "REJECTED"
    assert run.record.stage is PatchStage.REJECTED
    assert run.decision.reason.startswith("sandbox_failed:")
    assert len(run.stage_verdicts) == 1
    assert run.stage_verdicts[0].stage is PatchStage.SANDBOX
    assert not run.stage_verdicts[0].passed
    assert tuple(e.sub_kind for e in run.events) == (
        SystemEventKind.PATCH_PROPOSED,
        SystemEventKind.PATCH_STAGE_VERDICT,
        SystemEventKind.PATCH_DECISION,
    )


def test_static_analysis_failure_rejects_at_static_stage():
    orch, _ = _make_orchestrator()
    base = _clean_evidence()
    bad_findings = base.static_findings + (
        StaticAnalysisFinding(
            rule="authority_lint/L1",
            severity=FindingSeverity.ERROR,
            location="indira_engine/weights.py:1",
            detail="cross-engine import",
        ),
    )
    bad = StageEvidence(
        sandbox_touchpoints=base.sandbox_touchpoints,
        static_findings=bad_findings,
        backtest_summary=base.backtest_summary,
        shadow_samples=base.shadow_samples,
        shadow_matches=base.shadow_matches,
        canary_orders=base.canary_orders,
        canary_rejects=base.canary_rejects,
        canary_realised_pnl=base.canary_realised_pnl,
    )
    run = orch.run(proposal=_proposal(), evidence=bad, ts_ns=3000)

    assert run.decision.decision == "REJECTED"
    assert run.decision.reason.startswith("static_analysis_failed:")
    assert run.stage_verdicts[-1].stage is PatchStage.STATIC_ANALYSIS
    assert not run.stage_verdicts[-1].passed


def test_backtest_missing_summary_rejects():
    orch, _ = _make_orchestrator()
    base = _clean_evidence()
    bad = StageEvidence(
        sandbox_touchpoints=base.sandbox_touchpoints,
        static_findings=base.static_findings,
        backtest_summary=None,
        shadow_samples=base.shadow_samples,
        shadow_matches=base.shadow_matches,
        canary_orders=base.canary_orders,
        canary_rejects=base.canary_rejects,
        canary_realised_pnl=base.canary_realised_pnl,
    )
    run = orch.run(proposal=_proposal(), evidence=bad, ts_ns=4000)

    assert run.decision.decision == "REJECTED"
    assert run.decision.reason.startswith("backtest_failed:")
    assert run.stage_verdicts[-1].detail == "no backtest summary"


def test_shadow_under_min_samples_rejects():
    orch, _ = _make_orchestrator()
    base = _clean_evidence()
    bad = StageEvidence(
        sandbox_touchpoints=base.sandbox_touchpoints,
        static_findings=base.static_findings,
        backtest_summary=base.backtest_summary,
        shadow_samples=10,
        shadow_matches=10,
        canary_orders=base.canary_orders,
        canary_rejects=base.canary_rejects,
        canary_realised_pnl=base.canary_realised_pnl,
    )
    run = orch.run(proposal=_proposal(), evidence=bad, ts_ns=5000)

    assert run.decision.decision == "REJECTED"
    assert run.decision.reason.startswith("shadow_failed:")


def test_canary_high_error_rate_rejects():
    orch, _ = _make_orchestrator()
    base = _clean_evidence()
    bad = StageEvidence(
        sandbox_touchpoints=base.sandbox_touchpoints,
        static_findings=base.static_findings,
        backtest_summary=base.backtest_summary,
        shadow_samples=base.shadow_samples,
        shadow_matches=base.shadow_matches,
        canary_orders=20,
        canary_rejects=15,
        canary_realised_pnl=-1.0,
    )
    run = orch.run(proposal=_proposal(), evidence=bad, ts_ns=6000)

    assert run.decision.decision == "REJECTED"
    assert run.decision.reason.startswith("canary_failed:")


# ---------------------------------------------------------------------------
# determinism + replay parity
# ---------------------------------------------------------------------------


def test_run_replay_byte_identical_events():
    def _go() -> tuple[SystemEvent, ...]:
        orch, _ = _make_orchestrator()
        run = orch.run(
            proposal=_proposal(),
            evidence=_clean_evidence(),
            ts_ns=7000,
        )
        return run.events

    a = _go()
    b = _go()
    assert a == b
    payloads_a = tuple(e.payload for e in a)
    payloads_b = tuple(e.payload for e in b)
    assert payloads_a == payloads_b


def test_run_event_ts_ns_derives_from_base_not_proposal_ts_ns():
    """INV-66 regression: ``SystemEvent.ts_ns`` for every emission must be
    derived from the caller-supplied base ``ts_ns`` plus the per-stage
    offset, NOT from ``proposal.ts_ns``. Without this guarantee, a proposal
    whose creation timestamp is *after* the run base timestamp produces a
    non-monotonic event stream that breaks replay-deterministic ordering
    (the PROPOSED row would carry a ts_ns greater than every subsequent
    stage row). The original ``proposal.ts_ns`` must still survive inside
    the JSON body so :func:`proposal_from_system_event` round-trips.
    """
    orch, _ = _make_orchestrator()
    base_ts = 1_000
    proposal = _proposal(ts_ns=9_999)  # deliberately > base_ts
    run = orch.run(
        proposal=proposal, evidence=_clean_evidence(), ts_ns=base_ts
    )

    # Outer event timestamps must be base_ts + offset (0..6), not 9_999.
    assert [e.ts_ns for e in run.events] == [
        base_ts,
        base_ts + 1,
        base_ts + 2,
        base_ts + 3,
        base_ts + 4,
        base_ts + 5,
        base_ts + 6,
    ]
    # Strictly monotonic (the load-bearing INV-66 guarantee).
    assert [e.ts_ns for e in run.events] == sorted(
        e.ts_ns for e in run.events
    )
    # Body still preserves the original proposal.ts_ns for replay parity.
    proposed = run.events[0]
    assert proposed.sub_kind is SystemEventKind.PATCH_PROPOSED
    recovered = proposal_from_system_event(proposed)
    assert recovered.ts_ns == 9_999
    assert recovered == proposal


def test_run_rejects_negative_ts_and_empty_reason():
    orch, _ = _make_orchestrator()
    with pytest.raises(ValueError):
        orch.run(
            proposal=_proposal(),
            evidence=_clean_evidence(),
            ts_ns=-1,
        )
    with pytest.raises(ValueError):
        orch.run(
            proposal=_proposal(),
            evidence=_clean_evidence(),
            ts_ns=1000,
            approve_reason="",
        )


# ---------------------------------------------------------------------------
# event projection round-trip
# ---------------------------------------------------------------------------


def test_proposal_event_round_trip():
    proposal = _proposal()
    event = proposal_as_system_event(proposal)
    assert event.sub_kind is SystemEventKind.PATCH_PROPOSED
    assert event.source == PATCH_EVENT_SOURCE_PROPOSAL
    assert event.ts_ns == proposal.ts_ns
    assert proposal_from_system_event(event) == proposal


def test_verdict_event_round_trip():
    verdict = StageVerdict(
        ts_ns=42,
        stage=PatchStage.SHADOW,
        passed=True,
        detail="samples=120 matches=119 err=0.0083",
        meta={"samples": "120", "matches": "119", "error_rate": "0.008333"},
    )
    event = verdict_as_system_event(patch_id="px", verdict=verdict)
    assert event.sub_kind is SystemEventKind.PATCH_STAGE_VERDICT
    assert event.source == PATCH_EVENT_SOURCE_VERDICT
    pid, parsed = verdict_from_system_event(event)
    assert pid == "px"
    assert parsed == verdict


def test_decision_event_round_trip():
    decision = PatchApprovalDecision(
        ts_ns=99,
        patch_id="px",
        decision="APPROVED",
        reason="canary_clean",
        final_stage=PatchStage.APPROVED,
        meta={"who": "governance"},
    )
    event = decision_as_system_event(decision)
    assert event.sub_kind is SystemEventKind.PATCH_DECISION
    assert event.source == PATCH_EVENT_SOURCE_DECISION
    assert decision_from_system_event(event) == decision


def test_decision_event_rejects_unknown_decision():
    bad = PatchApprovalDecision(
        ts_ns=1,
        patch_id="px",
        decision="MAYBE",
        reason="?",
        final_stage=PatchStage.CANARY,
        meta={},
    )
    with pytest.raises(ValueError):
        decision_as_system_event(bad)


def test_proposal_event_rejects_empty_patch_id():
    bad = PatchProposal(
        ts_ns=1,
        patch_id="",
        source="x",
        target_strategy="y",
        touchpoints=("a",),
        rationale="z",
    )
    with pytest.raises(ValueError):
        proposal_as_system_event(bad)


def test_verdict_event_rejects_wrong_kind():
    other = SystemEvent(
        ts_ns=1,
        sub_kind=SystemEventKind.HEARTBEAT,
        source="x",
        payload={},
    )
    with pytest.raises(ValueError):
        verdict_from_system_event(other)


# ---------------------------------------------------------------------------
# Governance authority preserved (SAFE-69)
# ---------------------------------------------------------------------------


def test_orchestrator_does_not_bypass_bridge_for_terminal_transitions():
    """The orchestrator never calls pipeline.transition() directly for
    APPROVED / REJECTED — every terminal transition goes through the
    bridge so Governance retains sole authority (SAFE-69)."""
    orch, bridge = _make_orchestrator()
    run = orch.run(
        proposal=_proposal(), evidence=_clean_evidence(), ts_ns=8000
    )
    # Bridge.decisions captures every terminal transition driven by
    # Governance — exactly one for a happy-path APPROVED.
    assert len(bridge.decisions) == 1
    assert bridge.decisions[0] is run.decision


def test_orchestrator_run_terminal_record_matches_bridge_pipeline_view():
    orch, bridge = _make_orchestrator()
    proposal = _proposal()
    run = orch.run(
        proposal=proposal, evidence=_clean_evidence(), ts_ns=9000
    )
    assert bridge.pipeline.get(proposal.patch_id) == run.record


def test_orchestrator_rejects_duplicate_proposal_ids():
    orch, _ = _make_orchestrator()
    proposal = _proposal()
    orch.run(
        proposal=proposal, evidence=_clean_evidence(), ts_ns=10_000
    )
    # Re-running the same proposal must fail because the bridge's
    # underlying pipeline already owns that patch_id.
    with pytest.raises(PatchPipelineError):
        orch.run(
            proposal=proposal,
            evidence=_clean_evidence(),
            ts_ns=11_000,
        )
