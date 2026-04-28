"""Phase 5 — Learning + Evolution closed-loop coordination.

Build Compiler Spec §2 Phase 5 + §8: deterministic, IO-free, pure-Python.

    Execution → FeedbackCollector → PatchOutcomeFeedback → {
        UpdateEmitter (parameter mutation),
        MutationProposer (structural mutation),
    } → PatchApprovalBridge → PatchPipeline → APPROVED/REJECTED/ROLLED_BACK

Each unit + the end-to-end stitch are covered. Replay determinism
(INV-15) is verified per module.
"""

from __future__ import annotations

import pytest

from core.contracts.events import ExecutionStatus, SystemEventKind
from core.contracts.learning import (
    LearningUpdate,
    PatchProposal,
    StrategyStats,
    TradeOutcome,
)
from core.contracts.patch import (
    PatchPipelineError,
    PatchStage,
    StageVerdict,
)
from evolution_engine.intelligence_loops.mutation_proposer import (
    MutationProposer,
    MutationThresholds,
)
from evolution_engine.patch_pipeline import PatchPipeline
from execution_engine.protections.feedback import FeedbackCollector
from governance_engine.services.patch_pipeline_bridge import (
    PatchApprovalBridge,
)
from learning_engine.lanes.patch_outcome_feedback import PatchOutcomeFeedback
from learning_engine.update_emitter import UpdateEmitter

# ---------------------------------------------------------------------------
# EXEC-09 — FeedbackCollector
# ---------------------------------------------------------------------------


def _outcome(
    *,
    ts_ns: int = 1,
    strategy_id: str = "s1",
    pnl: float = 0.0,
    status: ExecutionStatus = ExecutionStatus.FILLED,
) -> TradeOutcome:
    return TradeOutcome(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        symbol="EURUSD",
        qty=1.0,
        pnl=pnl,
        status=status,
    )


def test_feedback_collector_terminal_only():
    fc = FeedbackCollector()
    assert (
        fc.record(
            ts_ns=1,
            strategy_id="s1",
            symbol="EURUSD",
            qty=1.0,
            pnl=10.0,
            status=ExecutionStatus.PROPOSED,
        )
        is None
    )
    assert (
        fc.record(
            ts_ns=2,
            strategy_id="s1",
            symbol="EURUSD",
            qty=1.0,
            pnl=10.0,
            status=ExecutionStatus.SUBMITTED,
        )
        is None
    )
    assert len(fc) == 0
    out = fc.record(
        ts_ns=3,
        strategy_id="s1",
        symbol="EURUSD",
        qty=1.0,
        pnl=10.0,
        status=ExecutionStatus.FILLED,
    )
    assert out is not None
    assert out.strategy_id == "s1"
    assert len(fc) == 1


def test_feedback_collector_drain_clears_buffer():
    fc = FeedbackCollector()
    for i, st in enumerate(
        (
            ExecutionStatus.FILLED,
            ExecutionStatus.PARTIALLY_FILLED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.REJECTED,
            ExecutionStatus.FAILED,
        )
    ):
        fc.record(
            ts_ns=i,
            strategy_id="s",
            symbol="X",
            qty=1.0,
            pnl=0.0,
            status=st,
        )
    drained = fc.drain()
    assert len(drained) == 5
    assert len(fc) == 0
    # peek vs drain
    fc.record(
        ts_ns=10,
        strategy_id="s",
        symbol="X",
        qty=1.0,
        pnl=1.0,
        status=ExecutionStatus.FILLED,
    )
    assert fc.peek() == fc.peek()
    assert len(fc) == 1


def test_feedback_collector_validates_required_fields():
    fc = FeedbackCollector()
    with pytest.raises(ValueError):
        fc.record(
            ts_ns=1,
            strategy_id="",
            symbol="X",
            qty=1.0,
            pnl=0.0,
            status=ExecutionStatus.FILLED,
        )
    with pytest.raises(ValueError):
        fc.record(
            ts_ns=1,
            strategy_id="s",
            symbol="",
            qty=1.0,
            pnl=0.0,
            status=ExecutionStatus.FILLED,
        )


def test_feedback_collector_replay_determinism():
    def run() -> tuple[TradeOutcome, ...]:
        fc = FeedbackCollector()
        for i in range(3):
            fc.record(
                ts_ns=i,
                strategy_id="s",
                symbol="X",
                qty=1.0,
                pnl=float(i),
                status=ExecutionStatus.FILLED,
            )
        return fc.drain()

    assert run() == run()


# ---------------------------------------------------------------------------
# DYN-L02 — PatchOutcomeFeedback
# ---------------------------------------------------------------------------


def test_outcome_feedback_initial_snapshot_is_zero():
    pof = PatchOutcomeFeedback()
    s = pof.snapshot(strategy_id="never_seen", ts_ns=1)
    assert s.n_trades == 0
    assert s.n_wins == 0
    assert s.win_rate == 0.0
    assert s.total_pnl == 0.0


def test_outcome_feedback_aggregates_wins_losses_pnl():
    pof = PatchOutcomeFeedback()
    pof.observe(_outcome(ts_ns=1, pnl=5.0))
    pof.observe(_outcome(ts_ns=2, pnl=-2.0))
    pof.observe(_outcome(ts_ns=3, pnl=3.0))
    s = pof.snapshot(strategy_id="s1", ts_ns=4)
    assert s.n_trades == 3
    assert s.n_wins == 2
    assert s.n_losses == 1
    assert s.total_pnl == pytest.approx(6.0)
    assert s.mean_pnl == pytest.approx(2.0)
    assert s.win_rate == pytest.approx(2 / 3)


def test_outcome_feedback_window_evicts_oldest():
    pof = PatchOutcomeFeedback(window=3)
    for i in range(5):
        pof.observe(_outcome(ts_ns=i, pnl=float(i)))
    s = pof.snapshot(strategy_id="s1", ts_ns=10)
    # window is 3, last 3 outcomes have pnl 2,3,4 → total 9
    assert s.n_trades == 3
    assert s.total_pnl == pytest.approx(9.0)


def test_outcome_feedback_isolates_strategies():
    pof = PatchOutcomeFeedback()
    pof.observe(_outcome(ts_ns=1, strategy_id="a", pnl=1.0))
    pof.observe(_outcome(ts_ns=2, strategy_id="b", pnl=-1.0))
    snaps = pof.all_snapshots(ts_ns=3)
    assert set(snaps.keys()) == {"a", "b"}
    assert snaps["a"].total_pnl == pytest.approx(1.0)
    assert snaps["b"].total_pnl == pytest.approx(-1.0)


def test_outcome_feedback_rejects_bad_window():
    with pytest.raises(ValueError):
        PatchOutcomeFeedback(window=0)


def test_outcome_feedback_replay_determinism():
    def run() -> tuple[StrategyStats, ...]:
        pof = PatchOutcomeFeedback(window=4)
        out: list[StrategyStats] = []
        for i, p in enumerate((1.0, -2.0, 3.0, -1.0, 2.0)):
            out.append(pof.observe(_outcome(ts_ns=i, pnl=p)))
        return tuple(out)

    assert run() == run()


# ---------------------------------------------------------------------------
# Learning UpdateEmitter
# ---------------------------------------------------------------------------


def test_update_emitter_propose_validates():
    with pytest.raises(ValueError):
        UpdateEmitter.propose(
            ts_ns=1,
            strategy_id="",
            parameter="p",
            old_value="0",
            new_value="1",
            reason="r",
        )
    with pytest.raises(ValueError):
        UpdateEmitter.propose(
            ts_ns=1,
            strategy_id="s",
            parameter="",
            old_value="0",
            new_value="1",
            reason="r",
        )
    with pytest.raises(ValueError):
        UpdateEmitter.propose(
            ts_ns=1,
            strategy_id="s",
            parameter="p",
            old_value="0",
            new_value="1",
            reason="",
        )


def test_update_emitter_emits_system_event():
    em = UpdateEmitter(source="learning_engine")
    upd = UpdateEmitter.propose(
        ts_ns=42,
        strategy_id="s1",
        parameter="threshold",
        old_value="0.5",
        new_value="0.6",
        reason="win_rate_drift",
        meta={"trace_id": "abc"},
    )
    ev = em.emit(upd)
    assert ev.ts_ns == 42
    assert ev.sub_kind is SystemEventKind.UPDATE_PROPOSED
    assert ev.source == "learning_engine"
    assert ev.payload["parameter"] == "threshold"
    assert ev.payload["old_value"] == "0.5"
    assert ev.payload["new_value"] == "0.6"
    assert ev.meta["trace_id"] == "abc"


def test_update_emitter_emit_many_preserves_order():
    em = UpdateEmitter()
    updates = tuple(
        UpdateEmitter.propose(
            ts_ns=i,
            strategy_id="s",
            parameter="p",
            old_value="x",
            new_value="y",
            reason="r",
        )
        for i in range(4)
    )
    events = em.emit_many(updates)
    assert tuple(e.ts_ns for e in events) == (0, 1, 2, 3)


def test_update_emitter_rejects_empty_source():
    with pytest.raises(ValueError):
        UpdateEmitter(source="")


def test_update_emitter_replay_determinism():
    def run() -> LearningUpdate:
        return UpdateEmitter.propose(
            ts_ns=7,
            strategy_id="s1",
            parameter="lr",
            old_value="0.01",
            new_value="0.02",
            reason="alpha_decay",
        )

    assert run() == run()


# ---------------------------------------------------------------------------
# DYN-L01 — MutationProposer
# ---------------------------------------------------------------------------


def _stats(
    *,
    ts_ns: int = 1,
    strategy_id: str = "s1",
    n_trades: int = 100,
    win_rate: float = 0.30,
    mean_pnl: float = -0.5,
) -> StrategyStats:
    n_wins = int(n_trades * win_rate)
    return StrategyStats(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_trades - n_wins,
        total_pnl=mean_pnl * n_trades,
        mean_pnl=mean_pnl,
        win_rate=win_rate,
    )


def test_mutation_proposer_silent_below_min_trades():
    mp = MutationProposer()
    out = mp.evaluate(_stats(n_trades=5, win_rate=0.10, mean_pnl=-1.0))
    assert out == ()


def test_mutation_proposer_emits_on_breach():
    mp = MutationProposer()
    out = mp.evaluate(_stats(win_rate=0.10, mean_pnl=-1.0))
    # Both win_rate AND mean_pnl breached
    assert len(out) == 2
    reasons = {p.meta["reason"] for p in out}
    assert reasons == {"win_rate_below_floor", "mean_pnl_below_floor"}
    for p in out:
        assert p.target_strategy == "s1"
        assert p.patch_id.startswith("PATCH-s1-")


def test_mutation_proposer_one_shot_until_recovery():
    mp = MutationProposer()
    first = mp.evaluate(_stats(ts_ns=1, win_rate=0.10, mean_pnl=-1.0))
    assert len(first) == 2
    # Same breach state — no re-emission
    second = mp.evaluate(_stats(ts_ns=2, win_rate=0.10, mean_pnl=-1.0))
    assert second == ()
    # Recover both → disarm
    mp.evaluate(_stats(ts_ns=3, win_rate=0.80, mean_pnl=1.0))
    # Re-breach → re-emits
    third = mp.evaluate(_stats(ts_ns=4, win_rate=0.10, mean_pnl=-1.0))
    assert len(third) == 2


def test_mutation_proposer_dropping_below_min_trades_clears_armed():
    mp = MutationProposer()
    mp.evaluate(_stats(ts_ns=1, win_rate=0.10, mean_pnl=-1.0))
    # n_trades drops below min — disarms
    mp.evaluate(_stats(ts_ns=2, n_trades=5, win_rate=0.10, mean_pnl=-1.0))
    # Re-breach with valid n_trades → re-emits
    out = mp.evaluate(_stats(ts_ns=3, win_rate=0.10, mean_pnl=-1.0))
    assert len(out) == 2


def test_mutation_proposer_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        MutationThresholds(min_trades=0)
    with pytest.raises(ValueError):
        MutationThresholds(min_win_rate=-0.1)
    with pytest.raises(ValueError):
        MutationThresholds(min_win_rate=1.5)


def test_mutation_proposer_partial_breach_only_emits_one():
    mp = MutationProposer()
    # Only mean_pnl breached, win_rate ok
    out = mp.evaluate(_stats(win_rate=0.80, mean_pnl=-0.5))
    assert len(out) == 1
    assert out[0].meta["reason"] == "mean_pnl_below_floor"


def test_mutation_proposer_replay_determinism():
    def run() -> tuple[PatchProposal, ...]:
        mp = MutationProposer()
        out: list[PatchProposal] = []
        for i, w in enumerate((0.10, 0.10, 0.80, 0.10)):
            r = mp.evaluate(
                _stats(ts_ns=i, win_rate=w, mean_pnl=-1.0)
            )
            out.extend(r)
        return tuple(out)

    assert run() == run()


# ---------------------------------------------------------------------------
# GOV-G18 — PatchApprovalBridge
# ---------------------------------------------------------------------------


def _proposal(*, ts_ns: int = 1, patch_id: str = "P1") -> PatchProposal:
    return PatchProposal(
        ts_ns=ts_ns,
        patch_id=patch_id,
        source="mutation_proposer",
        target_strategy="s1",
        touchpoints=("strategies.s1.entry_filter",),
        rationale="win_rate=0.10 < 0.40",
    )


def test_bridge_receive_proposal_creates_record():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    rec = bridge.receive_proposal(_proposal())
    assert rec.stage is PatchStage.PROPOSED
    assert rec.patch_id == "P1"


def test_bridge_advance_runs_full_pipeline():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    bridge.receive_proposal(_proposal())
    rec = bridge.advance(
        patch_id="P1",
        new_stage=PatchStage.SANDBOX,
        ts_ns=2,
        verdict=StageVerdict(
            ts_ns=2, stage=PatchStage.SANDBOX, passed=True
        ),
    )
    assert rec.stage is PatchStage.SANDBOX
    assert len(rec.verdicts) == 1
    for stage in (
        PatchStage.STATIC_ANALYSIS,
        PatchStage.BACKTEST,
        PatchStage.SHADOW,
        PatchStage.CANARY,
    ):
        rec = bridge.advance(patch_id="P1", new_stage=stage, ts_ns=2)
    decision = bridge.approve(patch_id="P1", ts_ns=10, reason="green")
    assert decision.decision == "APPROVED"
    assert decision.final_stage is PatchStage.APPROVED
    assert bridge.decisions == (decision,)


def test_bridge_reject_from_arbitrary_stage():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    bridge.receive_proposal(_proposal())
    decision = bridge.reject(patch_id="P1", ts_ns=2, reason="static_failed")
    assert decision.decision == "REJECTED"
    assert decision.final_stage is PatchStage.REJECTED


def test_bridge_approve_requires_canary():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    bridge.receive_proposal(_proposal())
    with pytest.raises(PatchPipelineError):
        bridge.approve(patch_id="P1", ts_ns=2)


def test_bridge_rollback_after_approval():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    bridge.receive_proposal(_proposal())
    for stage in (
        PatchStage.SANDBOX,
        PatchStage.STATIC_ANALYSIS,
        PatchStage.BACKTEST,
        PatchStage.SHADOW,
        PatchStage.CANARY,
    ):
        bridge.advance(patch_id="P1", new_stage=stage, ts_ns=2)
    bridge.approve(patch_id="P1", ts_ns=3)
    decision = bridge.rollback(
        patch_id="P1", ts_ns=4, reason="canary_regression"
    )
    assert decision.decision == "ROLLED_BACK"
    assert decision.final_stage is PatchStage.ROLLED_BACK


def test_bridge_rejects_empty_reason_on_reject_and_rollback():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    bridge.receive_proposal(_proposal())
    with pytest.raises(ValueError):
        bridge.reject(patch_id="P1", ts_ns=2, reason="")
    with pytest.raises(ValueError):
        bridge.rollback(patch_id="P1", ts_ns=2, reason="")


def test_bridge_rejects_proposal_without_id():
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)
    bad = PatchProposal(
        ts_ns=1,
        patch_id="",
        source="x",
        target_strategy="s1",
        touchpoints=(),
        rationale="r",
    )
    with pytest.raises(ValueError):
        bridge.receive_proposal(bad)


def test_bridge_replay_determinism():
    def run() -> tuple:
        pp = PatchPipeline()
        bridge = PatchApprovalBridge(pipeline=pp)
        bridge.receive_proposal(_proposal())
        bridge.reject(patch_id="P1", ts_ns=2, reason="static_failed")
        return tuple(
            (d.ts_ns, d.decision, d.final_stage) for d in bridge.decisions
        )

    assert run() == run()


# ---------------------------------------------------------------------------
# End-to-end stitch: Execution → Feedback → Learning → Evolution → Governance
# ---------------------------------------------------------------------------


def test_closed_loop_end_to_end_emits_proposal_and_approves():
    """Build Compiler Spec §8 closed loop, fully wired in-process."""
    fc = FeedbackCollector()
    pof = PatchOutcomeFeedback()
    mp = MutationProposer()
    pp = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pp)

    # 30 losing trades — breaches both win_rate AND mean_pnl
    for i in range(30):
        fc.record(
            ts_ns=i,
            strategy_id="s1",
            symbol="EURUSD",
            qty=1.0,
            pnl=-0.5,
            status=ExecutionStatus.FILLED,
        )

    proposals: list[PatchProposal] = []
    for outcome in fc.drain():
        stats = pof.observe(outcome)
        proposals.extend(mp.evaluate(stats))

    assert len(proposals) == 2
    assert {p.meta["reason"] for p in proposals} == {
        "win_rate_below_floor",
        "mean_pnl_below_floor",
    }

    # Governance receives + drives one to APPROVED, other to REJECTED
    approved = proposals[0]
    rejected = proposals[1]
    bridge.receive_proposal(approved)
    bridge.receive_proposal(rejected)
    bridge.reject(
        patch_id=rejected.patch_id, ts_ns=100, reason="duplicate_breach"
    )
    for stage in (
        PatchStage.SANDBOX,
        PatchStage.STATIC_ANALYSIS,
        PatchStage.BACKTEST,
        PatchStage.SHADOW,
        PatchStage.CANARY,
    ):
        bridge.advance(patch_id=approved.patch_id, new_stage=stage, ts_ns=101)
    decision = bridge.approve(
        patch_id=approved.patch_id, ts_ns=102, reason="canary_clean"
    )
    assert decision.decision == "APPROVED"
    assert {d.decision for d in bridge.decisions} == {"APPROVED", "REJECTED"}


def test_closed_loop_replay_determinism():
    def run() -> tuple:
        fc = FeedbackCollector()
        pof = PatchOutcomeFeedback()
        mp = MutationProposer()
        pp = PatchPipeline()
        bridge = PatchApprovalBridge(pipeline=pp)
        for i in range(30):
            fc.record(
                ts_ns=i,
                strategy_id="s1",
                symbol="EURUSD",
                qty=1.0,
                pnl=-0.5,
                status=ExecutionStatus.FILLED,
            )
        all_props: list[PatchProposal] = []
        for outcome in fc.drain():
            stats = pof.observe(outcome)
            all_props.extend(mp.evaluate(stats))
        for p in all_props:
            bridge.receive_proposal(p)
        return tuple(p.patch_id for p in all_props)

    assert run() == run()
