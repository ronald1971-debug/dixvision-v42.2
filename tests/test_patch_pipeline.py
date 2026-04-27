"""Tests for the patch pipeline FSM + 6 stages + rollback planner (Phase 4)."""

from __future__ import annotations

import pytest

from evolution_engine.patch_pipeline import (
    LEGAL_PATCH_TRANSITIONS,
    BacktestStage,
    BacktestSummary,
    CanaryStage,
    FindingSeverity,
    PatchPipeline,
    PatchPipelineError,
    PatchStage,
    RollbackPlanner,
    RollbackStep,
    SandboxStage,
    ShadowStage,
    StageVerdict,
    StaticAnalysisFinding,
    StaticAnalysisStage,
)

# ---------------------------------------------------------------------------
# PatchPipeline FSM
# ---------------------------------------------------------------------------


def test_patch_pipeline_propose_initial_state():
    p = PatchPipeline()
    rec = p.propose(patch_id="P1", ts_ns=1)
    assert rec.patch_id == "P1"
    assert rec.stage is PatchStage.PROPOSED
    assert len(rec.history) == 1
    assert rec.history[0].new is PatchStage.PROPOSED


def test_patch_pipeline_rejects_empty_id():
    p = PatchPipeline()
    with pytest.raises(ValueError):
        p.propose(patch_id="", ts_ns=1)


def test_patch_pipeline_rejects_duplicate_propose():
    p = PatchPipeline()
    p.propose(patch_id="P1", ts_ns=1)
    with pytest.raises(PatchPipelineError):
        p.propose(patch_id="P1", ts_ns=2)


def test_patch_pipeline_get_unknown_raises():
    p = PatchPipeline()
    with pytest.raises(PatchPipelineError):
        p.get("nope")


def test_patch_pipeline_full_happy_path():
    p = PatchPipeline()
    p.propose(patch_id="P1", ts_ns=1)
    for stage in (
        PatchStage.SANDBOX,
        PatchStage.STATIC_ANALYSIS,
        PatchStage.BACKTEST,
        PatchStage.SHADOW,
        PatchStage.CANARY,
        PatchStage.APPROVED,
    ):
        p.transition(patch_id="P1", new_stage=stage, ts_ns=10, reason=stage.value)
    assert p.get("P1").stage is PatchStage.APPROVED


def test_patch_pipeline_rejects_illegal_transition():
    p = PatchPipeline()
    p.propose(patch_id="P1", ts_ns=1)
    with pytest.raises(PatchPipelineError):
        p.transition(
            patch_id="P1",
            new_stage=PatchStage.APPROVED,  # PROPOSED → APPROVED is illegal
            ts_ns=2,
            reason="illegal",
        )


def test_patch_pipeline_reject_at_any_stage():
    p = PatchPipeline()
    p.propose(patch_id="P1", ts_ns=1)
    p.transition(
        patch_id="P1",
        new_stage=PatchStage.SANDBOX,
        ts_ns=2,
        reason="advance",
    )
    p.transition(
        patch_id="P1",
        new_stage=PatchStage.REJECTED,
        ts_ns=3,
        reason="rejected by reviewer",
    )
    rec = p.get("P1")
    assert rec.stage is PatchStage.REJECTED
    # Rejected is terminal.
    assert LEGAL_PATCH_TRANSITIONS[PatchStage.REJECTED] == frozenset()


def test_patch_pipeline_rollback_after_approval():
    p = PatchPipeline()
    p.propose(patch_id="P1", ts_ns=1)
    for stage in (
        PatchStage.SANDBOX,
        PatchStage.STATIC_ANALYSIS,
        PatchStage.BACKTEST,
        PatchStage.SHADOW,
        PatchStage.CANARY,
        PatchStage.APPROVED,
    ):
        p.transition(patch_id="P1", new_stage=stage, ts_ns=10, reason=stage.value)
    p.transition(
        patch_id="P1",
        new_stage=PatchStage.ROLLED_BACK,
        ts_ns=20,
        reason="hot-rollback",
    )
    assert p.get("P1").stage is PatchStage.ROLLED_BACK


def test_patch_pipeline_record_verdicts():
    p = PatchPipeline()
    p.propose(patch_id="P1", ts_ns=1)
    p.record_verdict(
        patch_id="P1",
        verdict=StageVerdict(
            ts_ns=2,
            stage=PatchStage.SANDBOX,
            passed=True,
        ),
    )
    rec = p.get("P1")
    assert len(rec.verdicts) == 1
    assert rec.verdicts[0].passed is True


def test_patch_pipeline_record_verdict_unknown_raises():
    p = PatchPipeline()
    with pytest.raises(PatchPipelineError):
        p.record_verdict(
            patch_id="X",
            verdict=StageVerdict(ts_ns=1, stage=PatchStage.SANDBOX, passed=True),
        )


def test_patch_pipeline_all_in():
    p = PatchPipeline()
    p.propose(patch_id="A", ts_ns=1)
    p.propose(patch_id="B", ts_ns=1)
    p.transition(
        patch_id="A",
        new_stage=PatchStage.SANDBOX,
        ts_ns=2,
        reason="advance",
    )
    proposed = p.all_in(PatchStage.PROPOSED)
    sandbox = p.all_in(PatchStage.SANDBOX)
    assert tuple(r.patch_id for r in proposed) == ("B",)
    assert tuple(r.patch_id for r in sandbox) == ("A",)


def test_patch_pipeline_replay_determinism():
    def run() -> tuple:
        p = PatchPipeline()
        p.propose(patch_id="P1", ts_ns=1)
        p.transition(
            patch_id="P1",
            new_stage=PatchStage.SANDBOX,
            ts_ns=2,
            reason="advance",
        )
        rec = p.get("P1")
        return tuple((t.prev, t.new, t.reason) for t in rec.history)

    assert run() == run()


# ---------------------------------------------------------------------------
# SandboxStage
# ---------------------------------------------------------------------------


def test_sandbox_clean_passes():
    s = SandboxStage()
    result, verdict = s.evaluate(
        ts_ns=1,
        touchpoints=("intelligence_engine.plugins.x", "core.contracts.market"),
    )
    assert verdict.passed is True
    assert result.forbidden_touchpoints == ()
    assert len(result.accepted_touchpoints) == 2


def test_sandbox_rejects_forbidden_imports():
    s = SandboxStage()
    result, verdict = s.evaluate(
        ts_ns=1,
        touchpoints=("subprocess.run", "core.contracts.market", "socket.socket"),
    )
    assert verdict.passed is False
    assert "subprocess.run" in result.forbidden_touchpoints
    assert "socket.socket" in result.forbidden_touchpoints
    assert "core.contracts.market" in result.accepted_touchpoints


def test_sandbox_custom_forbidden_prefixes():
    s = SandboxStage(forbidden_prefixes=("dangerous.",))
    _, verdict = s.evaluate(
        ts_ns=1,
        touchpoints=("dangerous.module", "subprocess.run"),
    )
    # only `dangerous.` is forbidden under custom rule
    assert verdict.passed is False
    # subprocess.run should now pass (not in custom list)


def test_sandbox_replay_determinism():
    def run() -> tuple:
        s = SandboxStage()
        result, verdict = s.evaluate(
            ts_ns=1,
            touchpoints=("subprocess.run", "core.contracts.market"),
        )
        return (result.forbidden_touchpoints, verdict.passed, verdict.detail)

    assert run() == run()


# ---------------------------------------------------------------------------
# StaticAnalysisStage
# ---------------------------------------------------------------------------


def test_static_analysis_no_findings_passes():
    s = StaticAnalysisStage()
    verdict = s.evaluate(ts_ns=1, findings=())
    assert verdict.passed is True


def test_static_analysis_warn_passes_by_default():
    s = StaticAnalysisStage()
    findings = (
        StaticAnalysisFinding(
            rule="W001",
            severity=FindingSeverity.WARN,
            location="x.py:1",
        ),
    )
    verdict = s.evaluate(ts_ns=1, findings=findings)
    assert verdict.passed is True


def test_static_analysis_error_fails_default():
    s = StaticAnalysisStage()
    findings = (
        StaticAnalysisFinding(
            rule="E001",
            severity=FindingSeverity.ERROR,
            location="x.py:1",
        ),
    )
    verdict = s.evaluate(ts_ns=1, findings=findings)
    assert verdict.passed is False


def test_static_analysis_strict_warn_threshold():
    s = StaticAnalysisStage(max_severity=FindingSeverity.INFO)
    findings = (
        StaticAnalysisFinding(
            rule="W001",
            severity=FindingSeverity.WARN,
            location="x.py:1",
        ),
    )
    verdict = s.evaluate(ts_ns=1, findings=findings)
    assert verdict.passed is False


def test_static_analysis_replay_determinism():
    def run() -> tuple:
        s = StaticAnalysisStage()
        findings = (
            StaticAnalysisFinding(
                rule="W001",
                severity=FindingSeverity.WARN,
                location="x.py:1",
            ),
        )
        v = s.evaluate(ts_ns=1, findings=findings)
        return (v.passed, v.detail, tuple(sorted(v.meta.items())))

    assert run() == run()


# ---------------------------------------------------------------------------
# BacktestStage
# ---------------------------------------------------------------------------


def test_backtest_passes_when_metrics_above_floor():
    s = BacktestStage(min_runs=10, min_sharpe=0.5, max_drawdown=0.3)
    summary = BacktestSummary(runs=20, pnl=100.0, sharpe=1.2, max_drawdown=0.1)
    verdict = s.evaluate(ts_ns=1, summary=summary)
    assert verdict.passed is True


def test_backtest_fails_when_runs_too_few():
    s = BacktestStage(min_runs=10, min_sharpe=0.5, max_drawdown=0.3)
    summary = BacktestSummary(runs=5, pnl=100.0, sharpe=1.2, max_drawdown=0.1)
    verdict = s.evaluate(ts_ns=1, summary=summary)
    assert verdict.passed is False


def test_backtest_fails_when_sharpe_too_low():
    s = BacktestStage(min_runs=10, min_sharpe=0.5, max_drawdown=0.3)
    summary = BacktestSummary(runs=20, pnl=100.0, sharpe=0.1, max_drawdown=0.1)
    verdict = s.evaluate(ts_ns=1, summary=summary)
    assert verdict.passed is False


def test_backtest_fails_when_drawdown_too_high():
    s = BacktestStage(min_runs=10, min_sharpe=0.5, max_drawdown=0.3)
    summary = BacktestSummary(runs=20, pnl=100.0, sharpe=1.0, max_drawdown=0.5)
    verdict = s.evaluate(ts_ns=1, summary=summary)
    assert verdict.passed is False


def test_backtest_validates_constructor_args():
    with pytest.raises(ValueError):
        BacktestStage(min_runs=0)
    with pytest.raises(ValueError):
        BacktestStage(max_drawdown=0.0)
    with pytest.raises(ValueError):
        BacktestStage(max_drawdown=1.0)


def test_backtest_replay_determinism():
    def run() -> tuple:
        s = BacktestStage(min_runs=1, min_sharpe=0.0, max_drawdown=0.5)
        summary = BacktestSummary(runs=10, pnl=42.0, sharpe=1.5, max_drawdown=0.1)
        v = s.evaluate(ts_ns=1, summary=summary)
        return (v.passed, v.detail)

    assert run() == run()


# ---------------------------------------------------------------------------
# ShadowStage
# ---------------------------------------------------------------------------


def test_shadow_passes_when_clean():
    s = ShadowStage(min_samples=10, max_error_rate=0.05)
    sv, verdict = s.evaluate(ts_ns=1, samples=100, matches=99)
    assert verdict.passed is True
    assert sv.error_rate == pytest.approx(0.01)


def test_shadow_fails_when_too_few_samples():
    s = ShadowStage(min_samples=10, max_error_rate=0.05)
    _, verdict = s.evaluate(ts_ns=1, samples=5, matches=5)
    assert verdict.passed is False


def test_shadow_fails_when_error_rate_too_high():
    s = ShadowStage(min_samples=10, max_error_rate=0.05)
    _, verdict = s.evaluate(ts_ns=1, samples=100, matches=80)
    assert verdict.passed is False


def test_shadow_zero_samples_zero_error_rate():
    s = ShadowStage(min_samples=10, max_error_rate=0.05)
    sv, _ = s.evaluate(ts_ns=1, samples=0, matches=0)
    assert sv.error_rate == 0.0


def test_shadow_rejects_invalid_counts():
    s = ShadowStage(min_samples=1, max_error_rate=0.05)
    with pytest.raises(ValueError):
        s.evaluate(ts_ns=1, samples=10, matches=20)
    with pytest.raises(ValueError):
        s.evaluate(ts_ns=1, samples=-1, matches=0)


def test_shadow_validates_constructor_args():
    with pytest.raises(ValueError):
        ShadowStage(min_samples=0)
    with pytest.raises(ValueError):
        ShadowStage(max_error_rate=-0.1)
    with pytest.raises(ValueError):
        ShadowStage(max_error_rate=1.1)


def test_shadow_replay_determinism():
    def run() -> tuple:
        s = ShadowStage(min_samples=10, max_error_rate=0.05)
        sv, verdict = s.evaluate(ts_ns=1, samples=100, matches=99)
        return (sv.error_rate, verdict.passed, verdict.detail)

    assert run() == run()


# ---------------------------------------------------------------------------
# CanaryStage
# ---------------------------------------------------------------------------


def test_canary_passes_when_clean():
    s = CanaryStage(min_orders=5, max_error_rate=0.10, min_pnl=0.0)
    cv, verdict = s.evaluate(ts_ns=1, orders=10, rejects=0, realised_pnl=5.0)
    assert verdict.passed is True
    assert cv.error_rate == 0.0


def test_canary_fails_when_pnl_negative():
    s = CanaryStage(min_orders=5, max_error_rate=0.10, min_pnl=0.0)
    _, verdict = s.evaluate(ts_ns=1, orders=10, rejects=0, realised_pnl=-1.0)
    assert verdict.passed is False


def test_canary_fails_when_too_few_orders():
    s = CanaryStage(min_orders=5, max_error_rate=0.10, min_pnl=0.0)
    _, verdict = s.evaluate(ts_ns=1, orders=2, rejects=0, realised_pnl=5.0)
    assert verdict.passed is False


def test_canary_fails_when_reject_rate_too_high():
    s = CanaryStage(min_orders=5, max_error_rate=0.10, min_pnl=0.0)
    _, verdict = s.evaluate(ts_ns=1, orders=10, rejects=5, realised_pnl=5.0)
    assert verdict.passed is False


def test_canary_zero_orders_zero_error():
    s = CanaryStage(min_orders=5, max_error_rate=0.10, min_pnl=0.0)
    cv, _ = s.evaluate(ts_ns=1, orders=0, rejects=0, realised_pnl=0.0)
    assert cv.error_rate == 0.0


def test_canary_rejects_invalid_counts():
    s = CanaryStage(min_orders=1, max_error_rate=0.10, min_pnl=0.0)
    with pytest.raises(ValueError):
        s.evaluate(ts_ns=1, orders=10, rejects=20, realised_pnl=0.0)


def test_canary_validates_constructor_args():
    with pytest.raises(ValueError):
        CanaryStage(min_orders=0)
    with pytest.raises(ValueError):
        CanaryStage(max_error_rate=-0.5)
    with pytest.raises(ValueError):
        CanaryStage(max_error_rate=1.5)


def test_canary_replay_determinism():
    def run() -> tuple:
        s = CanaryStage(min_orders=5, max_error_rate=0.10, min_pnl=0.0)
        cv, verdict = s.evaluate(ts_ns=1, orders=10, rejects=1, realised_pnl=2.5)
        return (cv.error_rate, verdict.passed, verdict.detail)

    assert run() == run()


# ---------------------------------------------------------------------------
# RollbackPlanner
# ---------------------------------------------------------------------------


def test_rollback_plan_reverses_touchpoints():
    p = RollbackPlanner()
    steps = p.plan(patch_id="P1", touchpoints=("a", "b", "c"))
    assert tuple(s.target for s in steps) == ("c", "b", "a")
    assert all(isinstance(s, RollbackStep) for s in steps)
    assert tuple(s.order for s in steps) == (0, 1, 2)


def test_rollback_plan_empty_touchpoints():
    p = RollbackPlanner()
    assert p.plan(patch_id="P1", touchpoints=()) == ()


def test_rollback_plan_rejects_empty_id():
    p = RollbackPlanner()
    with pytest.raises(ValueError):
        p.plan(patch_id="", touchpoints=("a",))


def test_rollback_plan_replay_determinism():
    def run() -> tuple:
        p = RollbackPlanner()
        steps = p.plan(patch_id="P1", touchpoints=("a", "b", "c"))
        return tuple((s.order, s.target, s.action) for s in steps)

    assert run() == run()
