"""Tests for :class:`StructuralEvolutionLoop` (P0-A).

Pins:

* HARDEN-04 / INV-70 — the loop refuses to drive the proposer /
  orchestrator when the live :class:`LearningEvolutionFreezePolicy` is
  frozen.
* The loop is the **single freeze gate** — the inner
  :class:`MutationProposer` must be constructed with ``freeze=None``.
* INV-15 byte-identical replay: same stats + policy snapshots →
  byte-identical :class:`StructuralLoopTickResult` sequence.
* B27 / B28 / INV-71 — the loop lives in ``evolution_engine.*`` so it
  IS allowed to host typed-event construction sites; pinned by an AST
  scan over ``evolution_engine/loops/structural_loop.py``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.contracts.events import SystemEventKind
from core.contracts.governance import SystemMode
from core.contracts.learning import PatchProposal, StrategyStats
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)
from core.contracts.patch import PatchStage
from evolution_engine.intelligence_loops.mutation_proposer import (
    MutationProposer,
    MutationThresholds,
)
from evolution_engine.loops.structural_loop import (
    StructuralEvolutionLoop,
    StructuralLoopTickResult,
)
from evolution_engine.patch_pipeline.backtest import BacktestSummary
from evolution_engine.patch_pipeline.orchestrator import (
    PatchPipelineOrchestrator,
    StageEvidence,
)
from evolution_engine.patch_pipeline.pipeline import PatchPipeline
from governance_engine.services.patch_pipeline_bridge import (
    PatchApprovalBridge,
)

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "evolution_engine"
    / "loops"
    / "structural_loop.py"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _frozen_policy() -> LearningEvolutionFreezePolicy:
    return LearningEvolutionFreezePolicy(mode=SystemMode.PAPER, operator_override=False)


def _unfrozen_policy() -> LearningEvolutionFreezePolicy:
    return LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=True)


def _breaching_stats(strategy_id: str = "alpha") -> StrategyStats:
    return StrategyStats(
        ts_ns=1_000_000_000,
        strategy_id=strategy_id,
        n_trades=50,
        n_wins=10,
        n_losses=40,
        total_pnl=-5.0,
        mean_pnl=-0.1,
        win_rate=0.20,
    )


def _healthy_stats(strategy_id: str = "alpha") -> StrategyStats:
    return StrategyStats(
        ts_ns=1_000_000_000,
        strategy_id=strategy_id,
        n_trades=50,
        n_wins=40,
        n_losses=10,
        total_pnl=5.0,
        mean_pnl=0.1,
        win_rate=0.80,
    )


def _clean_evidence(_proposal: PatchProposal) -> StageEvidence:
    return StageEvidence(
        sandbox_touchpoints=_proposal.touchpoints,
        static_findings=(),
        backtest_summary=BacktestSummary(runs=10, pnl=12.5, sharpe=1.5, max_drawdown=0.05),
        shadow_samples=120,
        shadow_matches=119,
        canary_orders=20,
        canary_rejects=1,
        canary_realised_pnl=0.5,
    )


def _make_loop(
    *,
    policy: LearningEvolutionFreezePolicy,
    stats: tuple[StrategyStats, ...] = (),
    proposer: MutationProposer | None = None,
) -> StructuralEvolutionLoop:
    pipeline = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pipeline)
    orchestrator = PatchPipelineOrchestrator(bridge=bridge)
    return StructuralEvolutionLoop(
        proposer=proposer
        or MutationProposer(
            thresholds=MutationThresholds(min_trades=10, min_win_rate=0.5, min_mean_pnl=0.0)
        ),
        orchestrator=orchestrator,
        policy_supplier=lambda: policy,
        stats_supplier=lambda: stats,
        evidence_builder=_clean_evidence,
    )


# ---------------------------------------------------------------------------
# Constructor invariants
# ---------------------------------------------------------------------------


def test_loop_refuses_proposer_with_inner_freeze() -> None:
    bad_proposer = MutationProposer(freeze=_unfrozen_policy())
    pipeline = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pipeline)
    orchestrator = PatchPipelineOrchestrator(bridge=bridge)
    with pytest.raises(ValueError, match="proposer.freeze=None"):
        StructuralEvolutionLoop(
            proposer=bad_proposer,
            orchestrator=orchestrator,
            policy_supplier=_unfrozen_policy,
            stats_supplier=tuple,
            evidence_builder=_clean_evidence,
        )


# ---------------------------------------------------------------------------
# Frozen path
# ---------------------------------------------------------------------------


def test_frozen_default_paper_mode_is_noop() -> None:
    stats = (_breaching_stats(),)
    loop = _make_loop(policy=_frozen_policy(), stats=stats)
    result = loop.tick(ts_ns=10_000_000_000)
    assert isinstance(result, StructuralLoopTickResult)
    assert result.frozen is True
    assert result.policy_mode_name == "PAPER"
    assert result.operator_override is False
    assert len(result.drained_stats) == 1
    assert result.proposals == ()
    assert result.runs == ()
    assert result.emitted_events == ()


# ---------------------------------------------------------------------------
# Unfrozen — breaching stats → proposal → APPROVED
# ---------------------------------------------------------------------------


def test_live_plus_override_proposes_and_drives_to_approved() -> None:
    stats = (_breaching_stats(),)
    loop = _make_loop(policy=_unfrozen_policy(), stats=stats)
    result = loop.tick(ts_ns=10_000_000_000)
    assert result.frozen is False
    assert result.policy_mode_name == "LIVE"
    assert result.operator_override is True
    assert len(result.proposals) >= 1
    assert len(result.runs) == len(result.proposals)
    for run in result.runs:
        assert run.decision.decision == "APPROVED"
        assert run.record.stage is PatchStage.APPROVED
    kinds = tuple(e.sub_kind for e in result.emitted_events)
    assert SystemEventKind.PATCH_PROPOSED in kinds
    assert SystemEventKind.PATCH_DECISION in kinds


def test_healthy_stats_produces_no_proposals() -> None:
    stats = (_healthy_stats(),)
    loop = _make_loop(policy=_unfrozen_policy(), stats=stats)
    result = loop.tick(ts_ns=10_000_000_000)
    assert result.frozen is False
    assert result.proposals == ()
    assert result.runs == ()
    assert result.emitted_events == ()


def test_empty_stats_unfrozen_is_noop_emit() -> None:
    loop = _make_loop(policy=_unfrozen_policy(), stats=())
    result = loop.tick(ts_ns=42)
    assert result.frozen is False
    assert result.drained_stats == ()
    assert result.proposals == ()
    assert result.runs == ()
    assert result.emitted_events == ()


# ---------------------------------------------------------------------------
# Mode flip mid-loop
# ---------------------------------------------------------------------------


def test_mode_flip_mid_loop() -> None:
    stats_batches: list[tuple[StrategyStats, ...]] = [
        (_breaching_stats(),),
        (_breaching_stats(strategy_id="beta"),),
        (_breaching_stats(strategy_id="gamma"),),
    ]
    policies: list[LearningEvolutionFreezePolicy] = [
        _frozen_policy(),
        _unfrozen_policy(),
        _frozen_policy(),
    ]
    iter_pol = iter(policies)
    iter_stats = iter(stats_batches)
    pipeline = PatchPipeline()
    bridge = PatchApprovalBridge(pipeline=pipeline)
    orchestrator = PatchPipelineOrchestrator(bridge=bridge)
    loop = StructuralEvolutionLoop(
        proposer=MutationProposer(
            thresholds=MutationThresholds(min_trades=10, min_win_rate=0.5, min_mean_pnl=0.0)
        ),
        orchestrator=orchestrator,
        policy_supplier=lambda: next(iter_pol),
        stats_supplier=lambda: next(iter_stats),
        evidence_builder=_clean_evidence,
    )
    r1 = loop.tick(ts_ns=10_000_000_000)
    assert r1.frozen is True
    assert r1.proposals == ()
    r2 = loop.tick(ts_ns=20_000_000_000)
    assert r2.frozen is False
    assert len(r2.proposals) >= 1
    r3 = loop.tick(ts_ns=30_000_000_000)
    assert r3.frozen is True
    assert r3.proposals == ()


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def _run_one_tick() -> StructuralLoopTickResult:
    loop = _make_loop(
        policy=_unfrozen_policy(),
        stats=(_breaching_stats(),),
    )
    return loop.tick(ts_ns=10_000_000_000)


def test_byte_identical_replay() -> None:
    a = _run_one_tick()
    b = _run_one_tick()
    c = _run_one_tick()
    # PatchProposal patch_ids embed an internal counter; the counter
    # is fresh per loop instance so two distinct loop runs produce
    # identical proposal IDs. Compare structurally.
    assert a.proposals == b.proposals == c.proposals
    assert a.emitted_events == b.emitted_events == c.emitted_events
    assert tuple(r.decision.decision for r in a.runs) == tuple(r.decision.decision for r in b.runs)


# ---------------------------------------------------------------------------
# B27 / B28 / INV-71 — AST scan
# ---------------------------------------------------------------------------


def _module_ast() -> ast.AST:
    return ast.parse(_MODULE_PATH.read_text())


def test_loop_module_does_not_directly_construct_typed_events() -> None:
    """Typed events come from the proposer + orchestrator the loop owns,
    never from the loop module itself."""
    tree = _module_ast()
    bad_constructors = {
        "PatchProposal",
        "GovernanceDecision",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else None)
            )
            assert name not in bad_constructors, f"structural_loop must not construct {name}"


def test_loop_module_does_not_import_runtime_engines() -> None:
    tree = _module_ast()
    forbidden_prefixes = (
        "execution_engine.",
        "system_engine.",
        "intelligence_engine.",
        "learning_engine.",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith(forbidden_prefixes), (
                f"structural_loop must not import {node.module}"
            )
