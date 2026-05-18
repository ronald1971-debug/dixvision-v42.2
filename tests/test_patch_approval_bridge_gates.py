"""P0-B — integration tests for the promotion-gate wiring on the bridge.

These tests advance a real :class:`PatchPipeline` through its FSM stages
up to ``CANARY`` and then exercise :meth:`PatchApprovalBridge.approve`
with the two promotion-gate evaluators wired in. The bridge is the
single authority that constructs :class:`PatchApprovalDecision` (B27 /
B28 / INV-71) — the evaluators only produce advisory verdicts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.constraint_engine import compile_rules
from core.contracts.learning import PatchProposal
from core.contracts.patch import PatchStage
from evolution_engine.patch_pipeline.pipeline import PatchPipeline
from governance_engine.gates import (
    QuantitativeEvaluator,
    QuantitativeMetrics,
    RuleGraphPatchEvaluator,
)
from governance_engine.services.patch_pipeline_bridge import (
    PatchApprovalBridge,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "registry" / "constraint_rules.yaml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _advance_to_canary(bridge: PatchApprovalBridge, *, patch_id: str, t0: int) -> None:
    """Drive a fresh patch through PROPOSED → … → CANARY."""

    bridge.receive_proposal(
        PatchProposal(
            ts_ns=t0,
            patch_id=patch_id,
            source="evolution_engine.cmaes",
            target_strategy="strat-x",
            touchpoints=("strat-x.param",),
            rationale="unit-test",
        )
    )
    for i, stage in enumerate(
        (
            PatchStage.SANDBOX,
            PatchStage.STATIC_ANALYSIS,
            PatchStage.BACKTEST,
            PatchStage.SHADOW,
            PatchStage.CANARY,
        ),
        start=1,
    ):
        bridge.advance(
            patch_id=patch_id,
            new_stage=stage,
            ts_ns=t0 + i,
        )


def _passing_metrics() -> QuantitativeMetrics:
    return QuantitativeMetrics(
        sharpe_ratio=1.5,
        max_drawdown=0.03,
        samples=250,
        is_score=0.10,
        oos_score=0.10,
        is_std=0.05,
    )


def _failing_metrics() -> QuantitativeMetrics:
    return QuantitativeMetrics(
        sharpe_ratio=0.3,
        max_drawdown=0.20,
        samples=250,
        is_score=0.10,
        oos_score=0.10,
        is_std=0.05,
    )


def _proposal(patch_id: str) -> PatchProposal:
    return PatchProposal(
        ts_ns=1_700_000_000_000_000_000,
        patch_id=patch_id,
        source="evolution_engine.cmaes",
        target_strategy="strat-x",
        touchpoints=("strat-x.param",),
        rationale="unit-test",
    )


@pytest.fixture(scope="module")
def rulegraph_evaluator() -> RuleGraphPatchEvaluator:
    return RuleGraphPatchEvaluator(rule_graph=compile_rules(RULES_PATH))


# ---------------------------------------------------------------------------
# Backwards compatibility: bridge without evaluators behaves as before.
# ---------------------------------------------------------------------------


def test_bridge_without_evaluators_approves_at_canary_unchanged() -> None:
    bridge = PatchApprovalBridge(pipeline=PatchPipeline())
    _advance_to_canary(bridge, patch_id="patch-bc-1", t0=10)
    dec = bridge.approve(patch_id="patch-bc-1", ts_ns=100)
    assert dec.decision == "APPROVED"
    assert dec.final_stage is PatchStage.APPROVED
    assert dec.meta == {}


# ---------------------------------------------------------------------------
# Quantitative-only gate
# ---------------------------------------------------------------------------


def test_quantitative_gate_approves_passing_metrics() -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        quantitative_evaluator=QuantitativeEvaluator(),
    )
    _advance_to_canary(bridge, patch_id="patch-q-1", t0=10)
    dec = bridge.approve(
        patch_id="patch-q-1",
        ts_ns=100,
        metrics=_passing_metrics(),
    )
    assert dec.decision == "APPROVED"
    assert dec.final_stage is PatchStage.APPROVED


def test_quantitative_gate_rejects_failing_metrics_and_surfaces_codes() -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        quantitative_evaluator=QuantitativeEvaluator(),
    )
    _advance_to_canary(bridge, patch_id="patch-q-2", t0=10)
    dec = bridge.approve(
        patch_id="patch-q-2",
        ts_ns=100,
        metrics=_failing_metrics(),
    )
    assert dec.decision == "REJECTED"
    assert dec.final_stage is PatchStage.REJECTED
    codes = dec.meta["gate_rejection_codes"].split(",")
    assert "QUANT_SHARPE_BELOW_FLOOR" in codes
    assert "QUANT_DRAWDOWN_EXCEEDS_CEILING" in codes


def test_quantitative_gate_raises_when_metrics_missing() -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        quantitative_evaluator=QuantitativeEvaluator(),
    )
    _advance_to_canary(bridge, patch_id="patch-q-3", t0=10)
    with pytest.raises(ValueError):
        bridge.approve(patch_id="patch-q-3", ts_ns=100)


# ---------------------------------------------------------------------------
# RuleGraph gate
# ---------------------------------------------------------------------------


def test_rulegraph_gate_approves_when_no_rules_fire(
    rulegraph_evaluator: RuleGraphPatchEvaluator,
) -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        rulegraph_evaluator=rulegraph_evaluator,
    )
    _advance_to_canary(bridge, patch_id="patch-r-1", t0=10)
    dec = bridge.approve(
        patch_id="patch-r-1",
        ts_ns=100,
        proposal=_proposal("patch-r-1"),
        metrics=_passing_metrics(),
    )
    assert dec.decision == "APPROVED"


def test_rulegraph_gate_rejects_when_gov_patch_rule_fires(
    rulegraph_evaluator: RuleGraphPatchEvaluator,
) -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        rulegraph_evaluator=rulegraph_evaluator,
    )
    _advance_to_canary(bridge, patch_id="patch-r-2", t0=10)
    dec = bridge.approve(
        patch_id="patch-r-2",
        ts_ns=100,
        proposal=_proposal("patch-r-2"),
        metrics=_failing_metrics(),
    )
    assert dec.decision == "REJECTED"
    assert dec.final_stage is PatchStage.REJECTED
    codes = dec.meta["gate_rejection_codes"].split(",")
    assert any(code.startswith("GOV-PATCH-") for code in codes)


def test_rulegraph_gate_raises_when_proposal_missing(
    rulegraph_evaluator: RuleGraphPatchEvaluator,
) -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        rulegraph_evaluator=rulegraph_evaluator,
    )
    _advance_to_canary(bridge, patch_id="patch-r-3", t0=10)
    with pytest.raises(ValueError):
        bridge.approve(
            patch_id="patch-r-3",
            ts_ns=100,
            metrics=_passing_metrics(),
        )


# ---------------------------------------------------------------------------
# Both gates wired
# ---------------------------------------------------------------------------


def test_both_gates_approve_when_passing(
    rulegraph_evaluator: RuleGraphPatchEvaluator,
) -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        quantitative_evaluator=QuantitativeEvaluator(),
        rulegraph_evaluator=rulegraph_evaluator,
    )
    _advance_to_canary(bridge, patch_id="patch-b-1", t0=10)
    dec = bridge.approve(
        patch_id="patch-b-1",
        ts_ns=100,
        proposal=_proposal("patch-b-1"),
        metrics=_passing_metrics(),
    )
    assert dec.decision == "APPROVED"


def test_both_gates_surface_both_rejection_sources_when_failing(
    rulegraph_evaluator: RuleGraphPatchEvaluator,
) -> None:
    bridge = PatchApprovalBridge(
        pipeline=PatchPipeline(),
        quantitative_evaluator=QuantitativeEvaluator(),
        rulegraph_evaluator=rulegraph_evaluator,
    )
    _advance_to_canary(bridge, patch_id="patch-b-2", t0=10)
    dec = bridge.approve(
        patch_id="patch-b-2",
        ts_ns=100,
        proposal=_proposal("patch-b-2"),
        metrics=_failing_metrics(),
    )
    assert dec.decision == "REJECTED"
    codes = dec.meta["gate_rejection_codes"].split(",")
    # Quantitative codes (QUANT_*) and rule-graph codes (GOV-PATCH-*)
    # MUST both appear when both gates are wired and both block.
    assert any(c.startswith("QUANT_") for c in codes)
    assert any(c.startswith("GOV-PATCH-") for c in codes)
    detail = dec.meta["gate_detail"]
    assert "quantitative=" in detail
    assert "rulegraph=" in detail


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_gate_rejection_replay_is_byte_identical(
    rulegraph_evaluator: RuleGraphPatchEvaluator,
) -> None:
    def _run() -> dict[str, str]:
        bridge = PatchApprovalBridge(
            pipeline=PatchPipeline(),
            quantitative_evaluator=QuantitativeEvaluator(),
            rulegraph_evaluator=rulegraph_evaluator,
        )
        _advance_to_canary(bridge, patch_id="patch-d-1", t0=10)
        dec = bridge.approve(
            patch_id="patch-d-1",
            ts_ns=100,
            proposal=_proposal("patch-d-1"),
            metrics=_failing_metrics(),
        )
        return dict(dec.meta)

    run1 = _run()
    run2 = _run()
    run3 = _run()
    assert run1 == run2 == run3
