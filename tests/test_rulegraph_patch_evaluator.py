"""P0-B — unit tests for the RuleGraph-backed patch evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.constraint_engine import compile_rules
from core.contracts.learning import PatchProposal
from governance_engine.gates.quantitative_evaluator import QuantitativeMetrics
from governance_engine.gates.rulegraph_patch_evaluator import (
    PatchEvaluationFacts,
    RuleGraphPatchEvaluator,
    RuleGraphPatchVerdictKind,
    build_patch_facts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "registry" / "constraint_rules.yaml"


@pytest.fixture(scope="module")
def evaluator() -> RuleGraphPatchEvaluator:
    rg = compile_rules(RULES_PATH)
    return RuleGraphPatchEvaluator(rule_graph=rg)


def _facts(**overrides: object) -> PatchEvaluationFacts:
    defaults: dict[str, object] = {
        "patch_id": "patch-001",
        "source": "evolution_engine.cmaes",
        "target_strategy": "strat-x",
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.03,
        "samples": 250,
        "is_oos_divergence_sigma": 0.1,
        "sharpe_ratio_min": 1.0,
        "max_drawdown_max": 0.05,
        "samples_min": 200,
        "is_oos_divergence_max_sigma": 0.5,
    }
    defaults.update(overrides)
    return PatchEvaluationFacts(**defaults)  # type: ignore[arg-type]


def _proposal(patch_id: str = "patch-001") -> PatchProposal:
    return PatchProposal(
        ts_ns=1_700_000_000_000_000_000,
        patch_id=patch_id,
        source="evolution_engine.cmaes",
        target_strategy="strat-x",
        touchpoints=("strat-x.param",),
        rationale="unit-test",
    )


# ---------------------------------------------------------------------------
# Passing path
# ---------------------------------------------------------------------------


def test_evaluator_approves_when_no_gov_patch_rule_fires(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    v = evaluator.evaluate(_facts())
    assert v.kind is RuleGraphPatchVerdictKind.APPROVED
    assert v.passed is True
    assert v.blocking_rule_ids == ()


def test_evaluator_filters_unrelated_fired_rules(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    """Unrelated rules firing on the shared fact mapping must not block."""

    v = evaluator.evaluate(_facts())
    # Any non-GOV-PATCH rule firing is fine; the evaluator must only
    # surface IDs that begin with the prefix.
    for rid in v.fired_rule_ids:
        assert rid.startswith("GOV-PATCH-")


# ---------------------------------------------------------------------------
# Rejection paths — one rule each
# ---------------------------------------------------------------------------


def test_gov_patch_sharpe_rule_fires_when_below_floor(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    v = evaluator.evaluate(_facts(sharpe_ratio=0.5))
    assert v.kind is RuleGraphPatchVerdictKind.REJECTED
    assert v.passed is False
    assert "GOV-PATCH-SHARPE" in v.blocking_rule_ids


def test_gov_patch_drawdown_rule_fires_above_ceiling(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    v = evaluator.evaluate(_facts(max_drawdown=0.10))
    assert v.kind is RuleGraphPatchVerdictKind.REJECTED
    assert "GOV-PATCH-DRAWDOWN" in v.blocking_rule_ids


def test_gov_patch_samples_rule_fires_below_floor(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    v = evaluator.evaluate(_facts(samples=50))
    assert v.kind is RuleGraphPatchVerdictKind.REJECTED
    assert "GOV-PATCH-SAMPLES" in v.blocking_rule_ids


def test_gov_patch_divergence_rule_fires_above_sigma_ceiling(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    v = evaluator.evaluate(_facts(is_oos_divergence_sigma=1.0))
    assert v.kind is RuleGraphPatchVerdictKind.REJECTED
    assert "GOV-PATCH-IS-OOS-DIVERGENCE" in v.blocking_rule_ids


def test_multiple_rules_fire_at_once_blocking_ids_sorted(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    v = evaluator.evaluate(_facts(sharpe_ratio=0.1, max_drawdown=0.20, samples=10))
    assert v.passed is False
    assert list(v.blocking_rule_ids) == sorted(v.blocking_rule_ids)
    assert "GOV-PATCH-SHARPE" in v.blocking_rule_ids
    assert "GOV-PATCH-DRAWDOWN" in v.blocking_rule_ids
    assert "GOV-PATCH-SAMPLES" in v.blocking_rule_ids


# ---------------------------------------------------------------------------
# build_patch_facts
# ---------------------------------------------------------------------------


def test_build_patch_facts_normalises_divergence_by_sigma() -> None:
    facts = build_patch_facts(
        proposal=_proposal(),
        metrics=QuantitativeMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.03,
            samples=250,
            is_score=0.25,
            oos_score=0.20,
            is_std=0.05,
        ),
        sharpe_ratio_min=1.0,
        max_drawdown_max=0.05,
        samples_min=200,
        is_oos_divergence_max_sigma=0.5,
    )
    # |0.25 - 0.20| / 0.05 == 1.0σ.
    assert facts.is_oos_divergence_sigma == pytest.approx(1.0)
    assert facts.patch_id == "patch-001"
    assert facts.target_strategy == "strat-x"


def test_build_patch_facts_falls_back_to_absolute_when_is_std_zero() -> None:
    facts = build_patch_facts(
        proposal=_proposal(),
        metrics=QuantitativeMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.03,
            samples=250,
            is_score=0.30,
            oos_score=0.20,
            is_std=0.0,
        ),
        sharpe_ratio_min=1.0,
        max_drawdown_max=0.05,
        samples_min=200,
        is_oos_divergence_max_sigma=0.5,
    )
    # is_std==0 → raw |0.30 - 0.20| == 0.10.
    assert facts.is_oos_divergence_sigma == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Determinism + mapping API
# ---------------------------------------------------------------------------


def test_facts_as_mapping_is_alphabetically_sorted_for_replay() -> None:
    facts = _facts()
    keys = list(facts.as_mapping().keys())
    assert keys == sorted(keys)


def test_evaluator_replay_is_byte_identical(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    facts = _facts(sharpe_ratio=0.5, max_drawdown=0.10)
    v1 = evaluator.evaluate(facts)
    v2 = evaluator.evaluate(facts)
    v3 = evaluator.evaluate(facts)
    assert v1 == v2 == v3


def test_evaluator_accepts_raw_mapping_in_addition_to_facts(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    facts = _facts(sharpe_ratio=0.5)
    v_from_facts = evaluator.evaluate(facts)
    v_from_mapping = evaluator.evaluate(facts.as_mapping())
    assert v_from_facts == v_from_mapping


def test_custom_rule_id_prefix_isolates_other_invariants(
    evaluator: RuleGraphPatchEvaluator,
) -> None:
    # Build a second evaluator that filters by a never-firing prefix —
    # GOV-PATCH-* rules must then be ignored and the verdict approve.
    rg = evaluator.rule_graph
    narrow = RuleGraphPatchEvaluator(rule_graph=rg, rule_id_prefix="NOT-A-REAL-PREFIX-")
    v = narrow.evaluate(_facts(sharpe_ratio=0.1, max_drawdown=0.9))
    assert v.kind is RuleGraphPatchVerdictKind.APPROVED
    assert v.passed is True
    assert v.fired_rule_ids == ()
    assert v.blocking_rule_ids == ()
