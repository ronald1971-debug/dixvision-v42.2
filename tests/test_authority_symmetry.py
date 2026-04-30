"""HARDEN-06 — authority symmetry invariant tests (INV-71).

The Triad Lock is a closed set of authority claims:

* :class:`ExecutionIntent` — only ``intelligence_engine.*`` /
  ``governance_engine.*`` may construct (B25 + runtime origin allowlist).
* :class:`SignalEvent` — only ``intelligence_engine.*`` may construct
  (B22 lint).
* :class:`ExecutionEvent` — only ``execution_engine.*`` may construct
  (B21 lint).
* :class:`HazardEvent` — only ``system_engine.*`` (or the Execution
  Gate, for ``HAZ-AUTHORITY``) may construct (HARDEN-03 producer set).

This module asserts the **symmetric** half of the contract that the
verdict flagged as exposed:

* :class:`LearningUpdate` — only ``learning_engine.*`` may construct
  (B27).
* :class:`PatchProposal` — only ``evolution_engine.*`` may construct
  (B28).

Plus a runtime invariant-breach probe for every authority surface that
must hard-fail on misuse:

* ``system_engine`` cannot construct an :class:`ExecutionIntent` even
  with a forged origin.
* A :class:`SignalEvent` carrying a producer string outside its
  registered set raises via :func:`assert_event_provenance`.
* :func:`mark_approved` cannot synthesise approval state without a
  ``governance_decision_id``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.contracts.event_provenance import (
    EVENT_PRODUCERS,
    EventProvenanceError,
    assert_event_provenance,
)
from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
)
from core.contracts.execution_intent import (
    UnauthorizedOriginError,
    create_execution_intent,
    mark_approved,
)
from core.contracts.learning import LearningUpdate, PatchProposal

# ---------------------------------------------------------------------------
# B27 — LearningUpdate construction restriction
# ---------------------------------------------------------------------------


def _run_b27(source: str, importer: str) -> list:
    from tools.authority_lint import _check_b27  # type: ignore

    repo_root = Path(__file__).resolve().parent.parent
    file = repo_root / "evolution_engine" / "_synthetic_b27_fixture.py"
    tree = ast.parse(source)
    return _check_b27(importer, file, repo_root, tree)


def test_b27_blocks_evolution_engine_constructing_learning_update():
    src = (
        "from core.contracts.learning import LearningUpdate\n"
        "LearningUpdate(ts_ns=1, strategy_id='s', parameter='p',"
        " old_value='', new_value='', reason='')\n"
    )
    violations = _run_b27(src, "evolution_engine.intelligence_loops")
    assert [v.rule for v in violations] == ["B27"]


def test_b27_blocks_intelligence_engine_constructing_learning_update():
    src = "LearningUpdate(ts_ns=1, strategy_id='s', parameter='p')\n"
    violations = _run_b27(src, "intelligence_engine.signal_pipeline")
    assert any(v.rule == "B27" for v in violations)


def test_b27_blocks_execution_engine_constructing_learning_update():
    src = "LearningUpdate(ts_ns=1, strategy_id='s', parameter='p')\n"
    violations = _run_b27(src, "execution_engine.hot_path.fast_execute")
    assert any(v.rule == "B27" for v in violations)


def test_b27_blocks_system_engine_constructing_learning_update():
    src = "LearningUpdate(ts_ns=1, strategy_id='s', parameter='p')\n"
    violations = _run_b27(src, "system_engine.dyon.health")
    assert any(v.rule == "B27" for v in violations)


def test_b27_allows_learning_engine_constructing_learning_update():
    src = "LearningUpdate(ts_ns=1, strategy_id='s', parameter='p')\n"
    violations = _run_b27(src, "learning_engine.update_emitter")
    assert violations == []


def test_b27_allows_contract_module():
    src = "LearningUpdate(ts_ns=1, strategy_id='s', parameter='p')\n"
    violations = _run_b27(src, "core.contracts.learning")
    assert violations == []


# ---------------------------------------------------------------------------
# B28 — PatchProposal construction restriction
# ---------------------------------------------------------------------------


def _run_b28(source: str, importer: str) -> list:
    from tools.authority_lint import _check_b28  # type: ignore

    repo_root = Path(__file__).resolve().parent.parent
    file = repo_root / "learning_engine" / "_synthetic_b28_fixture.py"
    tree = ast.parse(source)
    return _check_b28(importer, file, repo_root, tree)


def test_b28_blocks_learning_engine_constructing_patch_proposal():
    src = (
        "from core.contracts.learning import PatchProposal\n"
        "PatchProposal(ts_ns=1, patch_id='p', source='s',"
        " target_strategy='t', touchpoints=(), rationale='')\n"
    )
    violations = _run_b28(src, "learning_engine.lanes.weight_adjuster")
    assert [v.rule for v in violations] == ["B28"]


def test_b28_blocks_intelligence_engine_constructing_patch_proposal():
    src = "PatchProposal(ts_ns=1, patch_id='p', source='s')\n"
    violations = _run_b28(src, "intelligence_engine.meta_controller")
    assert any(v.rule == "B28" for v in violations)


def test_b28_blocks_governance_engine_constructing_patch_proposal():
    src = "PatchProposal(ts_ns=1, patch_id='p', source='s')\n"
    violations = _run_b28(src, "governance_engine.policy_engine")
    assert any(v.rule == "B28" for v in violations)


def test_b28_blocks_system_engine_constructing_patch_proposal():
    src = "PatchProposal(ts_ns=1, patch_id='p', source='s')\n"
    violations = _run_b28(src, "system_engine.dyon.patch_pipeline_shim")
    assert any(v.rule == "B28" for v in violations)


def test_b28_allows_evolution_engine_constructing_patch_proposal():
    src = "PatchProposal(ts_ns=1, patch_id='p', source='s')\n"
    violations = _run_b28(src, "evolution_engine.patch_pipeline.events")
    assert violations == []


def test_b28_allows_contract_module():
    src = "PatchProposal(ts_ns=1, patch_id='p', source='s')\n"
    violations = _run_b28(src, "core.contracts.learning")
    assert violations == []


# ---------------------------------------------------------------------------
# B29 — TraderObservation construction restriction (Wave-04 PR-1)
# ---------------------------------------------------------------------------


def _run_b29(source: str, importer: str) -> list:
    from tools.authority_lint import _check_b29  # type: ignore

    repo_root = Path(__file__).resolve().parent.parent
    file = repo_root / "learning_engine" / "_synthetic_b29_fixture.py"
    tree = ast.parse(source)
    return _check_b29(importer, file, repo_root, tree)


def test_b29_blocks_learning_engine_constructing_trader_observation():
    src = (
        "from core.contracts.trader_intelligence import "
        "TraderObservation, TraderModel\n"
        "TraderObservation(ts_ns=1, trader_id='t', "
        "observation_kind='PROFILE_UPDATE', "
        "model=TraderModel(trader_id='t', source_feed='SRC-A'))\n"
    )
    violations = _run_b29(src, "learning_engine.trader_aggregator")
    assert [v.rule for v in violations] == ["B29"]


def test_b29_blocks_evolution_engine_constructing_trader_observation():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "evolution_engine.trader_compose")
    assert any(v.rule == "B29" for v in violations)


def test_b29_blocks_intelligence_engine_root_constructing_trader_observation():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "intelligence_engine.signal_pipeline")
    assert any(v.rule == "B29" for v in violations)


def test_b29_blocks_execution_engine_constructing_trader_observation():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "execution_engine.hot_path.fast_execute")
    assert any(v.rule == "B29" for v in violations)


def test_b29_blocks_governance_engine_constructing_trader_observation():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "governance_engine.policy_engine")
    assert any(v.rule == "B29" for v in violations)


def test_b29_blocks_system_engine_constructing_trader_observation():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "system_engine.dyon.health")
    assert any(v.rule == "B29" for v in violations)


def test_b29_allows_trader_modeling_subsystem():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "intelligence_engine.trader_modeling.aggregator")
    assert violations == []


def test_b29_allows_trader_modeling_root():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "intelligence_engine.trader_modeling")
    assert violations == []


def test_b29_allows_contract_module():
    src = "TraderObservation(ts_ns=1, trader_id='t')\n"
    violations = _run_b29(src, "core.contracts.trader_intelligence")
    assert violations == []


# ---------------------------------------------------------------------------
# Production code is clean — none of B27 / B28 / B29 fire when the lint
# runs over the repo as a whole. Regression guard so a future commit
# cannot accidentally regress any of the symmetric authority rules.
# ---------------------------------------------------------------------------


def test_b27_b28_b29_clean_on_repo():
    from tools.authority_lint import lint_repo

    repo_root = Path(__file__).resolve().parent.parent
    violations = [
        v
        for v in lint_repo(repo_root)
        if v.rule in {"B27", "B28", "B29"}
    ]
    assert violations == [], f"unexpected B27/B28/B29 violations: {violations}"


# ---------------------------------------------------------------------------
# Invariant-breach probes — runtime defences hard-fail on misuse.
# ---------------------------------------------------------------------------


def _signal(produced_by_engine: str = "intelligence_engine") -> SignalEvent:
    return SignalEvent(
        ts_ns=1,
        symbol="EURUSD",
        side=Side.BUY,
        confidence=0.5,
        produced_by_engine=produced_by_engine,
    )


def test_system_engine_cannot_forge_execution_intent_origin():
    """``system_engine`` is not in :data:`AUTHORISED_INTENT_ORIGINS`.

    B25 catches this at lint time; this test pins the runtime defence
    so a missed lint cannot escalate into a real intent.
    """

    with pytest.raises(UnauthorizedOriginError):
        create_execution_intent(
            ts_ns=1,
            origin="system_engine.dyon.controller",
            signal=_signal(),
        )


def test_evolution_engine_cannot_forge_execution_intent_origin():
    with pytest.raises(UnauthorizedOriginError):
        create_execution_intent(
            ts_ns=1,
            origin="evolution_engine.patch_pipeline",
            signal=_signal(),
        )


def test_learning_engine_cannot_forge_execution_intent_origin():
    with pytest.raises(UnauthorizedOriginError):
        create_execution_intent(
            ts_ns=1,
            origin="learning_engine.update_emitter",
            signal=_signal(),
        )


def test_governance_engine_cannot_forge_execution_intent_origin():
    """Governance approves intents; it does not author them.

    B25 *imports* are allowed for ``governance_engine.*`` (so the
    harness shim can call :func:`mark_approved`), but the runtime
    origin allowlist still rejects ``governance_engine.*`` as a source.
    Authority symmetry: governance is the approver, not the proposer.
    """

    with pytest.raises(UnauthorizedOriginError):
        create_execution_intent(
            ts_ns=1,
            origin="governance_engine.policy_engine",
            signal=_signal(),
        )


def test_forged_signal_provenance_rejected_at_runtime():
    """A SignalEvent stamped with a non-intelligence producer raises."""

    forged = SignalEvent(
        ts_ns=2,
        symbol="EURUSD",
        side=Side.BUY,
        confidence=0.5,
        produced_by_engine="execution_engine",
    )
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(forged)


def test_forged_execution_provenance_rejected_at_runtime():
    """An ExecutionEvent stamped with a non-execution producer raises."""

    forged = ExecutionEvent(
        ts_ns=3,
        symbol="EURUSD",
        side=Side.BUY,
        price=1.1,
        qty=1.0,
        status=ExecutionStatus.FILLED,
        venue="paper",
        order_id="X",
        produced_by_engine="intelligence_engine",
    )
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(forged)


def test_forged_hazard_provenance_rejected_at_runtime():
    forged = HazardEvent(
        ts_ns=4,
        code="HAZ-FAKE",
        severity=HazardSeverity.LOW,
        source="system",
        produced_by_engine="learning_engine",
    )
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(forged)


def test_event_producers_registry_is_frozen_view_of_authority():
    """Pin the authority symmetry:

    * SignalEvent producers ⊆ {intelligence_engine, intelligence_engine.cognitive}
    * ExecutionEvent producers ⊆ {execution_engine}
    * HazardEvent producers ⊆ {system_engine, execution_engine}

    Adding a producer is a deliberate Triad Lock change; this test
    catches accidental loosening.
    """

    assert EVENT_PRODUCERS[SignalEvent] == frozenset(
        {"intelligence_engine", "intelligence_engine.cognitive"}
    )
    assert EVENT_PRODUCERS[ExecutionEvent] == frozenset({"execution_engine"})
    assert EVENT_PRODUCERS[HazardEvent] == frozenset(
        {"system_engine", "execution_engine"}
    )


def test_mark_approved_cannot_synthesise_approval_without_decision_id():
    proposed = create_execution_intent(
        ts_ns=5, origin="tests.fixtures", signal=_signal()
    )
    with pytest.raises(ValueError):
        mark_approved(proposed, governance_decision_id="")


# ---------------------------------------------------------------------------
# Schema-level symmetry — every contract dataclass we just gated has
# the expected field surface, so a future field rename can't silently
# break the lint pattern.
# ---------------------------------------------------------------------------


def test_learning_update_schema_unchanged():
    """LearningUpdate is the carrier B27 protects."""

    upd = LearningUpdate(
        ts_ns=10,
        strategy_id="s",
        parameter="p",
        old_value="0.1",
        new_value="0.2",
        reason="closed-loop",
    )
    assert upd.ts_ns == 10
    assert upd.strategy_id == "s"
    assert upd.parameter == "p"


def test_patch_proposal_schema_unchanged():
    """PatchProposal is the carrier B28 protects."""

    prop = PatchProposal(
        ts_ns=11,
        patch_id="p",
        source="src",
        target_strategy="t",
        touchpoints=("a", "b"),
        rationale="rat",
    )
    assert prop.ts_ns == 11
    assert prop.touchpoints == ("a", "b")
