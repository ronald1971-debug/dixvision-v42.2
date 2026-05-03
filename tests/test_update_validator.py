"""Tests for Wave-04.6 PR-E — UpdateValidator + UpdateApplier."""

from __future__ import annotations

import pytest

from core.contracts.events import SystemEvent, SystemEventKind
from core.contracts.governance import SystemMode
from core.contracts.strategy_registry import (
    StrategyLifecycle,
    StrategyLifecycleError,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.update_applier import UpdateApplier
from governance_engine.control_plane.update_validator import (
    ProposedUpdate,
    UpdateDecision,
    UpdateRejectCode,
    UpdateValidator,
    UpdateVerdict,
)
from governance_engine.engine import GovernanceEngine
from governance_engine.strategy_registry import (
    LEDGER_KIND_STRATEGY_PARAMETER_UPDATE,
    StrategyRegistry,
)


def _approved_strategy(
    *,
    strategy_id: str = "s1",
    mutable: tuple[str, ...] = ("alpha",),
    bounds: dict[str, tuple[float, float]] | None = None,
    parameters: dict[str, str] | None = None,
) -> StrategyRegistry:
    """Build a registry with one APPROVED strategy ready for updates."""
    reg = StrategyRegistry(ledger=LedgerAuthorityWriter())
    reg.register_draft(
        strategy_id=strategy_id,
        ts_ns=1,
        parameters=parameters or {"alpha": "0.5"},
        mutable_parameters=mutable,
        parameter_bounds=bounds or {},
    )
    reg.transition(
        strategy_id=strategy_id,
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="bootstrap",
    )
    reg.transition(
        strategy_id=strategy_id,
        new_lifecycle=StrategyLifecycle.APPROVED,
        ts_ns=3,
        reason="bootstrap",
    )
    return reg


def _learning_update(**overrides) -> ProposedUpdate:
    base = {
        "ts_ns": 100,
        "strategy_id": "s1",
        "parameter": "alpha",
        "old_value": "0.5",
        "new_value": "0.6",
        "reason": "win-rate-up",
    }
    base.update(overrides)
    return ProposedUpdate(**base)


# ---------------------------------------------------------------------------
# Validator — rule-by-rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.SHADOW,
        SystemMode.LOCKED,
    ],
)
def test_validator_rejects_when_mode_disables_learning(mode):
    reg = _approved_strategy()
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(), mode=mode
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.MODE_LEARNING_DISABLED


@pytest.mark.parametrize(
    "mode", [SystemMode.CANARY, SystemMode.LIVE, SystemMode.AUTO]
)
def test_validator_admits_modes_that_enable_learning(mode):
    reg = _approved_strategy(
        bounds={"alpha": (0.1, 0.9)},
    )
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(), mode=mode
    )
    assert decision.verdict is UpdateVerdict.RATIFY
    assert decision.code is None


def test_validator_rejects_unknown_strategy():
    reg = StrategyRegistry(ledger=LedgerAuthorityWriter())
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(), mode=SystemMode.LIVE
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.UNKNOWN_STRATEGY


def test_validator_rejects_empty_reason_before_apply():
    """Empty reason rejected up-front — guards INV-15.

    StrategyRegistry.apply_parameter_update raises ValueError on empty
    reason. If the validator returned RATIFY, the engine would write
    UPDATE_RATIFIED to the ledger and *then* the applier would raise,
    leaving an orphaned ledger row with no STRATEGY_PARAMETER_UPDATE
    follow-up. EMPTY_REASON closes that gap by rejecting before any
    ledger row is written.
    """
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(reason=""),
        mode=SystemMode.LIVE,
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.EMPTY_REASON


@pytest.mark.parametrize(
    "lifecycle",
    [
        StrategyLifecycle.DRAFT,
        StrategyLifecycle.VALIDATING,
        StrategyLifecycle.RETIRED,
    ],
)
def test_validator_rejects_non_approved_lifecycle(lifecycle):
    reg = StrategyRegistry(ledger=LedgerAuthorityWriter())
    reg.register_draft(
        strategy_id="s1",
        ts_ns=1,
        parameters={"alpha": "0.5"},
        mutable_parameters=("alpha",),
    )
    if lifecycle is StrategyLifecycle.VALIDATING:
        reg.transition(
            strategy_id="s1",
            new_lifecycle=StrategyLifecycle.VALIDATING,
            ts_ns=2,
            reason="x",
        )
    elif lifecycle is StrategyLifecycle.RETIRED:
        reg.transition(
            strategy_id="s1",
            new_lifecycle=StrategyLifecycle.RETIRED,
            ts_ns=2,
            reason="x",
        )
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(), mode=SystemMode.LIVE
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.LIFECYCLE_NOT_APPROVED


def test_validator_rejects_immutable_parameter():
    reg = _approved_strategy(mutable=("alpha",))
    update = _learning_update(parameter="beta")
    decision = UpdateValidator(registry=reg).validate(
        update=update, mode=SystemMode.LIVE
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.PARAMETER_NOT_MUTABLE


def test_validator_rejects_out_of_bounds_value():
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(new_value="1.5"),
        mode=SystemMode.LIVE,
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.NEW_VALUE_OUT_OF_BOUNDS


def test_validator_accepts_value_at_inclusive_bounds():
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    for value in ("0.1", "0.9"):
        decision = UpdateValidator(registry=reg).validate(
            update=_learning_update(new_value=value),
            mode=SystemMode.LIVE,
        )
        assert decision.verdict is UpdateVerdict.RATIFY


def test_validator_rejects_non_numeric_when_bounds_declared():
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(new_value="not-a-number"),
        mode=SystemMode.LIVE,
    )
    assert decision.verdict is UpdateVerdict.REJECT
    assert decision.code is UpdateRejectCode.NEW_VALUE_NOT_NUMERIC


def test_validator_admits_non_numeric_when_no_bounds():
    """Unbounded parameters accept arbitrary string values."""
    reg = _approved_strategy()  # no bounds for alpha
    decision = UpdateValidator(registry=reg).validate(
        update=_learning_update(new_value="aggressive"),
        mode=SystemMode.LIVE,
    )
    assert decision.verdict is UpdateVerdict.RATIFY


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


def test_applier_mutates_registry_on_ratify():
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    validator = UpdateValidator(registry=reg)
    applier = UpdateApplier(registry=reg)
    update = _learning_update(new_value="0.6")
    decision = validator.validate(update=update, mode=SystemMode.LIVE)
    record = applier.apply(decision=decision, update=update)
    assert record.parameters["alpha"] == "0.6"
    assert record.lifecycle is StrategyLifecycle.APPROVED
    assert record.version == 4  # 1 draft + 2 transitions + 1 update


def test_applier_refuses_reject_decisions():
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    applier = UpdateApplier(registry=reg)
    rejected = UpdateDecision(
        verdict=UpdateVerdict.REJECT,
        code=UpdateRejectCode.NEW_VALUE_OUT_OF_BOUNDS,
        detail="x",
    )
    with pytest.raises(ValueError):
        applier.apply(decision=rejected, update=_learning_update())


def test_apply_parameter_update_requires_approved_lifecycle():
    reg = StrategyRegistry(ledger=LedgerAuthorityWriter())
    reg.register_draft(
        strategy_id="s1",
        ts_ns=1,
        parameters={"alpha": "0.5"},
        mutable_parameters=("alpha",),
    )
    with pytest.raises(StrategyLifecycleError):
        reg.apply_parameter_update(
            strategy_id="s1",
            parameter="alpha",
            new_value="0.6",
            ts_ns=2,
            reason="x",
        )


def test_apply_parameter_update_requires_whitelisted_parameter():
    reg = _approved_strategy(mutable=("alpha",))
    with pytest.raises(StrategyLifecycleError):
        reg.apply_parameter_update(
            strategy_id="s1",
            parameter="beta",
            new_value="0.6",
            ts_ns=10,
            reason="x",
        )


def test_apply_parameter_update_writes_ledger_row():
    ledger = LedgerAuthorityWriter()
    reg = StrategyRegistry(ledger=ledger)
    reg.register_draft(
        strategy_id="s1",
        ts_ns=1,
        parameters={"alpha": "0.5"},
        mutable_parameters=("alpha",),
    )
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="x",
    )
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.APPROVED,
        ts_ns=3,
        reason="x",
    )
    reg.apply_parameter_update(
        strategy_id="s1",
        parameter="alpha",
        new_value="0.6",
        ts_ns=10,
        reason="winrate-up",
    )
    rows = ledger.read()
    update_rows = [
        r for r in rows if r.kind == LEDGER_KIND_STRATEGY_PARAMETER_UPDATE
    ]
    assert len(update_rows) == 1
    payload = update_rows[0].payload
    assert payload["strategy_id"] == "s1"
    assert payload["parameter"] == "alpha"
    assert payload["old_value"] == "0.5"
    assert payload["new_value"] == "0.6"
    assert payload["reason"] == "winrate-up"


# ---------------------------------------------------------------------------
# Replay determinism (lifecycle + parameter updates)
# ---------------------------------------------------------------------------


def test_replay_includes_parameter_updates():
    ledger = LedgerAuthorityWriter()
    reg = StrategyRegistry(ledger=ledger)
    reg.register_draft(
        strategy_id="s1",
        ts_ns=1,
        parameters={"alpha": "0.5"},
        mutable_parameters=("alpha",),
        parameter_bounds={"alpha": (0.1, 0.9)},
    )
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="x",
    )
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.APPROVED,
        ts_ns=3,
        reason="x",
    )
    reg.apply_parameter_update(
        strategy_id="s1",
        parameter="alpha",
        new_value="0.6",
        ts_ns=10,
        reason="winrate-up",
    )
    reg.apply_parameter_update(
        strategy_id="s1",
        parameter="alpha",
        new_value="0.7",
        ts_ns=20,
        reason="more confidence",
    )

    replay = StrategyRegistry(ledger=LedgerAuthorityWriter())
    replay.replay_from_ledger(ledger.read())
    assert replay.get("s1") == reg.get("s1")
    assert replay.get("s1").parameters["alpha"] == "0.7"


# ---------------------------------------------------------------------------
# GovernanceEngine wiring — closed loop
# ---------------------------------------------------------------------------


def _engine_with_registry() -> GovernanceEngine:
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    eng = GovernanceEngine(
        initial_mode=SystemMode.LIVE,
        strategy_registry=reg,
    )
    return eng


def test_governance_engine_legacy_audit_when_no_registry():
    """With no registry wired, a *complete* payload still falls back
    to the legacy ``UPDATE_PROPOSED_AUDIT`` row. Hardening-S1 item 4
    additionally requires that all five payload fields be present —
    a malformed payload now fails closed regardless of whether the
    registry is wired.
    """
    eng = GovernanceEngine(initial_mode=SystemMode.LIVE)
    event = SystemEvent(
        ts_ns=100,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={
            "strategy_id": "s1",
            "parameter": "alpha",
            "old_value": "0.5",
            "new_value": "0.6",
            "reason": "winrate-up",
        },
        meta={},
    )
    eng.process(event)
    rows = eng.ledger.read()
    audit_rows = [r for r in rows if r.kind == "UPDATE_PROPOSED_AUDIT"]
    assert len(audit_rows) == 1


def test_governance_engine_no_registry_malformed_payload_rejected():
    """Hardening-S1 item 4 — even on the no-registry legacy path, a
    payload missing required fields fails closed (UPDATE_REJECTED
    with code MALFORMED_PAYLOAD), not silently audit-logged.
    """
    eng = GovernanceEngine(initial_mode=SystemMode.LIVE)
    event = SystemEvent(
        ts_ns=101,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={"strategy_id": "s1", "parameter": "alpha"},  # 3 fields missing
        meta={},
    )
    eng.process(event)
    rows = eng.ledger.read()
    rejected = [r for r in rows if r.kind == "UPDATE_REJECTED"]
    audit_rows = [r for r in rows if r.kind == "UPDATE_PROPOSED_AUDIT"]
    assert len(rejected) == 1
    assert rejected[0].payload["code"] == "MALFORMED_PAYLOAD"
    assert audit_rows == []


def test_governance_engine_ratifies_and_applies_in_live():
    eng = _engine_with_registry()
    event = SystemEvent(
        ts_ns=100,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={
            "strategy_id": "s1",
            "parameter": "alpha",
            "old_value": "0.5",
            "new_value": "0.6",
            "reason": "winrate-up",
        },
        meta={},
    )
    eng.process(event)
    rows = eng.ledger.read()
    kinds = [r.kind for r in rows]
    assert "UPDATE_RATIFIED" in kinds
    assert LEDGER_KIND_STRATEGY_PARAMETER_UPDATE in kinds
    record = eng.strategy_registry.get("s1")
    assert record.parameters["alpha"] == "0.6"


def test_governance_engine_rejects_in_paper_mode():
    """PAPER mode must reject every learning update."""
    reg = _approved_strategy(bounds={"alpha": (0.1, 0.9)})
    eng = GovernanceEngine(
        initial_mode=SystemMode.PAPER,
        strategy_registry=reg,
    )
    event = SystemEvent(
        ts_ns=100,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={
            "strategy_id": "s1",
            "parameter": "alpha",
            "old_value": "0.5",
            "new_value": "0.6",
            "reason": "winrate-up",
        },
        meta={},
    )
    eng.process(event)
    rows = eng.ledger.read()
    rejected = [r for r in rows if r.kind == "UPDATE_REJECTED"]
    assert len(rejected) == 1
    assert (
        rejected[0].payload["code"]
        == UpdateRejectCode.MODE_LEARNING_DISABLED.value
    )
    # parameter must NOT have changed
    assert eng.strategy_registry.get("s1").parameters["alpha"] == "0.5"


def test_governance_engine_handles_malformed_payload_gracefully():
    eng = _engine_with_registry()
    event = SystemEvent(
        ts_ns=100,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={"strategy_id": "s1"},  # missing fields
        meta={},
    )
    eng.process(event)
    rows = eng.ledger.read()
    rejected = [r for r in rows if r.kind == "UPDATE_REJECTED"]
    assert len(rejected) == 1
    assert rejected[0].payload["code"] == "MALFORMED_PAYLOAD"


def test_golden_trace_paper_then_canary_then_live():
    """Reviewer #3's required golden trace.

    In PAPER mode the proposal is rejected; transitioning to CANARY
    and emitting another proposal ratifies it; same in LIVE.
    """
    reg = _approved_strategy(bounds={"alpha": (0.0, 1.0)})
    eng = GovernanceEngine(
        initial_mode=SystemMode.PAPER,
        strategy_registry=reg,
    )

    def propose(ts_ns: int, new_value: str) -> SystemEvent:
        return SystemEvent(
            ts_ns=ts_ns,
            sub_kind=SystemEventKind.UPDATE_PROPOSED,
            source="learning",
            payload={
                "strategy_id": "s1",
                "parameter": "alpha",
                "old_value": "0.5",
                "new_value": new_value,
                "reason": "trace",
            },
            meta={},
        )

    # 1) PAPER → reject
    eng.process(propose(100, "0.6"))
    assert eng.strategy_registry.get("s1").parameters["alpha"] == "0.5"

    # 2) Transition to CANARY (forward-chain via SAFE→PAPER→SHADOW→CANARY)
    from core.contracts.governance import ModeTransitionRequest

    for next_mode in (
        SystemMode.SHADOW,
        SystemMode.CANARY,
    ):
        eng.state_transitions.propose(
            ModeTransitionRequest(
                ts_ns=200,
                requestor="operator",
                current_mode=eng.state_transitions.current_mode(),
                target_mode=next_mode,
                reason="trace",
                operator_authorized=True,
            )
        )

    # 3) CANARY → ratify
    eng.process(propose(300, "0.6"))
    assert eng.strategy_registry.get("s1").parameters["alpha"] == "0.6"

    # 4) CANARY → LIVE → ratify again
    eng.state_transitions.propose(
        ModeTransitionRequest(
            ts_ns=400,
            requestor="operator",
            current_mode=eng.state_transitions.current_mode(),
            target_mode=SystemMode.LIVE,
            reason="trace",
            operator_authorized=True,
        )
    )
    eng.process(propose(500, "0.7"))
    assert eng.strategy_registry.get("s1").parameters["alpha"] == "0.7"

    # Ledger contains exactly one REJECTED + two RATIFIED
    rows = eng.ledger.read()
    assert sum(1 for r in rows if r.kind == "UPDATE_REJECTED") == 1
    assert sum(1 for r in rows if r.kind == "UPDATE_RATIFIED") == 2
