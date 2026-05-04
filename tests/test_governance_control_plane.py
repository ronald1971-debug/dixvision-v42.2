"""Phase 1 — Governance Control Plane (GOV-CP-01..07) tests.

Build Compiler Spec §6 / §7 enforcement:

* dashboard requests are the only operator write path
* Mode FSM: SAFE → PAPER → SHADOW → CANARY → LIVE → AUTO ratchet,
  emergency LOCK from anywhere, LOCKED → SAFE only
* every approved transition is one ledger row
* AUTO requires ``operator_authorized``
"""

from __future__ import annotations

import pytest

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from core.contracts.governance import (
    Constraint,
    ConstraintKind,
    ConstraintScope,
    DecisionKind,
    ModeTransitionRequest,
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from core.contracts.operator_consent import OperatorConsent
from governance_engine.control_plane import (
    ComplianceValidator,
    EventClassifier,
    LedgerAuthorityWriter,
    PolicyEngine,
    RiskEvaluator,
    StateTransitionManager,
)
from governance_engine.control_plane.event_classifier import (
    PipelineRoute,
    PipelineStage,
)
from governance_engine.engine import GovernanceEngine

# ---------------------------------------------------------------------------
# LedgerAuthorityWriter (GOV-CP-05)
# ---------------------------------------------------------------------------


def test_ledger_writer_chains_hashes_and_verifies():
    ledger = LedgerAuthorityWriter()
    e0 = ledger.append(ts_ns=1, kind="A", payload={"x": "1"})
    e1 = ledger.append(ts_ns=2, kind="B", payload={"y": "2"})
    e2 = ledger.append(ts_ns=3, kind="A", payload={"x": "3"})

    assert e0.seq == 0 and e1.seq == 1 and e2.seq == 2
    assert e0.prev_hash == "0" * 64
    assert e1.prev_hash == e0.hash_chain
    assert e2.prev_hash == e1.hash_chain
    assert len({e0.hash_chain, e1.hash_chain, e2.hash_chain}) == 3
    assert ledger.verify() is True
    assert ledger.head_hash() == e2.hash_chain


def test_ledger_writer_is_deterministic():
    a = LedgerAuthorityWriter()
    b = LedgerAuthorityWriter()
    rows = [
        (1, "MODE_TRANSITION", {"prev_mode": "SAFE", "new_mode": "PAPER"}),
        (2, "OPERATOR_REJECTED", {"action": "REQUEST_KILL", "rejection_code": "X"}),
        (3, "MODE_TRANSITION", {"prev_mode": "PAPER", "new_mode": "SHADOW"}),
    ]
    for ts, kind, payload in rows:
        a.append(ts_ns=ts, kind=kind, payload=payload)
        b.append(ts_ns=ts, kind=kind, payload=payload)
    assert [r.hash_chain for r in a.read()] == [
        r.hash_chain for r in b.read()
    ]


def test_ledger_writer_rejects_empty_kind():
    ledger = LedgerAuthorityWriter()
    with pytest.raises(ValueError):
        ledger.append(ts_ns=1, kind="", payload={})


# ---------------------------------------------------------------------------
# StateTransitionManager (GOV-CP-03) — Mode FSM
# ---------------------------------------------------------------------------


def _build_state(initial: SystemMode = SystemMode.SAFE) -> tuple[
    StateTransitionManager, LedgerAuthorityWriter, PolicyEngine
]:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    state = StateTransitionManager(
        policy=policy, ledger=ledger, initial_mode=initial
    )
    return state, ledger, policy


def _consent(
    *,
    ts_ns: int,
    mode_from: SystemMode,
    mode_to: SystemMode,
    policy: PolicyEngine,
    operator_id: str = "op",
    nonce: str | None = None,
) -> OperatorConsent:
    """Build a fresh, valid OperatorConsent for the SAFE→PAPER and
    LIVE→AUTO consent-required edges (Hardening-S1 item 8)."""

    return OperatorConsent(
        ts_ns=ts_ns,
        operator_id=operator_id,
        mode_from=mode_from,
        mode_to=mode_to,
        policy_hash=policy.table_hash,
        nonce=nonce or f"nonce-{ts_ns}-{mode_from.name}-{mode_to.name}",
    )


def test_mode_fsm_forward_ratchet_one_step():
    state, ledger, policy = _build_state()
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=10,
            requestor="op",
            current_mode=SystemMode.SAFE,
            target_mode=SystemMode.PAPER,
            reason="bring up paper trading",
            consent=_consent(
                ts_ns=10,
                mode_from=SystemMode.SAFE,
                mode_to=SystemMode.PAPER,
                policy=policy,
            ),
        )
    )
    assert decision.approved is True
    assert state.current_mode() is SystemMode.PAPER
    # Hardening-S1 item 8 — OPERATOR_CONSENT_ACCEPTED row is
    # paired with the MODE_TRANSITION row (consent first, then
    # transition). decision.ledger_seq points at the
    # MODE_TRANSITION row.
    rows = ledger.read()
    assert rows[0].kind == "OPERATOR_CONSENT_ACCEPTED"
    assert rows[1].kind == "MODE_TRANSITION"
    assert decision.ledger_seq == 1


def test_mode_fsm_forward_skip_rejected():
    state, _, _ = _build_state()
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=11,
            requestor="op",
            current_mode=SystemMode.SAFE,
            target_mode=SystemMode.LIVE,
            reason="skip ahead",
            operator_authorized=True,
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "FSM_FORWARD_SKIP"
    assert state.current_mode() is SystemMode.SAFE


def test_mode_fsm_live_requires_operator_authorisation():
    state, _, _ = _build_state(initial=SystemMode.CANARY)
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=12,
            requestor="op",
            current_mode=SystemMode.CANARY,
            target_mode=SystemMode.LIVE,
            reason="go live",
            operator_authorized=False,
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "POLICY_OPERATOR_REQUIRED"
    assert state.current_mode() is SystemMode.CANARY


def test_mode_fsm_auto_requires_operator_authorisation():
    state, _, policy = _build_state(initial=SystemMode.LIVE)
    # Hardening-S1 item 8 — LIVE → AUTO requires a typed consent
    # envelope. The legacy ``operator_authorized=False`` path now
    # surfaces ``CONSENT_MISSING`` instead of the old
    # ``POLICY_OPERATOR_REQUIRED`` because the consent gate runs
    # before the policy gate.
    no_op = state.propose(
        ModeTransitionRequest(
            ts_ns=13,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="autonomous",
            operator_authorized=False,
        )
    )
    assert no_op.approved is False
    assert no_op.rejection_code == "CONSENT_MISSING"

    ok = state.propose(
        ModeTransitionRequest(
            ts_ns=14,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="autonomous",
            consent=_consent(
                ts_ns=14,
                mode_from=SystemMode.LIVE,
                mode_to=SystemMode.AUTO,
                policy=policy,
            ),
        )
    )
    assert ok.approved is True
    assert state.current_mode() is SystemMode.AUTO


def test_consent_nonce_not_burned_when_promotion_gates_rejects():
    """Regression for the orphaned-consent-row bug.

    When LIVE → AUTO is consent-gated AND promotion-gated, a consent
    envelope that passes the consent validator must NOT have its nonce
    registered (and the OPERATOR_CONSENT_ACCEPTED audit row must NOT
    be written) until promotion gates and policy have also accepted
    the transition. Otherwise a downstream rejection would burn a
    semantically-valid nonce and leave an orphan audit row in the
    ledger, breaking the consent → MODE_TRANSITION pairing contract.
    """

    class _RejectingPromotionGates:
        """Stub gates that always reject AUTO."""

        def check(self, target_mode_name: str) -> tuple[bool, str]:
            if target_mode_name == "AUTO":
                return False, "PROMOTION_GATES_HASH_MISMATCH"
            return True, ""

        def bind(self, *, ts_ns: int, requestor: str) -> str:
            return "0" * 64

    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    state = StateTransitionManager(
        policy=policy,
        ledger=ledger,
        initial_mode=SystemMode.LIVE,
        promotion_gates=_RejectingPromotionGates(),
    )

    consent = _consent(
        ts_ns=100,
        mode_from=SystemMode.LIVE,
        mode_to=SystemMode.AUTO,
        policy=policy,
        nonce="reusable-nonce-1",
    )
    rejected = state.propose(
        ModeTransitionRequest(
            ts_ns=100,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="autonomous",
            consent=consent,
        )
    )
    assert rejected.approved is False
    assert rejected.rejection_code == "PROMOTION_GATES_HASH_MISMATCH"

    # No OPERATOR_CONSENT_ACCEPTED row was written for the rejected
    # transition (the row only ships paired with MODE_TRANSITION).
    kinds = [row.kind for row in ledger.read()]
    assert "OPERATOR_CONSENT_ACCEPTED" not in kinds
    assert "MODE_TRANSITION_REJECTED" in kinds

    # The same nonce can be reused -- it was not burned by the
    # rejected attempt. Swap in a passthrough gates so the retry
    # actually reaches the approval branch and the nonce gets
    # committed exactly once.
    class _PassThroughPromotionGates:
        def check(self, target_mode_name: str) -> tuple[bool, str]:
            return True, ""

        def bind(self, *, ts_ns: int, requestor: str) -> str:
            return "0" * 64

    state._promotion_gates = _PassThroughPromotionGates()
    retry = state.propose(
        ModeTransitionRequest(
            ts_ns=101,
            requestor="op",
            current_mode=SystemMode.LIVE,
            target_mode=SystemMode.AUTO,
            reason="autonomous (retry)",
            consent=_consent(
                ts_ns=101,
                mode_from=SystemMode.LIVE,
                mode_to=SystemMode.AUTO,
                policy=policy,
                nonce="reusable-nonce-1",
            ),
        )
    )
    assert retry.approved is True
    assert state.current_mode() is SystemMode.AUTO

    # Now exactly one OPERATOR_CONSENT_ACCEPTED, paired with the
    # successful MODE_TRANSITION row.
    kinds = [row.kind for row in ledger.read()]
    assert kinds.count("OPERATOR_CONSENT_ACCEPTED") == 1
    accepted_idx = kinds.index("OPERATOR_CONSENT_ACCEPTED")
    assert kinds[accepted_idx + 1] == "MODE_TRANSITION"


def test_mode_fsm_emergency_lock_from_any_state():
    for start in (
        SystemMode.PAPER,
        SystemMode.SHADOW,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
    ):
        state, _, _ = _build_state(initial=start)
        decision = state.propose(
            ModeTransitionRequest(
                ts_ns=20,
                requestor="dyon",
                current_mode=start,
                target_mode=SystemMode.LOCKED,
                reason="HAZ-04 stale data",
                operator_authorized=True,
            )
        )
        assert decision.approved is True
        assert state.current_mode() is SystemMode.LOCKED


def test_mode_fsm_locked_only_to_safe():
    state, _, _ = _build_state(initial=SystemMode.LOCKED)
    bad = state.propose(
        ModeTransitionRequest(
            ts_ns=30,
            requestor="op",
            current_mode=SystemMode.LOCKED,
            target_mode=SystemMode.PAPER,
            reason="resume",
            operator_authorized=True,
        )
    )
    assert bad.approved is False
    assert bad.rejection_code == "FSM_LOCKED_ONLY_TO_SAFE"

    good = state.propose(
        ModeTransitionRequest(
            ts_ns=31,
            requestor="op",
            current_mode=SystemMode.LOCKED,
            target_mode=SystemMode.SAFE,
            reason="reset",
            operator_authorized=True,
        )
    )
    assert good.approved is True
    assert state.current_mode() is SystemMode.SAFE


def test_mode_fsm_de_escalation_always_allowed():
    state, _, _ = _build_state(initial=SystemMode.AUTO)
    for target in (SystemMode.LIVE, SystemMode.SHADOW, SystemMode.SAFE):
        d = state.propose(
            ModeTransitionRequest(
                ts_ns=40,
                requestor="op",
                current_mode=state.current_mode(),
                target_mode=target,
                reason="de-escalate",
            )
        )
        assert d.approved is True
        assert state.current_mode() is target


def test_mode_fsm_no_op_rejected():
    state, _, _ = _build_state()
    d = state.propose(
        ModeTransitionRequest(
            ts_ns=50,
            requestor="op",
            current_mode=SystemMode.SAFE,
            target_mode=SystemMode.SAFE,
            reason="noop",
        )
    )
    assert d.approved is False
    assert d.rejection_code == "FSM_NO_OP"


# ---------------------------------------------------------------------------
# RiskEvaluator (GOV-CP-02)
# ---------------------------------------------------------------------------


def _risk_with_limits(qty_cap: float, exp_cap: float) -> RiskEvaluator:
    constraints = (
        Constraint(
            id="GLOBAL_QTY",
            scope=ConstraintScope.GLOBAL,
            kind=ConstraintKind.MAX_POSITION_QTY,
            params={"limit": str(qty_cap)},
        ),
        Constraint(
            id="GLOBAL_EXP",
            scope=ConstraintScope.GLOBAL,
            kind=ConstraintKind.MAX_SYMBOL_EXPOSURE,
            params={"limit": str(exp_cap)},
        ),
    )
    return RiskEvaluator(constraints=constraints)


def test_risk_evaluator_approves_within_limits():
    evaluator = _risk_with_limits(qty_cap=10.0, exp_cap=20.0)
    a = evaluator.assess(ts_ns=1, symbol="BTC-USD", side="BUY", qty=5.0)
    assert a.approved is True
    assert a.exposure_after == 5.0
    evaluator.commit(a)
    assert evaluator.book.get("BTC-USD") == 5.0


def test_risk_evaluator_rejects_oversize_qty():
    evaluator = _risk_with_limits(qty_cap=10.0, exp_cap=20.0)
    a = evaluator.assess(ts_ns=2, symbol="BTC-USD", side="BUY", qty=11.0)
    assert a.approved is False
    assert any(b.startswith("MAX_POSITION_QTY") for b in a.breached_limits)


def test_risk_evaluator_rejects_exposure_breach():
    evaluator = _risk_with_limits(qty_cap=100.0, exp_cap=10.0)
    evaluator.book.set("BTC-USD", 8.0)
    a = evaluator.assess(ts_ns=3, symbol="BTC-USD", side="BUY", qty=5.0)
    assert a.approved is False
    assert any(b.startswith("MAX_SYMBOL_EXPOSURE") for b in a.breached_limits)


def test_risk_evaluator_symbol_scope_overrides_global():
    constraints = (
        Constraint(
            id="GLOBAL",
            scope=ConstraintScope.GLOBAL,
            kind=ConstraintKind.MAX_POSITION_QTY,
            params={"limit": "10"},
        ),
        Constraint(
            id="ETH_ONLY",
            scope=ConstraintScope.SYMBOL,
            kind=ConstraintKind.MAX_POSITION_QTY,
            params={"limit": "2", "symbol": "ETH-USD"},
        ),
    )
    evaluator = RiskEvaluator(constraints=constraints)
    bad = evaluator.assess(ts_ns=4, symbol="ETH-USD", side="BUY", qty=5.0)
    good = evaluator.assess(ts_ns=4, symbol="BTC-USD", side="BUY", qty=5.0)
    assert bad.approved is False
    assert good.approved is True


def test_risk_evaluator_rejects_invalid_inputs():
    evaluator = _risk_with_limits(qty_cap=10.0, exp_cap=20.0)
    a = evaluator.assess(ts_ns=5, symbol="X", side="BUY", qty=0.0)
    assert a.approved is False
    assert a.rejection_code == "RISK_NON_POSITIVE_QTY"
    b = evaluator.assess(ts_ns=5, symbol="X", side="HOLD", qty=1.0)
    assert b.approved is False
    assert b.rejection_code == "RISK_INVALID_SIDE"


# ---------------------------------------------------------------------------
# ComplianceValidator (GOV-CP-06)
# ---------------------------------------------------------------------------


def test_compliance_blocks_trade_in_safe_mode():
    cv = ComplianceValidator()
    r = cv.validate_order(
        domain="NORMAL_TRADING", notional_usd=100.0, mode=SystemMode.SAFE
    )
    assert r.passed is False
    assert "COMPLIANCE_NO_TRADE_IN_SAFE" in r.violations


def test_compliance_memecoin_per_trade_cap():
    cv = ComplianceValidator()
    r = cv.validate_order(
        domain="MEMECOIN", notional_usd=300.0, mode=SystemMode.LIVE
    )
    assert r.passed is False
    assert any(v.startswith("COMPLIANCE_PER_TRADE_CAP:MEMECOIN") for v in r.violations)


def test_compliance_memecoin_daily_cap():
    cv = ComplianceValidator()
    # Four 200-USD trades within the per-trade cap (250) — total 800.
    for _ in range(4):
        ok = cv.validate_order(
            domain="MEMECOIN", notional_usd=200.0, mode=SystemMode.LIVE
        )
        assert ok.passed is True
    breach = cv.validate_order(
        domain="MEMECOIN", notional_usd=250.0, mode=SystemMode.LIVE
    )
    assert breach.passed is False
    assert any(v.startswith("COMPLIANCE_DAILY_CAP:MEMECOIN") for v in breach.violations)


def test_compliance_unknown_domain_blocked():
    cv = ComplianceValidator()
    r = cv.validate_order(
        domain="WAT", notional_usd=10.0, mode=SystemMode.LIVE
    )
    assert r.passed is False
    assert any(v.startswith("COMPLIANCE_UNKNOWN_DOMAIN") for v in r.violations)


# ---------------------------------------------------------------------------
# EventClassifier (GOV-CP-04)
# ---------------------------------------------------------------------------


def test_classifier_high_severity_hazard_triggers_lock():
    ec = EventClassifier()
    h = HazardEvent(
        ts_ns=1, code="HAZ-04", severity=HazardSeverity.CRITICAL,
        source="dyon", detail="stale data 5s",
    )
    route = ec.classify(h)
    assert route.emergency_lock is True
    assert PipelineStage.STATE_TRANSITION in route.stages


def test_classifier_low_hazard_audit_only():
    ec = EventClassifier()
    h = HazardEvent(
        ts_ns=1, code="HAZ-08", severity=HazardSeverity.LOW,
        source="dyon", detail="benign",
    )
    route = ec.classify(h)
    assert route.emergency_lock is False
    assert route.stages == (PipelineStage.LEDGER,)


def test_classifier_update_proposed_routes_through_policy():
    ec = EventClassifier()
    s = SystemEvent(
        ts_ns=1,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={"target": "microstructure_v1"},
    )
    route = ec.classify(s)
    assert PipelineStage.POLICY in route.stages
    assert PipelineStage.COMPLIANCE in route.stages
    assert PipelineStage.LEDGER in route.stages


def test_classifier_heartbeat_is_noop():
    ec = EventClassifier()
    s = SystemEvent(
        ts_ns=1,
        sub_kind=SystemEventKind.HEARTBEAT,
        source="dyon",
    )
    route = ec.classify(s)
    assert isinstance(route, PipelineRoute)
    assert route.stages == (PipelineStage.NOOP,)


# ---------------------------------------------------------------------------
# OperatorInterfaceBridge (GOV-CP-07)
# ---------------------------------------------------------------------------


def _build_engine() -> GovernanceEngine:
    return GovernanceEngine()


def _consent_payload(
    *,
    eng: GovernanceEngine,
    ts_ns: int,
    target_mode: str,
    operator_id: str = "ronald",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a REQUEST_MODE payload with the four ``consent_*`` fields
    expected by ``OperatorInterfaceBridge._handle_mode`` for
    consent-required edges (Hardening-S1 item 8).
    """

    payload: dict[str, str] = {
        "target_mode": target_mode,
        "consent_operator_id": operator_id,
        "consent_policy_hash": eng.policy.table_hash,
        "consent_nonce": f"nonce-bridge-{ts_ns}-{target_mode}",
        "consent_ts_ns": str(ts_ns),
    }
    if extra:
        payload.update(extra)
    return payload


def test_operator_bridge_mode_transition_approved():
    eng = _build_engine()
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="ronald",
            action=OperatorAction.REQUEST_MODE,
            payload=_consent_payload(
                eng=eng,
                ts_ns=1,
                target_mode="PAPER",
                extra={"reason": "bring up"},
            ),
        )
    )
    assert decision.approved is True
    assert decision.kind is DecisionKind.MODE_TRANSITION
    assert eng.current_mode() is SystemMode.PAPER
    assert decision.ledger_seq >= 0


def test_operator_bridge_kill_locks_system():
    eng = _build_engine()
    eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="ronald",
            action=OperatorAction.REQUEST_MODE,
            payload=_consent_payload(eng=eng, ts_ns=1, target_mode="PAPER"),
        )
    )
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=2,
            requestor="ronald",
            action=OperatorAction.REQUEST_KILL,
            payload={"reason": "panic"},
        )
    )
    assert decision.approved is True
    assert decision.kind is DecisionKind.KILL
    assert eng.current_mode() is SystemMode.LOCKED


def test_operator_bridge_locked_only_unlock_allowed():
    eng = _build_engine()
    eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_KILL,
            payload={},
        )
    )
    bad = eng.operator.submit(
        OperatorRequest(
            ts_ns=2,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload={"target_mode": "PAPER"},
        )
    )
    assert bad.approved is False
    assert bad.rejection_code == "POLICY_LOCKED"

    good = eng.operator.submit(
        OperatorRequest(
            ts_ns=3,
            requestor="op",
            action=OperatorAction.REQUEST_UNLOCK,
            payload={"reason": "resume"},
        )
    )
    assert good.approved is True
    assert eng.current_mode() is SystemMode.SAFE


def test_operator_bridge_unknown_target_mode_rejected():
    eng = _build_engine()
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload={"target_mode": "TURBO"},
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "BRIDGE_UNKNOWN_MODE"


def test_operator_bridge_plugin_lifecycle_logs_audit_row():
    eng = _build_engine()
    # First leave SAFE so the policy gate permits ACTIVE transitions.
    eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload=_consent_payload(
                eng=eng, ts_ns=1, target_mode="PAPER", operator_id="op"
            ),
        )
    )
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=2,
            requestor="op",
            action=OperatorAction.REQUEST_PLUGIN_LIFECYCLE,
            payload={
                "plugin_path": "intelligence_engine.plugins.microstructure",
                "target_status": "ACTIVE",
                "reason": "promote v1 from SHADOW",
            },
        )
    )
    assert decision.approved is True
    assert decision.kind is DecisionKind.PLUGIN_LIFECYCLE
    rows = [r for r in eng.ledger.read() if r.kind == "PLUGIN_LIFECYCLE"]
    assert len(rows) == 1
    assert rows[0].payload["plugin_path"].endswith("microstructure")


# ---------------------------------------------------------------------------
# GovernanceEngine.process(event)
# ---------------------------------------------------------------------------


def test_engine_critical_hazard_locks_via_process():
    eng = _build_engine()
    eng.process(
        HazardEvent(
            ts_ns=1,
            code="HAZ-04",
            severity=HazardSeverity.CRITICAL,
            source="dyon",
            detail="stale data > 5s",
        )
    )
    assert eng.current_mode() is SystemMode.LOCKED


def test_engine_low_hazard_does_not_lock():
    eng = _build_engine()
    eng.process(
        HazardEvent(
            ts_ns=1,
            code="HAZ-08",
            severity=HazardSeverity.LOW,
            source="dyon",
            detail="latency uptick",
        )
    )
    assert eng.current_mode() is SystemMode.SAFE
    rows = [r for r in eng.ledger.read() if r.kind == "HAZARD_AUDIT"]
    assert len(rows) == 1


def test_engine_signal_event_writes_audit_row():
    eng = _build_engine()
    eng.process(
        SignalEvent(
            ts_ns=42,
            symbol="BTC-USD",
            side=Side.BUY,
            confidence=0.7,
            plugin_chain=("microstructure_v1",),
        )
    )
    rows = [r for r in eng.ledger.read() if r.kind == "SIGNAL_AUDIT"]
    assert len(rows) == 1
    assert rows[0].ts_ns == 42


def test_engine_execution_event_writes_audit_row():
    eng = _build_engine()
    eng.process(
        ExecutionEvent(
            ts_ns=43,
            symbol="BTC-USD",
            side=Side.BUY,
            qty=1.0,
            price=50_000.0,
            status=ExecutionStatus.FILLED,
            venue="paper",
            order_id="o1",
        )
    )
    rows = [r for r in eng.ledger.read() if r.kind == "EXECUTION_AUDIT"]
    assert len(rows) == 1
    assert rows[0].ts_ns == 43


def test_engine_check_self_reports_cp_modules():
    eng = _build_engine()
    health = eng.check_self()
    assert "control_plane" in health.plugin_states
    cp = health.plugin_states["control_plane"]
    for spec in (
        "GOV-CP-01",
        "GOV-CP-02",
        "GOV-CP-03",
        "GOV-CP-04",
        "GOV-CP-05",
        "GOV-CP-06",
        "GOV-CP-07",
    ):
        assert spec in cp


def test_engine_check_self_degrades_when_locked():
    eng = _build_engine()
    eng.process(
        HazardEvent(
            ts_ns=1,
            code="HAZ-04",
            severity=HazardSeverity.CRITICAL,
            source="dyon",
        )
    )
    health = eng.check_self()
    assert "mode=LOCKED" in health.detail


# ---------------------------------------------------------------------------
# Replay determinism (INV-15 / TEST-01)
# ---------------------------------------------------------------------------


def test_two_engines_same_inputs_produce_same_ledger():
    a = _build_engine()
    b = _build_engine()
    requests = [
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload=_consent_payload(
                eng=a, ts_ns=1, target_mode="PAPER", operator_id="op"
            ),
        ),
        OperatorRequest(
            ts_ns=2,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload={"target_mode": "SHADOW"},
        ),
        OperatorRequest(
            ts_ns=3,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload={"target_mode": "AUTO"},  # FSM_FORWARD_SKIP — rejected
        ),
        OperatorRequest(
            ts_ns=4,
            requestor="op",
            action=OperatorAction.REQUEST_KILL,
            payload={},
        ),
    ]
    for r in requests:
        a.operator.submit(r)
        b.operator.submit(r)

    rows_a = [(r.seq, r.kind, r.hash_chain) for r in a.ledger.read()]
    rows_b = [(r.seq, r.kind, r.hash_chain) for r in b.ledger.read()]
    assert rows_a == rows_b
    assert a.current_mode() is b.current_mode() is SystemMode.LOCKED
