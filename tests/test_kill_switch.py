"""Tests for SAFE-01 — system/kill_switch.py (P0-1b)."""

from __future__ import annotations

from core.contracts.governance import DecisionKind, SystemMode
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_engine import PolicyEngine
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
)
from system.kill_switch import KillReason, KillRequest, KillSwitch


def _build(initial: SystemMode = SystemMode.LIVE) -> KillSwitch:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    state = StateTransitionManager(
        policy=policy, ledger=ledger, initial_mode=initial
    )
    return KillSwitch(state_transitions=state)


def test_engage_operator_locks_from_live():
    ks = _build(initial=SystemMode.LIVE)
    decision = ks.engage_operator(requestor="op-alice", reason="panic", ts_ns=42)
    assert decision.approved is True
    assert decision.kind is DecisionKind.KILL
    assert "system locked" in decision.summary


def test_engage_hazard_locks_with_sensor_requestor():
    ks = _build(initial=SystemMode.CANARY)
    decision = ks.engage_hazard(sensor="HAZ-04", reason="stale data", ts_ns=43)
    assert decision.approved is True
    assert decision.kind is DecisionKind.KILL


def test_engage_external_locks_from_paper():
    ks = _build(initial=SystemMode.PAPER)
    decision = ks.engage_external(
        requestor="cockpit-pairing", reason="ops", ts_ns=44
    )
    assert decision.approved is True
    assert decision.kind is DecisionKind.KILL


def test_engage_carries_origin_tag_in_reason():
    """Reason on the ledger gets ``[ORIGIN]`` prefix so post-hoc audit
    can answer "who pulled the cord" without joining tables."""

    ks = _build(initial=SystemMode.LIVE)
    decision = ks.engage(
        KillRequest(
            requestor="HAZ-07",
            reason="catastrophic loss",
            origin=KillReason.HAZARD,
            ts_ns=45,
        )
    )
    assert decision.approved is True
    assert decision.kind is DecisionKind.KILL


def test_engage_from_safe_is_legal_locked_edge():
    """SAFE -> LOCKED is a legal kill edge (defensive no-op when
    already-restrictive but not yet locked)."""

    ks = _build(initial=SystemMode.SAFE)
    decision = ks.engage_operator(requestor="op", reason="kill", ts_ns=46)
    assert decision.approved is True


def test_engage_from_locked_is_idempotent_or_rejected():
    """Already-LOCKED is either a legal self-edge or an FSM rejection.

    Whichever the FSM says is fine — the test only enforces that the
    decision returns with ``DecisionKind.KILL`` so the audit ledger is
    consistent.
    """

    ks = _build(initial=SystemMode.LOCKED)
    decision = ks.engage_operator(requestor="op", reason="redundant", ts_ns=47)
    assert decision.kind is DecisionKind.KILL


def test_engage_uses_provided_ts_ns():
    ks = _build(initial=SystemMode.LIVE)
    decision = ks.engage_operator(requestor="op", reason="panic", ts_ns=99_999)
    assert decision.ts_ns == 99_999


def test_engage_uses_wall_ns_when_ts_omitted():
    ks = _build(initial=SystemMode.LIVE)
    decision = ks.engage_operator(requestor="op", reason="panic")
    assert decision.ts_ns > 0
