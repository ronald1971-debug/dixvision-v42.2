"""DASH-MCP-01 — MemecoinControlPanel widget tests."""

from __future__ import annotations

from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from dashboard_backend.control_plane.memecoin_control_panel import MemecoinControlPanel


def _move_to_paper(router, consent_payload) -> None:
    out = router.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_MODE,
            payload=consent_payload(
                ts_ns=1,
                target_mode="PAPER",
                extra={
                    "reason": "bring up paper",
                    "operator_authorized": "false",
                },
            ),
        )
    )
    assert out.approved is True


def test_default_status_is_disabled(governance_stack, router):
    panel = MemecoinControlPanel(router=router)
    status = panel.status()
    assert status.enabled is False
    assert status.killed is False
    assert "disabled" in status.summary


def test_request_enable_in_safe_is_rejected_by_policy(governance_stack, router):
    # Build Compiler Spec §7 PolicyEngine forbids ACTIVE plugin
    # lifecycle in SAFE mode (POLICY_LIFECYCLE_REQUIRES_NON_SAFE).
    panel = MemecoinControlPanel(router=router)
    outcome = panel.request_enable(
        ts_ns=1, requestor="op", reason="bring up memecoin"
    )
    assert outcome.approved is False
    assert panel.status().enabled is False


def test_request_enable_in_paper_is_approved(governance_stack, router, consent_payload):
    _move_to_paper(router, consent_payload)
    panel = MemecoinControlPanel(router=router)
    outcome = panel.request_enable(
        ts_ns=2, requestor="op", reason="bring up memecoin"
    )
    assert outcome.approved is True
    assert panel.status().enabled is True


def test_request_disable_after_enable(governance_stack, router, consent_payload):
    _move_to_paper(router, consent_payload)
    panel = MemecoinControlPanel(router=router)
    panel.request_enable(ts_ns=2, requestor="op", reason="up")
    outcome = panel.request_disable(
        ts_ns=3, requestor="op", reason="winding down"
    )
    assert outcome.approved is True
    status = panel.status()
    assert status.enabled is False
    assert status.killed is False


def test_request_kill_marks_status_killed(governance_stack, router, consent_payload):
    _, _, state, _ = governance_stack
    _move_to_paper(router, consent_payload)
    panel = MemecoinControlPanel(router=router)
    panel.request_enable(ts_ns=2, requestor="op", reason="up")
    outcome = panel.request_kill(
        ts_ns=3, requestor="op", reason="emergency"
    )
    assert outcome.approved is True
    status = panel.status()
    assert status.killed is True
    assert status.enabled is False
    # request_kill on memecoin doesn't change Mode FSM
    assert state.current_mode() is SystemMode.PAPER


def test_rejected_request_does_not_change_status(governance_stack, router):
    _, _, state, _ = governance_stack
    panel = MemecoinControlPanel(router=router)
    # Force LOCKED mode by killing the system first.
    router.submit(
        OperatorRequest(
            ts_ns=10,
            requestor="op",
            action=OperatorAction.REQUEST_KILL,
            payload={"reason": "force lock"},
        )
    )
    assert state.current_mode() is SystemMode.LOCKED
    # In LOCKED, plugin lifecycle is rejected by the policy engine.
    outcome = panel.request_enable(ts_ns=11, requestor="op", reason="x")
    assert outcome.approved is False
    assert panel.status().enabled is False
