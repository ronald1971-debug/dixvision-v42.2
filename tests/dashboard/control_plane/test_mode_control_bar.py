"""DASH-02 — ModeControlBar widget tests."""

from __future__ import annotations

from core.contracts.governance import SystemMode
from dashboard.control_plane.mode_control_bar import ModeControlBar


def test_snapshot_returns_legal_targets_in_safe(governance_stack, router):
    _, _, state, _ = governance_stack
    bar = ModeControlBar(state_transitions=state, router=router)
    snap = bar.snapshot()
    assert snap.current_mode == "SAFE"
    assert "PAPER" in snap.legal_targets
    assert "LOCKED" in snap.legal_targets  # emergency lock from anywhere
    assert "LIVE" not in snap.legal_targets  # forward-skip forbidden
    assert snap.is_locked is False


def test_request_transition_routes_through_router(governance_stack, router):
    ledger, _, state, _ = governance_stack
    bar = ModeControlBar(state_transitions=state, router=router)
    outcome = bar.request_transition(
        ts_ns=10,
        requestor="op",
        target_mode="PAPER",
        reason="bring up paper",
    )
    assert outcome.approved is True
    assert state.current_mode() is SystemMode.PAPER
    assert any(row.kind == "MODE_TRANSITION" for row in ledger.read())


def test_request_transition_unknown_mode_raises(governance_stack, router):
    _, _, state, _ = governance_stack
    bar = ModeControlBar(state_transitions=state, router=router)
    try:
        bar.request_transition(
            ts_ns=11,
            requestor="op",
            target_mode="NOPE",
            reason="",
        )
    except ValueError as exc:
        assert "unknown target mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_request_kill_locks_system(governance_stack, router):
    _, _, state, _ = governance_stack
    bar = ModeControlBar(state_transitions=state, router=router)
    outcome = bar.request_kill(ts_ns=12, requestor="op", reason="emergency")
    assert outcome.approved is True
    assert state.current_mode() is SystemMode.LOCKED


def test_snapshot_after_lock_marks_is_locked(governance_stack, router):
    _, _, state, _ = governance_stack
    bar = ModeControlBar(state_transitions=state, router=router)
    bar.request_kill(ts_ns=13, requestor="op", reason="emergency")
    snap = bar.snapshot()
    assert snap.current_mode == "LOCKED"
    assert snap.is_locked is True
