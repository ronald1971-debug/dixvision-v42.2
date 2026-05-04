"""Tests for ``governance_engine.control_plane.policy_drift_sentry``.

The sentry is the wiring that makes ``HAZ-POLICY-DRIFT`` actually
*do* something at runtime. The pure detector
:class:`PolicyHashAnchor` is covered by ``test_policy_hash_anchor.py``;
these tests verify only the routing contract:

* no drift -> no governance call, returns ``()``;
* drift -> hazard handed to ``governance_process``;
* drift -> caller sees whatever events Governance emits downstream;
* end-to-end with the real :class:`GovernanceEngine`, drift on a
  CRITICAL hazard transitions the FSM to ``LOCKED`` via the same
  ``emergency_lock`` path the classifier uses for every other CRITICAL
  hazard (B32 / GOV-CP-03 single FSM mutator).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.contracts.events import Event, HazardEvent
from core.contracts.governance import SystemMode
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_drift_sentry import (
    PolicyDriftSentry,
)
from governance_engine.control_plane.policy_hash_anchor import (
    HAZARD_CODE_POLICY_DRIFT,
    PolicyHashAnchor,
)
from governance_engine.engine import GovernanceEngine


def _make_files(tmp_path: Path) -> tuple[tuple[str, Path], ...]:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_bytes(b"version: 1\n")
    b.write_bytes(b"rules: []\n")
    return (("a", a), ("b", b))


def test_check_returns_empty_and_skips_governance_when_no_drift(
    tmp_path: Path,
) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    calls: list[HazardEvent] = []

    def fake_process(event: Event) -> Sequence[Event]:
        # Should never be called when there is no drift.
        calls.append(event)  # type: ignore[arg-type]
        return ()

    sentry = PolicyDriftSentry(anchor=anchor, governance_process=fake_process)

    result = sentry.check(now_ns=200)

    assert result == ()
    assert calls == []


def test_check_routes_critical_hazard_through_governance_on_mutation(
    tmp_path: Path,
) -> None:
    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    files[0][1].write_bytes(b"version: 2\n")  # mid-session edit

    routed: list[HazardEvent] = []

    def fake_process(event: Event) -> Sequence[Event]:
        routed.append(event)  # type: ignore[arg-type]
        return ()

    sentry = PolicyDriftSentry(anchor=anchor, governance_process=fake_process)

    result = sentry.check(now_ns=200)

    assert result == ()  # no downstream events from the fake processor
    assert len(routed) == 1
    assert routed[0].code == HAZARD_CODE_POLICY_DRIFT
    assert routed[0].severity.name == "CRITICAL"
    assert routed[0].meta["a_status"] == "mismatch"


def test_check_returns_governance_downstream_events(tmp_path: Path) -> None:
    """The sentry must propagate Governance's emitted events to the caller."""

    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    # Never bind -- forces a hazard on every check (unbound state).

    sentinel = object()

    def fake_process(event: Event) -> Sequence[Event]:
        return (sentinel,)  # type: ignore[return-value]

    sentry = PolicyDriftSentry(anchor=anchor, governance_process=fake_process)

    result = sentry.check(now_ns=100)

    assert result == (sentinel,)


def test_end_to_end_drift_locks_governance_fsm(tmp_path: Path) -> None:
    """End-to-end: a real Governance + drift -> FSM transitions to LOCKED.

    This is the protection the wiring delivers. Detection alone is
    inert; the sentry plumbing is what turns a CRITICAL hazard into a
    forced mode downgrade through the single FSM mutator (B32 /
    GOV-CP-03). The classifier folds CRITICAL into ``emergency_lock=True``
    so :meth:`StateTransitions.propose` is the only mutator touched.
    """

    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    governance = GovernanceEngine(ledger=ledger)
    sentry = PolicyDriftSentry(anchor=anchor, governance_process=governance.process)

    # Take the FSM into a non-LOCKED state so the LOCKED transition is
    # observable. SAFE is the default starting mode for a fresh
    # GovernanceEngine, but be defensive in case the default changes.
    starting_mode = governance.state_transitions.current_mode()
    assert starting_mode is not SystemMode.LOCKED, (
        "fresh GovernanceEngine should not start LOCKED"
    )

    # No drift -> sentry is inert, FSM unchanged.
    sentry.check(now_ns=150)
    assert governance.state_transitions.current_mode() is starting_mode

    # Mid-session edit -> drift -> sentry routes hazard -> FSM -> LOCKED.
    files[1][1].write_bytes(b"rules: [danger]\n")
    sentry.check(now_ns=200)

    assert governance.state_transitions.current_mode() is SystemMode.LOCKED


def test_check_repeats_emit_one_governance_call_per_invocation(
    tmp_path: Path,
) -> None:
    """Each ``check`` call routes one hazard while drift persists.

    This is the desired behaviour: the audit chain accumulates one
    ledger entry per detection so the operator can see *when* drift
    was first observed and *for how long* it persisted before
    remediation.
    """

    files = _make_files(tmp_path)
    ledger = LedgerAuthorityWriter()
    anchor = PolicyHashAnchor(ledger=ledger, files=files)
    anchor.bind_session(ts_ns=100, requestor="test")

    files[0][1].write_bytes(b"version: 2\n")

    routed: list[HazardEvent] = []

    def fake_process(event: Event) -> Sequence[Event]:
        routed.append(event)  # type: ignore[arg-type]
        return ()

    sentry = PolicyDriftSentry(anchor=anchor, governance_process=fake_process)

    sentry.check(now_ns=200)
    sentry.check(now_ns=210)
    sentry.check(now_ns=220)

    assert len(routed) == 3
    assert all(h.code == HAZARD_CODE_POLICY_DRIFT for h in routed)
