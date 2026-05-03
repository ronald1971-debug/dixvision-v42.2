"""Hardening-S1 item 4 — fail-closed governance handler tests.

The architecture critique flagged ``GovernanceEngine._handle_system``
+ ``EventClassifier._classify_system`` as default-permissive: an
unknown :class:`SystemEventKind` (or a future enum addition nobody
routes) silently dropped on the floor with no audit trail. The fix
treats the classifier's known-kind sets as an explicit allowlist and
writes a loud ``UNKNOWN_SYSTEM_KIND`` ledger row for any miss.

These tests pin the new contract:

* The two classifier allowlists are disjoint and together cover every
  ``SystemEventKind`` member — so adding a new enum value without
  thinking about routing trips a test, not a silent runtime
  regression.
* An unknown ``sub_kind`` reaches the engine via a dedicated route
  whose ``note`` starts with ``unknown_system_kind:`` and the engine
  writes one ``UNKNOWN_SYSTEM_KIND`` row instead of dispatching.
* PLUGIN_LIFECYCLE payloads missing ``plugin_id`` or ``lifecycle``
  produce a ``PLUGIN_LIFECYCLE_REJECTED`` row with code
  ``MALFORMED_PAYLOAD`` — matching the existing UPDATE_PROPOSED
  fail-closed pattern.
* Valid PLUGIN_LIFECYCLE / UPDATE_PROPOSED / audit-only events still
  process unchanged (no regression).
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from core.contracts.events import (
    SystemEvent,
    SystemEventKind,
)
from governance_engine.control_plane.event_classifier import (
    _AUDIT_ONLY_SYSTEM_KINDS,
    _GOVERNANCE_HANDLED_SYSTEM_KINDS,
    EventClassifier,
    PipelineRoute,
    PipelineStage,
)
from governance_engine.engine import GovernanceEngine

# ---------------------------------------------------------------------------
# Allowlist invariants
# ---------------------------------------------------------------------------


def test_handled_and_audit_only_sets_are_disjoint():
    overlap = _GOVERNANCE_HANDLED_SYSTEM_KINDS & _AUDIT_ONLY_SYSTEM_KINDS
    assert overlap == frozenset(), (
        "Hardening-S1 item 4: a SystemEventKind cannot simultaneously "
        "be governance-handled and audit-only — overlap leaks the "
        "fail-closed invariant. Offending: " + str(overlap)
    )


def test_allowlists_cover_every_system_event_kind():
    covered = _GOVERNANCE_HANDLED_SYSTEM_KINDS | _AUDIT_ONLY_SYSTEM_KINDS
    missing = set(SystemEventKind) - covered
    assert missing == set(), (
        "Hardening-S1 item 4: every SystemEventKind must be classified "
        "either as governance-handled or audit-only. Missing kinds "
        "cause UNKNOWN_SYSTEM_KIND audit rows at runtime — add them "
        "deliberately to the appropriate allowlist in "
        "governance_engine.control_plane.event_classifier. "
        "Missing: " + ", ".join(sorted(k.value for k in missing))
    )


# ---------------------------------------------------------------------------
# Classifier — unknown kind path
# ---------------------------------------------------------------------------


def test_classifier_unknown_sub_kind_returns_unknown_route():
    """Construct a SystemEvent whose sub_kind is not in either allowlist.

    SystemEventKind is a StrEnum and the dataclass is slot-only, so we
    bypass the type by passing a plain string — this mirrors what
    happens when a future enum addition (or a buggy producer) emits a
    sub_kind nobody routes.
    """
    ec = EventClassifier()
    s = SystemEvent(
        ts_ns=1,
        sub_kind=cast(SystemEventKind, "TOTALLY_UNKNOWN_KIND"),
        source="future_engine",
    )
    route = ec.classify(s)
    assert isinstance(route, PipelineRoute)
    assert route.note.startswith("unknown_system_kind:")
    # The route still terminates at LEDGER so the engine can write an
    # audit row — but it never visits POLICY / RISK / COMPLIANCE.
    assert route.stages == (PipelineStage.LEDGER,)
    assert PipelineStage.POLICY not in route.stages
    assert PipelineStage.RISK not in route.stages
    assert PipelineStage.COMPLIANCE not in route.stages


def test_classifier_known_audit_only_kind_is_noop():
    ec = EventClassifier()
    for kind in _AUDIT_ONLY_SYSTEM_KINDS:
        s = SystemEvent(ts_ns=1, sub_kind=kind, source="dyon")
        route = ec.classify(s)
        assert route.stages == (PipelineStage.NOOP,), (
            f"audit-only kind {kind} must classify to NOOP, "
            f"got {route.stages}"
        )


# ---------------------------------------------------------------------------
# Engine — unknown kind writes UNKNOWN_SYSTEM_KIND row, no handler runs
# ---------------------------------------------------------------------------


def test_process_unknown_sub_kind_writes_audit_row_and_returns_empty():
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=42,
        sub_kind=cast(SystemEventKind, "FUTURE_KIND_XYZ"),
        source="future_engine",
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    assert len(rows) == 1
    assert rows[0].kind == "UNKNOWN_SYSTEM_KIND"
    assert rows[0].payload["code"] == "FAIL_CLOSED_UNKNOWN_KIND"
    assert "FUTURE_KIND_XYZ" in rows[0].payload["sub_kind_repr"]
    assert rows[0].payload["source"] == "future_engine"


# ---------------------------------------------------------------------------
# Engine — PLUGIN_LIFECYCLE missing required fields → fail closed
# ---------------------------------------------------------------------------


def test_plugin_lifecycle_missing_plugin_id_rejected():
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=10,
        sub_kind=SystemEventKind.PLUGIN_LIFECYCLE,
        source="dashboard",
        payload={"lifecycle": "ACTIVE"},  # no plugin_id
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    assert len(rows) == 1
    assert rows[0].kind == "PLUGIN_LIFECYCLE_REJECTED"
    assert rows[0].payload["code"] == "MALFORMED_PAYLOAD"
    assert "plugin_id" in rows[0].payload["detail"]


def test_plugin_lifecycle_missing_lifecycle_rejected():
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=11,
        sub_kind=SystemEventKind.PLUGIN_LIFECYCLE,
        source="dashboard",
        payload={"plugin_id": "microstructure_v1"},  # no lifecycle
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    assert len(rows) == 1
    assert rows[0].kind == "PLUGIN_LIFECYCLE_REJECTED"
    assert rows[0].payload["code"] == "MALFORMED_PAYLOAD"
    assert "lifecycle" in rows[0].payload["detail"]


def test_plugin_lifecycle_missing_both_fields_lists_both():
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=12,
        sub_kind=SystemEventKind.PLUGIN_LIFECYCLE,
        source="dashboard",
        payload={},
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    assert len(rows) == 1
    assert rows[0].kind == "PLUGIN_LIFECYCLE_REJECTED"
    detail = rows[0].payload["detail"]
    assert "plugin_id" in detail
    assert "lifecycle" in detail


def test_plugin_lifecycle_with_required_fields_writes_audit_row():
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=13,
        sub_kind=SystemEventKind.PLUGIN_LIFECYCLE,
        source="dashboard",
        payload={"plugin_id": "microstructure_v1", "lifecycle": "ACTIVE"},
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    audit_rows = [r for r in rows if r.kind == "PLUGIN_LIFECYCLE_AUDIT"]
    rejected = [r for r in rows if r.kind == "PLUGIN_LIFECYCLE_REJECTED"]
    assert len(audit_rows) == 1
    assert rejected == []
    assert audit_rows[0].payload["p_plugin_id"] == "microstructure_v1"
    assert audit_rows[0].payload["p_lifecycle"] == "ACTIVE"


# ---------------------------------------------------------------------------
# Regression — known good events still flow
# ---------------------------------------------------------------------------


def test_audit_only_kinds_still_process_as_noop():
    """An audit-only sub_kind must remain NOOP — no UNKNOWN row written."""
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=20,
        sub_kind=SystemEventKind.HEARTBEAT,
        source="dyon",
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    # Heartbeats should not write any governance audit rows.
    unknown = [r for r in rows if r.kind == "UNKNOWN_SYSTEM_KIND"]
    assert unknown == []


def test_update_proposed_missing_field_existing_pattern_still_works():
    """The pre-existing UPDATE_PROPOSED fail-closed path must remain
    intact — Hardening-S1 item 4 is additive, not a refactor of the
    UPDATE_PROPOSED path."""
    eng = GovernanceEngine()
    before = len(eng.ledger.read())
    s = SystemEvent(
        ts_ns=30,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={
            # missing strategy_id
            "parameter": "lookback",
            "old_value": "10",
            "new_value": "20",
            "reason": "drift",
        },
    )
    out = eng.process(s)
    assert out == ()
    rows = eng.ledger.read()[before:]
    rejected = [r for r in rows if r.kind == "UPDATE_REJECTED"]
    assert len(rejected) == 1
    assert rejected[0].payload["code"] == "MALFORMED_PAYLOAD"


# Touch ``replace`` so import lint stays quiet if we later need it for
# constructing edge-case events with mutable copies.
_ = replace
