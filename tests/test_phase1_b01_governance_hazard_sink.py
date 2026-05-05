"""Phase 1 backlog item B-01 -- IGovernanceHazardSink Protocol surface.

Asserts:

* ``IGovernanceHazardSink`` is a runtime-checkable Protocol on
  ``core.contracts.governance``.
* ``GovernanceEngine`` satisfies the Protocol (structural check).
* ``accept_hazard(HazardEvent)`` is equivalent to feeding the same
  event through ``process(...)`` -- both append the same
  ``HAZARD_AUDIT`` row to the authority ledger.
* Critical-severity hazards transition the FSM to ``LOCKED`` via the
  typed surface, matching the bus-surface behaviour.
"""

from __future__ import annotations

from core.contracts.events import (
    EventKind,
    HazardEvent,
    HazardSeverity,
)
from core.contracts.governance import (
    IGovernanceHazardSink,
    SystemMode,
)
from governance_engine.engine import GovernanceEngine


def _build_engine() -> GovernanceEngine:
    return GovernanceEngine(initial_mode=SystemMode.PAPER)


def _hazard(
    severity: HazardSeverity = HazardSeverity.MEDIUM,
    code: str = "TEST_HAZARD",
    ts_ns: int = 1_000_000_000,
) -> HazardEvent:
    return HazardEvent(
        ts_ns=ts_ns,
        kind=EventKind.HAZARD,
        produced_by_engine="system_engine",
        code=code,
        severity=severity,
        source="test",
    )


def test_protocol_is_runtime_checkable() -> None:
    """``IGovernanceHazardSink`` is a runtime-checkable Protocol."""

    engine = _build_engine()

    assert isinstance(engine, IGovernanceHazardSink)


def test_accept_hazard_appends_audit_row() -> None:
    """``accept_hazard`` writes the same audit row as ``process``."""

    engine = _build_engine()
    before = len(list(engine.ledger.read()))

    engine.accept_hazard(_hazard())

    after = list(engine.ledger.read())
    assert len(after) == before + 1
    last = after[-1]
    assert last.kind == "HAZARD_AUDIT"
    assert last.payload["code"] == "TEST_HAZARD"
    assert last.payload["severity"] == HazardSeverity.MEDIUM.value


def test_accept_hazard_critical_locks_mode() -> None:
    """Critical hazards delivered through the typed sink trip LOCKED."""

    engine = _build_engine()
    assert engine.state_transitions.current_mode() is SystemMode.PAPER

    engine.accept_hazard(_hazard(severity=HazardSeverity.CRITICAL))

    assert engine.state_transitions.current_mode() is SystemMode.LOCKED


def test_accept_hazard_equivalent_to_process_bus_surface() -> None:
    """``accept_hazard(h)`` and ``process(h)`` produce identical effects."""

    engine_a = _build_engine()
    engine_b = _build_engine()
    haz = _hazard(code="EQUIV_CHECK")

    engine_a.accept_hazard(haz)
    engine_b.process(haz)

    rows_a = [
        (row.kind, row.payload.get("code"), row.payload.get("severity"))
        for row in engine_a.ledger.read()
    ]
    rows_b = [
        (row.kind, row.payload.get("code"), row.payload.get("severity"))
        for row in engine_b.ledger.read()
    ]
    assert rows_a == rows_b
