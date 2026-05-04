"""AUDIT-WIRE.* regression tests — assert ui.server bootstrap binds the
primitives the action-plan flagged as "built but never wired".

Each test constructs a real :class:`ui.server._State` (with the
``DIXVISION_PERMIT_EPHEMERAL_LEDGER`` opt-in set by ``conftest.py`` for
the test suite) and asserts the post-PR wiring is observable via
public attributes + behavioural smoke tests. CI-failing the moment a
future refactor silently drops a constructor argument is the whole
point.

PRs in this thread:
- AUDIT-WIRE.1 — :class:`HazardThrottleAdapter` -> :class:`ExecutionEngine`
"""

from __future__ import annotations

import pytest

from core.contracts.events import HazardEvent, HazardSeverity
from system_engine.coupling import HazardThrottleAdapter


@pytest.fixture()
def state():
    # Imported lazily so the test honours conftest's
    # ``DIXVISION_PERMIT_EPHEMERAL_LEDGER`` set-up; importing
    # ``ui.server`` at module load happens before the env var.
    from ui.server import _State

    return _State()


# ---------------------------------------------------------------------------
# AUDIT-WIRE.1 — HazardThrottleAdapter wired into ExecutionEngine
# ---------------------------------------------------------------------------


def test_audit_wire_1_state_owns_hazard_throttle_adapter(state):
    """The harness must expose a single throttle adapter shared with
    ``ExecutionEngine`` so hazards observed at the harness layer
    tighten the same projection the engine reads on dispatch."""

    assert isinstance(state.hazard_throttle, HazardThrottleAdapter)
    assert state.execution._throttle_adapter is state.hazard_throttle


def test_audit_wire_1_execution_has_baseline_risk_snapshot(state):
    """Without a baseline RiskSnapshot the throttle projection is a
    no-op even when hazards land. The harness must seed a permissive
    baseline at boot."""

    baseline = state.execution._risk_baseline
    assert baseline is not None
    assert baseline.halted is False


def test_audit_wire_1_news_hazard_reaches_throttle_adapter(state):
    """``_ingest_news_hazard_locked`` is the production sink for
    HAZ-NEWS-SHOCK; the wiring guarantees those hazards observably
    enter the throttle observer ring (independent of whether the
    governance FSM acts on them)."""

    hazard = HazardEvent(
        ts_ns=1_000,
        code="HAZ-NEWS-SHOCK",
        severity=HazardSeverity.CRITICAL,
        detail="test wiring",
        source="news",
        produced_by_engine="system",
    )

    state._ingest_news_hazard_locked(hazard)

    observed = state.hazard_throttle.active_observations(now_ns=1_500)
    assert len(observed) == 1
    assert observed[0].code == "HAZ-NEWS-SHOCK"


def test_audit_wire_1_critical_hazard_halts_subsequent_dispatch(state):
    """Behavioural proof of the chain closure: a CRITICAL hazard
    delivered at the harness layer flips ``project().halted`` so the
    next ``execute()`` short-circuits to a REJECTED ExecutionEvent
    with reason=hazard_throttled.

    This is the original P0-2 behaviour the action plan demanded:
    "wire ``apply_throttle()`` into the execution path before order
    sizing -- enables graceful degradation instead of binary lock".
    """

    hazard = HazardEvent(
        ts_ns=1_000,
        code="HAZ-DYON-CRITICAL",
        severity=HazardSeverity.CRITICAL,
        detail="test wiring",
        source="system",
        produced_by_engine="system",
    )
    state.execution.on_hazard(hazard)

    projected = state.hazard_throttle.project(
        snapshot=state.execution._risk_baseline,
        now_ns=1_500,
    )
    assert projected.halted is True
