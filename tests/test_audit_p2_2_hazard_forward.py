"""AUDIT-P2.2 — SystemEngine.process hazards reach Governance.

PR #193 (WIRE.4) bound the ``SensorArray`` into ``SystemEngine``;
PR #202 (P1.3) made ``SystemEngine.process`` actually invoke the
pollable sensors. The remaining gap was that ``ui/server.STATE``
never *called* ``self.system.process(event)`` on the bus path, so
the pollable sensors were polled by nothing and the only hazards
that ever reached ``GovernanceEngine.process`` were the ones the
``NewsFanout`` injected directly through
``_ingest_news_hazard_locked``.

This module pins the contract: every event recorded on the bus
that is **not** itself a ``HazardEvent`` is forwarded into
``self.system.process``, and any ``HazardEvent`` it returns is fed
through the canonical hazard ingestion seam — ``execution.on_hazard``
followed by ``governance.process`` — so the authority ledger gains a
``HAZARD`` audit row and the throttle adapter observes the hazard.
"""

from __future__ import annotations

import importlib

import pytest

from core.contracts.events import HazardEvent, HazardSeverity

ui_server = importlib.import_module("ui.server")


class _FixedHazardSensor:
    """Test sensor that fires HAZ-TEST exactly once on first poll.

    Implements the canonical pollable-sensor signature
    ``observe(ts_ns: int)`` so ``SystemEngine.process`` will pick it
    up via ``_is_pollable``. Arms-once so a single bus event
    produces a single hazard, mirroring real sensor semantics.
    """

    name: str = "test_hazard_sensor"
    code: str = "HAZ-TEST"
    source: str = "tests.test_audit_p2_2_hazard_forward"

    def __init__(self) -> None:
        self._armed = False

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        if self._armed:
            return ()
        self._armed = True
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.LOW,
                source=self.source,
                detail="P2.2 forwarding regression sensor fired",
                meta={"test": "audit-p2-2"},
                produced_by_engine="system_engine",
            ),
        )


@pytest.fixture
def state_with_test_sensor():
    state = ui_server._State()  # type: ignore[attr-defined]
    sensor = _FixedHazardSensor()
    state.sensor_array.register(sensor)
    state.system._pollable = state.system._pollable + (sensor,)
    return state, sensor


def _ledger_kinds(state) -> list[str]:
    return [row.kind for row in state.governance.ledger.read()]


def test_record_forwards_emitted_hazards_into_governance(
    state_with_test_sensor,
):
    """``record`` must invoke ``system.process`` and route the
    returned hazards into ``governance.process`` so the authority
    ledger gains a ``HAZARD`` audit row.
    """

    state, sensor = state_with_test_sensor

    from core.contracts.events import EventKind, SignalEvent, Side

    sig = SignalEvent(
        ts_ns=1_234_000_000,
        symbol="BTC-USD",
        side=Side.BUY,
        confidence=0.6,
        produced_by_engine="intelligence_engine",
    )
    assert sig.kind == EventKind.SIGNAL

    before = _ledger_kinds(state)
    state.record("test", sig)
    after = _ledger_kinds(state)

    assert sensor._armed is True, (
        "SystemEngine.process was never invoked from STATE.record; "
        "the AUDIT-P2.2 forwarding wiring regressed"
    )
    new_kinds = after[len(before):]
    assert "HAZARD_AUDIT" in new_kinds, (
        "SystemEngine emitted a HazardEvent but the canonical "
        "ingestion seam did not append a HAZARD_AUDIT audit row "
        f"(new ledger kinds: {new_kinds})"
    )


def test_record_does_not_repoll_on_hazard_events(
    state_with_test_sensor,
):
    """Polling ``SystemEngine.process`` on a ``HazardEvent`` would
    re-invoke the same sensor on every emitted hazard and could
    arm-cycle when sensors share keys. The forwarding branch must
    skip ``EventKind.HAZARD`` to keep the loop bounded.
    """

    state, sensor = state_with_test_sensor

    hazard = HazardEvent(
        ts_ns=2_222_000_000,
        code="HAZ-FAKE",
        severity=HazardSeverity.LOW,
        source="tests.test_audit_p2_2_hazard_forward",
        detail="external hazard recorded directly",
        meta={},
        produced_by_engine="system_engine",
    )

    before = _ledger_kinds(state)
    state.record("test", hazard)
    after = _ledger_kinds(state)

    assert sensor._armed is False, (
        "STATE.record polled SystemEngine on a HAZARD event; this "
        "would create an unbounded fan-out when multiple sensors "
        "share keys"
    )
    new_kinds = after[len(before):]
    assert new_kinds.count("HAZARD_AUDIT") <= 1, (
        "Recording a HazardEvent should produce at most one "
        "HAZARD_AUDIT ledger row; sensor re-polling regressed"
    )
