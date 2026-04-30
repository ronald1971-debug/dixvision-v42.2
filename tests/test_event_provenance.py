"""HARDEN-03 — runtime Triad-Lock provenance assertions (INV-69).

Pairs with the static B20/B21/B22 lint rules. The lint stops obvious
import-time role violations; this suite locks the runtime half of the
Triad Lock against dynamic dispatch (factories, registry-loaded
plugins, mocks).
"""

from __future__ import annotations

import pytest

from core.contracts.event_provenance import (
    EVENT_PRODUCERS,
    EventProvenanceError,
    assert_event_provenance,
    is_event_provenance_known,
)
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _signal(*, producer: str = "intelligence_engine") -> SignalEvent:
    return SignalEvent(
        ts_ns=1_000_000_000,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.5,
        produced_by_engine=producer,
    )


def _execution(*, producer: str = "execution_engine") -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=1_000_000_000,
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=1.0,
        price=100.0,
        status=ExecutionStatus.FILLED,
        produced_by_engine=producer,
    )


def _hazard(*, producer: str = "system_engine") -> HazardEvent:
    return HazardEvent(
        ts_ns=1_000_000_000,
        code="HAZ-99",
        severity=HazardSeverity.MEDIUM,
        source="tests.fixtures",
        detail="synthetic",
        produced_by_engine=producer,
    )


def _system(*, producer: str = "system_engine") -> SystemEvent:
    return SystemEvent(
        ts_ns=1_000_000_000,
        sub_kind=SystemEventKind.HEARTBEAT,
        source="tests.fixtures",
        produced_by_engine=producer,
    )


# ---------------------------------------------------------------------------
# EVENT_PRODUCERS registry shape
# ---------------------------------------------------------------------------


def test_event_producers_keys_are_the_four_canonical_events():
    assert set(EVENT_PRODUCERS) == {SignalEvent, ExecutionEvent, HazardEvent, SystemEvent}


def test_event_producers_values_are_frozensets():
    for cls, producers in EVENT_PRODUCERS.items():
        assert isinstance(producers, frozenset), cls
        assert producers, f"{cls.__name__} producer set is empty"


def test_event_producers_signal_includes_intelligence_and_cognitive():
    # Wave-03 PR-5 added the cognitive sub-prefix so the operator-
    # approval edge can stamp SignalEvents without tripping
    # assert_event_provenance. B26 lint pins the inverse.
    assert EVENT_PRODUCERS[SignalEvent] == frozenset(
        {"intelligence_engine", "intelligence_engine.cognitive"},
    )


def test_event_producers_execution_is_execution_only():
    assert EVENT_PRODUCERS[ExecutionEvent] == frozenset({"execution_engine"})


def test_event_producers_hazard_includes_execution_for_haz_authority():
    # The Execution Gate emits HAZ-AUTHORITY on guard fail (HARDEN-02).
    assert "execution_engine" in EVENT_PRODUCERS[HazardEvent]
    assert "system_engine" in EVENT_PRODUCERS[HazardEvent]


# ---------------------------------------------------------------------------
# assert_event_provenance — happy paths
# ---------------------------------------------------------------------------


def test_signal_with_intelligence_producer_is_accepted():
    assert_event_provenance(_signal())


def test_signal_with_cognitive_producer_is_accepted():
    # Wave-03 PR-5 — the operator-approval edge stamps this prefix when
    # it promotes a queued ProposedSignalApi to a real SignalEvent.
    assert_event_provenance(_signal(producer="intelligence_engine.cognitive"))


def test_execution_with_execution_producer_is_accepted():
    assert_event_provenance(_execution())


def test_hazard_with_system_producer_is_accepted():
    assert_event_provenance(_hazard())


def test_hazard_with_execution_producer_is_accepted_for_haz_authority():
    assert_event_provenance(_hazard(producer="execution_engine"))


def test_system_event_with_each_known_producer_is_accepted():
    for producer in EVENT_PRODUCERS[SystemEvent]:
        assert_event_provenance(_system(producer=producer))


# ---------------------------------------------------------------------------
# assert_event_provenance — Triad-Lock violations
# ---------------------------------------------------------------------------


def test_signal_from_execution_engine_raises():
    with pytest.raises(EventProvenanceError) as excinfo:
        assert_event_provenance(_signal(producer="execution_engine"))
    msg = str(excinfo.value)
    assert "SignalEvent" in msg
    assert "execution_engine" in msg
    assert "intelligence_engine" in msg


def test_signal_from_governance_engine_raises():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_signal(producer="governance_engine"))


def test_execution_from_intelligence_engine_raises():
    with pytest.raises(EventProvenanceError) as excinfo:
        assert_event_provenance(_execution(producer="intelligence_engine"))
    assert "ExecutionEvent" in str(excinfo.value)


def test_execution_from_governance_engine_raises():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_execution(producer="governance_engine"))


def test_hazard_from_intelligence_engine_raises():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_hazard(producer="intelligence_engine"))


def test_hazard_from_governance_engine_raises():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_hazard(producer="governance_engine"))


def test_system_event_from_unknown_producer_raises():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_system(producer="rogue_engine"))


# ---------------------------------------------------------------------------
# assert_event_provenance — strict vs soft on empty produced_by_engine
# ---------------------------------------------------------------------------


def test_strict_mode_rejects_empty_produced_by_engine_signal():
    sig = _signal(producer="")
    with pytest.raises(EventProvenanceError) as excinfo:
        assert_event_provenance(sig)
    assert "empty" in str(excinfo.value).lower()


def test_strict_mode_rejects_empty_produced_by_engine_execution():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_execution(producer=""))


def test_strict_mode_rejects_empty_produced_by_engine_hazard():
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_hazard(producer=""))


def test_soft_mode_allows_empty_produced_by_engine():
    # Backwards-compat path used during migration.
    assert_event_provenance(_signal(producer=""), strict=False)
    assert_event_provenance(_execution(producer=""), strict=False)
    assert_event_provenance(_hazard(producer=""), strict=False)
    assert_event_provenance(_system(producer=""), strict=False)


def test_soft_mode_still_rejects_wrong_producer():
    # Soft mode only relaxes the empty-string check, not Triad Lock.
    with pytest.raises(EventProvenanceError):
        assert_event_provenance(_signal(producer="execution_engine"), strict=False)


# ---------------------------------------------------------------------------
# Unknown event types
# ---------------------------------------------------------------------------


def test_unknown_event_class_raises():
    class NotAnEvent:
        produced_by_engine = "intelligence_engine"

    with pytest.raises(EventProvenanceError) as excinfo:
        assert_event_provenance(NotAnEvent())  # type: ignore[arg-type]
    assert "unknown event class" in str(excinfo.value)


# ---------------------------------------------------------------------------
# is_event_provenance_known — advisory probe
# ---------------------------------------------------------------------------


def test_is_event_provenance_known_returns_true_for_valid_pair():
    assert is_event_provenance_known(_signal()) is True
    assert is_event_provenance_known(_execution()) is True
    assert is_event_provenance_known(_hazard()) is True
    assert is_event_provenance_known(_hazard(producer="execution_engine")) is True


def test_is_event_provenance_known_returns_false_for_violation():
    assert is_event_provenance_known(_signal(producer="execution_engine")) is False
    assert is_event_provenance_known(_execution(producer="intelligence_engine")) is False


def test_is_event_provenance_known_returns_false_for_empty():
    assert is_event_provenance_known(_signal(producer="")) is False


def test_is_event_provenance_known_returns_false_for_unknown_class():
    class NotAnEvent:
        produced_by_engine = "intelligence_engine"

    assert is_event_provenance_known(NotAnEvent()) is False  # type: ignore[arg-type]
