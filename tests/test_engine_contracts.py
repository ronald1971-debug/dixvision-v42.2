"""Phase E0 — Engine instantiation + contract conformance tests."""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts.engine import (
    EngineTier,
    HealthState,
    OfflineEngine,
    RuntimeEngine,
)
from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from evolution_engine import EvolutionEngine
from execution_engine import ExecutionEngine
from governance_engine import GovernanceEngine
from intelligence_engine import IntelligenceEngine
from learning_engine import LearningEngine
from system_engine import SystemEngine

RUNTIME_ENGINE_CLASSES = (
    IntelligenceEngine,
    ExecutionEngine,
    SystemEngine,
    GovernanceEngine,
)

OFFLINE_ENGINE_CLASSES = (
    LearningEngine,
    EvolutionEngine,
)


@pytest.mark.parametrize("cls", RUNTIME_ENGINE_CLASSES)
def test_runtime_engine_instantiates(cls):
    eng = cls()
    assert isinstance(eng, RuntimeEngine)
    assert eng.tier is EngineTier.RUNTIME
    assert eng.name in {"intelligence", "execution", "system", "governance"}
    status = eng.check_self()
    assert status.state is HealthState.OK


@pytest.mark.parametrize("cls", OFFLINE_ENGINE_CLASSES)
def test_offline_engine_instantiates(cls):
    eng = cls()
    assert isinstance(eng, OfflineEngine)
    assert eng.tier is EngineTier.OFFLINE
    assert eng.name in {"learning", "evolution"}
    # OfflineEngine.schedule must return a non-empty cron string.
    assert eng.schedule()
    status = eng.check_self()
    assert status.state is HealthState.OK


def test_intelligence_engine_signal_passthrough():
    eng = IntelligenceEngine()
    sig = SignalEvent(ts_ns=1, symbol="EURUSD", side=Side.HOLD, confidence=0.0)
    out = eng.process(sig)
    assert tuple(out) == (sig,)


def test_event_types_have_kind_discriminator():
    assert SignalEvent(
        ts_ns=0, symbol="X", side=Side.BUY, confidence=0.5
    ).kind is EventKind.SIGNAL
    assert ExecutionEvent(
        ts_ns=0,
        symbol="X",
        side=Side.SELL,
        qty=1.0,
        price=1.0,
        status=ExecutionStatus.PROPOSED,
    ).kind is EventKind.EXECUTION
    assert SystemEvent(
        ts_ns=0,
        sub_kind=SystemEventKind.HEARTBEAT,
        source="system",
    ).kind is EventKind.SYSTEM
    assert HazardEvent(
        ts_ns=0,
        code="HAZ-01",
        severity=HazardSeverity.LOW,
        source="system",
    ).kind is EventKind.HAZARD


def test_events_are_immutable():
    sig = SignalEvent(ts_ns=0, symbol="X", side=Side.BUY, confidence=0.5)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        sig.symbol = "Y"  # type: ignore[misc]


def test_engine_tier_registry_consistency():
    # Mirror of registry/engines.yaml — keep these in lockstep.
    expected = {
        "intelligence": EngineTier.RUNTIME,
        "execution": EngineTier.RUNTIME,
        "system": EngineTier.RUNTIME,
        "governance": EngineTier.RUNTIME,
        "learning": EngineTier.OFFLINE,
        "evolution": EngineTier.OFFLINE,
    }
    actual = {
        eng().name: eng().tier
        for eng in (*RUNTIME_ENGINE_CLASSES, *OFFLINE_ENGINE_CLASSES)
    }
    assert actual == expected
