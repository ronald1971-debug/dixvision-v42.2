"""End-to-end test for the P0-2 hazard throttle chain closure.

Wires HazardThrottleAdapter into ExecutionEngine and verifies that
``apply_throttle()`` is now reachable from the live execute path:

  HazardEvent -> ExecutionEngine.on_hazard ->
  HazardThrottleAdapter.observe -> compute_throttle ->
  apply_throttle -> tightened RiskSnapshot.halted -> REJECTED.
"""

from __future__ import annotations

from core.contracts.events import (
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
)
from core.contracts.execution_intent import (
    create_execution_intent,
    mark_approved,
)
from core.contracts.risk import RiskSnapshot
from execution_engine.adapters.paper import PaperBroker
from execution_engine.engine import ExecutionEngine
from execution_engine.execution_gate import AuthorityGuard
from system_engine.coupling import HazardThrottleAdapter


def _signal(ts_ns: int = 1_000_000_000) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.9,
        plugin_chain=("microstructure_v1",),
        meta={},
    )


def _intent(*, ts_ns: int = 1_000_000_000):
    proposal = create_execution_intent(
        ts_ns=ts_ns,
        origin="tests.fixtures",
        signal=_signal(ts_ns=ts_ns),
    )
    return mark_approved(proposal, governance_decision_id="GOV-DECISION-1")


def _baseline(ts_ns: int = 0) -> RiskSnapshot:
    return RiskSnapshot(
        version=1,
        ts_ns=ts_ns,
        max_position_qty=10.0,
        max_signal_confidence=0.0,
        symbol_caps={},
        halted=False,
    )


def _guard() -> AuthorityGuard:
    return AuthorityGuard(
        caller_allowlist=frozenset({"execution_engine", "tests.fixtures"})
    )


def test_engine_without_throttle_adapter_dispatches_normally() -> None:
    """Pre-P0-2 callers that never wire the adapter keep their behaviour."""

    engine = ExecutionEngine(adapter=PaperBroker(), guard=_guard())
    engine.on_market(
        type(
            "T",
            (),
            {"ts_ns": 1, "symbol": "BTCUSDT", "last": 50_000.0},
        )()
    )

    events = engine.execute(_intent(), caller="tests.fixtures")

    assert len(events) == 1
    assert events[0].status is not ExecutionStatus.REJECTED


def test_engine_short_circuits_when_critical_hazard_halts_snapshot() -> None:
    adapter = HazardThrottleAdapter()
    engine = ExecutionEngine(
        adapter=PaperBroker(),
        guard=_guard(),
        throttle_adapter=adapter,
        risk_baseline=_baseline(),
    )

    engine.on_hazard(
        HazardEvent(
            ts_ns=1_000_000_000,
            code="HAZ-DATA-STALENESS",
            severity=HazardSeverity.CRITICAL,
            detail="critical staleness",
            source="system",
            produced_by_engine="system",
        )
    )

    events = engine.execute(_intent(), caller="tests.fixtures")

    assert len(events) == 1
    event = events[0]
    assert event.status is ExecutionStatus.REJECTED
    assert event.meta["reason"] == "hazard_throttled"
    assert event.qty == 0.0
    assert event.produced_by_engine == "execution_engine"


def test_engine_with_baseline_but_no_hazards_dispatches_normally() -> None:
    """Throttle wired but no observations => no halt, no short-circuit."""

    engine = ExecutionEngine(
        adapter=PaperBroker(),
        guard=_guard(),
        throttle_adapter=HazardThrottleAdapter(),
        risk_baseline=_baseline(),
    )
    engine.on_market(
        type(
            "T",
            (),
            {"ts_ns": 1, "symbol": "BTCUSDT", "last": 50_000.0},
        )()
    )

    events = engine.execute(_intent(), caller="tests.fixtures")

    assert events[0].status is not ExecutionStatus.REJECTED


def test_decayed_hazards_no_longer_halt_execution() -> None:
    adapter = HazardThrottleAdapter()
    engine = ExecutionEngine(
        adapter=PaperBroker(),
        guard=_guard(),
        throttle_adapter=adapter,
        risk_baseline=_baseline(),
    )
    engine.on_market(
        type(
            "T",
            (),
            {"ts_ns": 1, "symbol": "BTCUSDT", "last": 50_000.0},
        )()
    )

    engine.on_hazard(
        HazardEvent(
            ts_ns=1,
            code="HAZ-DATA-STALENESS",
            severity=HazardSeverity.CRITICAL,
            detail="ancient",
            source="system",
            produced_by_engine="system",
        )
    )

    far_future = 10**15
    events = engine.execute(_intent(ts_ns=far_future), caller="tests.fixtures")

    assert events[0].status is not ExecutionStatus.REJECTED
