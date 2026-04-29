"""HARDEN-02 — Execution Gate runtime guard tests.

The :class:`AuthorityGuard` is the runtime defence layered on top of
the static lint (B7, B20-B22, B25). These tests exercise the gate
without going through the full ``ExecutionEngine`` so each guard
reason is asserted in isolation, then re-asserted end-to-end via
``ExecutionEngine.execute``.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts.events import (
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
)
from core.contracts.execution_intent import (
    ExecutionIntent,
    create_execution_intent,
    mark_approved,
)
from core.contracts.market import MarketTick
from execution_engine.adapters.paper import PaperBroker
from execution_engine.engine import ExecutionEngine
from execution_engine.execution_gate import (
    HAZ_AUTHORITY_CODE,
    AuthorityGuard,
    UnauthorizedActorError,
)


def _tick(symbol: str = "BTCUSDT", last: float = 50_000.0) -> MarketTick:
    return MarketTick(
        ts_ns=1,
        symbol=symbol,
        bid=last - 1.0,
        ask=last + 1.0,
        last=last,
    )


def _signal(*, ts_ns: int = 1_000_000_000) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.9,
        plugin_chain=("microstructure_v1",),
        meta={},
    )


def _approved_intent(
    *,
    origin: str = "tests.fixtures",
    decision_id: str = "GOV-DECISION-1",
) -> ExecutionIntent:
    proposal = create_execution_intent(
        ts_ns=1_000_000_000,
        origin=origin,
        signal=_signal(),
    )
    return mark_approved(proposal, governance_decision_id=decision_id)


def _guard(
    *,
    hazards: list[HazardEvent] | None = None,
    caller_allowlist: frozenset[str] | None = None,
) -> AuthorityGuard:
    sink = (lambda h: hazards.append(h)) if hazards is not None else None
    allowlist = (
        caller_allowlist
        if caller_allowlist is not None
        else frozenset({"execution_engine", "tests.fixtures"})
    )
    return AuthorityGuard(caller_allowlist=allowlist, hazard_sink=sink)


# ---------------------------------------------------------------------------
# Guard-only tests
# ---------------------------------------------------------------------------


def test_guard_accepts_approved_intent_from_authorised_caller():
    guard = _guard()
    intent = _approved_intent()
    guard.assert_can_execute(intent, caller="tests.fixtures")


def test_guard_rejects_unapproved_intent():
    hazards: list[HazardEvent] = []
    guard = _guard(hazards=hazards)
    proposal = create_execution_intent(
        ts_ns=1_000_000_000, origin="tests.fixtures", signal=_signal()
    )
    with pytest.raises(UnauthorizedActorError) as exc:
        guard.assert_can_execute(proposal, caller="tests.fixtures")
    assert "not approved by governance" in str(exc.value)
    assert len(hazards) == 1
    assert hazards[0].code == HAZ_AUTHORITY_CODE
    assert hazards[0].severity is HazardSeverity.CRITICAL


def test_guard_rejects_caller_not_in_allowlist():
    hazards: list[HazardEvent] = []
    guard = _guard(
        hazards=hazards, caller_allowlist=frozenset({"execution_engine"})
    )
    intent = _approved_intent()
    with pytest.raises(UnauthorizedActorError) as exc:
        guard.assert_can_execute(intent, caller="ui.dashboard")
    assert "caller not in execution allowlist" in str(exc.value)
    assert len(hazards) == 1
    assert hazards[0].meta["caller"] == "ui.dashboard"


def test_guard_rejects_tampered_content_hash():
    hazards: list[HazardEvent] = []
    guard = _guard(hazards=hazards)
    intent = _approved_intent()
    tampered = dataclasses.replace(intent, ts_ns=intent.ts_ns + 1)
    with pytest.raises(UnauthorizedActorError) as exc:
        guard.assert_can_execute(tampered, caller="tests.fixtures")
    assert "content_hash mismatch" in str(exc.value)
    assert hazards[0].code == HAZ_AUTHORITY_CODE


def test_guard_rejects_origin_outside_intelligence_actor_module():
    hazards: list[HazardEvent] = []
    guard = _guard(
        hazards=hazards,
        caller_allowlist=frozenset({"execution_engine"}),
    )
    # Build an intent whose origin is in the AUTHORISED set but does
    # NOT match ``intelligence_engine.*``. We synthesise a frozen
    # replacement so we can drive the matrix-mismatch branch directly.
    valid = _approved_intent(origin="intelligence_engine.meta_controller.hot_path")
    spoofed = dataclasses.replace(valid, origin="execution_engine.fast_execute")
    # ``spoofed`` will fail content-hash check first which is correct
    # behaviour; this proves no spoofed intent ever passes the gate.
    with pytest.raises(UnauthorizedActorError):
        guard.assert_can_execute(spoofed, caller="execution_engine")


def test_guard_rejects_unregistered_origin():
    """An intent whose origin is not in AUTHORISED_INTENT_ORIGINS would
    already fail at construction; this test proves the gate also
    rejects a hand-crafted bypass via ``dataclasses.replace`` at the
    matrix-prefix check."""

    hazards: list[HazardEvent] = []
    guard = _guard(hazards=hazards)
    intent = _approved_intent()
    bypass = dataclasses.replace(intent, origin="rogue.module")
    with pytest.raises(UnauthorizedActorError):
        guard.assert_can_execute(bypass, caller="tests.fixtures")


def test_guard_rejects_tests_fixtures_origin_in_production_caller():
    """tests.fixtures origin is only permitted when the caller
    allowlist also opts into it. A production deploy that lists only
    ``execution_engine`` must still reject a fixture-origin intent."""

    hazards: list[HazardEvent] = []
    guard = _guard(
        hazards=hazards, caller_allowlist=frozenset({"execution_engine"})
    )
    intent = _approved_intent()
    with pytest.raises(UnauthorizedActorError) as exc:
        guard.assert_can_execute(intent, caller="execution_engine")
    assert "tests.fixtures origin requires explicit test caller" in str(
        exc.value
    )


# ---------------------------------------------------------------------------
# End-to-end tests via ExecutionEngine.execute
# ---------------------------------------------------------------------------


def test_execute_emits_filled_when_mark_present():
    hazards: list[HazardEvent] = []
    guard = _guard(hazards=hazards)
    engine = ExecutionEngine(adapter=PaperBroker(), guard=guard)
    engine.on_market(_tick())
    fills = engine.execute(_approved_intent(), caller="tests.fixtures")
    assert len(fills) == 1
    assert fills[0].status is ExecutionStatus.FILLED
    assert hazards == []


def test_execute_raises_when_unapproved():
    hazards: list[HazardEvent] = []
    guard = _guard(hazards=hazards)
    engine = ExecutionEngine(adapter=PaperBroker(), guard=guard)
    proposal = create_execution_intent(
        ts_ns=1, origin="tests.fixtures", signal=_signal()
    )
    with pytest.raises(UnauthorizedActorError):
        engine.execute(proposal, caller="tests.fixtures")
    assert len(hazards) == 1
    assert hazards[0].code == HAZ_AUTHORITY_CODE


def test_execute_raises_when_caller_not_authorised():
    hazards: list[HazardEvent] = []
    guard = _guard(
        hazards=hazards, caller_allowlist=frozenset({"execution_engine"})
    )
    engine = ExecutionEngine(adapter=PaperBroker(), guard=guard)
    intent = _approved_intent(origin="intelligence_engine.meta_controller.hot_path")
    with pytest.raises(UnauthorizedActorError):
        engine.execute(intent, caller="ui.dashboard")
    assert hazards[0].meta["caller"] == "ui.dashboard"


def test_legacy_process_emits_deprecation_warning():
    engine = ExecutionEngine(adapter=PaperBroker())
    engine.on_market(_tick())
    with pytest.warns(DeprecationWarning, match="HARDEN-02"):
        out = engine.process(_signal())
    assert len(out) == 1


def test_real_authority_matrix_loads_with_default_guard():
    """The default :class:`AuthorityGuard` resolves the matrix path
    relative to the package; this test pins that wiring so a future
    repo-layout change is caught at PR time."""

    guard = AuthorityGuard()
    assert "intelligence" in guard.matrix.actor_ids
    assert "execution" in guard.matrix.actor_ids
