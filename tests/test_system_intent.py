"""Phase 6.T1d — System Intent Engine tests (INV-38).

Covers:

* :class:`SystemIntent` is frozen / read-only.
* :data:`DEFAULT_SYSTEM_INTENT` is the boot default.
* :func:`derive_system_intent` replays ``INTENT_TRANSITION`` ledger rows
  in order, deterministically (INV-15).
* ``StateTransitionManager.propose_intent`` is the only writer of
  ``INTENT_TRANSITION`` ledger rows; rejects malformed enums.
* ``OperatorInterfaceBridge`` routes ``REQUEST_INTENT`` end-to-end and
  the resulting ledger replays back into the same ``SystemIntent``.
* B8 lint rule rejects engine imports inside
  ``core/coherence/system_intent.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.coherence.system_intent import (
    DEFAULT_SYSTEM_INTENT,
    GENESIS_INTENT_ID,
    INTENT_KEY_FOCUS,
    INTENT_KEY_HORIZON,
    INTENT_KEY_OBJECTIVE,
    INTENT_KEY_REASON,
    INTENT_KEY_RISK_MODE,
    INTENT_TRANSITION_KIND,
    SYSTEM_INTENT_VERSION,
    SystemIntent,
    decode_focus,
    derive_system_intent,
    encode_focus,
)
from core.contracts.governance import (
    DecisionKind,
    IntentHorizon,
    IntentObjective,
    IntentRiskMode,
    IntentTransitionRequest,
    OperatorAction,
    OperatorRequest,
)
from governance_engine.control_plane import (
    LedgerAuthorityWriter,
    PolicyEngine,
    StateTransitionManager,
)
from governance_engine.engine import GovernanceEngine
from tools.authority_lint import _check_b8

# ---------------------------------------------------------------------------
# SystemIntent dataclass invariants
# ---------------------------------------------------------------------------


def test_system_intent_is_frozen():
    intent = SystemIntent(
        ts_ns=1,
        objective=IntentObjective.RISK_ADJUSTED_GROWTH,
        risk_mode=IntentRiskMode.BALANCED,
        horizon=IntentHorizon.SHORT_TERM,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.objective = IntentObjective.ABSOLUTE_RETURN  # type: ignore[misc]


def test_default_system_intent_is_capital_preservation_defensive_intraday():
    d = DEFAULT_SYSTEM_INTENT
    assert d.objective is IntentObjective.CAPITAL_PRESERVATION
    assert d.risk_mode is IntentRiskMode.DEFENSIVE
    assert d.horizon is IntentHorizon.INTRADAY
    assert d.focus == ()
    assert d.intent_id == GENESIS_INTENT_ID
    assert d.set_at == -1
    assert d.version == SYSTEM_INTENT_VERSION


def test_focus_round_trip_preserves_order():
    focus = ("crypto_microstructure", "fx_carry", "vol_arb")
    encoded = encode_focus(focus)
    assert decode_focus(encoded) == focus
    assert decode_focus("") == ()


# ---------------------------------------------------------------------------
# derive_system_intent — read-only projection
# ---------------------------------------------------------------------------


def _submit_intent(
    bridge_state: tuple[StateTransitionManager, LedgerAuthorityWriter],
    *,
    ts_ns: int,
    objective: IntentObjective,
    risk_mode: IntentRiskMode,
    horizon: IntentHorizon,
    focus: tuple[str, ...] = (),
    requestor: str = "op",
) -> None:
    state, _ = bridge_state
    state.propose_intent(
        IntentTransitionRequest(
            ts_ns=ts_ns,
            requestor=requestor,
            objective=objective,
            risk_mode=risk_mode,
            horizon=horizon,
            focus=focus,
        )
    )


def _fresh_state() -> tuple[StateTransitionManager, LedgerAuthorityWriter]:
    ledger = LedgerAuthorityWriter()
    state = StateTransitionManager(policy=PolicyEngine(), ledger=ledger)
    return state, ledger


def test_derive_returns_default_when_ledger_has_no_intent_rows():
    ledger = LedgerAuthorityWriter()
    ledger.append(ts_ns=1, kind="MODE_TRANSITION", payload={"x": "y"})
    intent = derive_system_intent(ledger.read())
    assert intent == DEFAULT_SYSTEM_INTENT


def test_derive_returns_latest_intent_after_single_transition():
    state, ledger = _fresh_state()
    _submit_intent(
        (state, ledger),
        ts_ns=10,
        objective=IntentObjective.RISK_ADJUSTED_GROWTH,
        risk_mode=IntentRiskMode.BALANCED,
        horizon=IntentHorizon.SHORT_TERM,
        focus=("crypto_microstructure",),
    )
    intent = derive_system_intent(ledger.read())
    assert intent.objective is IntentObjective.RISK_ADJUSTED_GROWTH
    assert intent.risk_mode is IntentRiskMode.BALANCED
    assert intent.horizon is IntentHorizon.SHORT_TERM
    assert intent.focus == ("crypto_microstructure",)
    assert intent.set_at >= 0
    assert intent.intent_id != GENESIS_INTENT_ID
    assert intent.ts_ns == 10


def test_derive_returns_last_committed_intent_across_many():
    state, ledger = _fresh_state()
    sequence = [
        (
            10,
            IntentObjective.EXPLORATION,
            IntentRiskMode.AGGRESSIVE,
            IntentHorizon.INTRADAY,
            ("a",),
        ),
        (
            20,
            IntentObjective.RISK_ADJUSTED_GROWTH,
            IntentRiskMode.BALANCED,
            IntentHorizon.SHORT_TERM,
            ("b", "c"),
        ),
        (
            30,
            IntentObjective.CAPITAL_PRESERVATION,
            IntentRiskMode.DEFENSIVE,
            IntentHorizon.MEDIUM_TERM,
            (),
        ),
    ]
    for ts, obj, rm, hz, foc in sequence:
        _submit_intent(
            (state, ledger),
            ts_ns=ts,
            objective=obj,
            risk_mode=rm,
            horizon=hz,
            focus=foc,
        )
    intent = derive_system_intent(ledger.read())
    assert intent.objective is IntentObjective.CAPITAL_PRESERVATION
    assert intent.risk_mode is IntentRiskMode.DEFENSIVE
    assert intent.horizon is IntentHorizon.MEDIUM_TERM
    assert intent.focus == ()
    assert intent.ts_ns == 30


def test_derive_is_deterministic_under_replay():
    """INV-15 — replaying the same ledger yields the same intent."""
    state_a, ledger_a = _fresh_state()
    state_b, ledger_b = _fresh_state()
    rows = [
        (
            5,
            IntentObjective.RISK_ADJUSTED_GROWTH,
            IntentRiskMode.BALANCED,
            IntentHorizon.SHORT_TERM,
            ("crypto",),
        ),
        (
            7,
            IntentObjective.ABSOLUTE_RETURN,
            IntentRiskMode.AGGRESSIVE,
            IntentHorizon.INTRADAY,
            ("fx", "rates"),
        ),
    ]
    for ts, obj, rm, hz, foc in rows:
        _submit_intent(
            (state_a, ledger_a),
            ts_ns=ts,
            objective=obj,
            risk_mode=rm,
            horizon=hz,
            focus=foc,
        )
        _submit_intent(
            (state_b, ledger_b),
            ts_ns=ts,
            objective=obj,
            risk_mode=rm,
            horizon=hz,
            focus=foc,
        )
    a = derive_system_intent(ledger_a.read())
    b = derive_system_intent(ledger_b.read())
    assert a == b


def test_derive_skips_non_intent_rows():
    state, ledger = _fresh_state()
    ledger.append(ts_ns=1, kind="MODE_TRANSITION", payload={"x": "y"})
    _submit_intent(
        (state, ledger),
        ts_ns=10,
        objective=IntentObjective.EXPLORATION,
        risk_mode=IntentRiskMode.AGGRESSIVE,
        horizon=IntentHorizon.INTRADAY,
    )
    ledger.append(ts_ns=11, kind="OPERATOR_REJECTED", payload={"x": "y"})
    intent = derive_system_intent(ledger.read())
    assert intent.objective is IntentObjective.EXPLORATION


# ---------------------------------------------------------------------------
# StateTransitionManager.propose_intent — GOV-CP-03 writer
# ---------------------------------------------------------------------------


def test_propose_intent_writes_one_intent_transition_row():
    state, ledger = _fresh_state()
    decision = state.propose_intent(
        IntentTransitionRequest(
            ts_ns=42,
            requestor="op",
            objective=IntentObjective.RISK_ADJUSTED_GROWTH,
            risk_mode=IntentRiskMode.BALANCED,
            horizon=IntentHorizon.SHORT_TERM,
            focus=("crypto_microstructure", "fx_carry"),
            reason="weekly review",
        )
    )
    assert decision.approved is True
    assert decision.ledger_seq >= 0
    rows = list(ledger.read())
    [row] = [r for r in rows if r.kind == INTENT_TRANSITION_KIND]
    assert row.payload[INTENT_KEY_OBJECTIVE] == "RISK_ADJUSTED_GROWTH"
    assert row.payload[INTENT_KEY_RISK_MODE] == "BALANCED"
    assert row.payload[INTENT_KEY_HORIZON] == "SHORT_TERM"
    assert row.payload[INTENT_KEY_REASON] == "weekly review"
    assert decode_focus(row.payload[INTENT_KEY_FOCUS]) == (
        "crypto_microstructure",
        "fx_carry",
    )


def test_propose_intent_rejects_malformed_enum():
    """Tampered ``IntentTransitionRequest`` falls into INTENT_INVALID_ENUM."""
    state, ledger = _fresh_state()
    request = IntentTransitionRequest(
        ts_ns=1,
        requestor="op",
        objective=IntentObjective.RISK_ADJUSTED_GROWTH,
        risk_mode=IntentRiskMode.BALANCED,
        horizon=IntentHorizon.SHORT_TERM,
    )
    object.__setattr__(request, "objective", "NOT_AN_ENUM")
    decision = state.propose_intent(request)
    assert decision.approved is False
    assert decision.rejection_code == "INTENT_INVALID_ENUM"
    assert any(
        r.kind == "INTENT_TRANSITION_REJECTED" for r in ledger.read()
    )
    assert all(
        r.kind != INTENT_TRANSITION_KIND for r in ledger.read()
    )


# ---------------------------------------------------------------------------
# OperatorInterfaceBridge — REQUEST_INTENT routing (GOV-CP-07)
# ---------------------------------------------------------------------------


def test_operator_bridge_request_intent_approved_and_ledger_replays():
    eng = GovernanceEngine()
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=100,
            requestor="ronald",
            action=OperatorAction.REQUEST_INTENT,
            payload={
                INTENT_KEY_OBJECTIVE: "RISK_ADJUSTED_GROWTH",
                INTENT_KEY_RISK_MODE: "BALANCED",
                INTENT_KEY_HORIZON: "SHORT_TERM",
                INTENT_KEY_FOCUS: encode_focus(
                    ("crypto_microstructure", "fx_carry")
                ),
                INTENT_KEY_REASON: "weekly review",
            },
        )
    )
    assert decision.approved is True
    assert decision.kind is DecisionKind.INTENT_TRANSITION
    assert decision.ledger_seq >= 0

    rows = eng.ledger.read()
    intent = derive_system_intent(rows)
    assert intent.objective is IntentObjective.RISK_ADJUSTED_GROWTH
    assert intent.risk_mode is IntentRiskMode.BALANCED
    assert intent.horizon is IntentHorizon.SHORT_TERM
    assert intent.focus == ("crypto_microstructure", "fx_carry")
    assert intent.set_at == decision.ledger_seq


def test_operator_bridge_request_intent_unknown_objective_rejected():
    eng = GovernanceEngine()
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_INTENT,
            payload={
                INTENT_KEY_OBJECTIVE: "BURN_IT_DOWN",
                INTENT_KEY_RISK_MODE: "BALANCED",
                INTENT_KEY_HORIZON: "SHORT_TERM",
            },
        )
    )
    assert decision.approved is False
    assert decision.kind is DecisionKind.REJECTED
    assert decision.rejection_code == "BRIDGE_UNKNOWN_INTENT"
    # No INTENT_TRANSITION row was written.
    assert all(r.kind != INTENT_TRANSITION_KIND for r in eng.ledger.read())


def test_operator_bridge_request_intent_unknown_horizon_rejected():
    eng = GovernanceEngine()
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_INTENT,
            payload={
                INTENT_KEY_OBJECTIVE: "RISK_ADJUSTED_GROWTH",
                INTENT_KEY_RISK_MODE: "BALANCED",
                INTENT_KEY_HORIZON: "FOREVER",
            },
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "BRIDGE_UNKNOWN_INTENT"


def test_operator_bridge_request_intent_locked_is_rejected():
    """Intent submission while LOCKED falls into POLICY_LOCKED, not write."""
    eng = GovernanceEngine()
    eng.operator.submit(
        OperatorRequest(
            ts_ns=1,
            requestor="op",
            action=OperatorAction.REQUEST_KILL,
            payload={},
        )
    )
    decision = eng.operator.submit(
        OperatorRequest(
            ts_ns=2,
            requestor="op",
            action=OperatorAction.REQUEST_INTENT,
            payload={
                INTENT_KEY_OBJECTIVE: "RISK_ADJUSTED_GROWTH",
                INTENT_KEY_RISK_MODE: "BALANCED",
                INTENT_KEY_HORIZON: "SHORT_TERM",
            },
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "POLICY_LOCKED"
    assert all(
        r.kind != INTENT_TRANSITION_KIND for r in eng.ledger.read()
    )


# ---------------------------------------------------------------------------
# Authority Lint — B8 (system_intent isolation)
# ---------------------------------------------------------------------------


_FAKE_FILE = Path("core/coherence/system_intent.py")


def test_b8_allows_core_contracts_import():
    assert (
        _check_b8(
            "core.coherence.system_intent",
            "core.contracts.governance",
            _FAKE_FILE,
            1,
        )
        is None
    )


def test_b8_allows_state_ledger_reader_import():
    assert (
        _check_b8(
            "core.coherence.system_intent",
            "state.ledger.reader",
            _FAKE_FILE,
            1,
        )
        is None
    )


def test_b8_blocks_intelligence_engine_import():
    v = _check_b8(
        "core.coherence.system_intent",
        "intelligence_engine.meta_controller.hot_path",
        _FAKE_FILE,
        1,
    )
    assert v is not None
    assert v.rule == "B8"


def test_b8_blocks_governance_engine_import():
    v = _check_b8(
        "core.coherence.system_intent",
        "governance_engine.control_plane.state_transition_manager",
        _FAKE_FILE,
        1,
    )
    assert v is not None
    assert v.rule == "B8"


def test_b8_blocks_execution_engine_import():
    v = _check_b8(
        "core.coherence.system_intent",
        "execution_engine.hot_path",
        _FAKE_FILE,
        1,
    )
    assert v is not None
    assert v.rule == "B8"


def test_b8_only_targets_system_intent_module():
    """Other ``core.coherence.*`` modules are not subject to B8."""
    assert (
        _check_b8(
            "core.coherence.belief_state",
            "intelligence_engine.meta_controller.hot_path",
            _FAKE_FILE,
            1,
        )
        is None
    )
