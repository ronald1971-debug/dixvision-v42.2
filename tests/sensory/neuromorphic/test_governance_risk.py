"""Unit tests for sensory.neuromorphic.governance_risk (NEUR-03)."""

from __future__ import annotations

import pytest

from sensory.neuromorphic.governance_risk import (
    RISK_HAZARD_DENSITY,
    RISK_REJECT_RATE,
    RISK_UNAUTH_DIRECTIVE_RATE,
    DecisionObservation,
    assess_risk,
)


def _o(
    *,
    approved: bool = True,
    had_hazard: bool = False,
    unauthorized_directive: bool = False,
) -> DecisionObservation:
    return DecisionObservation(
        approved=approved,
        had_hazard=had_hazard,
        unauthorized_directive=unauthorized_directive,
    )


def test_reject_rate_zero() -> None:
    r = assess_risk(
        ts_ns=1,
        source="governance.decision_audit",
        risk_kind=RISK_REJECT_RATE,
        window=[_o(approved=True), _o(approved=True)],
    )
    assert r.risk_score == 0.0
    assert r.sample_count == 2


def test_reject_rate_partial() -> None:
    r = assess_risk(
        ts_ns=2,
        source="governance.decision_audit",
        risk_kind=RISK_REJECT_RATE,
        window=[
            _o(approved=True),
            _o(approved=False),
            _o(approved=True),
            _o(approved=False),
        ],
    )
    assert r.risk_score == pytest.approx(0.5)


def test_hazard_density() -> None:
    r = assess_risk(
        ts_ns=3,
        source="governance.decision_audit",
        risk_kind=RISK_HAZARD_DENSITY,
        window=[
            _o(had_hazard=True),
            _o(had_hazard=False),
            _o(had_hazard=False),
            _o(had_hazard=True),
        ],
    )
    assert r.risk_score == 0.5


def test_unauth_directive_rate() -> None:
    r = assess_risk(
        ts_ns=4,
        source="governance.decision_audit",
        risk_kind=RISK_UNAUTH_DIRECTIVE_RATE,
        window=[
            _o(unauthorized_directive=True),
            _o(unauthorized_directive=False),
        ],
    )
    assert r.risk_score == 0.5


def test_unknown_risk_kind_raises() -> None:
    with pytest.raises(ValueError, match="risk_kind"):
        assess_risk(
            ts_ns=5,
            source="governance.decision_audit",
            risk_kind="LIQUIDITY_DRIFT",
            window=[_o()],
        )


def test_empty_window_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        assess_risk(
            ts_ns=6,
            source="governance.decision_audit",
            risk_kind=RISK_REJECT_RATE,
            window=[],
        )


def test_is_deterministic_inv15() -> None:
    window = [_o(approved=True), _o(approved=False)]
    r1 = assess_risk(
        ts_ns=7,
        source="governance.decision_audit",
        risk_kind=RISK_REJECT_RATE,
        window=window,
    )
    r2 = assess_risk(
        ts_ns=7,
        source="governance.decision_audit",
        risk_kind=RISK_REJECT_RATE,
        window=window,
    )
    assert r1 == r2
