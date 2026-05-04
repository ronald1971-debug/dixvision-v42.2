"""AUDIT-P1.1 regression tests — DecisionSigner wired into AuthorityGuard.

Hardening-S1 item 2 closure. The ``DecisionSigner`` HMAC primitive
(PR #170) and the ``AuthorityGuard`` verifier hook (same PR) had
existed for weeks, but the production harness in :mod:`ui.server`
never constructed a signer or injected a verifier. Every
``ExecutionEngine`` built in the harness path therefore lazily
materialised an ``AuthorityGuard()`` with ``signature_verifier=None``,
and the documented "no signature -> no execution" guarantee was
inoperative in production.

After this PR the harness:

* constructs one :class:`DecisionSigner` at boot (``state.decision_signer``);
* constructs one :class:`AuthorityGuard` whose
  ``signature_verifier`` calls
  :meth:`DecisionSigner.verify` (``state.authority_guard``);
* injects the guard into the production
  :class:`ExecutionEngine` (``state.execution._guard``);
* threads the signer into every harness call site of
  :func:`approve_signal_for_execution` so approved intents carry a
  non-empty ``decision_signature``.

The tests below pin each link of the chain, plus assert that an
unsigned intent is rejected by the live wiring and that a forged
signature is rejected too. This makes the ``HMAC enforced everywhere``
property a continuously-tested invariant rather than a doc claim.
"""

from __future__ import annotations

import pytest

from core.contracts.events import Side, SignalEvent
from core.contracts.execution_intent import (
    create_execution_intent,
    mark_approved,
)
from execution_engine.execution_gate import (
    AuthorityGuard,
    UnauthorizedActorError,
)
from governance_engine.control_plane.decision_signer import DecisionSigner
from governance_engine.harness_approver import (
    DEFAULT_HARNESS_ORIGIN,
    approve_signal_for_execution,
)


@pytest.fixture()
def state():
    from ui.server import _State

    return _State()


def _signal(ts_ns: int = 1_700_000_000) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="BTC-USD",
        side=Side.BUY,
        confidence=0.7,
        produced_by_engine="intelligence_engine.signal_pipeline.orchestrator",
    )


def test_audit_p1_1_state_owns_decision_signer(state):
    """The harness must hold the canonical signer + guard pair."""

    assert isinstance(state.decision_signer, DecisionSigner)
    assert isinstance(state.authority_guard, AuthorityGuard)


def test_audit_p1_1_execution_engine_received_the_guard(state):
    """The ExecutionEngine guard must be the same instance the
    harness constructed -- not a lazily-materialised default that
    bypasses the verifier."""

    assert state.execution.guard is state.authority_guard


def test_audit_p1_1_harness_approved_intent_carries_signature(state):
    intent = approve_signal_for_execution(
        _signal(),
        ts_ns=1_700_000_000,
        signer=state.decision_signer,
    )

    assert intent.decision_signature
    assert intent.approved_by_governance
    # The guard accepts the signed intent without raising.
    state.authority_guard.assert_can_execute(
        intent, caller="execution_engine"
    )


def test_audit_p1_1_unsigned_intent_rejected_by_live_guard(state):
    intent = approve_signal_for_execution(
        _signal(),
        ts_ns=1_700_000_000,
        # No signer -> empty decision_signature.
    )
    assert intent.decision_signature == ""
    with pytest.raises(UnauthorizedActorError):
        state.authority_guard.assert_can_execute(
            intent, caller="execution_engine"
        )


def test_audit_p1_1_forged_signature_rejected_by_live_guard(state):
    base = create_execution_intent(
        ts_ns=1_700_000_000,
        origin=DEFAULT_HARNESS_ORIGIN,
        signal=_signal(),
    )
    forged = mark_approved(
        base,
        governance_decision_id="harness:auto:1700000000",
        decision_signature="00" * 32,  # plausible-looking but unsigned
    )
    with pytest.raises(UnauthorizedActorError):
        state.authority_guard.assert_can_execute(
            forged, caller="execution_engine"
        )
