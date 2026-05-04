"""AUDIT-WIRE.3 regression tests \u2014 FeedbackCollector + LearningInterface
wired into ExecutionEngine.

P0-3 closure. The harness constructed ExecutionEngine with both sinks
defaulting to ``None``, so the closed-loop dispatch helper short-circuited
to a no-op early return and the learning engine observed zero trade
outcomes. After this PR both sinks are constructed at boot and held on
``_State`` so future tests / read-side callers can drain them.
"""

from __future__ import annotations

import pytest

from execution_engine.protections.feedback import FeedbackCollector
from intelligence_engine.learning_interface import LearningInterface


@pytest.fixture()
def state():
    from ui.server import _State

    return _State()


def test_audit_wire_3_state_owns_feedback_collector(state):
    assert isinstance(state.feedback_collector, FeedbackCollector)
    assert state.execution._feedback_collector is state.feedback_collector


def test_audit_wire_3_state_owns_learning_interface(state):
    assert isinstance(state.learning_interface, LearningInterface)
    assert state.execution._intelligence_feedback is state.learning_interface


def test_audit_wire_3_dispatch_helper_no_longer_short_circuits(state):
    """The early-return guard in ``ExecutionEngine`` only fires when
    BOTH sinks are ``None``. Pinning the wiring with this test stops
    a future refactor that drops one of the two arguments from
    re-introducing a silent learning-loop break."""

    assert (
        state.execution._feedback_collector is not None
        or state.execution._intelligence_feedback is not None
    )
    assert state.execution._feedback_collector is not None
    assert state.execution._intelligence_feedback is not None
