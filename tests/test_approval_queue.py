"""Tests for :mod:`intelligence_engine.cognitive.approval_queue` (Wave-03 PR-5).

The queue is the operator-approval ledger: it lands a chat proposal as
``PENDING`` and flips it once to ``APPROVED`` / ``REJECTED``. Every
transition is exercised here so the route handlers in PR-5 can rely on
the exception contract (``ApprovalNotFoundError`` → 404,
``ApprovalAlreadyDecidedError`` → 409).
"""

from __future__ import annotations

import itertools

import pytest

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalSideApi,
    ApprovalStatusApi,
    ProposedSignalApi,
)
from intelligence_engine.cognitive.approval_queue import (
    ApprovalAlreadyDecidedError,
    ApprovalNotFoundError,
    ApprovalQueue,
)


def _proposal(
    symbol: str = "EURUSD",
    side: ApprovalSideApi = ApprovalSideApi.BUY,
) -> ProposedSignalApi:
    return ProposedSignalApi(
        symbol=symbol,
        side=side,
        confidence=0.75,
        rationale="unit test",
    )


def _make_queue() -> ApprovalQueue:
    """Deterministic queue: monotonic id counter + monotonic clock."""

    ids = itertools.count(1)
    clock = itertools.count(1_000)
    return ApprovalQueue(
        id_factory=lambda: f"req-{next(ids):04d}",
        ts_ns=lambda: next(clock),
    )


def test_submit_lands_pending_row_with_injected_ts() -> None:
    queue = _make_queue()
    row = queue.submit(thread_id="t1", proposal=_proposal(), requested_at_ts_ns=42)
    assert row.request_id == "req-0001"
    assert row.status is ApprovalStatusApi.PENDING
    assert row.requested_at_ts_ns == 42
    assert row.decided_at_ts_ns is None
    assert row.decided_by == ""
    assert len(queue) == 1


def test_submit_uses_clock_when_ts_omitted() -> None:
    queue = _make_queue()
    row = queue.submit(thread_id="t1", proposal=_proposal())
    # _make_queue's clock starts at 1_000.
    assert row.requested_at_ts_ns == 1_000


def test_submit_rejects_hold_proposal() -> None:
    queue = _make_queue()
    with pytest.raises(ValueError, match="HOLD"):
        queue.submit(thread_id="t1", proposal=_proposal(side=ApprovalSideApi.HOLD))


def test_submit_raises_on_id_collision() -> None:
    """A buggy id_factory must not silently overwrite an existing row."""

    queue = ApprovalQueue(id_factory=lambda: "fixed", ts_ns=lambda: 1)
    queue.submit(thread_id="t1", proposal=_proposal())
    with pytest.raises(RuntimeError, match="collision"):
        queue.submit(thread_id="t2", proposal=_proposal())


def test_decide_flips_pending_to_approved() -> None:
    queue = _make_queue()
    row = queue.submit(thread_id="t1", proposal=_proposal(), requested_at_ts_ns=42)
    decided = queue.decide(request_id=row.request_id, approved=True, decided_by="op1")
    assert decided.status is ApprovalStatusApi.APPROVED
    assert decided.decided_by == "op1"
    assert decided.decided_at_ts_ns == 1_000  # first ts_ns() call
    # original row's other fields preserved
    assert decided.proposal.symbol == "EURUSD"
    assert decided.requested_at_ts_ns == 42


def test_decide_flips_pending_to_rejected() -> None:
    queue = _make_queue()
    row = queue.submit(thread_id="t1", proposal=_proposal(), requested_at_ts_ns=42)
    decided = queue.decide(request_id=row.request_id, approved=False, decided_by="op1")
    assert decided.status is ApprovalStatusApi.REJECTED


def test_decide_unknown_id_raises_not_found() -> None:
    queue = _make_queue()
    with pytest.raises(ApprovalNotFoundError):
        queue.decide(request_id="missing", approved=True, decided_by="op1")


def test_decide_already_decided_raises_already_decided() -> None:
    queue = _make_queue()
    row = queue.submit(thread_id="t1", proposal=_proposal(), requested_at_ts_ns=42)
    queue.decide(request_id=row.request_id, approved=True, decided_by="op1")
    with pytest.raises(ApprovalAlreadyDecidedError):
        queue.decide(request_id=row.request_id, approved=True, decided_by="op2")
    with pytest.raises(ApprovalAlreadyDecidedError):
        queue.decide(request_id=row.request_id, approved=False, decided_by="op2")


def test_get_returns_current_row_after_decision() -> None:
    queue = _make_queue()
    row = queue.submit(thread_id="t1", proposal=_proposal(), requested_at_ts_ns=42)
    queue.decide(request_id=row.request_id, approved=True, decided_by="op1")
    fetched = queue.get(row.request_id)
    assert fetched.status is ApprovalStatusApi.APPROVED


def test_get_unknown_id_raises_not_found() -> None:
    queue = _make_queue()
    with pytest.raises(ApprovalNotFoundError):
        queue.get("missing")


def test_list_default_pending_only_in_submission_order() -> None:
    queue = _make_queue()
    a = queue.submit(thread_id="t1", proposal=_proposal("AAA"), requested_at_ts_ns=1)
    b = queue.submit(thread_id="t1", proposal=_proposal("BBB"), requested_at_ts_ns=2)
    c = queue.submit(thread_id="t1", proposal=_proposal("CCC"), requested_at_ts_ns=3)
    queue.decide(request_id=b.request_id, approved=True, decided_by="op1")
    snapshot = queue.list()
    assert [r.request_id for r in snapshot] == [a.request_id, c.request_id]


def test_list_include_decided_returns_full_history() -> None:
    queue = _make_queue()
    a = queue.submit(thread_id="t1", proposal=_proposal("AAA"), requested_at_ts_ns=1)
    b = queue.submit(thread_id="t1", proposal=_proposal("BBB"), requested_at_ts_ns=2)
    queue.decide(request_id=b.request_id, approved=False, decided_by="op1")
    snapshot = queue.list(include_decided=True)
    assert [r.request_id for r in snapshot] == [a.request_id, b.request_id]
    assert snapshot[1].status is ApprovalStatusApi.REJECTED
