"""Tests for :mod:`intelligence_engine.cognitive.approval_edge` (Wave-03 PR-5).

Pins the contract that PR-5 closes:

* ``approve`` flips the queue row, *writes the ledger first*, then emits
  the ``SignalEvent`` with the cognitive ``produced_by_engine`` stamp.
* ``reject`` flips the queue row, writes the ledger, never emits.
* Both verbs raise ``ApprovalNotFoundError`` / ``ApprovalAlreadyDecidedError``
  so the route handlers can map to 404 / 409.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import pytest

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalDecisionRequest,
    ApprovalSideApi,
    ApprovalStatusApi,
    ProposedSignalApi,
)
from core.contracts.events import Side, SignalEvent
from intelligence_engine.cognitive.approval_edge import (
    COGNITIVE_PRODUCED_BY_ENGINE,
    ApprovalAlreadyDecidedError,
    ApprovalEdge,
    ApprovalNotFoundError,
)
from intelligence_engine.cognitive.approval_queue import ApprovalQueue


def _proposal(
    symbol: str = "EURUSD",
    side: ApprovalSideApi = ApprovalSideApi.BUY,
) -> ProposedSignalApi:
    return ProposedSignalApi(
        symbol=symbol,
        side=side,
        confidence=0.62,
        rationale="macro setup",
    )


def _make_edge() -> tuple[
    ApprovalEdge,
    list[SignalEvent],
    list[tuple[str, dict[str, str]]],
]:
    """Construct an ApprovalEdge with recording stubs.

    Returns ``(edge, emitted_events, ledger_writes)`` so tests can assert
    the *order* of side effects (ledger first, bus second).
    """

    ids = itertools.count(1)
    queue_clock = itertools.count(1_000)
    edge_clock = itertools.count(2_000)
    queue = ApprovalQueue(
        id_factory=lambda: f"req-{next(ids):04d}",
        ts_ns=lambda: next(queue_clock),
    )
    emitted: list[SignalEvent] = []
    ledger: list[tuple[str, dict[str, str]]] = []

    def signal_emitter(sig: SignalEvent) -> None:
        emitted.append(sig)

    def ledger_append(kind: str, payload: Mapping[str, str]) -> None:
        ledger.append((kind, dict(payload)))

    edge = ApprovalEdge(
        queue=queue,
        signal_emitter=signal_emitter,
        ledger_append=ledger_append,
        ts_ns=lambda: next(edge_clock),
    )
    return edge, emitted, ledger


def test_approve_flips_row_and_emits_signal_with_cognitive_stamp() -> None:
    edge, emitted, ledger = _make_edge()
    submitted = edge.queue.submit(thread_id="t1", proposal=_proposal())
    decided, sig = edge.approve(
        request_id=submitted.request_id,
        decision=ApprovalDecisionRequest(decided_by="op1", note="ok"),
    )
    assert decided.status is ApprovalStatusApi.APPROVED
    assert decided.decided_by == "op1"
    assert sig.symbol == "EURUSD"
    assert sig.side is Side.BUY
    assert sig.confidence == pytest.approx(0.62)
    assert sig.produced_by_engine == COGNITIVE_PRODUCED_BY_ENGINE
    assert "cognitive_chat" in sig.plugin_chain
    assert sig.meta.get("approval_id") == submitted.request_id
    assert sig.meta.get("decided_by") == "op1"
    assert sig.meta.get("approval_note") == "ok"
    assert emitted == [sig]


def test_approve_writes_ledger_before_emitting_to_bus() -> None:
    edge, emitted, ledger = _make_edge()
    submitted = edge.queue.submit(thread_id="t1", proposal=_proposal())

    # Order witness: capture the order of side effects regardless of
    # internal implementation by injecting a recorder that tracks each
    # invocation's call number.
    order: list[str] = []

    def signal_emitter(sig: SignalEvent) -> None:
        order.append("bus")
        emitted.append(sig)

    def ledger_append(kind: str, payload: Mapping[str, str]) -> None:
        order.append(f"ledger:{kind}")
        ledger.append((kind, dict(payload)))

    edge.signal_emitter = signal_emitter
    edge.ledger_append = ledger_append

    edge.approve(
        request_id=submitted.request_id,
        decision=ApprovalDecisionRequest(decided_by="op1"),
    )
    assert order == ["ledger:OPERATOR_APPROVED_SIGNAL", "bus"]


def test_approve_rejects_unknown_id() -> None:
    edge, _, _ = _make_edge()
    with pytest.raises(ApprovalNotFoundError):
        edge.approve(
            request_id="missing",
            decision=ApprovalDecisionRequest(decided_by="op1"),
        )


def test_approve_rejects_already_decided_row() -> None:
    edge, _, _ = _make_edge()
    submitted = edge.queue.submit(thread_id="t1", proposal=_proposal())
    edge.queue.decide(
        request_id=submitted.request_id, approved=False, decided_by="other"
    )
    with pytest.raises(ApprovalAlreadyDecidedError):
        edge.approve(
            request_id=submitted.request_id,
            decision=ApprovalDecisionRequest(decided_by="op1"),
        )


def test_reject_flips_row_and_writes_ledger_without_emit() -> None:
    edge, emitted, ledger = _make_edge()
    submitted = edge.queue.submit(thread_id="t1", proposal=_proposal())
    decided = edge.reject(
        request_id=submitted.request_id,
        decision=ApprovalDecisionRequest(decided_by="op1", note="too noisy"),
    )
    assert decided.status is ApprovalStatusApi.REJECTED
    assert emitted == []
    assert ledger and ledger[0][0] == "OPERATOR_REJECTED_SIGNAL"
    payload = ledger[0][1]
    assert payload["approval_id"] == submitted.request_id
    assert payload["approval_note"] == "too noisy"
    assert payload["decided_by"] == "op1"


def test_reject_rejects_unknown_id() -> None:
    edge, _, _ = _make_edge()
    with pytest.raises(ApprovalNotFoundError):
        edge.reject(
            request_id="missing",
            decision=ApprovalDecisionRequest(decided_by="op1"),
        )


def test_reject_rejects_already_decided_row() -> None:
    edge, _, _ = _make_edge()
    submitted = edge.queue.submit(thread_id="t1", proposal=_proposal())
    edge.queue.decide(
        request_id=submitted.request_id, approved=True, decided_by="other"
    )
    with pytest.raises(ApprovalAlreadyDecidedError):
        edge.reject(
            request_id=submitted.request_id,
            decision=ApprovalDecisionRequest(decided_by="op1"),
        )


def test_approve_emits_sell_side_correctly() -> None:
    edge, emitted, _ = _make_edge()
    submitted = edge.queue.submit(
        thread_id="t1", proposal=_proposal(side=ApprovalSideApi.SELL)
    )
    _, sig = edge.approve(
        request_id=submitted.request_id,
        decision=ApprovalDecisionRequest(decided_by="op1"),
    )
    assert sig.side is Side.SELL
    assert emitted[0].side is Side.SELL
