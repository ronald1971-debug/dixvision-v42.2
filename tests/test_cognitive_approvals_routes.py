"""Tests for the /api/cognitive/chat/approvals/* HTTP routes (Wave-03 PR-5).

Drives the FastAPI surface with the live ``STATE`` (the harness uses a
process-wide singleton — there's no per-request DI). Each test cleans
up its own queue rows via the tail of the queue's order list so the
suite stays order-independent.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalSideApi,
    ApprovalStatusApi,
    ProposedSignalApi,
)
from intelligence_engine.cognitive.approval_queue import ApprovalQueue
from ui import server as server_module
from ui.server import STATE, app


def _proposal(symbol: str = "EURUSD") -> ProposedSignalApi:
    return ProposedSignalApi(
        symbol=symbol,
        side=ApprovalSideApi.BUY,
        confidence=0.55,
        rationale="route test",
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_queue() -> Iterator[None]:
    """Replace the runtime's queue with an empty one for each test.

    The chat runtime + STATE singletons hold a queue across the whole
    test session; isolating per-test prevents bleed-through.
    """

    runtime = STATE.chat_runtime
    original_queue = runtime.approval_queue
    original_edge_queue = STATE.approval_edge.queue

    fresh = ApprovalQueue()
    runtime.approval_queue = fresh
    STATE.approval_edge.queue = fresh

    # Snapshot the ledger length so per-test assertions can pin the
    # *new* rows independently of unrelated harness writes.
    server_module.STATE = STATE  # ensure rebound runtime is visible

    try:
        yield
    finally:
        runtime.approval_queue = original_queue
        STATE.approval_edge.queue = original_edge_queue


def test_list_returns_pending_only_by_default(client: TestClient) -> None:
    submitted = STATE.chat_runtime.approval_queue.submit(
        thread_id="t1", proposal=_proposal()
    )
    res = client.get("/api/cognitive/chat/approvals")
    assert res.status_code == 200
    data = res.json()
    assert len(data["requests"]) == 1
    assert data["requests"][0]["request_id"] == submitted.request_id
    assert data["requests"][0]["status"] == "PENDING"


def test_list_excludes_decided_when_default(client: TestClient) -> None:
    queue = STATE.chat_runtime.approval_queue
    queue.submit(thread_id="t1", proposal=_proposal("AAA"))
    decided = queue.submit(thread_id="t1", proposal=_proposal("BBB"))
    queue.decide(
        request_id=decided.request_id, approved=False, decided_by="op1"
    )
    res = client.get("/api/cognitive/chat/approvals")
    assert res.status_code == 200
    data = res.json()
    assert [r["proposal"]["symbol"] for r in data["requests"]] == ["AAA"]


def test_list_include_decided_returns_full_history(
    client: TestClient,
) -> None:
    queue = STATE.chat_runtime.approval_queue
    pending = queue.submit(thread_id="t1", proposal=_proposal("AAA"))
    decided = queue.submit(thread_id="t1", proposal=_proposal("BBB"))
    queue.decide(
        request_id=decided.request_id, approved=False, decided_by="op1"
    )
    res = client.get(
        "/api/cognitive/chat/approvals", params={"include_decided": "true"}
    )
    assert res.status_code == 200
    data = res.json()
    ids = [r["request_id"] for r in data["requests"]]
    assert pending.request_id in ids
    assert decided.request_id in ids


def test_approve_flips_row_and_writes_ledger(client: TestClient) -> None:
    submitted = STATE.chat_runtime.approval_queue.submit(
        thread_id="t1", proposal=_proposal()
    )
    ledger_before = len(STATE.governance.ledger)
    res = client.post(
        f"/api/cognitive/chat/approvals/{submitted.request_id}/approve",
        json={"decided_by": "op1"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["request"]["status"] == "APPROVED"
    assert data["request"]["decided_by"] == "op1"
    assert data["emitted_signal_id"] != ""
    # Lookup confirms persistence
    persisted = STATE.chat_runtime.approval_queue.get(submitted.request_id)
    assert persisted.status is ApprovalStatusApi.APPROVED
    # Ledger row was appended
    ledger_after = STATE.governance.ledger.read()
    assert len(ledger_after) > ledger_before
    new_rows = [r.kind for r in ledger_after[ledger_before:]]
    assert "OPERATOR_APPROVED_SIGNAL" in new_rows


def test_approve_with_empty_body_uses_defaults(client: TestClient) -> None:
    submitted = STATE.chat_runtime.approval_queue.submit(
        thread_id="t1", proposal=_proposal()
    )
    res = client.post(
        f"/api/cognitive/chat/approvals/{submitted.request_id}/approve",
    )
    assert res.status_code == 200
    # ApprovalDecisionRequest defaults decided_by to "operator" so an
    # empty client body still produces a non-empty audit attribution.
    assert res.json()["request"]["decided_by"] == "operator"


def test_approve_unknown_id_returns_404(client: TestClient) -> None:
    res = client.post(
        "/api/cognitive/chat/approvals/missing-id/approve",
        json={"decided_by": "op1"},
    )
    assert res.status_code == 404


def test_approve_already_decided_returns_409(client: TestClient) -> None:
    submitted = STATE.chat_runtime.approval_queue.submit(
        thread_id="t1", proposal=_proposal()
    )
    STATE.chat_runtime.approval_queue.decide(
        request_id=submitted.request_id, approved=False, decided_by="other"
    )
    res = client.post(
        f"/api/cognitive/chat/approvals/{submitted.request_id}/approve",
        json={"decided_by": "op1"},
    )
    assert res.status_code == 409


def test_reject_flips_row_and_writes_ledger(client: TestClient) -> None:
    submitted = STATE.chat_runtime.approval_queue.submit(
        thread_id="t1", proposal=_proposal()
    )
    ledger_before = len(STATE.governance.ledger)
    res = client.post(
        f"/api/cognitive/chat/approvals/{submitted.request_id}/reject",
        json={"decided_by": "op1", "note": "no"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["request"]["status"] == "REJECTED"
    assert data["emitted_signal_id"] == ""
    ledger_after = STATE.governance.ledger.read()
    new_rows = [r.kind for r in ledger_after[ledger_before:]]
    assert "OPERATOR_REJECTED_SIGNAL" in new_rows


def test_reject_unknown_id_returns_404(client: TestClient) -> None:
    res = client.post(
        "/api/cognitive/chat/approvals/missing-id/reject",
        json={"decided_by": "op1"},
    )
    assert res.status_code == 404


def test_reject_already_decided_returns_409(client: TestClient) -> None:
    submitted = STATE.chat_runtime.approval_queue.submit(
        thread_id="t1", proposal=_proposal()
    )
    STATE.chat_runtime.approval_queue.decide(
        request_id=submitted.request_id, approved=True, decided_by="other"
    )
    res = client.post(
        f"/api/cognitive/chat/approvals/{submitted.request_id}/reject",
        json={"decided_by": "op1"},
    )
    assert res.status_code == 409
