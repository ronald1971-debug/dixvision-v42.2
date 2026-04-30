"""Wave-03 PR-7 — projection tests for the ledger-backed approval queue.

Covers:

* Pure-projection contract — same rows in, same rows out.
* PENDING → APPROVED / REJECTED transitions resolved in order.
* Insertion-order preservation across mixed kinds.
* Defensive behaviours: orphan decision row, duplicate PENDING,
  duplicate decision, unknown kind.
* End-to-end "restart safety": submit / approve / reject sequence on
  one queue, ledger replay into a second queue, public surfaces
  identical.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalSideApi,
    ApprovalStatusApi,
    ProposedSignalApi,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from intelligence_engine.cognitive.approval_projection import (
    APPROVED_KIND,
    DECISION_KINDS,
    PENDING_KIND,
    REJECTED_KIND,
    ProjectionLedgerRow,
    projection_rows_from_payloads,
)
from intelligence_engine.cognitive.approval_queue import ApprovalQueue
from ui.cognitive_chat_runtime import rehydrate_approval_queue_from_ledger


def _pending_row(
    *,
    approval_id: str,
    thread_id: str = "thr-1",
    symbol: str = "EURUSD",
    side: str = "BUY",
    confidence: str = "0.700000",
    rationale: str = "test",
    ts_ns: str = "1000",
) -> ProjectionLedgerRow:
    return ProjectionLedgerRow(
        kind=PENDING_KIND,
        payload={
            "approval_id": approval_id,
            "thread_id": thread_id,
            "symbol": symbol,
            "side": side,
            "confidence": confidence,
            "rationale": rationale,
            "ts_ns": ts_ns,
        },
    )


def _decision_row(
    *,
    kind: str,
    approval_id: str,
    decided_by: str = "operator-1",
    ts_ns: str = "2000",
) -> ProjectionLedgerRow:
    return ProjectionLedgerRow(
        kind=kind,
        payload={
            "approval_id": approval_id,
            "decided_by": decided_by,
            "ts_ns": ts_ns,
        },
    )


# ---------------------------------------------------------------------------
# Pure projection contract
# ---------------------------------------------------------------------------


def test_empty_chain_yields_empty_projection() -> None:
    assert projection_rows_from_payloads([]) == ()


def test_single_pending_row_lands_as_pending() -> None:
    rows = projection_rows_from_payloads([_pending_row(approval_id="a-1")])
    assert len(rows) == 1
    rid, req = rows[0]
    assert rid == "a-1"
    assert req.status is ApprovalStatusApi.PENDING
    assert req.proposal.symbol == "EURUSD"
    assert req.proposal.side is ApprovalSideApi.BUY
    assert abs(req.proposal.confidence - 0.7) < 1e-9
    assert req.proposal.rationale == "test"
    assert req.requested_at_ts_ns == 1000
    assert req.decided_at_ts_ns is None
    assert req.decided_by == ""


def test_pending_then_approve_resolves_to_approved() -> None:
    rows = projection_rows_from_payloads(
        [
            _pending_row(approval_id="a-1"),
            _decision_row(
                kind=APPROVED_KIND, approval_id="a-1", ts_ns="2500"
            ),
        ]
    )
    assert len(rows) == 1
    _, req = rows[0]
    assert req.status is ApprovalStatusApi.APPROVED
    assert req.decided_at_ts_ns == 2500
    assert req.decided_by == "operator-1"


def test_pending_then_reject_resolves_to_rejected() -> None:
    rows = projection_rows_from_payloads(
        [
            _pending_row(approval_id="a-1"),
            _decision_row(kind=REJECTED_KIND, approval_id="a-1"),
        ]
    )
    _, req = rows[0]
    assert req.status is ApprovalStatusApi.REJECTED


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_insertion_order_preserved_across_mixed_kinds() -> None:
    rows = projection_rows_from_payloads(
        [
            _pending_row(approval_id="a-1", ts_ns="1000"),
            _pending_row(approval_id="a-2", ts_ns="1100"),
            _decision_row(kind=APPROVED_KIND, approval_id="a-1"),
            _pending_row(approval_id="a-3", ts_ns="1200"),
            _decision_row(kind=REJECTED_KIND, approval_id="a-2"),
        ]
    )
    assert [rid for rid, _ in rows] == ["a-1", "a-2", "a-3"]
    by_id = dict(rows)
    assert by_id["a-1"].status is ApprovalStatusApi.APPROVED
    assert by_id["a-2"].status is ApprovalStatusApi.REJECTED
    assert by_id["a-3"].status is ApprovalStatusApi.PENDING


# ---------------------------------------------------------------------------
# Defensive behaviours
# ---------------------------------------------------------------------------


def test_orphan_decision_row_is_skipped() -> None:
    rows = projection_rows_from_payloads(
        [_decision_row(kind=APPROVED_KIND, approval_id="ghost")]
    )
    assert rows == ()


def test_duplicate_pending_keeps_first_and_ignores_second() -> None:
    rows = projection_rows_from_payloads(
        [
            _pending_row(approval_id="a-1", confidence="0.500000", ts_ns="1000"),
            _pending_row(approval_id="a-1", confidence="0.900000", ts_ns="1100"),
        ]
    )
    assert len(rows) == 1
    _, req = rows[0]
    assert abs(req.proposal.confidence - 0.5) < 1e-9
    assert req.requested_at_ts_ns == 1000


def test_duplicate_decision_does_not_overwrite_first() -> None:
    rows = projection_rows_from_payloads(
        [
            _pending_row(approval_id="a-1"),
            _decision_row(
                kind=APPROVED_KIND,
                approval_id="a-1",
                decided_by="op-A",
                ts_ns="2000",
            ),
            _decision_row(
                kind=REJECTED_KIND,
                approval_id="a-1",
                decided_by="op-B",
                ts_ns="3000",
            ),
        ]
    )
    _, req = rows[0]
    assert req.status is ApprovalStatusApi.APPROVED
    assert req.decided_by == "op-A"
    assert req.decided_at_ts_ns == 2000


def test_unrelated_ledger_kinds_are_ignored() -> None:
    rows = projection_rows_from_payloads(
        [
            ProjectionLedgerRow(kind="MODE_TRANSITION", payload={"to": "LIVE"}),
            _pending_row(approval_id="a-1"),
            ProjectionLedgerRow(kind="HAZ-AUTHORITY", payload={"sev": "HIGH"}),
        ]
    )
    assert len(rows) == 1
    assert rows[0][0] == "a-1"


def test_kind_constants_are_disjoint() -> None:
    assert PENDING_KIND not in DECISION_KINDS
    assert {APPROVED_KIND, REJECTED_KIND} == set(DECISION_KINDS)


# ---------------------------------------------------------------------------
# Restart safety: end-to-end queue ↔ ledger ↔ queue equivalence
# ---------------------------------------------------------------------------


def _deterministic_id_factory() -> Callable[[], str]:
    counter = itertools.count(1)

    def _next() -> str:
        return f"req-{next(counter):03d}"

    return _next


def _build_runtime_pair() -> tuple[
    LedgerAuthorityWriter, ApprovalQueue, list[int]
]:
    """Set up a queue + ledger pair, mirroring the live wiring in
    ``ui.cognitive_chat_runtime``.

    Returns ``(ledger, queue, ts_box)``. ``ts_box[0]`` is a
    monotonic counter the test ticks before each ledger write so the
    rows stay in insertion order without depending on wall clock.
    """

    ledger = LedgerAuthorityWriter()
    ts_box = [1_000]

    def _ts() -> int:
        ts_box[0] += 1
        return ts_box[0]

    queue = ApprovalQueue(id_factory=_deterministic_id_factory(), ts_ns=_ts)
    return ledger, queue, ts_box


def _ledger_submit(
    ledger: LedgerAuthorityWriter,
    queue: ApprovalQueue,
    *,
    thread_id: str,
    proposal: ProposedSignalApi,
    ts_ns: int,
) -> str:
    """Mirror what ``CognitiveChatRuntime.handle_turn`` does on submit."""

    queued = queue.submit(
        thread_id=thread_id, proposal=proposal, requested_at_ts_ns=ts_ns
    )
    ledger.append(
        ts_ns=ts_ns,
        kind=PENDING_KIND,
        payload={
            "approval_id": queued.request_id,
            "thread_id": queued.thread_id,
            "symbol": queued.proposal.symbol,
            "side": queued.proposal.side.value,
            "confidence": f"{queued.proposal.confidence:.6f}",
            "rationale": queued.proposal.rationale,
            "ts_ns": str(queued.requested_at_ts_ns),
        },
    )
    return queued.request_id


def _ledger_decide(
    ledger: LedgerAuthorityWriter,
    queue: ApprovalQueue,
    *,
    request_id: str,
    approved: bool,
    decided_by: str,
    ts_ns: int,
) -> None:
    """Mirror what ``ApprovalEdge.approve``/``reject`` writes."""

    queue.decide(request_id=request_id, approved=approved, decided_by=decided_by)
    ledger.append(
        ts_ns=ts_ns,
        kind=APPROVED_KIND if approved else REJECTED_KIND,
        payload={
            "approval_id": request_id,
            "decided_by": decided_by,
            "ts_ns": str(ts_ns),
        },
    )


def _proposal(symbol: str, side: ApprovalSideApi, *, conf: float) -> ProposedSignalApi:
    return ProposedSignalApi(
        symbol=symbol, side=side, confidence=conf, rationale=f"why-{symbol}"
    )


def test_restart_safety_round_trip() -> None:
    """Submit / approve / reject on queue1, replay ledger into queue2,
    assert public list output is identical."""

    ledger, queue1, ts_box = _build_runtime_pair()

    a = _ledger_submit(
        ledger,
        queue1,
        thread_id="thr-A",
        proposal=_proposal("EURUSD", ApprovalSideApi.BUY, conf=0.7),
        ts_ns=ts_box[0] + 1,
    )
    ts_box[0] += 1
    b = _ledger_submit(
        ledger,
        queue1,
        thread_id="thr-B",
        proposal=_proposal("BTCUSDT", ApprovalSideApi.SELL, conf=0.42),
        ts_ns=ts_box[0] + 1,
    )
    ts_box[0] += 1
    _ledger_decide(
        ledger,
        queue1,
        request_id=a,
        approved=True,
        decided_by="op-1",
        ts_ns=ts_box[0] + 1,
    )
    ts_box[0] += 1
    c = _ledger_submit(
        ledger,
        queue1,
        thread_id="thr-C",
        proposal=_proposal("ETHUSDT", ApprovalSideApi.BUY, conf=0.95),
        ts_ns=ts_box[0] + 1,
    )
    ts_box[0] += 1
    _ledger_decide(
        ledger,
        queue1,
        request_id=b,
        approved=False,
        decided_by="op-2",
        ts_ns=ts_box[0] + 1,
    )

    # queue2 = a fresh queue rehydrated from the ledger only.
    queue2 = ApprovalQueue()
    n = rehydrate_approval_queue_from_ledger(queue2, ledger)
    assert n == 3

    full1 = queue1.list(include_decided=True)
    full2 = queue2.list(include_decided=True)
    assert full1 == full2
    assert [r.request_id for r in full2] == [a, b, c]
    assert queue2.list() == queue1.list()  # pending-only view


def test_rehydrate_resets_existing_state() -> None:
    """Calling rehydrate on a non-empty queue replaces its contents."""

    ledger, queue, _ = _build_runtime_pair()
    _ledger_submit(
        ledger,
        queue,
        thread_id="thr-A",
        proposal=_proposal("EURUSD", ApprovalSideApi.BUY, conf=0.7),
        ts_ns=2_000,
    )
    assert len(queue) == 1

    other_ledger = LedgerAuthorityWriter()
    n = rehydrate_approval_queue_from_ledger(queue, other_ledger)
    assert n == 0
    assert len(queue) == 0
    assert queue.list(include_decided=True) == ()


def test_rehydrate_is_idempotent() -> None:
    """Replaying twice produces the same projection."""

    ledger, queue1, ts_box = _build_runtime_pair()
    _ledger_submit(
        ledger,
        queue1,
        thread_id="thr-A",
        proposal=_proposal("EURUSD", ApprovalSideApi.BUY, conf=0.7),
        ts_ns=ts_box[0] + 1,
    )

    queue2 = ApprovalQueue()
    rehydrate_approval_queue_from_ledger(queue2, ledger)
    snapshot1 = queue2.list(include_decided=True)
    rehydrate_approval_queue_from_ledger(queue2, ledger)
    snapshot2 = queue2.list(include_decided=True)
    assert snapshot1 == snapshot2
