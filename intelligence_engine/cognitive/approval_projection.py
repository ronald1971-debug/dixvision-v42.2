"""Wave-03 PR-7 — ledger-backed approval queue projection.

The :class:`~intelligence_engine.cognitive.approval_queue.ApprovalQueue`
keeps an in-memory dict of approval rows so that ``GET /api/cognitive/
chat/approvals`` is O(1). Wave-03 PR-5 already wrote one row per
lifecycle transition to the audit ledger:

* ``OPERATOR_APPROVAL_PENDING`` on ``submit`` (chat runtime)
* ``OPERATOR_APPROVED_SIGNAL`` on approve (approval edge)
* ``OPERATOR_REJECTED_SIGNAL`` on reject (approval edge)

…but the in-memory dict was the source of truth. A process restart
(or a crash before approval) lost every pending row even though the
ledger had captured them.

This module makes the ledger the source of truth: the queue is now a
projection over those three ledger row kinds. On startup, the HTTP
layer replays the chain through :func:`projection_rows_from_ledger`
and feeds the result back into a fresh :class:`ApprovalQueue` via
:meth:`ApprovalQueue.rehydrate`. Same dict, same shape — just sourced
from the audit chain instead of in-memory submit calls.

Determinism contract (INV-15): the projection is a pure function of
``(rows, …)`` — same chain in, same projection out, byte-identical.
No clock, no PRNG, no IO.

Isolation contract (B1): this module lives under
``intelligence_engine.cognitive.*`` and may not import
``governance_engine.*`` or ``system_engine.*``. The HTTP layer reads
from ``LedgerAuthorityWriter.read()`` and hands the rows over as
typed records that this module walks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalRequestApi,
    ApprovalSideApi,
    ApprovalStatusApi,
    ProposedSignalApi,
)

__all__ = [
    "PENDING_KIND",
    "APPROVED_KIND",
    "REJECTED_KIND",
    "DECISION_KINDS",
    "ProjectionLedgerRow",
    "projection_rows_from_payloads",
]


# Ledger row kinds the projection consumes. The chat runtime
# (``ui.cognitive_chat_runtime``) writes ``PENDING_KIND``; the
# approval edge (``intelligence_engine.cognitive.approval_edge``)
# writes the two decision kinds. Other kinds in the ledger are
# ignored — the projection is opt-in.

PENDING_KIND = "OPERATOR_APPROVAL_PENDING"
APPROVED_KIND = "OPERATOR_APPROVED_SIGNAL"
REJECTED_KIND = "OPERATOR_REJECTED_SIGNAL"
DECISION_KINDS = frozenset({APPROVED_KIND, REJECTED_KIND})


@dataclass(frozen=True, slots=True)
class ProjectionLedgerRow:
    """The minimal slice of a ledger entry the projection needs.

    The audit ledger's :class:`~core.contracts.governance.LedgerEntry`
    carries hash-chain fields (``seq``, ``prev_hash``, ``hash_chain``)
    that the projection ignores; this dataclass keeps the projection
    decoupled from the governance contract, which would otherwise
    drag a B1-forbidden import into ``intelligence_engine.cognitive``.

    Construction is intentionally cheap so the HTTP layer can map
    ``LedgerAuthorityWriter.read()`` into a tuple of these in one
    pass at startup.
    """

    kind: str
    payload: Mapping[str, str]


def _parse_pending(payload: Mapping[str, str]) -> ApprovalRequestApi:
    """Materialise a fresh PENDING row from a ledger payload.

    Raises :class:`KeyError` on a missing required field — every
    payload written by ``ui.cognitive_chat_runtime`` has the full
    set, so a missing field is a chain corruption / version mismatch
    that the caller must surface.
    """

    side = ApprovalSideApi(payload["side"])
    proposal = ProposedSignalApi(
        symbol=payload["symbol"],
        side=side,
        confidence=float(payload["confidence"]),
        rationale=payload.get("rationale", ""),
    )
    return ApprovalRequestApi(
        request_id=payload["approval_id"],
        thread_id=payload["thread_id"],
        requested_at_ts_ns=int(payload["ts_ns"]),
        proposal=proposal,
        status=ApprovalStatusApi.PENDING,
        decided_at_ts_ns=None,
        decided_by="",
    )


def _apply_decision(
    pending: ApprovalRequestApi,
    *,
    payload: Mapping[str, str],
    approved: bool,
) -> ApprovalRequestApi:
    """Flip a PENDING row to APPROVED / REJECTED via a payload.

    Mirrors :meth:`ApprovalQueue.decide` byte-for-byte: same
    ``decided_at_ts_ns``, same ``decided_by`` value, same status
    enum.
    """

    return pending.model_copy(
        update={
            "status": (
                ApprovalStatusApi.APPROVED
                if approved
                else ApprovalStatusApi.REJECTED
            ),
            "decided_at_ts_ns": int(payload["ts_ns"]),
            "decided_by": payload.get("decided_by", ""),
        },
    )


def projection_rows_from_payloads(
    rows: Iterable[ProjectionLedgerRow],
) -> tuple[tuple[str, ApprovalRequestApi], ...]:
    """Walk ``rows`` in order, return ``(approval_id, row)`` pairs.

    The pairs are in *insertion* order (first PENDING wins for ordering),
    which matches :meth:`ApprovalQueue.list` semantics. A decision row
    that references an unknown ``approval_id`` is silently skipped —
    in practice this only happens if a chain prefix is replayed (e.g.
    a partial export) where the corresponding PENDING is on the other
    side of the cut. The behaviour is intentional: the projection is
    forgiving so an operator can still inspect the resulting queue.

    Args:
        rows: Iterable of :class:`ProjectionLedgerRow` — typically
            built from ``LedgerAuthorityWriter.read()`` filtered to the
            three :data:`DECISION_KINDS` ∪ :data:`PENDING_KIND`.

    Returns:
        A tuple of ``(approval_id, ApprovalRequestApi)`` pairs in
        insertion order. The ``ApprovalRequestApi`` carries the
        terminal lifecycle state implied by the chain.
    """

    by_id: dict[str, ApprovalRequestApi] = {}
    order: list[str] = []
    for row in rows:
        if row.kind == PENDING_KIND:
            req = _parse_pending(row.payload)
            if req.request_id in by_id:
                # Duplicate PENDING — ignore. The first row sets
                # ordering; a second PENDING with the same id
                # would be a chain anomaly the lint should catch
                # at write time, not silently overwrite here.
                continue
            by_id[req.request_id] = req
            order.append(req.request_id)
        elif row.kind in DECISION_KINDS:
            approval_id = row.payload.get("approval_id", "")
            pending = by_id.get(approval_id)
            if pending is None:
                continue
            if pending.status is not ApprovalStatusApi.PENDING:
                # A second decision row for the same approval is a
                # chain anomaly (the approval edge refuses double
                # decides at write time). Skip the duplicate so the
                # projection stays idempotent.
                continue
            by_id[approval_id] = _apply_decision(
                pending,
                payload=row.payload,
                approved=row.kind == APPROVED_KIND,
            )
        # Other kinds are ignored by design.
    return tuple((rid, by_id[rid]) for rid in order)
