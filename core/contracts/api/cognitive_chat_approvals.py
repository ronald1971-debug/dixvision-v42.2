"""Pydantic request/response models for ``/api/cognitive/chat/approvals/*``.

Wave-03 PR-5 — operator-approval edge gating ``SignalEvent`` proposal
emission. The cognitive chat graph (PR-3, PR-4) replies in natural
language; if the reply contains a structured "propose" block, the
runtime drops the proposal into a *pending queue* keyed by request
id and surfaces it here. The bus only sees a real ``SignalEvent``
after the operator clicks **Approve** — and at that point the
request flows through the same ``produced_by_engine`` /
``AuthorityGuard`` chain as every other intelligence-side signal
(HARDEN-02, HARDEN-03).

Design constraints:

1. **One direction of growth only.** The chat-turn / chat-status
   contracts in :mod:`core.contracts.api.cognitive_chat` stay
   *unchanged* on the wire — PR-4 clients keep working without
   modification. PR-5 adds a new optional ``proposal_id`` field
   on :class:`ChatTurnResponse` (in the original module) and a
   wholly new namespace here for the approval queue itself.

2. **No transport-engine coupling.** These models describe the
   wire shape only — the queue's internal state, the parser, and
   the emit-on-approve bridge live in
   ``intelligence_engine.cognitive.approval_*`` and never leak
   across the HTTP boundary.

3. **Pydantic→TS codegen is the source of truth.** Every field
   here is registered in
   ``.github/workflows/dashboard2026.yml`` so the dashboard
   ``api.ts`` cannot drift from this module without CI failing.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApprovalSideApi(StrEnum):
    """Direction of the proposed signal.

    Mirrors :class:`core.contracts.events.Side` flattened to string
    literals so the TS client doesn't need any cross-module union.
    The ``HOLD`` value is rejected by the proposal parser (a
    "propose HOLD" payload is not actionable); listing it here
    keeps the union stable in case the parser grows."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ApprovalStatusApi(StrEnum):
    """Lifecycle of a queued approval request.

    A request lands as ``PENDING`` from the chat turn, then
    transitions exactly once to ``APPROVED`` or ``REJECTED`` via
    the operator-side endpoints. Re-decisions are rejected by the
    queue; the dashboard hides decided rows by default but can
    show them in a history panel."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ProposedSignalApi(BaseModel):
    """The :class:`SignalEvent` shape, *as proposed* — not yet emitted.

    This is the operator-visible draft of what the chat graph wants
    to put on the bus. Only the fields the LLM can fill are exposed;
    ``ts_ns`` and ``produced_by_engine`` are stamped server-side at
    the moment of emission so the operator cannot accidentally
    forge or replay a stale proposal."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        min_length=1,
        max_length=32,
        description=(
            "Instrument identifier — e.g. ``\"EURUSD\"`` or "
            "``\"BTCUSDT\"``. Validated against the registry on "
            "approval; an unknown symbol turns the approval into a "
            "400."
        ),
    )
    side: ApprovalSideApi = Field(
        description=(
            "Direction of the proposed action. ``HOLD`` is rejected "
            "by the parser before ever reaching this model."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence band ``[0.0, 1.0]`` carried straight onto "
            "the emitted ``SignalEvent``. The chat graph is asked "
            "to bound itself; the queue clamps anyway."
        ),
    )
    rationale: str = Field(
        max_length=2000,
        description=(
            "Operator-facing free-text explanation of *why* the "
            "graph is proposing this action. Stamped onto the "
            "``SignalEvent.meta`` as ``rationale=...`` on approval "
            "so the audit ledger captures it without a separate "
            "lookup. Bounded to keep ledger rows small."
        ),
    )


class ApprovalRequestApi(BaseModel):
    """One row in the operator-visible pending queue.

    Surfaced by ``GET /api/cognitive/chat/approvals``. The
    ``request_id`` is the URL key for the approve / reject
    endpoints. Decided rows carry ``decided_at_ts_ns`` and
    ``decided_by`` for audit; pending rows leave them as ``None``
    / ``\"\"``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Server-generated stable id (uuid4) — survives across "
            "page reloads and is the URL key for "
            "``POST /api/cognitive/chat/approvals/{id}/...``."
        ),
    )
    thread_id: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Chat thread that produced the proposal — lets the "
            "operator click back to the originating conversation."
        ),
    )
    requested_at_ts_ns: int = Field(
        ge=0,
        description=(
            "Monotonic timestamp of the originating chat turn "
            "(``TimeAuthority``). Used purely for display ordering."
        ),
    )
    proposal: ProposedSignalApi
    status: ApprovalStatusApi = Field(
        description=(
            "Current lifecycle state. The list endpoint by default "
            "returns only ``PENDING`` rows; decided rows are "
            "available via ``?include_decided=true``."
        ),
    )
    decided_at_ts_ns: int | None = Field(
        default=None,
        description=(
            "When the operator clicked approve / reject "
            "(``TimeAuthority``). ``None`` while pending."
        ),
    )
    decided_by: str = Field(
        default="",
        max_length=64,
        description=(
            "Operator id from the bridge / cockpit auth context. "
            "Empty string while pending. Today the harness uses "
            "``\"operator\"`` as a placeholder; PR-N will plumb the "
            "authenticated identity through."
        ),
    )


class ApprovalsListResponse(BaseModel):
    """Server → operator: a snapshot of the approval queue."""

    model_config = ConfigDict(extra="forbid")

    requests: list[ApprovalRequestApi] = Field(
        description=(
            "All matching approval requests, oldest first by "
            "``requested_at_ts_ns``. Default filter is "
            "``PENDING``-only."
        ),
    )


class ApprovalDecisionRequest(BaseModel):
    """Operator → server: approve or reject a queued proposal.

    The decision verb is in the URL path
    (``/approvals/{id}/approve`` vs ``/approvals/{id}/reject``)
    so this body only carries the audit-context fields. Empty
    bodies are accepted for the harness; production deployments
    fill ``decided_by`` from the cockpit session."""

    model_config = ConfigDict(extra="forbid")

    decided_by: str = Field(
        default="operator",
        max_length=64,
        description=(
            "Operator identity recorded on the ledger row. "
            "Defaults to ``\"operator\"`` so the harness works "
            "without an authenticated session."
        ),
    )
    note: str = Field(
        default="",
        max_length=500,
        description=(
            "Free-text annotation written into the ledger row "
            "(``approval_note`` payload field). Bounded to keep "
            "ledger rows small."
        ),
    )


class ApprovalDecisionResponse(BaseModel):
    """Server → operator: result of an approve / reject click."""

    model_config = ConfigDict(extra="forbid")

    request: ApprovalRequestApi = Field(
        description=(
            "The decided request, with status flipped to "
            "``APPROVED`` or ``REJECTED``. The list endpoint will "
            "no longer return it under the default filter."
        ),
    )
    emitted_signal_id: str = Field(
        default="",
        description=(
            "Audit-ledger row id of the resulting "
            "``OPERATOR_APPROVED_SIGNAL`` (or "
            "``OPERATOR_REJECTED_SIGNAL``) entry. Empty string "
            "before persistence is wired (today the chain hash "
            "stands in)."
        ),
    )


__all__ = [
    "ApprovalDecisionRequest",
    "ApprovalDecisionResponse",
    "ApprovalRequestApi",
    "ApprovalSideApi",
    "ApprovalStatusApi",
    "ApprovalsListResponse",
    "ProposedSignalApi",
]
