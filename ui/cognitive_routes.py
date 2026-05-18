"""C-2 / P2-4 / R-1 part 3 — cognitive chat operator routes.

Extracted from :mod:`ui.server` to keep the FastAPI host module
focused on harness boot and runtime wiring. The five endpoints
mounted here back the operator chat panel (Wave-03 PR-4 and PR-5):

* ``GET  /api/cognitive/chat/status``
* ``POST /api/cognitive/chat/turn``
* ``GET  /api/cognitive/chat/approvals``
* ``POST /api/cognitive/chat/approvals/{request_id}/approve``
* ``POST /api/cognitive/chat/approvals/{request_id}/reject``

URL paths, HTTP methods, request bodies, response models and HTTP
status codes are preserved byte-for-byte from the inline handlers
that lived in ``ui/server.py``. The route module never imports
``ui.server`` or any ``*_engine`` package directly — it reads its
dependencies through a Protocol-based state accessor, the same
pattern used by :mod:`ui.dashboard_routes`,
:mod:`ui.execution_routes`, :mod:`ui.governance_routes`,
:mod:`ui.runtime_routes`, and :mod:`ui.feeds_routes`.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException

from core.contracts.api.cognitive_chat import (
    ChatStatusResponse,
    ChatTurnRequest,
    ChatTurnResponse,
)
from core.contracts.api.cognitive_chat_approvals import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalsListResponse,
)
from intelligence_engine.cognitive.approval_edge import (
    ApprovalAlreadyDecidedError,
    ApprovalNotFoundError,
)
from ui.cognitive_chat_runtime import (
    ChatTurnDisabled,
    ChatTurnNoProvider,
    ChatTurnTransportFailed,
)


class _CognitiveStateLike(Protocol):
    """Read-only accessor the host installs into FastAPI app.

    Only the attributes the five cognitive routes actually read are
    declared here so the route module stays decoupled from the
    full harness ``_State`` type.
    """

    @property
    def lock(self) -> Lock: ...
    @property
    def chat_runtime(self) -> Any: ...
    @property
    def approval_edge(self) -> Any: ...


def build_cognitive_router(
    state_accessor: Callable[[], _CognitiveStateLike],
) -> APIRouter:
    """Construct the operator cognitive-chat router.

    Args:
        state_accessor: Callable returning the live state object.
            The route module never holds a direct reference to the
            object; it re-reads through the accessor on every
            request so the same factory works in tests with a
            stub state and in the production harness.

    Returns:
        An :class:`APIRouter` mounting the five chat endpoints
        under ``/api/cognitive/chat``.
    """

    router = APIRouter(prefix="/api/cognitive/chat", tags=["cognitive"])

    @router.get("/status", response_model=ChatStatusResponse)
    def cognitive_chat_status() -> ChatStatusResponse:
        """Wave-03 PR-4 — feature-flag + provider availability snapshot.

        Polled by the operator chat page on mount so the UI can
        decide whether to render the input box or a "feature
        disabled" notice. Read-only; never writes the ledger.
        """

        state = state_accessor()
        with state.lock:
            return state.chat_runtime.status()  # type: ignore[no-any-return]

    @router.post("/turn", response_model=ChatTurnResponse)
    def cognitive_chat_turn(body: ChatTurnRequest) -> ChatTurnResponse:
        """Wave-03 PR-4 — drive one turn of the cognitive chat graph.

        Honors ``DIX_COGNITIVE_CHAT_ENABLED`` (on by default since
        PR #165 — only the explicit falsy set ``0`` / ``false`` /
        ``no`` / ``off`` flips it off; 503 is returned in that
        case). Dispatches through the registry-driven chat model
        from PR-1 so no vendor name appears on the wire. State is
        persisted to the audit ledger via PR-2's saver.
        Operator-approval edges that gate ``SignalEvent`` proposal
        emission land in PR-5.
        """

        # Snapshot the runtime under the process-wide lock, then
        # drop it before calling ``turn`` — the LLM round-trip can
        # take seconds, and holding the harness lock across it
        # would block every other endpoint (health, ticks,
        # operator summary, …). ``CognitiveChatRuntime`` has its
        # own lock guarding the bundle lazy-init path; the graph
        # itself is invocation-safe under concurrent calls because
        # LangGraph keys state by ``thread_id``.
        state = state_accessor()
        with state.lock:
            runtime = state.chat_runtime
        try:
            return runtime.turn(body)  # type: ignore[no-any-return]
        except ChatTurnDisabled as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ChatTurnNoProvider as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ChatTurnTransportFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            # Bad request shape (empty messages / wrong tail role /
            # SYSTEM message before PR-5 lands).
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/approvals", response_model=ApprovalsListResponse)
    def cognitive_chat_approvals_list(
        include_decided: bool = False,
    ) -> ApprovalsListResponse:
        """Wave-03 PR-5 — snapshot the operator-approval queue.

        Default returns ``PENDING``-only rows so the dashboard
        panel only shows what still needs an operator click. Pass
        ``?include_decided=true`` to also surface the recent
        ``APPROVED`` / ``REJECTED`` history (used by the audit
        panel). Read-only; never writes the ledger.
        """

        state = state_accessor()
        with state.lock:
            rows = state.chat_runtime.approval_queue.list(
                include_decided=include_decided,
            )
        return ApprovalsListResponse(requests=list(rows))

    @router.post(
        "/approvals/{request_id}/approve",
        response_model=ApprovalDecisionResponse,
    )
    def cognitive_chat_approval_approve(
        request_id: str,
        body: ApprovalDecisionRequest | None = None,
    ) -> ApprovalDecisionResponse:
        """Wave-03 PR-5 — operator approves a queued cognitive proposal.

        The approval edge stamps ``produced_by_engine=
        "intelligence_engine.cognitive"`` on the resulting
        ``SignalEvent`` (B26 / HARDEN-03), routes it through the
        intelligence → execution chain (HARDEN-02 chokepoint), and
        writes an ``OPERATOR_APPROVED_SIGNAL`` ledger row. Returns
        the decided request and the new event's audit-ledger id.
        """

        decision = body if body is not None else ApprovalDecisionRequest()
        state = state_accessor()
        try:
            decided, sig = state.approval_edge.approve(
                request_id=request_id,
                decision=decision,
            )
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ApprovalAlreadyDecidedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return ApprovalDecisionResponse(
            request=decided,
            emitted_signal_id=f"{sig.symbol}:{sig.side.value}:{sig.ts_ns}",
        )

    @router.post(
        "/approvals/{request_id}/reject",
        response_model=ApprovalDecisionResponse,
    )
    def cognitive_chat_approval_reject(
        request_id: str,
        body: ApprovalDecisionRequest | None = None,
    ) -> ApprovalDecisionResponse:
        """Wave-03 PR-5 — operator rejects a queued cognitive proposal.

        No event hits the bus; an ``OPERATOR_REJECTED_SIGNAL`` row
        is written to the ledger so the audit chain captures every
        decision (not just the approvals). Returns the decided
        request with ``emitted_signal_id`` left empty.
        """

        decision = body if body is not None else ApprovalDecisionRequest()
        state = state_accessor()
        try:
            decided = state.approval_edge.reject(
                request_id=request_id,
                decision=decision,
            )
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ApprovalAlreadyDecidedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return ApprovalDecisionResponse(request=decided, emitted_signal_id="")

    return router


__all__ = ["build_cognitive_router"]
