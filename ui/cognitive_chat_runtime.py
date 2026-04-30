"""HTTP-side glue for the cognitive chat surface (Wave-03 PR-4).

The LangGraph chat graph from PR-3 (``assemble_cognitive_chat``) is
intentionally pure: it accepts seams (``provider_resolver``,
``transport``, ``ledger_append``) and returns a
:class:`CognitiveChatBundle`. This module is where the *FastAPI
process* binds those seams to live registry / transport / ledger
instances, and exposes a small, testable façade
(:class:`CognitiveChatRuntime`) that the route handlers in
``ui/server.py`` consume.

Two design constraints govern the shape of this module:

1. **B1 / B24 isolation stays intact.**
   ``intelligence_engine.cognitive.chat.*`` may not import
   ``governance_engine.*`` or ``system_engine.*`` (the seams in
   PR-1/2/3 enforce this). The HTTP layer (``ui.*``) is *allowed*
   to depend on all engines — so this is the right place to
   instantiate a ``LedgerAuthorityWriter``-backed ``LedgerAppend``
   adapter, the live SCVS ``SourceRegistry``, and the production
   :class:`ChatTransport`.

2. **Test seams stay surgical.**
   The runtime ships a default :class:`NotConfiguredTransport`
   that raises :class:`NoEligibleProviderError` for every turn —
   so a fresh deployment with the feature flag on but no
   credentialed provider returns a clean 502 instead of a stack
   trace. Tests inject fake transports via the
   :func:`build_runtime` factory rather than monkey-patching
   module globals.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.cognitive_router import (
    AIProvider,
    TaskClass,
    select_providers,
)
from core.contracts.api.cognitive_chat import (
    ChatMessageApi,
    ChatRoleApi,
    ChatStatusResponse,
    ChatTurnRequest,
    ChatTurnResponse,
)
from core.contracts.api.cognitive_chat_approvals import (
    ApprovalDecisionRequest,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.operator_attention import (
    AUTO_DECIDED_BY_TAG,
    OperatorAttention,
)
from intelligence_engine.cognitive.approval_edge import ApprovalEdge
from intelligence_engine.cognitive.approval_projection import (
    DECISION_KINDS,
    PENDING_KIND,
    ProjectionLedgerRow,
    projection_rows_from_payloads,
)
from intelligence_engine.cognitive.approval_queue import ApprovalQueue
from intelligence_engine.cognitive.chat import (
    AllProvidersFailedError,
    ChatTransport,
    CognitiveChatBundle,
    CognitiveChatDisabledError,
    CognitiveChatFeatureFlag,
    NoEligibleProviderError,
    ProviderResolver,
    assemble_cognitive_chat,
)
from intelligence_engine.cognitive.proposal_parser import extract_proposal
from system_engine.scvs.source_registry import SourceRegistry

__all__ = [
    "CognitiveChatRuntime",
    "ChatTurnDisabled",
    "ChatTurnNoProvider",
    "ChatTurnTransportFailed",
    "NotConfiguredTransport",
    "build_ledger_append",
    "build_runtime",
]


class ChatTurnDisabled(RuntimeError):
    """Raised when the runtime is asked for a turn but the flag is off."""


class ChatTurnNoProvider(RuntimeError):
    """Raised when no provider is eligible for the requested task."""


class ChatTurnTransportFailed(RuntimeError):
    """Raised when every eligible provider failed transiently."""


class NotConfiguredTransport:
    """Default transport — refuses every turn with a clear message.

    Production deployments replace this via :func:`build_runtime`
    once a real provider transport (HTTP / MCP / gRPC) is wired.
    Until then the runtime is reachable but every turn returns a
    502 so the chat page can render "no transport configured"
    without the operator having to read CI logs."""

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        raise NoEligibleProviderError(
            "no chat transport is configured in this process — "
            "the cognitive chat surface is reachable but cannot "
            "dispatch turns until a real provider transport is "
            "registered via build_runtime(...)"
        )


def build_ledger_append(
    writer: LedgerAuthorityWriter,
):
    """Adapt :class:`LedgerAuthorityWriter` to the ``LedgerAppend`` shape.

    The cognitive saver in PR-3 expects a ``Callable[[str,
    Mapping[str, str]], None]``; the production writer is a
    method on a stateful object. The adapter also stamps a
    monotonic ``ts_ns`` so the saver does not have to import
    ``system_engine.time_authority`` (which would re-introduce a
    B1 violation if the saver did the wrapping itself).
    """

    def _append(kind: str, payload: Mapping[str, str]) -> None:
        writer.append(ts_ns=time.time_ns(), kind=kind, payload=dict(payload))

    return _append


def _provider_resolver_from_registry(
    registry: SourceRegistry,
    task: TaskClass,
) -> ProviderResolver:
    """Bind ``registry`` and ``task`` into a zero-arg resolver.

    The chat model expects ``Callable[[], tuple[AIProvider, ...]]``.
    Currying ``select_providers`` here keeps the cognitive package
    free of any direct ``system_engine.scvs`` import (B1)."""

    def _resolve() -> tuple[AIProvider, ...]:
        return select_providers(registry, task)

    return _resolve


def _request_to_messages(req: ChatTurnRequest) -> list[BaseMessage]:
    """Translate the typed wire payload to LangChain messages.

    Anything other than a plain ``USER`` message at the tail
    raises ``ValueError`` so the route can surface a 400. The
    chat graph itself relies on this invariant — its single node
    always responds to the most recent message and would happily
    talk to itself otherwise."""

    if not req.messages:
        raise ValueError("messages must contain at least one entry")
    if req.messages[-1].role is not ChatRoleApi.USER:
        raise ValueError("the last message must be a USER message")

    out: list[BaseMessage] = []
    for msg in req.messages:
        if msg.role is ChatRoleApi.USER:
            out.append(HumanMessage(content=msg.content))
        elif msg.role is ChatRoleApi.ASSISTANT:
            out.append(AIMessage(content=msg.content))
        else:  # SYSTEM — reserved for PR-5
            raise ValueError(
                "system messages are not yet accepted on this surface"
            )
    return out


def _reply_to_api(reply: BaseMessage) -> ChatMessageApi:
    """Coerce LangChain reply content (str | list[part]) to our wire shape."""

    content = reply.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "")
            if isinstance(part, dict)
            else str(part)
            for part in content
        )
    else:
        text = str(content)
    return ChatMessageApi(role=ChatRoleApi.ASSISTANT, content=text)


@dataclass
class CognitiveChatRuntime:
    """Stateful façade over the chat graph used by FastAPI handlers.

    Construction is *eager*: the runtime tries to assemble the
    bundle once at startup; if the feature flag is off the bundle
    stays ``None`` and ``status()`` reports ``enabled=false``.
    Re-assembly happens on the next request after the operator
    flips the env var, so flipping the flag does not require a
    server restart in dev.

    Wave-03 PR-5 adds an :class:`ApprovalQueue`. The runtime no
    longer just returns the assistant reply — if the reply contains
    a structured ``propose`` block, it queues the proposal and the
    response carries the resulting ``proposal_id`` so the dashboard
    can route the operator to the pending-approvals panel."""

    task: TaskClass
    registry: SourceRegistry
    transport: ChatTransport
    ledger_append: Any
    feature_flag: CognitiveChatFeatureFlag
    approval_queue: ApprovalQueue
    bundle: CognitiveChatBundle | None = None
    # Wave-04.6 PR-F: OperatorAttention reads the canonical mode-effect
    # table to decide whether the proposal needs a per-trade operator
    # click (LIVE / CANARY / SHADOW / PAPER / SAFE) or may auto-emit
    # via the approval edge (AUTO with no active hazard). Both seams
    # are optional so PR-7 / Wave-03 tests that wire only the queue
    # remain green.
    operator_attention: OperatorAttention | None = None
    approval_edge: ApprovalEdge | None = None
    # Per-runtime lock guarding `bundle` lazy-init only. The graph
    # invocation itself (an LLM round-trip) must NOT be held under
    # this lock — and definitely not under the process-wide
    # ``STATE.lock`` — so other endpoints stay responsive while a
    # turn is in flight.
    _init_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def status(self) -> ChatStatusResponse:
        from intelligence_engine.cognitive.chat import FEATURE_FLAG_ENV_VAR

        if not self.feature_flag.enabled:
            return ChatStatusResponse(
                enabled=False,
                eligible_providers=[],
                feature_flag_env_var=FEATURE_FLAG_ENV_VAR,
            )
        providers = select_providers(self.registry, self.task)
        return ChatStatusResponse(
            enabled=True,
            eligible_providers=[p.id for p in providers],
            feature_flag_env_var=FEATURE_FLAG_ENV_VAR,
        )

    def _ensure_bundle(self) -> CognitiveChatBundle:
        # Double-checked: cheap read first, then lock + re-check so
        # concurrent first requests do not assemble two bundles.
        if self.bundle is not None:
            return self.bundle
        with self._init_lock:
            if self.bundle is not None:
                return self.bundle
            try:
                self.bundle = assemble_cognitive_chat(
                    task=self.task,
                    provider_resolver=_provider_resolver_from_registry(
                        self.registry, self.task
                    ),
                    transport=self.transport,
                    ledger_append=self.ledger_append,
                    feature_flag=self.feature_flag,
                )
            except CognitiveChatDisabledError as exc:
                raise ChatTurnDisabled(str(exc)) from exc
        return self.bundle

    def turn(self, req: ChatTurnRequest) -> ChatTurnResponse:
        if not self.feature_flag.enabled:
            raise ChatTurnDisabled(
                "cognitive chat is disabled on this server"
            )
        bundle = self._ensure_bundle()
        messages = _request_to_messages(req)

        config = {"configurable": {"thread_id": req.thread_id}}
        try:
            result = bundle.graph.invoke({"messages": messages}, config=config)
        except NoEligibleProviderError as exc:
            raise ChatTurnNoProvider(str(exc)) from exc
        except AllProvidersFailedError as exc:
            raise ChatTurnTransportFailed(str(exc)) from exc

        reply_msg = result["messages"][-1]
        # The actual provider id is stamped onto the AIMessage's
        # ``response_metadata`` by RegistryDrivenChatModel so the
        # value here reflects the provider that *served* the turn —
        # not just the first one in the registry. Fallback chains
        # (TransientProviderError on the first try, success on the
        # second) therefore surface the real id to operators.
        provider_id = ""
        metadata = getattr(reply_msg, "response_metadata", None) or {}
        candidate = metadata.get("provider_id")
        if isinstance(candidate, str):
            provider_id = candidate

        checkpoint_id = ""
        try:
            tup = bundle.saver.get_tuple(config)
            if tup is not None:
                checkpoint_id = str(
                    tup.config["configurable"].get("checkpoint_id", "")
                )
        except Exception:  # noqa: BLE001 — saver introspection is advisory
            checkpoint_id = ""

        reply_api = _reply_to_api(reply_msg)

        # PR-5: if the reply contains a structured propose block,
        # drop it onto the approval queue and surface the
        # request_id so the client can navigate to the panel.
        # Parsing failures and HOLD proposals are silently
        # treated as conversational — the chat reply still
        # reaches the operator.
        proposal_id = ""
        proposal = extract_proposal(reply_api.content)
        if proposal is not None:
            queued = self.approval_queue.submit(
                thread_id=req.thread_id,
                proposal=proposal,
                requested_at_ts_ns=time.time_ns(),
            )
            self.ledger_append(
                "OPERATOR_APPROVAL_PENDING",
                {
                    "approval_id": queued.request_id,
                    "thread_id": queued.thread_id,
                    "symbol": queued.proposal.symbol,
                    "side": queued.proposal.side.value,
                    "confidence": (
                        f"{queued.proposal.confidence:.6f}"
                    ),
                    "rationale": queued.proposal.rationale,
                    "ts_ns": str(queued.requested_at_ts_ns),
                },
            )
            proposal_id = queued.request_id
            # Wave-04.6 PR-F: AUTO mode oversight relaxation. When the
            # mode-effect table reports ``oversight_kind=exception_only``
            # *and* no hazard is active, OperatorAttention returns
            # ``per_trade_required() == False`` and the runtime drives
            # the approval edge directly with
            # ``decided_by=AUTO_DECIDED_BY_TAG``. The approval edge
            # writes ``OPERATOR_APPROVED_SIGNAL`` and emits the
            # ``SignalEvent`` exactly as it would for an operator click,
            # so HARDEN-02 / HARDEN-03 are unchanged. When either seam
            # is absent (legacy / test wiring) the proposal stays
            # PENDING — the conservative back-compat path.
            if (
                self.operator_attention is not None
                and self.approval_edge is not None
                and not self.operator_attention.per_trade_required()
            ):
                decided, _sig = self.approval_edge.approve(
                    request_id=queued.request_id,
                    decision=ApprovalDecisionRequest(
                        decided_by=AUTO_DECIDED_BY_TAG,
                    ),
                )
                proposal_id = decided.request_id

        return ChatTurnResponse(
            thread_id=req.thread_id,
            reply=reply_api,
            provider_id=provider_id,
            checkpoint_id=checkpoint_id,
            proposal_id=proposal_id,
        )


def rehydrate_approval_queue_from_ledger(
    queue: ApprovalQueue,
    ledger_writer: LedgerAuthorityWriter,
) -> int:
    """Rebuild ``queue`` from the ``OPERATOR_APPROVAL_*`` rows in the chain.

    Wave-03 PR-7 — single source of truth for the operator approval
    queue is the audit ledger; the in-memory queue is a projection
    over those rows. Called once on FastAPI startup so a process
    restart preserves pending approvals.

    Returns the number of approval ids replayed (a convenience for
    startup logs and tests). Rows of unrelated kinds are ignored;
    decisions referencing an unknown id are skipped (see
    :func:`projection_rows_from_payloads`).
    """

    projection_rows: list[ProjectionLedgerRow] = []
    for entry in ledger_writer.read():
        if entry.kind == PENDING_KIND or entry.kind in DECISION_KINDS:
            projection_rows.append(
                ProjectionLedgerRow(kind=entry.kind, payload=entry.payload),
            )
    rows = projection_rows_from_payloads(projection_rows)
    queue.rehydrate(rows)
    return len(rows)


def build_runtime(
    *,
    registry: SourceRegistry,
    ledger_writer: LedgerAuthorityWriter,
    transport: ChatTransport | None = None,
    task: TaskClass = TaskClass.INDIRA_REASONING,
    feature_flag: CognitiveChatFeatureFlag | None = None,
    approval_queue: ApprovalQueue | None = None,
) -> CognitiveChatRuntime:
    """Construct a runtime for the FastAPI process.

    ``transport`` defaults to :class:`NotConfiguredTransport` so a
    fresh deployment is reachable but turns return 502 until a
    real transport is wired. Tests pass a fake transport here.
    ``feature_flag`` defaults to env-driven; tests pass an
    injected getter. ``approval_queue`` defaults to a fresh
    in-memory queue using ``time.time_ns`` for stamping; tests
    inject a deterministic queue.

    PR-7: when the queue is created here (the production path) the
    factory replays ``ledger_writer`` so the projection picks up any
    pending or decided approvals from prior runs. Tests injecting
    their own queue keep full control — no implicit rehydrate."""

    if approval_queue is None:
        approval_queue = ApprovalQueue(ts_ns=time.time_ns)
        rehydrate_approval_queue_from_ledger(approval_queue, ledger_writer)
    return CognitiveChatRuntime(
        task=task,
        registry=registry,
        transport=transport if transport is not None else NotConfiguredTransport(),
        ledger_append=build_ledger_append(ledger_writer),
        feature_flag=(
            feature_flag if feature_flag is not None else CognitiveChatFeatureFlag()
        ),
        approval_queue=approval_queue,
    )


def new_thread_id() -> str:
    """Convenience wrapper used by the route when the client omits one."""

    return uuid.uuid4().hex
