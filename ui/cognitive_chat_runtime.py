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

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
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
    server restart in dev."""

    task: TaskClass
    registry: SourceRegistry
    transport: ChatTransport
    ledger_append: Any
    feature_flag: CognitiveChatFeatureFlag
    bundle: CognitiveChatBundle | None = None

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
        provider_id = ""
        try:
            provider_id = select_providers(self.registry, self.task)[0].id
        except (IndexError, NoEligibleProviderError):
            provider_id = ""

        checkpoint_id = ""
        try:
            tup = bundle.saver.get_tuple(config)
            if tup is not None:
                checkpoint_id = str(
                    tup.config["configurable"].get("checkpoint_id", "")
                )
        except Exception:  # noqa: BLE001 — saver introspection is advisory
            checkpoint_id = ""

        return ChatTurnResponse(
            thread_id=req.thread_id,
            reply=_reply_to_api(reply_msg),
            provider_id=provider_id,
            checkpoint_id=checkpoint_id,
        )


def build_runtime(
    *,
    registry: SourceRegistry,
    ledger_writer: LedgerAuthorityWriter,
    transport: ChatTransport | None = None,
    task: TaskClass = TaskClass.INDIRA_REASONING,
    feature_flag: CognitiveChatFeatureFlag | None = None,
) -> CognitiveChatRuntime:
    """Construct a runtime for the FastAPI process.

    ``transport`` defaults to :class:`NotConfiguredTransport` so a
    fresh deployment is reachable but turns return 502 until a
    real transport is wired. Tests pass a fake transport here.
    ``feature_flag`` defaults to env-driven; tests pass an
    injected getter."""

    return CognitiveChatRuntime(
        task=task,
        registry=registry,
        transport=transport if transport is not None else NotConfiguredTransport(),
        ledger_append=build_ledger_append(ledger_writer),
        feature_flag=(
            feature_flag if feature_flag is not None else CognitiveChatFeatureFlag()
        ),
    )


def new_thread_id() -> str:
    """Convenience wrapper used by the route when the client omits one."""

    return uuid.uuid4().hex
