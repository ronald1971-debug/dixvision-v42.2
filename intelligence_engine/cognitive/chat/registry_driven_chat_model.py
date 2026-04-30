"""RegistryDrivenChatModel — registry-driven LangChain ``BaseChatModel``.

Wave-03 PR-1 (Dashboard-2026 cognitive plan §4.3 + INV-67).

The chat layer is the first place inside the codebase that may invoke
an LLM. Per the design rule "registry-driven AI providers, no
hard-coded vendor names", every routing decision must be a projection
of ``registry/data_source_registry.yaml`` — chat code never mentions
any specific provider by name.

This module implements that contract as a :class:`BaseChatModel`
subclass with three pluggable seams:

1. :class:`ChatTransport` — a per-provider blocking call. Tests pass
   a fake transport; production wiring will pass a real HTTP/MCP
   client. The transport is the *only* place provider-specific code
   lives, and it never sees a :class:`TaskClass` — only the resolved
   :class:`AIProvider` row.
2. :class:`FallbackAuditSink` — receives one
   ``SOURCE_FALLBACK_ACTIVATED`` (SCVS-10) audit per transient
   transport failure, before the next provider is tried.
3. ``provider_resolver`` — a zero-arg callable returning the ordered
   tuple of eligible :class:`AIProvider` rows. Production wiring
   binds this to ``lambda: select_providers(registry, task)`` against
   the live SCVS registry; tests bind a list-returner. Inverting this
   dependency keeps the adapter free of any direct ``system_engine``
   import (B1 cross-engine isolation).

Errors form a tight hierarchy:

* :class:`TransientProviderError` — raised by the transport. The
  adapter records a fallback audit and tries the next provider.
* :class:`NoEligibleProviderError` — ``provider_resolver`` returned
  an empty tuple. Surfaced immediately; no fallback path exists.
* :class:`AllProvidersFailedError` — every eligible provider raised
  ``TransientProviderError``. Wraps the last failure as ``__cause__``.

Any other exception from the transport (auth failure, schema error,
operator-cancelled request, …) propagates unchanged: those are not
fallback-eligible, and silently swallowing them would mask real bugs.

The class is deliberately small. LangGraph integration (wave-03 PR-3)
imports this adapter; LangGraph itself is not imported here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from core.cognitive_router import AIProvider, TaskClass

__all__ = [
    "AllProvidersFailedError",
    "ChatTransport",
    "FallbackAuditSink",
    "NoEligibleProviderError",
    "RegistryDrivenChatModel",
    "TransientProviderError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransientProviderError(RuntimeError):
    """Raised by a :class:`ChatTransport` when the provider failed in a
    way that warrants trying the next eligible provider.

    Examples: 429 rate-limit, 503 overloaded, network timeout.
    Non-examples: 401 unauthenticated, 400 malformed request — those
    are deterministic and propagating them stops the chain.
    """


class NoEligibleProviderError(RuntimeError):
    """Raised when the registry has zero providers eligible for a task."""


class AllProvidersFailedError(RuntimeError):
    """Raised when every eligible provider raised :class:`TransientProviderError`."""


# ---------------------------------------------------------------------------
# Pluggable seams
# ---------------------------------------------------------------------------


@runtime_checkable
class ChatTransport(Protocol):
    """Per-turn dispatch to a single resolved AI provider.

    Implementations may be HTTP, MCP, gRPC, or a local stub. The
    transport is responsible for translating the registry row into a
    concrete client (auth, base URL, model name) — but not for
    selection, ordering, or fallback. Selection lives in
    :class:`RegistryDrivenChatModel`; the transport is dumb.
    """

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        """Send ``messages`` to ``provider`` and return the assistant's text.

        Raises:
            TransientProviderError: If the provider is temporarily
                unavailable. The adapter will log a fallback audit
                and try the next eligible provider.
            Exception: Any other exception type propagates without
                fallback. Reserve these for non-retriable failures
                (auth, schema, operator-cancelled).
        """


FallbackAuditSink = Callable[[AIProvider, str], None]
"""Callable invoked once per ``SOURCE_FALLBACK_ACTIVATED`` audit.

Signature: ``(provider, reason) -> None``. Production wiring routes
the call to ``state.ledger`` via the audit subsystem; tests pass a
list-appender. The default sink is a no-op."""


ProviderResolver = Callable[[], tuple[AIProvider, ...]]
"""Zero-arg callable returning eligible providers in priority order.

Production wiring: ``lambda: select_providers(registry, task)`` over
the SCVS source registry. Tests pass a tuple-returner."""


def _noop_audit(_provider: AIProvider, _reason: str) -> None:
    return None


# ---------------------------------------------------------------------------
# RegistryDrivenChatModel
# ---------------------------------------------------------------------------


class RegistryDrivenChatModel(BaseChatModel):
    """LangChain ``BaseChatModel`` that resolves providers from the registry.

    Construction parameters:

    * ``task`` — :class:`TaskClass` the operator is asking the chat
      surface to handle. Surfaced in ``generation_info`` for audit.
    * ``provider_resolver`` — :data:`ProviderResolver`. The single
      seam through which the adapter reads the eligible-provider
      tuple. Production wiring binds this to ``lambda:
      select_providers(registry, task)``; tests bind a tuple-returner.
    * ``transport`` — :class:`ChatTransport` instance. Production
      wiring constructs one transport that knows how to dispatch any
      eligible provider; tests pass a fake.
    * ``fallback_audit`` — :data:`FallbackAuditSink`. Defaults to a
      no-op so unit-test construction is cheap.

    The adapter is read-mostly. ``model_config`` permits ``arbitrary_
    types_allowed`` so the :class:`ChatTransport` and
    :data:`ProviderResolver` references survive Pydantic validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: TaskClass
    provider_resolver: ProviderResolver
    transport: ChatTransport
    fallback_audit: FallbackAuditSink = Field(default=_noop_audit)

    @property
    def _llm_type(self) -> str:  # pragma: no cover — LangChain plumbing
        return "dixvision_registry_driven_chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        eligible = self.provider_resolver()
        if not eligible:
            raise NoEligibleProviderError(
                f"no enabled AI providers in the registry have the"
                f" capabilities required for task={self.task.value!r}"
            )

        last_error: TransientProviderError | None = None
        for provider in eligible:
            try:
                text = self.transport.invoke(
                    provider, tuple(messages), stop=stop, **kwargs
                )
            except TransientProviderError as exc:
                last_error = exc
                self.fallback_audit(provider, str(exc))
                continue
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=text),
                        generation_info={
                            "provider_id": provider.id,
                            "provider": provider.provider,
                            "task": self.task.value,
                        },
                    )
                ],
                llm_output={
                    "provider_id": provider.id,
                    "provider": provider.provider,
                    "task": self.task.value,
                },
            )

        assert last_error is not None  # eligible was non-empty
        raise AllProvidersFailedError(
            f"every eligible provider for task={self.task.value!r}"
            f" raised TransientProviderError; last reason: {last_error}"
        ) from last_error
