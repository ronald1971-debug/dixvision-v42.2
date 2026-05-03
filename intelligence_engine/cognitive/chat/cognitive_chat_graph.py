"""Wave-03 PR-3 — LangGraph chat graph behind a feature flag.

This module assembles the first end-to-end cognitive surface for DIX
VISION: a LangGraph :class:`StateGraph` that drives multi-turn chat
through :class:`RegistryDrivenChatModel` (PR #82, INV-67 / B24) and
persists conversation state to the audit ledger via
:class:`AuditLedgerCheckpointSaver` (PR #83, GOV-CP-05).

Design pillars (Dashboard-2026 wave-03 plan §4):

* **Quarantined non-determinism (INV-67).** The chat graph is the
  only place in the codebase where LLM calls happen at runtime;
  it produces *advisory* messages, never typed bus events
  directly. Promotion to ``SignalEvent`` happens outside this
  module via Governance (out of scope for PR-3).
* **Registry-driven dispatch (B23 / B24).** The graph never names a
  vendor. The chat model is constructed from a
  :class:`ProviderResolver` callable injected at assembly time —
  same dependency-inversion pattern PR #82 used to keep
  ``intelligence_engine.cognitive.*`` free of any
  ``system_engine`` import (B1).
* **Audit-ledger checkpoints.** The graph's checkpointer is
  always :class:`AuditLedgerCheckpointSaver`; LangGraph's default
  ``SqliteSaver`` is forbidden. Conversation state lands in the
  same hash chain that protects governance.
* **Off by default.** Construction is gated by
  :class:`CognitiveChatFeatureFlag`, which reads the
  ``DIX_COGNITIVE_CHAT_ENABLED`` environment variable (default
  ``false``). :func:`assemble_cognitive_chat` raises
  :class:`CognitiveChatDisabledError` unless the flag is on,
  so an accidental import in production cannot bring up the
  graph.

The graph itself is intentionally minimal in PR-3: a single
``chat`` node that calls the model once per turn and appends the
reply to the conversation. Multi-agent supervision and operator-
gated proposal emission land in subsequent PRs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core.cognitive_router import TaskClass
from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    ChatTransport,
    ProviderResolver,
    RegistryDrivenChatModel,
)
from intelligence_engine.cognitive.checkpointing import (
    AuditLedgerCheckpointSaver,
    LedgerAppend,
)

__all__ = [
    "ChatGraphState",
    "CognitiveChatBundle",
    "CognitiveChatDisabledError",
    "CognitiveChatFeatureFlag",
    "FEATURE_FLAG_ENV_VAR",
    "assemble_cognitive_chat",
    "build_cognitive_chat_graph",
]


FEATURE_FLAG_ENV_VAR = "DIX_COGNITIVE_CHAT_ENABLED"
"""Environment variable that gates :func:`assemble_cognitive_chat`.

Cognitive chat is **on by default**. The flag is read whenever the
runtime evaluates :attr:`CognitiveChatFeatureFlag.enabled`; only the
explicit falsy set (``"0"``, ``"false"``, ``"no"``, ``"off"`` —
case-insensitive) flips it off. Unset / empty / unknown values keep
it enabled. Truthy values (``"1"``, ``"true"``, ``"yes"``, ``"on"``)
are accepted for symmetry with operator muscle memory but are
redundant given the on-by-default behaviour."""


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


class ChatGraphState(TypedDict):
    """Conversation state flowing through the cognitive chat graph.

    ``messages`` accumulates with the LangGraph ``add_messages``
    reducer so each node returns *just the new* messages it
    produced, not the full history.
    """

    messages: Annotated[list[BaseMessage], add_messages]


class CognitiveChatDisabledError(RuntimeError):
    """Raised by :func:`assemble_cognitive_chat` when the feature flag is off.

    Distinct from a configuration error — production processes
    should never see this raised; the import path stays cheap and
    the graph never wires up."""


@dataclass(frozen=True)
class CognitiveChatFeatureFlag:
    """Stateless reader for the cognitive-chat feature flag.

    Frozen + dataclass so multiple call sites observe the same
    semantics; an explicit ``getter`` parameter makes the flag
    testable without environment manipulation. Production wiring
    leaves ``getter`` at the default (``os.getenv``)."""

    getter: Callable[[str, str], str] = os.getenv  # type: ignore[assignment]

    @property
    def enabled(self) -> bool:
        raw = self.getter(FEATURE_FLAG_ENV_VAR, "").strip().lower()
        if raw in _FALSY:
            return False
        # Empty / unset / unknown / truthy → enabled.
        return True


@dataclass(frozen=True)
class CognitiveChatBundle:
    """Bundle returned by :func:`assemble_cognitive_chat`.

    Carries the compiled graph alongside the model and saver it
    was built from so callers can hold references for inspection
    (e.g. an operator dashboard wanting to surface the active
    provider list, or a forensic auditor cross-referencing the
    saver's ledger rows against the graph's run)."""

    graph: Any
    model: RegistryDrivenChatModel
    saver: AuditLedgerCheckpointSaver


def _chat_node_factory(
    model: RegistryDrivenChatModel,
) -> Callable[[ChatGraphState], dict[str, list[BaseMessage]]]:
    """Build the single ``chat`` node used by the graph.

    The node calls the model with the full conversation so far and
    returns a one-element list with the reply. The
    ``add_messages`` reducer on :class:`ChatGraphState` then
    appends it to the persisted history."""

    def _chat(state: ChatGraphState) -> dict[str, list[BaseMessage]]:
        reply = model.invoke(state["messages"])
        return {"messages": [reply]}

    return _chat


def build_cognitive_chat_graph(
    *,
    model: RegistryDrivenChatModel,
    saver: BaseCheckpointSaver[Any],
) -> Any:
    """Compile the LangGraph chat graph wired to ``model`` and ``saver``.

    ``saver`` is typed as :class:`BaseCheckpointSaver` rather than
    :class:`AuditLedgerCheckpointSaver` so tests can sub in a
    fake. Production wiring (via :func:`assemble_cognitive_chat`)
    still pins it to the audit-ledger saver — the type relaxation
    is unit-test scaffolding, not a public contract."""

    builder: StateGraph = StateGraph(ChatGraphState)
    builder.add_node("chat", _chat_node_factory(model))
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=saver)


def assemble_cognitive_chat(
    *,
    task: TaskClass,
    provider_resolver: ProviderResolver,
    transport: ChatTransport,
    ledger_append: LedgerAppend,
    feature_flag: CognitiveChatFeatureFlag | None = None,
) -> CognitiveChatBundle:
    """Bring up the chat graph end-to-end.

    Wires :class:`RegistryDrivenChatModel` against
    ``provider_resolver`` + ``transport``, an
    :class:`AuditLedgerCheckpointSaver` against ``ledger_append``,
    and compiles them into the LangGraph chat graph.

    Raises :class:`CognitiveChatDisabledError` unless the cognitive
    chat feature flag is on. The check happens *before* any object
    is constructed so an accidentally-imported call site is cheap."""

    flag = feature_flag if feature_flag is not None else CognitiveChatFeatureFlag()
    if not flag.enabled:
        raise CognitiveChatDisabledError(
            "cognitive chat is disabled — set"
            f" {FEATURE_FLAG_ENV_VAR}=true to enable"
        )

    model = RegistryDrivenChatModel(
        task=task,
        provider_resolver=provider_resolver,
        transport=transport,
    )
    saver = AuditLedgerCheckpointSaver(ledger_append=ledger_append)
    graph = build_cognitive_chat_graph(model=model, saver=saver)
    return CognitiveChatBundle(graph=graph, model=model, saver=saver)


def _ensure_message_sequence(
    messages: Sequence[BaseMessage],
) -> tuple[BaseMessage, ...]:
    """Defensive coercion used by callers that read raw operator input.

    Kept as a free function so the dashboard handler in PR-4 can
    reuse it without re-importing LangGraph internals; for PR-3 it
    is exercised only by tests as a public-surface check."""

    return tuple(messages)
