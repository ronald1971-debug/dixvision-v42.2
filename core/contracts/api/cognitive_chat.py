"""Pydantic response/request models for the ``/api/cognitive/chat/*`` HTTP surface.

Wave-03 PR-4 — first end-user-visible cognitive surface. The
operator dashboard's chat page (``/dash2/#/chat``) drives a
multi-turn conversation through the LangGraph chat graph from PR-3
(``intelligence_engine.cognitive.chat.assemble_cognitive_chat``).

These models live alongside the rest of ``core/contracts`` so the
TypeScript codegen (``tools/codegen/pydantic_to_ts.py``) can import a
single, stable namespace for the wave-02/03 dashboard. The route
handler in ``ui/server.py`` uses these as ``response_model=`` so
FastAPI's own validation matches the schema we ship to the client.

Only request and response shapes live here. The chat *graph* is
constructed inside ``intelligence_engine.cognitive.chat`` and
referenced by the route via callable seams — no graph internals
leak through the HTTP contract. The wire shape is deliberately
narrow (role + content + thread id) so PR-5's operator-approval
edge can grow it without breaking the existing client.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ChatRoleApi(StrEnum):
    """Role tag carried on every message in the conversation.

    Mirrors LangChain's ``HumanMessage`` / ``AIMessage`` distinction
    but flattened to a string-literal union so the TS client does
    not need to import any LangChain type. ``system`` is reserved
    for PR-5 (operator-approved system prompt) and currently unused;
    listing it now keeps the union stable across PRs."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessageApi(BaseModel):
    """One turn of the rendered transcript.

    ``content`` is a plain string today. PR-5 may grow it to a
    structured payload (citations, tool calls); when that happens
    this model gains an optional ``parts`` field rather than
    breaking the existing ``content`` contract."""

    model_config = ConfigDict(extra="forbid")

    role: ChatRoleApi
    content: str


class ChatTurnRequest(BaseModel):
    """Operator → server: one user turn plus the prior transcript.

    The client submits the *full* conversation so far. The server
    is the source of truth for thread state (LangGraph's checkpoint
    saver), but the request also carries the in-memory transcript
    so a freshly opened tab can resume a thread without a separate
    ``GET`` round-trip. The server validates that the last message
    is from the user; otherwise the request is rejected so the
    graph never replies to its own message."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Stable identifier scoping the LangGraph checkpoint. "
            "Operator-supplied; generated client-side as a uuid4 "
            "the first time the page is opened."
        ),
    )
    messages: list[ChatMessageApi] = Field(
        min_length=1,
        description=(
            "Full conversation so far, in order. The server uses "
            "the most recent ``USER`` message as the prompt and "
            "ignores any later assistant messages."
        ),
    )


class ChatTurnResponse(BaseModel):
    """Server → operator: the assistant's reply plus thread metadata.

    The reply is always present on a 200 response. Errors are
    returned as standard FastAPI ``HTTPException`` payloads
    (``{"detail": "..."}``) so the client does not need an
    error-typed branch in this schema."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    reply: ChatMessageApi
    provider_id: str = Field(
        description=(
            "Registry source id that served this turn. Surfaced "
            "so the operator can audit which provider was reached "
            "without grep'ing the audit ledger. ``\"\"`` if the "
            "transport returned a reply without resolving a "
            "registry row (test stubs)."
        ),
    )
    checkpoint_id: str = Field(
        description=(
            "Opaque LangGraph checkpoint id for this turn. "
            "Operators can cross-reference it against the "
            "``COGNITIVE_CHECKPOINT`` rows in the audit ledger."
        ),
    )


class ChatStatusResponse(BaseModel):
    """Server → operator: feature-flag / provider availability snapshot.

    The chat page polls this on mount so the UI can decide whether
    to show the input box or a "feature disabled" notice. Keeps
    the chat-turn endpoint focused on the actual turn — clients
    never have to inspect a 503 body to figure out why."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        description=(
            "True iff ``DIX_COGNITIVE_CHAT_ENABLED`` is set to a "
            "truthy value. False otherwise; the input box should "
            "be hidden in that case."
        ),
    )
    eligible_providers: list[str] = Field(
        description=(
            "Registry ids of providers eligible for "
            "``INDIRA_REASONING`` *right now*. Empty when the "
            "feature is disabled or no credentialed provider is "
            "configured. The chat page surfaces the count so the "
            "operator can see why a turn might fail before sending."
        ),
    )
    feature_flag_env_var: str = Field(
        description=(
            "Environment variable name the server reads. Surfaced "
            "so the operator can flip it locally without grep'ing "
            "the codebase."
        ),
    )
