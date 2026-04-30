"""Wave-03 PR-4 — HTTP-side runtime façade for the cognitive chat surface.

These tests exercise the :class:`CognitiveChatRuntime` glue that
``ui/server.py`` consumes — *not* the route handlers directly. The
route layer is a thin ``HTTPException`` wrapper, so locking the
runtime contract here keeps the tests free of TestClient + global
``STATE`` lock entanglement.

Coverage targets:

* feature flag off → ``ChatTurnDisabled`` (mapped to 503 in the route)
* feature flag on, happy path → ``ChatTurnResponse`` carries the
  fake transport's reply, the requested ``thread_id``, the registry
  ``provider_id``, and a non-empty ``checkpoint_id``
* no eligible providers → ``ChatTurnNoProvider`` (mapped to 502)
* the ``NotConfiguredTransport`` default raises
  ``NoEligibleProviderError`` (translated to ``ChatTurnNoProvider``)
* status response shape — ``enabled`` follows the flag, the
  registry ids surface in ``eligible_providers``, the env-var name
  is the canonical one
* the runtime never imports ``governance_engine`` or
  ``system_engine`` at the call boundary that would re-introduce a
  B1 violation (smoke check via the ledger-append adapter)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import BaseMessage

from core.cognitive_router import AIProvider
from core.contracts.api.cognitive_chat import (
    ChatMessageApi,
    ChatRoleApi,
    ChatTurnRequest,
)
from intelligence_engine.cognitive.chat import (
    FEATURE_FLAG_ENV_VAR,
    CognitiveChatFeatureFlag,
)
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
)
from ui.cognitive_chat_runtime import (
    ChatTurnDisabled,
    ChatTurnNoProvider,
    NotConfiguredTransport,
    build_ledger_append,
    build_runtime,
    new_thread_id,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingTransport:
    def __init__(self, *, reply: str = "fake-reply") -> None:
        self._reply = reply
        self.calls: list[tuple[AIProvider, tuple[BaseMessage, ...]]] = []

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        **kwargs: Any,
    ) -> str:
        self.calls.append((provider, tuple(messages)))
        return self._reply


class _RecordingLedger:
    """List-backed ``LedgerAuthorityWriter`` stand-in.

    Mirrors the ``append(ts_ns, kind, payload)`` signature used by
    the production writer so :func:`build_ledger_append` accepts it
    without a real governance engine in the picture."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, str, dict[str, str]]] = []

    def append(
        self,
        *,
        ts_ns: int,
        kind: str,
        payload: Mapping[str, str],
    ) -> None:
        self.rows.append((ts_ns, kind, dict(payload)))


def _registry_with(*provider_ids: str) -> SourceRegistry:
    decls = tuple(
        SourceDeclaration(
            id=pid,
            name=pid,
            category=SourceCategory.AI,
            provider=pid,
            endpoint=f"https://example.invalid/{pid}",
            schema="generic_chat",
            auth="bearer",
            enabled=True,
            critical=False,
            liveness_threshold_ms=0,
            capabilities=("reasoning",),
        )
        for pid in provider_ids
    )
    return SourceRegistry(version="test", sources=decls)


def _flag(value: str) -> CognitiveChatFeatureFlag:
    return CognitiveChatFeatureFlag(getter=lambda _n, _d: value)


def _request(text: str, thread_id: str = "thread-A") -> ChatTurnRequest:
    return ChatTurnRequest(
        thread_id=thread_id,
        messages=[ChatMessageApi(role=ChatRoleApi.USER, content=text)],
    )


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


def test_status_disabled_when_flag_off() -> None:
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag(""),
    )
    resp = runtime.status()
    assert resp.enabled is False
    assert resp.eligible_providers == []
    assert resp.feature_flag_env_var == FEATURE_FLAG_ENV_VAR


def test_status_lists_eligible_providers_when_flag_on() -> None:
    runtime = build_runtime(
        registry=_registry_with("provider-A", "provider-B"),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag("true"),
    )
    resp = runtime.status()
    assert resp.enabled is True
    assert sorted(resp.eligible_providers) == ["provider-A", "provider-B"]
    assert resp.feature_flag_env_var == FEATURE_FLAG_ENV_VAR


def test_status_enabled_with_no_providers_returns_empty_list() -> None:
    runtime = build_runtime(
        registry=_registry_with(),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag("on"),
    )
    resp = runtime.status()
    assert resp.enabled is True
    assert resp.eligible_providers == []


# ---------------------------------------------------------------------------
# turn() — happy path
# ---------------------------------------------------------------------------


def test_turn_happy_path_returns_reply_and_thread_metadata() -> None:
    transport = _RecordingTransport(reply="hello back")
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=transport,
        feature_flag=_flag("yes"),
    )
    resp = runtime.turn(_request("hi", thread_id="thread-X"))

    assert resp.thread_id == "thread-X"
    assert resp.reply.role is ChatRoleApi.ASSISTANT
    assert resp.reply.content == "hello back"
    assert resp.provider_id == "provider-A"
    assert resp.checkpoint_id != ""
    assert len(transport.calls) == 1
    provider, msgs = transport.calls[0]
    assert provider.id == "provider-A"
    assert len(msgs) == 1
    assert msgs[0].content == "hi"


def test_turn_persists_thread_state_across_calls() -> None:
    transport = _RecordingTransport(reply="ack")
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=transport,
        feature_flag=_flag("1"),
    )

    runtime.turn(_request("first", thread_id="thread-S"))
    # Second turn uses the same thread_id but only sends the new
    # user message — the LangGraph saver must replay the prior
    # state so the transport sees both messages.
    second = ChatTurnRequest(
        thread_id="thread-S",
        messages=[ChatMessageApi(role=ChatRoleApi.USER, content="second")],
    )
    runtime.turn(second)

    assert len(transport.calls) == 2
    _, second_call_msgs = transport.calls[1]
    contents = [m.content for m in second_call_msgs]
    assert "first" in contents
    assert "ack" in contents
    assert "second" in contents


def test_turn_reports_actual_provider_from_response_metadata() -> None:
    """``provider_id`` reflects the provider that *served* the turn.

    Regression for Devin Review BUG_0002: if the first eligible
    provider raises ``TransientProviderError``, the chat model
    falls back to the next one — and the operator must see that
    fallback in ``ChatTurnResponse.provider_id``, not the first
    registry entry.
    """

    from intelligence_engine.cognitive.chat import TransientProviderError

    class _FallbackTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def invoke(
            self,
            provider: AIProvider,
            messages: Sequence[BaseMessage],
            **kwargs: Any,
        ) -> str:
            self.calls.append(provider.id)
            if provider.id == "provider-A":
                raise TransientProviderError("provider-A is unavailable")
            return "served by B"

    transport = _FallbackTransport()
    runtime = build_runtime(
        registry=_registry_with("provider-A", "provider-B"),
        ledger_writer=_RecordingLedger(),
        transport=transport,
        feature_flag=_flag("yes"),
    )
    resp = runtime.turn(_request("hi"))
    assert transport.calls == ["provider-A", "provider-B"]
    assert resp.provider_id == "provider-B"
    assert resp.reply.content == "served by B"


def test_turn_isolates_separate_thread_ids() -> None:
    transport = _RecordingTransport(reply="ack")
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=transport,
        feature_flag=_flag("on"),
    )

    runtime.turn(_request("alpha", thread_id="thread-1"))
    runtime.turn(_request("beta", thread_id="thread-2"))

    _, second_call_msgs = transport.calls[1]
    contents = [m.content for m in second_call_msgs]
    assert "alpha" not in contents
    assert "beta" in contents


# ---------------------------------------------------------------------------
# turn() — error paths
# ---------------------------------------------------------------------------


def test_turn_raises_chat_turn_disabled_when_flag_off() -> None:
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag(""),
    )
    with pytest.raises(ChatTurnDisabled):
        runtime.turn(_request("hi"))


def test_turn_raises_no_provider_when_registry_empty() -> None:
    runtime = build_runtime(
        registry=_registry_with(),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag("true"),
    )
    with pytest.raises(ChatTurnNoProvider):
        runtime.turn(_request("hi"))


def test_default_transport_translates_to_chat_turn_no_provider() -> None:
    """Default :class:`NotConfiguredTransport` returns 502 not a 500."""

    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        feature_flag=_flag("yes"),
    )
    assert isinstance(runtime.transport, NotConfiguredTransport)
    with pytest.raises(ChatTurnNoProvider):
        runtime.turn(_request("hi"))


def test_turn_rejects_empty_messages() -> None:
    from pydantic import ValidationError

    # Pydantic min_length=1 stops empty lists at the contract layer;
    # this is a sanity check that the wire shape is locked before
    # the runtime ever sees the request.
    with pytest.raises(ValidationError):
        ChatTurnRequest(thread_id="t", messages=[])


def test_turn_rejects_assistant_tail_message() -> None:
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag("yes"),
    )
    req = ChatTurnRequest(
        thread_id="t",
        messages=[
            ChatMessageApi(role=ChatRoleApi.USER, content="a"),
            ChatMessageApi(role=ChatRoleApi.ASSISTANT, content="b"),
        ],
    )
    with pytest.raises(ValueError):
        runtime.turn(req)


def test_turn_rejects_system_messages_until_pr5() -> None:
    runtime = build_runtime(
        registry=_registry_with("provider-A"),
        ledger_writer=_RecordingLedger(),
        transport=_RecordingTransport(),
        feature_flag=_flag("yes"),
    )
    req = ChatTurnRequest(
        thread_id="t",
        messages=[
            ChatMessageApi(role=ChatRoleApi.SYSTEM, content="be helpful"),
            ChatMessageApi(role=ChatRoleApi.USER, content="hi"),
        ],
    )
    with pytest.raises(ValueError):
        runtime.turn(req)


# ---------------------------------------------------------------------------
# Ledger adapter / helpers
# ---------------------------------------------------------------------------


def test_build_ledger_append_stamps_ts_and_forwards_payload() -> None:
    writer = _RecordingLedger()
    append = build_ledger_append(writer)
    append("CHAT_TEST", {"k": "v"})
    assert len(writer.rows) == 1
    ts_ns, kind, payload = writer.rows[0]
    assert kind == "CHAT_TEST"
    assert payload == {"k": "v"}
    assert isinstance(ts_ns, int) and ts_ns > 0


def test_new_thread_id_is_unique() -> None:
    a = new_thread_id()
    b = new_thread_id()
    assert a != b
    assert a.isalnum() and len(a) == 32
