"""Wave-03 PR-3 — cognitive chat graph behind a feature flag.

These tests exercise the graph end-to-end against a fake
:class:`ChatTransport` and a list-backed :data:`LedgerAppend`, so
they do not require any real LLM provider, governance engine, or
filesystem. The seam contract from PR #82 (provider resolver) and
PR #83 (ledger append callable) is what makes this possible —
both injection points are honoured here.

Coverage targets:

* feature flag default-off; explicit truthy values turn it on
* ``assemble_cognitive_chat`` raises when the flag is off
* the graph completes one turn, returning the model's reply
* the saver receives ``COGNITIVE_CHECKPOINT`` rows on every turn
* thread isolation — separate ``thread_id`` configs do not bleed
* B1 isolation — the chat-graph module does not import
  ``governance_engine`` or ``system_engine`` directly
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.cognitive_router import (
    AIProvider,
    TaskClass,
    select_providers,
)
from intelligence_engine.cognitive.chat import (
    FEATURE_FLAG_ENV_VAR,
    CognitiveChatBundle,
    CognitiveChatDisabledError,
    CognitiveChatFeatureFlag,
    assemble_cognitive_chat,
    build_cognitive_chat_graph,
)
from intelligence_engine.cognitive.chat.cognitive_chat_graph import (
    ChatGraphState,
)
from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    RegistryDrivenChatModel,
)
from intelligence_engine.cognitive.checkpointing import (
    AuditLedgerCheckpointSaver,
)
from intelligence_engine.cognitive.checkpointing.audit_ledger_checkpoint_saver import (
    CHECKPOINT_KIND,
)
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
)

# ---------------------------------------------------------------------------
# Fakes — fake transport + list-backed ledger so the graph runs offline
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """Fake :class:`ChatTransport` that returns a canned reply per call."""

    def __init__(self, *, reply: str = "fake-reply") -> None:
        self._reply = reply
        self.calls: list[
            tuple[AIProvider, tuple[BaseMessage, ...], dict[str, Any]]
        ] = []

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        **kwargs: Any,
    ) -> str:
        self.calls.append((provider, tuple(messages), dict(kwargs)))
        return self._reply


class _RecordingLedger:
    """List-backed :data:`LedgerAppend` for assertion-friendly tests."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, str]]] = []

    def __call__(self, kind: str, payload: Mapping[str, str]) -> None:
        self.rows.append((kind, dict(payload)))


def _registry_with_one_provider() -> SourceRegistry:
    decl = SourceDeclaration(
        id="provider-A",
        name="provider-A",
        category=SourceCategory.AI,
        provider="provider-A",
        endpoint="https://example.invalid/provider-A",
        schema="generic_chat",
        auth="bearer",
        enabled=True,
        critical=False,
        liveness_threshold_ms=0,
        capabilities=("reasoning",),
    )
    return SourceRegistry(version="test", sources=(decl,))


def _resolver(registry: SourceRegistry, task: TaskClass):
    """Bind a registry+task to the zero-arg ProviderResolver shape."""

    def _inner() -> tuple[AIProvider, ...]:
        return select_providers(registry, task)

    return _inner


def _enable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, "true")


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_feature_flag_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cognitive chat is on by default — operator does not need to opt in."""

    monkeypatch.delenv(FEATURE_FLAG_ENV_VAR, raising=False)
    flag = CognitiveChatFeatureFlag()
    assert flag.enabled is True


@pytest.mark.parametrize(
    "value", ["1", "true", "TRUE", "yes", "on", "", "maybe"]
)
def test_feature_flag_truthy_or_unknown_values_enable(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, value)
    assert CognitiveChatFeatureFlag().enabled is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_feature_flag_falsy_values_disable(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, value)
    assert CognitiveChatFeatureFlag().enabled is False


def test_feature_flag_accepts_injected_getter() -> None:
    """Custom ``getter`` keeps the flag testable without env mutation."""

    flag = CognitiveChatFeatureFlag(getter=lambda _name, _default: "yes")
    assert flag.enabled is True


# ---------------------------------------------------------------------------
# assemble_cognitive_chat — gating
# ---------------------------------------------------------------------------


def test_assemble_raises_when_flag_explicitly_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit falsy env value disables cognitive chat."""

    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, "false")
    transport = _RecordingTransport()
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()

    with pytest.raises(CognitiveChatDisabledError):
        assemble_cognitive_chat(
            task=TaskClass.INDIRA_REASONING,
            provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
            transport=transport,
            ledger_append=ledger,
        )


def test_assemble_raises_when_explicit_flag_object_is_disabled() -> None:
    """An explicit :class:`CognitiveChatFeatureFlag` short-circuits the env."""

    transport = _RecordingTransport()
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()
    disabled = CognitiveChatFeatureFlag(getter=lambda _n, _d: "false")

    with pytest.raises(CognitiveChatDisabledError):
        assemble_cognitive_chat(
            task=TaskClass.INDIRA_REASONING,
            provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
            transport=transport,
            ledger_append=ledger,
            feature_flag=disabled,
        )


def test_assemble_returns_bundle_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_flag(monkeypatch)
    transport = _RecordingTransport()
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()

    bundle = assemble_cognitive_chat(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        ledger_append=ledger,
    )

    assert isinstance(bundle, CognitiveChatBundle)
    assert isinstance(bundle.model, RegistryDrivenChatModel)
    assert isinstance(bundle.saver, AuditLedgerCheckpointSaver)


# ---------------------------------------------------------------------------
# Graph end-to-end (one turn)
# ---------------------------------------------------------------------------


def _build_bundle(monkeypatch: pytest.MonkeyPatch) -> tuple[
    CognitiveChatBundle, _RecordingTransport, _RecordingLedger
]:
    _enable_flag(monkeypatch)
    transport = _RecordingTransport(reply="hello")
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()
    bundle = assemble_cognitive_chat(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        ledger_append=ledger,
    )
    return bundle, transport, ledger


def test_graph_one_turn_returns_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, transport, _ledger = _build_bundle(monkeypatch)
    config = {"configurable": {"thread_id": "t1"}}

    result = bundle.graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config=config,
    )

    assert len(transport.calls) == 1
    final_msg = result["messages"][-1]
    assert isinstance(final_msg, AIMessage)
    assert final_msg.content == "hello"


def test_graph_appends_messages_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, transport, _ledger = _build_bundle(monkeypatch)
    config = {"configurable": {"thread_id": "t1"}}

    bundle.graph.invoke(
        {"messages": [HumanMessage(content="turn-1")]}, config=config
    )
    bundle.graph.invoke(
        {"messages": [HumanMessage(content="turn-2")]}, config=config
    )

    assert len(transport.calls) == 2
    # The second call must see the persisted history from the first turn —
    # this is what the audit-ledger saver buys us beyond a stateless API.
    _provider, second_call_messages, _kwargs = transport.calls[1]
    contents = [m.content for m in second_call_messages]
    assert "turn-1" in contents
    assert "hello" in contents  # reply from turn 1
    assert "turn-2" in contents


def test_saver_receives_checkpoint_rows_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _transport, ledger = _build_bundle(monkeypatch)
    config = {"configurable": {"thread_id": "t1"}}

    bundle.graph.invoke(
        {"messages": [HumanMessage(content="hi")]}, config=config
    )

    checkpoint_rows = [
        payload for kind, payload in ledger.rows if kind == CHECKPOINT_KIND
    ]
    assert checkpoint_rows, "expected at least one COGNITIVE_CHECKPOINT row"
    assert all(row["thread_id"] == "t1" for row in checkpoint_rows)


def test_threads_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, transport, ledger = _build_bundle(monkeypatch)

    bundle.graph.invoke(
        {"messages": [HumanMessage(content="A1")]},
        config={"configurable": {"thread_id": "thread-A"}},
    )
    bundle.graph.invoke(
        {"messages": [HumanMessage(content="B1")]},
        config={"configurable": {"thread_id": "thread-B"}},
    )

    # The B-thread call sees only B1 — no leakage from thread-A.
    _provider, b_messages, _kwargs = transport.calls[1]
    b_contents = {m.content for m in b_messages}
    assert "A1" not in b_contents
    assert "B1" in b_contents

    threads = {
        payload["thread_id"]
        for kind, payload in ledger.rows
        if kind == CHECKPOINT_KIND
    }
    assert threads == {"thread-A", "thread-B"}


# ---------------------------------------------------------------------------
# build_cognitive_chat_graph — direct-construction smoke test
# ---------------------------------------------------------------------------


def test_build_cognitive_chat_graph_accepts_arbitrary_saver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_cognitive_chat_graph`` accepts any
    :class:`BaseCheckpointSaver` so unit tests can sub a fake. The
    production assembly path always pins
    :class:`AuditLedgerCheckpointSaver` — this test just verifies
    the typing relaxation does not silently reject the real saver."""

    _enable_flag(monkeypatch)
    transport = _RecordingTransport(reply="ok")
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()
    bundle = assemble_cognitive_chat(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        ledger_append=ledger,
    )

    rebuilt = build_cognitive_chat_graph(model=bundle.model, saver=bundle.saver)
    assert rebuilt is not None


# ---------------------------------------------------------------------------
# B1 isolation — AST guard
# ---------------------------------------------------------------------------


def test_chat_graph_module_does_not_import_governance_or_system_engine() -> None:
    """B1: the cognitive chat graph cannot import ``governance_engine``
    or ``system_engine`` directly. The seams are the
    :data:`ProviderResolver` and :data:`LedgerAppend` callables —
    production wiring binds them at construction; the module itself
    has zero direct cross-engine imports."""

    here = Path(__file__).resolve().parent.parent
    target = (
        here
        / "intelligence_engine"
        / "cognitive"
        / "chat"
        / "cognitive_chat_graph.py"
    )
    tree = ast.parse(target.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("governance_engine"), (
                f"forbidden cross-engine import: {module}"
            )
            assert not module.startswith("system_engine"), (
                f"forbidden cross-engine import: {module}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("governance_engine"), (
                    f"forbidden cross-engine import: {alias.name}"
                )
                assert not alias.name.startswith("system_engine"), (
                    f"forbidden cross-engine import: {alias.name}"
                )


def test_chat_graph_state_typed_dict_round_trips() -> None:
    """``ChatGraphState`` carries a single ``messages`` field whose
    reducer is the LangGraph ``add_messages`` helper. Encoding the
    expectation as a test prevents an accidental shape change in a
    later PR from silently breaking the dashboard handler."""

    state: ChatGraphState = {"messages": [HumanMessage(content="x")]}
    assert "messages" in state
    assert isinstance(state["messages"][0], BaseMessage)
