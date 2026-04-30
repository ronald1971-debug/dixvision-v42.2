"""Wave-03 PR-1 — RegistryDrivenChatModel adapter (INV-67 / B24).

Locks the contract that the cognitive layer dispatches to AI providers
exclusively through the SCVS registry, never by vendor name. The
adapter is a thin :class:`BaseChatModel` subclass; tests use a fake
:class:`ChatTransport` so no live provider is contacted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from core.cognitive_router import AIProvider, TaskClass, select_providers
from intelligence_engine.cognitive.chat import (
    AllProvidersFailedError,
    ChatTransport,
    NoEligibleProviderError,
    RegistryDrivenChatModel,
    TransientProviderError,
)
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
)


def _resolver(registry: SourceRegistry, task: TaskClass):
    """Bind a registry+task to the zero-arg ProviderResolver shape."""

    def _inner() -> tuple[AIProvider, ...]:
        return select_providers(registry, task)

    return _inner


# ---------------------------------------------------------------------------
# Fixtures — frozen registries for the eligible / mixed / empty paths
# ---------------------------------------------------------------------------


def _ai_row(
    *,
    source_id: str,
    capabilities: tuple[str, ...],
    enabled: bool = True,
) -> SourceDeclaration:
    return SourceDeclaration(
        id=source_id,
        name=source_id,
        category=SourceCategory.AI,
        provider=source_id,
        endpoint=f"https://example.invalid/{source_id}",
        schema="generic_chat",
        auth="bearer",
        enabled=enabled,
        critical=False,
        liveness_threshold_ms=0,
        capabilities=capabilities,
    )


def _registry_with_two_reasoning_providers() -> SourceRegistry:
    return SourceRegistry(
        version="test",
        sources=(
            _ai_row(source_id="prov-a", capabilities=("reasoning",)),
            _ai_row(source_id="prov-b", capabilities=("reasoning",)),
        ),
    )


def _registry_no_eligible() -> SourceRegistry:
    # Has an AI row but it lacks the 'reasoning' capability.
    return SourceRegistry(
        version="test",
        sources=(
            _ai_row(source_id="prov-x", capabilities=("realtime_search",)),
        ),
    )


# ---------------------------------------------------------------------------
# Fake transports
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """Returns a deterministic reply, records every (provider, kwargs) call."""

    def __init__(self, *, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[tuple[AIProvider, tuple[BaseMessage, ...], dict[str, Any]]] = []

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        self.calls.append((provider, tuple(messages), dict(kwargs)))
        return self.reply


class _ScriptedTransport:
    """Per-call behaviour driven by a script of ('ok'|'transient'|'fatal'|str)."""

    def __init__(self, *, script: tuple[str, ...]) -> None:
        self._script = list(script)
        self.calls: list[AIProvider] = []

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        self.calls.append(provider)
        if not self._script:
            raise AssertionError("transport called more times than scripted")
        action = self._script.pop(0)
        if action == "transient":
            raise TransientProviderError(f"transient on {provider.id}")
        if action == "fatal":
            raise RuntimeError(f"fatal on {provider.id}")
        return action  # treated as the reply text


# ---------------------------------------------------------------------------
# Construction is registry-driven (no vendor names in the adapter)
# ---------------------------------------------------------------------------


def test_construction_takes_task_registry_transport() -> None:
    transport = _RecordingTransport()
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
    )
    assert model.task is TaskClass.INDIRA_REASONING
    assert model.transport is transport


def test_llm_type_does_not_name_a_vendor() -> None:
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=_RecordingTransport(),
    )
    # The LangChain identity must not name a vendor.
    assert model._llm_type == "dixvision_registry_driven_chat"
    for vendor in ("vendor-a", "vendor-b", "vendor-c"):
        assert vendor not in model._llm_type.lower()


# ---------------------------------------------------------------------------
# Happy path — first eligible provider answers
# ---------------------------------------------------------------------------


def test_invoke_dispatches_to_first_eligible_provider() -> None:
    transport = _RecordingTransport(reply="hello")
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
    )
    result = model.invoke([HumanMessage(content="hi")])
    assert result.content == "hello"
    assert len(transport.calls) == 1
    chosen, msgs, _kwargs = transport.calls[0]
    assert chosen.id == "prov-a"  # first row in registry
    assert isinstance(msgs[0], HumanMessage)


def test_generation_info_carries_provider_provenance() -> None:
    transport = _RecordingTransport(reply="hi")
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
    )
    out = model._generate([HumanMessage(content="x")])
    info = out.generations[0].generation_info
    assert info is not None
    assert info["provider_id"] == "prov-a"
    assert info["provider"] == "prov-a"
    assert info["task"] == TaskClass.INDIRA_REASONING.value


# ---------------------------------------------------------------------------
# Fallback — transient on prov-a, prov-b answers
# ---------------------------------------------------------------------------


def test_transient_failure_triggers_fallback_to_next_provider() -> None:
    transport = _ScriptedTransport(script=("transient", "fallback-reply"))
    audits: list[tuple[str, str]] = []

    def audit(provider: AIProvider, reason: str) -> None:
        audits.append((provider.id, reason))

    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        fallback_audit=audit,
    )
    result = model.invoke([HumanMessage(content="x")])
    assert result.content == "fallback-reply"
    assert [p.id for p in transport.calls] == ["prov-a", "prov-b"]
    assert len(audits) == 1
    assert audits[0][0] == "prov-a"
    assert "transient on prov-a" in audits[0][1]


def test_audit_is_optional_and_defaults_to_noop() -> None:
    # Same scripted-fallback scenario, but no audit sink wired. The
    # call should still succeed.
    transport = _ScriptedTransport(script=("transient", "ok"))
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
    )
    result = model.invoke([HumanMessage(content="x")])
    assert result.content == "ok"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_no_eligible_provider_raises_with_clear_message() -> None:
    registry = _registry_no_eligible()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=_RecordingTransport(),
    )
    with pytest.raises(NoEligibleProviderError) as excinfo:
        model.invoke([HumanMessage(content="x")])
    assert TaskClass.INDIRA_REASONING.value in str(excinfo.value)


def test_all_providers_transient_raises_all_providers_failed() -> None:
    transport = _ScriptedTransport(script=("transient", "transient"))
    audits: list[str] = []
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        fallback_audit=lambda p, r: audits.append(p.id),
    )
    with pytest.raises(AllProvidersFailedError) as excinfo:
        model.invoke([HumanMessage(content="x")])
    # Both providers were tried, both audited.
    assert audits == ["prov-a", "prov-b"]
    # The last transient is wrapped as __cause__ so the operator can see it.
    assert isinstance(excinfo.value.__cause__, TransientProviderError)


def test_non_transient_exception_propagates_without_fallback() -> None:
    transport = _ScriptedTransport(script=("fatal", "should-not-reach"))
    audits: list[str] = []
    registry = _registry_with_two_reasoning_providers()
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        fallback_audit=lambda p, r: audits.append(p.id),
    )
    with pytest.raises(RuntimeError) as excinfo:
        model.invoke([HumanMessage(content="x")])
    # Non-transient errors are NOT fallback-eligible — silently swallowing
    # them would mask real bugs.
    assert "fatal on prov-a" in str(excinfo.value)
    assert audits == []
    # prov-b must not have been contacted.
    assert [p.id for p in transport.calls] == ["prov-a"]


# ---------------------------------------------------------------------------
# Disabled rows are skipped (the cognitive router enforces this; we
# pin the contract here as well).
# ---------------------------------------------------------------------------


def test_disabled_provider_is_not_selected() -> None:
    registry = SourceRegistry(
        version="test",
        sources=(
            _ai_row(source_id="prov-a", capabilities=("reasoning",), enabled=False),
            _ai_row(source_id="prov-b", capabilities=("reasoning",)),
        ),
    )
    transport = _RecordingTransport(reply="ok")
    model = RegistryDrivenChatModel(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
    )
    model.invoke([HumanMessage(content="x")])
    assert [p.id for p, _msgs, _kw in transport.calls] == ["prov-b"]


# ---------------------------------------------------------------------------
# Vendor-name negative — the adapter source must not contain provider tokens
# ---------------------------------------------------------------------------


def test_adapter_source_does_not_name_a_vendor() -> None:
    """Mirrors B23 at the adapter level — the registry-driven adapter
    must not contain any provider token in its source code. We allow
    them to appear in *docstrings* (where they are pedagogical), so
    this is a "no token outside docstrings" check via AST."""

    import ast
    import inspect

    from intelligence_engine.cognitive.chat import (
        registry_driven_chat_model,
    )

    src = inspect.getsource(registry_driven_chat_model)
    tree = ast.parse(src)

    forbidden = (
        "o" + "penai",
        "a" + "nthropic",
        "c" + "laude",
        "g" + "emini",
        "m" + "istral",
        "q" + "wen",
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.lower()
            for token in forbidden:
                if token in value:
                    raise AssertionError(
                        f"forbidden vendor token {token!r} appears in"
                        f" string literal: {value!r}"
                    )


# ---------------------------------------------------------------------------
# Pluggable seam contract — ChatTransport is a runtime-checkable Protocol
# ---------------------------------------------------------------------------


def test_recording_transport_satisfies_chat_transport_protocol() -> None:
    assert isinstance(_RecordingTransport(), ChatTransport)


def test_scripted_transport_satisfies_chat_transport_protocol() -> None:
    assert isinstance(_ScriptedTransport(script=()), ChatTransport)
