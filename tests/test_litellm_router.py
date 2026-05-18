"""Tests for ``intelligence_engine/cognitive/litellm_router.py`` (S-12)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from core.cognitive_router import AIProvider, TaskClass
from intelligence_engine.cognitive import litellm_router as router_mod
from intelligence_engine.cognitive.litellm_router import (
    DEFAULT_TIMEOUT_S,
    MAX_TIMEOUT_S,
    AllProvidersFailedError,
    ChatMessage,
    LiteLLMRouter,
    LLMRequest,
    LLMResponse,
    LLMTransport,
    LLMUsage,
    NoEligibleProviderError,
    ProviderTimeoutError,
    TransientProviderError,
    litellm_completion_transport,
)

_MOD_PATH = Path(router_mod.__file__)


# ---------------------------------------------------------------------------
# Module metadata + AST authority pins
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_litellm_only() -> None:
    assert router_mod.NEW_PIP_DEPENDENCIES == ("litellm",)


def test_adapted_from_header_present() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    assert "ADAPTED FROM: BerriAI/litellm" in src


def test_max_timeout_constant_is_30s() -> None:
    assert MAX_TIMEOUT_S == 30.0
    assert DEFAULT_TIMEOUT_S == 30.0


def _module_imports() -> set[str]:
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                names.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module)
    return names


def test_module_does_not_import_litellm_at_top_level() -> None:
    """The router must never top-level-import litellm; the production
    transport lazy-imports it inside the factory."""

    imports = _module_imports()
    # Function-local imports for litellm are present, but they must
    # not appear in the module-level import set.
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                top_level.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                top_level.add(node.module)
    assert "litellm" not in top_level
    # The function-body import is still present somewhere in the AST.
    assert "litellm" in imports


def test_module_does_not_import_clock_or_engine_modules() -> None:
    """RUNTIME_SAFE: no clock reads, no system_engine/governance_engine
    cross-imports, no asyncio/websockets, no os/datetime/time."""

    imports = _module_imports()
    forbidden = {
        "asyncio",
        "datetime",
        "execution_engine",
        "governance_engine",
        "numpy",
        "os",
        "polars",
        "system_engine",
        "time",
        "websockets",
    }
    leaked = imports & forbidden
    assert not leaked, f"forbidden imports leaked into router: {leaked}"


# ---------------------------------------------------------------------------
# ChatMessage validation
# ---------------------------------------------------------------------------


def test_chat_message_accepts_valid_roles() -> None:
    for role in ("system", "user", "assistant"):
        m = ChatMessage(role=role, content="hi")
        assert m.role == role


@pytest.mark.parametrize("role", ["", "tool", "function", "USER", "buy"])
def test_chat_message_rejects_invalid_role(role: str) -> None:
    with pytest.raises(ValueError, match="role"):
        ChatMessage(role=role, content="hi")


def test_chat_message_rejects_non_str_role() -> None:
    with pytest.raises(TypeError, match="role"):
        ChatMessage(role=1, content="hi")  # type: ignore[arg-type]


def test_chat_message_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content"):
        ChatMessage(role="user", content="")


def test_chat_message_rejects_non_str_content() -> None:
    with pytest.raises(TypeError, match="content"):
        ChatMessage(role="user", content=123)  # type: ignore[arg-type]


def test_chat_message_is_frozen() -> None:
    m = ChatMessage(role="user", content="hi")
    with pytest.raises((AttributeError, TypeError)):
        m.role = "system"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMRequest validation
# ---------------------------------------------------------------------------


def _msgs() -> tuple[ChatMessage, ...]:
    return (ChatMessage(role="user", content="hello"),)


def test_llm_request_defaults_use_30s_timeout() -> None:
    r = LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs())
    assert r.timeout_s == 30.0
    assert r.max_tokens == 1024
    assert r.temperature == 0.0


def test_llm_request_rejects_non_taskclass_task() -> None:
    with pytest.raises(TypeError, match="task"):
        LLMRequest(task="reasoning", messages=_msgs())  # type: ignore[arg-type]


def test_llm_request_rejects_non_tuple_messages() -> None:
    with pytest.raises(TypeError, match="messages"):
        LLMRequest(
            task=TaskClass.INDIRA_REASONING,
            messages=[ChatMessage(role="user", content="hi")],  # type: ignore[arg-type]
        )


def test_llm_request_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="messages"):
        LLMRequest(task=TaskClass.INDIRA_REASONING, messages=())


def test_llm_request_rejects_non_chatmessage_in_messages() -> None:
    with pytest.raises(TypeError, match="messages"):
        LLMRequest(
            task=TaskClass.INDIRA_REASONING,
            messages=({"role": "user", "content": "hi"},),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_tokens", [0, -1, -100])
def test_llm_request_rejects_non_positive_max_tokens(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs(), max_tokens=max_tokens)


def test_llm_request_rejects_bool_max_tokens() -> None:
    with pytest.raises(TypeError, match="max_tokens"):
        LLMRequest(
            task=TaskClass.INDIRA_REASONING,
            messages=_msgs(),
            max_tokens=True,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("temp", [-0.1, 2.01, 5.0, -1.0])
def test_llm_request_rejects_temperature_out_of_range(temp: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs(), temperature=temp)


@pytest.mark.parametrize("ts", [0.0, -0.001, -1.0])
def test_llm_request_rejects_non_positive_timeout(ts: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs(), timeout_s=ts)


@pytest.mark.parametrize("ts", [30.001, 31.0, 60.0, 1_000_000.0])
def test_llm_request_rejects_timeout_above_30s(ts: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs(), timeout_s=ts)


def test_llm_request_accepts_timeout_at_ceiling() -> None:
    r = LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs(), timeout_s=30.0)
    assert r.timeout_s == 30.0


def test_llm_request_is_frozen() -> None:
    r = LLMRequest(task=TaskClass.INDIRA_REASONING, messages=_msgs())
    with pytest.raises((AttributeError, TypeError)):
        r.max_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMUsage validation
# ---------------------------------------------------------------------------


def test_llm_usage_total_tokens_is_sum() -> None:
    u = LLMUsage(prompt_tokens=10, completion_tokens=20, cost_usd=0.01)
    assert u.total_tokens == 30


@pytest.mark.parametrize(
    "p,c",
    [(-1, 0), (0, -1), (-5, -10)],
)
def test_llm_usage_rejects_negative_tokens(p: int, c: int) -> None:
    with pytest.raises(ValueError):
        LLMUsage(prompt_tokens=p, completion_tokens=c, cost_usd=0.0)


def test_llm_usage_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        LLMUsage(prompt_tokens=1, completion_tokens=1, cost_usd=-0.01)


def test_llm_usage_rejects_nan_cost() -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        LLMUsage(prompt_tokens=1, completion_tokens=1, cost_usd=float("nan"))


def test_llm_usage_rejects_bool_tokens() -> None:
    with pytest.raises(TypeError, match="prompt_tokens"):
        LLMUsage(
            prompt_tokens=True,  # type: ignore[arg-type]
            completion_tokens=0,
            cost_usd=0.0,
        )


def test_llm_usage_is_frozen() -> None:
    u = LLMUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0)
    with pytest.raises((AttributeError, TypeError)):
        u.cost_usd = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMResponse validation
# ---------------------------------------------------------------------------


def _usage() -> LLMUsage:
    return LLMUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0)


def test_llm_response_rejects_non_positive_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        LLMResponse(
            request_id="r1",
            ts_ns=0,
            provider_id="p1",
            content="hi",
            usage=_usage(),
            attempts=("p1",),
        )


def test_llm_response_rejects_empty_request_id() -> None:
    with pytest.raises(ValueError, match="request_id"):
        LLMResponse(
            request_id="",
            ts_ns=1,
            provider_id="p1",
            content="hi",
            usage=_usage(),
            attempts=("p1",),
        )


def test_llm_response_rejects_attempts_tail_mismatch() -> None:
    with pytest.raises(ValueError, match="attempts"):
        LLMResponse(
            request_id="r1",
            ts_ns=1,
            provider_id="p1",
            content="hi",
            usage=_usage(),
            attempts=("p2",),
        )


def test_llm_response_rejects_empty_attempts() -> None:
    with pytest.raises(ValueError, match="attempts"):
        LLMResponse(
            request_id="r1",
            ts_ns=1,
            provider_id="p1",
            content="hi",
            usage=_usage(),
            attempts=(),
        )


def test_llm_response_is_frozen() -> None:
    r = LLMResponse(
        request_id="r1",
        ts_ns=1,
        provider_id="p1",
        content="hi",
        usage=_usage(),
        attempts=("p1",),
    )
    with pytest.raises((AttributeError, TypeError)):
        r.content = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_provider_timeout_is_transient_subclass() -> None:
    assert issubclass(ProviderTimeoutError, TransientProviderError)


def test_transient_is_runtime_subclass() -> None:
    assert issubclass(TransientProviderError, RuntimeError)


def test_no_eligible_is_runtime_subclass() -> None:
    assert issubclass(NoEligibleProviderError, RuntimeError)


def test_all_failed_is_runtime_subclass() -> None:
    assert issubclass(AllProvidersFailedError, RuntimeError)


# ---------------------------------------------------------------------------
# LiteLLMRouter — fakes + helpers
# ---------------------------------------------------------------------------


def _provider(pid: str, *, endpoint: str | None = None) -> AIProvider:
    return AIProvider(
        id=pid,
        name=pid,
        provider="fake",
        endpoint=endpoint or f"fake/{pid}",
        capabilities=("reasoning",),
    )


class _RecordingTransport:
    """Test transport that scripts a per-provider response."""

    def __init__(self, plan: dict[str, Any], *, usage: LLMUsage | None = None) -> None:
        self._plan = plan
        self._usage = usage or LLMUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
        self.calls: list[tuple[str, LLMRequest]] = []

    def complete(self, provider: AIProvider, request: LLMRequest, /) -> tuple[str, LLMUsage]:
        self.calls.append((provider.id, request))
        action = self._plan.get(provider.id)
        if action is None:
            raise KeyError(f"no plan for provider {provider.id!r}")
        if isinstance(action, BaseException):
            raise action
        return str(action), self._usage


# ---------------------------------------------------------------------------
# LiteLLMRouter — happy path
# ---------------------------------------------------------------------------


def _request(timeout_s: float = 30.0) -> LLMRequest:
    return LLMRequest(
        task=TaskClass.INDIRA_REASONING,
        messages=(ChatMessage(role="user", content="ping"),),
        timeout_s=timeout_s,
    )


def test_router_returns_first_provider_response() -> None:
    providers = (_provider("p1"), _provider("p2"))
    transport = _RecordingTransport({"p1": "hello-from-p1", "p2": "n/a"})
    cost_calls: list[tuple[str, LLMUsage]] = []
    audits: list[tuple[str, str]] = []
    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=transport,
        fallback_audit=lambda p, reason: audits.append((p.id, reason)),
        cost_ledger=lambda p, u: cost_calls.append((p.id, u)),
    )
    resp = r.complete(_request(), ts_ns=1, request_id="req-1")

    assert resp.provider_id == "p1"
    assert resp.content == "hello-from-p1"
    assert resp.attempts == ("p1",)
    assert audits == []
    assert len(cost_calls) == 1
    assert cost_calls[0][0] == "p1"
    # Only the first provider was tried.
    assert [c[0] for c in transport.calls] == ["p1"]


def test_router_records_cost_only_on_success() -> None:
    providers = (
        _provider("p1"),
        _provider("p2"),
        _provider("p3"),
    )
    transport = _RecordingTransport(
        {
            "p1": TransientProviderError("503 overloaded"),
            "p2": ProviderTimeoutError("timeout"),
            "p3": "finally",
        }
    )
    cost_calls: list[tuple[str, LLMUsage]] = []
    audits: list[tuple[str, str]] = []
    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=transport,
        fallback_audit=lambda p, reason: audits.append((p.id, reason)),
        cost_ledger=lambda p, u: cost_calls.append((p.id, u)),
    )
    resp = r.complete(_request(), ts_ns=1, request_id="req-1")

    # All three tried in resolver order; cost recorded once for p3 only.
    assert resp.attempts == ("p1", "p2", "p3")
    assert resp.provider_id == "p3"
    assert [c[0] for c in cost_calls] == ["p3"]
    # Two fallback audits (p1 and p2).
    assert [a[0] for a in audits] == ["p1", "p2"]


def test_router_audit_reason_includes_exception_type() -> None:
    providers = (_provider("p1"), _provider("p2"))
    transport = _RecordingTransport(
        {
            "p1": ProviderTimeoutError("upstream slept"),
            "p2": "ok",
        }
    )
    audits: list[tuple[str, str]] = []
    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=transport,
        fallback_audit=lambda p, reason: audits.append((p.id, reason)),
    )
    r.complete(_request(), ts_ns=1, request_id="req-1")
    assert audits == [("p1", "ProviderTimeoutError: upstream slept")]


def test_router_attempts_list_is_in_resolver_order() -> None:
    providers = (
        _provider("a"),
        _provider("b"),
        _provider("c"),
    )
    transport = _RecordingTransport(
        {
            "a": TransientProviderError("a-fail"),
            "b": TransientProviderError("b-fail"),
            "c": "won",
        }
    )
    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=transport,
    )
    resp = r.complete(_request(), ts_ns=1, request_id="req-1")
    assert resp.attempts == ("a", "b", "c")


# ---------------------------------------------------------------------------
# LiteLLMRouter — failure modes
# ---------------------------------------------------------------------------


def test_router_raises_no_eligible_on_empty_resolver() -> None:
    transport = _RecordingTransport({})
    r = LiteLLMRouter(
        provider_resolver=lambda: (),
        transport=transport,
    )
    with pytest.raises(NoEligibleProviderError):
        r.complete(_request(), ts_ns=1, request_id="req-1")


def test_router_raises_all_failed_when_every_provider_transient() -> None:
    providers = (_provider("p1"), _provider("p2"))
    last_err = TransientProviderError("p2 down")
    transport = _RecordingTransport(
        {
            "p1": TransientProviderError("p1 down"),
            "p2": last_err,
        }
    )
    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=transport,
    )
    with pytest.raises(AllProvidersFailedError) as ei:
        r.complete(_request(), ts_ns=1, request_id="req-1")
    # __cause__ wraps the last transient error.
    assert ei.value.__cause__ is last_err


def test_router_propagates_non_transient_exception() -> None:
    """Auth / 400 / unknown errors must propagate without rotating —
    the chain stops on deterministic failures."""

    providers = (_provider("p1"), _provider("p2"))

    class _AuthError(Exception):
        pass

    transport = _RecordingTransport(
        {
            "p1": _AuthError("401 unauthenticated"),
            "p2": "should not run",
        }
    )
    audits: list[Any] = []
    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=transport,
        fallback_audit=lambda p, reason: audits.append((p.id, reason)),
    )
    with pytest.raises(_AuthError):
        r.complete(_request(), ts_ns=1, request_id="req-1")
    # No fallback audit, no second provider call.
    assert audits == []
    assert [c[0] for c in transport.calls] == ["p1"]


def test_router_rejects_non_tuple_resolver_return() -> None:
    transport = _RecordingTransport({"p1": "ok"})
    r = LiteLLMRouter(
        provider_resolver=lambda: [_provider("p1")],  # type: ignore[return-value,arg-type]
        transport=transport,
    )
    with pytest.raises(TypeError, match="tuple"):
        r.complete(_request(), ts_ns=1, request_id="req-1")


def test_router_rejects_non_aiprovider_in_tuple() -> None:
    transport = _RecordingTransport({})

    def _bad_resolver() -> tuple[Any, ...]:
        return ("not-a-provider",)

    r = LiteLLMRouter(
        provider_resolver=_bad_resolver,  # type: ignore[arg-type]
        transport=transport,
    )
    with pytest.raises(TypeError, match="AIProvider"):
        r.complete(_request(), ts_ns=1, request_id="req-1")


def test_router_rejects_transport_returning_non_str_content() -> None:
    providers = (_provider("p1"),)

    class _BadTransport:
        def complete(self, provider: AIProvider, request: LLMRequest, /) -> tuple[str, LLMUsage]:
            return (123, _usage())  # type: ignore[return-value]

    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=_BadTransport(),
    )
    with pytest.raises(TypeError, match="content"):
        r.complete(_request(), ts_ns=1, request_id="req-1")


def test_router_rejects_transport_returning_non_usage() -> None:
    providers = (_provider("p1"),)

    class _BadTransport:
        def complete(self, provider: AIProvider, request: LLMRequest, /) -> tuple[str, LLMUsage]:
            return ("ok", "not-usage")  # type: ignore[return-value]

    r = LiteLLMRouter(
        provider_resolver=lambda: providers,
        transport=_BadTransport(),
    )
    with pytest.raises(TypeError, match="usage"):
        r.complete(_request(), ts_ns=1, request_id="req-1")


# ---------------------------------------------------------------------------
# LiteLLMRouter — argument validation
# ---------------------------------------------------------------------------


def test_router_rejects_non_request_arg() -> None:
    r = LiteLLMRouter(
        provider_resolver=lambda: (_provider("p1"),),
        transport=_RecordingTransport({"p1": "ok"}),
    )
    with pytest.raises(TypeError, match="LLMRequest"):
        r.complete("hi", ts_ns=1, request_id="r")  # type: ignore[arg-type]


def test_router_rejects_non_int_ts_ns() -> None:
    r = LiteLLMRouter(
        provider_resolver=lambda: (_provider("p1"),),
        transport=_RecordingTransport({"p1": "ok"}),
    )
    with pytest.raises(TypeError, match="ts_ns"):
        r.complete(_request(), ts_ns=1.0, request_id="r")  # type: ignore[arg-type]


def test_router_rejects_non_positive_ts_ns() -> None:
    r = LiteLLMRouter(
        provider_resolver=lambda: (_provider("p1"),),
        transport=_RecordingTransport({"p1": "ok"}),
    )
    with pytest.raises(ValueError, match="ts_ns"):
        r.complete(_request(), ts_ns=0, request_id="r")


def test_router_rejects_empty_request_id() -> None:
    r = LiteLLMRouter(
        provider_resolver=lambda: (_provider("p1"),),
        transport=_RecordingTransport({"p1": "ok"}),
    )
    with pytest.raises(ValueError, match="request_id"):
        r.complete(_request(), ts_ns=1, request_id="")


def test_router_constructor_rejects_non_callable_resolver() -> None:
    with pytest.raises(TypeError, match="provider_resolver"):
        LiteLLMRouter(
            provider_resolver="not-callable",  # type: ignore[arg-type]
            transport=_RecordingTransport({}),
        )


def test_router_constructor_rejects_non_protocol_transport() -> None:
    with pytest.raises(TypeError, match="LLMTransport"):
        LiteLLMRouter(
            provider_resolver=lambda: (),
            transport="not-a-transport",  # type: ignore[arg-type]
        )


def test_router_constructor_rejects_non_callable_audit() -> None:
    with pytest.raises(TypeError, match="fallback_audit"):
        LiteLLMRouter(
            provider_resolver=lambda: (),
            transport=_RecordingTransport({}),
            fallback_audit="x",  # type: ignore[arg-type]
        )


def test_router_constructor_rejects_non_callable_cost() -> None:
    with pytest.raises(TypeError, match="cost_ledger"):
        LiteLLMRouter(
            provider_resolver=lambda: (),
            transport=_RecordingTransport({}),
            cost_ledger="x",  # type: ignore[arg-type]
        )


def test_router_is_frozen() -> None:
    r = LiteLLMRouter(
        provider_resolver=lambda: (),
        transport=_RecordingTransport({}),
    )
    with pytest.raises((AttributeError, TypeError)):
        r.transport = _RecordingTransport({})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_recording_transport_is_llm_transport() -> None:
    assert isinstance(_RecordingTransport({}), LLMTransport)


# ---------------------------------------------------------------------------
# INV-15 replay determinism
# ---------------------------------------------------------------------------


def test_router_replay_byte_identical_across_three_runs() -> None:
    """Same provider list + same plan + same ts_ns + same request_id
    produces byte-identical LLMResponse instances across replays."""

    providers = (_provider("p1"), _provider("p2"), _provider("p3"))

    def _run() -> LLMResponse:
        transport = _RecordingTransport(
            {
                "p1": TransientProviderError("flake"),
                "p2": TransientProviderError("flake"),
                "p3": "deterministic-content",
            }
        )
        r = LiteLLMRouter(
            provider_resolver=lambda: providers,
            transport=transport,
        )
        return r.complete(_request(), ts_ns=42, request_id="req-fixed")

    a, b, c = _run(), _run(), _run()
    assert a == b == c


def test_router_replay_byte_identical_with_audit_and_cost() -> None:
    providers = (_provider("p1"), _provider("p2"))

    def _run() -> tuple[LLMResponse, list[tuple[str, str]], list[str]]:
        transport = _RecordingTransport(
            {
                "p1": TransientProviderError("flake"),
                "p2": "ok",
            }
        )
        audits: list[tuple[str, str]] = []
        cost: list[str] = []
        r = LiteLLMRouter(
            provider_resolver=lambda: providers,
            transport=transport,
            fallback_audit=lambda p, reason: audits.append((p.id, reason)),
            cost_ledger=lambda p, _u: cost.append(p.id),
        )
        resp = r.complete(_request(), ts_ns=7, request_id="rid")
        return resp, audits, cost

    a, b, c = _run(), _run(), _run()
    assert a == b == c


# ---------------------------------------------------------------------------
# litellm_completion_transport — lazy import contract
# ---------------------------------------------------------------------------


def test_litellm_completion_transport_callable_exists() -> None:
    """The factory must exist as a top-level attribute regardless of
    whether litellm is installed."""

    assert callable(litellm_completion_transport)


def test_litellm_completion_transport_lazy_imports() -> None:
    """Calling the factory triggers the litellm import; the attempt
    must either succeed (litellm installed) or raise ImportError —
    never any other exception type."""

    try:
        t = litellm_completion_transport()
    except ImportError:
        pytest.skip("litellm not installed; lazy-import path validated")
    else:
        # If we got here, litellm is installed; just sanity-check the
        # returned object implements the protocol.
        assert isinstance(t, LLMTransport)
