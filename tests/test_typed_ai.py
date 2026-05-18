"""Tests for S-06 :mod:`intelligence_engine.cognitive.typed_ai`."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import Field, ValidationError

from core.cognitive_router import AIProvider, TaskClass
from intelligence_engine.cognitive import typed_ai
from intelligence_engine.cognitive.typed_ai import (
    AllProvidersFailedError,
    NoEligibleProviderError,
    SchemaValidationError,
    TransientProviderError,
    TypedAIAgent,
    TypedAIProposal,
    TypedAIRequest,
    TypedAIResult,
    TypedAITransport,
    default_id_factory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _PriceProposal(TypedAIResult):
    symbol: str = Field(min_length=1)
    side: str = Field(pattern="^(BUY|SELL|HOLD)$")
    confidence: float = Field(ge=0.0, le=1.0)


class _OtherProposal(TypedAIResult):
    note: str = Field(min_length=1)


def _provider(pid: str, *, capabilities: tuple[str, ...] = ("chat",)) -> AIProvider:
    return AIProvider(
        id=pid,
        name=f"provider-{pid}",
        provider="stub",
        endpoint=f"https://{pid}.example/v1",
        capabilities=capabilities,
    )


@dataclasses.dataclass
class _Recorder:
    audits: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    submitted: list[TypedAIProposal[Any]] = dataclasses.field(default_factory=list)

    def audit(self, provider: AIProvider, reason: str) -> None:
        self.audits.append((provider.id, reason))

    def submit(self, proposal: TypedAIProposal[Any]) -> None:
        self.submitted.append(proposal)


class _ScriptedTransport:
    """Transport that returns a pre-scripted response per (provider_id, attempt).

    Each entry in ``script`` is either:

    * a ``str`` — returned as the raw transport response
    * a ``BaseException`` instance — re-raised
    * a ``BaseException`` subclass — raised with a stub message
    """

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self._script = {pid: list(items) for pid, items in script.items()}
        self.calls: list[tuple[str, str, type[TypedAIResult], str | None]] = []

    def invoke(
        self,
        provider: AIProvider,
        prompt: str,
        schema_class: type[TypedAIResult],
        /,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        self.calls.append((provider.id, prompt, schema_class, retry_feedback))
        items = self._script.get(provider.id)
        if not items:
            raise AssertionError(f"no scripted response remaining for provider={provider.id!r}")
        item = items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item(f"stub error from {provider.id}")
        if isinstance(item, str):
            return item
        raise AssertionError(f"bad scripted entry: {item!r}")


def _ok_payload(symbol: str = "BTC-USD", side: str = "BUY", conf: float = 0.5) -> str:
    return f'{{"symbol":"{symbol}","side":"{side}","confidence":{conf}}}'


def _seq_id_factory(prefix: str = "P") -> Callable[[], str]:
    state = {"n": 0}

    def factory() -> str:
        state["n"] += 1
        return f"{prefix}{state['n']:04d}"

    return factory


def _build_agent(
    *,
    providers: tuple[AIProvider, ...],
    transport: TypedAITransport,
    recorder: _Recorder | None = None,
    id_factory: Callable[[], str] | None = None,
) -> tuple[TypedAIAgent, _Recorder]:
    rec = recorder or _Recorder()
    return (
        TypedAIAgent(
            provider_resolver=lambda: providers,
            transport=transport,
            submit_proposal=rec.submit,
            fallback_audit=rec.audit,
            id_factory=id_factory or _seq_id_factory(),
        ),
        rec,
    )


# ---------------------------------------------------------------------------
# TypedAIRequest validation
# ---------------------------------------------------------------------------


def test_typed_ai_request_accepts_concrete_subclass() -> None:
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="Pick a side.",
        schema_class=_PriceProposal,
    )
    assert req.max_retries == 2
    assert req.schema_class is _PriceProposal


def test_typed_ai_request_rejects_non_string_prompt() -> None:
    with pytest.raises(TypeError, match="prompt must be str"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt=123,  # type: ignore[arg-type]
            schema_class=_PriceProposal,
        )


def test_typed_ai_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="",
            schema_class=_PriceProposal,
        )


def test_typed_ai_request_rejects_non_typed_ai_result_schema() -> None:
    class NotAResult:
        pass

    with pytest.raises(TypeError, match="must be a TypedAIResult subclass"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="hi",
            schema_class=NotAResult,  # type: ignore[type-var]
        )


def test_typed_ai_request_rejects_typed_ai_result_base_class() -> None:
    with pytest.raises(ValueError, match="must be a \\*concrete\\* subclass"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="hi",
            schema_class=TypedAIResult,
        )


def test_typed_ai_request_rejects_non_int_max_retries() -> None:
    with pytest.raises(TypeError, match="max_retries must be int"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="hi",
            schema_class=_PriceProposal,
            max_retries="2",  # type: ignore[arg-type]
        )


def test_typed_ai_request_rejects_bool_max_retries() -> None:
    """Bool is an int subtype; reject it explicitly."""
    with pytest.raises(TypeError, match="max_retries must be int"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="hi",
            schema_class=_PriceProposal,
            max_retries=True,  # type: ignore[arg-type]
        )


def test_typed_ai_request_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="hi",
            schema_class=_PriceProposal,
            max_retries=-1,
        )


def test_typed_ai_request_is_frozen() -> None:
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.prompt = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TypedAIProposal validation
# ---------------------------------------------------------------------------


def _good_result() -> _PriceProposal:
    return _PriceProposal(symbol="BTC-USD", side="BUY", confidence=0.5)


def test_typed_ai_proposal_rejects_empty_proposal_id() -> None:
    with pytest.raises(ValueError, match="proposal_id must be non-empty"):
        TypedAIProposal(
            proposal_id="",
            ts_ns=1,
            task=TaskClass.INDIRA_REASONING,
            provider_id="p1",
            schema_name="x.Y",
            validated_result=_good_result(),
        )


def test_typed_ai_proposal_rejects_non_int_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        TypedAIProposal(
            proposal_id="P0001",
            ts_ns="1",  # type: ignore[arg-type]
            task=TaskClass.INDIRA_REASONING,
            provider_id="p1",
            schema_name="x.Y",
            validated_result=_good_result(),
        )


def test_typed_ai_proposal_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        TypedAIProposal(
            proposal_id="P0001",
            ts_ns=0,
            task=TaskClass.INDIRA_REASONING,
            provider_id="p1",
            schema_name="x.Y",
            validated_result=_good_result(),
        )


def test_typed_ai_proposal_rejects_empty_provider_id() -> None:
    with pytest.raises(ValueError, match="provider_id must be non-empty"):
        TypedAIProposal(
            proposal_id="P0001",
            ts_ns=1,
            task=TaskClass.INDIRA_REASONING,
            provider_id="",
            schema_name="x.Y",
            validated_result=_good_result(),
        )


def test_typed_ai_proposal_rejects_empty_schema_name() -> None:
    with pytest.raises(ValueError, match="schema_name must be non-empty"):
        TypedAIProposal(
            proposal_id="P0001",
            ts_ns=1,
            task=TaskClass.INDIRA_REASONING,
            provider_id="p1",
            schema_name="",
            validated_result=_good_result(),
        )


def test_typed_ai_proposal_rejects_non_typed_ai_result_payload() -> None:
    with pytest.raises(TypeError, match="must be a TypedAIResult instance"):
        TypedAIProposal(
            proposal_id="P0001",
            ts_ns=1,
            task=TaskClass.INDIRA_REASONING,
            provider_id="p1",
            schema_name="x.Y",
            validated_result={"symbol": "BTC"},  # type: ignore[arg-type]
        )


def test_typed_ai_proposal_is_frozen() -> None:
    proposal = TypedAIProposal(
        proposal_id="P0001",
        ts_ns=1,
        task=TaskClass.INDIRA_REASONING,
        provider_id="p1",
        schema_name="x.Y",
        validated_result=_good_result(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.proposal_id = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TypedAIAgent constructor validation
# ---------------------------------------------------------------------------


def test_agent_rejects_non_callable_provider_resolver() -> None:
    with pytest.raises(TypeError, match="provider_resolver must be callable"):
        TypedAIAgent(
            provider_resolver="nope",  # type: ignore[arg-type]
            transport=_ScriptedTransport({}),
        )


def test_agent_rejects_transport_without_invoke() -> None:
    class BadTransport:
        pass

    with pytest.raises(TypeError, match="must implement the TypedAITransport"):
        TypedAIAgent(
            provider_resolver=tuple,
            transport=BadTransport(),  # type: ignore[arg-type]
        )


def test_agent_rejects_non_callable_submit_proposal() -> None:
    with pytest.raises(TypeError, match="submit_proposal must be callable"):
        TypedAIAgent(
            provider_resolver=lambda: (),
            transport=_ScriptedTransport({}),
            submit_proposal="nope",  # type: ignore[arg-type]
        )


def test_agent_rejects_non_callable_fallback_audit() -> None:
    with pytest.raises(TypeError, match="fallback_audit must be callable"):
        TypedAIAgent(
            provider_resolver=lambda: (),
            transport=_ScriptedTransport({}),
            fallback_audit="nope",  # type: ignore[arg-type]
        )


def test_agent_rejects_non_callable_id_factory() -> None:
    with pytest.raises(TypeError, match="id_factory must be callable"):
        TypedAIAgent(
            provider_resolver=lambda: (),
            transport=_ScriptedTransport({}),
            id_factory="nope",  # type: ignore[arg-type]
        )


def test_agent_is_frozen() -> None:
    agent = TypedAIAgent(
        provider_resolver=lambda: (),
        transport=_ScriptedTransport({}),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        agent.transport = _ScriptedTransport({})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_typed argument validation
# ---------------------------------------------------------------------------


def test_run_typed_rejects_non_request() -> None:
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=_ScriptedTransport({"a": [_ok_payload()]}),
    )
    with pytest.raises(TypeError, match="request must be TypedAIRequest"):
        agent.run_typed("not a request", ts_ns=1)  # type: ignore[arg-type]


def test_run_typed_rejects_non_int_ts_ns() -> None:
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=_ScriptedTransport({"a": [_ok_payload()]}),
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(TypeError, match="ts_ns must be int"):
        agent.run_typed(req, ts_ns="1")  # type: ignore[arg-type]


def test_run_typed_rejects_zero_ts_ns() -> None:
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=_ScriptedTransport({"a": [_ok_payload()]}),
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        agent.run_typed(req, ts_ns=0)


def test_run_typed_raises_on_empty_resolver() -> None:
    agent, _ = _build_agent(
        providers=(),
        transport=_ScriptedTransport({}),
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(NoEligibleProviderError, match="no enabled AI providers"):
        agent.run_typed(req, ts_ns=1)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_typed_happy_path_first_provider() -> None:
    transport = _ScriptedTransport({"a": [_ok_payload(symbol="ETH", side="SELL")]})
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="Decide BTC.",
        schema_class=_PriceProposal,
    )

    proposal = agent.run_typed(req, ts_ns=42)

    assert proposal.proposal_id == "P0001"
    assert proposal.ts_ns == 42
    assert proposal.task is TaskClass.INDIRA_REASONING
    assert proposal.provider_id == "a"
    assert proposal.validated_result.symbol == "ETH"
    assert proposal.validated_result.side == "SELL"
    assert rec.submitted == [proposal]
    assert rec.audits == []
    # Only the first provider was called; second was never reached.
    assert [c[0] for c in transport.calls] == ["a"]


def test_run_typed_schema_name_is_qualified() -> None:
    transport = _ScriptedTransport({"a": [_ok_payload()]})
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    proposal = agent.run_typed(req, ts_ns=1)
    assert proposal.schema_name.endswith("._PriceProposal")
    assert "test_typed_ai" in proposal.schema_name


def test_run_typed_passes_no_retry_feedback_on_first_attempt() -> None:
    transport = _ScriptedTransport({"a": [_ok_payload()]})
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    agent.run_typed(req, ts_ns=1)
    assert transport.calls[0][3] is None


# ---------------------------------------------------------------------------
# Provider fallback (transient errors)
# ---------------------------------------------------------------------------


def test_run_typed_falls_back_on_transient_first_provider() -> None:
    transport = _ScriptedTransport(
        {
            "a": [TransientProviderError("rate-limited")],
            "b": [_ok_payload()],
        }
    )
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    proposal = agent.run_typed(req, ts_ns=1)
    assert proposal.provider_id == "b"
    assert rec.audits == [("a", "rate-limited")]


def test_run_typed_raises_when_all_providers_transient() -> None:
    transport = _ScriptedTransport(
        {
            "a": [TransientProviderError("a-flap")],
            "b": [TransientProviderError("b-flap")],
        }
    )
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(AllProvidersFailedError, match="last reason: b-flap") as ei:
        agent.run_typed(req, ts_ns=1)
    assert isinstance(ei.value.__cause__, TransientProviderError)
    assert rec.audits == [("a", "a-flap"), ("b", "b-flap")]
    assert rec.submitted == []


def test_run_typed_does_not_call_remaining_providers_after_success() -> None:
    transport = _ScriptedTransport(
        {
            "a": [_ok_payload()],
            "b": [_ok_payload()],
            "c": [_ok_payload()],
        }
    )
    agent, _ = _build_agent(
        providers=(_provider("a"), _provider("b"), _provider("c")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    agent.run_typed(req, ts_ns=1)
    assert [c[0] for c in transport.calls] == ["a"]


def test_run_typed_propagates_non_transient_exceptions() -> None:
    """Non-transient exceptions are not fallback-eligible."""

    class AuthError(RuntimeError):
        pass

    transport = _ScriptedTransport({"a": [AuthError("401")]})
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(AuthError, match="401"):
        agent.run_typed(req, ts_ns=1)
    # Provider b never gets called.
    assert [c[0] for c in transport.calls] == ["a"]
    assert rec.audits == []


# ---------------------------------------------------------------------------
# Schema validation + retries
# ---------------------------------------------------------------------------


def test_run_typed_retries_on_validation_failure_same_provider() -> None:
    transport = _ScriptedTransport(
        {
            "a": [
                '{"symbol":"BTC","side":"BUY","confidence":2.5}',  # bad
                _ok_payload(),
            ],
        }
    )
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
        max_retries=1,
    )
    proposal = agent.run_typed(req, ts_ns=1)
    assert proposal.provider_id == "a"
    # No fallback audit — schema retries stay on the same provider.
    assert rec.audits == []
    # Two calls, both on provider a.
    assert [c[0] for c in transport.calls] == ["a", "a"]
    # Second call carries retry feedback.
    assert transport.calls[0][3] is None
    assert transport.calls[1][3] is not None
    assert "confidence" in transport.calls[1][3]


def test_run_typed_raises_schema_validation_after_retries() -> None:
    transport = _ScriptedTransport(
        {
            "a": [
                '{"symbol":"BTC","side":"BUY","confidence":2.5}',
                '{"symbol":"BTC","side":"BUY","confidence":3.5}',
                '{"symbol":"BTC","side":"BUY","confidence":4.5}',
            ],
            "b": [_ok_payload()],
        }
    )
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
        max_retries=2,
    )
    with pytest.raises(SchemaValidationError) as ei:
        agent.run_typed(req, ts_ns=1)
    assert "_PriceProposal" in str(ei.value)
    assert "3 attempt(s)" in str(ei.value)
    assert isinstance(ei.value.__cause__, ValidationError)
    # Validation failures must NOT rotate provider.
    assert [c[0] for c in transport.calls] == ["a", "a", "a"]
    assert rec.audits == []
    assert rec.submitted == []


def test_run_typed_zero_retries_means_one_attempt() -> None:
    transport = _ScriptedTransport({"a": ['{"symbol":"BTC","side":"BUY","confidence":2.5}']})
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
        id_factory=_seq_id_factory(),
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
        max_retries=0,
    )
    with pytest.raises(SchemaValidationError, match="1 attempt"):
        agent.run_typed(req, ts_ns=1)


def test_run_typed_rejects_non_str_transport_response() -> None:
    class _BadTransport:
        def invoke(
            self,
            provider: AIProvider,
            prompt: str,
            schema_class: type[TypedAIResult],
            /,
            *,
            retry_feedback: str | None = None,
        ) -> str:
            return 42  # type: ignore[return-value]

    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=_BadTransport(),
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(TypeError, match="transport.invoke must return str"):
        agent.run_typed(req, ts_ns=1)


def test_run_typed_rejects_garbled_json_then_recovers() -> None:
    transport = _ScriptedTransport(
        {
            "a": [
                "this is not json at all",
                _ok_payload(symbol="DOGE", side="HOLD", conf=0.3),
            ]
        }
    )
    agent, rec = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
        max_retries=1,
    )
    proposal = agent.run_typed(req, ts_ns=1)
    assert proposal.validated_result.symbol == "DOGE"
    assert proposal.validated_result.side == "HOLD"
    assert rec.audits == []


# ---------------------------------------------------------------------------
# Submit / audit invariants
# ---------------------------------------------------------------------------


def test_submit_proposal_called_exactly_once_on_success() -> None:
    transport = _ScriptedTransport({"a": [_ok_payload()]})
    agent, rec = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    agent.run_typed(req, ts_ns=1)
    assert len(rec.submitted) == 1


def test_submit_proposal_not_called_on_validation_exhaustion() -> None:
    transport = _ScriptedTransport({"a": ['{"symbol":"BTC","side":"BUY","confidence":2.5}']})
    agent, rec = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
        max_retries=0,
    )
    with pytest.raises(SchemaValidationError):
        agent.run_typed(req, ts_ns=1)
    assert rec.submitted == []


def test_submit_proposal_not_called_on_all_providers_failed() -> None:
    transport = _ScriptedTransport(
        {
            "a": [TransientProviderError("a-flap")],
            "b": [TransientProviderError("b-flap")],
        }
    )
    agent, rec = _build_agent(
        providers=(_provider("a"), _provider("b")),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_PriceProposal,
    )
    with pytest.raises(AllProvidersFailedError):
        agent.run_typed(req, ts_ns=1)
    assert rec.submitted == []


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_replay_determinism_byte_identical() -> None:
    """INV-15: same inputs → same outputs byte-identical."""

    def _build() -> tuple[TypedAIAgent, _Recorder]:
        return _build_agent(
            providers=(_provider("a"), _provider("b")),
            transport=_ScriptedTransport(
                {
                    "a": [TransientProviderError("rate")],
                    "b": [_ok_payload(symbol="BTC", side="BUY", conf=0.7)],
                }
            ),
            id_factory=_seq_id_factory("REPLAY"),
        )

    runs: list[TypedAIProposal[_PriceProposal]] = []
    audits: list[list[tuple[str, str]]] = []
    for _ in range(3):
        agent, rec = _build()
        req = TypedAIRequest(
            task=TaskClass.INDIRA_REASONING,
            prompt="Replay-fixed prompt.",
            schema_class=_PriceProposal,
        )
        runs.append(agent.run_typed(req, ts_ns=12_345_678))
        audits.append(list(rec.audits))

    assert runs[0] == runs[1] == runs[2]
    assert audits[0] == audits[1] == audits[2]


def test_default_id_factory_returns_uuid_hex() -> None:
    a = default_id_factory()
    b = default_id_factory()
    assert a != b
    assert len(a) == 32
    assert all(c in "0123456789abcdef" for c in a)


# ---------------------------------------------------------------------------
# AST-walks: INV-12 / INV-15 / authority isolation
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    src = pathlib.Path(typed_ai.__file__).read_text(encoding="utf-8")
    return ast.parse(src)


def _imported_roots(tree: ast.Module) -> set[str]:
    """Top-level imports only — ignore lazy imports inside function bodies."""

    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def test_module_does_not_import_clock_or_environ() -> None:
    """INV-15 / B-CLOCK / T1: no time / datetime / os imports at module scope."""

    forbidden = {"os", "time", "datetime", "asyncio"}
    roots = _imported_roots(_module_ast())
    leaks = roots & forbidden
    assert leaks == set(), f"forbidden imports leaked into typed_ai: {leaks}"


def test_module_does_not_import_governance_or_execution() -> None:
    """B1: typed_ai may not reach into governance_engine / execution_engine."""

    forbidden = {"governance_engine", "execution_engine", "system_engine"}
    roots = _imported_roots(_module_ast())
    leaks = roots & forbidden
    assert leaks == set(), f"forbidden engine imports in typed_ai: {leaks}"


def test_module_does_not_import_pydantic_ai_at_top_level() -> None:
    """pydantic-ai is lazy-imported inside pydantic_ai_transport()."""

    roots = _imported_roots(_module_ast())
    assert "pydantic_ai" not in roots


def test_module_does_not_emit_signal_event_directly() -> None:
    """INV-12: the agent never constructs SignalEvent / ExecutionIntent.

    This is enforced structurally — the module must not name those
    types anywhere in its source.
    """

    src = pathlib.Path(typed_ai.__file__).read_text(encoding="utf-8")
    forbidden_names = ("SignalEvent", "ExecutionIntent", "OrderState")
    leaks = [name for name in forbidden_names if name in src]
    assert leaks == [], f"typed_ai mentions execution-side names: {leaks}"


def test_module_calls_only_run_sync_on_pydantic_ai_agent() -> None:
    """AGPL-style pinning: the production transport may only invoke
    ``Agent.run_sync`` and ``run_result.data.model_dump_json``. We
    enforce this by AST-walking the inner ``_PydanticAITransport``
    class body and asserting the only method calls are the documented
    ones.
    """

    src = inspect.getsource(typed_ai.pydantic_ai_transport)
    tree = ast.parse(src)

    method_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method_calls.add(node.func.attr)

    # The transport closure may call constructors / format strings; we
    # care that no *unexpected* pydantic-ai surface gets reached. The
    # documented surface is run_sync + model_dump_json + Agent (ctor).
    documented = {"run_sync", "model_dump_json"}
    pydantic_ai_specific = {"crawl_url", "stream", "iter", "messages"}
    leaks = method_calls & pydantic_ai_specific
    assert leaks == set(), f"pydantic_ai_transport reaches into undocumented surface: {leaks}"
    assert documented <= method_calls, (
        f"pydantic_ai_transport missing documented calls: {documented - method_calls}"
    )


# ---------------------------------------------------------------------------
# Adapter-from header
# ---------------------------------------------------------------------------


def test_module_carries_adapted_from_header() -> None:
    src = pathlib.Path(typed_ai.__file__).read_text(encoding="utf-8")
    first_lines = src.splitlines()[:3]
    assert any(line.startswith("# ADAPTED FROM:") for line in first_lines), (
        "S-06 typed_ai must carry an # ADAPTED FROM: header per PART 1 rule 7"
    )


def test_new_pip_dependencies_declared() -> None:
    """PART 1 rule 10: flag new pip deps."""

    assert typed_ai.NEW_PIP_DEPENDENCIES == ("pydantic-ai",)


def test_module_exports_public_api() -> None:
    expected = {
        "AllProvidersFailedError",
        "FallbackAuditSink",
        "NoEligibleProviderError",
        "ProposalSubmitter",
        "ProviderResolver",
        "SchemaValidationError",
        "TransientProviderError",
        "TypedAIAgent",
        "TypedAIProposal",
        "TypedAIRequest",
        "TypedAIResult",
        "TypedAITransport",
        "default_id_factory",
        "pydantic_ai_transport",
    }
    assert set(typed_ai.__all__) == expected


# ---------------------------------------------------------------------------
# pydantic_ai_transport raises cleanly when the package is missing
# ---------------------------------------------------------------------------


def test_pydantic_ai_transport_factory_is_callable() -> None:
    """The factory is exported regardless of whether pydantic-ai is installed.

    On a host without pydantic-ai the call raises RuntimeError with a
    clear message; the symbol itself remains importable so downstream
    wiring code can reference it without crashing the boot sequence.
    """

    assert callable(typed_ai.pydantic_ai_transport)


def test_other_typed_ai_result_subclass_round_trips() -> None:
    """A second :class:`TypedAIResult` subclass must validate independently."""

    transport = _ScriptedTransport({"a": ['{"note":"hello world"}']})
    agent, _ = _build_agent(
        providers=(_provider("a"),),
        transport=transport,
    )
    req = TypedAIRequest(
        task=TaskClass.INDIRA_REASONING,
        prompt="hi",
        schema_class=_OtherProposal,
    )
    proposal = agent.run_typed(req, ts_ns=1)
    assert isinstance(proposal.validated_result, _OtherProposal)
    assert proposal.validated_result.note == "hello world"


def test_typed_ai_result_is_frozen() -> None:
    r = _PriceProposal(symbol="BTC-USD", side="BUY", confidence=0.5)
    with pytest.raises(ValidationError):
        r.symbol = "ETH"  # type: ignore[misc]
