"""Unit tests for the Wave-03 PR-6 HTTP chat transports.

Mirrors :mod:`tests.test_credentials_verifiers` — every test patches
the module-level :func:`http_chat_transport._open` so no real
outbound HTTP is ever made. Coverage targets every outcome class
(200 OK, 4xx auth, 429, 5xx, timeout, network error, malformed
JSON, missing required field, missing API key) for each of the
three concrete transports plus the dispatcher.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.cognitive_router import AIProvider
from intelligence_engine.cognitive.chat import http_chat_transport
from intelligence_engine.cognitive.chat.http_chat_transport import (
    CognitionDevinChatTransport,
    GoogleGeminiChatTransport,
    OpenAICompatChatTransport,
    RegistryDispatchChatTransport,
    build_default_dispatch_transport,
)
from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    TransientProviderError,
)

# --------------------------------------------------------------------
# Fakes / helpers
# --------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``http.client.HTTPResponse``."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = BytesIO(body)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)


def _patch_open(monkeypatch, behaviour) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_open(request, timeout):
        body = request.data.decode("utf-8") if request.data else ""
        calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "timeout": timeout,
                "body": body,
            }
        )
        return behaviour(request, timeout)

    monkeypatch.setattr(http_chat_transport, "_open", fake_open)
    return calls


def _ok(payload: dict[str, Any]):
    return lambda r, t: _FakeResponse(200, json.dumps(payload).encode("utf-8"))


def _http_error(status: int, msg: str = "err"):
    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, status, msg, {}, None)

    return behaviour


def _provider(provider_id: str, provider: str) -> AIProvider:
    return AIProvider(
        id=provider_id,
        name=provider_id,
        provider=provider,
        endpoint=f"https://example.invalid/{provider}",
        capabilities=("reasoning",),
    )


# --------------------------------------------------------------------
# OpenAICompatChatTransport — happy path + every outcome class
# --------------------------------------------------------------------


def _openai_ok_body(reply: str = "hi back") -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": reply}}]}


def test_openai_compat_happy_path_returns_reply(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_openai_ok_body("hi back")))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk-fake"})
    reply = transport.invoke(
        _provider("SRC-AI-OPENAI-001", "openai"),
        [HumanMessage(content="hi")],
    )
    assert reply == "hi back"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-fake"
    body = json.loads(call["body"])
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_compat_routes_xai_to_xai_base_url(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_openai_ok_body("ok")))
    transport = OpenAICompatChatTransport(env={"XAI_API_KEY": "xai-fake"})
    transport.invoke(
        _provider("SRC-AI-GROK-001", "xai"),
        [HumanMessage(content="hi")],
    )
    assert calls[0]["url"] == "https://api.x.ai/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer xai-fake"


def test_openai_compat_routes_deepseek_to_deepseek_base_url(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_openai_ok_body("ok")))
    transport = OpenAICompatChatTransport(env={"DEEPSEEK_API_KEY": "ds-fake"})
    transport.invoke(
        _provider("SRC-AI-DEEPSEEK-001", "deepseek"),
        [HumanMessage(content="hi")],
    )
    assert calls[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer ds-fake"


def test_openai_compat_passes_stop_temperature_max_tokens(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_openai_ok_body("ok")))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    transport.invoke(
        _provider("SRC-AI-OPENAI-001", "openai"),
        [HumanMessage(content="hi")],
        model="gpt-4o",
        stop=["</end>"],
        temperature=0.2,
        max_tokens=64,
    )
    body = json.loads(calls[0]["body"])
    assert body["model"] == "gpt-4o"
    assert body["stop"] == ["</end>"]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64


def test_openai_compat_maps_role_for_system_and_assistant(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_openai_ok_body("ok")))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    transport.invoke(
        _provider("SRC-AI-OPENAI-001", "openai"),
        [
            SystemMessage(content="you are X"),
            HumanMessage(content="hi"),
            AIMessage(content="prior reply"),
            HumanMessage(content="hi again"),
        ],
    )
    msgs = json.loads(calls[0]["body"])["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_openai_compat_missing_api_key_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_openai_ok_body("never reached")))
    transport = OpenAICompatChatTransport(env={})
    with pytest.raises(TransientProviderError, match="OPENAI_API_KEY"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_401_raises_runtime_error_not_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(401, "Unauthorized"))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk-bad"})
    with pytest.raises(RuntimeError, match="non-transient HTTP 401") as exc_info:
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )
    assert not isinstance(exc_info.value, TransientProviderError)


def test_openai_compat_429_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(429, "Too Many Requests"))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(TransientProviderError, match="HTTP 429"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_503_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(503, "Service Unavailable"))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(TransientProviderError, match="HTTP 503"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_timeout_raises_transient(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.URLError(TimeoutError("timed out"))

    _patch_open(monkeypatch, behaviour)
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(TransientProviderError, match="timed out"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_network_error_raises_transient(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.URLError(ConnectionRefusedError())

    _patch_open(monkeypatch, behaviour)
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(TransientProviderError, match="network error"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_malformed_json_raises_runtime_error(monkeypatch) -> None:
    _patch_open(
        monkeypatch,
        lambda r, t: _FakeResponse(200, b"this is not JSON {"),
    )
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(RuntimeError, match="malformed JSON"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_missing_choices_raises_runtime_error(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok({"unexpected": "shape"}))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(RuntimeError, match="missing 'choices'"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_unknown_provider_raises_runtime_error(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_openai_ok_body("never")))
    transport = OpenAICompatChatTransport(env={"OPENAI_API_KEY": "sk"})
    with pytest.raises(RuntimeError, match="not in the OpenAI-compat profile"):
        transport.invoke(
            _provider("SRC-AI-WHO-001", "who"),
            [HumanMessage(content="hi")],
        )


def test_openai_compat_error_messages_do_not_leak_api_key(monkeypatch) -> None:
    """Mirrors the verifiers contract — no key value or prefix in errors."""

    _patch_open(monkeypatch, _http_error(401))
    transport = OpenAICompatChatTransport(
        env={"OPENAI_API_KEY": "sk-supersecret-1234567890"}
    )
    with pytest.raises(RuntimeError) as exc_info:
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )
    assert "sk-supersecret" not in str(exc_info.value)
    assert "1234567890" not in str(exc_info.value)


# --------------------------------------------------------------------
# GoogleGeminiChatTransport
# --------------------------------------------------------------------


def _gemini_ok_body(reply: str = "gemini reply") -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"parts": [{"text": reply}], "role": "model"}}
        ]
    }


def test_gemini_happy_path_returns_reply(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_gemini_ok_body("ok google")))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "ggk"})
    reply = transport.invoke(
        _provider("SRC-AI-GEMINI-001", "google"),
        [HumanMessage(content="hi")],
    )
    assert reply == "ok google"
    assert "?key=ggk" in calls[0]["url"]
    assert "models/gemini-1.5-flash:generateContent" in calls[0]["url"]


def test_gemini_concatenates_system_prompts(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_gemini_ok_body("ok")))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "k"})
    transport.invoke(
        _provider("SRC-AI-GEMINI-001", "google"),
        [
            SystemMessage(content="be terse"),
            SystemMessage(content="be helpful"),
            HumanMessage(content="hi"),
        ],
    )
    body = json.loads(calls[0]["body"])
    assert body["systemInstruction"]["parts"][0]["text"] == "be terse\nbe helpful"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_gemini_maps_assistant_to_model_role(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_gemini_ok_body("ok")))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "k"})
    transport.invoke(
        _provider("SRC-AI-GEMINI-001", "google"),
        [
            HumanMessage(content="hi"),
            AIMessage(content="prior"),
            HumanMessage(content="next"),
        ],
    )
    body = json.loads(calls[0]["body"])
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]


def test_gemini_url_encodes_api_key(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, _ok(_gemini_ok_body("ok")))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "key with space&"})
    transport.invoke(
        _provider("SRC-AI-GEMINI-001", "google"),
        [HumanMessage(content="hi")],
    )
    assert "?key=key%20with%20space%26" in calls[0]["url"]


def test_gemini_refuses_non_google_provider(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_gemini_ok_body("never")))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="refusing to dispatch"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_gemini_missing_api_key_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_gemini_ok_body("never")))
    transport = GoogleGeminiChatTransport(env={})
    with pytest.raises(TransientProviderError, match="GEMINI_API_KEY"):
        transport.invoke(
            _provider("SRC-AI-GEMINI-001", "google"),
            [HumanMessage(content="hi")],
        )


def test_gemini_429_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(429))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "k"})
    with pytest.raises(TransientProviderError, match="HTTP 429"):
        transport.invoke(
            _provider("SRC-AI-GEMINI-001", "google"),
            [HumanMessage(content="hi")],
        )


def test_gemini_403_raises_runtime_error_not_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(403))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="non-transient HTTP 403") as exc_info:
        transport.invoke(
            _provider("SRC-AI-GEMINI-001", "google"),
            [HumanMessage(content="hi")],
        )
    assert not isinstance(exc_info.value, TransientProviderError)


def test_gemini_missing_candidates_raises_runtime_error(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok({"unexpected": "shape"}))
    transport = GoogleGeminiChatTransport(env={"GEMINI_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="missing 'candidates'"):
        transport.invoke(
            _provider("SRC-AI-GEMINI-001", "google"),
            [HumanMessage(content="hi")],
        )


# --------------------------------------------------------------------
# CognitionDevinChatTransport
# --------------------------------------------------------------------


def _devin_ok_body(session_id: str = "sess-123", url: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"session_id": session_id}
    if url:
        out["url"] = url
    return out


def test_devin_happy_path_returns_session_acknowledgement(monkeypatch) -> None:
    calls = _patch_open(
        monkeypatch,
        _ok(_devin_ok_body("sess-abc", "https://app.devin.ai/sessions/sess-abc")),
    )
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "dvk"})
    reply = transport.invoke(
        _provider("SRC-AI-DEVIN-001", "cognition"),
        [HumanMessage(content="please refactor X")],
    )
    assert "Devin session sess-abc started" in reply
    assert "https://app.devin.ai/sessions/sess-abc" in reply
    assert calls[0]["url"] == "https://api.devin.ai/v1/sessions"
    assert calls[0]["headers"]["Authorization"] == "Bearer dvk"
    body = json.loads(calls[0]["body"])
    assert body["prompt"] == "please refactor X"
    assert body["idempotent"] is True


def test_devin_omits_session_url_when_response_lacks_one(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_devin_ok_body("sess-xyz", url="")))
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "dvk"})
    reply = transport.invoke(
        _provider("SRC-AI-DEVIN-001", "cognition"),
        [HumanMessage(content="hi")],
    )
    assert "Devin session sess-xyz started" in reply


def test_devin_uses_id_when_session_id_field_is_missing(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok({"id": "alt-id"}))
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "dvk"})
    reply = transport.invoke(
        _provider("SRC-AI-DEVIN-001", "cognition"),
        [HumanMessage(content="hi")],
    )
    assert "Devin session alt-id started" in reply


def test_devin_refuses_non_cognition_provider(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_devin_ok_body()))
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="refusing to dispatch"):
        transport.invoke(
            _provider("SRC-AI-OPENAI-001", "openai"),
            [HumanMessage(content="hi")],
        )


def test_devin_missing_api_key_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _ok(_devin_ok_body()))
    transport = CognitionDevinChatTransport(env={})
    with pytest.raises(TransientProviderError, match="DEVIN_API_KEY"):
        transport.invoke(
            _provider("SRC-AI-DEVIN-001", "cognition"),
            [HumanMessage(content="hi")],
        )


def test_devin_429_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(429))
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "k"})
    with pytest.raises(TransientProviderError, match="HTTP 429"):
        transport.invoke(
            _provider("SRC-AI-DEVIN-001", "cognition"),
            [HumanMessage(content="hi")],
        )


def test_devin_500_raises_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(500))
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "k"})
    with pytest.raises(TransientProviderError, match="HTTP 500"):
        transport.invoke(
            _provider("SRC-AI-DEVIN-001", "cognition"),
            [HumanMessage(content="hi")],
        )


def test_devin_401_raises_runtime_error_not_transient(monkeypatch) -> None:
    _patch_open(monkeypatch, _http_error(401))
    transport = CognitionDevinChatTransport(env={"DEVIN_API_KEY": "bad"})
    with pytest.raises(RuntimeError, match="non-transient HTTP 401") as exc_info:
        transport.invoke(
            _provider("SRC-AI-DEVIN-001", "cognition"),
            [HumanMessage(content="hi")],
        )
    assert not isinstance(exc_info.value, TransientProviderError)


# --------------------------------------------------------------------
# RegistryDispatchChatTransport
# --------------------------------------------------------------------


class _RecordingBackend:
    """Fake backend that returns a deterministic reply per provider."""

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.calls: list[tuple[AIProvider, int]] = []

    def invoke(self, provider, messages, /, **kwargs):
        self.calls.append((provider, len(messages)))
        return self._reply


def test_dispatch_routes_to_matching_backend() -> None:
    a, b = _RecordingBackend("a"), _RecordingBackend("b")
    dispatcher = RegistryDispatchChatTransport({"openai": a, "google": b})
    out = dispatcher.invoke(
        _provider("SRC-AI-GEMINI-001", "google"),
        [HumanMessage(content="hi")],
    )
    assert out == "b"
    assert len(a.calls) == 0
    assert len(b.calls) == 1


def test_dispatch_unknown_provider_raises_runtime_error() -> None:
    dispatcher = RegistryDispatchChatTransport(
        {"openai": _RecordingBackend()},
    )
    with pytest.raises(RuntimeError, match="no backend wired"):
        dispatcher.invoke(
            _provider("SRC-AI-MYSTERY-001", "mystery"),
            [HumanMessage(content="hi")],
        )


def test_build_default_dispatch_wires_every_registry_ai_provider() -> None:
    """The dispatch table covers every provider field used in the registry.

    We can't introspect ``_table`` without poking private attrs, so
    instead we round-trip every known provider field through the
    dispatcher with an empty ``env``. Each call should fail with
    :class:`TransientProviderError` ("<X>_API_KEY is not set"),
    which only happens *inside* a real backend — proving the
    dispatcher routed, since "no backend wired" would surface as a
    non-transient :class:`RuntimeError`.
    """

    for provider_field in ("openai", "xai", "deepseek", "google", "cognition"):
        dispatcher = build_default_dispatch_transport(env={})
        with pytest.raises(TransientProviderError):
            dispatcher.invoke(
                _provider(f"SRC-AI-{provider_field.upper()}-001", provider_field),
                [HumanMessage(content="hi")],
            )


# --------------------------------------------------------------------
# B1 + B24 isolation — module-level import audit
# --------------------------------------------------------------------


def test_http_chat_transport_does_not_import_governance_or_system_engine() -> None:
    """B1 + B24: cognitive layer never reaches into governance / system."""

    import ast
    import importlib

    module = importlib.import_module(
        "intelligence_engine.cognitive.chat.http_chat_transport"
    )
    source = open(module.__file__).read()
    tree = ast.parse(source)
    forbidden_prefixes = ("governance_engine", "system_engine")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert not name.name.startswith(forbidden_prefixes), (
                    f"forbidden import: {name.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            assert not node.module.startswith(forbidden_prefixes), (
                f"forbidden import: from {node.module}"
            )
