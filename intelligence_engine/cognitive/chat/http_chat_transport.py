"""HTTP chat transports (Wave-03 PR-6).

The chat layer's :class:`ChatTransport` protocol is the *only* place
provider-specific networking is allowed to live. PR-1 shipped the
adapter (:class:`RegistryDrivenChatModel`) plus a stub
:class:`NotConfiguredTransport`; this module is the production
implementation — three thin synchronous HTTP transports plus a
dispatcher that routes a turn to the right one based on the
``provider`` field of the resolved :class:`AIProvider`.

Design constraints (rooted in Wave-01 + Wave-03 PR-1):

* **Stdlib HTTP only.** ``urllib.request`` matches the credential
  ``verifiers`` module and keeps the runtime dependency count flat.
  Every blocking call goes through the module-level :func:`_open`
  shim so unit tests can monkey-patch network access without
  touching the real internet.
* **Errors map to the chat-model error hierarchy.** Anything the
  ``RegistryDrivenChatModel`` adapter is willing to fall back over
  (network errors, 429, 5xx, timeouts, missing key) is raised as
  :class:`TransientProviderError`. Anything that should propagate
  (auth, 4xx schema, malformed body) raises a generic
  :class:`RuntimeError` so the adapter does *not* silently fall
  through to another provider on a bug.
* **No secret values in error messages.** Mirrors the rule in
  ``system_engine.credentials.verifiers`` — nothing about the API
  key (length, prefix, presence in env) leaks into ``detail``
  strings beyond "key missing" / "key rejected".
* **Registry-driven routing.** The dispatcher table is keyed on
  ``provider.provider`` (e.g. ``"openai"``, ``"google"``,
  ``"cognition"``) so future registry rows that re-use a known
  provider field are picked up automatically. Adding a brand-new
  provider here is the only way to introduce vendor-specific code
  to the chat hot path.
* **B1 + B24 isolation preserved.** This module imports
  ``langchain_core`` types (allowed by B24 inside
  ``intelligence_engine.cognitive.chat.*``) and the
  registry-projection :class:`AIProvider` from
  ``core.cognitive_router`` (allowed by B1, since
  ``core.*`` is engine-neutral). It MUST NOT import
  ``governance_engine.*`` or ``system_engine.*`` directly.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core.cognitive_router import AIProvider
from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    ChatTransport,
    TransientProviderError,
)

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_RESPONSE_BYTES",
    "OpenAICompatChatTransport",
    "GoogleGeminiChatTransport",
    "CognitionDevinChatTransport",
    "RegistryDispatchChatTransport",
    "build_default_dispatch_transport",
]


# Conservative blocking HTTP timeout. Matches
# ``system_engine.credentials.verifiers.DEFAULT_TIMEOUT_S``. The chat
# graph is operator-initiated and never on the trade hot path, so we
# trade snappier failures for safer absolute caps.
DEFAULT_TIMEOUT_S: float = 30.0

# Hard cap on response body size. A misbehaving provider returning a
# 100MB blob would otherwise trip the request before the timeout. The
# limit comfortably absorbs even verbose multi-message responses.
MAX_RESPONSE_BYTES: int = 4 * 1024 * 1024  # 4 MiB


# --------------------------------------------------------------------
# Shared HTTP helpers
# --------------------------------------------------------------------


def _open(
    request: urllib.request.Request,
    timeout: float,
):
    """Indirection over :func:`urllib.request.urlopen` for tests.

    Tests monkey-patch this name so unit tests never make real
    outbound HTTP. Production code calls through unchanged.
    """

    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def _read_body(resp: Any) -> bytes:
    """Read at most :data:`MAX_RESPONSE_BYTES` from ``resp``."""

    return resp.read(MAX_RESPONSE_BYTES + 1)[:MAX_RESPONSE_BYTES]


def _is_transient_status(status: int) -> bool:
    """HTTP status codes that warrant fallback to the next provider."""

    return status == 408 or status == 429 or 500 <= status <= 599


def _execute(
    request: urllib.request.Request,
    *,
    timeout: float,
    provider_label: str,
) -> bytes:
    """Run ``request`` through :func:`_open` and classify failures.

    Returns the raw body on 2xx. Raises
    :class:`TransientProviderError` for the network-shaped failures
    the adapter is willing to fall back over, and a plain
    :class:`RuntimeError` for shapes that look like bugs (4xx auth /
    schema, oversized body, malformed framing).
    """

    try:
        with _open(request, timeout) as resp:  # type: ignore[no-untyped-call]
            status = int(getattr(resp, "status", None) or resp.getcode())
            body = _read_body(resp)
    except urllib.error.HTTPError as exc:
        if _is_transient_status(int(exc.code)):
            raise TransientProviderError(
                f"{provider_label}: HTTP {exc.code}"
            ) from exc
        raise RuntimeError(
            f"{provider_label}: non-transient HTTP {exc.code}"
        ) from exc
    except TimeoutError as exc:
        raise TransientProviderError(
            f"{provider_label}: timed out after {timeout:.1f}s"
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise TransientProviderError(
                f"{provider_label}: timed out after {timeout:.1f}s"
            ) from exc
        # ``type(exc.reason).__name__`` and not ``str(exc)`` —
        # avoid leaking proxy URLs / DNS hints / anything
        # environmental.
        reason = type(exc.reason).__name__ if exc.reason else "URLError"
        raise TransientProviderError(
            f"{provider_label}: network error ({reason})"
        ) from exc

    if status < 200 or status >= 300:
        if _is_transient_status(status):
            raise TransientProviderError(
                f"{provider_label}: HTTP {status}"
            )
        raise RuntimeError(
            f"{provider_label}: non-transient HTTP {status}"
        )
    return body


def _parse_json(body: bytes, *, provider_label: str) -> Mapping[str, Any]:
    """Decode ``body`` as JSON; misshapen → :class:`RuntimeError`."""

    try:
        decoded = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"{provider_label}: malformed JSON response"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeError(
            f"{provider_label}: response root is not a JSON object"
        )
    return decoded


def _read_required_env(provider_label: str, env_var: str) -> str:
    """Return ``os.environ[env_var]`` or :class:`TransientProviderError`.

    Treats a missing key as transient so the chat model falls
    through to the next eligible provider. Operators tend to add
    one provider at a time via ``/credentials``; raising
    non-transiently here would force them to fix every row before
    *any* turn could succeed.
    """

    raw = os.environ.get(env_var, "").strip()
    if not raw:
        raise TransientProviderError(
            f"{provider_label}: {env_var} is not set"
        )
    return raw


# --------------------------------------------------------------------
# OpenAI-compatible chat transport (openai / xai / deepseek)
# --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OpenAICompatProfile:
    """Per-provider knobs for the OpenAI-compatible POST shape.

    The four 2026 providers that speak this dialect (OpenAI, xAI,
    DeepSeek, plus a handful of self-hosted gateways) only differ in
    base URL, env var, and default model name. Centralising the
    profile here keeps the dispatch table declarative.
    """

    base_url: str
    env_var: str
    default_model: str


_OPENAI_COMPAT_PROFILES: Mapping[str, _OpenAICompatProfile] = {
    "openai": _OpenAICompatProfile(
        base_url="https://api.openai.com/v1",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    ),
    "xai": _OpenAICompatProfile(
        base_url="https://api.x.ai/v1",
        env_var="XAI_API_KEY",
        default_model="grok-2-latest",
    ),
    "deepseek": _OpenAICompatProfile(
        base_url="https://api.deepseek.com/v1",
        env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
}


def _role_for(message: BaseMessage) -> str:
    """Map LangChain message types to OpenAI Chat-Completion roles."""

    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, HumanMessage):
        return "user"
    raise RuntimeError(
        f"unsupported LangChain message type: {type(message).__name__}"
    )


def _coerce_text(content: Any) -> str:
    """Flatten LangChain-style content (str | list[part]) to plain text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            parts.append(str(part))
        return "".join(parts)
    return str(content)


class OpenAICompatChatTransport:
    """ChatTransport that POSTs the OpenAI Chat Completions schema.

    Used for ``provider in {"openai", "xai", "deepseek"}``. The
    schema is identical down to field names, so a single
    implementation covers all three; differences live in
    :data:`_OPENAI_COMPAT_PROFILES`.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._timeout = timeout
        # ``env`` is an explicit injection seam for tests. Production
        # call sites always read ``os.environ`` via the helper, but a
        # custom env mapping (``MappingProxyType`` etc.) is allowed
        # so harnesses can pin the credentials surface.
        self._env: Mapping[str, str] | None = env

    def _read_env(self, label: str, env_var: str) -> str:
        if self._env is None:
            return _read_required_env(label, env_var)
        raw = self._env.get(env_var, "").strip()
        if not raw:
            raise TransientProviderError(f"{label}: {env_var} is not set")
        return raw

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        profile = _OPENAI_COMPAT_PROFILES.get(provider.provider)
        if profile is None:
            raise RuntimeError(
                f"OpenAICompatChatTransport: provider"
                f" {provider.provider!r} is not in the OpenAI-compat"
                f" profile table"
            )
        label = f"{provider.id} ({provider.provider})"
        api_key = self._read_env(label, profile.env_var)

        payload: dict[str, Any] = {
            "model": kwargs.get("model", profile.default_model),
            "messages": [
                {
                    "role": _role_for(m),
                    "content": _coerce_text(m.content),
                }
                for m in messages
            ],
        }
        stop = kwargs.get("stop")
        if stop:
            payload["stop"] = list(stop)
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{profile.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "dixvision-cognitive-chat/1",
            },
            method="POST",
        )
        raw = _execute(request, timeout=self._timeout, provider_label=label)
        decoded = _parse_json(raw, provider_label=label)
        return _extract_openai_compat_text(decoded, provider_label=label)


def _extract_openai_compat_text(
    decoded: Mapping[str, Any],
    *,
    provider_label: str,
) -> str:
    """Pull ``choices[0].message.content`` out of an OpenAI-shape body."""

    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"{provider_label}: response missing 'choices'"
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError(
            f"{provider_label}: response 'choices[0]' is not an object"
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError(
            f"{provider_label}: response missing 'choices[0].message'"
        )
    content = message.get("content")
    if isinstance(content, str):
        return content
    # Some providers (notably newer OpenAI / xAI variants) return a
    # list of content parts. Flatten the same way LangChain does.
    if isinstance(content, list):
        return _coerce_text(content)
    raise RuntimeError(
        f"{provider_label}: 'choices[0].message.content' has unexpected type"
    )


# --------------------------------------------------------------------
# Google Gemini transport
# --------------------------------------------------------------------


class GoogleGeminiChatTransport:
    """ChatTransport for Google Gemini's REST ``generateContent`` API.

    Gemini differs from OpenAI-compat in three ways:

    * Auth is a ``?key=<api_key>`` query parameter, not a Bearer.
    * Roles are ``"user"`` / ``"model"`` (no ``"system"``); system
      prompts are concatenated onto the first user turn.
    * The reply body lives at
      ``candidates[0].content.parts[*].text``.

    Default model is ``gemini-1.5-flash`` — free-tier friendly and
    fast enough for the operator-chat use case.
    """

    DEFAULT_MODEL = "gemini-1.5-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    ENV_VAR = "GEMINI_API_KEY"

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._timeout = timeout
        self._env: Mapping[str, str] | None = env

    def _read_env(self, label: str) -> str:
        if self._env is None:
            return _read_required_env(label, self.ENV_VAR)
        raw = self._env.get(self.ENV_VAR, "").strip()
        if not raw:
            raise TransientProviderError(
                f"{label}: {self.ENV_VAR} is not set"
            )
        return raw

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        if provider.provider != "google":
            raise RuntimeError(
                f"GoogleGeminiChatTransport: refusing to dispatch"
                f" provider {provider.provider!r}"
            )
        label = f"{provider.id} ({provider.provider})"
        api_key = self._read_env(label)
        model = kwargs.get("model", self.DEFAULT_MODEL)

        contents, system_prefix = _gemini_messages(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_prefix:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prefix}],
            }
        if "temperature" in kwargs or "max_tokens" in kwargs:
            generation: dict[str, Any] = {}
            if "temperature" in kwargs:
                generation["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                generation["maxOutputTokens"] = kwargs["max_tokens"]
            payload["generationConfig"] = generation
        stop = kwargs.get("stop")
        if stop:
            payload.setdefault("generationConfig", {})["stopSequences"] = list(
                stop
            )

        encoded_key = urllib.parse.quote(api_key, safe="")
        url = f"{self.BASE_URL}/models/{model}:generateContent?key={encoded_key}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "dixvision-cognitive-chat/1",
            },
            method="POST",
        )
        raw = _execute(request, timeout=self._timeout, provider_label=label)
        decoded = _parse_json(raw, provider_label=label)
        return _extract_gemini_text(decoded, provider_label=label)


def _gemini_messages(
    messages: Sequence[BaseMessage],
) -> tuple[list[dict[str, Any]], str]:
    """Split ``messages`` into Gemini ``contents`` + a system prefix.

    Gemini has no first-class ``"system"`` role on the
    ``generateContent`` endpoint — instead, system prompts go in
    ``systemInstruction``. We concatenate every system message in
    order so the prompt-engineering shape from PR-1's chat graph
    survives the translation.
    """

    contents: list[dict[str, Any]] = []
    system_chunks: list[str] = []
    for msg in messages:
        text = _coerce_text(msg.content)
        if isinstance(msg, SystemMessage):
            system_chunks.append(text)
            continue
        role = "model" if isinstance(msg, AIMessage) else "user"
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents, "\n".join(system_chunks).strip()


def _extract_gemini_text(
    decoded: Mapping[str, Any],
    *,
    provider_label: str,
) -> str:
    """Pull text out of Gemini's ``candidates[0].content.parts``."""

    candidates = decoded.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(
            f"{provider_label}: response missing 'candidates'"
        )
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise RuntimeError(
            f"{provider_label}: 'candidates[0]' is not an object"
        )
    content = first.get("content")
    if not isinstance(content, Mapping):
        raise RuntimeError(
            f"{provider_label}: 'candidates[0].content' is not an object"
        )
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError(
            f"{provider_label}: 'candidates[0].content.parts' is not a list"
        )
    out: list[str] = []
    for part in parts:
        if isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                out.append(text)
    return "".join(out)


# --------------------------------------------------------------------
# Cognition / Devin transport
# --------------------------------------------------------------------


class CognitionDevinChatTransport:
    """ChatTransport that delegates a turn to the Devin AI session API.

    Devin's public API is built around long-running coding sessions,
    not a synchronous chat-completion endpoint
    (`POST /v1/sessions` returns a session id; the agent runs
    asynchronously and emits messages over a callback / dashboard).
    For an operator chat surface we therefore POST a fresh session
    with the operator prompt as the seed and return a short
    deterministic acknowledgement that includes the session id and
    URL — so the operator can click through to the live session
    while the chat surface continues with whatever provider is
    next in the registry's fallback chain.

    This is intentionally lightweight: a real interactive Devin
    integration belongs on a dedicated ``/dash2/#/devin`` panel
    (future PR) rather than the registry-driven chat. The transport
    exists today so the registry's ``cognition`` row can be flipped
    to ``enabled: true`` and the operator can verify their key end
    to end via ``POST /api/credentials/verify``.
    """

    BASE_URL = "https://api.devin.ai/v1"
    ENV_VAR = "DEVIN_API_KEY"

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._timeout = timeout
        self._env: Mapping[str, str] | None = env

    def _read_env(self, label: str) -> str:
        if self._env is None:
            return _read_required_env(label, self.ENV_VAR)
        raw = self._env.get(self.ENV_VAR, "").strip()
        if not raw:
            raise TransientProviderError(
                f"{label}: {self.ENV_VAR} is not set"
            )
        return raw

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        if provider.provider != "cognition":
            raise RuntimeError(
                f"CognitionDevinChatTransport: refusing to dispatch"
                f" provider {provider.provider!r}"
            )
        label = f"{provider.id} ({provider.provider})"
        api_key = self._read_env(label)

        # Use the most recent user turn as the session seed. The
        # cognitive chat graph's contract (PR-3 / PR-4) guarantees the
        # last message is always a USER message, so this is safe.
        prompt = _coerce_text(messages[-1].content) if messages else ""
        payload = {
            "prompt": prompt,
            "idempotent": True,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.BASE_URL}/sessions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "dixvision-cognitive-chat/1",
            },
            method="POST",
        )
        raw = _execute(request, timeout=self._timeout, provider_label=label)
        decoded = _parse_json(raw, provider_label=label)
        session_id = decoded.get("session_id") or decoded.get("id") or "?"
        url = decoded.get("url") or ""
        suffix = f" {url}" if isinstance(url, str) and url else ""
        return (
            f"Devin session {session_id} started — the agent is now"
            f" running asynchronously and will post results to its own"
            f" session log.{suffix}"
        )


# --------------------------------------------------------------------
# Registry-driven dispatcher
# --------------------------------------------------------------------


class RegistryDispatchChatTransport:
    """ChatTransport that picks the right backend per provider row.

    The dispatch table is keyed on the ``provider`` field of the
    resolved :class:`AIProvider`. Every key is the same string the
    registry YAML and the credentials manifest already use, so
    adding a new provider only requires:

    1. A new row in ``data_source_registry.yaml``
    2. A new ``CredentialBlueprint`` in
       ``system_engine/credentials/manifest.py``
    3. A new entry in this dispatch table

    Routing here is intentionally strict — a provider with no
    matching transport raises :class:`RuntimeError` (not transient),
    so the operator sees the misconfiguration immediately rather
    than a silent fallback to another provider.
    """

    def __init__(self, table: Mapping[str, ChatTransport]) -> None:
        self._table: Mapping[str, ChatTransport] = dict(table)

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        backend = self._table.get(provider.provider)
        if backend is None:
            raise RuntimeError(
                f"RegistryDispatchChatTransport: no backend wired for"
                f" provider {provider.provider!r} (registry id"
                f" {provider.id!r})"
            )
        return backend.invoke(provider, messages, **kwargs)


def build_default_dispatch_transport(
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> RegistryDispatchChatTransport:
    """Return the production dispatch transport.

    Wires the OpenAI-compat backend for ``openai``, ``xai`` and
    ``deepseek``; the Gemini backend for ``google``; and the Devin
    session backend for ``cognition``. Every backend reads its API
    key from ``env`` (or :func:`os.environ` if ``env`` is ``None``)
    on each turn, so an operator who adds a key via
    ``/credentials`` after the runtime starts gets immediate effect
    without a restart.
    """

    openai_compat = OpenAICompatChatTransport(timeout=timeout, env=env)
    gemini = GoogleGeminiChatTransport(timeout=timeout, env=env)
    devin = CognitionDevinChatTransport(timeout=timeout, env=env)
    return RegistryDispatchChatTransport(
        {
            "openai": openai_compat,
            "xai": openai_compat,
            "deepseek": openai_compat,
            "google": gemini,
            "cognition": devin,
        }
    )
