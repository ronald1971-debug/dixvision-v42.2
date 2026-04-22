"""
cockpit.llm \u2014 multi-provider AI router.

Every voice (INDIRA / DYON / GOVERNANCE / DEVIN) gets answers through
this router. Each request declares a set of **capability tags**
(reason / code / translate / sentiment / long_context / realtime_web /
math / offline_ok / multimodal). The router picks the cheapest available
provider with all required tags, falling back through primary \u2192 secondary
\u2192 local \u2192 templated stub.

Providers are pluggable. Missing API key \u2192 provider disabled.

Roles assigned at boot (see docs/AI_ROSTER.md):
    cognition_devin  \u2014 DEVIN voice + DYON coder backend
    anthropic_claude \u2014 GOVERNANCE reasoning + patch reviewer
    openai_gpt4o     \u2014 INDIRA strategy advisor + DEVIN fallback
    google_gemini    \u2014 long-context knowledge ingestor
    xai_grok         \u2014 realtime sentiment / X pulse / alt-data
    ollama_local     \u2014 offline default for chat / translate
    deepseek         \u2014 quant reasoner + cheap translate
    perplexity       \u2014 cited web research

All AI-to-AI hand-offs are logged to META/AI_HANDOFF (audit).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.secrets import get_secret
from system.config import get_config


class Capability(str, Enum):
    REASON = "reason"
    CODE = "code"
    TRANSLATE = "translate"
    SENTIMENT = "sentiment"
    LONG_CONTEXT = "long_context"
    REALTIME_WEB = "realtime_web"
    MATH = "math"
    OFFLINE_OK = "offline_ok"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class Provider:
    name: str                     # "anthropic_claude"
    role: str                     # "GOVERNANCE reasoning + patch reviewer"
    env_key: str                  # "ANTHROPIC_API_KEY"
    capabilities: frozenset[Capability]
    cost_per_1k_tokens_usd: float
    model: str
    endpoint: str = ""
    local: bool = False


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""

    def ok(self) -> bool:
        return not self.error


_ALL_PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="cognition_devin",
        role="DEVIN voice + DYON coder backend",
        env_key="COGNITION_API_KEY",
        capabilities=frozenset({Capability.CODE, Capability.REASON}),
        cost_per_1k_tokens_usd=0.01,
        model="devin-coder-1",
        endpoint="https://api.cognition.ai/v1/chat/completions",
    ),
    Provider(
        name="anthropic_claude",
        role="GOVERNANCE reasoning + patch reviewer",
        env_key="ANTHROPIC_API_KEY",
        capabilities=frozenset({Capability.REASON, Capability.CODE,
                                Capability.LONG_CONTEXT, Capability.MATH}),
        cost_per_1k_tokens_usd=0.015,
        model="claude-sonnet-4",
        endpoint="https://api.anthropic.com/v1/messages",
    ),
    Provider(
        name="openai_gpt4o",
        role="INDIRA strategy advisor + DEVIN fallback",
        env_key="OPENAI_API_KEY",
        capabilities=frozenset({Capability.REASON, Capability.CODE,
                                Capability.MATH, Capability.MULTIMODAL}),
        cost_per_1k_tokens_usd=0.0025,
        model="gpt-4o",
        endpoint="https://api.openai.com/v1/chat/completions",
    ),
    Provider(
        name="google_gemini",
        role="long-context knowledge ingestor",
        env_key="GOOGLE_GENAI_API_KEY",
        capabilities=frozenset({Capability.LONG_CONTEXT, Capability.REASON,
                                Capability.MULTIMODAL}),
        cost_per_1k_tokens_usd=0.0020,
        model="gemini-1.5-pro",
        endpoint="https://generativelanguage.googleapis.com/v1beta",
    ),
    Provider(
        name="xai_grok",
        role="realtime sentiment / X pulse / alt-data",
        env_key="XAI_API_KEY",
        capabilities=frozenset({Capability.SENTIMENT, Capability.REALTIME_WEB,
                                Capability.REASON}),
        cost_per_1k_tokens_usd=0.005,
        model="grok-2-latest",
        endpoint="https://api.x.ai/v1/chat/completions",
    ),
    Provider(
        name="ollama_local",
        role="offline default for chat / translate",
        env_key="",                   # no key \u2014 presence = local server up
        capabilities=frozenset({Capability.REASON, Capability.TRANSLATE,
                                Capability.OFFLINE_OK, Capability.CODE}),
        cost_per_1k_tokens_usd=0.0,
        model="llama3.1:8b",
        endpoint="http://127.0.0.1:11434/api/generate",
        local=True,
    ),
    Provider(
        name="deepseek",
        role="quant reasoner + cheap translate",
        env_key="DEEPSEEK_API_KEY",
        capabilities=frozenset({Capability.MATH, Capability.REASON,
                                Capability.TRANSLATE, Capability.CODE}),
        cost_per_1k_tokens_usd=0.0005,
        model="deepseek-reasoner",
        endpoint="https://api.deepseek.com/v1/chat/completions",
    ),
    Provider(
        name="perplexity",
        role="cited web research",
        env_key="PERPLEXITY_API_KEY",
        capabilities=frozenset({Capability.REALTIME_WEB, Capability.REASON}),
        cost_per_1k_tokens_usd=0.005,
        model="sonar-pro",
        endpoint="https://api.perplexity.ai/chat/completions",
    ),
)


@dataclass
class ProviderStatus:
    name: str
    role: str
    model: str
    enabled: bool
    has_key: bool
    capabilities: list[str]
    cost_per_1k_tokens_usd: float
    local: bool
    total_calls: int = 0
    total_cost_usd: float = 0.0
    last_error: str = ""


class LLMRouter:
    def __init__(self, providers: tuple[Provider, ...] = _ALL_PROVIDERS) -> None:
        self._providers = providers
        self._status: dict[str, ProviderStatus] = {
            p.name: ProviderStatus(
                name=p.name, role=p.role, model=p.model,
                enabled=self._enabled(p), has_key=self._has_key(p),
                capabilities=sorted(c.value for c in p.capabilities),
                cost_per_1k_tokens_usd=p.cost_per_1k_tokens_usd,
                local=p.local,
            )
            for p in providers
        }

    # ------------------------------------------------------------------
    # introspection
    def status(self) -> list[ProviderStatus]:
        return list(self._status.values())

    def available(self, required: frozenset[Capability]) -> list[Provider]:
        out: list[Provider] = []
        for p in self._providers:
            if not self._enabled(p):
                continue
            if required.issubset(p.capabilities):
                out.append(p)
        out.sort(key=lambda p: p.cost_per_1k_tokens_usd)
        return out

    # ------------------------------------------------------------------
    def ask(self, prompt: str, *, system: str = "",
            required: frozenset[Capability] = frozenset({Capability.REASON}),
            prefer: str | None = None,
            max_tokens: int = 512) -> LLMResponse:
        # Explicit preference wins, else cheapest available.
        candidates = self.available(required)
        if prefer:
            preferred = next((p for p in self._providers if p.name == prefer and self._enabled(p)), None)
            if preferred:
                candidates = [preferred] + [p for p in candidates if p.name != prefer]
        if not candidates:
            return self._templated(prompt, system, required, reason="no_provider_with_caps")
        for p in candidates:
            resp = self._dispatch(p, prompt, system, max_tokens)
            self._record(p, resp)
            if resp.ok():
                return resp
        return self._templated(prompt, system, required, reason="all_providers_failed")

    # ------------------------------------------------------------------
    def _enabled(self, p: Provider) -> bool:
        if p.local:
            return _config_get(f"llm.{p.name}.enabled", True)
        return self._has_key(p) and _config_get(f"llm.{p.name}.enabled", True)

    def _has_key(self, p: Provider) -> bool:
        if p.local or not p.env_key:
            return True
        return bool(get_secret(p.env_key, default="") or "")

    def _record(self, p: Provider, r: LLMResponse) -> None:
        s = self._status[p.name]
        s.total_calls += 1
        s.total_cost_usd += r.cost_usd
        s.last_error = r.error or ""

    def _dispatch(self, p: Provider, prompt: str, system: str, max_tokens: int) -> LLMResponse:
        # Kept deliberately minimal; real HTTP calls are best-effort and
        # guarded so the router never breaks the system if a provider is
        # down. Integration tests use monkeypatched transports.
        try:                                                                    # pragma: no cover
            if p.local:
                return _call_ollama(p, prompt, system, max_tokens)
            if p.name == "anthropic_claude":
                return _call_anthropic(p, prompt, system, max_tokens)
            if p.name in ("openai_gpt4o", "xai_grok", "deepseek", "perplexity",
                          "cognition_devin"):
                return _call_openai_compatible(p, prompt, system, max_tokens)
            if p.name == "google_gemini":
                return _call_gemini(p, prompt, system, max_tokens)
        except Exception as e:                                                  # pragma: no cover
            return LLMResponse(text="", provider=p.name, model=p.model,
                               error=repr(e))
        return LLMResponse(text="", provider=p.name, model=p.model,
                           error="unsupported_provider")

    def _templated(self, prompt: str, system: str,
                   required: frozenset[Capability], reason: str) -> LLMResponse:
        caps = "/".join(c.value for c in required)
        body = (f"[OFFLINE-TEMPLATE caps={caps} reason={reason}]\n"
                f"system: {system[:200]}\n"
                f"user:   {prompt[:500]}")
        return LLMResponse(text=body, provider="template", model="none")


def _config_get(key: str, default: bool) -> bool:
    try:
        v = get_config().get(key, default)
        return bool(v)
    except Exception:                                                           # pragma: no cover
        return default


# ---- network call helpers (thin, kept under pragma no cover) ---------
def _call_openai_compatible(p: Provider, prompt: str, system: str,
                            max_tokens: int) -> LLMResponse:                    # pragma: no cover
    import json
    import urllib.request
    key = get_secret(p.env_key, default="")
    payload = {
        "model": p.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        p.endpoint, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    choice = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    cost = (tin + tout) / 1000.0 * p.cost_per_1k_tokens_usd
    return LLMResponse(text=choice.strip(), provider=p.name, model=p.model,
                       cost_usd=cost, tokens_in=tin, tokens_out=tout)


def _call_anthropic(p: Provider, prompt: str, system: str,
                    max_tokens: int) -> LLMResponse:                            # pragma: no cover
    import json
    import urllib.request
    key = get_secret(p.env_key, default="")
    payload = {
        "model": p.model, "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        p.endpoint, data=json.dumps(payload).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    text = body["content"][0]["text"]
    usage = body.get("usage", {})
    tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    cost = (tin + tout) / 1000.0 * p.cost_per_1k_tokens_usd
    return LLMResponse(text=text.strip(), provider=p.name, model=p.model,
                       cost_usd=cost, tokens_in=tin, tokens_out=tout)


def _call_gemini(p: Provider, prompt: str, system: str,
                 max_tokens: int) -> LLMResponse:                               # pragma: no cover
    import json
    import urllib.request
    key = get_secret(p.env_key, default="")
    # Use the x-goog-api-key header rather than ?key= so the secret
    # never appears in HTTP access / proxy / CDN logs.
    url = f"{p.endpoint}/models/{p.model}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=30.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    return LLMResponse(text=text.strip(), provider=p.name, model=p.model)


def _call_ollama(p: Provider, prompt: str, system: str,
                 max_tokens: int) -> LLMResponse:                               # pragma: no cover
    import json
    import urllib.request
    payload = {"model": p.model, "prompt": prompt, "system": system, "stream": False,
               "options": {"num_predict": max_tokens, "temperature": 0.2}}
    req = urllib.request.Request(p.endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60.0) as r:
        body = json.loads(r.read().decode("utf-8"))
    text = body.get("response", "")
    return LLMResponse(text=text.strip(), provider=p.name, model=p.model)


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


__all__ = ["Capability", "Provider", "ProviderStatus", "LLMResponse",
           "LLMRouter", "get_router"]
