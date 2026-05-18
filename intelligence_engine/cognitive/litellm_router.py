# ADAPTED FROM: BerriAI/litellm
# (litellm/router.py — Router fallback chain + retry pattern;
#  litellm/main.py — completion() unified provider interface;
#  litellm/utils.py — token counting + cost tracking shape.)
"""S-12 — Unified LLM gateway with registry-driven fallback + cost ledger.

LiteLLM's ``Router`` walks a list of model deployments and picks the
first that accepts a completion call, recording fallbacks/cost as it
goes. We adapt that exact pattern behind DIX contracts:

1. The :class:`LiteLLMRouter` class is a thin coordinator. The actual
   ``litellm`` import is hidden behind an :class:`LLMTransport`
   Protocol. Production code constructs a transport that lazily
   imports ``litellm`` inside :func:`litellm_completion_transport`;
   unit tests inject a fake. The adapter never imports ``litellm``
   directly, so the module is importable on a host that has never
   installed the package.
2. Provider selection is delegated to a :data:`ProviderResolver`
   callable (same shape as
   :class:`~intelligence_engine.cognitive.chat.registry_driven_chat_model.\
RegistryDrivenChatModel` and
   :class:`~intelligence_engine.cognitive.typed_ai.TypedAIAgent`).
   The router itself never reads ``registry/`` or
   ``system_engine.scvs`` — it is leaf-pure (B1 cross-engine
   isolation).
3. Cost tracking writes through an injected :data:`CostLedgerSink`
   callable (signature ``(LLMUsage) -> None``). The router never
   imports the ledger; the operator wires the sink at construction
   time. Cost is recorded **once on success only** — transient
   failures and timeouts are ledger-silent (no double-billing on
   replay).
4. Hard timeout: every per-provider call carries an explicit
   ``timeout_s`` value (default ``DEFAULT_TIMEOUT_S = 30.0``,
   maximum ``MAX_TIMEOUT_S = 30.0``). The router never reads the
   wall clock — it forwards the value to the transport which is
   responsible for enforcing it (httpx ``timeout=`` parameter or
   ``litellm.completion(timeout=...)`` kwarg). On timeout the
   transport raises :class:`ProviderTimeoutError` (a
   :class:`TransientProviderError` subclass) and the router rotates
   to the next provider exactly like any other transient failure.
5. Determinism (INV-15): every output of :meth:`LiteLLMRouter.\
complete` is a function of the inputs (request + provider tuple +
   transport behaviour + ``ts_ns`` + ``request_id`` arguments). The
   router never reads the wall clock, never imports ``os``/``time``,
   and never mutates global state. ``LLMResponse.request_id`` is
   supplied by the caller so replays are byte-identical.

What survives from upstream
---------------------------

* The ``Router._async_function_with_fallbacks`` outer-loop shape:
  walk the deployments tuple, transient errors trigger fallback,
  non-transient errors propagate.
* The "deployment row + completion call + cost callback" tri-tuple
  from ``litellm/router.py`` and ``litellm/utils.py``.

What is rewritten behind DIX contracts
--------------------------------------

* Provider selection (``litellm`` reads its own model list — we
  delegate to :data:`ProviderResolver` against the registry).
* Cost callback (``litellm`` writes to its built-in callbacks list —
  we forward to a typed :data:`CostLedgerSink`).
* Retry strategy (``litellm`` has its own backoff / circuit-breaker
  layer — we keep the loop deterministic, no clock, no PRNG; the
  caller is responsible for any sleep between calls).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from core.cognitive_router import AIProvider, TaskClass

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("litellm",)
"""S-12 introduces a runtime-optional dependency on litellm.

The package is **only** required if the operator wires
:func:`litellm_completion_transport` as the production transport.
Test deployments and any host that exclusively uses an injected fake
transport do not need litellm installed; the module imports cleanly
without it.
"""

DEFAULT_TIMEOUT_S: float = 30.0
"""Default per-provider hard timeout. Caller may pass a smaller
value via :class:`LLMRequest.timeout_s`; values above
:data:`MAX_TIMEOUT_S` are rejected at construction time."""

MAX_TIMEOUT_S: float = 30.0
"""Hard ceiling on per-provider timeout. The S-12 spec states
"30s max per call — never block indefinitely"; this constant is the
machine-checked form of that rule."""

__all__ = [
    "AllProvidersFailedError",
    "ChatMessage",
    "CostLedgerSink",
    "DEFAULT_TIMEOUT_S",
    "FallbackAuditSink",
    "LLMRequest",
    "LLMResponse",
    "LLMTransport",
    "LLMUsage",
    "LiteLLMRouter",
    "MAX_TIMEOUT_S",
    "NoEligibleProviderError",
    "ProviderResolver",
    "ProviderTimeoutError",
    "TransientProviderError",
    "litellm_completion_transport",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransientProviderError(RuntimeError):
    """Raised by an :class:`LLMTransport` when the provider failed in a
    way that warrants trying the next eligible provider.

    Examples: 429 rate-limit, 503 overloaded, network timeout
    (see :class:`ProviderTimeoutError`).
    Non-examples: 401 unauthenticated, 400 malformed request — those
    propagate unchanged so the chain stops on deterministic failures.
    """


class ProviderTimeoutError(TransientProviderError):
    """Raised by an :class:`LLMTransport` when the per-provider hard
    timeout fires.

    Treated as transient by the router (rotates to the next provider).
    Distinct subclass so audits and tests can distinguish wall-clock
    timeouts from upstream rate-limit / 503 failures.
    """


class NoEligibleProviderError(RuntimeError):
    """Raised when :data:`ProviderResolver` returned an empty tuple."""


class AllProvidersFailedError(RuntimeError):
    """Raised when every eligible provider raised
    :class:`TransientProviderError`.

    Wraps the most recent transient error as ``__cause__`` so the
    traceback retains the underlying failure.
    """


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


@dataclasses.dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn in a chat completion request.

    Mirrors the OpenAI / litellm message envelope (``role`` +
    ``content``) with strict validation. ``role`` must be one of
    ``"system"`` / ``"user"`` / ``"assistant"``; ``content`` must be a
    non-empty string. The ``"tool"`` and ``"function"`` roles are
    intentionally not supported here — tool calls go through a
    separate channel (see ``intelligence_engine/cognitive/typed_ai.py``
    for structured outputs).
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise TypeError("ChatMessage.role must be str")
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"ChatMessage.role must be one of {sorted(_VALID_ROLES)!r}; got {self.role!r}"
            )
        if not isinstance(self.content, str):
            raise TypeError("ChatMessage.content must be str")
        if not self.content:
            raise ValueError("ChatMessage.content must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class LLMRequest:
    """Caller-supplied request to :meth:`LiteLLMRouter.complete`.

    Fields:

    * ``task`` — :class:`TaskClass` for provider eligibility (the
      :data:`ProviderResolver` is expected to filter on this).
    * ``messages`` — non-empty tuple of :class:`ChatMessage`.
    * ``max_tokens`` — upper bound on completion tokens. Positive int.
    * ``temperature`` — sampling temperature in ``[0.0, 2.0]``.
    * ``timeout_s`` — per-provider hard timeout, in seconds. Must be
      in ``(0, MAX_TIMEOUT_S]``. Defaults to :data:`DEFAULT_TIMEOUT_S`.
    """

    task: TaskClass
    messages: tuple[ChatMessage, ...]
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskClass):
            raise TypeError("LLMRequest.task must be a TaskClass")
        if not isinstance(self.messages, tuple):
            raise TypeError("LLMRequest.messages must be a tuple")
        if not self.messages:
            raise ValueError("LLMRequest.messages must be non-empty")
        for i, msg in enumerate(self.messages):
            if not isinstance(msg, ChatMessage):
                raise TypeError(f"LLMRequest.messages[{i}] must be a ChatMessage")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("LLMRequest.max_tokens must be int")
        if self.max_tokens <= 0:
            raise ValueError("LLMRequest.max_tokens must be > 0")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("LLMRequest.temperature must be int|float")
        if not (0.0 <= float(self.temperature) <= 2.0):
            raise ValueError("LLMRequest.temperature must be in [0.0, 2.0]")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)):
            raise TypeError("LLMRequest.timeout_s must be int|float")
        ts = float(self.timeout_s)
        if ts <= 0.0:
            raise ValueError("LLMRequest.timeout_s must be > 0")
        if ts > MAX_TIMEOUT_S:
            raise ValueError(
                f"LLMRequest.timeout_s must be <= {MAX_TIMEOUT_S} (S-12 hard ceiling); got {ts}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class LLMUsage:
    """Per-call token + cost accounting.

    Fields are non-negative; cost is denominated in USD. The transport
    is responsible for computing these from the upstream response —
    this object is just the canonical envelope used by the cost
    ledger sink.
    """

    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        for fname in ("prompt_tokens", "completion_tokens"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"LLMUsage.{fname} must be int")
            if v < 0:
                raise ValueError(f"LLMUsage.{fname} must be >= 0")
        if isinstance(self.cost_usd, bool) or not isinstance(self.cost_usd, (int, float)):
            raise TypeError("LLMUsage.cost_usd must be int|float")
        cost = float(self.cost_usd)
        # NaN check — `cost != cost` is the canonical idiom.
        if cost != cost:  # noqa: PLR0124
            raise ValueError("LLMUsage.cost_usd must not be NaN")
        if cost < 0.0:
            raise ValueError("LLMUsage.cost_usd must be >= 0")

    @property
    def total_tokens(self) -> int:
        """Sum of prompt + completion tokens (no separate field — kept
        as a derived view to match the litellm response shape)."""

        return self.prompt_tokens + self.completion_tokens


@dataclasses.dataclass(frozen=True, slots=True)
class LLMResponse:
    """Output of :meth:`LiteLLMRouter.complete`.

    Fields:

    * ``request_id`` — caller-supplied opaque id for ledger
      correlation (replays must pass the same id to be byte-identical).
    * ``ts_ns`` — caller-supplied monotonic ns timestamp. The router
      never reads the wall clock.
    * ``provider_id`` — id of the provider that produced ``content``
      (the first non-failing provider in the resolver order).
    * ``content`` — assistant reply text.
    * ``usage`` — :class:`LLMUsage` for cost-ledger accounting.
    * ``attempts`` — non-empty tuple of provider ids actually tried
      (in order). The last entry is always ``provider_id``; earlier
      entries are the providers that raised :class:`TransientProviderError`.
    """

    request_id: str
    ts_ns: int
    provider_id: str
    content: str
    usage: LLMUsage
    attempts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("LLMResponse.request_id must be non-empty str")
        if isinstance(self.ts_ns, bool) or not isinstance(self.ts_ns, int):
            raise TypeError("LLMResponse.ts_ns must be int")
        if self.ts_ns <= 0:
            raise ValueError("LLMResponse.ts_ns must be positive")
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("LLMResponse.provider_id must be non-empty str")
        if not isinstance(self.content, str):
            raise TypeError("LLMResponse.content must be str")
        if not isinstance(self.usage, LLMUsage):
            raise TypeError("LLMResponse.usage must be LLMUsage")
        if not isinstance(self.attempts, tuple):
            raise TypeError("LLMResponse.attempts must be a tuple")
        if not self.attempts:
            raise ValueError("LLMResponse.attempts must be non-empty")
        for i, a in enumerate(self.attempts):
            if not isinstance(a, str) or not a:
                raise ValueError(f"LLMResponse.attempts[{i}] must be non-empty str")
        if self.attempts[-1] != self.provider_id:
            raise ValueError(
                "LLMResponse.attempts[-1] must equal provider_id; got "
                f"attempts[-1]={self.attempts[-1]!r}, "
                f"provider_id={self.provider_id!r}"
            )


# ---------------------------------------------------------------------------
# Pluggable seams
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMTransport(Protocol):
    """Per-call dispatch to a single resolved AI provider.

    Implementations may be HTTP, MCP, gRPC, or a local litellm shim.
    The transport is responsible for:

    * translating the registry row into a concrete client (auth,
      base URL, model name);
    * enforcing the per-call ``timeout_s`` (passed to httpx /
      litellm as the ``timeout=`` kwarg);
    * computing :class:`LLMUsage` from the upstream response.

    The transport is **not** responsible for selection, ordering, or
    fallback. Selection lives in :class:`LiteLLMRouter`; the
    transport is dumb.
    """

    def complete(
        self,
        provider: AIProvider,
        request: LLMRequest,
        /,
    ) -> tuple[str, LLMUsage]:
        """Send ``request`` to ``provider`` and return
        ``(content, usage)``.

        The transport must enforce ``request.timeout_s``: if the
        upstream call has not returned within that many seconds, the
        transport must raise :class:`ProviderTimeoutError`. Any other
        timeout-shaped failure (network reset, 503, 429) raises
        :class:`TransientProviderError` so the router can rotate.

        Non-transient failures (401 auth, 400 malformed) propagate
        unchanged.
        """


FallbackAuditSink = Callable[[AIProvider, str], None]
"""Callable invoked once per ``SOURCE_FALLBACK_ACTIVATED`` audit
emitted on a transient provider failure (signature
``(provider, reason) -> None``)."""


CostLedgerSink = Callable[[AIProvider, LLMUsage], None]
"""Callable invoked once per **successful** LLM call. Production
wiring routes the call to the audit ledger row writer; tests pass a
list-appender. Signature: ``(provider, usage) -> None``."""


ProviderResolver = Callable[[], tuple[AIProvider, ...]]
"""Zero-arg callable returning eligible providers in priority order.

Production wiring binds this to
``lambda: select_providers(registry, task)`` against the live SCVS
registry; tests bind a list-returner. Inverting this dependency keeps
the router free of any direct ``system_engine`` import (B1
cross-engine isolation)."""


def _noop_audit(_provider: AIProvider, _reason: str) -> None:
    return None


def _noop_cost(_provider: AIProvider, _usage: LLMUsage) -> None:
    return None


# ---------------------------------------------------------------------------
# LiteLLMRouter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class LiteLLMRouter:
    """Coordinator that turns an :class:`LLMRequest` into an
    :class:`LLMResponse` via a registry-driven fallback chain.

    The router runs a single outer loop:

    * iterate eligible providers in :data:`ProviderResolver` order;
    * call :meth:`LLMTransport.complete` for each;
    * a :class:`TransientProviderError` (including
      :class:`ProviderTimeoutError`) drops a :data:`FallbackAuditSink`
      row and rotates to the next provider;
    * any other exception type propagates unchanged (auth, schema,
      operator-cancelled — these are not fallback-eligible);
    * on the first non-failing provider the router invokes
      :data:`CostLedgerSink` exactly once and returns the response.

    There is no inner retry loop — the router is deterministic and
    has no clock / no PRNG. If the operator wants a backoff between
    calls, that lives in a wrapping coroutine (e.g.
    :func:`execution_engine.market_data.aggregator.next_reconnect_delay_s`
    from S-11).

    All fields are frozen; :meth:`complete` is the only public method.
    """

    provider_resolver: ProviderResolver
    transport: LLMTransport
    fallback_audit: FallbackAuditSink = dataclasses.field(default=_noop_audit)
    cost_ledger: CostLedgerSink = dataclasses.field(default=_noop_cost)

    def __post_init__(self) -> None:
        if not callable(self.provider_resolver):
            raise TypeError("LiteLLMRouter.provider_resolver must be callable")
        if not isinstance(self.transport, LLMTransport):
            raise TypeError("LiteLLMRouter.transport must implement the LLMTransport Protocol")
        if not callable(self.fallback_audit):
            raise TypeError("LiteLLMRouter.fallback_audit must be callable")
        if not callable(self.cost_ledger):
            raise TypeError("LiteLLMRouter.cost_ledger must be callable")

    def complete(
        self,
        request: LLMRequest,
        *,
        ts_ns: int,
        request_id: str,
    ) -> LLMResponse:
        """Execute ``request`` and return one :class:`LLMResponse`.

        Args:
            request: Validated :class:`LLMRequest`.
            ts_ns: Monotonic ns timestamp supplied by the caller.
                The router never reads the wall clock.
            request_id: Caller-supplied opaque id for ledger
                correlation. Replays must pass the same id to be
                byte-identical (INV-15).

        Returns:
            :class:`LLMResponse` produced by the first non-failing
            provider in resolver order. ``response.attempts`` is the
            ordered tuple of provider ids actually tried.

        Raises:
            NoEligibleProviderError: ``provider_resolver`` returned
                an empty tuple. No fallback path exists.
            AllProvidersFailedError: Every eligible provider raised
                :class:`TransientProviderError`. Wraps the last
                transient failure as ``__cause__``.
            Exception: Any non-transient failure from the transport
                (auth, schema, operator-cancelled) propagates
                unchanged.
        """

        if not isinstance(request, LLMRequest):
            raise TypeError("LiteLLMRouter.complete: request must be LLMRequest")
        if isinstance(ts_ns, bool) or not isinstance(ts_ns, int):
            raise TypeError("LiteLLMRouter.complete: ts_ns must be int")
        if ts_ns <= 0:
            raise ValueError("LiteLLMRouter.complete: ts_ns must be positive")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("LiteLLMRouter.complete: request_id must be non-empty str")

        providers: Sequence[AIProvider] = self.provider_resolver()
        if not isinstance(providers, tuple):
            raise TypeError(
                "LiteLLMRouter.provider_resolver must return a tuple;"
                f" got {type(providers).__name__}"
            )
        if not providers:
            raise NoEligibleProviderError(
                "LiteLLMRouter: provider_resolver returned no eligible providers"
            )

        attempts: list[str] = []
        last_transient: TransientProviderError | None = None
        for provider in providers:
            if not isinstance(provider, AIProvider):
                raise TypeError(
                    "LiteLLMRouter.provider_resolver must return tuple[AIProvider, ...]"
                )
            attempts.append(provider.id)
            try:
                content, usage = self.transport.complete(provider, request)
            except TransientProviderError as exc:
                last_transient = exc
                self.fallback_audit(provider, _audit_reason(exc))
                continue

            if not isinstance(content, str):
                raise TypeError(
                    "LiteLLMRouter: transport must return (str, LLMUsage)"
                    f" tuple; got content={type(content).__name__}"
                )
            if not isinstance(usage, LLMUsage):
                raise TypeError(
                    "LiteLLMRouter: transport must return (str, LLMUsage)"
                    f" tuple; got usage={type(usage).__name__}"
                )

            self.cost_ledger(provider, usage)
            return LLMResponse(
                request_id=request_id,
                ts_ns=ts_ns,
                provider_id=provider.id,
                content=content,
                usage=usage,
                attempts=tuple(attempts),
            )

        # All providers raised TransientProviderError.
        msg = (
            "LiteLLMRouter: every eligible provider raised a transient"
            f" error (attempts={tuple(attempts)!r})"
        )
        raise AllProvidersFailedError(msg) from last_transient


def _audit_reason(exc: TransientProviderError) -> str:
    """Render a deterministic audit reason for ``exc``.

    The transport's exception type and message are public-audit-safe;
    the router does not include traceback data so the audit row is
    byte-stable across replays.
    """

    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Production transport (lazy litellm import)
# ---------------------------------------------------------------------------


def litellm_completion_transport() -> LLMTransport:
    """Return an :class:`LLMTransport` backed by the real ``litellm``
    library.

    Lazily imports ``litellm`` so the module imports cleanly on a
    host where the package is not installed; if the operator wires
    this transport into a router and then calls ``complete``, the
    underlying call will execute through ``litellm.completion``.

    Raises:
        ImportError: If ``litellm`` is not installed at construction
            time.
    """

    import litellm  # noqa: F401  # noqa: PLC0415

    return _LiteLLMCompletionTransport()


@dataclasses.dataclass(frozen=True, slots=True)
class _LiteLLMCompletionTransport:
    """Concrete :class:`LLMTransport` adapter over ``litellm.completion``.

    Translates :class:`AIProvider` + :class:`LLMRequest` into the
    ``litellm.completion(model=..., messages=..., timeout=...,
    max_tokens=..., temperature=...)`` call shape, and unpacks the
    response into ``(content, LLMUsage)``.

    Errors are mapped to the DIX hierarchy:

    * ``litellm.Timeout`` / ``litellm.APITimeoutError`` →
      :class:`ProviderTimeoutError`;
    * ``litellm.RateLimitError`` / ``litellm.ServiceUnavailableError`` /
      ``litellm.APIConnectionError`` → :class:`TransientProviderError`;
    * any other ``litellm.*Error`` propagates unchanged so the router
      stops the chain on deterministic failures (401 auth, 400
      malformed, schema, ...).
    """

    def complete(
        self,
        provider: AIProvider,
        request: LLMRequest,
        /,
    ) -> tuple[str, LLMUsage]:
        import litellm  # noqa: PLC0415

        timeout_cls = _safe_attr(litellm, "Timeout") or _safe_attr(litellm, "APITimeoutError")
        rate_limit_cls = _safe_attr(litellm, "RateLimitError")
        service_unavailable_cls = _safe_attr(litellm, "ServiceUnavailableError")
        api_connection_cls = _safe_attr(litellm, "APIConnectionError")

        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        try:
            resp = litellm.completion(
                model=provider.endpoint,
                messages=msgs,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            if timeout_cls is not None and isinstance(exc, timeout_cls):
                raise ProviderTimeoutError(str(exc)) from exc
            for cls in (
                rate_limit_cls,
                service_unavailable_cls,
                api_connection_cls,
            ):
                if cls is not None and isinstance(exc, cls):
                    raise TransientProviderError(str(exc)) from exc
            raise

        return _unpack_litellm_response(resp)


def _safe_attr(mod: Any, name: str) -> type | None:  # pragma: no cover - tiny
    cls = getattr(mod, name, None)
    return cls if isinstance(cls, type) else None


def _unpack_litellm_response(resp: Any) -> tuple[str, LLMUsage]:
    """Pull ``(content, LLMUsage)`` out of a ``litellm.ModelResponse``.

    Tolerant of both the ``ModelResponse`` dataclass and a plain
    ``dict`` (litellm's ``return_response_dict=True`` mode), which
    keeps unit tests cheap.
    """

    content = _dig(resp, ("choices", 0, "message", "content"))
    if not isinstance(content, str):
        raise RuntimeError(
            "_LiteLLMCompletionTransport: litellm response missing choices[0].message.content"
        )
    usage_obj = _dig(resp, ("usage",))
    prompt_tokens = int(_dig(usage_obj, ("prompt_tokens",)) or 0)
    completion_tokens = int(_dig(usage_obj, ("completion_tokens",)) or 0)
    cost_raw = _dig(resp, ("_hidden_params", "response_cost"))
    cost_usd = float(cost_raw) if cost_raw is not None else 0.0
    return content, LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def _dig(obj: Any, path: tuple[Any, ...]) -> Any:  # pragma: no cover - tiny
    cur: Any = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(key, int):
            try:
                cur = cur[key]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                cur = getattr(cur, key, None)
    return cur
