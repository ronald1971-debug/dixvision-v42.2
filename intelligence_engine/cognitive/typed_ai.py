# ADAPTED FROM: pydantic_ai/agent.py
"""S-06 — Typed AI provider abstraction for governance proposals.

This module is the canonical adaptation of `pydantic-ai`_ behind the DIX
contract surface. ``pydantic-ai`` ships an ``Agent[ResultT]`` class that
runs an LLM, validates the JSON response against a Pydantic schema, and
retries on validation failure. We **rewrite** that pattern behind DIX
contracts:

1. The :class:`TypedAIAgent` class is a thin coordinator. The actual
   pydantic-ai client is hidden behind a :class:`TypedAITransport`
   Protocol. Production code constructs a transport that lazily imports
   ``pydantic_ai`` inside :func:`pydantic_ai_transport`; unit tests
   inject a fake. The adapter never imports ``pydantic_ai`` directly,
   so the module remains importable on a host that has never installed
   the package.
2. Schema validation is **mandatory**. Every typed result must extend
   :class:`TypedAIResult`. If the LLM produces JSON that fails Pydantic
   validation we either retry (up to ``max_retries``) or raise
   :class:`SchemaValidationError` — there is no "best effort" path.
3. The agent's only public side-effect is enqueueing a
   :class:`TypedAIProposal` via an injected
   :data:`ProposalSubmitter` callable. **It never emits**
   execution-side events **directly**, never writes to the ledger, and never
   reaches the execution chokepoint. INV-12 (governance approval is
   the only path to live execution) is preserved structurally.
4. Provider selection is delegated to a :data:`ProviderResolver`
   callable (same pattern as
   :class:`~intelligence_engine.cognitive.chat.registry_driven_chat_model.\
RegistryDrivenChatModel`). ``TypedAIAgent`` itself never reads
   ``registry/`` or ``system_engine.scvs`` — it is leaf-pure.
5. Determinism (INV-15): every output of :meth:`TypedAIAgent.run_typed`
   is a function of the inputs (prompt + provider tuple + schema +
   transport behaviour + ``ts_ns`` argument). The agent never reads
   the wall clock, never imports ``os``, and never mutates global
   state. ``TypedAIProposal.proposal_id`` is supplied by an injected
   ``id_factory`` so replays are byte-identical.

What survives from upstream
---------------------------

* The validate-then-retry loop shape from
  ``pydantic_ai/agent.py::Agent._handle_text_response`` — JSON parse,
  Pydantic validate, on failure feed the validator's error message
  back to the LLM and retry up to ``max_retries`` times.
* The "result_type is a pydantic.BaseModel subclass" contract — the
  LLM's JSON is validated through ``schema.model_validate`` exactly
  the way pydantic-ai does.

What is rewritten behind DIX contracts
--------------------------------------

* Provider selection, fallback audits, and the proposal-emission path
  are DIX-native; ``pydantic_ai.models`` is never reached for through
  this module.
* The ``RunResult`` envelope is replaced with the frozen
  :class:`TypedAIProposal` dataclass that is structurally compatible
  with the existing governance approval queue.

.. _pydantic-ai: https://github.com/pydantic/pydantic-ai
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from core.cognitive_router import AIProvider, TaskClass

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("pydantic-ai",)
"""S-06 introduces a runtime-optional dependency on pydantic-ai.

The package is **only** required if the operator wires
:func:`pydantic_ai_transport` as the production transport. Test
deployments and any host that exclusively uses an injected fake
transport do not need pydantic-ai installed; the module imports
cleanly without it.
"""

__all__ = [
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
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransientProviderError(RuntimeError):
    """Raised by a :class:`TypedAITransport` when the provider failed in a
    way that warrants trying the next eligible provider.

    Examples: 429 rate-limit, 503 overloaded, network timeout. Schema
    validation failures are *not* transient — they raise
    :class:`SchemaValidationError` instead so the agent retries on the
    same provider rather than rotating.
    """


class NoEligibleProviderError(RuntimeError):
    """Raised when :data:`ProviderResolver` returned an empty tuple."""


class AllProvidersFailedError(RuntimeError):
    """Raised when every eligible provider raised :class:`TransientProviderError`.

    Wraps the most recent transient error as ``__cause__`` so the
    traceback retains the underlying failure.
    """


class SchemaValidationError(RuntimeError):
    """Raised when the LLM produced output that fails Pydantic validation
    after exhausting all retries.

    The agent **never** falls back to a "best effort" response: a
    validation failure is a hard rejection of the proposal.
    """


# ---------------------------------------------------------------------------
# Result base + frozen request/proposal envelopes
# ---------------------------------------------------------------------------


class TypedAIResult(BaseModel):
    """Base class every typed-AI proposal payload must extend.

    Marker base for :class:`TypedAIProposal.validated_result`. The
    governance approval queue uses the schema name to route proposals
    to the correct projection; refusing to instantiate
    :class:`TypedAIProposal` with a non-:class:`TypedAIResult` payload
    is what makes INV-12 enforceable at the type system rather than
    by convention.
    """

    model_config = {"frozen": True, "extra": "forbid"}


_ResultT = TypeVar("_ResultT", bound=TypedAIResult)


@dataclasses.dataclass(frozen=True, slots=True)
class TypedAIRequest(Generic[_ResultT]):
    """Caller-supplied request to :meth:`TypedAIAgent.run_typed`.

    Fields:

    * ``task`` — :class:`TaskClass` for provider eligibility.
    * ``prompt`` — operator-facing prompt text.
    * ``schema_class`` — concrete :class:`TypedAIResult` subclass that
      the LLM output must validate against. Mandatory; there is no
      "free-text" mode.
    * ``max_retries`` — number of additional attempts to retry on a
      Pydantic validation failure on the same provider before
      escalating to provider fallback. Defaults to ``2`` to match the
      pydantic-ai default.
    """

    task: TaskClass
    prompt: str
    schema_class: type[_ResultT]
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("TypedAIRequest.prompt must be str")
        if not self.prompt:
            raise ValueError("TypedAIRequest.prompt must be non-empty")
        if not isinstance(self.schema_class, type) or not issubclass(
            self.schema_class, TypedAIResult
        ):
            raise TypeError("TypedAIRequest.schema_class must be a TypedAIResult subclass")
        if self.schema_class is TypedAIResult:
            raise ValueError(
                "TypedAIRequest.schema_class must be a *concrete* subclass of"
                " TypedAIResult; the base class itself has no fields"
            )
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise TypeError("TypedAIRequest.max_retries must be int")
        if self.max_retries < 0:
            raise ValueError("TypedAIRequest.max_retries must be >= 0")


@dataclasses.dataclass(frozen=True, slots=True)
class TypedAIProposal(Generic[_ResultT]):
    """Output of :meth:`TypedAIAgent.run_typed`.

    The agent's only side-effect is emitting one of these via the
    injected :data:`ProposalSubmitter`. The receiver is responsible
    for routing it to the governance approval queue — typically via
    :class:`~intelligence_engine.cognitive.approval_queue.ApprovalQueue.\
submit`. The agent itself never imports ``governance_engine``.

    Fields:

    * ``proposal_id`` — generated by :data:`TypedAIAgent`'s injected
      ``id_factory`` (uuid4 hex by default; tests inject a counter).
    * ``ts_ns`` — caller-supplied monotonic ns timestamp. The agent
      never reads the wall clock.
    * ``task`` — original :class:`TaskClass`.
    * ``provider_id`` — id of the provider that produced the
      validated result.
    * ``schema_name`` — fully-qualified name of the
      :class:`TypedAIResult` subclass, e.g.
      ``"my.module.MyProposalPayload"``. Surfaced for audit so the
      governance projection knows which schema generated the row.
    * ``validated_result`` — the validated :class:`TypedAIResult`.
    """

    proposal_id: str
    ts_ns: int
    task: TaskClass
    provider_id: str
    schema_name: str
    validated_result: _ResultT

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise ValueError("TypedAIProposal.proposal_id must be non-empty str")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError("TypedAIProposal.ts_ns must be int")
        if self.ts_ns <= 0:
            raise ValueError("TypedAIProposal.ts_ns must be positive")
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("TypedAIProposal.provider_id must be non-empty str")
        if not isinstance(self.schema_name, str) or not self.schema_name:
            raise ValueError("TypedAIProposal.schema_name must be non-empty str")
        if not isinstance(self.validated_result, TypedAIResult):
            raise TypeError("TypedAIProposal.validated_result must be a TypedAIResult instance")


# ---------------------------------------------------------------------------
# Pluggable seams
# ---------------------------------------------------------------------------


@runtime_checkable
class TypedAITransport(Protocol):
    """Per-attempt dispatch to a single provider.

    The transport produces a *raw text response* — the agent is
    responsible for parsing/validating it. The transport is dumb on
    purpose: schema retries, fallback selection, and proposal emission
    are agent-level concerns, never transport-level.

    ``retry_feedback`` is the validator's ``str(ValidationError)``
    from the previous attempt, present on retries only. The transport
    is expected to splice it into the prompt so the LLM can correct
    itself; the canonical implementation in
    :func:`pydantic_ai_transport` does this with pydantic-ai's
    built-in mechanism.
    """

    def invoke(
        self,
        provider: AIProvider,
        prompt: str,
        schema_class: type[TypedAIResult],
        /,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        """Send ``prompt`` + schema instructions to ``provider`` and
        return the raw assistant text (expected to be JSON).

        Raises:
            TransientProviderError: If the provider is temporarily
                unavailable. The agent records a fallback audit and
                tries the next eligible provider.
            Exception: Any other exception type propagates without
                fallback (auth, bad schema, operator-cancelled).
        """


FallbackAuditSink = Callable[[AIProvider, str], None]
"""Callable invoked once per ``SOURCE_FALLBACK_ACTIVATED`` audit
emitted on a transient provider failure."""


ProviderResolver = Callable[[], tuple[AIProvider, ...]]
"""Zero-arg callable returning eligible providers in priority order."""


ProposalSubmitter = Callable[[TypedAIProposal[Any]], None]
"""Callable invoked once per validated proposal.

Signature: ``(proposal) -> None``. Production wiring routes the call
to :meth:`~intelligence_engine.cognitive.approval_queue.\
ApprovalQueue.submit`; tests pass a list-appender. The default
submitter is a no-op so unit-test construction is cheap."""


_IdFactory = Callable[[], str]


def default_id_factory() -> str:
    """Production proposal-id generator (uuid4 hex, 32 chars).

    Imported lazily inside the function body so replays in test mode
    can substitute a counter without ever instantiating ``uuid``.
    """

    import uuid

    return uuid.uuid4().hex


def _noop_audit(_provider: AIProvider, _reason: str) -> None:
    return None


def _noop_submit(_proposal: TypedAIProposal[Any]) -> None:
    return None


def _schema_qualname(cls: type[TypedAIResult]) -> str:
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", cls.__name__)
    if module and module not in ("__main__", "builtins"):
        return f"{module}.{qualname}"
    return qualname


# ---------------------------------------------------------------------------
# TypedAIAgent
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TypedAIAgent:
    """Coordinator that turns a :class:`TypedAIRequest` into a
    schema-validated :class:`TypedAIProposal`.

    The agent runs a two-level loop:

    * **Outer loop** — iterate eligible providers in resolver order.
      A :class:`TransientProviderError` from the transport drops a
      :data:`FallbackAuditSink` row and rotates to the next provider.
    * **Inner loop** — for each provider, attempt the LLM call up to
      ``request.max_retries + 1`` times. On a Pydantic
      :class:`pydantic.ValidationError` (or any JSON-parse failure)
      we *do not* rotate provider; we re-prompt with the validator's
      error message until the budget is exhausted, then re-raise as
      :class:`SchemaValidationError`.

    The exhaustion behaviour deliberately differs between transient
    and validation failures: a flaky provider should hand off, but a
    provider that consistently produces bad JSON should be punished
    by the schema-strictness signal rather than by silently rotating.

    All fields are frozen; ``run_typed`` is the only public method.
    """

    provider_resolver: ProviderResolver
    transport: TypedAITransport
    submit_proposal: ProposalSubmitter = dataclasses.field(default=_noop_submit)
    fallback_audit: FallbackAuditSink = dataclasses.field(default=_noop_audit)
    id_factory: _IdFactory = dataclasses.field(default=default_id_factory)

    def __post_init__(self) -> None:
        if not callable(self.provider_resolver):
            raise TypeError("TypedAIAgent.provider_resolver must be callable")
        if not isinstance(self.transport, TypedAITransport):
            raise TypeError("TypedAIAgent.transport must implement the TypedAITransport Protocol")
        if not callable(self.submit_proposal):
            raise TypeError("TypedAIAgent.submit_proposal must be callable")
        if not callable(self.fallback_audit):
            raise TypeError("TypedAIAgent.fallback_audit must be callable")
        if not callable(self.id_factory):
            raise TypeError("TypedAIAgent.id_factory must be callable")

    def run_typed(
        self,
        request: TypedAIRequest[_ResultT],
        *,
        ts_ns: int,
    ) -> TypedAIProposal[_ResultT]:
        """Execute ``request`` and emit one schema-validated proposal.

        The agent contract:

        * On success — emits exactly one :class:`TypedAIProposal` to
          :data:`ProposalSubmitter` and returns the same proposal
          instance.
        * On schema-validation exhaustion on a single provider —
          raises :class:`SchemaValidationError`. We do **not** rotate
          providers on schema failure (see class docstring).
        * On every provider failing transiently — raises
          :class:`AllProvidersFailedError` whose ``__cause__`` is the
          last :class:`TransientProviderError`.
        * On the resolver returning no providers — raises
          :class:`NoEligibleProviderError` immediately.

        ``ts_ns`` must be a positive monotonic-ns timestamp supplied
        by the caller; the agent never reads the wall clock.
        """

        if not isinstance(request, TypedAIRequest):
            raise TypeError("TypedAIAgent.run_typed: request must be TypedAIRequest")
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError("TypedAIAgent.run_typed: ts_ns must be int")
        if ts_ns <= 0:
            raise ValueError("TypedAIAgent.run_typed: ts_ns must be positive")

        eligible: Sequence[AIProvider] = tuple(self.provider_resolver())
        if not eligible:
            raise NoEligibleProviderError(
                "no enabled AI providers in the registry have the"
                f" capabilities required for task={request.task.value!r}"
            )

        last_transient: TransientProviderError | None = None
        for provider in eligible:
            try:
                validated = self._run_against_provider(provider, request)
            except TransientProviderError as exc:
                last_transient = exc
                self.fallback_audit(provider, str(exc))
                continue
            proposal = TypedAIProposal(
                proposal_id=self.id_factory(),
                ts_ns=ts_ns,
                task=request.task,
                provider_id=provider.id,
                schema_name=_schema_qualname(request.schema_class),
                validated_result=validated,
            )
            self.submit_proposal(proposal)
            return proposal

        assert last_transient is not None  # eligible was non-empty
        raise AllProvidersFailedError(
            f"every eligible provider for task={request.task.value!r}"
            " raised TransientProviderError;"
            f" last reason: {last_transient}"
        ) from last_transient

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_against_provider(
        self,
        provider: AIProvider,
        request: TypedAIRequest[_ResultT],
    ) -> _ResultT:
        retry_feedback: str | None = None
        last_validation: ValidationError | None = None
        attempts = request.max_retries + 1
        for _attempt in range(attempts):
            raw = self.transport.invoke(
                provider,
                request.prompt,
                request.schema_class,
                retry_feedback=retry_feedback,
            )
            if not isinstance(raw, str):
                raise TypeError(
                    f"TypedAIAgent: transport.invoke must return str, got {type(raw).__name__}"
                )
            try:
                validated = request.schema_class.model_validate_json(raw)
            except ValidationError as exc:
                last_validation = exc
                retry_feedback = str(exc)
                continue
            return validated

        assert last_validation is not None  # attempts >= 1
        raise SchemaValidationError(
            f"provider={provider.id!r} produced output that failed schema"
            f" {request.schema_class.__name__!r} after"
            f" {attempts} attempt(s); last error: {last_validation}"
        ) from last_validation


# ---------------------------------------------------------------------------
# Production transport (lazy pydantic-ai import)
# ---------------------------------------------------------------------------


def pydantic_ai_transport() -> TypedAITransport:
    """Construct the canonical :class:`TypedAITransport` backed by
    ``pydantic-ai``'s ``Agent.run_sync``.

    Lazily imports ``pydantic_ai`` at call time so this module remains
    importable on hosts that do not have the package installed (every
    test deployment that uses an injected fake transport).

    Raises:
        RuntimeError: If ``pydantic_ai`` is not installed.
    """

    try:
        from pydantic_ai import Agent  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — exercised via test fake
        raise RuntimeError(
            "pydantic_ai_transport: pydantic-ai is not installed; install"
            " 'pydantic-ai' or pass an injected TypedAITransport to"
            " TypedAIAgent"
        ) from exc

    class _PydanticAITransport:
        def invoke(
            self,
            provider: AIProvider,
            prompt: str,
            schema_class: type[TypedAIResult],
            /,
            *,
            retry_feedback: str | None = None,
        ) -> str:
            agent = Agent(
                model=provider.endpoint,
                result_type=schema_class,
            )
            full_prompt = prompt
            if retry_feedback is not None:
                full_prompt = (
                    f"{prompt}\n\n---\nPrevious response failed schema"
                    f" validation. Validator output:\n{retry_feedback}\n"
                    "Return JSON that conforms to the declared schema."
                )
            run_result = agent.run_sync(full_prompt)
            return run_result.data.model_dump_json()

    return _PydanticAITransport()
