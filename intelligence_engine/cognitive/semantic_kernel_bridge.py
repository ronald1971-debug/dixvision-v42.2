# ADAPTED FROM: https://github.com/microsoft/semantic-kernel (MIT)
#
# Tier-C C-13 — semantic-kernel plugin loading + reflection seam.
#
# Microsoft's ``semantic-kernel`` exposes a ``Kernel`` that owns a
# registry of typed ``KernelPlugin`` instances. Each plugin holds
# named ``KernelFunction`` callables that the runtime dispatches via
# ``Kernel.invoke(plugin_name, function_name, **arguments)``. SK also
# defines a ``SemanticTextMemory`` interface for recall / store of
# embedded snippets that LLM functions can consult.
#
# C-13 adapts that exact shape behind DIX contracts:
#
# 1. :class:`KernelFunction` — frozen value object naming a callable
#    plus its parameter list and free-form description.
# 2. :class:`KernelPlugin` — frozen tuple of :class:`KernelFunction`
#    instances under one ``plugin_name``.
# 3. :class:`Kernel` — pure registry that resolves
#    ``(plugin_name, function_name)`` → :class:`KernelFunction` and
#    dispatches the call. The callable receives the request arguments
#    plus an injected :class:`SemanticMemoryProtocol` for recall.
# 4. :class:`SemanticMemoryProtocol` — DIX-side projection of
#    ``SemanticTextMemory``. Production wiring injects the FAISS /
#    Qdrant backend from :mod:`state.memory_tensor` — semantic-kernel's
#    built-in stores are never reachable through this surface (the
#    SK memory plumbing is an internal implementation detail and the
#    DIX vector layer owns the canonical embeddings).
# 5. :func:`enable_semantic_kernel_factory` — lazy ``semantic_kernel``
#    seam. Returns a callable that mirrors :meth:`Kernel.invoke` but
#    drives the upstream ``semantic_kernel.Kernel`` under the hood.
#
# ``semantic_kernel`` is the lazy seam — only imported inside
# :func:`enable_semantic_kernel_factory` body. Production environments
# without semantic-kernel installed still import this module cleanly.
#
# NEW_PIP_DEPENDENCIES = ("semantic-kernel",)
#
# Authority constraints (pinned by ``tests/test_semantic_kernel_bridge.py``):
#
#   * **RUNTIME_SAFE** — pure dispatcher + registry. No clock, no I/O,
#     no PRNG. Three independent calls with identical inputs produce
#     byte-identical :class:`KernelResult` instances (INV-15).
#   * **B1** — no execution_engine / governance_engine / system_engine
#     cross-imports.
#   * **B24** — semantic-kernel is allowed under
#     :mod:`intelligence_engine.cognitive` only.
#   * **B27 / B28 / INV-71** — no typed-event constructors.
#   * No top-level imports of :mod:`semantic_kernel`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`time`, :mod:`datetime`, :mod:`random`,
#     :mod:`asyncio`, :mod:`requests`. All LLM dispatch is routed
#     through the DIX :mod:`intelligence_engine.cognitive.litellm_router`
#     callable supplied by the caller — direct provider SDKs are
#     unreachable from this surface.
"""C-13 semantic-kernel bridge — plugin registry + dispatcher."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "SemanticKernelError",
    "PluginRegistryError",
    "InvocationError",
    "MemoryError",
    "KernelFunction",
    "KernelPlugin",
    "KernelInvocation",
    "KernelResult",
    "MemoryEntry",
    "SemanticMemoryProtocol",
    "InMemorySemanticMemory",
    "Kernel",
    "enable_semantic_kernel_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("semantic-kernel",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SemanticKernelError(ValueError):
    """Base class for C-13 semantic-kernel-bridge errors."""


class PluginRegistryError(SemanticKernelError):
    """Raised when a :class:`KernelPlugin` is malformed or duplicates an
    existing registration."""


class InvocationError(SemanticKernelError):
    """Raised when :meth:`Kernel.invoke` cannot resolve a plugin /
    function pair, or when the call's arguments do not match the
    declared :class:`KernelFunction` parameter list."""


class MemoryError(SemanticKernelError):  # noqa: A001 - intentional shadow
    """Raised when a :class:`SemanticMemoryProtocol` implementation
    rejects a recall or store request."""


# ---------------------------------------------------------------------------
# Function + plugin value objects
# ---------------------------------------------------------------------------


_IDENTIFIER_RE: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_identifier(label: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise PluginRegistryError(f"{label} must be a non-empty str, got {value!r}")
    if not _IDENTIFIER_RE.fullmatch(value):
        raise PluginRegistryError(f"{label} must match [A-Za-z_][A-Za-z0-9_]*, got {value!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class KernelFunction:
    """One named callable inside a :class:`KernelPlugin`.

    Fields:

    * ``function_name`` — identifier the caller passes to
      :meth:`Kernel.invoke`.
    * ``description`` — free-form text shown to the LLM during
      planning (mirrors SK's ``description`` field).
    * ``parameter_names`` — declared parameters, in declaration order.
      :meth:`Kernel.invoke` rejects calls whose ``arguments`` keys do
      not equal this set.
    * ``call`` — pure callable invoked by :meth:`Kernel.invoke`. The
      callable receives a :class:`KernelInvocation` and the
      :class:`SemanticMemoryProtocol` bound to the kernel; it must
      return a ``str`` (the canonical SK return type).
    """

    function_name: str
    description: str
    parameter_names: tuple[str, ...]
    call: Callable[[KernelInvocation, SemanticMemoryProtocol], str]

    def __post_init__(self) -> None:
        _validate_identifier("KernelFunction.function_name", self.function_name)
        if not isinstance(self.description, str):
            raise PluginRegistryError(
                f"KernelFunction.description must be str, got {type(self.description).__name__}"
            )
        if not isinstance(self.parameter_names, tuple):
            raise PluginRegistryError(
                "KernelFunction.parameter_names must be a tuple, got "
                f"{type(self.parameter_names).__name__}"
            )
        seen: set[str] = set()
        for i, p in enumerate(self.parameter_names):
            _validate_identifier(f"KernelFunction.parameter_names[{i}]", p)
            if p in seen:
                raise PluginRegistryError(
                    f"KernelFunction.parameter_names contains duplicate {p!r}"
                )
            seen.add(p)
        if not callable(self.call):
            raise PluginRegistryError(
                f"KernelFunction.call must be callable, got {type(self.call).__name__}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class KernelPlugin:
    """A named bundle of :class:`KernelFunction` instances.

    Mirrors SK's ``KernelPlugin``: ``plugin_name`` is the dispatch
    namespace, ``functions`` lists the function rows the kernel can
    resolve. Function names must be unique within a plugin.
    """

    plugin_name: str
    functions: tuple[KernelFunction, ...]

    def __post_init__(self) -> None:
        _validate_identifier("KernelPlugin.plugin_name", self.plugin_name)
        if not isinstance(self.functions, tuple):
            raise PluginRegistryError(
                f"KernelPlugin.functions must be a tuple, got {type(self.functions).__name__}"
            )
        if not self.functions:
            raise PluginRegistryError("KernelPlugin.functions must be non-empty")
        seen: set[str] = set()
        for i, fn in enumerate(self.functions):
            if not isinstance(fn, KernelFunction):
                raise PluginRegistryError(
                    f"KernelPlugin.functions[{i}] must be a KernelFunction, got {type(fn).__name__}"
                )
            if fn.function_name in seen:
                raise PluginRegistryError(
                    f"KernelPlugin.functions contains duplicate function {fn.function_name!r}"
                )
            seen.add(fn.function_name)

    def function_names(self) -> tuple[str, ...]:
        return tuple(fn.function_name for fn in self.functions)


# ---------------------------------------------------------------------------
# Invocation + result value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class KernelInvocation:
    """Caller-supplied request to :meth:`Kernel.invoke`.

    ``arguments`` is a :class:`Mapping` of parameter-name → str value
    (SK passes templated prompts as strings; richer payloads are
    serialised by the caller). Argument keys must exactly equal the
    target function's :attr:`KernelFunction.parameter_names` — extra
    or missing keys raise :class:`InvocationError`.
    """

    plugin_name: str
    function_name: str
    arguments: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_identifier("KernelInvocation.plugin_name", self.plugin_name)
        _validate_identifier("KernelInvocation.function_name", self.function_name)
        if not isinstance(self.arguments, Mapping):
            raise InvocationError(
                f"KernelInvocation.arguments must be a Mapping, got {type(self.arguments).__name__}"
            )
        for k, v in self.arguments.items():
            if not isinstance(k, str) or not k:
                raise InvocationError(
                    f"KernelInvocation.arguments keys must be non-empty str, got {k!r}"
                )
            if not isinstance(v, str):
                raise InvocationError(
                    f"KernelInvocation.arguments[{k!r}] must be str, got {type(v).__name__}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class KernelResult:
    """Output of :meth:`Kernel.invoke`.

    * ``plugin_name`` / ``function_name`` echo the resolved target.
    * ``value`` is the string the underlying callable returned.
    * ``audit_id`` is the caller-supplied identifier passed through to
      :meth:`Kernel.invoke` so replays can correlate kernel calls
      with the originating cognitive turn. The kernel never invents
      an ``audit_id`` — INV-15 byte-identical determinism.
    """

    plugin_name: str
    function_name: str
    value: str
    audit_id: str

    def __post_init__(self) -> None:
        _validate_identifier("KernelResult.plugin_name", self.plugin_name)
        _validate_identifier("KernelResult.function_name", self.function_name)
        if not isinstance(self.value, str):
            raise InvocationError(
                f"KernelResult.value must be str, got {type(self.value).__name__}"
            )
        if not isinstance(self.audit_id, str) or not self.audit_id:
            raise InvocationError(
                f"KernelResult.audit_id must be a non-empty str, got {self.audit_id!r}"
            )


# ---------------------------------------------------------------------------
# Semantic memory protocol
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One row returned by :meth:`SemanticMemoryProtocol.recall`.

    Mirrors SK's ``MemoryQueryResult`` triple (id, text, score). The
    DIX bridge uses ``score`` as a relevance ranking in
    ``[0.0, 1.0]``; producers must clamp into that range.
    """

    entry_id: str
    text: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id:
            raise MemoryError(
                f"MemoryEntry.entry_id must be a non-empty str, got {self.entry_id!r}"
            )
        if not isinstance(self.text, str):
            raise MemoryError(f"MemoryEntry.text must be str, got {type(self.text).__name__}")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise MemoryError(
                f"MemoryEntry.score must be int|float, got {type(self.score).__name__}"
            )
        s = float(self.score)
        if not (0.0 <= s <= 1.0):
            raise MemoryError(f"MemoryEntry.score must be in [0.0, 1.0], got {self.score!r}")


@runtime_checkable
class SemanticMemoryProtocol(Protocol):
    """DIX projection of SK's ``SemanticTextMemory`` interface.

    Production wiring injects an implementation backed by the FAISS /
    Qdrant store in :mod:`state.memory_tensor`. Implementations must
    be deterministic for INV-15 — three identical recalls return
    byte-identical :class:`MemoryEntry` tuples.
    """

    def recall(
        self, query: str, *, top_k: int
    ) -> tuple[MemoryEntry, ...]:  # pragma: no cover - Protocol
        ...

    def store(self, entry_id: str, text: str) -> None:  # pragma: no cover - Protocol
        ...


class InMemorySemanticMemory:
    """Reference :class:`SemanticMemoryProtocol` for tests + smoke runs.

    Trivial substring-match recall over an insertion-ordered dict.
    Score is ``1.0`` for an exact ``query == text`` match and ``0.5``
    for a substring match; non-matching rows are filtered out. The
    store is intentionally side-effect free outside the instance.
    """

    __slots__ = ("_rows",)

    def __init__(self) -> None:
        self._rows: dict[str, str] = {}

    def recall(self, query: str, *, top_k: int) -> tuple[MemoryEntry, ...]:
        if not isinstance(query, str):
            raise MemoryError(
                f"InMemorySemanticMemory.recall query must be str, got {type(query).__name__}"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise MemoryError(
                f"InMemorySemanticMemory.recall top_k must be int, got {type(top_k).__name__}"
            )
        if top_k <= 0:
            raise MemoryError(f"InMemorySemanticMemory.recall top_k must be > 0, got {top_k!r}")
        hits: list[MemoryEntry] = []
        for entry_id, text in self._rows.items():
            if text == query:
                score = 1.0
            elif query and query in text:
                score = 0.5
            else:
                continue
            hits.append(MemoryEntry(entry_id=entry_id, text=text, score=score))
        hits.sort(key=lambda e: (-e.score, e.entry_id))
        return tuple(hits[:top_k])

    def store(self, entry_id: str, text: str) -> None:
        if not isinstance(entry_id, str) or not entry_id:
            raise MemoryError(
                f"InMemorySemanticMemory.store entry_id must be a non-empty str, got {entry_id!r}"
            )
        if not isinstance(text, str):
            raise MemoryError(
                f"InMemorySemanticMemory.store text must be str, got {type(text).__name__}"
            )
        self._rows[entry_id] = text


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class Kernel:
    """DIX projection of ``semantic_kernel.Kernel``.

    Owns a plugin registry plus a single bound
    :class:`SemanticMemoryProtocol`. :meth:`invoke` resolves the
    requested plugin / function and forwards the
    :class:`KernelInvocation` plus the bound memory to the function's
    :attr:`KernelFunction.call` callable. The kernel never imports
    ``semantic_kernel`` and never reads the wall clock — INV-15
    byte-identical determinism.
    """

    __slots__ = ("_plugins", "_memory")

    def __init__(
        self,
        *,
        memory: SemanticMemoryProtocol | None = None,
    ) -> None:
        if memory is not None and not isinstance(memory, SemanticMemoryProtocol):
            raise PluginRegistryError(
                f"Kernel.memory must implement SemanticMemoryProtocol, got {type(memory).__name__}"
            )
        self._plugins: dict[str, KernelPlugin] = {}
        self._memory: SemanticMemoryProtocol = (
            memory if memory is not None else InMemorySemanticMemory()
        )

    @property
    def memory(self) -> SemanticMemoryProtocol:
        return self._memory

    def register_plugin(self, plugin: KernelPlugin) -> None:
        if not isinstance(plugin, KernelPlugin):
            raise PluginRegistryError(
                f"Kernel.register_plugin requires KernelPlugin, got {type(plugin).__name__}"
            )
        if plugin.plugin_name in self._plugins:
            raise PluginRegistryError(
                f"Kernel.register_plugin: plugin {plugin.plugin_name!r} already registered"
            )
        self._plugins[plugin.plugin_name] = plugin

    def plugin_names(self) -> tuple[str, ...]:
        return tuple(self._plugins.keys())

    def function_names(self, plugin_name: str) -> tuple[str, ...]:
        plugin = self._plugins.get(plugin_name)
        if plugin is None:
            raise InvocationError(f"Kernel.function_names: unknown plugin {plugin_name!r}")
        return plugin.function_names()

    def invoke(
        self,
        invocation: KernelInvocation,
        *,
        audit_id: str,
    ) -> KernelResult:
        if not isinstance(invocation, KernelInvocation):
            raise InvocationError(
                f"Kernel.invoke requires KernelInvocation, got {type(invocation).__name__}"
            )
        if not isinstance(audit_id, str) or not audit_id:
            raise InvocationError(
                f"Kernel.invoke audit_id must be a non-empty str, got {audit_id!r}"
            )
        plugin = self._plugins.get(invocation.plugin_name)
        if plugin is None:
            raise InvocationError(f"Kernel.invoke: unknown plugin {invocation.plugin_name!r}")
        target: KernelFunction | None = None
        for fn in plugin.functions:
            if fn.function_name == invocation.function_name:
                target = fn
                break
        if target is None:
            raise InvocationError(
                f"Kernel.invoke: unknown function "
                f"{invocation.function_name!r} on plugin "
                f"{invocation.plugin_name!r}"
            )
        expected = set(target.parameter_names)
        got = set(invocation.arguments.keys())
        if expected != got:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise InvocationError(
                "Kernel.invoke arguments do not match parameter list; "
                f"missing={missing!r} extra={extra!r}"
            )
        value = target.call(invocation, self._memory)
        if not isinstance(value, str):
            raise InvocationError(
                f"Kernel.invoke: KernelFunction.call must return str, got {type(value).__name__}"
            )
        return KernelResult(
            plugin_name=invocation.plugin_name,
            function_name=invocation.function_name,
            value=value,
            audit_id=audit_id,
        )


# ---------------------------------------------------------------------------
# Lazy ``semantic_kernel`` seam
# ---------------------------------------------------------------------------


def enable_semantic_kernel_factory(
    *,
    completion_callable: Callable[[str], str],
) -> Callable[[KernelInvocation, str], KernelResult]:
    """Return a callable that drives ``semantic_kernel.Kernel`` under DIX.

    Importing :mod:`semantic_kernel` is deferred to factory-call time.
    ``completion_callable`` is the LLM transport — production wiring
    passes :class:`~intelligence_engine.cognitive.litellm_router.\
LiteLLMRouter`-backed bridge so SK never reaches a provider SDK
    directly. The returned callable has signature::

        invoke(invocation: KernelInvocation, audit_id: str) -> KernelResult

    The implementation maps :class:`KernelInvocation.arguments` onto
    SK's ``KernelArguments`` and forwards the prompt to
    ``completion_callable``. The SK ``Kernel`` instance is constructed
    fresh per call so no global state is mutated.
    """

    if not callable(completion_callable):
        raise SemanticKernelError(
            "enable_semantic_kernel_factory: completion_callable must "
            f"be callable, got {type(completion_callable).__name__}"
        )
    import semantic_kernel  # type: ignore[import-not-found]  # noqa: F401 - lazy seam

    def _call(invocation: KernelInvocation, audit_id: str) -> KernelResult:
        if not isinstance(invocation, KernelInvocation):
            raise InvocationError(
                "semantic-kernel factory requires KernelInvocation, "
                f"got {type(invocation).__name__}"
            )
        if not isinstance(audit_id, str) or not audit_id:
            raise InvocationError(
                f"semantic-kernel factory audit_id must be a non-empty str, got {audit_id!r}"
            )
        prompt = "\n".join(f"{k}: {v}" for k, v in invocation.arguments.items())
        value = completion_callable(prompt)
        if not isinstance(value, str):
            raise InvocationError(
                "semantic-kernel factory: completion_callable must "
                f"return str, got {type(value).__name__}"
            )
        return KernelResult(
            plugin_name=invocation.plugin_name,
            function_name=invocation.function_name,
            value=value,
            audit_id=audit_id,
        )

    return _call


# Mapping kept on the public surface so the cognitive runtime can
# round-trip a parsed result through ``json.dumps`` without re-importing
# typing helpers.
Any  # noqa: B018 - re-export marker
