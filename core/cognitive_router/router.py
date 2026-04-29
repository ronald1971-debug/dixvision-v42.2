"""Pure, registry-driven AI provider selection (Cognitive Router).

The router is a *projection*: given a :class:`SourceRegistry` and a
:class:`TaskClass`, return the ordered tuple of providers (registry
rows) eligible for the task. No I/O, no clock, no global state — two
calls with the same registry + task class always return the same
tuple, byte-for-byte (INV-15 determinism).

The router never *names* a provider. It only filters by:

1. ``category == ai``
2. ``enabled is True``
3. ``capabilities`` superset of the task's required capabilities

Selection order
---------------
Providers are returned in the order they appear in the registry YAML.
Operators control fallback order by reordering the YAML — there is no
hidden ranking. When the registry-aware transport (wave-02) calls
providers, it walks this tuple front-to-back and records a
``SOURCE_FALLBACK_ACTIVATED`` audit event for every retry (SCVS-10).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.cognitive_router.task_class import TaskClass, required_capabilities
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
)


@dataclass(frozen=True, slots=True)
class AIProvider:
    """Public projection of a registry AI row.

    Distinct from :class:`SourceDeclaration` because the router and
    chat widgets only need the AI-relevant subset, and the public
    projection deliberately omits ``schema`` / ``auth`` / ``enabled``
    / ``critical`` / ``liveness_threshold_ms`` — those are governance
    concerns, not chat-widget concerns.
    """

    id: str
    name: str
    provider: str
    endpoint: str
    capabilities: tuple[str, ...]


def _to_public(decl: SourceDeclaration) -> AIProvider:
    return AIProvider(
        id=decl.id,
        name=decl.name,
        provider=decl.provider,
        endpoint=decl.endpoint,
        capabilities=decl.capabilities,
    )


def enabled_ai_providers(
    registry: SourceRegistry,
) -> tuple[AIProvider, ...]:
    """Return every enabled ``category: ai`` row as a public projection.

    Order matches the YAML row order. Used by ``GET /api/ai/providers``
    and by the chat widget dropdowns to populate the provider list.
    """

    return tuple(
        _to_public(s)
        for s in registry.sources
        if s.category is SourceCategory.AI and s.enabled
    )


def select_providers(
    registry: SourceRegistry,
    task: TaskClass,
) -> tuple[AIProvider, ...]:
    """Return enabled AI providers eligible for ``task``.

    A provider is eligible iff its declared ``capabilities`` is a
    superset of the task's required capabilities (see
    :func:`required_capabilities`). This is intentionally a strict
    superset check — a provider that "almost" supports a task class
    (missing one tag) is excluded, which forces operators to either
    declare the capability explicitly (the registry is the single
    source of truth) or to add a different task class for the
    weaker shape.

    The function is pure: no clock, no I/O, no global state.
    """

    needed = frozenset(required_capabilities(task))
    return tuple(
        p
        for p in enabled_ai_providers(registry)
        if needed.issubset(p.capabilities)
    )


__all__ = [
    "AIProvider",
    "enabled_ai_providers",
    "select_providers",
]
