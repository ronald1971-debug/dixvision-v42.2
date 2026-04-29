"""Tests for ``core.cognitive_router``.

Coverage focuses on the load-bearing behaviours:

* ``enabled_ai_providers`` returns only ``category: ai`` rows whose
  ``enabled`` is True, in registry order, with the public projection
  shape (no auth / enabled / schema fields leaked).
* ``select_providers`` filters by capability superset, not just
  intersection — a provider missing any required capability is
  excluded.
* The router is pure: two calls with the same inputs return the same
  tuple, and the result tuples are immutable.
* The shipping registry (``registry/data_source_registry.yaml``) is
  exercised by tests so a future provider added to the YAML doesn't
  silently break the schema or capability set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.cognitive_router import (
    AIProvider,
    TaskClass,
    enabled_ai_providers,
    select_providers,
)
from core.cognitive_router.task_class import (
    _REQUIREMENTS,
    required_capabilities,
)
from system_engine.scvs.source_registry import (
    ALLOWED_AI_CAPABILITIES,
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
    load_source_registry,
)

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "registry"
    / "data_source_registry.yaml"
)


def _registry(*decls: SourceDeclaration) -> SourceRegistry:
    return SourceRegistry(version="test", sources=tuple(decls))


def _ai(
    sid: str,
    *,
    enabled: bool = True,
    capabilities: tuple[str, ...] = (),
    provider: str = "vendor",
) -> SourceDeclaration:
    return SourceDeclaration(
        id=sid,
        name=sid,
        category=SourceCategory.AI,
        provider=provider,
        endpoint=f"https://api.example.com/{sid}",
        schema="sensory.cognitive.contracts.AIResponse",
        auth="required",
        enabled=enabled,
        critical=False,
        capabilities=capabilities,
    )


def _market(sid: str) -> SourceDeclaration:
    return SourceDeclaration(
        id=sid,
        name=sid,
        category=SourceCategory.MARKET,
        provider="venue",
        endpoint=f"wss://example.com/{sid}",
        schema="core.contracts.market.MarketTick",
        auth="none",
        enabled=True,
        critical=False,
    )


# ---------------------------------------------------------------------------
# enabled_ai_providers
# ---------------------------------------------------------------------------


def test_enabled_ai_providers_filters_by_category_and_enabled() -> None:
    registry = _registry(
        _ai("SRC-AI-A-001", enabled=True, capabilities=("reasoning",)),
        _ai("SRC-AI-B-002", enabled=False, capabilities=("reasoning",)),
        _market("SRC-MARKET-X-001"),
        _ai(
            "SRC-AI-C-003",
            enabled=True,
            capabilities=("code_gen", "long_context"),
        ),
    )
    out = enabled_ai_providers(registry)
    assert tuple(p.id for p in out) == ("SRC-AI-A-001", "SRC-AI-C-003")
    # Public projection — no leaked governance fields.
    p = out[0]
    assert isinstance(p, AIProvider)
    assert hasattr(p, "id")
    assert hasattr(p, "capabilities")
    assert not hasattr(p, "auth")
    assert not hasattr(p, "enabled")


def test_enabled_ai_providers_preserves_registry_order() -> None:
    registry = _registry(
        _ai("SRC-AI-Z-001", capabilities=("reasoning",)),
        _ai("SRC-AI-A-002", capabilities=("reasoning",)),
        _ai("SRC-AI-M-003", capabilities=("reasoning",)),
    )
    ids = tuple(p.id for p in enabled_ai_providers(registry))
    assert ids == ("SRC-AI-Z-001", "SRC-AI-A-002", "SRC-AI-M-003")


# ---------------------------------------------------------------------------
# select_providers — capability filtering
# ---------------------------------------------------------------------------


def test_select_providers_requires_capability_superset() -> None:
    # DYON_CODING needs ("code_gen", "long_context").
    registry = _registry(
        # Missing ``long_context`` → excluded.
        _ai("SRC-AI-CODE-ONLY-001", capabilities=("code_gen",)),
        # Both → included.
        _ai(
            "SRC-AI-CODE-LC-002",
            capabilities=("code_gen", "long_context", "tool_use"),
        ),
        # Wrong tags → excluded.
        _ai("SRC-AI-MULTIMODAL-003", capabilities=("multimodal",)),
    )
    out = select_providers(registry, TaskClass.DYON_CODING)
    assert tuple(p.id for p in out) == ("SRC-AI-CODE-LC-002",)


def test_select_providers_skips_disabled_rows() -> None:
    registry = _registry(
        _ai(
            "SRC-AI-OFF-001",
            enabled=False,
            capabilities=("reasoning", "multimodal"),
        ),
        _ai(
            "SRC-AI-ON-002",
            enabled=True,
            capabilities=("reasoning", "multimodal"),
        ),
    )
    out = select_providers(registry, TaskClass.INDIRA_MULTIMODAL_RESEARCH)
    assert tuple(p.id for p in out) == ("SRC-AI-ON-002",)


def test_select_providers_returns_empty_tuple_when_none_match() -> None:
    # Registry has only realtime_search providers; ask for a task
    # class that needs reasoning.
    registry = _registry(
        _ai("SRC-AI-RT-001", capabilities=("realtime_search",)),
    )
    out = select_providers(registry, TaskClass.INDIRA_REASONING)
    assert out == ()


def test_select_providers_for_each_task_class_uses_declared_requirements() -> None:
    # Build one provider per task class, declaring exactly the tags
    # that task class requires. Each provider must be selected by its
    # corresponding task class and only by its corresponding task
    # class (when no provider has overlapping capabilities).
    registry = _registry(
        *[
            _ai(f"SRC-AI-T-{i:03d}", capabilities=required_capabilities(t))
            for i, t in enumerate(TaskClass)
        ]
    )
    for i, t in enumerate(TaskClass):
        sid = f"SRC-AI-T-{i:03d}"
        out = select_providers(registry, t)
        # The provider with this task's exact requirement is included.
        assert sid in {p.id for p in out}, (
            f"{t.value} did not select its own minimum-capability"
            f" provider {sid}"
        )


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_select_providers_is_pure_and_deterministic() -> None:
    registry = _registry(
        _ai(
            "SRC-AI-A-001",
            capabilities=("reasoning", "multimodal", "tool_use"),
        ),
        _ai(
            "SRC-AI-B-002",
            capabilities=("reasoning", "multimodal"),
        ),
    )
    a = select_providers(registry, TaskClass.INDIRA_MULTIMODAL_RESEARCH)
    b = select_providers(registry, TaskClass.INDIRA_MULTIMODAL_RESEARCH)
    assert a == b
    # Result is a tuple — caller cannot mutate.
    assert isinstance(a, tuple)


# ---------------------------------------------------------------------------
# Live registry — sanity that the YAML stays in sync
# ---------------------------------------------------------------------------


def test_shipped_registry_loads_with_capabilities() -> None:
    """The committed ``data_source_registry.yaml`` must parse cleanly
    under the capability-extended schema."""

    registry = load_source_registry(REGISTRY_PATH)
    ai_rows = [s for s in registry.sources if s.category is SourceCategory.AI]
    assert ai_rows, "expected at least one AI row in the shipped registry"
    for row in ai_rows:
        assert set(row.capabilities).issubset(ALLOWED_AI_CAPABILITIES), (
            f"{row.id} declares unknown capabilities"
            f" {sorted(set(row.capabilities) - ALLOWED_AI_CAPABILITIES)}"
        )


def test_devin_ai_is_registered_and_capable() -> None:
    """SRC-AI-DEVIN-001 must be in the registry with the capabilities
    Dyon Chat depends on (multi-step coding / connector tasks)."""

    registry = load_source_registry(REGISTRY_PATH)
    devin = registry.by_id("SRC-AI-DEVIN-001")
    assert devin is not None, "SRC-AI-DEVIN-001 missing from registry"
    assert devin.category is SourceCategory.AI
    assert devin.provider == "cognition"
    # Dyon's primary task class requires (code_gen, long_context).
    needed = set(required_capabilities(TaskClass.DYON_CODING))
    assert needed.issubset(devin.capabilities), (
        f"SRC-AI-DEVIN-001 missing Dyon-required capabilities:"
        f" {sorted(needed - set(devin.capabilities))}"
    )


# ---------------------------------------------------------------------------
# Defensive: requirements mapping internal invariants
# ---------------------------------------------------------------------------


def test_every_task_class_has_requirements() -> None:
    for t in TaskClass:
        # Will raise KeyError if a TaskClass member is added without a
        # matching _REQUIREMENTS entry.
        assert isinstance(_REQUIREMENTS[t], tuple)


def test_required_capabilities_returns_immutable_tuple() -> None:
    caps = required_capabilities(TaskClass.INDIRA_REASONING)
    assert isinstance(caps, tuple)
    with pytest.raises((TypeError, AttributeError)):
        caps.append("extra")  # type: ignore[attr-defined]
