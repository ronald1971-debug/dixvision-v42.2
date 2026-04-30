"""Source registry loader for SCVS Phase 1.

A :class:`SourceRegistry` is the in-memory, frozen projection of
``registry/data_source_registry.yaml``. Loading is strict — any
schema violation raises :class:`ValueError` at boot so misconfigured
sources never reach the runtime tracker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

REGISTRY_VERSION = "v0.1.0"

ALLOWED_AUTH = frozenset({"none", "required"})

# SCVS Phase 2 — runtime liveness defaults. Sources without an
# explicit ``liveness_threshold_ms`` inherit the category default; the
# fallback is 30 s for everything that is not explicitly tighter or
# looser. ``synthetic`` is exempt (replay buffers don't heartbeat) and
# ``regulatory`` filings publish on a slow human cadence.
_DEFAULT_LIVENESS_MS_BY_CATEGORY: Mapping[str, int] = {
    "market": 5_000,
    "onchain": 60_000,
    "news": 5 * 60_000,
    "social": 5 * 60_000,
    "macro": 24 * 60 * 60_000,
    "regulatory": 24 * 60 * 60_000,
    "dev": 60 * 60_000,
    "alt": 5 * 60_000,
    "trader": 5 * 60_000,
    "ai": 60_000,
    "synthetic": 0,  # 0 == not liveness-checked
}


class SourceCategory(StrEnum):
    """Canonical source category taxonomy (matches the v3.5 SCVS spec)."""

    MARKET = "market"
    NEWS = "news"
    SOCIAL = "social"
    ONCHAIN = "onchain"
    MACRO = "macro"
    REGULATORY = "regulatory"
    DEV = "dev"
    ALT = "alt"
    # Wave-04 PR-2 — external trader feeds (signals + ideas + alerts).
    # Producers: ``ui.feeds.tradingview_ideas`` and future trader-feed
    # adapters. Consumers: ``intelligence_engine.trader_modeling`` (the
    # only B29-allowed runtime constructor for ``TraderObservation``).
    TRADER = "trader"
    AI = "ai"
    SYNTHETIC = "synthetic"


# Allowed capability tags for ``category: ai`` rows. Chat widgets pick
# providers per task class by intersecting the operator-requested
# capabilities with each row's declared set, so adding a capability
# here is a load-bearing change — every existing AI row's
# ``capabilities`` must still be a subset of this set.
ALLOWED_AI_CAPABILITIES: frozenset[str] = frozenset(
    {
        "reasoning",
        "code_gen",
        "multimodal",
        "realtime_search",
        "long_context",
        "tool_use",
        "agent_orchestration",
    }
)


@dataclass(frozen=True, slots=True)
class SourceDeclaration:
    """One row of ``data_source_registry.yaml``."""

    id: str
    name: str
    category: SourceCategory
    provider: str
    endpoint: str
    schema: str
    auth: str
    enabled: bool
    critical: bool
    liveness_threshold_ms: int = 0  # 0 == not liveness-checked
    # Optional capability tags. Only meaningful for ``category: ai``
    # rows where the registry-driven Cognitive Router (Dashboard-2026
    # wave-01) selects providers per task class. Always a tuple so the
    # dataclass stays hashable; empty for non-AI rows.
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Immutable view of the full source registry."""

    version: str
    sources: tuple[SourceDeclaration, ...]

    def by_id(self, source_id: str) -> SourceDeclaration | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.sources)

    @property
    def enabled_ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.sources if s.enabled)


def _require(mapping: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{ctx}: missing required field '{key}'")
    return mapping[key]


def _parse_source(raw: Any, idx: int) -> SourceDeclaration:
    if not isinstance(raw, Mapping):
        raise ValueError(f"sources[{idx}] is not a mapping")

    ctx = f"sources[{idx}]"
    sid = str(_require(raw, "id", ctx))
    if not sid.startswith("SRC-"):
        raise ValueError(f"{ctx}: id '{sid}' must start with 'SRC-'")

    category_raw = str(_require(raw, "category", ctx))
    try:
        category = SourceCategory(category_raw)
    except ValueError as exc:
        raise ValueError(
            f"{ctx}: category '{category_raw}' is not a SourceCategory"
        ) from exc

    auth = str(_require(raw, "auth", ctx))
    if auth not in ALLOWED_AUTH:
        raise ValueError(
            f"{ctx}: auth '{auth}' must be one of {sorted(ALLOWED_AUTH)}"
        )

    default_liveness = _DEFAULT_LIVENESS_MS_BY_CATEGORY.get(category.value, 30_000)
    liveness_raw = raw.get("liveness_threshold_ms", default_liveness)
    try:
        liveness_ms = int(liveness_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ctx}: liveness_threshold_ms must be an integer (got {liveness_raw!r})"
        ) from exc
    if liveness_ms < 0:
        raise ValueError(
            f"{ctx}: liveness_threshold_ms must be >= 0 (got {liveness_ms})"
        )

    capabilities_raw = raw.get("capabilities", ())
    if capabilities_raw is None:
        capabilities_tuple: tuple[str, ...] = ()
    elif isinstance(capabilities_raw, (list, tuple)):
        seen: set[str] = set()
        cap_list: list[str] = []
        for cap in capabilities_raw:
            if not isinstance(cap, str):
                raise ValueError(
                    f"{ctx}: capabilities entries must be strings"
                    f" (got {type(cap).__name__})"
                )
            if cap in seen:
                raise ValueError(
                    f"{ctx}: capability '{cap}' listed more than once"
                )
            seen.add(cap)
            cap_list.append(cap)
        capabilities_tuple = tuple(cap_list)
    else:
        raise ValueError(
            f"{ctx}: capabilities must be a list (got {type(capabilities_raw).__name__})"
        )
    if capabilities_tuple and category is not SourceCategory.AI:
        raise ValueError(
            f"{ctx}: capabilities only valid on category=ai (got '{category.value}')"
        )
    if capabilities_tuple:
        unknown = set(capabilities_tuple) - ALLOWED_AI_CAPABILITIES
        if unknown:
            raise ValueError(
                f"{ctx}: unknown AI capabilities {sorted(unknown)};"
                f" allowed = {sorted(ALLOWED_AI_CAPABILITIES)}"
            )

    return SourceDeclaration(
        id=sid,
        name=str(_require(raw, "name", ctx)),
        category=category,
        provider=str(_require(raw, "provider", ctx)),
        endpoint=str(_require(raw, "endpoint", ctx)),
        schema=str(_require(raw, "schema", ctx)),
        auth=auth,
        enabled=bool(raw.get("enabled", False)),
        critical=bool(raw.get("critical", False)),
        liveness_threshold_ms=liveness_ms,
        capabilities=capabilities_tuple,
    )


def load_source_registry(path: str | Path) -> SourceRegistry:
    """Load + strictly validate the source registry YAML."""

    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"source registry at {path} is not a mapping")

    version = str(raw.get("version", ""))
    if not version:
        raise ValueError("source registry: missing 'version'")

    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list):
        raise ValueError("source registry: 'sources' must be a list")

    sources: list[SourceDeclaration] = []
    seen: set[str] = set()
    for idx, row in enumerate(sources_raw):
        decl = _parse_source(row, idx)
        if decl.id in seen:
            raise ValueError(f"source registry: duplicate id {decl.id!r}")
        seen.add(decl.id)
        sources.append(decl)

    return SourceRegistry(version=version, sources=tuple(sources))
