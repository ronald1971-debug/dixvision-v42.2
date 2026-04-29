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
    AI = "ai"
    SYNTHETIC = "synthetic"


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

    default_liveness = _DEFAULT_LIVENESS_MS_BY_CATEGORY.get(category.value, 0)
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
