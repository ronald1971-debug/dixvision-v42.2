"""Pydantic response models for the ``/api/governance/*`` HTTP surface.

Until AUDIT-P2.4 these payloads were hand-defined twice — once as
plain ``dict[str, Any]`` projections in :mod:`ui.governance_routes`,
and once as ``export interface`` declarations in
``dashboard2026/src/api/governance.ts``. The two surfaces drifted
silently whenever a field was added or renamed on either side.

This module is the single source of truth. The TypeScript client
consumes the generated mirror at
``dashboard2026/src/types/generated/api.ts`` (rendered by
``tools.codegen.pydantic_to_ts`` and pinned by
``tests/test_codegen_pydantic_to_ts.py``). Adding a new governance
field is now a one-place change here, plus a regen of the .ts file.

The shapes intentionally mirror the legacy dictionaries verbatim so
the route handlers can still return ``dict[str, Any]`` (FastAPI
coerces them through these models when ``response_model`` is set).
That keeps the diff tiny and avoids touching the runtime code paths
that produce the dictionaries.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PromotionGatesResponse(BaseModel):
    """``GET /api/governance/promotion_gates`` payload."""

    model_config = ConfigDict(extra="forbid")

    path: str
    file_present: bool
    file_hash: str | None
    bound_hash: str | None
    matches: bool | None
    backend_wired: bool
    gated_targets: list[str]
    doc_url: str


class DriftComponent(BaseModel):
    """One axis of the composite drift oracle (model / exec / latency / causal)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    threshold: float
    description: str
    value: float | None = None


class DriftResponse(BaseModel):
    """``GET /api/governance/drift`` payload."""

    model_config = ConfigDict(extra="forbid")

    backend_wired: bool
    composite: float | None
    expected_components: list[DriftComponent]
    components: list[DriftComponent]
    downgrade_threshold: float


class SourceRow(BaseModel):
    """One :class:`SourceRegistry` entry augmented with runtime liveness."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    name: str
    category: str
    provider: str
    auth: str
    enabled: bool
    critical: bool
    liveness_threshold_ms: int
    status: str
    last_heartbeat_ns: int
    last_data_ns: int
    gap_ns: int


class SourcesResponse(BaseModel):
    """``GET /api/governance/sources`` payload."""

    model_config = ConfigDict(extra="forbid")

    backend_wired: bool
    registry_loaded: bool
    rows: list[SourceRow]


class HazardTaxonomyRow(BaseModel):
    """Static HAZ-* taxonomy row (always present, even with no sensors)."""

    model_config = ConfigDict(extra="forbid")

    code: str
    label: str
    description: str


class HazardEventRow(BaseModel):
    """Recent live :class:`HazardEvent` projection from the sensor array."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: str
    ts_ns: int
    source: str
    summary: str


class HazardsResponse(BaseModel):
    """``GET /api/governance/hazards`` payload."""

    model_config = ConfigDict(extra="forbid")

    backend_wired: bool
    taxonomy: list[HazardTaxonomyRow]
    recent: list[HazardEventRow]


__all__ = (
    "DriftComponent",
    "DriftResponse",
    "HazardEventRow",
    "HazardTaxonomyRow",
    "HazardsResponse",
    "PromotionGatesResponse",
    "SourceRow",
    "SourcesResponse",
)
