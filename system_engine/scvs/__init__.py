"""SCVS — Source & Consumption Validation System.

Phase 1 surface (backend-only):

* :mod:`system_engine.scvs.source_registry` — strict loader + schema for
  ``registry/data_source_registry.yaml``.
* :mod:`system_engine.scvs.consumption_tracker` — strict loader for
  per-module ``consumes.yaml`` declarations + bidirectional closure
  against the source registry (SCVS-01 / SCVS-02).
* :mod:`system_engine.scvs.lint` — pure validator that returns the
  set of violations; the CI entry point (``tools/scvs_lint.py``) raises
  on any non-empty result.

Runtime liveness, schema enforcement, and hazard escalation are
intentionally deferred to SCVS Phase 2 + Phase 3 — see
``docs/manifest_v3.5_delta.md``.
"""

from system_engine.scvs.consumption_tracker import (
    ConsumptionDeclaration,
    ConsumptionInput,
    discover_consumption_declarations,
    load_consumption_declaration,
)
from system_engine.scvs.lint import SCVSViolation, validate_scvs
from system_engine.scvs.source_manager import (
    HAZ_CRITICAL_SOURCE_STALE,
    SourceLivenessReport,
    SourceManager,
    SourceStatus,
)
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
    load_source_registry,
)

__all__ = [
    "HAZ_CRITICAL_SOURCE_STALE",
    "ConsumptionDeclaration",
    "ConsumptionInput",
    "SCVSViolation",
    "SourceCategory",
    "SourceDeclaration",
    "SourceLivenessReport",
    "SourceManager",
    "SourceRegistry",
    "SourceStatus",
    "discover_consumption_declarations",
    "load_consumption_declaration",
    "load_source_registry",
    "validate_scvs",
]
