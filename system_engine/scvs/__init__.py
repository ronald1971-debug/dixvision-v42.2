"""SCVS — Source & Consumption Validation System.

Phase 1 surface — bidirectional closure (BUILD-FAIL):

* :mod:`system_engine.scvs.source_registry` — strict loader + schema for
  ``registry/data_source_registry.yaml``.
* :mod:`system_engine.scvs.consumption_tracker` — strict loader for
  per-module ``consumes.yaml`` declarations.
* :mod:`system_engine.scvs.lint` — pure SCVS-01 / SCVS-02 validator.

Phase 2 surface — runtime liveness FSM:

* :mod:`system_engine.scvs.source_manager` — pure FSM that classifies
  registered sources as UNKNOWN / LIVE / STALE based on caller-supplied
  heartbeats; emits transition events + ``HAZ-13`` for critical sources.

Phase 3 surface — per-packet validation + silent-fallback audit:

* :mod:`system_engine.scvs.schema_guard` — SCVS-04 schema enforcement +
  SCVS-09 stale-data rejection.
* :mod:`system_engine.scvs.ai_validator` — SCVS-07 AI provider response
  validation (latency / structure / empty-output) with critical-source
  HAZ-13 escalation.
* :mod:`system_engine.scvs.lint.find_redundant_sources` — SCVS-08
  duplicate-source WARN (non-fatal).
* :mod:`system_engine.scvs.fallback_audit` — SCVS-10 silent-fallback
  audit emitter.
"""

from system_engine.scvs.ai_validator import (
    AIOutcome,
    AIValidationResult,
    AIValidator,
)
from system_engine.scvs.consumption_tracker import (
    ConsumptionDeclaration,
    ConsumptionInput,
    discover_consumption_declarations,
    load_consumption_declaration,
)
from system_engine.scvs.fallback_audit import make_fallback_event
from system_engine.scvs.lint import (
    SCVSViolation,
    SCVSWarning,
    find_redundant_sources,
    validate_scvs,
)
from system_engine.scvs.schema_guard import (
    ContractRegistry,
    SchemaGuard,
    SchemaSpec,
    ValidationOutcome,
    ValidationResult,
)
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
    "AIOutcome",
    "AIValidationResult",
    "AIValidator",
    "ConsumptionDeclaration",
    "ConsumptionInput",
    "ContractRegistry",
    "HAZ_CRITICAL_SOURCE_STALE",
    "SCVSViolation",
    "SCVSWarning",
    "SchemaGuard",
    "SchemaSpec",
    "SourceCategory",
    "SourceDeclaration",
    "SourceLivenessReport",
    "SourceManager",
    "SourceRegistry",
    "SourceStatus",
    "ValidationOutcome",
    "ValidationResult",
    "discover_consumption_declarations",
    "find_redundant_sources",
    "load_consumption_declaration",
    "load_source_registry",
    "make_fallback_event",
    "validate_scvs",
]
