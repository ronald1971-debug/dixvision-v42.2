"""AI provider response validator for SCVS Phase 3 (rule **SCVS-07**).

External AI models (ChatGPT / Gemini / Grok / DeepSeek / Devin etc.)
are first-class sources in the v3.5 registry — category ``ai``. They
must be validated like any other source, plus three AI-specific
checks per the SCVS spec:

* response latency below a per-source threshold
* response structurally valid (non-empty, decodable)
* output non-empty (no whitespace-only "fallback" placeholder)

The validator is pure / deterministic and does **not** call any
provider — it inspects an already-collected response. No network, no
clock (caller supplies ``latency_ns``), no PRNG.

Critical AI sources that fail validation must escalate via the same
``HAZ-13`` seam as other critical sources (Phase 2 SCVS-06). The
validator emits the hazard alongside the outcome so callers don't
need to re-implement the escalation rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from core.contracts.events import HazardEvent, HazardSeverity
from system_engine.scvs.source_manager import HAZ_CRITICAL_SOURCE_STALE
from system_engine.scvs.source_registry import SourceRegistry

SOURCE = "system_engine.scvs.ai_validator"


class AIOutcome(StrEnum):
    """SCVS-07 AI validation outcome."""

    ACCEPTED = "ACCEPTED"
    REJECTED_NOT_AI = "REJECTED_NOT_AI"  # source is not category=ai
    REJECTED_UNKNOWN_SOURCE = "REJECTED_UNKNOWN_SOURCE"
    REJECTED_DISABLED_SOURCE = "REJECTED_DISABLED_SOURCE"
    REJECTED_LATENCY = "REJECTED_LATENCY"
    REJECTED_EMPTY = "REJECTED_EMPTY"
    REJECTED_STRUCTURE = "REJECTED_STRUCTURE"


@dataclass(frozen=True, slots=True)
class AIValidationResult:
    """One immutable AI provider validation outcome."""

    source_id: str
    outcome: AIOutcome
    detail: str = ""


@dataclass(slots=True)
class AIValidator:
    """Pure validator for AI provider responses (SCVS-07)."""

    registry: SourceRegistry
    max_latency_ns: int  # any latency above this fails the call
    required_top_keys: frozenset[str] = frozenset()  # structural minimum

    def __post_init__(self) -> None:
        if self.max_latency_ns <= 0:
            raise ValueError("max_latency_ns must be > 0")

    def validate(
        self,
        *,
        source_id: str,
        response: Mapping[str, object] | None,
        latency_ns: int,
        now_ns: int,
    ) -> tuple[AIValidationResult, tuple[HazardEvent, ...]]:
        decl = self.registry.by_id(source_id)
        if decl is None:
            return (
                AIValidationResult(
                    source_id=source_id,
                    outcome=AIOutcome.REJECTED_UNKNOWN_SOURCE,
                    detail=f"unknown source_id: {source_id!r}",
                ),
                (),
            )
        if decl.category.value != "ai":
            return (
                AIValidationResult(
                    source_id=source_id,
                    outcome=AIOutcome.REJECTED_NOT_AI,
                    detail=f"source {source_id!r} is category={decl.category.value}, not 'ai'",
                ),
                (),
            )
        if not decl.enabled:
            return (
                AIValidationResult(
                    source_id=source_id,
                    outcome=AIOutcome.REJECTED_DISABLED_SOURCE,
                    detail=f"source {source_id!r} has enabled=false",
                ),
                (),
            )

        outcome = _classify(
            response=response,
            latency_ns=latency_ns,
            max_latency_ns=self.max_latency_ns,
            required_top_keys=self.required_top_keys,
        )
        result = AIValidationResult(
            source_id=source_id,
            outcome=outcome.outcome,
            detail=outcome.detail,
        )
        if result.outcome == AIOutcome.ACCEPTED or not decl.critical:
            return result, ()
        # SCVS-06 escalation seam — critical AI source failed.
        hazard = HazardEvent(
            ts_ns=now_ns,
            code=HAZ_CRITICAL_SOURCE_STALE,
            severity=HazardSeverity.HIGH,
            source=SOURCE,
            detail=f"critical AI source '{decl.id}' failed: {result.detail}",
            meta={
                "source_id": decl.id,
                "outcome": result.outcome.value,
                "category": decl.category.value,
            },
        )
        return result, (hazard,)


@dataclass(frozen=True, slots=True)
class _Classification:
    outcome: AIOutcome
    detail: str = ""


def _classify(
    *,
    response: Mapping[str, object] | None,
    latency_ns: int,
    max_latency_ns: int,
    required_top_keys: frozenset[str],
) -> _Classification:
    if latency_ns > max_latency_ns:
        return _Classification(
            outcome=AIOutcome.REJECTED_LATENCY,
            detail=f"latency_ns={latency_ns} > max_latency_ns={max_latency_ns}",
        )
    if response is None or not response:
        return _Classification(
            outcome=AIOutcome.REJECTED_EMPTY,
            detail="response is None or empty",
        )
    missing = sorted(k for k in required_top_keys if k not in response)
    if missing:
        return _Classification(
            outcome=AIOutcome.REJECTED_STRUCTURE,
            detail=f"missing required top-level keys: {missing}",
        )
    # Whitespace-only string fields are treated as silent-empty fallback
    # placeholders — the model likely returned a degenerate completion.
    for key, value in response.items():
        if isinstance(value, str) and value.strip() == "":
            return _Classification(
                outcome=AIOutcome.REJECTED_EMPTY,
                detail=f"key {key!r} is whitespace-only",
            )
    return _Classification(outcome=AIOutcome.ACCEPTED)


__all__ = [
    "AIOutcome",
    "AIValidationResult",
    "AIValidator",
]
