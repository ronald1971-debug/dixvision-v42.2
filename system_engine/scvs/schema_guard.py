"""Per-packet schema + staleness guard for SCVS Phase 3.

Implements two of the five Phase 3 rules:

* **SCVS-04** — schema enforcement. Every packet emitted by a
  registered source must satisfy the contract declared in
  ``data_source_registry.yaml`` (column ``schema``). The contract is
  resolved through an injected :class:`ContractRegistry`; the guard is
  pure / deterministic and never imports user contracts directly.

* **SCVS-09** — stale-data rejection. Even from a *live* source (per
  the Phase 2 heartbeat FSM), a packet whose ``ts_ns`` lags
  ``now_ns`` by more than the guard's ``max_age_ns`` is rejected.
  This is the per-packet complement to the Phase 2 per-source heartbeat
  check.

INV-15 — pure / deterministic. The guard owns no clock and no PRNG;
``now_ns`` is always caller-supplied. The same input timeline always
produces the same outcome sequence.

Phase 3 covers SCVS-04, SCVS-07, SCVS-08, SCVS-09, SCVS-10. Each rule
ships in its own module so the dependency graph stays sharp.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from system_engine.scvs.source_registry import SourceDeclaration, SourceRegistry


class ValidationOutcome(StrEnum):
    """Per-packet decision returned by :meth:`SchemaGuard.validate`."""

    ACCEPTED = "ACCEPTED"
    REJECTED_UNKNOWN_SOURCE = "REJECTED_UNKNOWN_SOURCE"
    REJECTED_DISABLED_SOURCE = "REJECTED_DISABLED_SOURCE"
    REJECTED_UNKNOWN_SCHEMA = "REJECTED_UNKNOWN_SCHEMA"
    REJECTED_SCHEMA_MISMATCH = "REJECTED_SCHEMA_MISMATCH"
    REJECTED_EMPTY_PACKET = "REJECTED_EMPTY_PACKET"
    REJECTED_STALE = "REJECTED_STALE"
    REJECTED_FUTURE_TS = "REJECTED_FUTURE_TS"


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """Contract for a single ``schema`` string declared in the registry.

    The spec is intentionally minimal — required keys + a budget on
    extras. Strict typing per field belongs in the engine that owns
    the contract, not in the SCVS layer.
    """

    required_keys: frozenset[str]
    allow_extras: bool = True


@dataclass(frozen=True, slots=True)
class ContractRegistry:
    """Maps registry ``schema`` strings to a :class:`SchemaSpec`."""

    specs: Mapping[str, SchemaSpec]

    def by_name(self, name: str) -> SchemaSpec | None:
        return self.specs.get(name)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """One immutable per-packet validation outcome."""

    source_id: str
    outcome: ValidationOutcome
    detail: str = ""


@dataclass(slots=True)
class SchemaGuard:
    """Pure per-packet schema + staleness validator."""

    registry: SourceRegistry
    contracts: ContractRegistry
    max_age_ns: int  # SCVS-09 staleness threshold

    def __post_init__(self) -> None:
        if self.max_age_ns < 0:
            raise ValueError("max_age_ns must be >= 0")

    def validate(
        self,
        *,
        source_id: str,
        packet: Mapping[str, object],
        packet_ts_ns: int,
        now_ns: int,
    ) -> ValidationResult:
        decl = self.registry.by_id(source_id)
        if decl is None:
            return ValidationResult(
                source_id=source_id,
                outcome=ValidationOutcome.REJECTED_UNKNOWN_SOURCE,
                detail=f"unknown source_id: {source_id!r}",
            )
        if not decl.enabled:
            return ValidationResult(
                source_id=source_id,
                outcome=ValidationOutcome.REJECTED_DISABLED_SOURCE,
                detail=f"source {source_id!r} has enabled=false",
            )
        if not packet:
            return ValidationResult(
                source_id=source_id,
                outcome=ValidationOutcome.REJECTED_EMPTY_PACKET,
                detail="packet has no fields",
            )
        if packet_ts_ns > now_ns:
            return ValidationResult(
                source_id=source_id,
                outcome=ValidationOutcome.REJECTED_FUTURE_TS,
                detail=f"packet_ts_ns={packet_ts_ns} > now_ns={now_ns}",
            )
        gap = now_ns - packet_ts_ns
        if self.max_age_ns > 0 and gap > self.max_age_ns:
            return ValidationResult(
                source_id=source_id,
                outcome=ValidationOutcome.REJECTED_STALE,
                detail=(
                    f"packet age {gap}ns exceeds max_age_ns={self.max_age_ns}"
                ),
            )
        return _validate_schema(decl, self.contracts, packet)


def _validate_schema(
    decl: SourceDeclaration,
    contracts: ContractRegistry,
    packet: Mapping[str, object],
) -> ValidationResult:
    spec = contracts.by_name(decl.schema)
    if spec is None:
        return ValidationResult(
            source_id=decl.id,
            outcome=ValidationOutcome.REJECTED_UNKNOWN_SCHEMA,
            detail=f"contract {decl.schema!r} not registered",
        )
    missing = sorted(k for k in spec.required_keys if k not in packet)
    if missing:
        return ValidationResult(
            source_id=decl.id,
            outcome=ValidationOutcome.REJECTED_SCHEMA_MISMATCH,
            detail=f"missing required keys: {missing}",
        )
    if not spec.allow_extras:
        extras = sorted(k for k in packet if k not in spec.required_keys)
        if extras:
            return ValidationResult(
                source_id=decl.id,
                outcome=ValidationOutcome.REJECTED_SCHEMA_MISMATCH,
                detail=f"unexpected extra keys: {extras}",
            )
    return ValidationResult(source_id=decl.id, outcome=ValidationOutcome.ACCEPTED)


__all__ = [
    "ContractRegistry",
    "SchemaGuard",
    "SchemaSpec",
    "ValidationOutcome",
    "ValidationResult",
]
