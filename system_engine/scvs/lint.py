"""SCVS bidirectional-closure lint.

Phase 1 BUILD-FAIL rules:

* **SCVS-01** — every ``enabled: true`` source in the registry must be
  referenced by at least one ``consumes.yaml``. No unused live source.
* **SCVS-02** — every ``source_id`` referenced from any
  ``consumes.yaml`` must exist in the registry. No phantom consumption.

Phase 3 WARN rule (does **not** fail the build):

* **SCVS-08** — registry rows that share the same ``(category,
  provider, endpoint)`` triple are flagged as redundant. The spec
  ("WARN redundancy") accepts that some redundancy is intentional —
  e.g. shadow-mode AI providers — so this is surfaced as guidance, not
  a hard failure.

The functions are pure: they return their results so callers (tests,
CI tool) can decide how to surface them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from system_engine.scvs.consumption_tracker import ConsumptionDeclaration
from system_engine.scvs.source_registry import SourceRegistry


@dataclass(frozen=True, slots=True)
class SCVSViolation:
    """One SCVS lint violation (BUILD-FAIL)."""

    rule: str
    detail: str


@dataclass(frozen=True, slots=True)
class SCVSWarning:
    """One SCVS lint warning (non-fatal)."""

    rule: str
    detail: str


def validate_scvs(
    registry: SourceRegistry,
    declarations: Iterable[ConsumptionDeclaration],
) -> tuple[SCVSViolation, ...]:
    """Return all SCVS-01 / SCVS-02 violations (empty tuple == clean)."""

    declarations = tuple(declarations)
    declared_inputs: set[str] = set()
    for decl in declarations:
        for inp in decl.inputs:
            declared_inputs.add(inp.source_id)

    registry_ids = registry.ids
    enabled_ids = registry.enabled_ids

    violations: list[SCVSViolation] = []

    # SCVS-02 — every consumes.yaml entry must be in the registry.
    for decl in declarations:
        for inp in decl.inputs:
            if inp.source_id not in registry_ids:
                violations.append(
                    SCVSViolation(
                        rule="SCVS-02",
                        detail=(
                            f"phantom consumption: module '{decl.module}' "
                            f"declares unknown source_id '{inp.source_id}' "
                            f"(in {decl.path})"
                        ),
                    )
                )

    # SCVS-01 — every enabled source must be consumed by at least one
    # module. ``enabled: false`` rows are intentionally exempt: they
    # are pre-registered placeholders for adapters not yet wired.
    for sid in sorted(enabled_ids):
        if sid not in declared_inputs:
            violations.append(
                SCVSViolation(
                    rule="SCVS-01",
                    detail=(
                        f"unused live source: '{sid}' is enabled in the "
                        f"registry but no module declares it in "
                        f"consumes.yaml"
                    ),
                )
            )

    return tuple(violations)


def find_redundant_sources(registry: SourceRegistry) -> tuple[SCVSWarning, ...]:
    """SCVS-08 — flag registry rows sharing ``(category, provider, endpoint)``.

    Pure helper. Returns warnings (never violations) — duplicates are
    sometimes intentional (e.g. shadow-mode AI providers, multi-region
    failover endpoints), so the spec mandates a WARN, not a fail.
    """

    groups: dict[tuple[str, str, str], list[str]] = {}
    for src in registry.sources:
        key = (src.category.value, src.provider, src.endpoint)
        groups.setdefault(key, []).append(src.id)

    warnings: list[SCVSWarning] = []
    for (category, provider, endpoint), ids in groups.items():
        if len(ids) < 2:
            continue
        warnings.append(
            SCVSWarning(
                rule="SCVS-08",
                detail=(
                    f"redundant sources sharing "
                    f"category={category!r} provider={provider!r} "
                    f"endpoint={endpoint!r}: {sorted(ids)}"
                ),
            )
        )
    return tuple(warnings)
