"""SCVS bidirectional-closure lint (Phase 1).

Implements the two BUILD-FAIL rules:

* **SCVS-01** — every ``enabled: true`` source in the registry must be
  referenced by at least one ``consumes.yaml``. No unused live source.
* **SCVS-02** — every ``source_id`` referenced from any
  ``consumes.yaml`` must exist in the registry. No phantom consumption.

The function is pure: it returns the set of violations for callers
(tests, CI tool) to surface or raise on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from system_engine.scvs.consumption_tracker import ConsumptionDeclaration
from system_engine.scvs.source_registry import SourceRegistry


@dataclass(frozen=True, slots=True)
class SCVSViolation:
    """One SCVS lint violation."""

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
