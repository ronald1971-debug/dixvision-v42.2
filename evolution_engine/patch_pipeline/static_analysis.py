"""Static analysis stage — deterministic finding aggregation.

Caller passes a list of findings produced by an external tool (e.g.
ruff, mypy, authority_lint). The stage classifies the patch by the
worst severity and emits a stage verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from evolution_engine.patch_pipeline.pipeline import PatchStage, StageVerdict


class FindingSeverity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class StaticAnalysisFinding:
    rule: str
    severity: FindingSeverity
    location: str
    detail: str = ""


class StaticAnalysisStage:
    """GOV-G18-S2."""

    name: str = "static_analysis"
    spec_id: str = "GOV-G18-S2"

    __slots__ = ("_max_severity",)

    def __init__(
        self,
        *,
        max_severity: FindingSeverity = FindingSeverity.WARN,
    ) -> None:
        self._max_severity = max_severity

    def evaluate(
        self,
        *,
        ts_ns: int,
        findings: Sequence[StaticAnalysisFinding],
    ) -> StageVerdict:
        rank = {
            FindingSeverity.INFO: 0,
            FindingSeverity.WARN: 1,
            FindingSeverity.ERROR: 2,
        }
        worst = max(
            (rank[f.severity] for f in findings),
            default=-1,
        )
        passed = worst <= rank[self._max_severity]
        return StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.STATIC_ANALYSIS,
            passed=passed,
            detail=(
                f"{len(findings)} findings, worst="
                + (
                    [k.value for k, v in rank.items() if v == worst][0]
                    if worst >= 0
                    else "NONE"
                )
            ),
            meta={
                "findings": str(len(findings)),
                "max_severity": self._max_severity.value,
            },
        )


__all__ = [
    "FindingSeverity",
    "StaticAnalysisFinding",
    "StaticAnalysisStage",
]
