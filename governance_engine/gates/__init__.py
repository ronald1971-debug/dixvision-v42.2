"""P0-B promotion gates — quantitative + RuleGraph patch evaluation.

Gating evaluators consumed by
:class:`governance_engine.services.patch_pipeline_bridge.PatchApprovalBridge`
on the ``CANARY → APPROVED`` edge. Pure / deterministic; no clock, no IO.
"""

from __future__ import annotations

from governance_engine.gates.quantitative_evaluator import (
    DEFAULT_QUANTITATIVE_THRESHOLDS,
    QuantitativeEvaluator,
    QuantitativeMetrics,
    QuantitativeThresholds,
    QuantitativeVerdict,
    QuantitativeVerdictKind,
)
from governance_engine.gates.rulegraph_patch_evaluator import (
    PatchEvaluationFacts,
    RuleGraphPatchEvaluator,
    RuleGraphPatchVerdict,
    RuleGraphPatchVerdictKind,
    build_patch_facts,
)

__all__ = [
    "DEFAULT_QUANTITATIVE_THRESHOLDS",
    "PatchEvaluationFacts",
    "QuantitativeEvaluator",
    "QuantitativeMetrics",
    "QuantitativeThresholds",
    "QuantitativeVerdict",
    "QuantitativeVerdictKind",
    "RuleGraphPatchEvaluator",
    "RuleGraphPatchVerdict",
    "RuleGraphPatchVerdictKind",
    "build_patch_facts",
]
