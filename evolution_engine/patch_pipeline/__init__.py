"""Patch pipeline (Phase 4 / Build Compiler Spec §2 — Dyon).

Pure-Python, IO-free, deterministic FSM that promotes a candidate
patch through six gates::

    PROPOSED → SANDBOX → STATIC_ANALYSIS → BACKTEST →
    SHADOW → CANARY → APPROVED   (or REJECTED at any point)

Governance is the only authority that can promote (mirrors the strategy
lifecycle FSM). Each stage is a pure function: same inputs → same
verdict. The pipeline never imports learning_engine or governance_engine
(INV-08, INV-11). Outbound contract is:
``PatchPipeline.advance() -> (PatchRecord, list[StageVerdict])`` —
governance pulls and decides.
"""

from evolution_engine.patch_pipeline.backtest import (
    BacktestStage,
    BacktestSummary,
)
from evolution_engine.patch_pipeline.canary import CanaryStage, CanaryVerdict
from evolution_engine.patch_pipeline.pipeline import (
    LEGAL_PATCH_TRANSITIONS,
    PatchPipeline,
    PatchPipelineError,
    PatchRecord,
    PatchStage,
    StageVerdict,
)
from evolution_engine.patch_pipeline.rollback import RollbackPlanner, RollbackStep
from evolution_engine.patch_pipeline.sandbox import SandboxResult, SandboxStage
from evolution_engine.patch_pipeline.shadow import ShadowStage, ShadowVerdict
from evolution_engine.patch_pipeline.static_analysis import (
    FindingSeverity,
    StaticAnalysisFinding,
    StaticAnalysisStage,
)

__all__ = [
    "BacktestStage",
    "BacktestSummary",
    "CanaryStage",
    "CanaryVerdict",
    "FindingSeverity",
    "LEGAL_PATCH_TRANSITIONS",
    "PatchPipeline",
    "PatchPipelineError",
    "PatchRecord",
    "PatchStage",
    "RollbackPlanner",
    "RollbackStep",
    "SandboxResult",
    "SandboxStage",
    "ShadowStage",
    "ShadowVerdict",
    "StageVerdict",
    "StaticAnalysisFinding",
    "StaticAnalysisStage",
]
