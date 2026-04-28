"""Patch pipeline contracts (Phase 5).

Frozen, deterministic data types for the evolution-patch promotion FSM.
Lives in ``core.contracts`` so both the offline owner (Evolution) and the
sole runtime authority (Governance) can reference the same FSM shape
without violating the cross-engine import seam (INV-08 / INV-11 / INV-15
enforced by ``tools/authority_lint.py``).

* The concrete in-memory FSM implementation
  (:class:`evolution_engine.patch_pipeline.PatchPipeline`) lives in
  Evolution because it is the structural mutation owner per Build
  Compiler Spec §2 Phase 4 / Phase 5.
* Governance drives transitions through
  :class:`PatchPipelineProtocol` — never by importing the concrete
  class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


class PatchStage(StrEnum):
    """All valid stages of the patch promotion FSM."""

    PROPOSED = "PROPOSED"
    SANDBOX = "SANDBOX"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    BACKTEST = "BACKTEST"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


_LEGAL: dict[PatchStage, frozenset[PatchStage]] = {
    PatchStage.PROPOSED: frozenset({PatchStage.SANDBOX, PatchStage.REJECTED}),
    PatchStage.SANDBOX: frozenset(
        {PatchStage.STATIC_ANALYSIS, PatchStage.REJECTED}
    ),
    PatchStage.STATIC_ANALYSIS: frozenset(
        {PatchStage.BACKTEST, PatchStage.REJECTED}
    ),
    PatchStage.BACKTEST: frozenset({PatchStage.SHADOW, PatchStage.REJECTED}),
    PatchStage.SHADOW: frozenset({PatchStage.CANARY, PatchStage.REJECTED}),
    PatchStage.CANARY: frozenset(
        {PatchStage.APPROVED, PatchStage.REJECTED, PatchStage.ROLLED_BACK}
    ),
    PatchStage.APPROVED: frozenset({PatchStage.ROLLED_BACK}),
    PatchStage.REJECTED: frozenset(),
    PatchStage.ROLLED_BACK: frozenset(),
}

LEGAL_PATCH_TRANSITIONS: Mapping[PatchStage, frozenset[PatchStage]] = (
    MappingProxyType(_LEGAL)
)


class PatchPipelineError(RuntimeError):
    """Raised on illegal patch transitions or unknown patch IDs."""


@dataclass(frozen=True, slots=True)
class StageVerdict:
    """Deterministic verdict emitted by a single pipeline stage."""

    ts_ns: int
    stage: PatchStage
    passed: bool
    detail: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PatchTransition:
    """Single FSM step recorded on a :class:`PatchRecord`."""

    ts_ns: int
    prev: PatchStage
    new: PatchStage
    reason: str


@dataclass(frozen=True, slots=True)
class PatchRecord:
    """Frozen view of a patch's full FSM history + verdicts."""

    patch_id: str
    stage: PatchStage
    history: tuple[PatchTransition, ...]
    verdicts: tuple[StageVerdict, ...]


class PatchPipelineProtocol(Protocol):
    """Authority contract for the patch FSM.

    Implemented by :class:`evolution_engine.patch_pipeline.PatchPipeline`.
    Governance depends only on this Protocol — no direct import of the
    offline engine.
    """

    def propose(self, *, patch_id: str, ts_ns: int) -> PatchRecord: ...

    def get(self, patch_id: str) -> PatchRecord: ...

    def transition(
        self,
        *,
        patch_id: str,
        new_stage: PatchStage,
        ts_ns: int,
        reason: str,
    ) -> PatchRecord: ...

    def record_verdict(
        self, *, patch_id: str, verdict: StageVerdict
    ) -> None: ...

    def all_in(self, stage: PatchStage) -> tuple[PatchRecord, ...]: ...


__all__ = [
    "LEGAL_PATCH_TRANSITIONS",
    "PatchPipelineError",
    "PatchPipelineProtocol",
    "PatchRecord",
    "PatchStage",
    "PatchTransition",
    "StageVerdict",
]
