"""Patch pipeline FSM and record (Phase 4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class PatchStage(StrEnum):
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
    ts_ns: int
    stage: PatchStage
    passed: bool
    detail: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Transition:
    ts_ns: int
    prev: PatchStage
    new: PatchStage
    reason: str


@dataclass(frozen=True, slots=True)
class PatchRecord:
    patch_id: str
    stage: PatchStage
    history: tuple[_Transition, ...]
    verdicts: tuple[StageVerdict, ...]


class PatchPipeline:
    """Deterministic registry + FSM for patch promotion."""

    name: str = "patch_pipeline"
    spec_id: str = "GOV-G18"

    __slots__ = ("_records", "_history", "_verdicts")

    def __init__(self) -> None:
        self._records: dict[str, PatchStage] = {}
        self._history: dict[str, list[_Transition]] = {}
        self._verdicts: dict[str, list[StageVerdict]] = {}

    # ------------------------------------------------------------------
    def propose(self, *, patch_id: str, ts_ns: int) -> PatchRecord:
        if not patch_id:
            raise ValueError("patch_id must be non-empty")
        if patch_id in self._records:
            raise PatchPipelineError(f"patch already exists: {patch_id!r}")
        self._records[patch_id] = PatchStage.PROPOSED
        self._history[patch_id] = [
            _Transition(
                ts_ns=ts_ns,
                prev=PatchStage.PROPOSED,
                new=PatchStage.PROPOSED,
                reason="propose",
            )
        ]
        self._verdicts[patch_id] = []
        return self.get(patch_id)

    def get(self, patch_id: str) -> PatchRecord:
        if patch_id not in self._records:
            raise PatchPipelineError(f"unknown patch: {patch_id!r}")
        return PatchRecord(
            patch_id=patch_id,
            stage=self._records[patch_id],
            history=tuple(self._history[patch_id]),
            verdicts=tuple(self._verdicts[patch_id]),
        )

    def transition(
        self,
        *,
        patch_id: str,
        new_stage: PatchStage,
        ts_ns: int,
        reason: str,
    ) -> PatchRecord:
        if patch_id not in self._records:
            raise PatchPipelineError(f"unknown patch: {patch_id!r}")
        prev = self._records[patch_id]
        if new_stage not in LEGAL_PATCH_TRANSITIONS[prev]:
            raise PatchPipelineError(
                f"illegal transition {prev.value} → {new_stage.value} "
                f"for patch {patch_id!r}"
            )
        self._records[patch_id] = new_stage
        self._history[patch_id].append(
            _Transition(ts_ns=ts_ns, prev=prev, new=new_stage, reason=reason)
        )
        return self.get(patch_id)

    def record_verdict(
        self,
        *,
        patch_id: str,
        verdict: StageVerdict,
    ) -> None:
        if patch_id not in self._verdicts:
            raise PatchPipelineError(f"unknown patch: {patch_id!r}")
        self._verdicts[patch_id].append(verdict)

    def all_in(self, stage: PatchStage) -> tuple[PatchRecord, ...]:
        return tuple(
            self.get(pid)
            for pid, st in self._records.items()
            if st is stage
        )


__all__ = [
    "LEGAL_PATCH_TRANSITIONS",
    "PatchPipeline",
    "PatchPipelineError",
    "PatchRecord",
    "PatchStage",
    "StageVerdict",
]
