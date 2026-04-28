"""Patch pipeline FSM and record (Phase 4 / Phase 5 refactor).

Concrete in-memory implementation of
:class:`core.contracts.patch.PatchPipelineProtocol`. Data types
(:class:`PatchStage`, :class:`PatchRecord`, :class:`StageVerdict`, …) are
re-exported from :mod:`core.contracts.patch` for backward compatibility
with Phase 4 callers.
"""

from __future__ import annotations

from core.contracts.patch import (
    LEGAL_PATCH_TRANSITIONS,
    PatchPipelineError,
    PatchRecord,
    PatchStage,
    PatchTransition,
    StageVerdict,
)


class PatchPipeline:
    """Deterministic registry + FSM for patch promotion."""

    name: str = "patch_pipeline"
    spec_id: str = "GOV-G18"

    __slots__ = ("_records", "_history", "_verdicts")

    def __init__(self) -> None:
        self._records: dict[str, PatchStage] = {}
        self._history: dict[str, list[PatchTransition]] = {}
        self._verdicts: dict[str, list[StageVerdict]] = {}

    # ------------------------------------------------------------------
    def propose(self, *, patch_id: str, ts_ns: int) -> PatchRecord:
        if not patch_id:
            raise ValueError("patch_id must be non-empty")
        if patch_id in self._records:
            raise PatchPipelineError(f"patch already exists: {patch_id!r}")
        self._records[patch_id] = PatchStage.PROPOSED
        self._history[patch_id] = [
            PatchTransition(
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
            PatchTransition(
                ts_ns=ts_ns, prev=prev, new=new_stage, reason=reason
            )
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
    "PatchTransition",
    "StageVerdict",
]
