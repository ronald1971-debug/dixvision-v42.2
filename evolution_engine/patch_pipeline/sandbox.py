"""Sandbox stage — pure-Python deterministic isolation check.

The real implementation will run patch code in a sandbox. At Phase 4
the stage is a deterministic *gate*: the caller submits a manifest of
declared touchpoints and the stage rejects anything that crosses a
forbidden list (e.g. cross-engine imports, network access, file IO).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evolution_engine.patch_pipeline.pipeline import PatchStage, StageVerdict


@dataclass(frozen=True, slots=True)
class SandboxResult:
    forbidden_touchpoints: tuple[str, ...]
    accepted_touchpoints: tuple[str, ...]


class SandboxStage:
    """SAFE-26 / GOV-G18-S1."""

    name: str = "sandbox"
    spec_id: str = "GOV-G18-S1"

    __slots__ = ("_forbidden_prefixes",)

    def __init__(self, forbidden_prefixes: Sequence[str] | None = None) -> None:
        self._forbidden_prefixes: tuple[str, ...] = tuple(
            forbidden_prefixes
            or (
                "subprocess",
                "socket",
                "urllib",
                "requests",
                "ctypes",
            )
        )

    def evaluate(
        self,
        *,
        ts_ns: int,
        touchpoints: Sequence[str],
    ) -> tuple[SandboxResult, StageVerdict]:
        forbidden: list[str] = []
        accepted: list[str] = []
        for tp in touchpoints:
            if any(tp.startswith(p) for p in self._forbidden_prefixes):
                forbidden.append(tp)
            else:
                accepted.append(tp)
        result = SandboxResult(
            forbidden_touchpoints=tuple(forbidden),
            accepted_touchpoints=tuple(accepted),
        )
        passed = not forbidden
        verdict = StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.SANDBOX,
            passed=passed,
            detail=(
                "sandbox clean"
                if passed
                else f"forbidden: {','.join(forbidden)}"
            ),
            meta={"accepted": str(len(accepted)), "forbidden": str(len(forbidden))},
        )
        return result, verdict


__all__ = ["SandboxResult", "SandboxStage"]
