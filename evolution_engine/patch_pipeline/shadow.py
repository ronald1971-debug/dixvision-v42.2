"""Shadow stage — promotion gate based on shadow-mode samples."""

from __future__ import annotations

from dataclasses import dataclass

from evolution_engine.patch_pipeline.pipeline import PatchStage, StageVerdict


@dataclass(frozen=True, slots=True)
class ShadowVerdict:
    samples: int
    matches: int
    error_rate: float


class ShadowStage:
    """GOV-G18-S4."""

    name: str = "shadow"
    spec_id: str = "GOV-G18-S4"

    __slots__ = ("_min_samples", "_max_error_rate")

    def __init__(
        self,
        *,
        min_samples: int = 50,
        max_error_rate: float = 0.05,
    ) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if not 0.0 <= max_error_rate <= 1.0:
            raise ValueError("max_error_rate must be in [0, 1]")
        self._min_samples = min_samples
        self._max_error_rate = max_error_rate

    def evaluate(
        self,
        *,
        ts_ns: int,
        samples: int,
        matches: int,
    ) -> tuple[ShadowVerdict, StageVerdict]:
        if samples < 0 or matches < 0 or matches > samples:
            raise ValueError("invalid sample counts")
        error_rate = (
            0.0 if samples == 0 else 1.0 - (matches / samples)
        )
        sv = ShadowVerdict(
            samples=samples,
            matches=matches,
            error_rate=error_rate,
        )
        passed = (
            samples >= self._min_samples
            and error_rate <= self._max_error_rate
        )
        verdict = StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.SHADOW,
            passed=passed,
            detail=(
                f"samples={samples} matches={matches} err={error_rate:.4f}"
            ),
            meta={
                "samples": str(samples),
                "matches": str(matches),
                "error_rate": f"{error_rate:.6f}",
            },
        )
        return sv, verdict


__all__ = ["ShadowStage", "ShadowVerdict"]
