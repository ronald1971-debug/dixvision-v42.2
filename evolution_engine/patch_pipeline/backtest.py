"""Backtest stage — deterministic acceptance summary."""

from __future__ import annotations

from dataclasses import dataclass

from evolution_engine.patch_pipeline.pipeline import PatchStage, StageVerdict


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    runs: int
    pnl: float
    sharpe: float
    max_drawdown: float


class BacktestStage:
    """GOV-G18-S3."""

    name: str = "backtest"
    spec_id: str = "GOV-G18-S3"

    __slots__ = ("_min_runs", "_min_sharpe", "_max_drawdown")

    def __init__(
        self,
        *,
        min_runs: int = 1,
        min_sharpe: float = 0.0,
        max_drawdown: float = 0.5,
    ) -> None:
        if min_runs < 1:
            raise ValueError("min_runs must be >= 1")
        if max_drawdown <= 0.0 or max_drawdown >= 1.0:
            raise ValueError("max_drawdown must be in (0, 1)")
        self._min_runs = min_runs
        self._min_sharpe = min_sharpe
        self._max_drawdown = max_drawdown

    def evaluate(
        self,
        *,
        ts_ns: int,
        summary: BacktestSummary,
    ) -> StageVerdict:
        passed = (
            summary.runs >= self._min_runs
            and summary.sharpe >= self._min_sharpe
            and summary.max_drawdown <= self._max_drawdown
        )
        return StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.BACKTEST,
            passed=passed,
            detail=(
                f"runs={summary.runs} sharpe={summary.sharpe:.3f} "
                f"dd={summary.max_drawdown:.3f}"
            ),
            meta={
                "runs": str(summary.runs),
                "sharpe": f"{summary.sharpe:.6f}",
                "max_drawdown": f"{summary.max_drawdown:.6f}",
            },
        )


__all__ = ["BacktestStage", "BacktestSummary"]
