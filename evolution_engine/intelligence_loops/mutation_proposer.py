"""MutationProposer — DYN-L01.

Translates rolling :class:`StrategyStats` into structural
:class:`PatchProposal` records when performance breaches operator-set
thresholds. One-shot per (strategy_id, reason) episode — proposals are not
re-emitted until the strategy recovers above the threshold.

Pure / deterministic. Same input series → same output proposals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts.learning import PatchProposal, StrategyStats
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
    assert_unfrozen,
)


@dataclass(frozen=True, slots=True)
class MutationThresholds:
    """Operator-tuned thresholds for triggering mutation proposals."""

    min_trades: int = 30
    min_win_rate: float = 0.40
    min_mean_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.min_trades < 1:
            raise ValueError("min_trades must be >= 1")
        if not 0.0 <= self.min_win_rate <= 1.0:
            raise ValueError("min_win_rate must be within [0, 1]")


class MutationProposer:
    """Emits one :class:`PatchProposal` per breached threshold per strategy."""

    name: str = "mutation_proposer"
    spec_id: str = "DYN-L01"

    __slots__ = ("_thresholds", "_armed", "_counter", "_freeze")

    def __init__(
        self,
        *,
        thresholds: MutationThresholds | None = None,
        freeze: LearningEvolutionFreezePolicy | None = None,
    ) -> None:
        self._thresholds = thresholds or MutationThresholds()
        self._armed: dict[tuple[str, str], bool] = {}
        self._counter = 0
        self._freeze = freeze

    def evaluate(
        self, stats: StrategyStats
    ) -> tuple[PatchProposal, ...]:
        """Return proposals for any newly-breached threshold."""
        assert_unfrozen(self._freeze, action="propose_patch")
        if stats.n_trades < self._thresholds.min_trades:
            self._clear(stats.strategy_id)
            return ()
        out: list[PatchProposal] = []
        if stats.win_rate < self._thresholds.min_win_rate:
            p = self._maybe_emit(
                stats=stats,
                reason="win_rate_below_floor",
                rationale=(
                    f"win_rate={stats.win_rate:.3f} < "
                    f"{self._thresholds.min_win_rate:.3f}"
                ),
                touchpoints=(f"strategies.{stats.strategy_id}.entry_filter",),
            )
            if p is not None:
                out.append(p)
        else:
            self._disarm(stats.strategy_id, "win_rate_below_floor")
        if stats.mean_pnl < self._thresholds.min_mean_pnl:
            p = self._maybe_emit(
                stats=stats,
                reason="mean_pnl_below_floor",
                rationale=(
                    f"mean_pnl={stats.mean_pnl:.4f} < "
                    f"{self._thresholds.min_mean_pnl:.4f}"
                ),
                touchpoints=(f"strategies.{stats.strategy_id}.exit_filter",),
            )
            if p is not None:
                out.append(p)
        else:
            self._disarm(stats.strategy_id, "mean_pnl_below_floor")
        return tuple(out)

    # ------------------------------------------------------------------
    def _maybe_emit(
        self,
        *,
        stats: StrategyStats,
        reason: str,
        rationale: str,
        touchpoints: tuple[str, ...],
        meta: Mapping[str, str] | None = None,
    ) -> PatchProposal | None:
        key = (stats.strategy_id, reason)
        if self._armed.get(key, False):
            return None
        self._armed[key] = True
        self._counter += 1
        patch_id = f"PATCH-{stats.strategy_id}-{reason}-{self._counter:04d}"
        return PatchProposal(
            ts_ns=stats.ts_ns,
            patch_id=patch_id,
            source=self.name,
            target_strategy=stats.strategy_id,
            touchpoints=touchpoints,
            rationale=rationale,
            meta=dict(meta or {"reason": reason}),
        )

    def _disarm(self, strategy_id: str, reason: str) -> None:
        self._armed.pop((strategy_id, reason), None)

    def _clear(self, strategy_id: str) -> None:
        for k in list(self._armed.keys()):
            if k[0] == strategy_id:
                self._armed.pop(k, None)


__all__ = ["MutationProposer", "MutationThresholds"]
