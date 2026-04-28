"""PatchOutcomeFeedback — DYN-L02.

Aggregates trade outcomes into rolling per-strategy statistics that drive
both Learning parameter updates (``UpdateEmitter``) and Evolution structural
proposals (``MutationProposer``).

Pure: no IO, no clocks, no randomness. Same call sequence → same stats
(INV-15).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from core.contracts.learning import StrategyStats, TradeOutcome


class PatchOutcomeFeedback:
    """Rolling stats per strategy_id, with a fixed-size window."""

    name: str = "patch_outcome_feedback"
    spec_id: str = "DYN-L02"

    __slots__ = ("_window", "_outcomes")

    def __init__(self, *, window: int = 100) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._outcomes: dict[str, deque[TradeOutcome]] = {}

    def observe(self, outcome: TradeOutcome) -> StrategyStats:
        """Record an outcome; return the updated rolling stats."""
        bucket = self._outcomes.setdefault(
            outcome.strategy_id, deque(maxlen=self._window)
        )
        bucket.append(outcome)
        return self._compute(outcome.strategy_id, outcome.ts_ns)

    def snapshot(
        self, *, strategy_id: str, ts_ns: int
    ) -> StrategyStats:
        """Return current rolling stats for ``strategy_id``."""
        return self._compute(strategy_id, ts_ns)

    def all_snapshots(self, *, ts_ns: int) -> Mapping[str, StrategyStats]:
        """Return current rolling stats for every tracked strategy."""
        return {
            sid: self._compute(sid, ts_ns)
            for sid in sorted(self._outcomes.keys())
        }

    # ------------------------------------------------------------------
    def _compute(self, strategy_id: str, ts_ns: int) -> StrategyStats:
        bucket = self._outcomes.get(strategy_id)
        if not bucket:
            return StrategyStats(
                ts_ns=ts_ns,
                strategy_id=strategy_id,
                n_trades=0,
                n_wins=0,
                n_losses=0,
                total_pnl=0.0,
                mean_pnl=0.0,
                win_rate=0.0,
            )
        n = len(bucket)
        wins = sum(1 for o in bucket if o.pnl > 0.0)
        losses = sum(1 for o in bucket if o.pnl < 0.0)
        total = sum(o.pnl for o in bucket)
        mean = total / n
        win_rate = wins / n
        return StrategyStats(
            ts_ns=ts_ns,
            strategy_id=strategy_id,
            n_trades=n,
            n_wins=wins,
            n_losses=losses,
            total_pnl=total,
            mean_pnl=mean,
            win_rate=win_rate,
        )


__all__ = ["PatchOutcomeFeedback"]
