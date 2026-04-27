"""Strategy scheduler — Phase 3 / v2-B.

Decides *when* each strategy runs. Cadence is bar-aligned (number of
ticks since last fire) so replay determinism is preserved (INV-15).
The scheduler holds no clocks: the caller passes ``ts_ns`` from the
source tick.

A strategy that has not yet fired is considered "due" on its first
visit so cold start is deterministic across replays.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StrategyTick:
    """Per-strategy scheduler bookkeeping."""

    strategy_id: str
    cadence: int
    ticks_since_last_fire: int = 0
    last_fire_ts_ns: int = 0
    fires: int = 0


class StrategyScheduler:
    """Deterministic, tick-counting scheduler."""

    name: str = "strategy_scheduler"
    spec_id: str = "IND-SCH-01"

    def __init__(self) -> None:
        self._book: dict[str, StrategyTick] = {}

    # -- registration ------------------------------------------------------

    def register(
        self, *, strategy_id: str, cadence: int = 1
    ) -> StrategyTick:
        if not strategy_id:
            raise ValueError("strategy_id required")
        if cadence <= 0:
            raise ValueError("cadence must be > 0")
        if strategy_id in self._book:
            raise ValueError(f"already registered: {strategy_id}")
        record = StrategyTick(strategy_id=strategy_id, cadence=cadence)
        self._book[strategy_id] = record
        return record

    def deregister(self, strategy_id: str) -> None:
        self._book.pop(strategy_id, None)

    # -- queries -----------------------------------------------------------

    def get(self, strategy_id: str) -> StrategyTick | None:
        return self._book.get(strategy_id)

    def __len__(self) -> int:
        return len(self._book)

    # -- mutations ---------------------------------------------------------

    def step(self, ts_ns: int) -> tuple[str, ...]:
        """Advance one tick and return strategy_ids that should fire.

        Strategies fire when ``ticks_since_last_fire >= cadence``. The
        counter resets on fire.
        """
        due: list[str] = []
        for sid, rec in self._book.items():
            rec.ticks_since_last_fire += 1
            if rec.ticks_since_last_fire >= rec.cadence:
                rec.ticks_since_last_fire = 0
                rec.last_fire_ts_ns = ts_ns
                rec.fires += 1
                due.append(sid)
        return tuple(due)


__all__ = ["StrategyScheduler", "StrategyTick"]
