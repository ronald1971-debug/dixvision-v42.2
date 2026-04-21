"""mind.strategies \u2014 pluggable strategy interface + built-ins."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class StrategySignal:
    strategy: str
    symbol: str
    side: str                                            # "buy" | "sell" | "flat"
    strength: float                                      # [-1.0 .. 1.0]
    rationale: str = ""
    features: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    name: str

    def propose(self, *, symbol: str, features: dict[str, float]) -> StrategySignal | None:
        ...


class TrendStrategy:
    name = "trend"

    def propose(self, *, symbol: str, features: dict[str, float]) -> StrategySignal | None:
        ma_fast = features.get("ma_fast", 0.0)
        ma_slow = features.get("ma_slow", 0.0)
        if ma_fast == 0.0 or ma_slow == 0.0:
            return None
        diff = (ma_fast - ma_slow) / max(abs(ma_slow), 1e-9)
        if abs(diff) < 0.002:
            return StrategySignal(self.name, symbol, "flat", 0.0,
                                  rationale="trend too weak")
        side = "buy" if diff > 0 else "sell"
        return StrategySignal(self.name, symbol, side,
                              max(-1.0, min(1.0, diff * 50)),
                              rationale="fast vs slow MA cross")


class MeanReversionStrategy:
    name = "mean_reversion"

    def propose(self, *, symbol: str, features: dict[str, float]) -> StrategySignal | None:
        z = features.get("price_zscore", 0.0)
        if abs(z) < 1.0:
            return StrategySignal(self.name, symbol, "flat", 0.0,
                                  rationale="z-score too small")
        side = "sell" if z > 0 else "buy"
        return StrategySignal(self.name, symbol, side,
                              max(-1.0, min(1.0, -z / 3.0)),
                              rationale=f"z-score={z:.2f}")


class BreakoutStrategy:
    name = "breakout"

    def propose(self, *, symbol: str, features: dict[str, float]) -> StrategySignal | None:
        high20 = features.get("high_20d", 0.0)
        low20 = features.get("low_20d", 0.0)
        last = features.get("last", 0.0)
        if not high20 or not low20 or not last:
            return None
        if last >= high20 * 0.999:
            return StrategySignal(self.name, symbol, "buy", 0.6,
                                  rationale="20d high breakout")
        if last <= low20 * 1.001:
            return StrategySignal(self.name, symbol, "sell", 0.6,
                                  rationale="20d low breakdown")
        return StrategySignal(self.name, symbol, "flat", 0.0,
                              rationale="inside range")


_BUILTINS: list[Strategy] = [TrendStrategy(), MeanReversionStrategy(), BreakoutStrategy()]


def list_builtins() -> list[Strategy]:
    return list(_BUILTINS)


__all__ = ["Strategy", "StrategySignal", "TrendStrategy",
           "MeanReversionStrategy", "BreakoutStrategy", "list_builtins"]
