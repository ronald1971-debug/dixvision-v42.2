"""
mind/portfolio_manager.py
Tracks open positions + equity. Used by Indira to size orders and by
Governance (via constraint_compiler) to recompute risk limits.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Position:
    asset: str
    size: float = 0.0          # signed — positive = long
    avg_price: float = 0.0
    unrealized_pnl_usd: float = 0.0


@dataclass
class PortfolioSnapshot:
    equity_usd: float = 100_000.0
    cash_usd: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)

    def exposure_usd(self) -> float:
        return sum(abs(p.size) * p.avg_price for p in self.positions.values())


class PortfolioManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snap = PortfolioSnapshot()

    def snapshot(self) -> PortfolioSnapshot:
        with self._lock:
            return PortfolioSnapshot(
                equity_usd=self._snap.equity_usd,
                cash_usd=self._snap.cash_usd,
                positions={k: Position(**p.__dict__) for k, p in self._snap.positions.items()},
            )

    def apply_fill(self, asset: str, side: str, size: float, price: float) -> None:
        assert side in {"BUY", "SELL"}
        signed = size if side == "BUY" else -size
        with self._lock:
            pos = self._snap.positions.setdefault(asset, Position(asset=asset))
            new_size = pos.size + signed
            if pos.size == 0 or (pos.size > 0 and signed > 0) or (pos.size < 0 and signed < 0):
                total_notional = abs(pos.size) * pos.avg_price + abs(signed) * price
                denom = abs(pos.size) + abs(signed)
                pos.avg_price = total_notional / denom if denom else 0.0
            pos.size = new_size
            self._snap.cash_usd -= signed * price

    def mark_to_market(self, prices: dict[str, float]) -> None:
        with self._lock:
            pnl = 0.0
            for a, p in self._snap.positions.items():
                mark = prices.get(a, p.avg_price)
                p.unrealized_pnl_usd = (mark - p.avg_price) * p.size
                pnl += p.unrealized_pnl_usd
            self._snap.equity_usd = self._snap.cash_usd + pnl + sum(
                abs(p.size) * p.avg_price for p in self._snap.positions.values()
            )


_pm: PortfolioManager | None = None
_lock = threading.Lock()


def get_portfolio_manager() -> PortfolioManager:
    global _pm
    if _pm is None:
        with _lock:
            if _pm is None:
                _pm = PortfolioManager()
    return _pm
