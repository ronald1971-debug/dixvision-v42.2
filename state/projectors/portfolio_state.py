"""
state/projectors/portfolio_state.py
Projects MARKET.TRADE_EXECUTION events into portfolio read-state.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class PortfolioReadModel:
    equity_usd: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)  # asset -> signed size
    realized_pnl_usd: float = 0.0


class PortfolioStateProjector:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = PortfolioReadModel()

    def apply(self, event: dict) -> None:
        et = str(event.get("event_type", "")).upper()
        st = str(event.get("sub_type", "")).upper()
        if et != "MARKET" or st != "TRADE_EXECUTION":
            return
        p = event.get("payload", {}) or {}
        asset = str(p.get("asset", ""))
        side = str(p.get("side", "")).upper()
        size_usd = float(p.get("size_usd", 0.0))
        if not asset or side not in {"BUY", "SELL"} or size_usd <= 0:
            return
        signed = size_usd if side == "BUY" else -size_usd
        with self._lock:
            self._model.positions[asset] = self._model.positions.get(asset, 0.0) + signed

    def snapshot(self) -> PortfolioReadModel:
        with self._lock:
            return PortfolioReadModel(
                equity_usd=self._model.equity_usd,
                positions=dict(self._model.positions),
                realized_pnl_usd=self._model.realized_pnl_usd,
            )


_p: PortfolioStateProjector | None = None
_lock = threading.Lock()


def get_portfolio_projector() -> PortfolioStateProjector:
    global _p
    if _p is None:
        with _lock:
            if _p is None:
                _p = PortfolioStateProjector()
    return _p
