"""
mind/plugins/arbitrage.py
Baseline arbitrage plugin — scores price spread between two venues.
"""
from __future__ import annotations

from typing import Any

from . import _BasePlugin


class ArbitragePlugin(_BasePlugin):
    name = "arbitrage"

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        buy_price = float(data.get("buy_price", 0.0))
        sell_price = float(data.get("sell_price", 0.0))
        if buy_price <= 0 or sell_price <= 0:
            return {"signal": 0.0, "confidence": 0.0, "strategy": self.name}
        spread_pct = (sell_price - buy_price) / buy_price
        signal = max(-1.0, min(1.0, spread_pct * 200.0))
        confidence = min(1.0, abs(signal))
        return {"signal": signal, "confidence": confidence, "strategy": self.name}
