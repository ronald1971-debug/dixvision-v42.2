"""
mind/plugins/liquidity.py
Baseline liquidity plugin — confidence scales with top-of-book depth.
"""
from __future__ import annotations

from typing import Any

from . import _BasePlugin


class LiquidityPlugin(_BasePlugin):
    name = "liquidity"

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        bid_size = float(data.get("bid_size", 0.0))
        ask_size = float(data.get("ask_size", 0.0))
        total = bid_size + ask_size
        if total <= 0:
            return {"signal": 0.0, "confidence": 0.0, "strategy": self.name}
        imbalance = (bid_size - ask_size) / total
        signal = max(-1.0, min(1.0, imbalance))
        confidence = min(1.0, total / 1_000_000.0)
        return {"signal": signal, "confidence": confidence, "strategy": self.name}
