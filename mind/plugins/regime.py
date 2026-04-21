"""
mind/plugins/regime.py
Baseline regime-detection plugin — categorizes market regime as trend / chop
and produces a conservative trade-gating signal.
"""
from __future__ import annotations

from typing import Any

from . import _BasePlugin


class RegimePlugin(_BasePlugin):
    name = "regime"

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        volatility = max(0.0, float(data.get("volatility", 0.0)))
        trend_strength = max(-1.0, min(1.0, float(data.get("trend_strength", 0.0))))
        if volatility > 0.1:
            return {"signal": 0.0, "confidence": 0.1, "strategy": self.name, "regime": "chop"}
        return {"signal": trend_strength * 0.5, "confidence": 0.6,
                "strategy": self.name, "regime": "trend"}
