"""
mind/plugins/technical.py
Baseline technical-analysis plugin. Produces a signal from a short/long MA
delta.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from . import _BasePlugin


class TechnicalPlugin(_BasePlugin):
    name = "technical"

    def __init__(self, short: int = 10, long: int = 40) -> None:
        self.short_window: deque[float] = deque(maxlen=short)
        self.long_window: deque[float] = deque(maxlen=long)

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        price = float(data.get("price", 0.0))
        if price <= 0:
            return {"signal": 0.0, "confidence": 0.0, "strategy": self.name}
        self.short_window.append(price)
        self.long_window.append(price)
        if len(self.short_window) < self.short_window.maxlen or len(self.long_window) < self.long_window.maxlen:
            return {"signal": 0.0, "confidence": 0.2, "strategy": self.name}
        sma_short = sum(self.short_window) / len(self.short_window)
        sma_long = sum(self.long_window) / len(self.long_window)
        if sma_long == 0:
            return {"signal": 0.0, "confidence": 0.0, "strategy": self.name}
        delta = (sma_short - sma_long) / sma_long
        signal = max(-1.0, min(1.0, delta * 20.0))
        confidence = min(1.0, abs(signal) + 0.3)
        return {"signal": signal, "confidence": confidence, "strategy": self.name}
