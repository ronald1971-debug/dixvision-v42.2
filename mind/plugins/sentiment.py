"""
mind/plugins/sentiment.py
Baseline sentiment plugin — consumes precomputed polarity scores from upstream
feeds. Never parses text at run-time.
"""
from __future__ import annotations

from typing import Any

from . import _BasePlugin


class SentimentPlugin(_BasePlugin):
    name = "sentiment"

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        polarity = max(-1.0, min(1.0, float(data.get("sentiment_polarity", 0.0))))
        confidence = max(0.0, min(1.0, float(data.get("sentiment_confidence", 0.5))))
        return {"signal": polarity, "confidence": confidence, "strategy": self.name}
