"""
mind/plugins/macro.py
Baseline macro plugin — signal from headline macro indicator delta.
"""
from __future__ import annotations

from typing import Any

from . import _BasePlugin


class MacroPlugin(_BasePlugin):
    name = "macro"

    def evaluate(self, data: dict[str, Any]) -> dict[str, Any]:
        z = max(-3.0, min(3.0, float(data.get("macro_zscore", 0.0))))
        signal = max(-1.0, min(1.0, z / 3.0))
        confidence = min(1.0, abs(z) / 3.0 + 0.2)
        return {"signal": signal, "confidence": confidence, "strategy": self.name}
