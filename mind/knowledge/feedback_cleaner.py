"""
mind.knowledge.feedback_cleaner — normalizes raw feedback signals
(fills, cancels, reject reasons, user chat corrections) into a canonical
per-event record suitable for the strategy arbiter and RL reward shaping.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class CleanedFeedback:
    kind: str        # FILL | REJECT | CANCEL | USER_CORRECTION
    strategy: str
    asset: str
    reward: float    # normalized to [-1, +1]
    raw: dict[str, Any]


class FeedbackCleaner:
    def clean(self, kind: str, strategy: str, asset: str, raw: dict[str, Any]) -> CleanedFeedback:
        reward = 0.0
        if kind == "FILL":
            reward = float(raw.get("realized_pnl_norm", 0.0))
        elif kind == "REJECT":
            reward = -0.5
        elif kind == "CANCEL":
            reward = -0.1
        elif kind == "USER_CORRECTION":
            reward = float(raw.get("polarity", -1.0))
        return CleanedFeedback(
            kind=kind, strategy=strategy, asset=asset,
            reward=max(-1.0, min(1.0, reward)), raw=raw,
        )


_cleaner: FeedbackCleaner | None = None
_lock = threading.Lock()


def get_feedback_cleaner() -> FeedbackCleaner:
    global _cleaner
    if _cleaner is None:
        with _lock:
            if _cleaner is None:
                _cleaner = FeedbackCleaner()
    return _cleaner
