"""
state/projectors/governance_state.py
Projects GOVERNANCE events into a rolling decision read-model.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GovernanceReadModel:
    decision_counts: dict[str, int] = field(default_factory=dict)
    recent_decisions: deque[dict] = field(default_factory=lambda: deque(maxlen=256))


class GovernanceStateProjector:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = GovernanceReadModel()

    def apply(self, event: dict) -> None:
        if str(event.get("event_type", "")).upper() != "GOVERNANCE":
            return
        p = event.get("payload", {}) or {}
        outcome = str(p.get("outcome", ""))
        with self._lock:
            if outcome:
                self._model.decision_counts[outcome] = self._model.decision_counts.get(outcome, 0) + 1
            self._model.recent_decisions.append({
                "sub_type": event.get("sub_type"),
                "payload": p,
            })

    def snapshot(self) -> GovernanceReadModel:
        with self._lock:
            return GovernanceReadModel(
                decision_counts=dict(self._model.decision_counts),
                recent_decisions=deque(self._model.recent_decisions, maxlen=256),
            )


_p: GovernanceStateProjector | None = None
_lock = threading.Lock()


def get_governance_projector() -> GovernanceStateProjector:
    global _p
    if _p is None:
        with _lock:
            if _p is None:
                _p = GovernanceStateProjector()
    return _p
