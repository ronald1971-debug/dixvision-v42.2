"""
state/projectors/system_state.py
Projects SYSTEM and HAZARD events into a rolling system read-model.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SystemReadModel:
    boot_complete: bool = False
    last_mode: str = "INIT"
    recent_hazards: deque[dict] = field(default_factory=lambda: deque(maxlen=128))
    hazard_counts: dict[str, int] = field(default_factory=dict)


class SystemStateProjector:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = SystemReadModel()

    def apply(self, event: dict) -> None:
        et = str(event.get("event_type", "")).upper()
        st = str(event.get("sub_type", "")).upper()
        p = event.get("payload", {}) or {}
        with self._lock:
            if et == "SYSTEM" and st == "BOOT_COMPLETE":
                self._model.boot_complete = True
            elif et == "GOVERNANCE" and st == "MODE_CHANGE":
                self._model.last_mode = str(p.get("to", self._model.last_mode))
            elif et == "HAZARD":
                self._model.recent_hazards.append({
                    "sub_type": st,
                    "payload": p,
                })
                self._model.hazard_counts[st] = self._model.hazard_counts.get(st, 0) + 1

    def snapshot(self) -> SystemReadModel:
        with self._lock:
            return SystemReadModel(
                boot_complete=self._model.boot_complete,
                last_mode=self._model.last_mode,
                recent_hazards=deque(self._model.recent_hazards, maxlen=128),
                hazard_counts=dict(self._model.hazard_counts),
            )


_p: SystemStateProjector | None = None
_lock = threading.Lock()


def get_system_projector() -> SystemStateProjector:
    global _p
    if _p is None:
        with _lock:
            if _p is None:
                _p = SystemStateProjector()
    return _p
