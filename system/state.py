"""
system/state.py
DIX VISION v42.2 — System State Manager (Immutable Snapshots)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock


@dataclass(frozen=True)
class SystemState:
    mode: str = "INIT"
    health: float = 1.0
    last_heartbeat_ns: int = 0
    drawdown_pct: float = 0.0
    active_hazards: int = 0
    trading_allowed: bool = True
    governance_mode: str = "NORMAL"

class StateManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._state = SystemState()

    def get(self) -> SystemState:
        with self._lock:
            return self._state

    def update(self, **kw) -> SystemState:
        with self._lock:
            valid = {k: v for k, v in kw.items() if k in SystemState.__dataclass_fields__}
            self._state = replace(self._state, **valid)
            return self._state

    def set_mode(self, mode: str) -> SystemState:
        return self.update(mode=mode)

    def heartbeat(self) -> SystemState:
        import time
        return self.update(last_heartbeat_ns=time.monotonic_ns())

    def restore(self, d: dict) -> SystemState:
        with self._lock:
            valid = {k: v for k, v in d.items() if k in SystemState.__dataclass_fields__}
            self._state = SystemState(**valid)
            return self._state

_mgr: StateManager | None = None
_lock = RLock()

def get_state_manager() -> StateManager:
    global _mgr
    if _mgr is None:
        with _lock:
            if _mgr is None:
                _mgr = StateManager()
    return _mgr

def get_state() -> SystemState:
    return get_state_manager().get()
