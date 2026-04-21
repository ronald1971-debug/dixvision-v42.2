"""
governance/mode_manager.py
DIX VISION v42.2 — System Mode Manager

Controls: NORMAL → DEGRADED → SAFE_MODE → EMERGENCY_HALT
Transitions are governance-gated and ledger-logged.
"""
from __future__ import annotations

from enum import Enum

from state.ledger.event_store import append_event
from system.fast_risk_cache import get_risk_cache
from system.state import get_state_manager


class SystemMode(str, Enum):
    INIT = "INIT"
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    SAFE_MODE = "SAFE_MODE"
    EMERGENCY_HALT = "EMERGENCY_HALT"

_VALID_TRANSITIONS = {
    SystemMode.INIT: {SystemMode.NORMAL},
    SystemMode.NORMAL: {SystemMode.DEGRADED, SystemMode.SAFE_MODE, SystemMode.EMERGENCY_HALT},
    SystemMode.DEGRADED: {SystemMode.NORMAL, SystemMode.SAFE_MODE, SystemMode.EMERGENCY_HALT},
    SystemMode.SAFE_MODE: {SystemMode.NORMAL, SystemMode.EMERGENCY_HALT},
    SystemMode.EMERGENCY_HALT: set(),
}

class ModeManager:
    def __init__(self) -> None:
        self._state_mgr = get_state_manager()
        self._cache = get_risk_cache()

    def current_mode(self) -> SystemMode:
        return SystemMode(self._state_mgr.get().governance_mode)

    def transition(self, new_mode: SystemMode, reason: str = "") -> bool:
        current = self.current_mode()
        if new_mode not in _VALID_TRANSITIONS.get(current, set()):
            return False
        self._state_mgr.update(governance_mode=new_mode.value)
        if new_mode in {SystemMode.SAFE_MODE, SystemMode.EMERGENCY_HALT}:
            self._cache.enter_safe_mode()
        elif new_mode == SystemMode.NORMAL:
            self._cache.resume_trading()
        try:
            append_event("GOVERNANCE", "MODE_CHANGE", "mode_manager",
                         {"from": current.value, "to": new_mode.value, "reason": reason})
        except Exception:
            pass
        return True

    def halt(self, reason: str = "") -> None:
        self._cache.halt_trading(reason=reason)
        self._state_mgr.update(governance_mode=SystemMode.EMERGENCY_HALT.value,
                               trading_allowed=False)

_mgr: ModeManager | None = None

def get_mode_manager() -> ModeManager:
    global _mgr
    if _mgr is None:
        _mgr = ModeManager()
    return _mgr
