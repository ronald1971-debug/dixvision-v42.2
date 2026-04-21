"""governance.mode.degraded_mode — Enter/exit DEGRADED via the ModeManager."""
from __future__ import annotations

from governance.mode_manager import SystemMode, get_mode_manager


def enter_degraded_mode(reason: str = "unspecified") -> bool:
    return get_mode_manager().transition(SystemMode.DEGRADED, reason=reason)


def exit_degraded_mode(reason: str = "operator_clear") -> bool:
    return get_mode_manager().transition(SystemMode.NORMAL, reason=reason)
