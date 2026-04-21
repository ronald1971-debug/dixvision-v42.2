"""governance.mode.safe_mode — Enter/exit SAFE_MODE via the ModeManager."""
from __future__ import annotations

from governance.mode_manager import SystemMode, get_mode_manager


def enter_safe_mode(reason: str = "unspecified") -> bool:
    return get_mode_manager().transition(SystemMode.SAFE_MODE, reason=reason)


def exit_safe_mode(reason: str = "operator_clear") -> bool:
    return get_mode_manager().transition(SystemMode.NORMAL, reason=reason)
