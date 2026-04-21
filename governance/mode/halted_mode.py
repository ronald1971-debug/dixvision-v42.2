"""governance.mode.halted_mode — Trigger the terminal EMERGENCY_HALT state."""
from __future__ import annotations

from governance.mode_manager import SystemMode, get_mode_manager


def enter_halted_mode(reason: str = "unspecified") -> bool:
    mgr = get_mode_manager()
    ok = mgr.transition(SystemMode.EMERGENCY_HALT, reason=reason)
    if not ok:
        # EMERGENCY_HALT has no outbound edges. Force via halt() regardless.
        mgr.halt(reason=reason)
        ok = True
    return ok
