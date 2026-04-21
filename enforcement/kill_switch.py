"""
enforcement/kill_switch.py
Hard termination with safe logging. Armed by default; disarm for tests.
"""
from __future__ import annotations

import os
import signal
import threading

_lock = threading.Lock()
_armed = True


def is_armed() -> bool:
    with _lock:
        return _armed


def arm() -> None:
    global _armed
    with _lock:
        _armed = True


def disarm() -> None:
    global _armed
    with _lock:
        _armed = False


def trigger(reason: str = "unspecified", exit_code: int = 2) -> None:
    """Terminate the process after writing a KILL_SWITCH event to the ledger."""
    try:
        from state.ledger.event_store import append_event

        append_event("SYSTEM", "KILL_SWITCH", "enforcement.kill_switch", {
            "reason": reason, "exit_code": exit_code,
        })
    except Exception:
        pass
    if not is_armed():
        return
    try:
        from system.logger import get_logger

        get_logger("kill_switch").critical("kill_switch_triggered", reason=reason)
    except Exception:
        pass
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        pass
    os._exit(exit_code)
