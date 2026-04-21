"""
windows/tray/tray_actions.py
Plain-Python actions the tray can invoke (works on Linux too for dev).
"""
from __future__ import annotations

from typing import Any


def status() -> dict[str, Any]:
    from observability.dashboards.cockpit_adapter import build_cockpit_snapshot

    return build_cockpit_snapshot()


def enter_safe_mode(reason: str = "tray") -> bool:
    from governance.mode.safe_mode import enter_safe_mode as _enter

    return _enter(reason=reason)


def exit_safe_mode(reason: str = "tray") -> bool:
    from governance.mode.safe_mode import exit_safe_mode as _exit

    return _exit(reason=reason)


def halt(reason: str = "tray") -> bool:
    from governance.mode.halted_mode import enter_halted_mode

    return enter_halted_mode(reason=reason)
