"""
interrupt/interrupt_executor.py
Invokes the concrete emergency executor (in execution/) after the Resolver
has mapped a hazard event to a HazardAction.
"""
from __future__ import annotations

import threading

from .resolver import HazardAction


class InterruptExecutor:
    def execute(self, action: HazardAction) -> None:
        if action.action == "noop":
            return
        try:
            from execution.emergency_executor import get_emergency_executor

            get_emergency_executor().execute(action)
        except Exception as e:  # best-effort logging only — never raise
            try:
                from system.logger import get_logger

                get_logger("interrupt").critical(
                    "emergency_execute_failed",
                    action=action.action,
                    hazard_type=action.hazard_type,
                    error=str(e),
                )
            except Exception:
                pass


_ie: InterruptExecutor | None = None
_lock = threading.Lock()


def get_interrupt_executor() -> InterruptExecutor:
    global _ie
    if _ie is None:
        with _lock:
            if _ie is None:
                _ie = InterruptExecutor()
    return _ie
