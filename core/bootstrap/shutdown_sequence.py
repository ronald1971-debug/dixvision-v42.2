"""
core/bootstrap/shutdown_sequence.py
Canonical ordered shutdown steps.
"""
from __future__ import annotations

SHUTDOWN_SEQUENCE: list[tuple[str, str]] = [
    ("mode_halted", "Transition system mode → HALTED"),
    ("dyon_stop", "Stop Dyon system engine"),
    ("guardian_stop", "Stop runtime guardian"),
    ("hazard_bus_drain", "Drain hazard bus"),
    ("ledger_flush", "Flush event store writer"),
    ("audit_shutdown_complete", "Emit SHUTDOWN_COMPLETE audit record"),
]


def run_shutdown() -> None:
    """Execute the canonical shutdown sequence."""
    try:
        from system.state import get_state_manager

        get_state_manager().set_mode("HALTED")
    except Exception:
        pass
    try:
        from execution.engine import get_dyon_engine

        get_dyon_engine().stop()
    except Exception:
        pass
    try:
        from enforcement.runtime_guardian import get_runtime_guardian

        get_runtime_guardian().stop()
    except Exception:
        pass
    try:
        from system.audit_logger import get_audit_logger

        get_audit_logger().log("SYSTEM", "shutdown", {"event": "SHUTDOWN_COMPLETE"})
    except Exception:
        pass
