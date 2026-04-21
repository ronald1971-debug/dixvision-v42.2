"""
core/bootstrap/startup_sequence.py
Canonical ordered startup steps. The runtime implementation lives in
``bootstrap_kernel.run`` — this module exposes the order for introspection.
"""
from __future__ import annotations

# (step_name, description)
STARTUP_SEQUENCE: list[tuple[str, str]] = [
    ("foundation_integrity", "Verify immutable_core foundation hash"),
    ("config_load", "Load centralized configuration"),
    ("state_init", "Create singleton state manager"),
    ("ledger_init", "Open append-only event store"),
    ("governance_boot_gate", "Governance evaluates boot readiness"),
    ("fast_risk_cache", "Precompute risk constraints for hot path"),
    ("hazard_bus", "Start async hazard event bus"),
    ("runtime_guardian", "Start runtime guardian thread"),
    ("mode_transition_normal", "Transition system mode → NORMAL"),
    ("dyon_engine", "Start Dyon system monitor engine"),
    ("audit_boot_complete", "Emit BOOT_COMPLETE audit record"),
]


def run_startup(env: str = "dev", verify_only: bool = False) -> None:
    """Execute the canonical startup sequence via ``bootstrap_kernel.run``."""
    from bootstrap_kernel import run as _run

    _run(env=env, verify_only=verify_only)
