"""
enforcement/resource_enforcer.py
DIX VISION v42.2 — Resource Budget Enforcement
"""
from __future__ import annotations


def enforce_resources(state: object) -> None:
    compute_used = getattr(state, "compute_used", 0.0)
    compute_budget = getattr(state, "compute_budget", 100.0)
    if compute_used > compute_budget:
        from immutable_core.kill_switch import trigger_kill_switch
        trigger_kill_switch(
            reason=f"compute_budget_exceeded:{compute_used:.1f}>{compute_budget:.1f}",
            source="resource_enforcer"
        )
