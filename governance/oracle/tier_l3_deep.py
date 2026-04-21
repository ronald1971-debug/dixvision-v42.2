"""
governance/oracle/tier_l3_deep.py
Deep-tier approval: L2 + exposure + correlation checks.

Asynchronous by design; used when governance has time to spend (e.g. periodic
rebalance, large orders, strategy deployment approval).
"""
from __future__ import annotations

from typing import Any

from .tier_l2_balanced import approve_l2_balanced


def approve_l3_deep(ctx: dict[str, Any]) -> tuple[bool, str]:
    ok, reason = approve_l2_balanced(ctx)
    if not ok:
        return ok, reason
    exposure = float(ctx.get("current_exposure_pct", 0.0))
    if exposure > 0.40:
        return False, f"exposure_{exposure:.2f}_exceeds_40pct"
    correlation = float(ctx.get("correlation_to_open", 0.0))
    if correlation > 0.85:
        return False, f"correlation_{correlation:.2f}_exceeds_85pct"
    return True, "l3_deep_ok"
