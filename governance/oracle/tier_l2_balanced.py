"""
governance/oracle/tier_l2_balanced.py
Balanced-tier approval: FastRiskCache + declarative policy rules.
"""
from __future__ import annotations

from typing import Any

from governance.policy_engine import get_policy_engine

from .tier_l1_fast import approve_l1_fast


def approve_l2_balanced(ctx: dict[str, Any]) -> tuple[bool, str]:
    ok, reason = approve_l1_fast(ctx)
    if not ok:
        return ok, reason
    result = get_policy_engine().evaluate(ctx)
    if not result.allowed:
        return False, ";".join(result.reasons) or "policy_denied"
    return True, "l2_balanced_ok"
