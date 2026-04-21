"""
governance/oracle/tier_l1_fast.py
Fast-tier synchronous approval using only the precomputed FastRiskCache.
Target latency: sub-millisecond. No I/O.
"""
from __future__ import annotations

from typing import Any

from system.fast_risk_cache import get_risk_cache


def approve_l1_fast(ctx: dict[str, Any]) -> tuple[bool, str]:
    rc = get_risk_cache().get()
    ok, reason = rc.allows_trade(
        size_usd=float(ctx.get("size_usd", 0.0)),
        portfolio_usd=float(ctx.get("portfolio_usd", 100_000.0)),
    )
    return ok, reason
