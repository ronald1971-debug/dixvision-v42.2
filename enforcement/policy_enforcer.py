"""
enforcement/policy_enforcer.py
Attribute-level policy enforcement: wraps a function call and consults the
declarative policy_engine + risk cache before permitting the invocation.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from governance.policy_engine import get_policy_engine
from system.fast_risk_cache import get_risk_cache


@dataclass(frozen=True)
class EnforceResult:
    allowed: bool
    reason: str
    reasons: tuple[str, ...] = ()


class PolicyEnforcer:
    def allow(self, ctx: dict[str, Any]) -> EnforceResult:
        rc = get_risk_cache().get()
        if not rc.trading_allowed:
            return EnforceResult(False, "trading_disallowed")
        size_usd = float(ctx.get("size_usd", 0.0))
        portfolio_usd = float(ctx.get("portfolio_usd", 100_000.0))
        ok, reason = rc.allows_trade(size_usd=size_usd, portfolio_usd=portfolio_usd)
        if not ok:
            return EnforceResult(False, reason)
        result = get_policy_engine().evaluate(ctx)
        if not result.allowed:
            return EnforceResult(False, "policy_denied", tuple(result.reasons))
        return EnforceResult(True, "ok")

    def enforce(self, fn: Callable[..., Any], ctx: dict[str, Any]) -> Any:
        verdict = self.allow(ctx)
        if not verdict.allowed:
            raise PermissionError(f"policy_denied: {verdict.reason}")
        return fn()


_pe: PolicyEnforcer | None = None
_lock = threading.Lock()


def get_policy_enforcer() -> PolicyEnforcer:
    global _pe
    if _pe is None:
        with _lock:
            if _pe is None:
                _pe = PolicyEnforcer()
    return _pe
