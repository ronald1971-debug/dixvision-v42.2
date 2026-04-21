"""
governance/policy_engine.py
Evaluates declarative policy rules against an action request. The result is
folded into the GovernanceKernel decision.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyRule:
    name: str
    predicate: Callable[[dict[str, Any]], bool]
    reason: str
    deny: bool = True  # True = rule denies, False = rule approves


@dataclass
class PolicyResult:
    allowed: bool
    matched: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []

    def register(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        result = PolicyResult(allowed=True)
        for rule in self._rules:
            try:
                hit = rule.predicate(context)
            except Exception:
                hit = False
            if hit:
                result.matched.append(rule.name)
                result.reasons.append(rule.reason)
                if rule.deny:
                    result.allowed = False
        return result


# Default singleton seeded with common denies. Kernel can augment at runtime.
_engine = PolicyEngine()
_engine.register(
    PolicyRule(
        name="deny_martingale",
        predicate=lambda ctx: bool(ctx.get("strategy", "")).__eq__(True)
        and str(ctx.get("strategy", "")).lower() == "martingale",
        reason="martingale_forbidden_axiom",
    )
)
_engine.register(
    PolicyRule(
        name="deny_unbounded_leverage",
        predicate=lambda ctx: float(ctx.get("leverage", 0.0)) > 10.0,
        reason="unbounded_leverage_forbidden_axiom",
    )
)


def get_policy_engine() -> PolicyEngine:
    return _engine
