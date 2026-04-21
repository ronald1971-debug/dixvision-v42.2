"""
interrupt/resolver.py
Maps an incoming hazard event to a concrete HazardAction using the
preloaded PolicyCache. Zero RPC, no governance round-trip.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .policy_cache import EmergencyRule, get_policy_cache


@dataclass(frozen=True)
class HazardAction:
    action: str                  # "halt_trading" | "safe_mode" | "flatten_positions" | "kill"
    reason: str
    hazard_type: str
    severity: str
    flatten: bool = False


class Resolver:
    def resolve(self, hazard: Any) -> HazardAction:
        hazard_type = str(getattr(hazard, "hazard_type", "UNKNOWN"))
        severity = str(getattr(hazard, "severity", "MEDIUM"))
        rule: EmergencyRule | None = get_policy_cache().get(hazard_type)

        if rule is None:
            return HazardAction(
                action=get_policy_cache().get_snapshot().default_action,
                reason=f"no_rule:{hazard_type}",
                hazard_type=hazard_type,
                severity=severity,
            )

        # Severity gate: only trigger if observed severity ≥ rule's threshold.
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        if order.get(severity, 1) < order.get(rule.severity_threshold, 2):
            return HazardAction(
                action="noop",
                reason=f"below_threshold:{severity}<{rule.severity_threshold}",
                hazard_type=hazard_type,
                severity=severity,
            )

        return HazardAction(
            action=rule.action,
            reason=f"policy:{hazard_type}",
            hazard_type=hazard_type,
            severity=severity,
            flatten=rule.flatten_on_trigger,
        )


_resolver: Resolver | None = None
_lock = threading.Lock()


def get_resolver() -> Resolver:
    global _resolver
    if _resolver is None:
        with _lock:
            if _resolver is None:
                _resolver = Resolver()
    return _resolver
