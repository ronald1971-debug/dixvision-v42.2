"""
interrupt/policy_cache.py
Preloaded emergency-policy lookup. Compiled once from governance rules and
never re-evaluated at emission time (deterministic fast path).
"""
from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmergencyRule:
    hazard_type: str
    action: str                       # "halt_trading" | "safe_mode" | "flatten_positions" | "kill"
    severity_threshold: str = "HIGH"  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    flatten_on_trigger: bool = False


@dataclass(frozen=True)
class EmergencyPolicySnapshot:
    rules: dict[str, EmergencyRule] = field(default_factory=dict)
    default_action: str = "safe_mode"


# Default hazard-type → action map. Can be replaced at runtime by governance.
_DEFAULT_RULES: dict[str, EmergencyRule] = {
    "FEED_SILENCE":      EmergencyRule("FEED_SILENCE",      "safe_mode",         "HIGH"),
    "LATENCY_SPIKE":     EmergencyRule("LATENCY_SPIKE",     "safe_mode",         "HIGH"),
    "CLOCK_DRIFT":       EmergencyRule("CLOCK_DRIFT",       "halt_trading",      "HIGH"),
    "EXCHANGE_OFFLINE":  EmergencyRule("EXCHANGE_OFFLINE",  "safe_mode",         "HIGH"),
    "BAD_QUOTE":         EmergencyRule("BAD_QUOTE",         "halt_trading",      "MEDIUM"),
    "AUTH_FAILURE":      EmergencyRule("AUTH_FAILURE",      "halt_trading",      "HIGH"),
    "RISK_BREACH":       EmergencyRule("RISK_BREACH",       "flatten_positions", "HIGH", True),
    "INTEGRITY_BREACH":  EmergencyRule("INTEGRITY_BREACH",  "kill",              "CRITICAL"),
    "HEARTBEAT_TIMEOUT": EmergencyRule("HEARTBEAT_TIMEOUT", "kill",              "CRITICAL"),
}


class PolicyCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: EmergencyPolicySnapshot = EmergencyPolicySnapshot(
            rules=dict(_DEFAULT_RULES),
            default_action="safe_mode",
        )

    def get_snapshot(self) -> EmergencyPolicySnapshot:
        with self._lock:
            return self._snapshot

    def get(self, hazard_type: str) -> EmergencyRule | None:
        with self._lock:
            return self._snapshot.rules.get(hazard_type)

    def replace(self, snapshot: EmergencyPolicySnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def rules(self) -> Iterable[EmergencyRule]:
        with self._lock:
            return list(self._snapshot.rules.values())


_cache: PolicyCache | None = None
_lock = threading.Lock()


def get_policy_cache() -> PolicyCache:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = PolicyCache()
    return _cache
