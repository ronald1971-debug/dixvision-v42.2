"""
governance/emergency_policy.py
Source of truth for hazard → action mappings. Exports a snapshot to the
interrupt PolicyCache at boot (and whenever rules change).
"""
from __future__ import annotations

from interrupt.policy_cache import (
    EmergencyPolicySnapshot,
    EmergencyRule,
    get_policy_cache,
)

_CANONICAL_RULES = {
    "FEED_SILENCE":       EmergencyRule("FEED_SILENCE",       "safe_mode",         "HIGH"),
    "LATENCY_SPIKE":      EmergencyRule("LATENCY_SPIKE",      "safe_mode",         "HIGH"),
    "EXECUTION_LATENCY_SPIKE": EmergencyRule("EXECUTION_LATENCY_SPIKE", "safe_mode", "HIGH"),
    "CLOCK_DRIFT":        EmergencyRule("CLOCK_DRIFT",        "halt_trading",      "HIGH"),
    "EXCHANGE_OFFLINE":   EmergencyRule("EXCHANGE_OFFLINE",   "safe_mode",         "HIGH"),
    "EXCHANGE_TIMEOUT":   EmergencyRule("EXCHANGE_TIMEOUT",   "safe_mode",         "HIGH"),
    "BAD_QUOTE":          EmergencyRule("BAD_QUOTE",          "halt_trading",      "MEDIUM"),
    "AUTH_FAILURE":       EmergencyRule("AUTH_FAILURE",       "halt_trading",      "HIGH"),
    "API_CONNECTIVITY_FAILURE": EmergencyRule("API_CONNECTIVITY_FAILURE", "safe_mode", "HIGH"),
    "DATA_CORRUPTION_SUSPECTED": EmergencyRule("DATA_CORRUPTION_SUSPECTED", "halt_trading", "HIGH"),
    "SYSTEM_DEGRADATION": EmergencyRule("SYSTEM_DEGRADATION", "safe_mode",         "HIGH"),
    "MEMORY_PRESSURE":    EmergencyRule("MEMORY_PRESSURE",    "safe_mode",         "HIGH"),
    "CPU_OVERLOAD":       EmergencyRule("CPU_OVERLOAD",       "safe_mode",         "HIGH"),
    "RISK_BREACH":        EmergencyRule("RISK_BREACH",        "flatten_positions", "HIGH", True),
    "INTEGRITY_BREACH":   EmergencyRule("INTEGRITY_BREACH",   "kill",              "CRITICAL"),
    "LEDGER_INCONSISTENCY": EmergencyRule("LEDGER_INCONSISTENCY", "kill",          "CRITICAL"),
    "HEARTBEAT_TIMEOUT":  EmergencyRule("HEARTBEAT_TIMEOUT",  "kill",              "CRITICAL"),
}


def publish_canonical_policy() -> None:
    """Push the canonical hazard→action mapping into the interrupt PolicyCache."""
    snapshot = EmergencyPolicySnapshot(
        rules=dict(_CANONICAL_RULES),
        default_action="safe_mode",
    )
    get_policy_cache().replace(snapshot)


def get_snapshot() -> EmergencyPolicySnapshot:
    return EmergencyPolicySnapshot(
        rules=dict(_CANONICAL_RULES),
        default_action="safe_mode",
    )
