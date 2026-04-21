"""
immutable_core/system_identity.py
DIX VISION v42.2 — System Identity & Hard Behavioral Constraints
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemIdentity:
    forbidden_behaviors: frozenset[str] = field(default_factory=lambda: frozenset({
        "martingale", "unbounded_leverage", "runtime_core_mutation",
        "direct_dyon_trading_execution", "blocking_hazard_loop_in_hot_path",
        "silent_state_modification",
    }))
    required_behaviors: frozenset[str] = field(default_factory=lambda: frozenset({
        "deterministic_replay", "full_audit_trace",
        "governance_pre_approval_for_execution", "immutable_ledger_logging",
    }))
    def is_forbidden(self, b: str) -> bool:
        return b in self.forbidden_behaviors

IDENTITY = SystemIdentity()
