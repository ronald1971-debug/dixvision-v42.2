"""enforcement — Non-bypassable domain boundary guards."""
from .decorators import enforce_full, enforce_governance, record_attribution
from .kill_switch import arm, disarm, is_armed, trigger
from .policy_enforcer import PolicyEnforcer, get_policy_enforcer
from .runtime_guardian import get_runtime_guardian, start_runtime_guardian

__all__ = [
    "enforce_governance",
    "enforce_full",
    "record_attribution",
    "get_runtime_guardian",
    "start_runtime_guardian",
    "get_policy_enforcer",
    "PolicyEnforcer",
    "trigger",
    "is_armed",
    "arm",
    "disarm",
]
