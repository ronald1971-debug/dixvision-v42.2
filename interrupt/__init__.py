"""interrupt — Deterministic hazard interrupt path.

Flow:
    Dyon emits SYSTEM_HAZARD_EVENT
        → interrupt.dispatcher routes it
        → interrupt.resolver maps hazard -> action (via preloaded policy_cache)
        → interrupt.interrupt_executor invokes execution.emergency_executor

This path is independent of Indira's hot path. It is deterministic and never
re-evaluates policy at emission time.
"""
from .dispatcher import Dispatcher, get_dispatcher
from .interrupt_executor import InterruptExecutor, get_interrupt_executor
from .policy_cache import PolicyCache, get_policy_cache
from .resolver import HazardAction, Resolver, get_resolver

__all__ = [
    "Dispatcher",
    "get_dispatcher",
    "PolicyCache",
    "get_policy_cache",
    "Resolver",
    "get_resolver",
    "HazardAction",
    "InterruptExecutor",
    "get_interrupt_executor",
]
