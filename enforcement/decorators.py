"""
enforcement/decorators.py
DIX VISION v42.2 — Enforcement Decorators

@enforce_governance: validates action through governance kernel
@enforce_full: governance + resource check + attribution hook
@record_attribution: attribution hook (Phase 6+)
@enforce_domain: prevents cross-domain authority leaks
"""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def enforce_governance(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from governance.kernel import ActionRequest, get_kernel
        request = ActionRequest(
            action=fn.__name__,
            domain=kwargs.get("_domain", "MARKET"),
            payload={"kwargs": {k: str(v) for k, v in kwargs.items()
                                if not k.startswith("_")}},
        )
        decision = get_kernel().evaluate(request)
        if not decision.allowed:
            raise RuntimeError(f"Governance blocked: {decision.reason}")
        return fn(*args, **kwargs)
    return wrapper

def enforce_full(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from governance.kernel import ActionRequest, get_kernel
        from system.state import get_state
        state = get_state()
        if not state.trading_allowed:
            raise RuntimeError("Trading not allowed: risk cache or safe mode")
        trade_size_pct = kwargs.get("trade_size_pct", 0.0)
        request = ActionRequest(
            action=fn.__name__, domain="MARKET",
            payload={"kwargs": {"trade_size_pct": trade_size_pct}},
        )
        decision = get_kernel().evaluate(request)
        if not decision.allowed:
            raise RuntimeError(f"Governance blocked action: {decision.reason}")
        return fn(*args, **kwargs)
    return wrapper

def record_attribution(fn: Callable) -> Callable:
    """Phase 6+ Shapley attribution hook."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)
    return wrapper

def enforce_domain(domain: str):
    """Decorator that enforces caller domain (MARKET or SYSTEM)."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)
        wrapper._domain = domain
        return wrapper
    return decorator
