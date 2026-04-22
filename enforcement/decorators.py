"""
enforcement/decorators.py
DIX VISION v42.2 — Enforcement Decorators

@enforce_governance: validates action through governance kernel
@enforce_full: governance + resource check + attribution hook
@record_attribution: attribution hook (Phase 6+)
@enforce_domain: prevents cross-domain authority leaks
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any


def _bind_arguments(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Resolve positional + keyword args against ``fn``'s signature.

    Without this, ``@enforce_full`` / ``@enforce_governance`` read the
    sizing fields exclusively from ``**kwargs`` and callers that pass
    them positionally (e.g. ``place_trade("BTCUSDT", 5.0)``) bypass the
    governance check silently.
    """
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        # Fallback: callee has *args / **kwargs only; use kwargs as-is
        return dict(kwargs)


def enforce_governance(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from governance.kernel import ActionRequest, get_kernel
        bound = _bind_arguments(fn, args, kwargs)
        request = ActionRequest(
            action=fn.__name__,
            domain=bound.get("_domain", kwargs.get("_domain", "MARKET")),
            payload={"kwargs": {k: str(v) for k, v in bound.items()
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
        bound = _bind_arguments(fn, args, kwargs)
        trade_size_pct = float(bound.get("trade_size_pct", 0.0) or 0.0)
        # Forward the absolute sizing fields only when the caller
        # actually supplied them, so the governance kernel's own
        # portfolio_usd fallback (100_000) kicks in for callers that
        # only specify a percentage. Explicitly passing 0 here would
        # trigger fast_risk_cache.allows_trade's "portfolio_usd_required"
        # fail-closed and reject every percentage-only trade.
        payload: dict[str, Any] = {
            "kwargs": {"trade_size_pct": trade_size_pct},
            "trade_size_pct": trade_size_pct,
        }
        if bound.get("size_usd") is not None:
            payload["size_usd"] = float(bound["size_usd"])
        if bound.get("portfolio_usd") is not None:
            payload["portfolio_usd"] = float(bound["portfolio_usd"])
        request = ActionRequest(
            action=fn.__name__, domain="MARKET",
            payload=payload,
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
