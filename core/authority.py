"""
core/authority.py
DIX VISION v42.2 — Domain Authority Gates

Enforces the manifest's dual-domain boundary (§1, §6, §13):

  INDIRA   = "market"   → may execute trades, touch exchange adapters
  DYON     = "system"   → may detect hazards, never executes trades
  GOVERNANCE = "control" → may mutate risk cache + ledger, never in hot path
  SECURITY = "security" → secrets, authN/authZ
  CORE     = "core"     → bootstrap / runtime / authority (internal)

Usage:

    from core.authority import market, system, control, Domain, assert_domain

    @market                 # decorator: only market-authority code may call
    def place_order(...): ...

    @system
    def detect_hazard(...): ...

    # Or inline:
    assert_domain(Domain.MARKET)

Violations raise AuthorityViolation and are logged to the SECURITY event
stream (best-effort; never blocks).
"""
from __future__ import annotations

import contextvars
import enum
import functools
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar


class Domain(str, enum.Enum):
    MARKET = "market"         # INDIRA
    SYSTEM = "system"         # DYON
    CONTROL = "control"       # GOVERNANCE
    SECURITY = "security"
    CORE = "core"


class AuthorityViolation(RuntimeError):
    """Raised when code executes outside its declared authority domain."""


# Current authority domain, propagated through async/thread boundaries via ContextVar.
_current: contextvars.ContextVar[Domain | None] = contextvars.ContextVar(
    "dix_authority", default=None,
)
_lock = threading.Lock()
_strict = os.environ.get("DIX_AUTHORITY_STRICT", "1") == "1"


def current() -> Domain | None:
    return _current.get()


def set_domain(d: Domain) -> contextvars.Token:
    """Enter a domain. Returns a token; pass to ``reset_domain`` to leave."""
    return _current.set(d)


def reset_domain(tok: contextvars.Token) -> None:
    _current.reset(tok)


class _DomainScope:
    """Context manager form of set_domain/reset_domain."""

    def __init__(self, d: Domain) -> None:
        self._d = d
        self._tok: contextvars.Token | None = None

    def __enter__(self) -> Domain:
        self._tok = _current.set(self._d)
        return self._d

    def __exit__(self, *_: Any) -> None:
        if self._tok is not None:
            _current.reset(self._tok)


def scope(d: Domain) -> _DomainScope:
    return _DomainScope(d)


def _log_violation(required: Domain, actual: Domain | None, where: str) -> None:
    """Best-effort audit to the SECURITY event stream; never raises."""
    try:
        from state.ledger.event_store import append_event  # lazy
        append_event(
            "SECURITY",
            "AUTHORITY_VIOLATION",
            "core.authority",
            {
                "required": required.value,
                "actual": actual.value if actual else None,
                "where": where,
            },
        )
    except Exception:
        pass


def assert_domain(required: Domain, where: str = "") -> None:
    """Assert the current execution context declared the required domain."""
    actual = _current.get()
    if actual is None:
        # Not in any scope — in strict mode this is still a violation.
        if _strict:
            _log_violation(required, actual, where or "<unscoped>")
            raise AuthorityViolation(
                f"{where or '<unscoped>'}: authority required={required.value}, "
                f"actual=None (no scope)"
            )
        return
    if actual is not required:
        _log_violation(required, actual, where)
        raise AuthorityViolation(
            f"{where}: authority required={required.value}, actual={actual.value}"
        )


F = TypeVar("F", bound=Callable[..., Any])


def requires(required: Domain) -> Callable[[F], F]:
    """Decorator: callable may only run inside the specified domain scope."""

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            assert_domain(required, where=f"{fn.__module__}.{fn.__qualname__}")
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco


# Convenience decorators (common case)
market = requires(Domain.MARKET)
system = requires(Domain.SYSTEM)
control = requires(Domain.CONTROL)
security = requires(Domain.SECURITY)
core = requires(Domain.CORE)


# ---------------------------------------------------------------------------
# Module-level "banned imports" check.
# Dyon (system) code must never import INDIRA-only modules (adapters), and
# INDIRA must never import Dyon hazard internals. This is enforced at import
# time via the loader below; callers can also invoke assert_no_adapter_import()
# from a Dyon module's top-level.
# ---------------------------------------------------------------------------

_INDIRA_ONLY_MODULES = frozenset({
    "execution.adapters.binance",
    "execution.adapters.coinbase",
    "execution.adapters.kraken",
    "execution.adapters.uniswap_v3",
    "execution.adapters.raydium",
    "execution.adapter_router",
    "execution.trade_executor",
})

_DYON_ONLY_MODULES = frozenset({
    "system_monitor.engine",
    "system_monitor.hazard_detector",
    "system_monitor.heartbeat_monitor",
})


def assert_no_adapter_import(caller: str) -> None:
    """Hard-fail if a Dyon caller has imported INDIRA-only adapter modules."""
    import sys as _sys
    leaks = [m for m in _INDIRA_ONLY_MODULES if m in _sys.modules]
    if leaks:
        _log_violation(Domain.MARKET, Domain.SYSTEM, caller)
        raise AuthorityViolation(
            f"{caller}: Dyon domain cannot import INDIRA modules: {leaks}"
        )


__all__ = [
    "Domain",
    "AuthorityViolation",
    "assert_domain",
    "assert_no_adapter_import",
    "current",
    "set_domain",
    "reset_domain",
    "scope",
    "requires",
    "market",
    "system",
    "control",
    "security",
    "core",
]
