"""
governance/kernel.py
DIX VISION v42.2 — Governance Kernel (Async Control Plane)

Processes THREE input classes:
  A. MARKET_INTENT (from Indira)
  B. SYSTEM_INTENT (from Dyon)
  C. SYSTEM_HAZARD_EVENT (from Dyon via hazard bus)

Produces:
  EXECUTION_APPROVED | EXECUTION_REJECTED | EXECUTION_MODIFIED | SAFE_MODE

CRITICAL: Governance is async. It NEVER blocks Indira's fast path.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from execution.hazard.async_bus import HazardEvent, get_hazard_bus
from execution.hazard.severity_classifier import (
    classify_response,
    should_enter_safe_mode,
    should_halt_trading,
)
from state.ledger.event_store import append_event
from system.fast_risk_cache import get_risk_cache
from system.state import get_state_manager


class GovernanceOutcome(str, Enum):
    APPROVED = "EXECUTION_APPROVED"
    REJECTED = "EXECUTION_REJECTED"
    MODIFIED = "EXECUTION_MODIFIED"
    SAFE_MODE = "SYSTEM_SAFE_MODE_TRIGGERED"
    HALT = "TRADING_HALTED"

@dataclass
class GovernanceDecision:
    outcome: GovernanceOutcome
    reason: str
    modifications: dict[str, Any] = field(default_factory=dict)
    allowed: bool = True  # compatibility alias

    def __post_init__(self) -> None:
        self.allowed = self.outcome in {
            GovernanceOutcome.APPROVED, GovernanceOutcome.MODIFIED
        }

@dataclass
class ActionRequest:
    action: str
    domain: str  # "MARKET" | "SYSTEM"
    payload: dict[str, Any] = field(default_factory=dict)

class GovernanceKernel:
    """
    Single governance arbiter. Event-driven. Non-blocking.

    Trade decisions use the fast risk cache (precomputed).
    Hazard processing updates the cache asynchronously.
    """

    def __init__(self) -> None:
        self._risk_cache = get_risk_cache()
        self._state_mgr = get_state_manager()
        self._listeners: list[Callable[[GovernanceDecision], None]] = []
        self._lock = threading.Lock()

        # Subscribe to hazard bus
        bus = get_hazard_bus()
        bus.subscribe(self._on_hazard)

    def evaluate(self, request: ActionRequest) -> GovernanceDecision:
        """
        Synchronous evaluation for compatibility.
        Uses only the precomputed risk cache — no RPC.
        """
        if request.domain == "MARKET":
            return self._evaluate_market(request)
        elif request.domain == "SYSTEM":
            return self._evaluate_system(request)
        return GovernanceDecision(GovernanceOutcome.REJECTED, "unknown_domain")

    def _evaluate_market(self, request: ActionRequest) -> GovernanceDecision:
        constraints = self._risk_cache.get()
        payload = request.payload
        # kwargs may be nested (from enforce_full) or flat (direct calls)
        kwargs = payload.get("kwargs", payload)

        # Fail-closed: no trading allowed
        ok, reason = constraints.allows_trade(
            size_usd=float(kwargs.get("size_usd", payload.get("size_usd", 0.0))),
            portfolio_usd=float(kwargs.get("portfolio_usd", payload.get("portfolio_usd", 100_000.0))),
        )
        if not ok:
            decision = GovernanceDecision(GovernanceOutcome.REJECTED, reason)
        else:
            # Check trade size floor (1%) — convert from string if needed.
            # Use ``is not None`` so a missing key (defaulted to ``None``)
            # is NOT silently treated the same as an explicit ``0``,
            # which would let a trade with no ``trade_size_pct`` field
            # slip past the 1% circuit-breaker check.
            raw_size = kwargs.get("trade_size_pct",
                                  payload.get("trade_size_pct", None))
            size_pct = float(raw_size) if raw_size is not None else 0.0
            if size_pct > constraints.circuit_breaker_loss_pct * 100:
                decision = GovernanceDecision(
                    GovernanceOutcome.REJECTED,
                    f"trade_size_pct={size_pct:.2f} exceeds 1% floor",
                )
            else:
                decision = GovernanceDecision(GovernanceOutcome.APPROVED, "risk_validated")

        # Log decision to ledger (async safe)
        try:
            append_event("GOVERNANCE", "MARKET_DECISION", "governance.kernel", {
                "action": request.action, "outcome": decision.outcome.value,
                "reason": decision.reason,
            })
        except Exception:
            pass

        self._notify(decision)
        return decision

    def _evaluate_system(self, request: ActionRequest) -> GovernanceDecision:
        """Evaluate a system maintenance request from Dyon."""
        action = request.action
        allowed_system_actions = {
            "RESTART_SERVICE", "APPLY_PATCH", "ROLLBACK_PATCH",
            "BACKUP_DATA", "ROTATE_LOGS", "HEALTH_CHECK",
        }
        if action not in allowed_system_actions:
            return GovernanceDecision(GovernanceOutcome.REJECTED,
                                      f"unknown_system_action:{action}")
        decision = GovernanceDecision(GovernanceOutcome.APPROVED, "system_action_valid")
        try:
            append_event("GOVERNANCE", "SYSTEM_DECISION", "governance.kernel", {
                "action": action, "outcome": decision.outcome.value,
            })
        except Exception:
            pass
        return decision

    def _on_hazard(self, event: HazardEvent) -> None:
        """
        Handle SYSTEM_HAZARD_EVENT from Dyon.
        Updates risk cache asynchronously. NEVER blocks trading.
        """
        try:
            response_action = classify_response(event)

            if should_halt_trading(event):
                self._risk_cache.halt_trading(reason=event.hazard_type.value)
                self._state_mgr.update(trading_allowed=False)
                self._state_mgr.increment("active_hazards", 1)
                decision_type = GovernanceOutcome.HALT
            elif should_enter_safe_mode(event):
                self._risk_cache.enter_safe_mode()
                self._state_mgr.update(governance_mode="SAFE_MODE")
                decision_type = GovernanceOutcome.SAFE_MODE
            else:
                decision_type = GovernanceOutcome.APPROVED  # observe only

            # Log governance response to ledger
            append_event("GOVERNANCE", "HAZARD_RESPONSE", "governance.kernel", {
                "hazard_type": event.hazard_type.value,
                "severity": event.severity.value,
                "response_action": response_action,
                "governance_outcome": decision_type.value,
            })
        except Exception:
            pass  # governance failure never crashes the system

    def subscribe(self, handler: Callable[[GovernanceDecision], None]) -> None:
        with self._lock:
            self._listeners.append(handler)

    def _notify(self, decision: GovernanceDecision) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for l in listeners:
            try:
                l(decision)
            except Exception:
                pass

    def evaluate_boot(self, state: Any) -> GovernanceDecision:
        health = getattr(state, "health", 1.0)
        mode = getattr(state, "mode", "INIT")
        if health < 0.3:
            return GovernanceDecision(GovernanceOutcome.REJECTED,
                                      f"health_too_low:{health:.2f}")
        return GovernanceDecision(GovernanceOutcome.APPROVED, "boot_approved")

_kernel: GovernanceKernel | None = None
_kernel_lock = threading.Lock()

def get_kernel() -> GovernanceKernel:
    global _kernel
    if _kernel is None:
        with _kernel_lock:
            if _kernel is None:
                _kernel = GovernanceKernel()
    return _kernel
