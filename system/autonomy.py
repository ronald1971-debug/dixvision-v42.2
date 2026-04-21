"""
system.autonomy — three-tier autonomy mode register.

Orthogonal to :mod:`governance.mode_manager` (which tracks system *health*:
NORMAL / DEGRADED / SAFE_MODE / EMERGENCY_HALT).  This module tracks how
much latitude the operator has delegated to INDIRA:

    USER_CONTROLLED   every intent waits for an explicit operator
                      approve/deny click.  INDIRA only *proposes*.
                      Default on first boot.
    SEMI_AUTO         INDIRA may execute autonomously **inside** a
                      per-mode budget (per-trade size, per-hour count).
                      Anything outside the envelope becomes an
                      approval request.
    FULL_AUTO         INDIRA executes freely subject to the fast-risk
                      cache, wallet policy, dead-man and kill-switch.
                      Every fill still lands in the ledger; the
                      operator can kill-switch instantly.

Transitions are operator-only (enforced by ``security.operator`` in
the cockpit path) and every transition writes a
``GOVERNANCE/AUTONOMY_MODE_CHANGED`` ledger event.  A bounded
short-window counter (``_recent_autos``) tracks trades executed
without approval in the last hour for the SEMI_AUTO throttle.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

from state.ledger.writer import get_writer
from system.time_source import utc_now


class AutonomyMode(str, Enum):
    USER_CONTROLLED = "USER_CONTROLLED"
    SEMI_AUTO = "SEMI_AUTO"
    FULL_AUTO = "FULL_AUTO"


@dataclass(frozen=True)
class AutonomyBudget:
    """Per-mode envelope beyond which a trade requires approval."""
    max_size_usd: float
    max_trades_per_hour: int
    auto_asset_allowed: bool  # False: new assets still need per-asset approval


_DEFAULT_BUDGETS: dict[AutonomyMode, AutonomyBudget] = {
    AutonomyMode.USER_CONTROLLED: AutonomyBudget(0.0, 0, False),
    AutonomyMode.SEMI_AUTO: AutonomyBudget(50.0, 3, False),
    AutonomyMode.FULL_AUTO: AutonomyBudget(float("inf"),
                                           1_000_000, True),
}


@dataclass
class AutonomyStatus:
    mode: AutonomyMode
    budget: AutonomyBudget
    trades_last_hour: int
    last_changed_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "budget": {
                "max_size_usd": self.budget.max_size_usd,
                "max_trades_per_hour": self.budget.max_trades_per_hour,
                "auto_asset_allowed": self.budget.auto_asset_allowed,
            },
            "trades_last_hour": self.trades_last_hour,
            "last_changed_utc": self.last_changed_utc,
        }


class AutonomyManager:
    """Thread-safe register for the current autonomy mode.

    Reads are lock-free (single volatile reference); writes serialize
    through ``_lock`` so the transition invariant holds across
    concurrent cockpit calls.  This matches the pattern used by
    :mod:`system.fast_risk_cache` and :mod:`governance.mode_manager`.
    """

    def __init__(self) -> None:
        self._mode: AutonomyMode = AutonomyMode.USER_CONTROLLED
        self._budgets: dict[AutonomyMode, AutonomyBudget] = dict(_DEFAULT_BUDGETS)
        self._recent_autos: Deque[float] = deque(maxlen=10_000)
        self._last_changed_utc: str = ""
        self._lock = threading.Lock()

    # ---- read paths (cheap, no allocation) ---------------------------
    def mode(self) -> AutonomyMode:
        return self._mode

    def status(self) -> AutonomyStatus:
        with self._lock:
            return AutonomyStatus(
                mode=self._mode,
                budget=self._budgets[self._mode],
                trades_last_hour=self._count_last_hour_locked(),
                last_changed_utc=self._last_changed_utc,
            )

    # ---- write paths -------------------------------------------------
    def transition(self, new_mode: AutonomyMode, *,
                   operator_id: str = "operator",
                   reason: str = "") -> AutonomyStatus:
        """Switch mode. Every transition is ledger-logged."""
        with self._lock:
            old = self._mode
            self._mode = new_mode
            self._last_changed_utc = _iso_now()
            snapshot = AutonomyStatus(
                mode=self._mode,
                budget=self._budgets[self._mode],
                trades_last_hour=self._count_last_hour_locked(),
                last_changed_utc=self._last_changed_utc,
            )
        try:
            get_writer().write(
                "GOVERNANCE", "AUTONOMY_MODE_CHANGED", "system.autonomy",
                {"from": old.value, "to": new_mode.value,
                 "operator": operator_id, "reason": reason},
            )
        except Exception:
            pass
        return snapshot

    def set_budget(self, mode: AutonomyMode, budget: AutonomyBudget, *,
                   operator_id: str = "operator") -> None:
        with self._lock:
            self._budgets[mode] = budget
        try:
            get_writer().write(
                "GOVERNANCE", "AUTONOMY_BUDGET_CHANGED", "system.autonomy",
                {"mode": mode.value,
                 "max_size_usd": budget.max_size_usd,
                 "max_trades_per_hour": budget.max_trades_per_hour,
                 "auto_asset_allowed": budget.auto_asset_allowed,
                 "operator": operator_id},
            )
        except Exception:
            pass

    # ---- fast-path gate ----------------------------------------------
    def allows_auto(self, *, size_usd: float,
                    asset_known: bool = True) -> tuple[bool, str]:
        """Return (ok, reason).

        ``ok=True`` means INDIRA may execute without an individual
        operator click; ``ok=False`` means the trade must be staged
        as an approval request.  The hot path checks this BEFORE
        calling the adapter.
        """
        with self._lock:
            mode = self._mode
            budget = self._budgets[mode]
            if mode is AutonomyMode.USER_CONTROLLED:
                return False, "user_controlled_requires_approval"
            if size_usd > budget.max_size_usd:
                return False, "size_exceeds_autonomy_budget"
            if not budget.auto_asset_allowed and not asset_known:
                return False, "new_asset_requires_approval"
            # Evict any timestamps older than 1 hour, then check count.
            cutoff = _epoch_now() - 3600.0
            while self._recent_autos and self._recent_autos[0] < cutoff:
                self._recent_autos.popleft()
            if len(self._recent_autos) >= budget.max_trades_per_hour:
                return False, "autonomy_rate_exceeded"
        return True, "ok"

    def record_auto_trade(self) -> None:
        with self._lock:
            self._recent_autos.append(_epoch_now())

    def _count_last_hour_locked(self) -> int:
        cutoff = _epoch_now() - 3600.0
        while self._recent_autos and self._recent_autos[0] < cutoff:
            self._recent_autos.popleft()
        return len(self._recent_autos)


def _iso_now() -> str:
    n = utc_now()
    return n.isoformat() if n else ""


def _epoch_now() -> float:
    n = utc_now()
    return n.timestamp() if n else 0.0


_mgr: AutonomyManager | None = None
_mgr_lock = threading.Lock()


def get_autonomy() -> AutonomyManager:
    global _mgr
    if _mgr is None:
        with _mgr_lock:
            if _mgr is None:
                _mgr = AutonomyManager()
    return _mgr


__all__ = [
    "AutonomyMode", "AutonomyBudget", "AutonomyStatus",
    "AutonomyManager", "get_autonomy",
]
