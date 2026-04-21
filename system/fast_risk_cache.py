"""
system/fast_risk_cache.py
DIX VISION v42.2 — Precomputed Risk Cache (FAST PATH, NO RPC)

Updated asynchronously by Governance.
Consumed synchronously by Indira with zero latency.
Thread-safe reads via atomic reference swap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock

from system.time_source import utc_now


@dataclass(frozen=True)
class RiskConstraints:
    """Precomputed risk limits. Consumed by Indira fast path."""
    max_position_pct: float = 1.0       # max position as % of portfolio
    max_order_size_usd: float = 10_000.0
    volatility_band_high: float = 0.05  # 5%
    volatility_band_low: float = 0.001
    circuit_breaker_drawdown: float = 0.04  # 4%
    circuit_breaker_loss_pct: float = 0.01  # 1% per trade
    trading_allowed: bool = True
    safe_mode: bool = False
    last_updated_utc: str = ""

    def allows_trade(self, size_usd: float, portfolio_usd: float) -> tuple[bool, str]:
        if not self.trading_allowed:
            return False, "trading_not_allowed"
        if self.safe_mode:
            return False, "safe_mode_active"
        # Fail-closed: if we don't know the portfolio size we cannot
        # enforce the per-trade circuit breaker, so refuse rather than
        # silently skipping the percentage check.
        if portfolio_usd <= 0:
            return False, "portfolio_usd_required"
        # Absolute per-order cap governance sets via ConstraintCompiler.
        # Checked BEFORE the percentage rule so a large absolute size
        # on a very large portfolio still gets rejected.
        if size_usd > self.max_order_size_usd:
            return False, (
                f"size_usd_{size_usd:.2f}_exceeds_max_"
                f"{self.max_order_size_usd:.2f}"
            )
        pct = size_usd / portfolio_usd
        if pct > self.circuit_breaker_loss_pct:
            return False, f"size_pct_{pct:.4f}_exceeds_limit_{self.circuit_breaker_loss_pct}"
        return True, "ok"


class FastRiskCache:
    """
    Atomic single-writer, multi-reader risk cache.
    Governance is the sole writer (async).
    Indira reads every tick with zero lock contention.
    """
    def __init__(self) -> None:
        self._constraints = RiskConstraints(
            last_updated_utc=utc_now().isoformat()
        )
        self._lock = RLock()

    def get(self) -> RiskConstraints:
        """Lock-free read (atomic reference in CPython)."""
        return self._constraints

    def update(self, **kwargs) -> RiskConstraints:
        """Governance calls this asynchronously to update constraints."""
        with self._lock:
            self._constraints = replace(
                self._constraints,
                last_updated_utc=utc_now().isoformat(),
                **kwargs
            )
            return self._constraints

    def enter_safe_mode(self) -> RiskConstraints:
        return self.update(safe_mode=True, trading_allowed=False)

    def exit_safe_mode(self) -> RiskConstraints:
        return self.update(safe_mode=False, trading_allowed=True)

    def halt_trading(self, reason: str = "") -> RiskConstraints:
        return self.update(trading_allowed=False)

    def resume_trading(self) -> RiskConstraints:
        return self.update(trading_allowed=True, safe_mode=False)


_cache: FastRiskCache | None = None
_lock = RLock()

def get_risk_cache() -> FastRiskCache:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = FastRiskCache()
    return _cache
