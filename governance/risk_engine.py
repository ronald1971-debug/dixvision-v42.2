"""
governance/risk_engine.py
DIX VISION v42.2 — Async Risk Engine

Monitors portfolio exposure and updates the fast risk cache.
Runs in governance control plane — NEVER on trading hot path.
"""
from __future__ import annotations

import threading

from system.fast_risk_cache import get_risk_cache


class RiskEngine:
    """
    Computes and updates portfolio risk constraints.
    Results pushed to FastRiskCache for zero-latency Indira consumption.
    """
    def __init__(self) -> None:
        self._cache = get_risk_cache()
        self._portfolio_usd = 100_000.0
        self._open_exposure_usd = 0.0
        self._lock = threading.Lock()

    def record_fill(self, size_usd: float, side: str) -> None:
        with self._lock:
            if side == "BUY":
                self._open_exposure_usd += size_usd
            else:
                self._open_exposure_usd = max(0.0, self._open_exposure_usd - size_usd)
        self._recompute()

    def set_portfolio_value(self, usd: float) -> None:
        with self._lock:
            self._portfolio_usd = max(1.0, usd)
        self._recompute()

    def _recompute(self) -> None:
        with self._lock:
            portfolio = self._portfolio_usd
            exposure = self._open_exposure_usd
        exposure_pct = exposure / portfolio if portfolio > 0 else 0.0
        max_order_usd = portfolio * 0.01  # 1% per trade floor
        self._cache.update(
            max_order_size_usd=max_order_usd,
            circuit_breaker_loss_pct=0.01,
        )

_engine: RiskEngine | None = None
_lock = threading.Lock()

def get_risk_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = RiskEngine()
    return _engine
