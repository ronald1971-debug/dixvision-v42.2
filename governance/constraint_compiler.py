"""
governance/constraint_compiler.py
Compiles governance rules + risk limits into a ``RiskConstraints`` payload
that Indira's fast path consumes synchronously via the FastRiskCache.
"""
from __future__ import annotations

from dataclasses import dataclass

from immutable_core.constants import AXIOMS
from system.fast_risk_cache import RiskConstraints, get_risk_cache


@dataclass(frozen=True)
class CompiledConstraints:
    constraints: RiskConstraints
    trace: str


class ConstraintCompiler:
    def compile(
        self,
        portfolio_usd: float,
        drawdown_pct: float,
        safe_mode: bool = False,
        trading_allowed: bool = True,
    ) -> CompiledConstraints:
        max_loss_pct = AXIOMS.MAX_LOSS_PER_TRADE_FLOOR_PCT / 100.0  # 0.01
        max_drawdown = AXIOMS.MAX_DRAWDOWN_FLOOR_PCT / 100.0        # 0.04
        # Clamp to [0, max_drawdown] and fall back to the axiom floor
        # when the caller passes zero/negative. The manifest (§1) binds
        # the drawdown ceiling to 4 %; no caller, accidental or
        # adversarial, may raise it above that.
        clamped = max(0.0, min(1.0, drawdown_pct))
        effective_drawdown = min(clamped or max_drawdown, max_drawdown)

        c = RiskConstraints(
            max_position_pct=1.0,
            max_order_size_usd=max(0.0, portfolio_usd * max_loss_pct),
            volatility_band_high=0.05,
            volatility_band_low=0.001,
            circuit_breaker_drawdown=effective_drawdown,
            circuit_breaker_loss_pct=max_loss_pct,
            trading_allowed=trading_allowed and not safe_mode,
            safe_mode=safe_mode,
            last_updated_utc="",
        )
        trace = (
            f"compile(portfolio_usd={portfolio_usd},"
            f"drawdown_pct={drawdown_pct},safe_mode={safe_mode},"
            f"trading_allowed={trading_allowed})"
        )
        return CompiledConstraints(c, trace)

    def publish(self, compiled: CompiledConstraints) -> None:
        cache = get_risk_cache()
        cache.update(
            max_order_size_usd=compiled.constraints.max_order_size_usd,
            circuit_breaker_loss_pct=compiled.constraints.circuit_breaker_loss_pct,
            circuit_breaker_drawdown=compiled.constraints.circuit_breaker_drawdown,
            safe_mode=compiled.constraints.safe_mode,
            trading_allowed=compiled.constraints.trading_allowed,
        )


_cc: ConstraintCompiler | None = None


def get_constraint_compiler() -> ConstraintCompiler:
    global _cc
    if _cc is None:
        _cc = ConstraintCompiler()
    return _cc
