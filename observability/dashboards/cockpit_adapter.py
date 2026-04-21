"""
observability/dashboards/cockpit_adapter.py
Builds a single JSON-serializable snapshot for the cockpit endpoint.
"""
from __future__ import annotations

from typing import Any


def build_cockpit_snapshot() -> dict[str, Any]:
    from observability.metrics.metrics_registry import get_metrics_registry
    from state.projectors import (
        get_governance_projector,
        get_market_projector,
        get_portfolio_projector,
        get_system_projector,
    )
    from system.fast_risk_cache import get_risk_cache
    from system.state import get_state_manager

    state = get_state_manager().get()
    rc = get_risk_cache().get()
    mp = get_market_projector().snapshot()
    pp = get_portfolio_projector().snapshot()
    sp = get_system_projector().snapshot()
    gp = get_governance_projector().snapshot()

    return {
        "state": {
            "mode": state.governance_mode,
            "trading_allowed": state.trading_allowed,
            "health": state.health,
        },
        "risk": {
            "max_order_size_usd": rc.max_order_size_usd,
            "circuit_breaker_loss_pct": rc.circuit_breaker_loss_pct,
            "circuit_breaker_drawdown": rc.circuit_breaker_drawdown,
            "safe_mode": rc.safe_mode,
            "trading_allowed": rc.trading_allowed,
        },
        "market": {
            "prices": mp.last_price_by_asset,
            "volumes": mp.last_volume_by_asset,
        },
        "portfolio": {
            "equity_usd": pp.equity_usd,
            "positions": pp.positions,
            "realized_pnl_usd": pp.realized_pnl_usd,
        },
        "system": {
            "boot_complete": sp.boot_complete,
            "last_mode": sp.last_mode,
            "hazard_counts": sp.hazard_counts,
        },
        "governance": {
            "decision_counts": gp.decision_counts,
        },
        "metrics": get_metrics_registry().snapshot(),
    }
