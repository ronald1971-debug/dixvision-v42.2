"""
mind/fast_execute.py
DIX VISION v42.2 — Indira HOT-PATH Executor

This is the ONLY path a live signal should travel on:

    Indira signal
        → FastRiskCache (in-memory, sync, no RPC)
        → AdapterRouter → exchange adapter
        → ledger.writer.async (post-facto)

It MUST NOT:
  - call governance directly
  - call a policy_engine evaluate() in the hot path
  - block on the ledger (writer is async)
  - touch Dyon / system_monitor

Governance mutates FastRiskCache asynchronously elsewhere; this path
only *reads* the cache. That is the manifest §4/§13 invariant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.authority import Domain, scope
from execution.adapter_router import get_adapter_router
from state.ledger.writer import get_writer
from system.fast_risk_cache import get_risk_cache
from system.metrics import get_metrics


@dataclass
class FastExecuteResult:
    ok: bool
    reason: str
    adapter: str
    order: dict | None
    latency_ns: int


def fast_execute_trade(
    asset: str,
    side: str,
    size_usd: float,
    order_type: str = "MARKET",
    portfolio_usd: float = 0.0,
) -> FastExecuteResult:
    """
    Synchronous, non-governed trade submission.

    Caller is Indira (market domain). Governance is NOT consulted here;
    it has already shaped the risk constraints that the cache exposes.

    ``portfolio_usd`` MUST be a positive value. If the caller cannot
    supply a current portfolio value, trading is refused (fail-closed
    per manifest: a missing portfolio size means the per-trade
    circuit breaker cannot be enforced).
    """
    t0 = time.perf_counter_ns()
    with scope(Domain.MARKET):
        rc = get_risk_cache().get()

        # Fail-closed: the cache itself gates trading.
        if not rc.trading_allowed:
            _audit_reject(asset, side, size_usd, "trading_disallowed")
            return FastExecuteResult(False, "trading_disallowed", "", None,
                                     time.perf_counter_ns() - t0)
        if rc.safe_mode:
            _audit_reject(asset, side, size_usd, "safe_mode")
            return FastExecuteResult(False, "safe_mode", "", None,
                                     time.perf_counter_ns() - t0)

        # Fail-closed on missing portfolio size: without it the
        # per-trade loss cap cannot be evaluated, so we refuse the trade.
        if portfolio_usd <= 0:
            _audit_reject(asset, side, size_usd, "portfolio_usd_required")
            return FastExecuteResult(False, "portfolio_usd_required", "", None,
                                     time.perf_counter_ns() - t0)

        ok, reason = rc.allows_trade(size_usd=size_usd, portfolio_usd=portfolio_usd)
        if not ok:
            _audit_reject(asset, side, size_usd, reason)
            return FastExecuteResult(False, reason, "", None,
                                     time.perf_counter_ns() - t0)

        adapter = get_adapter_router().route(asset)
        if adapter is None:
            _audit_reject(asset, side, size_usd, "no_adapter")
            return FastExecuteResult(False, "no_adapter", "", None,
                                     time.perf_counter_ns() - t0)

        order = adapter.place_order(
            symbol=asset, side=side, size=size_usd, order_type=order_type
        )

        latency_ns = time.perf_counter_ns() - t0

        # Async ledger write; never blocks the hot path.
        get_writer().write(
            "MARKET",
            "TRADE_EXECUTION",
            "mind.fast_execute",
            {
                "asset": asset,
                "side": side,
                "size_usd": size_usd,
                "order_type": order_type,
                "order": order,
                "latency_ns": latency_ns,
            },
        )
        try:
            get_metrics().observe("fast_execute_latency_ms", latency_ns / 1e6)
        except Exception:
            pass

        return FastExecuteResult(
            ok=True,
            reason="ok",
            adapter=getattr(adapter, "name", type(adapter).__name__),
            order=order,
            latency_ns=latency_ns,
        )


def _audit_reject(asset: str, side: str, size_usd: float, reason: str) -> None:
    try:
        get_writer().write(
            "MARKET",
            "ORDER_REJECTED",
            "mind.fast_execute",
            {"asset": asset, "side": side, "size_usd": size_usd, "reason": reason},
        )
    except Exception:
        pass


__all__ = ["FastExecuteResult", "fast_execute_trade"]
