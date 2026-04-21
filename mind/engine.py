"""
mind/engine.py
DIX VISION v42.2 — Indira Market Execution Brain (FAST PATH OWNER)

CRITICAL DESIGN RULES:
  - Sub-5ms decision loop
  - NO synchronous governance calls
  - Uses precomputed FastRiskCache ONLY
  - Governance state consumed asynchronously
  - Indira is the ONLY entity that calls exchange APIs

Decision tiers:
  BUY/SELL:  signal > threshold + sufficient confidence
  HOLD:      weak signal or mid-range
  DELEGATE:  data/execution quality too low
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mind.intent_producer import IntentProducer, IntentType
from state.ledger.event_store import append_event
from system.fast_risk_cache import get_risk_cache
from system.metrics import get_metrics
from system.time_source import now


@dataclass
class ExecutionEvent:
    """Typed execution event emitted by Indira. Never free-text."""
    event_type: str        # TRADE_EXECUTION | HOLD | DELEGATE
    asset: str
    side: str              # BUY | SELL | NONE
    order_type: str        # MARKET | LIMIT | NONE
    size_usd: float
    price: float
    strategy: str
    confidence: float
    latency_ns: int
    timestamp_utc: str
    allowed: bool = True   # from risk cache check

class IndiraEngine:
    """
    Indira's core decision engine.

    Fast path: market_data → signal → risk_check (cache) → ExecutionEvent
    Total target latency: < 5ms
    """
    def __init__(self, on_execution: Callable[[ExecutionEvent], None] = None) -> None:
        self._intent = IntentProducer()
        self._risk_cache = get_risk_cache()
        self._metrics = get_metrics()
        self._on_execution = on_execution
        self._portfolio_usd = 100_000.0  # Updated by execution feedback

    def process_tick(self, market_data: dict[str, Any]) -> ExecutionEvent:
        """
        Main fast path. Produces typed execution event.
        Called on every market tick. Must complete < 5ms.
        """
        start_ns = time.monotonic_ns()

        signal = float(market_data.get("signal", 0.0))
        asset = str(market_data.get("asset", "UNKNOWN"))
        price = float(market_data.get("price", 0.0))
        data_quality = float(market_data.get("data_quality", 1.0))
        exec_quality = float(market_data.get("execution_confidence", 1.0))
        strategy = str(market_data.get("strategy", "default"))

        # Step 1: Intent classification (< 0.1ms)
        intent = self._intent.classify(
            signal_confidence=abs(signal),
            data_quality=data_quality,
            execution_quality=exec_quality,
        )

        # Step 2: Risk cache check (< 0.01ms, no RPC)
        constraints = self._risk_cache.get()

        if intent.intent_type == IntentType.DELEGATE:
            ev = self._make_event("DELEGATE", asset, "NONE", "NONE",
                                  0.0, price, strategy, intent.confidence, True)
        elif intent.intent_type in {IntentType.HOLD}:
            ev = self._make_event("HOLD", asset, "NONE", "NONE",
                                  0.0, price, strategy, intent.confidence, True)
        else:
            # Determine side
            side = "BUY" if signal > 0.7 else "SELL" if signal < -0.7 else "HOLD"
            if side == "HOLD":
                ev = self._make_event("HOLD", asset, "NONE", "NONE",
                                      0.0, price, strategy, intent.confidence, True)
            else:
                # Compute position size from risk constraints
                size_usd = min(
                    self._portfolio_usd * constraints.circuit_breaker_loss_pct,
                    constraints.max_order_size_usd,
                )
                ok, reason = constraints.allows_trade(size_usd, self._portfolio_usd)
                ev = self._make_event(
                    "TRADE_EXECUTION", asset, side, "MARKET",
                    size_usd, price, strategy, intent.confidence, ok
                )

        # Step 3: Measure and record latency
        latency_ns = time.monotonic_ns() - start_ns
        ev.latency_ns = latency_ns
        latency_ms = latency_ns / 1_000_000
        self._metrics.observe("trade_latency_ms", latency_ms)

        # Alert if latency exceeds fast-path target
        if latency_ms > 5.0:
            from execution.hazard.event_emitter import get_hazard_emitter
            get_hazard_emitter("indira").latency_spike("fast_path", latency_ms, 5.0)

        # Step 4: Write to ledger (non-blocking — queue-based)
        if ev.event_type == "TRADE_EXECUTION" and ev.allowed:
            self._log_to_ledger(ev)

        # Step 5: Callback for downstream (exchange adapters etc.)
        if self._on_execution:
            try:
                self._on_execution(ev)
            except Exception:
                pass

        return ev

    def _make_event(self, event_type: str, asset: str, side: str,
                    order_type: str, size_usd: float, price: float,
                    strategy: str, confidence: float, allowed: bool) -> ExecutionEvent:
        return ExecutionEvent(
            event_type=event_type, asset=asset, side=side,
            order_type=order_type, size_usd=size_usd, price=price,
            strategy=strategy, confidence=confidence, latency_ns=0,
            timestamp_utc=now().utc_time.isoformat(), allowed=allowed,
        )

    def _log_to_ledger(self, ev: ExecutionEvent) -> None:
        try:
            append_event("MARKET", ev.event_type, "indira", {
                "asset": ev.asset, "side": ev.side, "size_usd": ev.size_usd,
                "price": ev.price, "strategy": ev.strategy,
                "confidence": ev.confidence, "latency_ms": ev.latency_ns / 1_000_000,
            })
        except Exception:
            pass  # ledger failure never kills the trading loop

    def update_portfolio_value(self, usd: float) -> None:
        """Called by execution feedback after fills."""
        self._portfolio_usd = max(1.0, usd)

    # Backward-compat evaluate() for tests
    def evaluate(self, market_data: dict[str, Any]) -> IndiraDecision:
        ev = self.process_tick(market_data)
        from dataclasses import dataclass as _dc
        @_dc
        class IndiraDecision:
            decision: str
            intent: Any
        intent = self._intent.classify(
            signal_confidence=abs(float(market_data.get("signal", 0.0))),
            data_quality=float(market_data.get("data_quality", 1.0)),
            execution_quality=float(market_data.get("execution_confidence", 1.0)),
        )
        d = "BUY" if ev.side == "BUY" else "SELL" if ev.side == "SELL" else ev.event_type
        if d == "DELEGATE":
            d = "DELEGATE"
        return IndiraDecision(decision=d, intent=intent)
