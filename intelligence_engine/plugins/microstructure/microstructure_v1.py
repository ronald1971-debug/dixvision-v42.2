"""IND-L02 microstructure plugin v1 — Phase E2.

Deterministic mapping ``MarketTick -> SignalEvent`` based on the trade
print's distance from the bid/ask midpoint:

* ``last`` close to mid (within ``tolerance_bps``) -> ``HOLD``
* ``last`` above mid by more than ``tolerance_bps`` -> ``BUY``
* ``last`` below mid by more than ``tolerance_bps`` -> ``SELL``

Confidence is the (clipped) distance in bps divided by
``confidence_scale_bps``. The plugin holds **no clocks, no randomness,
no IO** so the same tick always produces the same signal (INV-15,
TEST-01).

Refs:

* ``docs/total_recall_index.md`` §4 IND-L02 (market microstructure)
* ``build_plan.md`` §Phase E2
* ``manifest.md`` §0.7 (Plugin Activation Surface)
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.engine import (
    HealthState,
    HealthStatus,
    PluginLifecycle,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick


@dataclass
class MicrostructureV1:
    """First concrete intelligence plugin (IND-L02 v1)."""

    name: str = "microstructure_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    tolerance_bps: float = 2.0
    confidence_scale_bps: float = 50.0
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.tolerance_bps < 0.0:
            raise ValueError("tolerance_bps must be >= 0")
        if self.confidence_scale_bps <= 0.0:
            raise ValueError("confidence_scale_bps must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.last <= 0.0:
            return ()
        if tick.ask < tick.bid:
            # Crossed book: refuse to emit; let System engine raise hazard.
            return ()

        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return ()

        diff_bps = (tick.last - mid) / mid * 10_000.0

        if diff_bps > self.tolerance_bps:
            side = Side.BUY
        elif diff_bps < -self.tolerance_bps:
            side = Side.SELL
        else:
            side = Side.HOLD

        confidence = min(1.0, abs(diff_bps) / self.confidence_scale_bps)
        if confidence < self.min_confidence:
            return ()

        return (
            SignalEvent(
                ts_ns=tick.ts_ns,
                symbol=tick.symbol,
                side=side,
                confidence=confidence,
                plugin_chain=(self.name,),
                meta={
                    "mid": f"{mid:.10f}",
                    "diff_bps": f"{diff_bps:.6f}",
                    "tolerance_bps": f"{self.tolerance_bps:.6f}",
                },
                produced_by_engine="intelligence_engine",
            ),
        )

    def check_self(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.OK,
            detail=(
                f"{self.name} v{self.version} "
                f"lifecycle={self.lifecycle} "
                f"tol={self.tolerance_bps}bps"
            ),
        )


__all__ = ["MicrostructureV1"]
