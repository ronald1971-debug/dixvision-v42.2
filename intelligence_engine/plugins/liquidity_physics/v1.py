"""IND-L05 liquidity_physics plugin v1 — Indira learning layer #4.

Newton-style **impulse** signal: each tick contributes a signed
``impulse = volume * (mid_t - mid_{t-1})``.  A fixed-size FIFO window
of impulses is averaged once full; the window-mean is a "physical"
estimate of liquidity-weighted directional drift.

* mean impulse ``> +impulse_threshold`` → ``BUY``  (sustained
  liquidity-weighted upward drift).
* mean impulse ``< -impulse_threshold`` → ``SELL`` (sustained
  liquidity-weighted downward drift).
* otherwise → no emit (incoherent / low-energy book).

This plugin complements:

* ``microstructure_v1`` (per-tick mid-distance signal),
* ``order_book_pressure_v1`` (resting-side depth-skew signal),

by reading the **flow** dimension as Newtonian impulse rather than
trade-flow signed volume — a venue without bid/ask sizes still
yields a usable signal here.

The plugin holds **state** (the rolling window + previous mid) but
**no clocks, no randomness, no IO**, so a given input tick sequence
always yields the same output sequence (INV-15, TEST-01).

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers" (this
  closes drift item H3.2 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from core.contracts.engine import (
    HealthState,
    HealthStatus,
    PluginLifecycle,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick


@dataclass
class LiquidityPhysicsV1:
    """Fourth concrete intelligence plugin (IND-L05 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        window_size: FIFO depth of the rolling-impulse buffer (>= 2).
        impulse_threshold: Absolute window-mean-impulse threshold
            (>= 0 in price * volume units) above which a directional
            signal fires.
        confidence_scale: Divisor mapping absolute mean impulse to
            confidence in ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "liquidity_physics_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    window_size: int = 16
    impulse_threshold: float = 0.05
    confidence_scale: float = 0.5
    min_confidence: float = 0.05
    _impulse_window: deque[float] = field(init=False, repr=False)
    _prev_mid: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.impulse_threshold < 0.0:
            raise ValueError("impulse_threshold must be >= 0")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._impulse_window = deque(maxlen=self.window_size)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0:
            return ()
        if tick.ask < tick.bid:
            return ()
        if tick.volume < 0.0:
            return ()

        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return ()

        prev = self._prev_mid
        self._prev_mid = mid
        if prev is None:
            return ()

        velocity = mid - prev
        impulse = float(tick.volume) * velocity
        self._impulse_window.append(impulse)

        if len(self._impulse_window) < self.window_size:
            return ()

        mean_impulse = sum(self._impulse_window) / float(self.window_size)

        if mean_impulse > self.impulse_threshold:
            side = Side.BUY
        elif mean_impulse < -self.impulse_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(mean_impulse) / self.confidence_scale)
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
                    "mean_impulse": f"{mean_impulse:.10f}",
                    "impulse_threshold": f"{self.impulse_threshold:.10f}",
                    "window_size": f"{self.window_size}",
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
                f"window={self.window_size} "
                f"thresh={self.impulse_threshold}"
            ),
        )


__all__ = ["LiquidityPhysicsV1"]
