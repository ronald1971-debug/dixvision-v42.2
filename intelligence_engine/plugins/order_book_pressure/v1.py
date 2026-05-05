"""IND-L04 order_book_pressure plugin v1 — Indira learning layer #3.

Complements the trade-flow imbalance signal (``orderflow_imbalance_v1``)
with the **resting-side** view: how skewed the order-book *depth* is
between bid-side and ask-side at the top of book.

Each :class:`MarketTick` is expected to carry ``bid_size`` and
``ask_size`` as string-encoded floats in ``tick.meta`` (the standard
no-PII typed-meta surface; venues that don't supply depth simply
produce no signal). The instantaneous pressure is

    pressure = (bid_size - ask_size) / (bid_size + ask_size)   in [-1, 1]

The plugin maintains a fixed-size FIFO window of pressure samples;
once full, the **mean** pressure defines the signal:

* mean ``>  threshold`` → ``BUY``  (resting bids dominate — buyers are queued up).
* mean ``< -threshold`` → ``SELL`` (resting asks dominate — sellers are queued up).
* otherwise → no emit (book is balanced).

The plugin holds **state** (the rolling window) but **no clocks, no
randomness, no IO**, so a given input tick sequence always yields the
same output sequence (INV-15, TEST-01).

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers" (this
  closes drift item H3.1 from the canonical-rebuild walk: 3rd plugin
  emitting a real :class:`SignalEvent`).
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


def _coerce_float(meta_value: str | None) -> float | None:
    """Parse a ``tick.meta`` string-encoded float; returns None on miss."""
    if meta_value is None:
        return None
    try:
        return float(meta_value)
    except (TypeError, ValueError):
        return None


@dataclass
class OrderBookPressureV1:
    """Third concrete intelligence plugin (IND-L04 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        window_size: FIFO depth of the rolling-pressure buffer (>= 2).
        pressure_threshold: Absolute mean-pressure threshold in
            ``(0, 1]`` above which a directional signal fires.
        confidence_scale: Divisor mapping absolute mean pressure to
            confidence in ``[0, 1]`` (clipped).
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "order_book_pressure_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    window_size: int = 16
    pressure_threshold: float = 0.2
    confidence_scale: float = 0.6
    min_confidence: float = 0.05
    _pressure_window: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not 0.0 < self.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be in (0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._pressure_window = deque(maxlen=self.window_size)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0:
            return ()
        if tick.ask < tick.bid:
            return ()

        bid_size = _coerce_float(tick.meta.get("bid_size"))
        ask_size = _coerce_float(tick.meta.get("ask_size"))
        if bid_size is None or ask_size is None:
            return ()
        if bid_size < 0.0 or ask_size < 0.0:
            return ()

        total = bid_size + ask_size
        if total <= 0.0:
            return ()

        pressure = (bid_size - ask_size) / total  # in [-1, 1]
        self._pressure_window.append(pressure)

        if len(self._pressure_window) < self.window_size:
            return ()

        mean_pressure = sum(self._pressure_window) / float(self.window_size)

        if mean_pressure > self.pressure_threshold:
            side = Side.BUY
        elif mean_pressure < -self.pressure_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(mean_pressure) / self.confidence_scale)
        if confidence < self.min_confidence:
            return ()

        mid = 0.5 * (tick.bid + tick.ask)

        return (
            SignalEvent(
                ts_ns=tick.ts_ns,
                symbol=tick.symbol,
                side=side,
                confidence=confidence,
                plugin_chain=(self.name,),
                meta={
                    "mid": f"{mid:.10f}",
                    "mean_book_pressure": f"{mean_pressure:.6f}",
                    "pressure_threshold": f"{self.pressure_threshold:.6f}",
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
                f"thresh={self.pressure_threshold}"
            ),
        )


__all__ = ["OrderBookPressureV1"]
