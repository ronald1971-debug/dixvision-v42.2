"""IND-L03 orderflow imbalance plugin v1 — Indira learning layer #2.

Derives directional pressure from a *rolling window* of signed
trade-flow, complementing the per-tick mid-distance signal that
``microstructure_v1`` emits.

For each :class:`MarketTick` the plugin computes the signed dollar
flow ``volume * sign(last - mid)`` (positive when the print lifted
the offer, negative when it hit the bid) and accumulates it inside a
fixed-size FIFO window. The window-summed imbalance, normalised by
the window's total absolute flow, defines the signal:

* normalised imbalance ``> threshold`` → ``BUY``
* normalised imbalance ``< -threshold`` → ``SELL``
* otherwise → no emit (orderflow neutral)

The plugin holds **state** (the rolling window) but **no clocks, no
randomness, no IO**, so a given input tick sequence always yields the
same output sequence (INV-15, TEST-01). State is wrapped in
:class:`collections.deque` with ``maxlen``, so eviction is
deterministic FIFO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers" (this
  closes drift item B from the canonical-rebuild walk: 2nd plugin
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


@dataclass
class OrderflowImbalanceV1:
    """Second concrete intelligence plugin (IND-L03 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        window_size: FIFO depth of the rolling-flow buffer (>= 2).
        imbalance_threshold: Absolute normalised-imbalance threshold
            in ``(0, 1]`` above which a directional signal fires.
        confidence_scale: Scale factor mapping absolute normalised
            imbalance to confidence in ``[0, 1]`` (clipped).
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "orderflow_imbalance_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    window_size: int = 32
    imbalance_threshold: float = 0.2
    confidence_scale: float = 0.6
    min_confidence: float = 0.05
    _flow_window: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not 0.0 < self.imbalance_threshold <= 1.0:
            raise ValueError("imbalance_threshold must be in (0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._flow_window = deque(maxlen=self.window_size)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.last <= 0.0:
            return ()
        if tick.ask < tick.bid:
            return ()

        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return ()

        if tick.last > mid:
            sign = 1.0
        elif tick.last < mid:
            sign = -1.0
        else:
            sign = 0.0

        signed_flow = sign * float(tick.volume)
        self._flow_window.append(signed_flow)

        if len(self._flow_window) < self.window_size:
            return ()

        net = sum(self._flow_window)
        gross = sum(abs(f) for f in self._flow_window)
        if gross <= 0.0:
            return ()

        normalised = net / gross  # in [-1, 1]

        if normalised > self.imbalance_threshold:
            side = Side.BUY
        elif normalised < -self.imbalance_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(normalised) / self.confidence_scale)
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
                    "normalised_imbalance": f"{normalised:.6f}",
                    "imbalance_threshold": f"{self.imbalance_threshold:.6f}",
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
                f"thresh={self.imbalance_threshold}"
            ),
        )


__all__ = ["OrderflowImbalanceV1"]
