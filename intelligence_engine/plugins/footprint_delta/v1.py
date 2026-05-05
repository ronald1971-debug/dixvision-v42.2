"""IND-L08 footprint_delta plugin v1 — Indira learning layer #7.

Per-tick **footprint delta** as a deterministic CVD-style signal.

A footprint chart attributes each trade's volume to either the bid
side or the ask side based on aggressor inference; the *delta* is
the running ``buy_vol - sell_vol``.  This plugin maintains:

* a rolling FIFO window of per-tick deltas (each tick contributes a
  signed delta = ``+volume`` if aggressor is BUY, ``-volume`` if
  SELL, ``0`` if neutral / mid-price print).
* the **window-cumulative delta** — the sum of the FIFO — once full.

When the cumulative delta sustains beyond the configured threshold,
the plugin emits a directional signal in the dominant direction:

* ``cum_delta >  delta_threshold`` → ``BUY``  (sustained net buying).
* ``cum_delta < -delta_threshold`` → ``SELL`` (sustained net selling).

Aggressor side is inferred via the Lee-Ready tick rule:
``last >= ask`` → BUY, ``last <= bid`` → SELL, else neutral.

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.5 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).

Note: this is **distinct** from
``intelligence_engine/plugins/vpin_imbalance/v1.py`` — VPIN
volume-buckets and reports a normalised toxicity ratio in
``[0, 1]``; this plugin reports a *raw signed cumulative delta*
in volume-units, which is the direct footprint-chart quantity
traders use to detect absorption / sustained pressure.
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
class FootprintDeltaV1:
    """Seventh concrete intelligence plugin (IND-L08 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        window_size: FIFO depth of the per-tick delta buffer (>= 2).
        delta_threshold: Absolute window-cumulative-delta threshold
            (>= 0, in volume units) above which a directional signal
            fires.
        confidence_scale: Divisor mapping ``|cum_delta|`` to
            confidence in ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "footprint_delta_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    window_size: int = 16
    delta_threshold: float = 50.0
    confidence_scale: float = 100.0
    min_confidence: float = 0.05
    _deltas: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.delta_threshold < 0.0:
            raise ValueError("delta_threshold must be >= 0")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._deltas = deque(maxlen=self.window_size)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0:
            return ()
        if tick.ask < tick.bid:
            return ()
        if tick.last <= 0.0:
            return ()
        if tick.volume < 0.0:
            return ()

        # Lee-Ready tick-rule aggressor inference -> signed delta.
        if tick.last >= tick.ask:
            delta = float(tick.volume)
        elif tick.last <= tick.bid:
            delta = -float(tick.volume)
        else:
            delta = 0.0

        self._deltas.append(delta)

        if len(self._deltas) < self.window_size:
            return ()

        cum_delta = sum(self._deltas)

        if cum_delta > self.delta_threshold:
            side = Side.BUY
        elif cum_delta < -self.delta_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(cum_delta) / self.confidence_scale)
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
                    "cum_delta": f"{cum_delta:.10f}",
                    "delta_threshold": f"{self.delta_threshold:.10f}",
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
                f"thresh={self.delta_threshold}"
            ),
        )


__all__ = ["FootprintDeltaV1"]
