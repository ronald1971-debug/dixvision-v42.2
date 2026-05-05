"""IND-L06 regime_classifier plugin v1 — Indira learning layer #5.

Tick-level volatility-and-drift regime classifier.

Maintains a rolling FIFO of mid-price log-style returns
(``r_t = (mid_t - mid_{t-1}) / mid_{t-1}``).  Once full, two
statistics are computed over the window:

* ``vol``  — population standard deviation of returns
  (volatility level).
* ``drift`` — arithmetic mean of returns (signed directional bias).

Three regimes are then resolved:

* ``vol > vol_high_threshold``                           → ``HIGH_VOL``  (no emit)
* ``vol <= vol_low_threshold``  AND ``drift > drift_threshold``  → ``LOW_VOL_BULL`` → ``BUY``
* ``vol <= vol_low_threshold``  AND ``drift < -drift_threshold`` → ``LOW_VOL_BEAR`` → ``SELL``
* otherwise (mid-vol band, or low-vol no-drift)          → ``RANGE``    (no emit)

The intent is **risk-on calm-market drift capture**: when vol is
elevated the plugin stays silent (deferring to other plugins or
to the hazard layer), and only when conditions are calm AND there
is a meaningfully signed drift does it emit.

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.3 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).

Note: this is **distinct** from
``intelligence_engine/macro/regime_engine.py``, which classifies
*macro* (cross-asset, multi-factor) regimes.  This plugin is a
single-symbol tick-stream regime gate; the two are complementary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt

from core.contracts.engine import (
    HealthState,
    HealthStatus,
    PluginLifecycle,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick


@dataclass
class RegimeClassifierV1:
    """Fifth concrete intelligence plugin (IND-L06 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        window_size: FIFO depth of the rolling-returns buffer (>= 2).
        vol_low_threshold: Volatility ``<=`` this is "calm". Must be
            ``>= 0`` and ``<= vol_high_threshold``.
        vol_high_threshold: Volatility ``>`` this is "stormy". Must
            be ``>= vol_low_threshold``.
        drift_threshold: Absolute mean-return threshold above which
            a directional signal fires (in calm vol). Must be ``>= 0``.
        confidence_scale: Divisor mapping absolute mean-return to
            confidence in ``[0, 1]`` (clipped). Must be ``> 0``.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "regime_classifier_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    window_size: int = 16
    vol_low_threshold: float = 0.001
    vol_high_threshold: float = 0.01
    drift_threshold: float = 0.0005
    confidence_scale: float = 0.005
    min_confidence: float = 0.05
    _returns_window: deque[float] = field(init=False, repr=False)
    _prev_mid: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.vol_low_threshold < 0.0:
            raise ValueError("vol_low_threshold must be >= 0")
        if self.vol_high_threshold < self.vol_low_threshold:
            raise ValueError("vol_high_threshold must be >= vol_low_threshold")
        if self.drift_threshold < 0.0:
            raise ValueError("drift_threshold must be >= 0")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._returns_window = deque(maxlen=self.window_size)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0:
            return ()
        if tick.ask < tick.bid:
            return ()

        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return ()

        prev = self._prev_mid
        self._prev_mid = mid
        if prev is None or prev <= 0.0:
            return ()

        ret = (mid - prev) / prev
        self._returns_window.append(ret)

        if len(self._returns_window) < self.window_size:
            return ()

        n = float(self.window_size)
        mean = sum(self._returns_window) / n
        var = sum((r - mean) * (r - mean) for r in self._returns_window) / n
        vol = sqrt(var)

        if vol > self.vol_high_threshold:
            return ()
        if vol > self.vol_low_threshold:
            return ()

        if mean > self.drift_threshold:
            side = Side.BUY
            regime = "low_vol_bull"
        elif mean < -self.drift_threshold:
            side = Side.SELL
            regime = "low_vol_bear"
        else:
            return ()

        confidence = min(1.0, abs(mean) / self.confidence_scale)
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
                    "vol": f"{vol:.10f}",
                    "mean_return": f"{mean:.10f}",
                    "regime": regime,
                    "vol_low_threshold": f"{self.vol_low_threshold:.10f}",
                    "vol_high_threshold": f"{self.vol_high_threshold:.10f}",
                    "drift_threshold": f"{self.drift_threshold:.10f}",
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
                f"vol_low={self.vol_low_threshold} "
                f"vol_high={self.vol_high_threshold} "
                f"drift={self.drift_threshold}"
            ),
        )


__all__ = ["RegimeClassifierV1"]
