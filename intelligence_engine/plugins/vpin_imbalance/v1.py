"""IND-L07 vpin_imbalance plugin v1 — Indira learning layer #6.

Volume-synchronized Probability of Informed Trading (VPIN, Easley
et al. 2012) as a deterministic, tick-level imbalance gate.

Algorithm:

1. Each tick contributes its trade volume into a *bucket* keyed by
   a fixed ``bucket_volume`` quantum.  The aggressor side is
   inferred from the deterministic tick rule: ``last >= ask`` → BUY,
   ``last <= bid`` → SELL, otherwise neutral (split half-half).
2. When a bucket fills (cumulative volume >= ``bucket_volume``),
   it is sealed and pushed onto a FIFO of completed buckets.
3. Once ``window_size`` buckets are sealed, VPIN is the window-mean of
   ``|buy_vol - sell_vol| / total_vol`` per bucket — a number in
   ``[0, 1]`` measuring sustained order-flow imbalance.
4. If ``vpin > vpin_threshold``, the plugin emits a directional signal
   whose side is the sign of the *most recent bucket's* signed
   imbalance — i.e. follow the most recent informed-flow direction
   when sustained imbalance is detected.

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.4 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).
* Easley, López de Prado, O'Hara (2012), "Flow Toxicity and
  Liquidity in a High-Frequency World", RFS.

Note: this is a *tick-rule* approximation of VPIN — without a true
trade tape (signed prints) we infer aggressor side from the
``last`` price relative to bid/ask, which is the canonical Lee-Ready
deterministic substitute and keeps the plugin a pure leaf module.
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
class _Bucket:
    buy_vol: float = 0.0
    sell_vol: float = 0.0

    @property
    def total(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def signed_imbalance(self) -> float:
        t = self.total
        if t <= 0.0:
            return 0.0
        return (self.buy_vol - self.sell_vol) / t

    @property
    def abs_imbalance(self) -> float:
        return abs(self.signed_imbalance)


@dataclass
class VpinImbalanceV1:
    """Sixth concrete intelligence plugin (IND-L07 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        bucket_volume: Volume quantum that seals a bucket (>0).
        window_size: FIFO depth of sealed buckets (>= 2).
        vpin_threshold: VPIN level above which a directional signal
            fires. Must be in ``[0, 1]``.
        confidence_scale: Divisor mapping ``vpin - vpin_threshold``
            slack to confidence in ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "vpin_imbalance_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    bucket_volume: float = 100.0
    window_size: int = 8
    vpin_threshold: float = 0.3
    confidence_scale: float = 0.5
    min_confidence: float = 0.05
    _buckets: deque[_Bucket] = field(init=False, repr=False)
    _open: _Bucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bucket_volume <= 0.0:
            raise ValueError("bucket_volume must be > 0")
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not 0.0 <= self.vpin_threshold <= 1.0:
            raise ValueError("vpin_threshold must be in [0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._buckets = deque(maxlen=self.window_size)
        self._open = _Bucket()

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        if tick.bid <= 0.0 or tick.ask <= 0.0:
            return ()
        if tick.ask < tick.bid:
            return ()
        if tick.volume <= 0.0:
            # No trade volume to allocate — VPIN ignores this tick.
            return ()
        if tick.last <= 0.0:
            return ()

        # Lee-Ready tick-rule aggressor inference.
        if tick.last >= tick.ask:
            buy_share, sell_share = 1.0, 0.0
        elif tick.last <= tick.bid:
            buy_share, sell_share = 0.0, 1.0
        else:
            buy_share, sell_share = 0.5, 0.5

        remaining = float(tick.volume)
        # Allocate volume into the open bucket; seal whenever full.
        # Note: a single tick may seal multiple buckets if its volume
        # exceeds bucket_volume — handled in the loop below.
        while remaining > 0.0:
            capacity = self.bucket_volume - self._open.total
            if capacity <= 0.0:
                self._buckets.append(self._open)
                self._open = _Bucket()
                continue
            allocated = min(remaining, capacity)
            self._open.buy_vol += allocated * buy_share
            self._open.sell_vol += allocated * sell_share
            remaining -= allocated
            if self._open.total >= self.bucket_volume:
                self._buckets.append(self._open)
                self._open = _Bucket()

        if len(self._buckets) < self.window_size:
            return ()

        vpin = sum(b.abs_imbalance for b in self._buckets) / float(self.window_size)
        if vpin <= self.vpin_threshold:
            return ()

        last_bucket = self._buckets[-1]
        last_imbalance = last_bucket.signed_imbalance
        if last_imbalance > 0.0:
            side = Side.BUY
        elif last_imbalance < 0.0:
            side = Side.SELL
        else:
            return ()

        slack = vpin - self.vpin_threshold
        confidence = min(1.0, slack / self.confidence_scale)
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
                    "vpin": f"{vpin:.10f}",
                    "vpin_threshold": f"{self.vpin_threshold:.10f}",
                    "last_bucket_imbalance": f"{last_imbalance:.10f}",
                    "bucket_volume": f"{self.bucket_volume:.10f}",
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
                f"bucket_volume={self.bucket_volume} "
                f"window={self.window_size} "
                f"thresh={self.vpin_threshold}"
            ),
        )


__all__ = ["VpinImbalanceV1"]
