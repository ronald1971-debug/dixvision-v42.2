"""IND-L10 on_chain_pulse plugin v1 — Indira learning layer #9.

Tick-level **on-chain pulse** as a deterministic rolling exchange-
netflow gate.

On-chain analytics feeds (Glassnode, Dune, Etherscan, mempool RPCs)
decorate each :class:`MarketTick` with an ``exchange_netflow`` value
in ``meta`` — the net token flow into centralised exchanges over
the source-attributed window:

* ``netflow > 0`` → coins flowing **into** exchanges (sell intent
  building).
* ``netflow < 0`` → coins flowing **out of** exchanges (HODL intent
  building, often pre-rally).
* ``netflow == 0`` → neutral.

This plugin maintains a rolling FIFO of per-tick netflow samples
and, once full, sums them and emits:

* ``cum_netflow >  netflow_threshold`` → ``SELL`` (sustained
  exchange inflows).
* ``cum_netflow < -netflow_threshold`` → ``BUY`` (sustained exchange
  outflows / HODL-pulse).
* otherwise silent.

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.7 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).

Note: the *upstream* on-chain metric pipeline (RPC pulls, ETL,
schema validation, source-trust scoring) lives elsewhere in
``sensory/onchain``; this leaf merely consolidates an already-
aggregated netflow score into a single signal channel.
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
class OnChainPulseV1:
    """Ninth concrete intelligence plugin (IND-L10 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        meta_key: Key in ``MarketTick.meta`` carrying the per-tick
            netflow value. Default ``"exchange_netflow"``.
        window_size: FIFO depth (>= 2).
        netflow_threshold: Absolute window-cumulative-netflow magnitude
            (>= 0, in tokens or USD) above which a directional signal
            fires.
        confidence_scale: Divisor mapping ``|cum_netflow|`` to
            confidence in ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "on_chain_pulse_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    meta_key: str = "exchange_netflow"
    window_size: int = 16
    netflow_threshold: float = 1000.0
    confidence_scale: float = 5000.0
    min_confidence: float = 0.05
    _samples: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.meta_key:
            raise ValueError("meta_key must be a non-empty string")
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.netflow_threshold < 0.0:
            raise ValueError("netflow_threshold must be >= 0")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._samples = deque(maxlen=self.window_size)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        raw = tick.meta.get(self.meta_key)
        if raw is None:
            return ()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return ()
        # Reject NaN / +inf / -inf via plain comparison (NaN auto-fails).
        if not (-1e18 <= value <= 1e18):
            return ()

        self._samples.append(value)

        if len(self._samples) < self.window_size:
            return ()

        cum_netflow = sum(self._samples)

        if cum_netflow > self.netflow_threshold:
            side = Side.SELL  # sustained inflows = sell pressure
        elif cum_netflow < -self.netflow_threshold:
            side = Side.BUY  # sustained outflows = HODL / accumulate
        else:
            return ()

        confidence = min(1.0, abs(cum_netflow) / self.confidence_scale)
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
                    "cum_netflow": f"{cum_netflow:.10f}",
                    "netflow_threshold": (
                        f"{self.netflow_threshold:.10f}"
                    ),
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
                f"thresh={self.netflow_threshold}"
            ),
        )


__all__ = ["OnChainPulseV1"]
