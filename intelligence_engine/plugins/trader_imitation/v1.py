"""IND-L11 trader_imitation plugin v1 — Indira learning layer #10.

Tick-level **trader imitation** as a deterministic leader-consensus
gate.

The trader-archetype registry (``registry/trader_archetypes.yaml``,
300 rows shipped in H1) describes 30 archetypes × 5 dimensions. The
upstream observation pipeline (``intelligence_engine.trader_modeling``)
maps live trader feeds into per-tick intent scores in ``[-1, 1]``
(``+1`` = leader is going full long, ``-1`` = full short, ``0`` =
flat). Those scores arrive on the tick under a single meta key
``leader_intents`` as a tuple of floats — one entry per tracked
leader.

This plugin is the deterministic *consumer*:

1. Per tick, take the *mean* of the leader intents (consensus).
2. Maintain a rolling FIFO of the last ``window_size`` consensus
   values.
3. Once full, take the rolling mean of consensus.
4. Emit:

   * ``BUY`` when ``rolling_consensus > consensus_threshold``,
   * ``SELL`` when ``rolling_consensus < -consensus_threshold``,
   * silent otherwise.

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.8 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).

Note: the *upstream* leader-tracking pipeline (TradingView trader
feeds, on-chain follow-the-money, smart-money cluster detection)
lives in ``intelligence_engine.trader_modeling`` and the existing
TradingView adapter. This leaf merely consolidates an already-
mapped leader-intent vector into a single signal channel.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.contracts.engine import (
    HealthState,
    HealthStatus,
    PluginLifecycle,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick


@dataclass
class TraderImitationV1:
    """Tenth concrete intelligence plugin (IND-L11 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        meta_key: Key in ``MarketTick.meta`` carrying the per-tick
            leader-intent vector. Default ``"leader_intents"``.
        window_size: FIFO depth (>= 2).
        min_leaders: Minimum number of valid leader intents required
            for a tick to count toward the rolling mean (>= 1).
        consensus_threshold: Absolute rolling-mean magnitude in
            ``[0, 1]`` above which a directional signal fires.
        confidence_scale: Divisor mapping ``|rolling_consensus|`` to
            confidence in ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "trader_imitation_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    meta_key: str = "leader_intents"
    window_size: int = 8
    min_leaders: int = 2
    consensus_threshold: float = 0.3
    confidence_scale: float = 1.0
    min_confidence: float = 0.05
    _consensus: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.meta_key:
            raise ValueError("meta_key must be a non-empty string")
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.min_leaders < 1:
            raise ValueError("min_leaders must be >= 1")
        if not 0.0 <= self.consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be in [0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._consensus = deque(maxlen=self.window_size)

    @staticmethod
    def _coerce(intents: object) -> tuple[float, ...] | None:
        if isinstance(intents, (str, bytes)):
            return None
        if not isinstance(intents, Iterable):
            return None
        out: list[float] = []
        for raw in intents:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            # Reject NaN / inf / out-of-range silently.
            if not (-1.0 <= v <= 1.0):
                continue
            out.append(v)
        return tuple(out)

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        raw = tick.meta.get(self.meta_key)
        if raw is None:
            return ()
        valid = self._coerce(raw)
        if valid is None or len(valid) < self.min_leaders:
            return ()

        consensus = sum(valid) / len(valid)
        self._consensus.append(consensus)

        if len(self._consensus) < self.window_size:
            return ()

        rolling = sum(self._consensus) / len(self._consensus)

        if rolling > self.consensus_threshold:
            side = Side.BUY
        elif rolling < -self.consensus_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(rolling) / self.confidence_scale)
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
                    "rolling_consensus": f"{rolling:.10f}",
                    "tick_consensus": f"{consensus:.10f}",
                    "n_leaders": f"{len(valid)}",
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
                f"min_leaders={self.min_leaders} "
                f"thresh={self.consensus_threshold}"
            ),
        )


__all__ = ["TraderImitationV1"]
