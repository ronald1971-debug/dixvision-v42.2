"""IND-L12 news_reaction plugin v1 — Indira learning layer #11.

Tick-level **news reaction** as a deterministic impulse-with-decay
gate.

The upstream news pipeline (``ui/feeds/coindesk_rss.py``,
``intelligence_engine/news/news_projection.py``,
``intelligence_engine/news/news_fanout.py``,
``sensory/web_autolearn``) classifies each headline into a signed
magnitude in ``[-1, 1]`` (``+1`` = maximally bullish news, ``-1`` =
maximally bearish, ``0`` = neutral). When a news event arrives, the
fanout decorates the *next* :class:`MarketTick` with
``news_event_magnitude`` in ``meta``.

This plugin is the deterministic *consumer*:

1. When a news event arrives (``news_event_magnitude`` present and
   valid), latch the impulse to that magnitude.
2. On every subsequent tick (whether news or not) decay the impulse
   geometrically: ``impulse_t = impulse_{t-1} * decay_rate``.
3. Emit:

   * ``BUY``  when ``impulse > impulse_threshold`` and confidence floor met
   * ``SELL`` when ``impulse < -impulse_threshold`` and confidence floor met
   * silent otherwise.

Different shape from the other H3 plugins:

* ``sentiment_aggregator_v1`` averages a continuously-flowing
  sentiment score (EMA).
* ``trader_imitation_v1`` rolls a consensus across leaders.
* ``news_reaction_v1`` reacts to *discrete* events with explicit
  decay between events, the simplest kernel of an event-driven
  feature without any clock dependency (decay is per-tick, not
  per-second, so replay determinism INV-15 is preserved).

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.9, the final H3 leaf, in the
  canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.contracts.engine import (
    HealthState,
    HealthStatus,
    PluginLifecycle,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick


@dataclass
class NewsReactionV1:
    """Eleventh concrete intelligence plugin (IND-L12 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        meta_key: Key in ``MarketTick.meta`` carrying the per-tick
            news-event magnitude. Default ``"news_event_magnitude"``.
        decay_rate: Per-tick geometric decay factor in ``(0, 1]``.
            ``1.0`` = no decay (impulse persists forever); ``0.0`` =
            instantaneous decay after the trigger tick. Default
            ``0.85``.
        impulse_threshold: Absolute impulse magnitude in ``[0, 1]``
            above which a directional signal fires.
        confidence_scale: Divisor mapping ``|impulse|`` to confidence
            in ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "news_reaction_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    meta_key: str = "news_event_magnitude"
    decay_rate: float = 0.85
    impulse_threshold: float = 0.15
    confidence_scale: float = 1.0
    min_confidence: float = 0.05
    _impulse: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.meta_key:
            raise ValueError("meta_key must be a non-empty string")
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError("decay_rate must be in (0, 1]")
        if not 0.0 <= self.impulse_threshold <= 1.0:
            raise ValueError("impulse_threshold must be in [0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._impulse = 0.0

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        # Decay first, so that a news-event tick's impulse is
        # measured at full magnitude on the firing tick rather than
        # being decayed before it has a chance to act.
        self._impulse = self._impulse * self.decay_rate

        raw = tick.meta.get(self.meta_key)
        if raw is not None:
            try:
                magnitude = float(raw)
            except (TypeError, ValueError):
                magnitude = None  # type: ignore[assignment]
            else:
                # Reject NaN / inf / out-of-range silently.
                if -1.0 <= magnitude <= 1.0:
                    # Latch (overwrite — most recent news dominates).
                    self._impulse = magnitude

        if self._impulse > self.impulse_threshold:
            side = Side.BUY
        elif self._impulse < -self.impulse_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(self._impulse) / self.confidence_scale)
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
                    "impulse": f"{self._impulse:.10f}",
                    "decay_rate": f"{self.decay_rate:.10f}",
                    "impulse_threshold": f"{self.impulse_threshold:.10f}",
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
                f"decay={self.decay_rate} "
                f"thresh={self.impulse_threshold}"
            ),
        )


__all__ = ["NewsReactionV1"]
