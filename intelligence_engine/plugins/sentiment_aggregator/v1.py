"""IND-L09 sentiment_aggregator plugin v1 — Indira learning layer #8.

Tick-level **sentiment aggregator** as a deterministic
exponentially-weighted moving-average gate.

External sentiment feeds (news, social, regulatory) decorate each
:class:`MarketTick` with a ``sentiment`` score in ``meta`` — a
canonical IEEE-754 float in ``[-1, 1]`` where ``+1`` is maximally
bullish, ``-1`` is maximally bearish, and ``0`` is neutral. This
plugin is the deterministic *consumer*:

* maintains a single-coefficient EMA of the per-tick score:
  ``ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}``
* once ``warmup_ticks`` valid samples have been seen, emits:

  * ``BUY``  if ``ema > sentiment_threshold``
  * ``SELL`` if ``ema < -sentiment_threshold``
  * silent otherwise

Pure (INV-15 / TEST-01): no clock, no PRNG, no IO.

Refs:

* dixvision_executive_summary.md — "31 Indira learning layers"
  (this closes drift item H3.6 from the canonical-rebuild walk).
* dixvision_build_plan.md §Phase 3 INDIRA — plugin contract.
* manifest.md §0.7 (Plugin Activation Surface).

Note: this plugin is a *deterministic aggregator over external
sentiment scores*. The upstream news / social / regulatory pipeline
that produces those scores lives elsewhere (``news_projection.py``,
``sensory/web_autolearn``); this leaf merely consolidates them into
a single signal channel — same pattern as a smoothing transform.
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
class SentimentAggregatorV1:
    """Eighth concrete intelligence plugin (IND-L09 v1).

    Attributes:
        name: Plugin identifier (matches registry row).
        version: Semantic version.
        lifecycle: Activation state per :class:`PluginLifecycle`.
        meta_key: Key in ``MarketTick.meta`` carrying the sentiment
            score. Default ``"sentiment"``.
        alpha: EMA coefficient in ``(0, 1]``. Higher = more weight on
            recent samples. Default ``0.2``.
        warmup_ticks: Minimum number of valid sentiment samples
            required before the plugin may emit (>= 1).
        sentiment_threshold: Absolute EMA magnitude in ``[0, 1]`` above
            which a directional signal fires.
        confidence_scale: Divisor mapping ``|ema|`` to confidence in
            ``[0, 1]`` (clipped). Must be > 0.
        min_confidence: Floor below which no signal is emitted.
    """

    name: str = "sentiment_aggregator_v1"
    version: str = "0.1.0"
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    meta_key: str = "sentiment"
    alpha: float = 0.2
    warmup_ticks: int = 8
    sentiment_threshold: float = 0.25
    confidence_scale: float = 1.0
    min_confidence: float = 0.05
    _ema: float = field(default=0.0, init=False, repr=False)
    _samples: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.warmup_ticks < 1:
            raise ValueError("warmup_ticks must be >= 1")
        if not 0.0 <= self.sentiment_threshold <= 1.0:
            raise ValueError("sentiment_threshold must be in [0, 1]")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be > 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not self.meta_key:
            raise ValueError("meta_key must be a non-empty string")
        self._ema = 0.0
        self._samples = 0

    def on_tick(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        raw = tick.meta.get(self.meta_key)
        if raw is None:
            return ()
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return ()
        if not (-1.0 <= score <= 1.0):
            return ()
        # NaN: -1.0 <= NaN <= 1.0 is False under IEEE 754 so the bound
        # check above already rejects NaN; explicit isnan-style check
        # is therefore redundant.

        if self._samples == 0:
            self._ema = score
        else:
            self._ema = self.alpha * score + (1.0 - self.alpha) * self._ema
        self._samples += 1

        if self._samples < self.warmup_ticks:
            return ()

        if self._ema > self.sentiment_threshold:
            side = Side.BUY
        elif self._ema < -self.sentiment_threshold:
            side = Side.SELL
        else:
            return ()

        confidence = min(1.0, abs(self._ema) / self.confidence_scale)
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
                    "ema": f"{self._ema:.10f}",
                    "samples": f"{self._samples}",
                    "alpha": f"{self.alpha:.10f}",
                    "sentiment_threshold": (
                        f"{self.sentiment_threshold:.10f}"
                    ),
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
                f"alpha={self.alpha} "
                f"warmup={self.warmup_ticks} "
                f"thresh={self.sentiment_threshold}"
            ),
        )


__all__ = ["SentimentAggregatorV1"]
