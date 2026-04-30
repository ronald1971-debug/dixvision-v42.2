"""Signal pipeline — Phase 3.

The signal pipeline is the deterministic *coordinator* that turns one
``MarketTick`` into zero-or-more coalesced ``SignalEvent`` rows by
sequencing the strategy-runtime components in a fixed order:

    tick
      → RegimeDetector.observe(tick)        # classify market regime
      → StrategyScheduler.step(tick.ts_ns)  # which strategies are due
      → StrategyOrchestrator.eligible(reg)  # which strategies are allowed
      → run intersection(due, eligible)     # call each plugin .on_tick(tick)
      → ConflictResolver.resolve(signals)   # collapse same-symbol conflicts
      → emit (SignalEvent, ConflictResolution) tuples

It is pure-Python, IO-free, and clock-free — the same input tick stream
produces bit-identical outputs across runs (INV-15). It does **not**
import any other engine; it consumes only ``core.contracts.*`` plus the
intelligence runtime modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts.engine import MicrostructurePlugin, PluginLifecycle
from core.contracts.events import SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.strategy_runtime.conflict_resolver import (
    ConflictResolution,
    ConflictResolver,
)
from intelligence_engine.strategy_runtime.orchestrator import StrategyOrchestrator
from intelligence_engine.strategy_runtime.regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeReading,
)
from intelligence_engine.strategy_runtime.scheduler import StrategyScheduler


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """One pass of the pipeline."""

    tick: MarketTick
    regime: RegimeReading
    fired: tuple[str, ...]
    raw_signals: tuple[SignalEvent, ...]
    resolved: tuple[tuple[SignalEvent, ConflictResolution], ...]


class SignalPipeline:
    """Stateless-from-the-outside coordinator over the strategy runtime.

    Args:
        plugins: Mapping ``strategy_id -> MicrostructurePlugin``. The
            pipeline only ever calls ``plugin.on_tick(tick)`` and reads
            ``plugin.lifecycle`` — it never mutates plugin state.
        regime_detector: Provides per-tick :class:`RegimeReading`.
        scheduler: Owns "is this strategy due?" cadence.
        orchestrator: Owns "is this strategy eligible?" lifecycle/regime
            gating.
        conflict_resolver: Collapses conflicting signals.
    """

    name: str = "signal_pipeline"
    spec_id: str = "IND-SP-01"

    def __init__(
        self,
        *,
        plugins: Mapping[str, MicrostructurePlugin],
        regime_detector: RegimeDetector,
        scheduler: StrategyScheduler,
        orchestrator: StrategyOrchestrator,
        conflict_resolver: ConflictResolver,
    ) -> None:
        self._plugins = dict(plugins)
        self._regime_detector = regime_detector
        self._scheduler = scheduler
        self._orchestrator = orchestrator
        self._resolver = conflict_resolver

    # -- queries -----------------------------------------------------------

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(self._plugins.keys())

    # -- step --------------------------------------------------------------

    def on_tick(self, tick: MarketTick) -> PipelineOutput:
        regime = self._regime_detector.observe(tick)
        due = set(self._scheduler.step(tick.ts_ns))
        eligible = set(self._orchestrator.eligible(regime.regime))
        # Preserve plugin registration order for deterministic outputs.
        fired_order: list[str] = []
        raw: list[SignalEvent] = []
        for sid, plugin in self._plugins.items():
            if sid not in due or sid not in eligible:
                continue
            if plugin.lifecycle is PluginLifecycle.DISABLED:
                continue
            fired_order.append(sid)
            for s in plugin.on_tick(tick):
                if plugin.lifecycle is PluginLifecycle.SHADOW:
                    meta = dict(s.meta)
                    meta["shadow"] = "true"
                    s = SignalEvent(
                        ts_ns=s.ts_ns,
                        symbol=s.symbol,
                        side=s.side,
                        confidence=s.confidence,
                        plugin_chain=s.plugin_chain,
                        meta=meta,
                        produced_by_engine="intelligence_engine",
                    )
                raw.append(s)

        resolved = self._resolver.resolve(raw)
        return PipelineOutput(
            tick=tick,
            regime=regime,
            fired=tuple(fired_order),
            raw_signals=tuple(raw),
            resolved=resolved,
        )


__all__ = ["MarketRegime", "PipelineOutput", "SignalPipeline"]
