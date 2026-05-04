"""Tests for the Phase 3 SignalPipeline coordinator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.contracts.engine import HealthState, HealthStatus, PluginLifecycle
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.signal_pipeline import SignalPipeline
from intelligence_engine.strategy_runtime import (
    ConflictResolver,
    MarketRegime,
    RegimeDetector,
    StrategyOrchestrator,
    StrategyScheduler,
    StrategyState,
    StrategyStateMachine,
)


@dataclass
class _ScriptedPlugin:
    name: str
    side: Side
    confidence: float
    lifecycle: PluginLifecycle = PluginLifecycle.ACTIVE
    version: str = "0.1.0"

    def on_tick(self, tick: MarketTick) -> Sequence[SignalEvent]:
        return (
            SignalEvent(
                ts_ns=tick.ts_ns,
                symbol=tick.symbol,
                side=self.side,
                confidence=self.confidence,
                plugin_chain=(self.name,),
                meta={},
            ),
        )

    def check_self(self) -> HealthStatus:
        return HealthStatus(state=HealthState.OK, detail=self.name)


def _tick(ts: int, mid: float = 100.0) -> MarketTick:
    return MarketTick(
        ts_ns=ts,
        symbol="BTCUSDT",
        bid=mid - 0.01,
        ask=mid + 0.01,
        last=mid,
    )


def _build_pipeline(*plugins: _ScriptedPlugin) -> SignalPipeline:
    fsm = StrategyStateMachine()
    sch = StrategyScheduler()
    orc = StrategyOrchestrator(fsm)
    for p in plugins:
        fsm.propose(strategy_id=p.name, ts_ns=0)
        fsm.transition(
            strategy_id=p.name,
            new_state=StrategyState.SHADOW,
            ts_ns=0,
            reason="promote",
        )
        fsm.transition(
            strategy_id=p.name,
            new_state=StrategyState.CANARY,
            ts_ns=0,
            reason="promote",
        )
        fsm.transition(
            strategy_id=p.name,
            new_state=StrategyState.LIVE,
            ts_ns=0,
            reason="promote",
        )
        sch.register(strategy_id=p.name, cadence=1)
        orc.register(strategy_id=p.name, min_state=StrategyState.LIVE)
    return SignalPipeline(
        plugins={p.name: p for p in plugins},
        regime_detector=RegimeDetector(
            window=4,
            trend_threshold_bps=1e9,
            volatility_threshold_bps=1e9,
        ),
        scheduler=sch,
        orchestrator=orc,
        conflict_resolver=ConflictResolver(),
    )


def test_pipeline_fires_eligible_due_plugins():
    a = _ScriptedPlugin("a", Side.BUY, 0.6)
    b = _ScriptedPlugin("b", Side.BUY, 0.3)
    p = _build_pipeline(a, b)
    out = p.on_tick(_tick(1))
    assert set(out.fired) == {"a", "b"}
    assert len(out.raw_signals) == 2
    assert len(out.resolved) == 1
    coalesced, _ = out.resolved[0]
    assert coalesced.side is Side.BUY


def test_pipeline_skips_due_but_ineligible_plugin():
    fsm = StrategyStateMachine()
    sch = StrategyScheduler()
    orc = StrategyOrchestrator(fsm)
    plugin = _ScriptedPlugin("a", Side.BUY, 0.5)
    fsm.propose(strategy_id="a", ts_ns=0)
    # Stays PROPOSED — orchestrator default min_state is SHADOW.
    sch.register(strategy_id="a", cadence=1)
    orc.register(strategy_id="a", min_state=StrategyState.SHADOW)
    pipe = SignalPipeline(
        plugins={"a": plugin},
        regime_detector=RegimeDetector(window=4),
        scheduler=sch,
        orchestrator=orc,
        conflict_resolver=ConflictResolver(),
    )
    out = pipe.on_tick(_tick(1))
    assert out.fired == ()
    assert out.raw_signals == ()


def test_pipeline_skips_eligible_but_not_due_plugin():
    fsm = StrategyStateMachine()
    sch = StrategyScheduler()
    orc = StrategyOrchestrator(fsm)
    plugin = _ScriptedPlugin("a", Side.BUY, 0.5)
    fsm.propose(strategy_id="a", ts_ns=0)
    fsm.transition(
        strategy_id="a",
        new_state=StrategyState.SHADOW,
        ts_ns=0,
        reason="promote",
    )
    sch.register(strategy_id="a", cadence=3)  # only fires every 3rd
    orc.register(strategy_id="a", min_state=StrategyState.SHADOW)
    pipe = SignalPipeline(
        plugins={"a": plugin},
        regime_detector=RegimeDetector(window=4),
        scheduler=sch,
        orchestrator=orc,
        conflict_resolver=ConflictResolver(),
    )
    assert pipe.on_tick(_tick(1)).fired == ()
    assert pipe.on_tick(_tick(2)).fired == ()
    assert pipe.on_tick(_tick(3)).fired == ("a",)


def test_pipeline_disabled_plugin_does_not_fire():
    plugin = _ScriptedPlugin(
        "a", Side.BUY, 0.6, lifecycle=PluginLifecycle.DISABLED
    )
    pipe = _build_pipeline(plugin)
    out = pipe.on_tick(_tick(1))
    # The orchestrator/scheduler still mark it due+eligible, but the
    # pipeline drops DISABLED plugins before invoking them.
    assert out.raw_signals == ()


def test_pipeline_replay_determinism():
    a = _ScriptedPlugin("a", Side.BUY, 0.6)
    b = _ScriptedPlugin("b", Side.SELL, 0.3)
    pipe1 = _build_pipeline(a, b)
    pipe2 = _build_pipeline(a, b)
    ticks = [_tick(i + 1, 100.0 + i * 0.01) for i in range(10)]
    out1 = tuple(
        tuple((s.symbol, s.side, round(s.confidence, 9)) for s, _ in pipe1.on_tick(t).resolved)
        for t in ticks
    )
    out2 = tuple(
        tuple((s.symbol, s.side, round(s.confidence, 9)) for s, _ in pipe2.on_tick(t).resolved)
        for t in ticks
    )
    assert out1 == out2


def test_pipeline_regime_used_for_eligibility():
    fsm = StrategyStateMachine()
    sch = StrategyScheduler()
    orc = StrategyOrchestrator(fsm)
    plugin = _ScriptedPlugin("trend_only", Side.BUY, 0.5)
    fsm.propose(strategy_id="trend_only", ts_ns=0)
    fsm.transition(
        strategy_id="trend_only",
        new_state=StrategyState.SHADOW,
        ts_ns=0,
        reason="promote",
    )
    sch.register(strategy_id="trend_only", cadence=1)
    orc.register(
        strategy_id="trend_only",
        allowed_regimes=(MarketRegime.TRENDING_UP,),
        min_state=StrategyState.SHADOW,
    )
    pipe = SignalPipeline(
        plugins={"trend_only": plugin},
        regime_detector=RegimeDetector(
            window=4,
            trend_threshold_bps=5.0,
            volatility_threshold_bps=1e9,
        ),
        scheduler=sch,
        orchestrator=orc,
        conflict_resolver=ConflictResolver(),
    )
    # Flat ticks → RANGING → not eligible.
    for i in range(4):
        out = pipe.on_tick(_tick(i + 1, 100.0))
    assert out.regime.regime in (MarketRegime.RANGING, MarketRegime.UNKNOWN)
    assert out.fired == ()
    # Strong upward drift → TRENDING_UP → fires.
    for i in range(8):
        out = pipe.on_tick(_tick(10 + i, 100.0 + i * 0.5))
    assert out.regime.regime is MarketRegime.TRENDING_UP
    assert out.fired == ("trend_only",)
