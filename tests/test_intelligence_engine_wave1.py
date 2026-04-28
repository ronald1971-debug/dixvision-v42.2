"""Wave 1 — IntelligenceEngine ↔ MetaControllerHotPath wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.coherence.performance_pressure import load_pressure_config
from core.contracts.engine import PluginLifecycle
from core.contracts.events import Side, SignalEvent, SystemEventKind
from core.contracts.market import MarketTick
from intelligence_engine import (
    DEFAULT_SIGNAL_WINDOW_SIZE,
    IntelligenceEngine,
    RuntimeContext,
)
from intelligence_engine.meta_controller import (
    MetaControllerHotPath,
    load_meta_controller_config,
)
from intelligence_engine.plugins import MicrostructureV1

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tick(
    symbol: str = "EURUSD",
    bid: float = 99.99,
    ask: float = 100.01,
    last: float = 100.10,
    ts: int = 1,
) -> MarketTick:
    return MarketTick(ts_ns=ts, symbol=symbol, bid=bid, ask=ask, last=last)


def _hot_path() -> MetaControllerHotPath:
    return MetaControllerHotPath(
        meta_config=load_meta_controller_config(),
        pressure_config=load_pressure_config(
            REPO_ROOT / "registry" / "pressure.yaml"
        ),
    )


def _context(elapsed_ns: int = 1_000) -> RuntimeContext:
    return RuntimeContext(
        perf=0.5,
        risk=0.0,
        drift=0.0,
        latency=0.0,
        vol_spike_z=0.0,
        elapsed_ns=elapsed_ns,
    )


# ---------------------------------------------------------------------------
# Constructor / defaults
# ---------------------------------------------------------------------------


def test_engine_default_window_size_when_not_configured() -> None:
    engine = IntelligenceEngine()
    assert engine.meta_controller_hot_path is None
    assert engine.signal_window == ()
    assert DEFAULT_SIGNAL_WINDOW_SIZE > 0


def test_engine_rejects_zero_window_size() -> None:
    with pytest.raises(ValueError):
        IntelligenceEngine(signal_window_size=0)


def test_engine_accepts_explicit_hot_path() -> None:
    hot = _hot_path()
    engine = IntelligenceEngine(meta_controller_hot_path=hot)
    assert engine.meta_controller_hot_path is hot


# ---------------------------------------------------------------------------
# on_market still works without the hot path (backward compat)
# ---------------------------------------------------------------------------


def test_on_market_without_hot_path_is_unchanged() -> None:
    plugin = MicrostructureV1(tolerance_bps=2.0, confidence_scale_bps=50.0)
    plugin.lifecycle = PluginLifecycle.ACTIVE
    engine = IntelligenceEngine(microstructure_plugins=[plugin])
    signals = engine.on_market(_tick())
    assert len(signals) == 1
    assert signals[0].side is Side.BUY
    # The window now tracks emitted signals.
    assert engine.signal_window == signals


def test_on_market_appends_to_rolling_window() -> None:
    plugin = MicrostructureV1(tolerance_bps=2.0, confidence_scale_bps=50.0)
    plugin.lifecycle = PluginLifecycle.ACTIVE
    engine = IntelligenceEngine(
        microstructure_plugins=[plugin],
        signal_window_size=2,
    )
    engine.on_market(_tick(ts=1, last=100.10))
    engine.on_market(_tick(ts=2, last=100.20))
    engine.on_market(_tick(ts=3, last=100.30))
    # Bounded to 2 most recent.
    window = engine.signal_window
    assert len(window) == 2
    assert window[-1].ts_ns == 3


# ---------------------------------------------------------------------------
# run_meta_tick contract
# ---------------------------------------------------------------------------


def test_run_meta_tick_requires_hot_path() -> None:
    engine = IntelligenceEngine()
    with pytest.raises(RuntimeError, match="meta_controller_hot_path"):
        engine.run_meta_tick(tick=_tick(), context=_context())


def test_run_meta_tick_returns_signals_decision_ledger() -> None:
    plugin = MicrostructureV1(tolerance_bps=2.0, confidence_scale_bps=50.0)
    plugin.lifecycle = PluginLifecycle.ACTIVE
    engine = IntelligenceEngine(
        microstructure_plugins=[plugin],
        meta_controller_hot_path=_hot_path(),
    )
    signals, decision, ledger = engine.run_meta_tick(
        tick=_tick(last=100.10),
        context=_context(),
    )
    # The microstructure plugin emitted a BUY for last>mid.
    assert len(signals) == 1
    assert signals[0].side is Side.BUY
    assert decision.side is Side.BUY
    # Four-event ledger: BELIEF / PRESSURE / META_AUDIT (and optional META_DIVERGENCE).
    assert ledger[0].sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ledger[1].sub_kind is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT


def test_run_meta_tick_propagates_extra_signals_into_window() -> None:
    engine = IntelligenceEngine(meta_controller_hot_path=_hot_path())
    extra = SignalEvent(ts_ns=10, symbol="X", side=Side.BUY, confidence=0.8)
    signals, decision, ledger = engine.run_meta_tick(
        tick=_tick(),
        context=_context(),
        extra_signals=[extra],
    )
    assert extra in signals
    assert extra in engine.signal_window
    # The decision sees the extra in the window.
    assert decision.side is Side.BUY


def test_run_meta_tick_uses_rolling_window_across_calls() -> None:
    plugin = MicrostructureV1(tolerance_bps=2.0, confidence_scale_bps=50.0)
    plugin.lifecycle = PluginLifecycle.ACTIVE
    engine = IntelligenceEngine(
        microstructure_plugins=[plugin],
        meta_controller_hot_path=_hot_path(),
    )
    # 4 BUY ticks → router persistence threshold met → committed regime
    # transitions to TREND_UP.
    for i in range(4):
        engine.run_meta_tick(
            tick=_tick(ts=10 + i, last=100.10),
            context=_context(),
        )
    state = engine.meta_controller_hot_path.state  # type: ignore[union-attr]
    from core.coherence.belief_state import Regime

    assert state.router_state.current_regime is Regime.TREND_UP


# ---------------------------------------------------------------------------
# INV-15 determinism end-to-end
# ---------------------------------------------------------------------------


def test_run_meta_tick_is_replay_deterministic() -> None:
    """Two engines fed the same tick + context sequence must produce the
    same (signals, decision, ledger) streams."""
    runs = []
    for _ in range(2):
        plugin = MicrostructureV1(
            tolerance_bps=2.0, confidence_scale_bps=50.0
        )
        plugin.lifecycle = PluginLifecycle.ACTIVE
        engine = IntelligenceEngine(
            microstructure_plugins=[plugin],
            meta_controller_hot_path=_hot_path(),
        )
        ticks = [_tick(ts=20 + i, last=100.05 + i * 0.01) for i in range(5)]
        run: list[tuple[object, ...]] = []
        for tk in ticks:
            run.append(engine.run_meta_tick(tick=tk, context=_context()))
        runs.append(tuple(run))
    assert runs[0] == runs[1]


# ---------------------------------------------------------------------------
# check_self surfaces meta-controller wiring
# ---------------------------------------------------------------------------


def test_check_self_reports_meta_controller_wired() -> None:
    engine = IntelligenceEngine(meta_controller_hot_path=_hot_path())
    assert "meta_controller=wired" in engine.check_self().detail
