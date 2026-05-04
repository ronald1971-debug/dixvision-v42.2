"""Phase E2 — IntelligenceEngine + microstructure plugin tests."""

from __future__ import annotations

import pytest

from core.contracts.engine import (
    HealthState,
    MicrostructurePlugin,
    PluginLifecycle,
)
from core.contracts.events import (
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
)
from core.contracts.market import MarketTick
from execution_engine.engine import ExecutionEngine
from governance_engine.harness_approver import (
    approve_signal_for_execution,
)
from intelligence_engine.engine import IntelligenceEngine
from intelligence_engine.plugins import MicrostructureV1

# ---------------------------------------------------------------------------
# MicrostructureV1 — unit
# ---------------------------------------------------------------------------


def _tick(symbol: str, bid: float, ask: float, last: float, ts: int = 1) -> MarketTick:
    return MarketTick(ts_ns=ts, symbol=symbol, bid=bid, ask=ask, last=last)


def test_microstructure_v1_implements_protocol():
    assert isinstance(MicrostructureV1(), MicrostructurePlugin)


def test_microstructure_v1_emits_buy_when_last_above_mid():
    plugin = MicrostructureV1(tolerance_bps=2.0, confidence_scale_bps=50.0)
    # mid = 100.0, last = 100.10 -> +10 bps -> BUY
    out = plugin.on_tick(_tick("EURUSD", 99.99, 100.01, 100.10))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.confidence == pytest.approx(min(1.0, 10.0 / 50.0))
    assert sig.plugin_chain == ("microstructure_v1",)


def test_microstructure_v1_emits_sell_when_last_below_mid():
    plugin = MicrostructureV1(tolerance_bps=2.0, confidence_scale_bps=50.0)
    out = plugin.on_tick(_tick("EURUSD", 99.99, 100.01, 99.90))
    assert len(out) == 1
    assert out[0].side is Side.SELL


def test_microstructure_v1_emits_hold_within_tolerance():
    plugin = MicrostructureV1(tolerance_bps=2.0)
    # last = mid -> diff_bps = 0 -> HOLD
    out = plugin.on_tick(_tick("EURUSD", 99.99, 100.01, 100.0))
    assert len(out) == 1
    assert out[0].side is Side.HOLD


def test_microstructure_v1_is_deterministic():
    plugin_a = MicrostructureV1()
    plugin_b = MicrostructureV1()
    tick = _tick("BTCUSDT", 49990.0, 50010.0, 50050.0, ts=42)
    assert plugin_a.on_tick(tick) == plugin_b.on_tick(tick)


def test_microstructure_v1_skips_invalid_quotes():
    plugin = MicrostructureV1()
    assert plugin.on_tick(_tick("X", 0.0, 1.0, 1.0)) == ()
    assert plugin.on_tick(_tick("X", 1.0, 0.0, 1.0)) == ()
    assert plugin.on_tick(_tick("X", 1.0, 1.0, 0.0)) == ()


def test_microstructure_v1_skips_crossed_book():
    plugin = MicrostructureV1()
    assert plugin.on_tick(_tick("X", 1.10, 1.05, 1.07)) == ()


def test_microstructure_v1_min_confidence_filter():
    plugin = MicrostructureV1(
        tolerance_bps=0.0,
        confidence_scale_bps=50.0,
        min_confidence=0.5,
    )
    # +10 bps -> confidence 0.2 -> filtered out by min_confidence=0.5
    assert plugin.on_tick(_tick("X", 99.99, 100.01, 100.10)) == ()


def test_microstructure_v1_validates_config():
    with pytest.raises(ValueError):
        MicrostructureV1(tolerance_bps=-1.0)
    with pytest.raises(ValueError):
        MicrostructureV1(confidence_scale_bps=0.0)
    with pytest.raises(ValueError):
        MicrostructureV1(min_confidence=1.5)


# ---------------------------------------------------------------------------
# IntelligenceEngine — integration
# ---------------------------------------------------------------------------


def test_intelligence_engine_no_plugins_emits_nothing():
    engine = IntelligenceEngine()
    out = engine.on_market(_tick("EURUSD", 1.0, 1.01, 1.02))
    assert out == ()


def test_intelligence_engine_runs_loaded_plugin():
    engine = IntelligenceEngine(
        microstructure_plugins=(
            MicrostructureV1(lifecycle=PluginLifecycle.ACTIVE),
        )
    )
    out = engine.on_market(_tick("EURUSD", 99.99, 100.01, 100.10))
    assert len(out) == 1
    assert out[0].side is Side.BUY
    assert out[0].meta.get("shadow") != "true"


def test_intelligence_engine_skips_disabled_plugins():
    engine = IntelligenceEngine(
        microstructure_plugins=(
            MicrostructureV1(lifecycle=PluginLifecycle.DISABLED),
        )
    )
    assert engine.on_market(_tick("EURUSD", 99.99, 100.01, 100.10)) == ()


def test_intelligence_engine_process_passthrough_for_signals():
    engine = IntelligenceEngine()
    sig = SignalEvent(ts_ns=10, symbol="X", side=Side.BUY, confidence=0.5)
    assert engine.process(sig) == (sig,)


def test_intelligence_engine_process_ignores_non_signal_events():
    engine = IntelligenceEngine()
    haz = HazardEvent(
        ts_ns=11, code="K", severity=HazardSeverity.LOW, source="system"
    )
    assert engine.process(haz) == ()


def test_intelligence_engine_check_self_reports_plugins():
    engine = IntelligenceEngine(
        microstructure_plugins=(MicrostructureV1(),),
    )
    status = engine.check_self()
    assert status.state is HealthState.OK
    assert "microstructure_v1" in status.plugin_states["microstructure"]


def test_intelligence_engine_hold_only_stream_engine_contract():
    """Phase E2 exit gate: engine contract must hold for an all-HOLD stream."""
    engine = IntelligenceEngine(
        microstructure_plugins=(
            MicrostructureV1(
                tolerance_bps=10.0,
                lifecycle=PluginLifecycle.ACTIVE,
            ),
        )
    )
    signals: list[SignalEvent] = []
    for i in range(10):
        signals.extend(
            engine.on_market(
                MarketTick(
                    ts_ns=100 + i,
                    symbol="EURUSD",
                    bid=99.99,
                    ask=100.01,
                    last=100.0,  # exactly mid
                )
            )
        )
    assert len(signals) == 10
    assert all(s.side is Side.HOLD for s in signals)
    assert all(s.kind is EventKind.SIGNAL for s in signals)


# ---------------------------------------------------------------------------
# End-to-end: tick -> intelligence -> execution
# ---------------------------------------------------------------------------


def test_end_to_end_tick_drives_filled_execution_when_active():
    intel = IntelligenceEngine(
        microstructure_plugins=(
            MicrostructureV1(
                tolerance_bps=2.0,
                lifecycle=PluginLifecycle.ACTIVE,
            ),
        )
    )
    execution = ExecutionEngine()

    tick = MarketTick(
        ts_ns=200, symbol="BTCUSDT", bid=49_990.0, ask=50_010.0, last=50_050.0
    )
    execution.on_market(tick)
    signals = intel.on_market(tick)
    assert len(signals) == 1
    assert signals[0].side is Side.BUY

    executions: list[ExecutionEvent] = []
    for sig in signals:
        intent = approve_signal_for_execution(sig, ts_ns=sig.ts_ns)
        for ev in execution.execute(intent):
            assert isinstance(ev, ExecutionEvent)
            executions.append(ev)
    assert len(executions) == 1
    assert executions[0].status is ExecutionStatus.FILLED
    assert executions[0].price == pytest.approx(50_050.0)


def test_end_to_end_replay_determinism():
    """Same tick stream must produce structurally identical signals."""

    def run() -> tuple[SignalEvent, ...]:
        intel = IntelligenceEngine(
            microstructure_plugins=(
                MicrostructureV1(lifecycle=PluginLifecycle.ACTIVE),
            )
        )
        out: list[SignalEvent] = []
        for i in range(5):
            out.extend(
                intel.on_market(
                    MarketTick(
                        ts_ns=300 + i,
                        symbol="EURUSD",
                        bid=99.99,
                        ask=100.01,
                        last=100.0 + 0.01 * (i - 2),
                    )
                )
            )
        return tuple(out)

    a = run()
    b = run()
    assert a == b
