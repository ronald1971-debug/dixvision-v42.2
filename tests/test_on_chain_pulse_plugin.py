"""Tests for the IND-L10 on_chain_pulse v1 plugin (Indira plugin #9)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.on_chain_pulse import OnChainPulseV1


def _tick(ts: int, *, netflow: object | None) -> MarketTick:
    meta: dict[str, object] = {}
    if netflow is not None:
        meta["exchange_netflow"] = netflow
    return MarketTick(
        ts_ns=ts,
        symbol="BTC-USD",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta=meta,
    )


def test_no_signal_until_window_full() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=0.0,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(3):
        out = p.on_tick(_tick(i, netflow=500.0))
    assert out == ()


def test_sustained_inflows_emit_sell() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=1000.0,
        confidence_scale=10000.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, netflow=600.0))
    # cum = 2400 > 1000 → SELL (inflows = sell pressure)
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert float(out[0].meta["cum_netflow"]) > 1000.0


def test_sustained_outflows_emit_buy() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=1000.0,
        confidence_scale=10000.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, netflow=-600.0))
    # cum = -2400 < -1000 → BUY (outflows = HODL / accumulate)
    assert len(out) == 1
    assert out[0].side is Side.BUY
    assert float(out[0].meta["cum_netflow"]) < -1000.0


def test_balanced_flows_no_emit() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=100.0,
        min_confidence=0.0,
    )
    flows = [500.0, -500.0, 500.0, -500.0]
    out: tuple = ()
    for i, f in enumerate(flows):
        out = p.on_tick(_tick(i, netflow=f))
    # cum_netflow = 0
    assert out == ()


def test_below_threshold_no_emit() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=10000.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, netflow=100.0))
    assert out == ()


def test_zero_netflow_window_no_emit() -> None:
    p = OnChainPulseV1(window_size=4, netflow_threshold=0.0)
    # threshold = 0 but cum = 0 also doesn't satisfy > 0 nor < 0
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, netflow=0.0))
    assert out == ()


def test_missing_meta_drops() -> None:
    p = OnChainPulseV1(window_size=2)
    assert p.on_tick(_tick(0, netflow=None)) == ()


def test_non_numeric_drops() -> None:
    p = OnChainPulseV1(window_size=2)
    assert p.on_tick(_tick(0, netflow="huge")) == ()
    assert p.on_tick(_tick(1, netflow=None)) == ()


def test_nan_inf_drops() -> None:
    p = OnChainPulseV1(window_size=2)
    assert p.on_tick(_tick(0, netflow=float("nan"))) == ()
    assert p.on_tick(_tick(1, netflow=float("inf"))) == ()
    assert p.on_tick(_tick(2, netflow=float("-inf"))) == ()


def test_replay_determinism() -> None:
    seq = [
        _tick(i, netflow=400.0 if i % 2 == 0 else 200.0) for i in range(10)
    ]
    p1 = OnChainPulseV1(window_size=4, netflow_threshold=500.0)
    p2 = OnChainPulseV1(window_size=4, netflow_threshold=500.0)
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=100.0,
        confidence_scale=1e9,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, netflow=200.0))
    assert out == ()


def test_window_rotation_flips_signal() -> None:
    p = OnChainPulseV1(
        window_size=4,
        netflow_threshold=1000.0,
        confidence_scale=10000.0,
        min_confidence=0.0,
    )
    out_a: tuple = ()
    for i in range(4):
        out_a = p.on_tick(_tick(i, netflow=600.0))
    assert len(out_a) == 1
    assert out_a[0].side is Side.SELL
    out_b: tuple = ()
    for i in range(4, 12):
        out_b = p.on_tick(_tick(i, netflow=-600.0))
    assert len(out_b) == 1
    assert out_b[0].side is Side.BUY


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        OnChainPulseV1(meta_key="")
    with pytest.raises(ValueError):
        OnChainPulseV1(window_size=1)
    with pytest.raises(ValueError):
        OnChainPulseV1(netflow_threshold=-1.0)
    with pytest.raises(ValueError):
        OnChainPulseV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        OnChainPulseV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        OnChainPulseV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    status = OnChainPulseV1().check_self()
    assert status.state is HealthState.OK
    assert "on_chain_pulse_v1" in status.detail


def test_lifecycle_default_active() -> None:
    assert OnChainPulseV1().lifecycle is PluginLifecycle.ACTIVE


def test_custom_meta_key_routes_correctly() -> None:
    p = OnChainPulseV1(
        meta_key="whale_netflow",
        window_size=2,
        netflow_threshold=100.0,
        confidence_scale=1000.0,
        min_confidence=0.0,
    )
    t1 = MarketTick(
        ts_ns=0,
        symbol="X",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta={"exchange_netflow": 9999.0, "whale_netflow": 80.0},
    )
    t2 = MarketTick(
        ts_ns=1,
        symbol="X",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta={"whale_netflow": 80.0},
    )
    p.on_tick(t1)
    out = p.on_tick(t2)
    assert len(out) == 1
    assert out[0].side is Side.SELL  # cum = 160 > 100
