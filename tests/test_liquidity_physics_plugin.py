"""Tests for the IND-L05 liquidity_physics v1 plugin (Indira plugin #4)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.liquidity_physics import LiquidityPhysicsV1


def _tick(
    ts: int,
    *,
    bid: float = 99.0,
    ask: float = 101.0,
    volume: float = 1.0,
) -> MarketTick:
    return MarketTick(
        ts_ns=ts,
        symbol="BTC-USD",
        bid=bid,
        ask=ask,
        last=0.5 * (bid + ask),
        volume=volume,
        meta={},
    )


def test_no_signal_until_window_full() -> None:
    p = LiquidityPhysicsV1(window_size=4)
    assert p.on_tick(_tick(0, bid=99.0, ask=101.0)) == ()
    for i in range(1, 4):
        bid = 99.0 + 0.5 * i
        ask = 101.0 + 0.5 * i
        assert p.on_tick(_tick(i, bid=bid, ask=ask, volume=1.0)) == ()


def test_strong_upward_drift_emits_buy() -> None:
    p = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        bid = 99.0 + 0.5 * i
        ask = 101.0 + 0.5 * i
        out = p.on_tick(_tick(i, bid=bid, ask=ask, volume=2.0))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert 0.0 < sig.confidence <= 1.0
    assert sig.plugin_chain == ("liquidity_physics_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert "mean_impulse" in sig.meta
    assert sig.meta["window_size"] == "4"


def test_strong_downward_drift_emits_sell() -> None:
    p = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        bid = 99.0 - 0.5 * i
        ask = 101.0 - 0.5 * i
        out = p.on_tick(_tick(i, bid=bid, ask=ask, volume=2.0))
    assert len(out) == 1
    assert out[0].side is Side.SELL


def test_flat_market_emits_nothing() -> None:
    p = LiquidityPhysicsV1(window_size=4, impulse_threshold=0.1)
    for i in range(8):
        out = p.on_tick(_tick(i, bid=99.0, ask=101.0, volume=2.0))
        assert out == ()


def test_below_threshold_emits_nothing() -> None:
    p = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=10.0,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        bid = 99.0 + 0.01 * i
        ask = 101.0 + 0.01 * i
        out = p.on_tick(_tick(i, bid=bid, ask=ask, volume=0.1))
    assert out == ()


def test_zero_volume_emits_nothing() -> None:
    p = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=0.001,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        bid = 99.0 + 0.5 * i
        ask = 101.0 + 0.5 * i
        out = p.on_tick(_tick(i, bid=bid, ask=ask, volume=0.0))
    # Mean impulse with volume=0 is exactly 0; below any positive threshold.
    assert out == ()


def test_invalid_book_drops() -> None:
    p = LiquidityPhysicsV1(window_size=2)
    assert p.on_tick(_tick(0, bid=0.0, ask=101.0)) == ()
    assert p.on_tick(_tick(1, bid=99.0, ask=0.0)) == ()
    assert p.on_tick(_tick(2, bid=101.0, ask=99.0)) == ()


def test_negative_volume_drops() -> None:
    p = LiquidityPhysicsV1(window_size=2)
    assert p.on_tick(_tick(0, bid=99.0, ask=101.0, volume=-1.0)) == ()


def test_replay_determinism() -> None:
    seq = [
        _tick(i, bid=99.0 + 0.3 * i, ask=101.0 + 0.3 * i, volume=2.0)
        for i in range(8)
    ]
    p1 = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    p2 = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    # Drift is barely above threshold, but min_confidence floors emission.
    p = LiquidityPhysicsV1(
        window_size=4,
        impulse_threshold=0.001,
        confidence_scale=1000.0,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(5):
        bid = 99.0 + 0.001 * i
        ask = 101.0 + 0.001 * i
        out = p.on_tick(_tick(i, bid=bid, ask=ask, volume=1.0))
    assert out == ()


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        LiquidityPhysicsV1(window_size=1)
    with pytest.raises(ValueError):
        LiquidityPhysicsV1(impulse_threshold=-0.1)
    with pytest.raises(ValueError):
        LiquidityPhysicsV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        LiquidityPhysicsV1(confidence_scale=-1.0)
    with pytest.raises(ValueError):
        LiquidityPhysicsV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        LiquidityPhysicsV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    p = LiquidityPhysicsV1()
    status = p.check_self()
    assert status.state is HealthState.OK
    assert "liquidity_physics_v1" in status.detail


def test_lifecycle_default_active() -> None:
    p = LiquidityPhysicsV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE
