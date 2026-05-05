"""Tests for the IND-L08 footprint_delta v1 plugin (Indira plugin #7)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.footprint_delta import FootprintDeltaV1


def _tick(
    ts: int, *, bid: float, ask: float, last: float, volume: float
) -> MarketTick:
    return MarketTick(
        ts_ns=ts,
        symbol="BTC-USD",
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        meta={},
    )


def test_no_signal_until_window_full() -> None:
    p = FootprintDeltaV1(window_size=4, delta_threshold=10.0)
    out: tuple = ()
    for i in range(3):
        out = p.on_tick(_tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0))
    assert out == ()


def test_sustained_buy_pressure_emits_buy() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=10.0,
        confidence_scale=40.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.plugin_chain == ("footprint_delta_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert float(sig.meta["cum_delta"]) > 10.0


def test_sustained_sell_pressure_emits_sell() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=10.0,
        confidence_scale=40.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=99.0, volume=10.0)
        )
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert float(out[0].meta["cum_delta"]) < -10.0


def test_balanced_flow_emits_nothing() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=1.0,
        confidence_scale=10.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        last = 101.0 if i % 2 == 0 else 99.0
        out = p.on_tick(_tick(i, bid=99.0, ask=101.0, last=last, volume=10.0))
    assert out == ()


def test_neutral_aggressor_contributes_zero_delta() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=1.0,
        confidence_scale=10.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=100.0, volume=10.0)
        )
    assert out == ()


def test_below_threshold_emits_nothing() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=1000.0,
        confidence_scale=10.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    assert out == ()


def test_zero_volume_contributes_zero() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=1.0,
        confidence_scale=10.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=0.0)
        )
    assert out == ()


def test_invalid_book_drops() -> None:
    p = FootprintDeltaV1(window_size=2)
    assert p.on_tick(_tick(0, bid=0.0, ask=101.0, last=100.0, volume=10.0)) == ()
    assert p.on_tick(_tick(1, bid=99.0, ask=0.0, last=100.0, volume=10.0)) == ()
    assert p.on_tick(_tick(2, bid=101.0, ask=99.0, last=100.0, volume=10.0)) == ()
    assert p.on_tick(_tick(3, bid=99.0, ask=101.0, last=0.0, volume=10.0)) == ()


def test_negative_volume_drops() -> None:
    p = FootprintDeltaV1(window_size=2)
    assert p.on_tick(_tick(0, bid=99.0, ask=101.0, last=101.0, volume=-1.0)) == ()


def test_replay_determinism() -> None:
    seq = [
        _tick(
            i,
            bid=99.0,
            ask=101.0,
            last=101.0 if i % 3 != 0 else 99.0,
            volume=10.0 + i,
        )
        for i in range(8)
    ]
    p1 = FootprintDeltaV1(window_size=4)
    p2 = FootprintDeltaV1(window_size=4)
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = FootprintDeltaV1(
        window_size=4,
        delta_threshold=10.0,
        confidence_scale=10000.0,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    assert out == ()


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        FootprintDeltaV1(window_size=1)
    with pytest.raises(ValueError):
        FootprintDeltaV1(delta_threshold=-1.0)
    with pytest.raises(ValueError):
        FootprintDeltaV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        FootprintDeltaV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        FootprintDeltaV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    p = FootprintDeltaV1()
    status = p.check_self()
    assert status.state is HealthState.OK
    assert "footprint_delta_v1" in status.detail


def test_lifecycle_default_active() -> None:
    p = FootprintDeltaV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE


def test_window_rotates_old_deltas_out() -> None:
    """Old deltas must drop out of the rolling window."""
    p = FootprintDeltaV1(
        window_size=2,
        delta_threshold=5.0,
        confidence_scale=10.0,
        min_confidence=0.0,
    )
    # First 2 ticks: strong buy -> emits BUY.
    out_a = p.on_tick(_tick(0, bid=99.0, ask=101.0, last=101.0, volume=10.0))
    out_b = p.on_tick(_tick(1, bid=99.0, ask=101.0, last=101.0, volume=10.0))
    assert out_a == ()
    assert len(out_b) == 1
    # Then 2 ticks of strong sell -> the buys should have rotated out
    # entirely after 2 sells, leaving net negative delta -> emits SELL.
    p.on_tick(_tick(2, bid=99.0, ask=101.0, last=99.0, volume=10.0))
    out_d = p.on_tick(_tick(3, bid=99.0, ask=101.0, last=99.0, volume=10.0))
    assert len(out_d) == 1
    assert out_d[0].side is Side.SELL
