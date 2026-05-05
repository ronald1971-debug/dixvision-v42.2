"""Tests for the IND-L03 orderflow_imbalance v1 plugin (Indira plugin #2)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.orderflow_imbalance import OrderflowImbalanceV1


def _tick(ts: int, last: float, volume: float, bid: float = 99.0, ask: float = 101.0) -> MarketTick:
    return MarketTick(ts_ns=ts, symbol="BTC-USD", bid=bid, ask=ask, last=last, volume=volume)


def test_no_signal_until_window_full() -> None:
    p = OrderflowImbalanceV1(window_size=4)
    for i in range(3):
        assert p.on_tick(_tick(i, last=101.0, volume=1.0)) == ()


def test_strong_buy_imbalance_emits_buy() -> None:
    p = OrderflowImbalanceV1(window_size=4, imbalance_threshold=0.2, min_confidence=0.0)
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, last=101.0, volume=1.0))  # all lifts
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert 0.0 < sig.confidence <= 1.0
    assert sig.plugin_chain == ("orderflow_imbalance_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert sig.meta["normalised_imbalance"].startswith("1.")


def test_strong_sell_imbalance_emits_sell() -> None:
    p = OrderflowImbalanceV1(window_size=4, imbalance_threshold=0.2, min_confidence=0.0)
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, last=99.0, volume=1.0))  # all hits
    assert len(out) == 1
    assert out[0].side is Side.SELL


def test_balanced_flow_emits_nothing() -> None:
    p = OrderflowImbalanceV1(window_size=4, imbalance_threshold=0.2)
    p.on_tick(_tick(0, last=101.0, volume=1.0))
    p.on_tick(_tick(1, last=99.0, volume=1.0))
    p.on_tick(_tick(2, last=101.0, volume=1.0))
    out = p.on_tick(_tick(3, last=99.0, volume=1.0))
    assert out == ()


def test_below_threshold_emits_nothing() -> None:
    p = OrderflowImbalanceV1(window_size=4, imbalance_threshold=0.6, min_confidence=0.0)
    p.on_tick(_tick(0, last=101.0, volume=1.0))
    p.on_tick(_tick(1, last=101.0, volume=1.0))
    p.on_tick(_tick(2, last=99.0, volume=1.0))
    out = p.on_tick(_tick(3, last=99.0, volume=1.0))
    assert out == ()


def test_zero_volume_buffers_zero_flow() -> None:
    p = OrderflowImbalanceV1(window_size=4)
    for i in range(4):
        out = p.on_tick(_tick(i, last=101.0, volume=0.0))
    assert out == ()


def test_invalid_tick_drops() -> None:
    p = OrderflowImbalanceV1(window_size=2, imbalance_threshold=0.1, min_confidence=0.0)
    assert p.on_tick(MarketTick(ts_ns=0, symbol="X", bid=0.0, ask=1.0, last=1.0)) == ()
    assert p.on_tick(MarketTick(ts_ns=1, symbol="X", bid=2.0, ask=1.0, last=1.0)) == ()  # crossed


def test_replay_determinism() -> None:
    a = OrderflowImbalanceV1(window_size=8, imbalance_threshold=0.2, min_confidence=0.0)
    b = OrderflowImbalanceV1(window_size=8, imbalance_threshold=0.2, min_confidence=0.0)
    seq = [
        (0, 101.0, 1.5),
        (1, 99.0, 0.5),
        (2, 101.0, 2.0),
        (3, 101.0, 1.0),
        (4, 99.0, 0.3),
        (5, 101.0, 1.7),
        (6, 101.0, 0.9),
        (7, 101.0, 1.2),
    ]
    out_a, out_b = (), ()
    for ts, last, vol in seq:
        out_a = a.on_tick(_tick(ts, last=last, volume=vol))
        out_b = b.on_tick(_tick(ts, last=last, volume=vol))
        assert out_a == out_b


def test_check_self_ok() -> None:
    p = OrderflowImbalanceV1()
    h = p.check_self()
    assert h.state is HealthState.OK
    assert "orderflow_imbalance_v1" in h.detail


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        OrderflowImbalanceV1(window_size=1)
    with pytest.raises(ValueError):
        OrderflowImbalanceV1(imbalance_threshold=0.0)
    with pytest.raises(ValueError):
        OrderflowImbalanceV1(imbalance_threshold=1.5)
    with pytest.raises(ValueError):
        OrderflowImbalanceV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        OrderflowImbalanceV1(min_confidence=1.5)


def test_lifecycle_default_active() -> None:
    p = OrderflowImbalanceV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE
