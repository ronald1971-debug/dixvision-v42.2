"""Tests for the IND-L04 order_book_pressure v1 plugin (Indira plugin #3)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.order_book_pressure import OrderBookPressureV1


def _tick(
    ts: int,
    bid_size: float | None = 1.0,
    ask_size: float | None = 1.0,
    bid: float = 99.0,
    ask: float = 101.0,
) -> MarketTick:
    meta: dict[str, str] = {}
    if bid_size is not None:
        meta["bid_size"] = f"{bid_size:.6f}"
    if ask_size is not None:
        meta["ask_size"] = f"{ask_size:.6f}"
    return MarketTick(
        ts_ns=ts,
        symbol="BTC-USD",
        bid=bid,
        ask=ask,
        last=100.0,
        volume=0.0,
        meta=meta,
    )


def test_no_signal_until_window_full() -> None:
    p = OrderBookPressureV1(window_size=4)
    for i in range(3):
        assert p.on_tick(_tick(i, bid_size=10.0, ask_size=1.0)) == ()


def test_strong_bid_pressure_emits_buy() -> None:
    p = OrderBookPressureV1(window_size=4, pressure_threshold=0.2, min_confidence=0.0)
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, bid_size=10.0, ask_size=1.0))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert 0.0 < sig.confidence <= 1.0
    assert sig.plugin_chain == ("order_book_pressure_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert sig.meta["mean_book_pressure"].startswith("0.")
    assert sig.meta["window_size"] == "4"


def test_strong_ask_pressure_emits_sell() -> None:
    p = OrderBookPressureV1(window_size=4, pressure_threshold=0.2, min_confidence=0.0)
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, bid_size=1.0, ask_size=10.0))
    assert len(out) == 1
    assert out[0].side is Side.SELL


def test_balanced_pressure_emits_nothing() -> None:
    p = OrderBookPressureV1(window_size=4, pressure_threshold=0.1)
    for i in range(4):
        out = p.on_tick(_tick(i, bid_size=1.0, ask_size=1.0))
    assert out == ()


def test_below_threshold_emits_nothing() -> None:
    p = OrderBookPressureV1(window_size=4, pressure_threshold=0.5, min_confidence=0.0)
    out: tuple = ()
    for i in range(4):
        # 1.5 vs 1.0 -> pressure ~ 0.2, below 0.5
        out = p.on_tick(_tick(i, bid_size=1.5, ask_size=1.0))
    assert out == ()


def test_missing_meta_drops() -> None:
    p = OrderBookPressureV1(window_size=2)
    assert p.on_tick(_tick(0, bid_size=None, ask_size=1.0)) == ()
    assert p.on_tick(_tick(1, bid_size=1.0, ask_size=None)) == ()


def test_negative_or_zero_sizes_drop() -> None:
    p = OrderBookPressureV1(window_size=2, min_confidence=0.0)
    assert p.on_tick(_tick(0, bid_size=-1.0, ask_size=1.0)) == ()
    assert p.on_tick(_tick(1, bid_size=0.0, ask_size=0.0)) == ()


def test_invalid_book_drops() -> None:
    p = OrderBookPressureV1(window_size=2)
    # crossed
    assert p.on_tick(_tick(0, bid=2.0, ask=1.0)) == ()
    # zero bid
    assert p.on_tick(_tick(1, bid=0.0)) == ()


def test_unparseable_meta_value_drops() -> None:
    p = OrderBookPressureV1(window_size=2)
    bad = MarketTick(
        ts_ns=0,
        symbol="X",
        bid=99.0,
        ask=101.0,
        last=100.0,
        meta={"bid_size": "NaNNN", "ask_size": "1.0"},
    )
    assert p.on_tick(bad) == ()


def test_replay_determinism() -> None:
    a = OrderBookPressureV1(window_size=8, pressure_threshold=0.2, min_confidence=0.0)
    b = OrderBookPressureV1(window_size=8, pressure_threshold=0.2, min_confidence=0.0)
    seq = [(i, 5.0 + (i % 2), 1.0 + (i % 3)) for i in range(20)]
    out_a = [a.on_tick(_tick(i, bid_size=bs, ask_size=asz)) for i, bs, asz in seq]
    out_b = [b.on_tick(_tick(i, bid_size=bs, ask_size=asz)) for i, bs, asz in seq]
    assert out_a == out_b


def test_min_confidence_floor() -> None:
    p = OrderBookPressureV1(
        window_size=4,
        pressure_threshold=0.05,  # very low → fires
        confidence_scale=10.0,    # very large → tiny confidence
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, bid_size=1.5, ask_size=1.0))
    assert out == ()  # confidence below floor


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        OrderBookPressureV1(window_size=1)
    with pytest.raises(ValueError):
        OrderBookPressureV1(pressure_threshold=0.0)
    with pytest.raises(ValueError):
        OrderBookPressureV1(pressure_threshold=1.5)
    with pytest.raises(ValueError):
        OrderBookPressureV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        OrderBookPressureV1(min_confidence=-0.1)


def test_check_self_reports_ok() -> None:
    p = OrderBookPressureV1()
    h = p.check_self()
    assert h.state is HealthState.OK
    assert "order_book_pressure_v1" in h.detail


def test_lifecycle_default_active() -> None:
    p = OrderBookPressureV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE
