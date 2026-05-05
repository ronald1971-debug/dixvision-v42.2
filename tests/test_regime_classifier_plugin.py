"""Tests for the IND-L06 regime_classifier v1 plugin (Indira plugin #5)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.regime_classifier import RegimeClassifierV1


def _tick(ts: int, *, bid: float, ask: float) -> MarketTick:
    return MarketTick(
        ts_ns=ts,
        symbol="BTC-USD",
        bid=bid,
        ask=ask,
        last=0.5 * (bid + ask),
        volume=1.0,
        meta={},
    )


def test_no_signal_until_window_full() -> None:
    p = RegimeClassifierV1(window_size=4)
    for i in range(4):  # first tick yields no return; 3 more returns < window
        out = p.on_tick(_tick(i, bid=99.0 + i * 0.05, ask=101.0 + i * 0.05))
        assert out == ()


def test_low_vol_bull_emits_buy() -> None:
    p = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=1.0,
        vol_high_threshold=1.0,
        drift_threshold=0.0001,
        confidence_scale=0.001,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(6):
        bid = 99.0 + i * 0.05
        ask = 101.0 + i * 0.05
        out = p.on_tick(_tick(i, bid=bid, ask=ask))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.meta["regime"] == "low_vol_bull"
    assert sig.plugin_chain == ("regime_classifier_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert sig.meta["window_size"] == "4"


def test_low_vol_bear_emits_sell() -> None:
    p = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=1.0,
        vol_high_threshold=1.0,
        drift_threshold=0.0001,
        confidence_scale=0.001,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(6):
        bid = 105.0 - i * 0.05
        ask = 107.0 - i * 0.05
        out = p.on_tick(_tick(i, bid=bid, ask=ask))
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert out[0].meta["regime"] == "low_vol_bear"


def test_high_vol_silences_emit() -> None:
    p = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=0.00001,
        vol_high_threshold=0.0001,
        drift_threshold=0.0,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    # Strong oscillation -> high vol.
    bids = [100.0, 110.0, 90.0, 110.0, 90.0, 110.0]
    out: tuple = ()
    for i, b in enumerate(bids):
        out = p.on_tick(_tick(i, bid=b, ask=b + 2.0))
    assert out == ()


def test_mid_vol_band_silences_emit() -> None:
    # vol falls between low and high bands -> RANGE no-emit.
    p = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=0.0,
        vol_high_threshold=1.0,
        drift_threshold=0.0,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(6):
        bid = 99.0 + i * 0.05
        ask = 101.0 + i * 0.05
        out = p.on_tick(_tick(i, bid=bid, ask=ask))
    assert out == ()


def test_calm_no_drift_emits_nothing() -> None:
    p = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=1.0,
        vol_high_threshold=1.0,
        drift_threshold=0.001,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(6):
        out = p.on_tick(_tick(i, bid=99.0, ask=101.0))
    assert out == ()


def test_invalid_book_drops() -> None:
    p = RegimeClassifierV1(window_size=2)
    assert p.on_tick(_tick(0, bid=0.0, ask=101.0)) == ()
    assert p.on_tick(_tick(1, bid=99.0, ask=0.0)) == ()
    assert p.on_tick(_tick(2, bid=101.0, ask=99.0)) == ()


def test_replay_determinism() -> None:
    seq = [
        _tick(i, bid=99.0 + i * 0.05, ask=101.0 + i * 0.05) for i in range(8)
    ]
    p1 = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=1.0,
        vol_high_threshold=1.0,
        drift_threshold=0.0001,
        confidence_scale=0.001,
        min_confidence=0.0,
    )
    p2 = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=1.0,
        vol_high_threshold=1.0,
        drift_threshold=0.0001,
        confidence_scale=0.001,
        min_confidence=0.0,
    )
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = RegimeClassifierV1(
        window_size=4,
        vol_low_threshold=1.0,
        vol_high_threshold=1.0,
        drift_threshold=0.0001,
        confidence_scale=1000.0,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(6):
        bid = 99.0 + i * 0.05
        ask = 101.0 + i * 0.05
        out = p.on_tick(_tick(i, bid=bid, ask=ask))
    assert out == ()


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        RegimeClassifierV1(window_size=1)
    with pytest.raises(ValueError):
        RegimeClassifierV1(vol_low_threshold=-0.1)
    with pytest.raises(ValueError):
        RegimeClassifierV1(vol_low_threshold=0.5, vol_high_threshold=0.1)
    with pytest.raises(ValueError):
        RegimeClassifierV1(drift_threshold=-0.1)
    with pytest.raises(ValueError):
        RegimeClassifierV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        RegimeClassifierV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        RegimeClassifierV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    p = RegimeClassifierV1()
    status = p.check_self()
    assert status.state is HealthState.OK
    assert "regime_classifier_v1" in status.detail


def test_lifecycle_default_active() -> None:
    p = RegimeClassifierV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE
