"""Tests for the IND-L07 vpin_imbalance v1 plugin (Indira plugin #6)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.vpin_imbalance import VpinImbalanceV1


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
    p = VpinImbalanceV1(bucket_volume=10.0, window_size=4)
    out: tuple = ()
    for i in range(3):  # 3 sealed buckets only — window not yet full
        out = p.on_tick(_tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0))
    assert out == ()


def test_sustained_buy_imbalance_emits_buy() -> None:
    p = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=0.3,
        confidence_scale=0.5,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):  # 5 buckets sealed; on the 5th, window is full
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.plugin_chain == ("vpin_imbalance_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert float(sig.meta["vpin"]) > 0.3
    assert float(sig.meta["last_bucket_imbalance"]) > 0.0


def test_sustained_sell_imbalance_emits_sell() -> None:
    p = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=0.3,
        confidence_scale=0.5,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=99.0, volume=10.0)
        )
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert float(out[0].meta["last_bucket_imbalance"]) < 0.0


def test_balanced_flow_emits_nothing() -> None:
    p = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=0.05,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    # Alternating BUY / SELL aggressors fill each bucket 50/50.
    for i in range(8):
        last = 101.0 if i % 2 == 0 else 99.0
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=last, volume=5.0)
        )
    assert out == ()


def test_below_threshold_emits_nothing() -> None:
    p = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=0.95,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    # All buckets are 100% buy → vpin == 1.0; with threshold 0.95 we
    # still emit, then re-run below with threshold == 1.0 to confirm
    # the strict-greater check silences a saturated stream.
    assert len(out) == 1
    p3 = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=1.0,  # vpin can be at most 1.0; "<=" never strict-greater
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out_silent: tuple = ()
    for i in range(5):
        out_silent = p3.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    assert out_silent == ()


def test_zero_volume_drops() -> None:
    p = VpinImbalanceV1(bucket_volume=10.0, window_size=2)
    assert p.on_tick(_tick(0, bid=99.0, ask=101.0, last=101.0, volume=0.0)) == ()


def test_invalid_book_drops() -> None:
    p = VpinImbalanceV1(bucket_volume=10.0, window_size=2)
    assert p.on_tick(_tick(0, bid=0.0, ask=101.0, last=100.0, volume=10.0)) == ()
    assert p.on_tick(_tick(1, bid=99.0, ask=0.0, last=100.0, volume=10.0)) == ()
    assert p.on_tick(_tick(2, bid=101.0, ask=99.0, last=100.0, volume=10.0)) == ()
    assert p.on_tick(_tick(3, bid=99.0, ask=101.0, last=0.0, volume=10.0)) == ()


def test_negative_volume_drops() -> None:
    p = VpinImbalanceV1(bucket_volume=10.0, window_size=2)
    assert p.on_tick(_tick(0, bid=99.0, ask=101.0, last=101.0, volume=-1.0)) == ()


def test_replay_determinism() -> None:
    seq = [
        _tick(i, bid=99.0, ask=101.0, last=101.0 if i % 2 == 0 else 99.0,
              volume=10.0)
        for i in range(10)
    ]
    p1 = VpinImbalanceV1(bucket_volume=10.0, window_size=4)
    p2 = VpinImbalanceV1(bucket_volume=10.0, window_size=4)
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=0.95,
        confidence_scale=1000.0,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(5):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=101.0, volume=10.0)
        )
    assert out == ()


def test_oversized_tick_seals_multiple_buckets() -> None:
    p = VpinImbalanceV1(
        bucket_volume=5.0,
        window_size=4,
        vpin_threshold=0.5,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    # Single tick with volume = 25 = 5 full buckets, all buy.
    out = p.on_tick(_tick(0, bid=99.0, ask=101.0, last=101.0, volume=25.0))
    assert len(out) == 1
    assert out[0].side is Side.BUY


def test_neutral_aggressor_inferred_for_mid_price_trade() -> None:
    # last is strictly between bid and ask -> 50/50 split, not informed,
    # so no signal should fire.
    p = VpinImbalanceV1(
        bucket_volume=10.0,
        window_size=4,
        vpin_threshold=0.05,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(5):
        out = p.on_tick(
            _tick(i, bid=99.0, ask=101.0, last=100.0, volume=10.0)
        )
    assert out == ()


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        VpinImbalanceV1(bucket_volume=0.0)
    with pytest.raises(ValueError):
        VpinImbalanceV1(bucket_volume=-1.0)
    with pytest.raises(ValueError):
        VpinImbalanceV1(window_size=1)
    with pytest.raises(ValueError):
        VpinImbalanceV1(vpin_threshold=-0.1)
    with pytest.raises(ValueError):
        VpinImbalanceV1(vpin_threshold=1.5)
    with pytest.raises(ValueError):
        VpinImbalanceV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        VpinImbalanceV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        VpinImbalanceV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    p = VpinImbalanceV1()
    status = p.check_self()
    assert status.state is HealthState.OK
    assert "vpin_imbalance_v1" in status.detail


def test_lifecycle_default_active() -> None:
    p = VpinImbalanceV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE
