"""Unit tests for ``intelligence_engine.portfolio.exposure_manager``."""

from __future__ import annotations

import pytest

from intelligence_engine.portfolio import ExposureManager


def test_starts_empty() -> None:
    em = ExposureManager()
    assert em.notional("BTC-USD") == 0.0
    assert dict(em.view()) == {}


def test_buy_adds_positive_notional() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=1, symbol="BTC-USD", side="BUY", notional_usd=1_000.0)
    assert em.notional("BTC-USD") == pytest.approx(1_000.0)


def test_sell_adds_negative_notional() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=1, symbol="BTC-USD", side="SELL", notional_usd=400.0)
    assert em.notional("BTC-USD") == pytest.approx(-400.0)


def test_fills_aggregate_signed() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=1, symbol="BTC-USD", side="BUY", notional_usd=1_000.0)
    em.apply_fill(ts_ns=2, symbol="BTC-USD", side="SELL", notional_usd=300.0)
    em.apply_fill(ts_ns=3, symbol="ETH-USD", side="BUY", notional_usd=500.0)
    assert em.notional("BTC-USD") == pytest.approx(700.0)
    assert em.notional("ETH-USD") == pytest.approx(500.0)


def test_view_is_a_copy() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=1, symbol="BTC-USD", side="BUY", notional_usd=1_000.0)
    v1 = em.view()
    v1_dict = dict(v1)  # type: ignore[arg-type]
    v1_dict["MUTATED"] = 999.0
    # Internal state must be unaffected by mutating the snapshot dict.
    assert "MUTATED" not in em.view()


def test_snapshot_materialises_exposuresnapshot() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=1, symbol="BTC-USD", side="BUY", notional_usd=1_000.0)
    snap = em.snapshot(ts_ns=2)
    assert snap.ts_ns == 2
    assert snap.notional("BTC-USD") == pytest.approx(1_000.0)
    assert snap.notional("UNKNOWN") == 0.0


def test_apply_fill_rejects_bad_inputs() -> None:
    em = ExposureManager()
    with pytest.raises(ValueError):
        em.apply_fill(ts_ns=1, symbol="", side="BUY", notional_usd=1.0)
    with pytest.raises(ValueError):
        em.apply_fill(ts_ns=1, symbol="X", side="HOLD", notional_usd=1.0)
    with pytest.raises(ValueError):
        em.apply_fill(ts_ns=1, symbol="X", side="BUY", notional_usd=-1.0)
    with pytest.raises(ValueError):
        em.apply_fill(ts_ns=0, symbol="X", side="BUY", notional_usd=1.0)


def test_apply_fill_rejects_out_of_order_ts() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=10, symbol="X", side="BUY", notional_usd=1.0)
    with pytest.raises(ValueError):
        em.apply_fill(ts_ns=5, symbol="X", side="BUY", notional_usd=1.0)


def test_snapshot_rejects_predating_last_update() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=10, symbol="X", side="BUY", notional_usd=1.0)
    with pytest.raises(ValueError):
        em.snapshot(ts_ns=5)


def test_reset_returns_to_initial() -> None:
    em = ExposureManager()
    em.apply_fill(ts_ns=1, symbol="X", side="BUY", notional_usd=1.0)
    em.reset()
    assert dict(em.view()) == {}
    em.apply_fill(ts_ns=1, symbol="X", side="BUY", notional_usd=1.0)
    assert em.notional("X") == pytest.approx(1.0)


def test_replay_determinism_same_fill_sequence_same_state() -> None:
    fills = [
        (1, "BTC-USD", "BUY", 100.0),
        (2, "ETH-USD", "BUY", 200.0),
        (3, "BTC-USD", "SELL", 50.0),
    ]
    a = ExposureManager()
    b = ExposureManager()
    for ts, sym, side, notional in fills:
        a.apply_fill(ts_ns=ts, symbol=sym, side=side, notional_usd=notional)
        b.apply_fill(ts_ns=ts, symbol=sym, side=side, notional_usd=notional)
    assert dict(a.view()) == dict(b.view())
    assert a.snapshot(ts_ns=4) == b.snapshot(ts_ns=4)
