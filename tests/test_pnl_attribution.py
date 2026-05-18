"""Tests for ``learning_engine/performance_analysis/pnl_attribution.py`` (S-04.1)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from core.contracts.backtest_result import BacktestTrade
from core.contracts.events import Side
from learning_engine.performance_analysis.pnl_attribution import (
    NEW_PIP_DEPENDENCIES,
    AttributedTrade,
    PnLAttribution,
    attribute_pnl,
    attribute_pnl_by_symbol,
    empty_attribution,
)


def _trade(
    *,
    ts_ns: int = 1_000,
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    qty: float = 1.0,
    price: float = 100.0,
    pnl_usd: float = 0.0,
    fee_usd: float = 0.0,
) -> BacktestTrade:
    return BacktestTrade(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        pnl_usd=pnl_usd,
        fee_usd=fee_usd,
    )


# ---------------------------------------------------------------------------
# Sanity / contract surface
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_empty_attribution_is_zero() -> None:
    e = empty_attribution()
    assert e.n_trades == 0
    assert e.notional_usd == 0.0
    assert e.realised_pnl_usd == 0.0
    assert e.signal_pnl_usd == 0.0
    assert e.slippage_pnl_usd == 0.0
    assert e.fee_pnl_usd == 0.0
    assert e.slippage_bps() == 0.0
    assert e.fee_bps() == 0.0


def test_empty_attribution_is_canonical_singleton_value() -> None:
    assert empty_attribution() == empty_attribution()
    assert empty_attribution() == attribute_pnl([])


# ---------------------------------------------------------------------------
# AttributedTrade validation
# ---------------------------------------------------------------------------


def test_attributed_trade_rejects_non_backtest_trade() -> None:
    with pytest.raises(TypeError, match="trade must be BacktestTrade"):
        AttributedTrade(trade="not-a-trade", signal_price=100.0)  # type: ignore[arg-type]


def test_attributed_trade_rejects_non_numeric_signal_price() -> None:
    with pytest.raises(TypeError, match="signal_price must be float"):
        AttributedTrade(trade=_trade(), signal_price="100")  # type: ignore[arg-type]


def test_attributed_trade_rejects_nan_signal_price() -> None:
    with pytest.raises(ValueError, match="must not be NaN"):
        AttributedTrade(trade=_trade(), signal_price=float("nan"))


def test_attributed_trade_rejects_zero_signal_price() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        AttributedTrade(trade=_trade(), signal_price=0.0)


def test_attributed_trade_rejects_negative_signal_price() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        AttributedTrade(trade=_trade(), signal_price=-100.0)


def test_attributed_trade_accepts_int_signal_price() -> None:
    AttributedTrade(trade=_trade(), signal_price=100)


def test_attributed_trade_is_frozen() -> None:
    at = AttributedTrade(trade=_trade(), signal_price=100.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        at.signal_price = 200.0  # type: ignore[misc]


def test_attributed_trade_is_hashable() -> None:
    at = AttributedTrade(trade=_trade(), signal_price=100.0)
    assert hash(at) == hash(at)
    assert len({at, at}) == 1


# ---------------------------------------------------------------------------
# PnLAttribution validation
# ---------------------------------------------------------------------------


def test_pnl_attribution_rejects_negative_n_trades() -> None:
    with pytest.raises(ValueError, match="n_trades must be >= 0"):
        PnLAttribution(
            n_trades=-1,
            notional_usd=0.0,
            realised_pnl_usd=0.0,
            signal_pnl_usd=0.0,
            slippage_pnl_usd=0.0,
            fee_pnl_usd=0.0,
        )


def test_pnl_attribution_rejects_negative_notional() -> None:
    with pytest.raises(ValueError, match="notional_usd must be >= 0"):
        PnLAttribution(
            n_trades=0,
            notional_usd=-1.0,
            realised_pnl_usd=0.0,
            signal_pnl_usd=0.0,
            slippage_pnl_usd=0.0,
            fee_pnl_usd=0.0,
        )


def test_pnl_attribution_rejects_positive_fee_pnl() -> None:
    with pytest.raises(ValueError, match="fee_pnl_usd must be <= 0"):
        PnLAttribution(
            n_trades=0,
            notional_usd=0.0,
            realised_pnl_usd=0.0,
            signal_pnl_usd=0.0,
            slippage_pnl_usd=0.0,
            fee_pnl_usd=1.0,
        )


def test_pnl_attribution_is_frozen_and_hashable() -> None:
    p = empty_attribution()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.n_trades = 5  # type: ignore[misc]
    assert hash(p) == hash(empty_attribution())
    assert len({p, empty_attribution()}) == 1


# ---------------------------------------------------------------------------
# attribute_pnl: single-trade decompositions
# ---------------------------------------------------------------------------


def test_buy_with_no_slippage_no_fees_attributes_all_to_signal() -> None:
    # BUY 1 unit at signal=100, fill=100, no fees, realised PnL = 5
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=100.0, pnl_usd=5.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.n_trades == 1
    assert r.notional_usd == 100.0
    assert r.realised_pnl_usd == 5.0
    assert r.signal_pnl_usd == 5.0
    assert r.slippage_pnl_usd == 0.0
    assert r.fee_pnl_usd == 0.0


def test_buy_paying_more_than_signal_is_negative_slippage() -> None:
    # BUY 2 units at signal=100, fill=101 → cost = (101-100) * 2 = 2
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=2.0, price=101.0, pnl_usd=10.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.slippage_pnl_usd == pytest.approx(-2.0)
    # realised = signal + slippage + fee  →  signal = 10 - (-2) - 0 = 12
    assert r.signal_pnl_usd == pytest.approx(12.0)
    assert r.fee_pnl_usd == 0.0


def test_sell_receiving_less_than_signal_is_negative_slippage() -> None:
    # SELL 2 units at signal=100, fill=99 → loss = (100-99) * 2 = 2 (worse)
    at = AttributedTrade(
        trade=_trade(side=Side.SELL, qty=2.0, price=99.0, pnl_usd=8.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.slippage_pnl_usd == pytest.approx(-2.0)
    assert r.signal_pnl_usd == pytest.approx(10.0)
    assert r.fee_pnl_usd == 0.0


def test_buy_paying_less_than_signal_is_positive_slippage() -> None:
    # BUY 1 unit at signal=100, fill=99 → execution beat the signal by 1
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=99.0, pnl_usd=10.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.slippage_pnl_usd == pytest.approx(1.0)
    # signal pnl is then realised - slippage - fee = 10 - 1 - 0 = 9
    assert r.signal_pnl_usd == pytest.approx(9.0)


def test_sell_receiving_more_than_signal_is_positive_slippage() -> None:
    # SELL 1 unit at signal=100, fill=101 → execution beat the signal by 1
    at = AttributedTrade(
        trade=_trade(side=Side.SELL, qty=1.0, price=101.0, pnl_usd=10.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.slippage_pnl_usd == pytest.approx(1.0)
    assert r.signal_pnl_usd == pytest.approx(9.0)


def test_fee_is_negative_pnl_component() -> None:
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=100.0, pnl_usd=5.0, fee_usd=0.5),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.fee_pnl_usd == pytest.approx(-0.5)
    # realised = signal + slippage + fee → signal = 5 - 0 - (-0.5) = 5.5
    assert r.signal_pnl_usd == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Aggregation invariants
# ---------------------------------------------------------------------------


def test_aggregation_preserves_decomposition_identity() -> None:
    trades = [
        AttributedTrade(
            trade=_trade(
                ts_ns=i * 1_000,
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                qty=float(i + 1),
                price=100.0 + 0.5 * i,
                pnl_usd=float(i),
                fee_usd=0.05 * float(i),
            ),
            signal_price=100.0 + 0.4 * i,
        )
        for i in range(10)
    ]
    r = attribute_pnl(trades)
    # invariant: signal + slippage + fee == realised (within float epsilon)
    reconstructed = r.signal_pnl_usd + r.slippage_pnl_usd + r.fee_pnl_usd
    assert math.isclose(reconstructed, r.realised_pnl_usd, abs_tol=1e-9)


def test_aggregation_sums_n_trades_and_notional() -> None:
    a = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=100.0, pnl_usd=1.0),
        signal_price=100.0,
    )
    b = AttributedTrade(
        trade=_trade(side=Side.SELL, qty=2.0, price=50.0, pnl_usd=2.0),
        signal_price=50.0,
    )
    r = attribute_pnl([a, b])
    assert r.n_trades == 2
    assert r.notional_usd == pytest.approx(100.0 + 100.0)


def test_aggregation_empty_returns_zero_record() -> None:
    assert attribute_pnl([]) == empty_attribution()


# ---------------------------------------------------------------------------
# Type validation on the iterable
# ---------------------------------------------------------------------------


def test_attribute_pnl_rejects_non_attributed_trade() -> None:
    with pytest.raises(TypeError, match="trades must contain AttributedTrade"):
        attribute_pnl([object()])  # type: ignore[list-item]


def test_attribute_pnl_by_symbol_rejects_non_attributed_trade() -> None:
    with pytest.raises(TypeError, match="trades must contain AttributedTrade"):
        attribute_pnl_by_symbol([object()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# bps helpers
# ---------------------------------------------------------------------------


def test_slippage_bps_is_positive_when_slippage_costs_money() -> None:
    # BUY 1 unit at signal=100, fill=101 → -1 USD slippage on 101 notional
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=101.0, pnl_usd=0.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    # bps = -(-1.0) / 101.0 * 10000 ≈ 99.0099
    assert r.slippage_bps() == pytest.approx(99.0099, abs=1e-3)


def test_slippage_bps_is_negative_when_execution_beats_signal() -> None:
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=99.0, pnl_usd=0.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    # slippage is +1.0; bps = -1.0 / 99.0 * 10000 ≈ -101.01
    assert r.slippage_bps() < 0.0


def test_fee_bps_is_positive_when_fees_charged() -> None:
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=100.0, pnl_usd=0.0, fee_usd=1.0),
        signal_price=100.0,
    )
    r = attribute_pnl([at])
    assert r.fee_bps() == pytest.approx(100.0)  # 1/100 * 10000 = 100 bps


def test_bps_helpers_return_zero_when_notional_is_zero() -> None:
    e = empty_attribution()
    assert e.slippage_bps() == 0.0
    assert e.fee_bps() == 0.0


# ---------------------------------------------------------------------------
# By-symbol grouping
# ---------------------------------------------------------------------------


def test_by_symbol_groups_each_symbol_independently() -> None:
    trades = [
        AttributedTrade(
            trade=_trade(symbol="BTCUSDT", side=Side.BUY, qty=1.0, price=100.0, pnl_usd=10.0),
            signal_price=100.0,
        ),
        AttributedTrade(
            trade=_trade(
                symbol="ETHUSDT",
                side=Side.BUY,
                qty=2.0,
                price=50.0,
                pnl_usd=4.0,
                fee_usd=1.0,
            ),
            signal_price=50.0,
        ),
        AttributedTrade(
            trade=_trade(symbol="BTCUSDT", side=Side.SELL, qty=1.0, price=110.0, pnl_usd=5.0),
            signal_price=110.0,
        ),
    ]
    by = attribute_pnl_by_symbol(trades)
    assert set(by.keys()) == {"BTCUSDT", "ETHUSDT"}
    assert by["BTCUSDT"].n_trades == 2
    assert by["ETHUSDT"].n_trades == 1
    assert by["BTCUSDT"].realised_pnl_usd == pytest.approx(15.0)
    assert by["ETHUSDT"].fee_pnl_usd == pytest.approx(-1.0)


def test_by_symbol_preserves_first_seen_symbol_order() -> None:
    trades = [
        AttributedTrade(trade=_trade(symbol="ZZZ"), signal_price=100.0),
        AttributedTrade(trade=_trade(symbol="AAA"), signal_price=100.0),
        AttributedTrade(trade=_trade(symbol="MMM"), signal_price=100.0),
        AttributedTrade(trade=_trade(symbol="AAA"), signal_price=100.0),
    ]
    by = attribute_pnl_by_symbol(trades)
    assert list(by.keys()) == ["ZZZ", "AAA", "MMM"]


def test_by_symbol_empty_input_returns_empty_dict() -> None:
    assert attribute_pnl_by_symbol([]) == {}


def test_by_symbol_aggregate_equals_global_aggregate() -> None:
    trades = [
        AttributedTrade(
            trade=_trade(
                symbol="BTC" if i % 2 == 0 else "ETH",
                side=Side.BUY if i % 3 == 0 else Side.SELL,
                qty=float(i + 1),
                price=100.0 + i,
                pnl_usd=float(i),
                fee_usd=0.05 * i,
            ),
            signal_price=100.0 + 0.5 * i,
        )
        for i in range(8)
    ]
    by = attribute_pnl_by_symbol(trades)
    glob = attribute_pnl(trades)
    sum_n = sum(p.n_trades for p in by.values())
    sum_real = sum(p.realised_pnl_usd for p in by.values())
    sum_fee = sum(p.fee_pnl_usd for p in by.values())
    sum_slip = sum(p.slippage_pnl_usd for p in by.values())
    sum_signal = sum(p.signal_pnl_usd for p in by.values())
    sum_notional = sum(p.notional_usd for p in by.values())
    assert sum_n == glob.n_trades
    assert math.isclose(sum_real, glob.realised_pnl_usd, abs_tol=1e-9)
    assert math.isclose(sum_fee, glob.fee_pnl_usd, abs_tol=1e-9)
    assert math.isclose(sum_slip, glob.slippage_pnl_usd, abs_tol=1e-9)
    assert math.isclose(sum_signal, glob.signal_pnl_usd, abs_tol=1e-9)
    assert math.isclose(sum_notional, glob.notional_usd, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Replay determinism (INV-15)
# ---------------------------------------------------------------------------


def test_replay_determinism_across_three_runs() -> None:
    trades = [
        AttributedTrade(
            trade=_trade(
                ts_ns=i * 1_000,
                symbol=f"SYM{i % 3}",
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                qty=float(i + 1),
                price=100.0 + 0.1 * i,
                pnl_usd=float(i) * 0.5,
                fee_usd=0.01 * i,
            ),
            signal_price=99.5 + 0.1 * i,
        )
        for i in range(20)
    ]
    a = attribute_pnl(trades)
    b = attribute_pnl(trades)
    c = attribute_pnl(trades)
    assert a == b == c

    by_a = attribute_pnl_by_symbol(trades)
    by_b = attribute_pnl_by_symbol(trades)
    assert by_a == by_b
    assert list(by_a.keys()) == list(by_b.keys())


def test_zero_qty_trade_contributes_no_notional_no_slippage() -> None:
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=0.0, price=100.0, pnl_usd=0.0),
        signal_price=99.0,
    )
    r = attribute_pnl([at])
    assert r.notional_usd == 0.0
    assert r.slippage_pnl_usd == 0.0
    assert r.signal_pnl_usd == 0.0


def test_single_trade_attribution_consumes_iterator_once() -> None:
    at = AttributedTrade(
        trade=_trade(side=Side.BUY, qty=1.0, price=100.0, pnl_usd=5.0),
        signal_price=100.0,
    )

    def gen():
        yield at

    r = attribute_pnl(gen())
    assert r.n_trades == 1
    assert r.realised_pnl_usd == 5.0
