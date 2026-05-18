"""Tests for ``learning_engine/performance_analysis/execution_quality.py`` (S-04.3)."""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts.backtest_result import BacktestTrade
from core.contracts.events import Side
from learning_engine.performance_analysis.execution_quality import (
    NEW_PIP_DEPENDENCIES,
    BenchmarkedFill,
    ExecutionQualityReport,
    empty_report,
    is_cost_bps,
    score_execution,
    score_execution_by_symbol,
    timing_cost_bps,
    vwap_deviation_bps,
)


def _trade(
    *,
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    qty: float = 1.0,
    price: float = 100.0,
    pnl_usd: float = 0.0,
    fee_usd: float = 0.0,
) -> BacktestTrade:
    return BacktestTrade(
        ts_ns=0,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        pnl_usd=pnl_usd,
        fee_usd=fee_usd,
    )


def _fill(
    *,
    trade: BacktestTrade | None = None,
    arrival_price: float = 100.0,
    interval_vwap: float = 100.0,
    interval_volume: float = 0.0,
) -> BenchmarkedFill:
    return BenchmarkedFill(
        trade=trade if trade is not None else _trade(),
        arrival_price=arrival_price,
        interval_vwap=interval_vwap,
        interval_volume=interval_volume,
    )


# ---------------------------------------------------------------------------
# Sanity / surface
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_empty_report_is_zero() -> None:
    r = empty_report()
    assert r.n_fills == 0
    assert r.notional_usd == 0.0
    assert r.is_cost_usd == 0.0
    assert r.vwap_deviation_usd == 0.0
    assert r.timing_cost_usd == 0.0
    assert r.participation_rate == 0.0


def test_score_execution_empty_returns_canonical_zero() -> None:
    assert score_execution([]) == empty_report()


def test_score_execution_by_symbol_empty_returns_empty_map() -> None:
    assert score_execution_by_symbol([]) == {}


# ---------------------------------------------------------------------------
# BenchmarkedFill validation
# ---------------------------------------------------------------------------


def test_fill_rejects_non_trade() -> None:
    with pytest.raises(TypeError, match="trade must be BacktestTrade"):
        BenchmarkedFill(
            trade="not-a-trade",  # type: ignore[arg-type]
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=0.0,
        )


def test_fill_rejects_zero_arrival_price() -> None:
    with pytest.raises(ValueError, match="arrival_price must be > 0"):
        _fill(arrival_price=0.0)


def test_fill_rejects_negative_arrival_price() -> None:
    with pytest.raises(ValueError, match="arrival_price must be > 0"):
        _fill(arrival_price=-1.0)


def test_fill_rejects_zero_interval_vwap() -> None:
    with pytest.raises(ValueError, match="interval_vwap must be > 0"):
        _fill(interval_vwap=0.0)


def test_fill_rejects_nan_arrival_price() -> None:
    with pytest.raises(ValueError, match="arrival_price must be finite"):
        _fill(arrival_price=float("nan"))


def test_fill_rejects_inf_interval_vwap() -> None:
    with pytest.raises(ValueError, match="interval_vwap must be finite"):
        _fill(interval_vwap=float("inf"))


def test_fill_rejects_negative_interval_volume() -> None:
    with pytest.raises(ValueError, match="interval_volume must be >= 0"):
        _fill(interval_volume=-1.0)


def test_fill_rejects_nan_interval_volume() -> None:
    with pytest.raises(ValueError, match="interval_volume must be finite"):
        _fill(interval_volume=float("nan"))


def test_fill_is_frozen() -> None:
    f = _fill()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.arrival_price = 200.0  # type: ignore[misc]


def test_fill_is_hashable() -> None:
    f = _fill()
    assert len({f, f}) == 1


def test_fill_accepts_zero_interval_volume() -> None:
    """Zero volume == 'participation rate undefined'; must not raise."""
    f = _fill(interval_volume=0.0)
    assert f.interval_volume == 0.0


# ---------------------------------------------------------------------------
# ExecutionQualityReport validation
# ---------------------------------------------------------------------------


def _r(**overrides) -> ExecutionQualityReport:
    base: dict[str, object] = dict(
        n_fills=0,
        notional_usd=0.0,
        is_cost_usd=0.0,
        vwap_deviation_usd=0.0,
        timing_cost_usd=0.0,
        participation_rate=0.0,
    )
    base.update(overrides)
    return ExecutionQualityReport(**base)  # type: ignore[arg-type]


def test_report_rejects_negative_n_fills() -> None:
    with pytest.raises(ValueError, match="n_fills must be >= 0"):
        _r(n_fills=-1)


def test_report_rejects_negative_notional() -> None:
    with pytest.raises(ValueError, match="notional_usd must be >= 0"):
        _r(notional_usd=-1.0)


def test_report_rejects_negative_participation() -> None:
    with pytest.raises(ValueError, match="participation_rate must be >= 0"):
        _r(participation_rate=-0.1)


def test_report_rejects_nan_field() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _r(is_cost_usd=float("nan"))


# ---------------------------------------------------------------------------
# Single-fill decomposition
# ---------------------------------------------------------------------------


def test_buy_at_arrival_with_no_drift_yields_zero_cost() -> None:
    f = _fill(
        trade=_trade(side=Side.BUY, qty=2.0, price=100.0),
        arrival_price=100.0,
        interval_vwap=100.0,
    )
    r = score_execution([f])
    assert r.n_fills == 1
    assert r.notional_usd == pytest.approx(200.0)
    assert r.is_cost_usd == 0.0
    assert r.vwap_deviation_usd == 0.0
    assert r.timing_cost_usd == 0.0


def test_buy_paying_above_arrival_is_negative_is_cost() -> None:
    # BUY 1 @ 101 vs arrival 100 ⇒ paid +1 ⇒ is_cost = -1 USD
    f = _fill(
        trade=_trade(side=Side.BUY, qty=1.0, price=101.0),
        arrival_price=100.0,
        interval_vwap=101.0,
    )
    r = score_execution([f])
    assert r.is_cost_usd == pytest.approx(-1.0)
    assert r.vwap_deviation_usd == 0.0
    assert r.timing_cost_usd == pytest.approx(-1.0)


def test_buy_below_vwap_is_positive_vwap_deviation() -> None:
    # BUY 1 @ 100 vs vwap 101 ⇒ saved +1 ⇒ vwap_dev = +1 USD
    f = _fill(
        trade=_trade(side=Side.BUY, qty=1.0, price=100.0),
        arrival_price=100.0,
        interval_vwap=101.0,
    )
    r = score_execution([f])
    assert r.is_cost_usd == 0.0
    assert r.vwap_deviation_usd == pytest.approx(1.0)
    # timing = is - vwap_dev = 0 - 1 = -1 (drift between arrival and interval was bad)
    assert r.timing_cost_usd == pytest.approx(-1.0)


def test_sell_above_arrival_is_positive_is_cost() -> None:
    # SELL 1 @ 101 vs arrival 100 ⇒ received +1 ⇒ is_cost = +1 USD
    f = _fill(
        trade=_trade(side=Side.SELL, qty=1.0, price=101.0),
        arrival_price=100.0,
        interval_vwap=101.0,
    )
    r = score_execution([f])
    assert r.is_cost_usd == pytest.approx(1.0)
    assert r.vwap_deviation_usd == 0.0
    assert r.timing_cost_usd == pytest.approx(1.0)


def test_sell_below_vwap_is_negative_vwap_deviation() -> None:
    # SELL 1 @ 100 vs vwap 101 ⇒ left +1 on table ⇒ vwap_dev = -1 USD
    f = _fill(
        trade=_trade(side=Side.SELL, qty=1.0, price=100.0),
        arrival_price=100.0,
        interval_vwap=101.0,
    )
    r = score_execution([f])
    assert r.is_cost_usd == 0.0
    assert r.vwap_deviation_usd == pytest.approx(-1.0)
    assert r.timing_cost_usd == pytest.approx(1.0)


def test_decomposition_identity_holds_per_fill() -> None:
    f = _fill(
        trade=_trade(side=Side.BUY, qty=3.0, price=99.0),
        arrival_price=100.0,
        interval_vwap=98.0,
    )
    r = score_execution([f])
    assert r.is_cost_usd == pytest.approx(r.vwap_deviation_usd + r.timing_cost_usd)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_sums_notional_and_costs() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=101.0),
            arrival_price=100.0,
            interval_vwap=100.5,
        ),
        _fill(
            trade=_trade(side=Side.SELL, qty=2.0, price=102.0),
            arrival_price=101.0,
            interval_vwap=101.5,
        ),
    ]
    r = score_execution(fills)
    assert r.n_fills == 2
    assert r.notional_usd == pytest.approx(101.0 + 204.0)
    # IS BUY: -(101-100)*1*1 = -1
    # IS SELL: -(102-101)*2*-1 = +2
    assert r.is_cost_usd == pytest.approx(-1.0 + 2.0)
    # VWAP-dev BUY: -(101-100.5)*1*1 = -0.5
    # VWAP-dev SELL: -(102-101.5)*2*-1 = +1
    assert r.vwap_deviation_usd == pytest.approx(-0.5 + 1.0)
    # Timing = IS - VWAP-dev
    assert r.timing_cost_usd == pytest.approx(r.is_cost_usd - r.vwap_deviation_usd)


def test_decomposition_identity_holds_across_aggregate() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.5, price=100.5),
            arrival_price=100.0,
            interval_vwap=100.2,
        ),
        _fill(
            trade=_trade(side=Side.BUY, qty=2.0, price=99.5),
            arrival_price=100.0,
            interval_vwap=99.8,
        ),
        _fill(
            trade=_trade(side=Side.SELL, qty=0.5, price=101.0),
            arrival_price=100.5,
            interval_vwap=100.7,
        ),
    ]
    r = score_execution(fills)
    assert r.is_cost_usd == pytest.approx(r.vwap_deviation_usd + r.timing_cost_usd)


def test_participation_rate_weighted_by_notional() -> None:
    # Two fills: BUY 1 @ 100 (notional=100) part=0.10, BUY 4 @ 100 (notional=400) part=0.40
    # weighted = (0.10 * 100 + 0.40 * 400) / 500 = (10 + 160) / 500 = 0.34
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=10.0,
        ),
        _fill(
            trade=_trade(side=Side.BUY, qty=4.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=10.0,
        ),
    ]
    r = score_execution(fills)
    assert r.participation_rate == pytest.approx(0.34)


def test_participation_rate_skips_zero_volume_fills() -> None:
    # First fill has zero volume → excluded from participation; second sets rate
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=0.0,
        ),
        _fill(
            trade=_trade(side=Side.BUY, qty=2.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=10.0,
        ),
    ]
    r = score_execution(fills)
    assert r.participation_rate == pytest.approx(0.20)


def test_participation_rate_zero_when_all_volumes_zero() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=0.0,
        ),
        _fill(
            trade=_trade(side=Side.BUY, qty=2.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=100.0,
            interval_volume=0.0,
        ),
    ]
    r = score_execution(fills)
    assert r.participation_rate == 0.0


# ---------------------------------------------------------------------------
# By-symbol grouping
# ---------------------------------------------------------------------------


def test_by_symbol_groups_each_symbol_independently() -> None:
    fills = [
        _fill(
            trade=_trade(symbol="BTCUSDT", side=Side.BUY, qty=1.0, price=101.0),
            arrival_price=100.0,
            interval_vwap=100.0,
        ),
        _fill(
            trade=_trade(symbol="ETHUSDT", side=Side.BUY, qty=2.0, price=200.0),
            arrival_price=200.0,
            interval_vwap=200.0,
        ),
        _fill(
            trade=_trade(symbol="BTCUSDT", side=Side.SELL, qty=1.0, price=99.0),
            arrival_price=100.0,
            interval_vwap=100.0,
        ),
    ]
    by_sym = score_execution_by_symbol(fills)
    assert set(by_sym) == {"BTCUSDT", "ETHUSDT"}
    assert by_sym["BTCUSDT"].n_fills == 2
    assert by_sym["ETHUSDT"].n_fills == 1


def test_by_symbol_preserves_first_seen_order() -> None:
    fills = [
        _fill(trade=_trade(symbol="ZZZ")),
        _fill(trade=_trade(symbol="AAA")),
        _fill(trade=_trade(symbol="ZZZ")),
        _fill(trade=_trade(symbol="MMM")),
    ]
    keys = list(score_execution_by_symbol(fills))
    assert keys == ["ZZZ", "AAA", "MMM"]


def test_by_symbol_aggregate_matches_total() -> None:
    fills = [
        _fill(
            trade=_trade(symbol="A", side=Side.BUY, qty=1.0, price=101.0),
            arrival_price=100.0,
            interval_vwap=100.5,
        ),
        _fill(
            trade=_trade(symbol="B", side=Side.SELL, qty=2.0, price=102.0),
            arrival_price=101.0,
            interval_vwap=101.5,
        ),
    ]
    total = score_execution(fills)
    by_sym = score_execution_by_symbol(fills)
    is_sum = sum(r.is_cost_usd for r in by_sym.values())
    vwap_sum = sum(r.vwap_deviation_usd for r in by_sym.values())
    timing_sum = sum(r.timing_cost_usd for r in by_sym.values())
    assert total.is_cost_usd == pytest.approx(is_sum)
    assert total.vwap_deviation_usd == pytest.approx(vwap_sum)
    assert total.timing_cost_usd == pytest.approx(timing_sum)


def test_by_symbol_rejects_non_fill() -> None:
    with pytest.raises(TypeError, match="must contain BenchmarkedFill"):
        score_execution_by_symbol([object()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Bps helpers
# ---------------------------------------------------------------------------


def test_is_cost_bps_normalises_against_notional() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=101.0),
            arrival_price=100.0,
            interval_vwap=101.0,
        ),
    ]
    r = score_execution(fills)
    # is_cost = -1, notional = 101 → bps = -1/101 * 10_000 ≈ -99.0099
    assert is_cost_bps(r) == pytest.approx(-1.0 / 101.0 * 10_000.0)


def test_vwap_deviation_bps_normalises_against_notional() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=101.0,
        ),
    ]
    r = score_execution(fills)
    assert vwap_deviation_bps(r) == pytest.approx(1.0 / 100.0 * 10_000.0)


def test_timing_cost_bps_normalises_against_notional() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=100.0),
            arrival_price=100.0,
            interval_vwap=101.0,
        ),
    ]
    r = score_execution(fills)
    # timing = is - vwap_dev = 0 - 1 = -1, notional = 100
    assert timing_cost_bps(r) == pytest.approx(-1.0 / 100.0 * 10_000.0)


def test_bps_helpers_return_zero_when_notional_zero() -> None:
    r = empty_report()
    assert is_cost_bps(r) == 0.0
    assert vwap_deviation_bps(r) == 0.0
    assert timing_cost_bps(r) == 0.0


# ---------------------------------------------------------------------------
# Replay determinism (INV-15) + iterator inputs
# ---------------------------------------------------------------------------


def test_replay_determinism_across_three_runs() -> None:
    fills: list[BenchmarkedFill] = []
    for i in range(20):
        side = Side.BUY if i % 2 == 0 else Side.SELL
        fills.append(
            _fill(
                trade=_trade(
                    symbol=f"S{i % 3}",
                    side=side,
                    qty=1.0 + 0.1 * i,
                    price=100.0 + 0.05 * i,
                ),
                arrival_price=100.0 + 0.04 * i,
                interval_vwap=100.0 + 0.06 * i,
                interval_volume=10.0 + i,
            )
        )
    a = score_execution(fills)
    b = score_execution(fills)
    c = score_execution(fills)
    assert a == b == c

    by_a = score_execution_by_symbol(fills)
    by_b = score_execution_by_symbol(fills)
    assert by_a == by_b
    assert list(by_a) == list(by_b)


def test_score_execution_consumes_generator() -> None:
    fills = [
        _fill(
            trade=_trade(side=Side.BUY, qty=1.0, price=101.0),
            arrival_price=100.0,
            interval_vwap=100.5,
        ),
        _fill(
            trade=_trade(side=Side.SELL, qty=2.0, price=102.0),
            arrival_price=101.0,
            interval_vwap=101.5,
        ),
    ]

    def gen():
        yield from fills

    r = score_execution(gen())
    assert r.n_fills == 2


def test_score_execution_rejects_non_fill() -> None:
    with pytest.raises(TypeError, match="must contain BenchmarkedFill"):
        score_execution([object()])  # type: ignore[list-item]
