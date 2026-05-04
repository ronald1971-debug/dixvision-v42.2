"""Paper-S3 — BacktestResult contract + ingestion seam unit tests.

Covers:

1. Frozen-dataclass shape (hashable, slotted, immutable).
2. Validation in ``__post_init__`` (period window, equity ordering,
   trade window, metric ranges, source pattern, policy_hash pattern,
   meta typing).
3. ``project_to_trade_outcomes`` projects 1-to-1, lossless, and is
   replay-deterministic across two identical inputs (INV-15).
4. ``BacktestIngester`` Protocol is ``runtime_checkable`` and
   ``BacktestIngestionError`` is a ``ValueError`` subclass.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.contracts.api.backtest_ingestion import (
    BacktestIngester,
    BacktestIngestionError,
)
from core.contracts.backtest_result import (
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
    build_equity_curve,
    project_to_trade_outcomes,
)
from core.contracts.events import ExecutionStatus, Side
from core.contracts.signal_trust import SignalTrust

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade(
    ts_ns: int,
    *,
    symbol: str = "BTC-USDT",
    side: Side = Side.BUY,
    qty: float = 1.0,
    price: float = 50_000.0,
    pnl: float = 100.0,
    fee: float = 0.5,
) -> BacktestTrade:
    return BacktestTrade(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        pnl_usd=pnl,
        fee_usd=fee,
    )


def _metrics(n: int = 0, **overrides: float) -> BacktestMetrics:
    base: dict[str, float | int] = dict(
        n_trades=n,
        win_rate=0.55,
        total_return=0.15,
        max_drawdown=0.10,
        sharpe=1.4,
        sortino=2.1,
    )
    base.update(overrides)
    return BacktestMetrics(**base)  # type: ignore[arg-type]


def _result(
    *,
    n_trades: int = 2,
    period_start: int = 1_000,
    period_end: int = 10_000,
    source: str = "tradingview",
    trust: SignalTrust = SignalTrust.EXTERNAL_LOW,
    equity: tuple[EquityPoint, ...] | None = None,
    trades: tuple[BacktestTrade, ...] | None = None,
) -> BacktestResult:
    equity = equity if equity is not None else (
        EquityPoint(ts_ns=period_start, equity_usd=10_000.0),
        EquityPoint(ts_ns=period_end, equity_usd=11_500.0),
    )
    trades = trades if trades is not None else tuple(
        _trade(period_start + i * 100) for i in range(n_trades)
    )
    return BacktestResult(
        ts_ns=period_end + 1,
        source=source,
        backtest_id="bt-001",
        strategy_id="meanrev_btc_v1",
        symbol="BTC-USDT",
        period_start_ns=period_start,
        period_end_ns=period_end,
        equity_curve=equity,
        trades=trades,
        metrics=_metrics(n=len(trades)),
        trust=trust,
    )


# ---------------------------------------------------------------------------
# 1. Construction & immutability
# ---------------------------------------------------------------------------


def test_construction_minimal_valid() -> None:
    r = _result()
    assert r.source == "tradingview"
    assert r.trust is SignalTrust.EXTERNAL_LOW
    assert len(r.trades) == 2
    assert len(r.equity_curve) == 2


def test_result_is_frozen() -> None:
    r = _result()
    with pytest.raises(FrozenInstanceError):
        r.source = "mt5"  # type: ignore[misc]


def test_result_equality_structural() -> None:
    """Structural equality across two identically-built results
    (note: dict-typed ``meta`` makes the dataclass unhashable, same
    as :class:`TradeOutcome` — equality is what we rely on, not hash)."""
    r1 = _result()
    r2 = _result()
    assert r1 == r2


def test_trade_and_metrics_frozen() -> None:
    t = _trade(1_000)
    with pytest.raises(FrozenInstanceError):
        t.qty = 99.0  # type: ignore[misc]
    m = _metrics()
    with pytest.raises(FrozenInstanceError):
        m.sharpe = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------


def test_source_pattern_rejected() -> None:
    with pytest.raises(ValueError, match="source must match"):
        _result(source="TradingView")
    with pytest.raises(ValueError, match="source must match"):
        _result(source="x")  # too short
    with pytest.raises(ValueError, match="source must match"):
        _result(source="1abc")  # leading digit


def test_period_window_inverted() -> None:
    with pytest.raises(ValueError, match="period_end_ns must be >="):
        BacktestResult(
            ts_ns=10,
            source="mt5",
            backtest_id="b",
            strategy_id="s",
            symbol="EURUSD",
            period_start_ns=100,
            period_end_ns=10,
            equity_curve=(),
            trades=(),
            metrics=_metrics(),
        )


def test_negative_ts_rejected() -> None:
    with pytest.raises(ValueError, match="ts_ns must be >= 0"):
        BacktestResult(
            ts_ns=-1,
            source="mt5",
            backtest_id="b",
            strategy_id="s",
            symbol="EURUSD",
            period_start_ns=0,
            period_end_ns=0,
            equity_curve=(),
            trades=(),
            metrics=_metrics(),
        )


def test_equity_curve_outside_window() -> None:
    bad = (
        EquityPoint(ts_ns=500, equity_usd=10_000.0),  # before period
        EquityPoint(ts_ns=2_000, equity_usd=10_500.0),
    )
    with pytest.raises(ValueError, match="outside period window"):
        _result(equity=bad)


def test_equity_curve_unsorted() -> None:
    bad = (
        EquityPoint(ts_ns=5_000, equity_usd=10_500.0),
        EquityPoint(ts_ns=2_000, equity_usd=10_400.0),  # earlier ts
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        _result(equity=bad)


def test_trade_outside_window() -> None:
    bad_trades = (_trade(50),)  # before period_start=1000
    with pytest.raises(ValueError, match="trade ts_ns=.*outside period window"):
        _result(trades=bad_trades)


def test_metrics_n_trades_mismatch() -> None:
    trades = (_trade(1_500), _trade(2_500))
    bad = BacktestMetrics(
        n_trades=99,
        win_rate=0.5,
        total_return=0.0,
        max_drawdown=0.0,
    )
    with pytest.raises(ValueError, match="metrics.n_trades.*!="):
        BacktestResult(
            ts_ns=10_000,
            source="quantconnect",
            backtest_id="b",
            strategy_id="s",
            symbol="BTC-USDT",
            period_start_ns=1_000,
            period_end_ns=5_000,
            equity_curve=(),
            trades=trades,
            metrics=bad,
        )


def test_metrics_win_rate_out_of_range() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        BacktestMetrics(
            n_trades=1, win_rate=1.5, total_return=0.0, max_drawdown=0.0
        )


def test_metrics_max_drawdown_out_of_range() -> None:
    with pytest.raises(ValueError, match="max_drawdown"):
        BacktestMetrics(
            n_trades=1, win_rate=0.5, total_return=0.0, max_drawdown=2.0
        )


def test_trade_negative_qty() -> None:
    with pytest.raises(ValueError, match="qty must be"):
        BacktestTrade(
            ts_ns=1,
            symbol="X",
            side=Side.BUY,
            qty=-1.0,
            price=10.0,
            pnl_usd=0.0,
        )


def test_policy_hash_invalid() -> None:
    with pytest.raises(ValueError, match="policy_hash"):
        BacktestResult(
            ts_ns=10_000,
            source="quantconnect",
            backtest_id="b",
            strategy_id="s",
            symbol="BTC-USDT",
            period_start_ns=1_000,
            period_end_ns=5_000,
            equity_curve=(),
            trades=(),
            metrics=_metrics(),
            policy_hash="ZZZZ",  # not lowercase hex / too short
        )


def test_policy_hash_valid_hex_accepted() -> None:
    r = BacktestResult(
        ts_ns=10_000,
        source="quantconnect",
        backtest_id="b",
        strategy_id="s",
        symbol="BTC-USDT",
        period_start_ns=1_000,
        period_end_ns=5_000,
        equity_curve=(),
        trades=(),
        metrics=_metrics(),
        policy_hash="deadbeefcafebabe",
    )
    assert r.policy_hash == "deadbeefcafebabe"


def test_meta_non_str_rejected() -> None:
    with pytest.raises(TypeError, match="meta keys and values"):
        BacktestResult(
            ts_ns=10_000,
            source="mt5",
            backtest_id="b",
            strategy_id="s",
            symbol="EURUSD",
            period_start_ns=1_000,
            period_end_ns=5_000,
            equity_curve=(),
            trades=(),
            metrics=_metrics(),
            meta={"x": 1},  # type: ignore[dict-item]
        )


def test_empty_required_fields_rejected() -> None:
    with pytest.raises(ValueError, match="backtest_id"):
        BacktestResult(
            ts_ns=10,
            source="mt5",
            backtest_id="",
            strategy_id="s",
            symbol="X",
            period_start_ns=0,
            period_end_ns=10,
            equity_curve=(),
            trades=(),
            metrics=_metrics(),
        )


# ---------------------------------------------------------------------------
# 3. Projection to TradeOutcome (the ingestion seam)
# ---------------------------------------------------------------------------


def test_project_emits_one_outcome_per_trade() -> None:
    r = _result(n_trades=3)
    out = project_to_trade_outcomes(r)
    assert len(out) == 3
    assert all(o.status is ExecutionStatus.FILLED for o in out)
    assert all(o.strategy_id == r.strategy_id for o in out)


def test_project_meta_carries_provenance() -> None:
    r = _result(source="quantconnect", trust=SignalTrust.EXTERNAL_MED)
    out = project_to_trade_outcomes(r)
    for o in out:
        assert o.meta["source"] == "quantconnect"
        assert o.meta["backtest_id"] == "bt-001"
        assert o.meta["trust"] == str(SignalTrust.EXTERNAL_MED)
        assert "price" in o.meta
        assert "side" in o.meta


def test_project_carries_policy_hash_when_set() -> None:
    r = BacktestResult(
        ts_ns=10_000,
        source="quantconnect",
        backtest_id="bt-002",
        strategy_id="s",
        symbol="BTC-USDT",
        period_start_ns=1_000,
        period_end_ns=5_000,
        equity_curve=(),
        trades=(_trade(1_500),),
        metrics=_metrics(n=1),
        policy_hash="0123456789abcdef",
    )
    out = project_to_trade_outcomes(r)
    assert out[0].meta["policy_hash"] == "0123456789abcdef"


def test_project_no_trades_returns_empty() -> None:
    r = _result(n_trades=0)
    assert project_to_trade_outcomes(r) == ()


def test_project_is_deterministic() -> None:
    """INV-15 — replay determinism for the projection seam."""
    r1 = _result(n_trades=4)
    r2 = _result(n_trades=4)
    assert project_to_trade_outcomes(r1) == project_to_trade_outcomes(r2)


def test_project_venue_falls_back_to_source() -> None:
    r = _result()
    out = project_to_trade_outcomes(r)
    assert all(o.venue == r.source for o in out)


# ---------------------------------------------------------------------------
# 4. build_equity_curve helper
# ---------------------------------------------------------------------------


def test_build_equity_curve_sorts_ascending() -> None:
    curve = build_equity_curve([(3, 30.0), (1, 10.0), (2, 20.0)])
    assert [p.ts_ns for p in curve] == [1, 2, 3]
    assert [p.equity_usd for p in curve] == [10.0, 20.0, 30.0]


def test_build_equity_curve_empty() -> None:
    assert build_equity_curve([]) == ()


# ---------------------------------------------------------------------------
# 5. BacktestIngester Protocol + error
# ---------------------------------------------------------------------------


class _ConformingIngester:
    source = "tradingview"

    def ingest(self, payload):  # type: ignore[no-untyped-def]
        return _result()


class _NonConformingIngester:
    source = "mt5"

    # missing ingest()


def test_protocol_runtime_checkable_positive() -> None:
    assert isinstance(_ConformingIngester(), BacktestIngester)


def test_protocol_runtime_checkable_negative() -> None:
    assert not isinstance(_NonConformingIngester(), BacktestIngester)


def test_ingestion_error_is_value_error() -> None:
    err = BacktestIngestionError("bad payload")
    assert isinstance(err, ValueError)
    assert str(err) == "bad payload"


# ---------------------------------------------------------------------------
# 6. Default trust + edge cases
# ---------------------------------------------------------------------------


def test_default_trust_is_external_low() -> None:
    r = BacktestResult(
        ts_ns=10_000,
        source="mt5",
        backtest_id="b",
        strategy_id="s",
        symbol="EURUSD",
        period_start_ns=1_000,
        period_end_ns=5_000,
        equity_curve=(),
        trades=(),
        metrics=_metrics(),
    )
    assert r.trust is SignalTrust.EXTERNAL_LOW


def test_zero_period_window_accepted() -> None:
    r = BacktestResult(
        ts_ns=10,
        source="mt5",
        backtest_id="b",
        strategy_id="s",
        symbol="EURUSD",
        period_start_ns=5,
        period_end_ns=5,
        equity_curve=(EquityPoint(ts_ns=5, equity_usd=100.0),),
        trades=(),
        metrics=_metrics(),
    )
    assert r.period_start_ns == r.period_end_ns
