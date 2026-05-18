"""B-11 backtest_scorer test suite — authority + math + INV-15."""

from __future__ import annotations

import ast
import math
import pathlib

import pytest

from core.contracts.backtest_result import (
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
)
from core.contracts.events import Side
from learning_engine.analytics import backtest_scorer as bs
from learning_engine.analytics.backtest_scorer import (
    DAILY_PERIODS_PER_YEAR,
    HOURLY_PERIODS_PER_YEAR,
    MAX_EQUITY_POINTS_PER_BATCH,
    MAX_TRADES_PER_BATCH,
    MINUTE_PERIODS_PER_YEAR,
    NEW_PIP_DEPENDENCIES,
    BacktestScore,
    BacktestScorerError,
    score_backtest,
)

MODULE_PATH = pathlib.Path(bs.__file__)
MODULE_SRC = MODULE_PATH.read_text(encoding="utf-8")
MODULE_AST = ast.parse(MODULE_SRC, filename=str(MODULE_PATH))


# ---------------------------------------------------------------------------
# Authority pins (AST)
# ---------------------------------------------------------------------------


def test_authority_adapted_from_header() -> None:
    assert "# ADAPTED FROM: polakowo/vectorbt" in MODULE_SRC


def test_authority_no_vectorbt_import() -> None:
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "vectorbt" not in alias.name.lower()
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or "vectorbt" not in node.module.lower()


def test_authority_no_runtime_imports() -> None:
    forbidden = {
        "pandas",
        "numpy",
        "polars",
        "torch",
        "scipy",
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "socket",
        "secrets",
        "uuid",
        "requests",
        "httpx",
        "aiohttp",
        "websockets",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                root = node.module.split(".")[0]
                assert root not in forbidden, f"forbidden import: {node.module}"


def test_authority_no_engine_cross_imports() -> None:
    forbidden_prefixes = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
        "intelligence_engine",
    )
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in forbidden_prefixes:
                assert not node.module.startswith(prefix), (
                    f"forbidden engine cross-import: {node.module}"
                )


def test_authority_no_typed_event_construction() -> None:
    forbidden_types = {
        "SignalEvent",
        "ExecutionIntent",
        "HazardEvent",
        "GovernanceDecision",
        "PatchProposal",
        "TradeOutcome",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_types, (
                f"forbidden typed event construction: {node.func.id}"
            )


def test_authority_no_top_level_io() -> None:
    for node in MODULE_AST.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name):
                assert func.id not in {"open", "print", "input", "exec", "eval"}


def test_authority_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_authority_score_is_pure_function() -> None:
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.FunctionDef) and node.name == "score_backtest":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    # No filesystem / network / wall-clock attribute calls.
                    forbidden_attrs = {
                        "open",
                        "read",
                        "write",
                        "request",
                        "get",
                        "post",
                        "now",
                        "time",
                        "monotonic",
                    }
                    if sub.func.attr in forbidden_attrs:
                        # Permit math.fsum and tuple methods. The only
                        # blanket ban is on stdlib IO/clock attributes.
                        attr_target = (
                            sub.func.value.id if isinstance(sub.func.value, ast.Name) else None
                        )
                        assert attr_target not in {
                            "time",
                            "datetime",
                            "os",
                        }, f"forbidden attribute call: {attr_target}.{sub.func.attr}"


# ---------------------------------------------------------------------------
# BacktestScore validation
# ---------------------------------------------------------------------------


def _ok_score() -> BacktestScore:
    return BacktestScore(
        total_return=0.1,
        cagr=0.05,
        volatility=0.2,
        sharpe=0.5,
        sortino=0.7,
        calmar=0.4,
        max_drawdown=0.125,
        win_rate=0.55,
        profit_factor=1.8,
        n_trades=10,
        avg_trade_pnl=12.5,
        best_trade_pnl=50.0,
        worst_trade_pnl=-30.0,
    )


def test_score_ok() -> None:
    s = _ok_score()
    assert s.n_trades == 10


def test_score_rejects_nan_profit_factor() -> None:
    with pytest.raises(BacktestScorerError, match="profit_factor"):
        BacktestScore(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=float("nan"),
            n_trades=0,
            avg_trade_pnl=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
        )


def test_score_allows_inf_profit_factor() -> None:
    s = BacktestScore(
        total_return=0.0,
        cagr=0.0,
        volatility=0.0,
        sharpe=0.0,
        sortino=0.0,
        calmar=0.0,
        max_drawdown=0.0,
        win_rate=1.0,
        profit_factor=math.inf,
        n_trades=1,
        avg_trade_pnl=10.0,
        best_trade_pnl=10.0,
        worst_trade_pnl=10.0,
    )
    assert math.isinf(s.profit_factor)


def test_score_rejects_negative_profit_factor() -> None:
    with pytest.raises(BacktestScorerError, match="profit_factor"):
        BacktestScore(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=-0.1,
            n_trades=0,
            avg_trade_pnl=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
        )


def test_score_rejects_out_of_range_win_rate() -> None:
    with pytest.raises(BacktestScorerError, match="win_rate"):
        BacktestScore(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            win_rate=1.5,
            profit_factor=1.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
        )


def test_score_rejects_out_of_range_max_drawdown() -> None:
    with pytest.raises(BacktestScorerError, match="max_drawdown"):
        BacktestScore(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=1.5,
            win_rate=0.5,
            profit_factor=1.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
        )


def test_score_rejects_negative_volatility() -> None:
    with pytest.raises(BacktestScorerError, match="volatility"):
        BacktestScore(
            total_return=0.0,
            cagr=0.0,
            volatility=-0.1,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            n_trades=0,
            avg_trade_pnl=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
        )


def test_score_rejects_negative_n_trades() -> None:
    with pytest.raises(BacktestScorerError, match="n_trades"):
        BacktestScore(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            n_trades=-1,
            avg_trade_pnl=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
        )


def test_score_is_frozen() -> None:
    s = _ok_score()
    with pytest.raises((AttributeError, Exception)):
        s.total_return = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# score_backtest — orchestration / validation
# ---------------------------------------------------------------------------


def _build_result(
    equity: list[tuple[int, float]],
    trades: list[tuple[int, str, Side, float, float, float]],
) -> BacktestResult:
    """Compact constructor used across the math tests."""

    eq_points = tuple(EquityPoint(ts_ns=ts, equity_usd=v) for ts, v in equity)
    bt_trades = tuple(
        BacktestTrade(
            ts_ns=ts,
            symbol=sym,
            side=side,
            qty=qty,
            price=price,
            pnl_usd=pnl,
            order_id=f"o-{i}",
        )
        for i, (ts, sym, side, qty, price, pnl) in enumerate(trades)
    )
    start = min((p[0] for p in equity), default=0)
    end = max((p[0] for p in equity), default=0)
    return BacktestResult(
        ts_ns=end + 1,
        source="internal_replay",
        backtest_id="bt-1",
        strategy_id="strat-1",
        symbol="BTCUSDT",
        period_start_ns=start,
        period_end_ns=end,
        equity_curve=eq_points,
        trades=bt_trades,
        metrics=BacktestMetrics(
            n_trades=len(bt_trades),
            win_rate=0.5,
            total_return=0.0,
            max_drawdown=0.0,
        ),
    )


def test_score_rejects_non_backtest_result() -> None:
    with pytest.raises(TypeError, match="BacktestResult"):
        score_backtest("not-a-result")  # type: ignore[arg-type]


def test_score_rejects_zero_periods_per_year() -> None:
    r = _build_result([(0, 100.0), (1, 110.0)], [])
    with pytest.raises(BacktestScorerError, match="periods_per_year"):
        score_backtest(r, periods_per_year=0)


def test_score_rejects_non_finite_rfr() -> None:
    r = _build_result([(0, 100.0), (1, 110.0)], [])
    with pytest.raises(BacktestScorerError, match="risk_free"):
        score_backtest(r, risk_free_rate_per_period=float("nan"))


def test_score_handles_empty_trades() -> None:
    r = _build_result([(0, 100.0), (1, 110.0), (2, 121.0)], [])
    s = score_backtest(r)
    assert s.n_trades == 0
    assert s.win_rate == 0.0
    assert s.profit_factor == 0.0
    assert s.avg_trade_pnl == 0.0


def test_score_handles_empty_equity_curve() -> None:
    eq_points: tuple[EquityPoint, ...] = ()
    bt_trades = (
        BacktestTrade(
            ts_ns=10,
            symbol="BTCUSDT",
            side=Side.BUY,
            qty=1.0,
            price=100.0,
            pnl_usd=5.0,
            order_id="x",
        ),
    )
    r = BacktestResult(
        ts_ns=100,
        source="internal_replay",
        backtest_id="bt-1",
        strategy_id="strat-1",
        symbol="BTCUSDT",
        period_start_ns=0,
        period_end_ns=100,
        equity_curve=eq_points,
        trades=bt_trades,
        metrics=BacktestMetrics(
            n_trades=1,
            win_rate=1.0,
            total_return=0.0,
            max_drawdown=0.0,
        ),
    )
    s = score_backtest(r)
    assert s.total_return == 0.0
    assert s.max_drawdown == 0.0
    assert s.sharpe == 0.0
    assert s.n_trades == 1
    assert s.win_rate == 1.0


# ---------------------------------------------------------------------------
# Textbook formula correctness
# ---------------------------------------------------------------------------


def test_total_return_basic() -> None:
    r = _build_result([(0, 100.0), (1, 110.0), (2, 121.0)], [])
    s = score_backtest(r)
    assert math.isclose(s.total_return, 0.21, rel_tol=1e-12)


def test_max_drawdown_basic() -> None:
    # peak 200 -> trough 100 -> recovers to 150
    r = _build_result([(0, 100.0), (1, 200.0), (2, 100.0), (3, 150.0)], [])
    s = score_backtest(r)
    assert math.isclose(s.max_drawdown, 0.5, rel_tol=1e-12)


def test_max_drawdown_zero_on_monotone_rising() -> None:
    r = _build_result([(0, 100.0), (1, 110.0), (2, 121.0)], [])
    s = score_backtest(r)
    assert s.max_drawdown == 0.0


def test_volatility_constant_returns_zero() -> None:
    # 10% per period, exactly — stddev must be zero.
    r = _build_result([(0, 100.0), (1, 110.0), (2, 121.0), (3, 133.1)], [])
    s = score_backtest(r)
    assert math.isclose(s.volatility, 0.0, abs_tol=1e-9)


def test_sharpe_constant_returns_zero_volatility_yields_zero() -> None:
    r = _build_result([(0, 100.0), (1, 110.0), (2, 121.0), (3, 133.1)], [])
    s = score_backtest(r)
    # Convention: zero volatility -> zero Sharpe (avoid div by zero).
    assert s.sharpe == 0.0
    assert s.sortino == 0.0


def test_sortino_no_downside_returns_zero() -> None:
    # Strictly rising — no downside deviations.
    r = _build_result([(0, 100.0), (1, 105.0), (2, 110.0)], [])
    s = score_backtest(r)
    assert s.sortino == 0.0


def test_sharpe_sign_matches_excess_mean() -> None:
    r = _build_result([(0, 100.0), (1, 90.0), (2, 95.0), (3, 80.0)], [])
    s = score_backtest(r)
    assert s.sharpe < 0.0  # negative excess mean -> negative Sharpe


def test_win_rate_basic() -> None:
    trades = [
        (10, "BTCUSDT", Side.BUY, 1.0, 100.0, 10.0),
        (20, "BTCUSDT", Side.SELL, 1.0, 100.0, -5.0),
        (30, "BTCUSDT", Side.BUY, 1.0, 100.0, 3.0),
        (40, "BTCUSDT", Side.SELL, 1.0, 100.0, 0.0),  # break-even loses
    ]
    r = _build_result([(0, 100.0), (50, 108.0)], trades)
    s = score_backtest(r)
    assert s.n_trades == 4
    assert math.isclose(s.win_rate, 0.5, rel_tol=1e-12)


def test_profit_factor_basic() -> None:
    trades = [
        (10, "X", Side.BUY, 1.0, 1.0, 10.0),
        (20, "X", Side.BUY, 1.0, 1.0, 20.0),
        (30, "X", Side.SELL, 1.0, 1.0, -5.0),
        (40, "X", Side.SELL, 1.0, 1.0, -5.0),
    ]
    r = _build_result([(0, 100.0), (50, 120.0)], trades)
    s = score_backtest(r)
    assert math.isclose(s.profit_factor, 3.0, rel_tol=1e-12)


def test_profit_factor_inf_on_only_wins() -> None:
    trades = [
        (10, "X", Side.BUY, 1.0, 1.0, 5.0),
        (20, "X", Side.BUY, 1.0, 1.0, 7.0),
    ]
    r = _build_result([(0, 100.0), (50, 112.0)], trades)
    s = score_backtest(r)
    assert math.isinf(s.profit_factor)


def test_profit_factor_zero_on_only_losses_and_zeros() -> None:
    trades = [
        (10, "X", Side.BUY, 1.0, 1.0, -5.0),
        (20, "X", Side.BUY, 1.0, 1.0, 0.0),
    ]
    r = _build_result([(0, 100.0), (50, 95.0)], trades)
    s = score_backtest(r)
    assert s.profit_factor == 0.0


def test_avg_best_worst_trade() -> None:
    trades = [
        (10, "X", Side.BUY, 1.0, 1.0, 10.0),
        (20, "X", Side.BUY, 1.0, 1.0, -20.0),
        (30, "X", Side.BUY, 1.0, 1.0, 5.0),
    ]
    r = _build_result([(0, 100.0), (50, 95.0)], trades)
    s = score_backtest(r)
    assert math.isclose(s.avg_trade_pnl, -5.0 / 3.0, rel_tol=1e-12)
    assert s.best_trade_pnl == 10.0
    assert s.worst_trade_pnl == -20.0


def test_cagr_one_year() -> None:
    # 252 daily periods, doubled equity -> CAGR = 100%
    eq = [(i, 100.0 * (2.0 ** (i / 252))) for i in range(253)]
    r = _build_result(eq, [])
    s = score_backtest(r)
    assert math.isclose(s.cagr, 1.0, rel_tol=1e-6)


def test_calmar_uses_cagr_and_drawdown() -> None:
    eq = [(0, 100.0), (1, 200.0), (2, 100.0), (3, 200.0)]
    r = _build_result(eq, [])
    s = score_backtest(r)
    assert s.max_drawdown > 0.0
    assert math.isclose(s.calmar, s.cagr / s.max_drawdown, rel_tol=1e-12)


def test_calmar_zero_when_no_drawdown() -> None:
    r = _build_result([(0, 100.0), (1, 110.0), (2, 121.0)], [])
    s = score_backtest(r)
    assert s.max_drawdown == 0.0
    assert s.calmar == 0.0


def test_score_rejects_non_positive_equity_value() -> None:
    eq_points = (
        EquityPoint(ts_ns=0, equity_usd=100.0),
        EquityPoint(ts_ns=1, equity_usd=0.0),
    )
    r = BacktestResult(
        ts_ns=10,
        source="internal_replay",
        backtest_id="bt-1",
        strategy_id="strat-1",
        symbol="X",
        period_start_ns=0,
        period_end_ns=10,
        equity_curve=eq_points,
        trades=(),
        metrics=BacktestMetrics(n_trades=0, win_rate=0.0, total_return=0.0, max_drawdown=0.0),
    )
    with pytest.raises(BacktestScorerError, match="strictly positive"):
        score_backtest(r)


def test_periods_per_year_constants_sane() -> None:
    assert DAILY_PERIODS_PER_YEAR == 252
    assert HOURLY_PERIODS_PER_YEAR == 252 * 24
    assert MINUTE_PERIODS_PER_YEAR == 252 * 24 * 60
    assert MAX_TRADES_PER_BATCH > 0
    assert MAX_EQUITY_POINTS_PER_BATCH > 0


def test_rfr_shifts_sharpe_down() -> None:
    eq = [(0, 100.0), (1, 110.0), (2, 121.0), (3, 133.1)]
    r = _build_result(eq, [])
    s0 = score_backtest(r, risk_free_rate_per_period=0.0)
    s_high = score_backtest(r, risk_free_rate_per_period=0.5)
    # Returns are constant so Sharpe is 0 either way; but Sortino /
    # excess mean must shift.
    assert s0.sharpe == 0.0
    assert s_high.sharpe == 0.0
    # With rfr > period return, excess mean goes negative -> Sortino < 0.
    assert s_high.sortino <= 0.0


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def test_replay_three_runs_identical() -> None:
    trades = [
        (10, "X", Side.BUY, 1.0, 1.0, 10.0),
        (20, "X", Side.BUY, 1.0, 1.0, -5.0),
        (30, "X", Side.BUY, 1.0, 1.0, 7.5),
    ]
    r = _build_result([(0, 100.0), (40, 112.5)], trades)
    s1 = score_backtest(r)
    s2 = score_backtest(r)
    s3 = score_backtest(r)
    assert s1 == s2 == s3


def test_replay_trade_order_independent() -> None:
    trades_a = [
        (10, "X", Side.BUY, 1.0, 1.0, 10.0),
        (20, "X", Side.BUY, 1.0, 1.0, -5.0),
        (30, "X", Side.BUY, 1.0, 1.0, 7.5),
    ]
    trades_b = list(reversed(trades_a))
    r_a = _build_result([(0, 100.0), (40, 112.5)], trades_a)
    r_b = _build_result([(0, 100.0), (40, 112.5)], trades_b)
    # Sorted-key projection inside the scorer must collapse both orders.
    s_a = score_backtest(r_a)
    s_b = score_backtest(r_b)
    assert s_a.win_rate == s_b.win_rate
    assert s_a.profit_factor == s_b.profit_factor
    assert s_a.avg_trade_pnl == s_b.avg_trade_pnl
    assert s_a.best_trade_pnl == s_b.best_trade_pnl
    assert s_a.worst_trade_pnl == s_b.worst_trade_pnl


def test_replay_module_reimport_clean() -> None:
    import importlib

    fresh = importlib.import_module("learning_engine.analytics.backtest_scorer")
    assert fresh is bs
    assert fresh.NEW_PIP_DEPENDENCIES == ()
