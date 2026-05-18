"""Tests for B-08 backtrader-style reference backtester.

Coverage:
  - AST authority pins (no backtrader / numpy / pandas / clock / random imports;
    no PatchProposal / SignalEvent / GovernanceDecision constructors;
    no engine cross-imports; ADAPTED FROM header present;
    NEW_PIP_DEPENDENCIES = ()).
  - Bar / OrderRequest / BrokerConfig / BacktestConfig validation.
  - run_backtest correctness: HOLD-only flat equity, single round-trip P&L,
    commission, deterministic slippage jitter, equity-curve monotonic ts.
  - Output is a fully-validated BacktestResult.
  - INV-15 byte-identical replay across 3 runs.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from core.contracts.backtest_result import BacktestResult
from core.contracts.events import Side
from core.contracts.signal_trust import SignalTrust
from simulation.backtester import (
    NEW_PIP_DEPENDENCIES,
    BacktestConfig,
    BacktesterError,
    Bar,
    BrokerConfig,
    OrderAction,
    OrderRequest,
    StrategyContext,
    run_backtest,
)

MODULE = Path("simulation/backtester.py")
SOURCE_TEXT = MODULE.read_text(encoding="utf-8")
SOURCE_AST = ast.parse(SOURCE_TEXT)


# ---------------------------------------------------------------------------
# Strategy fixtures
# ---------------------------------------------------------------------------
class _HoldOnly:
    def next(self, ctx: StrategyContext) -> OrderRequest:
        return OrderRequest(action=OrderAction.HOLD, qty=0.0)


class _BuyOnceSellOnce:
    """Buy 1 unit on bar 0, sell 1 unit on bar 2."""

    def next(self, ctx: StrategyContext) -> OrderRequest:
        if ctx.bar_index == 0:
            return OrderRequest(action=OrderAction.BUY, qty=1.0)
        if ctx.bar_index == 2:
            return OrderRequest(action=OrderAction.SELL, qty=1.0)
        return OrderRequest(action=OrderAction.HOLD, qty=0.0)


class _BadReturn:
    def next(self, ctx: StrategyContext) -> OrderRequest:  # type: ignore[override]
        return "not an order request"  # type: ignore[return-value]


def _bars(
    n: int = 5,
    start_ns: int = 1_000_000_000,
    step_ns: int = 60_000_000_000,
) -> tuple[Bar, ...]:
    out: list[Bar] = []
    for i in range(n):
        price = 100.0 + i
        out.append(
            Bar(
                ts_ns=start_ns + i * step_ns,
                symbol="BTCUSDT",
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=10.0,
            )
        )
    return tuple(out)


def _cfg(seed: int = 0, **kw) -> BacktestConfig:
    return BacktestConfig(
        backtest_id=kw.pop("backtest_id", "bt-001"),
        strategy_id=kw.pop("strategy_id", "buy-once"),
        symbol=kw.pop("symbol", "BTCUSDT"),
        broker=kw.pop("broker", BrokerConfig(initial_cash_usd=10_000.0)),
        seed=seed,
        history_window=kw.pop("history_window", 0),
        meta=kw.pop("meta", {}),
    )


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------
class TestAuthorityPins:
    def test_adapted_from_header_present(self) -> None:
        assert SOURCE_TEXT.startswith("# ADAPTED FROM: backtrader"), (
            "module must declare GPL mitigation header"
        )

    def test_no_banned_imports(self) -> None:
        banned = {
            "backtrader",
            "numpy",
            "pandas",
            "polars",
            "torch",
            "scipy",
            "random",
            "time",
            "datetime",
            "asyncio",
            "os",
            "system.time_source",
            "websockets",
            "langsmith",
        }
        for node in ast.walk(SOURCE_AST):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned, f"banned import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root = node.module.split(".")[0]
                assert root not in banned, f"banned import: {node.module}"
                assert node.module not in banned

    def test_no_engine_cross_imports(self) -> None:
        banned_prefixes = (
            "execution_engine",
            "governance_engine",
            "system_engine",
            "evolution_engine",
            "intelligence_engine",
        )
        for node in ast.walk(SOURCE_AST):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in banned_prefixes:
                    assert not node.module.startswith(prefix), f"engine cross-import: {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in banned_prefixes:
                        assert not alias.name.startswith(prefix), (
                            f"engine cross-import: {alias.name}"
                        )

    def test_no_typed_event_constructors(self) -> None:
        banned_ctors = {"PatchProposal", "SignalEvent", "GovernanceDecision"}
        for node in ast.walk(SOURCE_AST):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in banned_ctors, f"banned constructor: {node.func.id}"

    def test_pip_dependencies_empty(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ()

    def test_no_top_level_io(self) -> None:
        for node in ast.iter_child_nodes(SOURCE_AST):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                pytest.fail(f"top-level call: {ast.dump(node.value)}")


# ---------------------------------------------------------------------------
# Bar validation
# ---------------------------------------------------------------------------
class TestBarValidation:
    def test_valid_bar(self) -> None:
        Bar(ts_ns=1, symbol="BTCUSDT", open=100.0, high=101.0, low=99.0, close=100.5, volume=5.0)

    def test_negative_ts(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=-1, symbol="BTCUSDT", open=1.0, high=1.0, low=1.0, close=1.0, volume=0.0)

    def test_empty_symbol(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=1, symbol="", open=1.0, high=1.0, low=1.0, close=1.0, volume=0.0)

    def test_high_lt_low(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=1, symbol="X", open=1.0, high=0.5, low=1.0, close=0.5, volume=0.0)

    def test_open_outside_window(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=1, symbol="X", open=10.0, high=2.0, low=1.0, close=1.5, volume=0.0)

    def test_close_outside_window(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=1, symbol="X", open=1.5, high=2.0, low=1.0, close=10.0, volume=0.0)

    def test_negative_volume(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=1, symbol="X", open=1.0, high=1.0, low=1.0, close=1.0, volume=-1.0)

    def test_nan_open(self) -> None:
        with pytest.raises(BacktesterError):
            Bar(ts_ns=1, symbol="X", open=float("nan"), high=1.0, low=1.0, close=1.0, volume=0.0)

    def test_frozen(self) -> None:
        bar = Bar(ts_ns=1, symbol="X", open=1.0, high=1.0, low=1.0, close=1.0, volume=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            bar.open = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OrderRequest validation
# ---------------------------------------------------------------------------
class TestOrderRequestValidation:
    def test_buy(self) -> None:
        OrderRequest(action=OrderAction.BUY, qty=1.0)

    def test_sell(self) -> None:
        OrderRequest(action=OrderAction.SELL, qty=1.0)

    def test_hold_zero_qty(self) -> None:
        OrderRequest(action=OrderAction.HOLD, qty=0.0)

    def test_hold_must_have_zero_qty(self) -> None:
        with pytest.raises(BacktesterError):
            OrderRequest(action=OrderAction.HOLD, qty=1.0)

    def test_buy_zero_qty_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            OrderRequest(action=OrderAction.BUY, qty=0.0)

    def test_negative_qty_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            OrderRequest(action=OrderAction.BUY, qty=-1.0)


# ---------------------------------------------------------------------------
# BrokerConfig validation
# ---------------------------------------------------------------------------
class TestBrokerConfig:
    def test_defaults(self) -> None:
        BrokerConfig()

    def test_zero_cash_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            BrokerConfig(initial_cash_usd=0.0)

    def test_negative_cash_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            BrokerConfig(initial_cash_usd=-1.0)

    def test_excessive_commission_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            BrokerConfig(commission_rate=0.5)

    def test_negative_commission_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            BrokerConfig(commission_rate=-0.001)

    def test_excessive_slippage_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            BrokerConfig(slippage_perc=0.5)


# ---------------------------------------------------------------------------
# BacktestConfig validation
# ---------------------------------------------------------------------------
class TestBacktestConfigValidation:
    def test_minimal(self) -> None:
        _cfg()

    def test_empty_backtest_id_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            _cfg(backtest_id="")

    def test_empty_strategy_id_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            _cfg(strategy_id="")

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            _cfg(symbol="")

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            _cfg(seed=-1)

    def test_history_window_too_large(self) -> None:
        with pytest.raises(BacktesterError):
            _cfg(history_window=10_000)

    def test_meta_non_string_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            _cfg(meta={"k": 1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_backtest behaviour
# ---------------------------------------------------------------------------
class TestRunBacktest:
    def test_hold_only_flat_equity(self) -> None:
        bars = _bars(5)
        cfg = _cfg()
        result = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=cfg,
            result_ts_ns=bars[-1].ts_ns,
        )
        assert isinstance(result, BacktestResult)
        assert result.metrics.n_trades == 0
        assert result.trades == ()
        assert result.metrics.total_return == 0.0
        assert result.metrics.max_drawdown == 0.0
        for pt in result.equity_curve:
            assert pt.equity_usd == cfg.broker.initial_cash_usd

    def test_buy_then_sell_records_trades(self) -> None:
        bars = _bars(5)
        cfg = _cfg(strategy_id="buy-once")
        result = run_backtest(
            bars=bars,
            strategy=_BuyOnceSellOnce(),
            config=cfg,
            result_ts_ns=bars[-1].ts_ns,
        )
        assert result.metrics.n_trades == 2
        assert result.trades[0].side == Side.BUY
        assert result.trades[1].side == Side.SELL
        # No slippage / commission → P&L on sell == bar2-fill - bar0-fill.
        # bar0.close=100.0 → buy fills at bar1.open=101.0
        # bar2.close=102.0 → sell fills at bar3.open=103.0
        assert result.trades[0].price == 101.0
        assert result.trades[1].price == 103.0
        assert result.trades[1].pnl_usd == pytest.approx(2.0)

    def test_commission_is_charged(self) -> None:
        bars = _bars(5)
        broker = BrokerConfig(initial_cash_usd=10_000.0, commission_rate=0.001)
        cfg = _cfg(broker=broker)
        result = run_backtest(
            bars=bars,
            strategy=_BuyOnceSellOnce(),
            config=cfg,
            result_ts_ns=bars[-1].ts_ns,
        )
        # Each fill carries 1 unit * fill_price * 0.001 commission.
        for trade in result.trades:
            expected_fee = trade.price * trade.qty * 0.001
            assert trade.fee_usd == pytest.approx(expected_fee)

    def test_slippage_is_deterministic(self) -> None:
        bars = _bars(5)
        broker = BrokerConfig(initial_cash_usd=10_000.0, slippage_perc=0.005)
        cfg = _cfg(broker=broker, seed=42)
        r1 = run_backtest(
            bars=bars,
            strategy=_BuyOnceSellOnce(),
            config=cfg,
            result_ts_ns=bars[-1].ts_ns,
        )
        r2 = run_backtest(
            bars=bars,
            strategy=_BuyOnceSellOnce(),
            config=cfg,
            result_ts_ns=bars[-1].ts_ns,
        )
        assert r1.trades == r2.trades

    def test_slippage_diverges_across_seeds(self) -> None:
        bars = _bars(5)
        broker = BrokerConfig(initial_cash_usd=10_000.0, slippage_perc=0.005)
        a = run_backtest(
            bars=bars,
            strategy=_BuyOnceSellOnce(),
            config=_cfg(broker=broker, seed=1),
            result_ts_ns=bars[-1].ts_ns,
        )
        b = run_backtest(
            bars=bars,
            strategy=_BuyOnceSellOnce(),
            config=_cfg(broker=broker, seed=2),
            result_ts_ns=bars[-1].ts_ns,
        )
        # Buys add slippage upward, sells push downward — differing seeds
        # produce different fill prices.
        assert a.trades[0].price != b.trades[0].price

    def test_equity_curve_monotonic_ts(self) -> None:
        bars = _bars(5)
        result = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=_cfg(),
            result_ts_ns=bars[-1].ts_ns,
        )
        assert len(result.equity_curve) == 5
        ts_seq = [pt.ts_ns for pt in result.equity_curve]
        assert ts_seq == sorted(ts_seq)

    def test_history_window_exposed_to_strategy(self) -> None:
        bars = _bars(4)
        observed: list[int] = []

        class S:
            def next(self, ctx: StrategyContext) -> OrderRequest:
                observed.append(len(ctx.history))
                return OrderRequest(action=OrderAction.HOLD, qty=0.0)

        run_backtest(
            bars=bars,
            strategy=S(),
            config=_cfg(history_window=2),
            result_ts_ns=bars[-1].ts_ns,
        )
        assert observed == [0, 1, 2, 2]

    def test_empty_bars_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            run_backtest(
                bars=(),
                strategy=_HoldOnly(),
                config=_cfg(),
                result_ts_ns=1,
            )

    def test_non_monotonic_bars_rejected(self) -> None:
        a = Bar(
            ts_ns=2_000_000_000,
            symbol="BTCUSDT",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=0.0,
        )
        b = Bar(
            ts_ns=1_000_000_000,
            symbol="BTCUSDT",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=0.0,
        )
        with pytest.raises(BacktesterError):
            run_backtest(
                bars=(a, b),
                strategy=_HoldOnly(),
                config=_cfg(),
                result_ts_ns=2_000_000_000,
            )

    def test_symbol_mismatch_rejected(self) -> None:
        bar = Bar(ts_ns=1, symbol="ETHUSDT", open=1.0, high=1.0, low=1.0, close=1.0, volume=0.0)
        with pytest.raises(BacktesterError):
            run_backtest(
                bars=(bar,),
                strategy=_HoldOnly(),
                config=_cfg(),
                result_ts_ns=1,
            )

    def test_strategy_protocol_required(self) -> None:
        with pytest.raises(BacktesterError):
            run_backtest(
                bars=_bars(3),
                strategy="not a strategy",  # type: ignore[arg-type]
                config=_cfg(),
                result_ts_ns=10,
            )

    def test_strategy_must_return_order_request(self) -> None:
        with pytest.raises(BacktesterError):
            run_backtest(
                bars=_bars(3),
                strategy=_BadReturn(),
                config=_cfg(),
                result_ts_ns=10,
            )

    def test_negative_result_ts_rejected(self) -> None:
        with pytest.raises(BacktesterError):
            run_backtest(
                bars=_bars(3),
                strategy=_HoldOnly(),
                config=_cfg(),
                result_ts_ns=-1,
            )

    def test_result_trust_is_external_low(self) -> None:
        bars = _bars(3)
        result = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=_cfg(),
            result_ts_ns=bars[-1].ts_ns,
        )
        assert result.trust is SignalTrust.EXTERNAL_LOW

    def test_policy_hash_is_blake2b_16(self) -> None:
        bars = _bars(3)
        result = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=_cfg(),
            result_ts_ns=bars[-1].ts_ns,
        )
        assert re.fullmatch(r"[a-f0-9]{32}", result.policy_hash)

    def test_meta_includes_seed_and_window(self) -> None:
        bars = _bars(3)
        result = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=_cfg(seed=7, history_window=2, meta={"note": "x"}),
            result_ts_ns=bars[-1].ts_ns,
        )
        assert result.meta["seed"] == "7"
        assert result.meta["history_window"] == "2"
        assert result.meta["note"] == "x"


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------
class TestInv15Replay:
    def test_three_runs_identical(self) -> None:
        bars = _bars(8)
        cfg = _cfg(
            broker=BrokerConfig(
                initial_cash_usd=10_000.0,
                commission_rate=0.001,
                slippage_perc=0.002,
            ),
            seed=99,
        )
        runs = [
            run_backtest(
                bars=bars,
                strategy=_BuyOnceSellOnce(),
                config=cfg,
                result_ts_ns=bars[-1].ts_ns,
            )
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]

    def test_different_strategy_id_changes_policy_hash(self) -> None:
        bars = _bars(3)
        a = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=_cfg(strategy_id="strat-a"),
            result_ts_ns=bars[-1].ts_ns,
        )
        b = run_backtest(
            bars=bars,
            strategy=_HoldOnly(),
            config=_cfg(strategy_id="strat-b"),
            result_ts_ns=bars[-1].ts_ns,
        )
        assert a.policy_hash != b.policy_hash
