"""Tests for ``learning_engine.analytics.regime_stats`` (S-10.2 polars).

Covers:

* Module metadata (ADAPTED-FROM header, ``NEW_PIP_DEPENDENCIES``).
* Lazy-import contract — polars is imported inside
  :func:`compute_regime_stats`, never at module top-level.
* No clock / no engine cross-imports / no global mutable state.
* Frozen+slotted dataclass validation
  (:class:`RegimeTradeRow`, :class:`RegimeStats`, :class:`RegimeStatsReport`).
* Functional aggregation correctness against hand-rolled reference.
* INV-15 byte-identical replay (3-run + permutation invariance).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import io
import sys
import tokenize
from pathlib import Path

import pytest

from core.contracts.macro_regime import MacroRegime
from learning_engine.analytics import regime_stats as rs
from learning_engine.analytics.regime_stats import (
    NEW_PIP_DEPENDENCIES,
    RegimeStats,
    RegimeStatsReport,
    RegimeTradeRow,
    compute_regime_stats,
)

# Polars is required to actually exercise compute_regime_stats(). Skip the
# whole module when it is not installed (matches S-10.1 pnl_attribution
# pattern). Module-level metadata + dataclass-validation tests still
# require polars only at call time, but it is simpler to gate the file.
pl = pytest.importorskip("polars")  # noqa: F841

MODULE_PATH = Path(rs.__file__)


# ----------------------------------------------------------------------
# Module metadata
# ----------------------------------------------------------------------


def test_adapted_from_header_present() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    first_lines = src.splitlines()[:6]
    joined = "\n".join(first_lines)
    assert "ADAPTED FROM:" in joined
    assert "polars" in joined.lower()


def test_new_pip_dependencies_declares_polars() -> None:
    assert NEW_PIP_DEPENDENCIES == ("polars",)


def test_module_has_no_forbidden_top_level_imports() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {
        "polars",
        "numpy",
        "pandas",
        "torch",
        "datetime",
        "time",
        "ccxt",
        "river",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden, (
                    f"top-level import of {alias.name} forbidden in OFFLINE module"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".", 1)[0]
            assert mod not in forbidden, f"top-level from-import of {node.module} forbidden"


def test_module_has_no_clock_calls() -> None:
    """No ``time.time()`` / ``datetime.now()`` anywhere in the source.

    Uses tokenize so docstrings and string contents do not contribute
    false positives.
    """
    src = MODULE_PATH.read_text(encoding="utf-8")
    name_tokens = [
        t.string
        for t in tokenize.generate_tokens(io.StringIO(src).readline)
        if t.type == tokenize.NAME
    ]
    joined = " ".join(name_tokens)
    assert "datetime now" not in joined
    assert "time time" not in joined
    assert "time monotonic" not in joined
    assert "time perf_counter" not in joined


def test_module_has_no_engine_cross_imports() -> None:
    """No imports from runtime/hot-path tiers (AST-only — docstring mentions OK)."""
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden_roots, f"OFFLINE module must not import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            assert root not in forbidden_roots, f"OFFLINE module must not import from {mod}"
            if mod.startswith("intelligence_engine.meta_controller.hot_path"):
                raise AssertionError(f"OFFLINE module must not import from {mod}")


def test_module_has_no_random_or_prng() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or (node.names[0].name if node.names else "")
            mod_root = (mod or "").split(".", 1)[0]
            assert mod_root != "random", "no PRNG in deterministic OFFLINE module"


# ----------------------------------------------------------------------
# Lazy-import contract
# ----------------------------------------------------------------------


def test_polars_lazy_import_lives_inside_compute_regime_stats() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func_imports: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            inner = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        inner.append(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    inner.append(sub.module or "")
            func_imports[node.name] = inner

    assert "polars" in func_imports.get("compute_regime_stats", []), (
        "polars must be lazy-imported inside compute_regime_stats"
    )


def test_module_globals_do_not_leak_polars() -> None:
    assert "polars" not in vars(rs), "polars must not leak into module globals after lazy import"
    assert "pl" not in vars(rs)


def test_module_imports_without_polars_in_sys_modules() -> None:
    """Reimporting the module should not pull polars into sys.modules itself.

    Pins the lazy-import contract: nothing at toplevel should crash
    when polars is unavailable. Mirrors the S-10.1 pnl_attribution pattern.
    """
    saved = {k: v for k, v in sys.modules.items() if k.startswith("polars")}
    for k in list(sys.modules):
        if k.startswith("polars"):
            del sys.modules[k]
    if "learning_engine.analytics.regime_stats" in sys.modules:
        del sys.modules["learning_engine.analytics.regime_stats"]
    try:
        mod = importlib.import_module("learning_engine.analytics.regime_stats")
        assert mod.NEW_PIP_DEPENDENCIES == ("polars",)
        assert "polars" not in sys.modules, (
            "module import pulled polars in despite lazy-import contract"
        )
    finally:
        sys.modules.update(saved)
        importlib.import_module("learning_engine.analytics.regime_stats")


# ----------------------------------------------------------------------
# RegimeTradeRow validation
# ----------------------------------------------------------------------


def test_regime_trade_row_is_frozen_and_slotted() -> None:
    assert dataclasses.is_dataclass(RegimeTradeRow)
    spec = dataclasses.fields(RegimeTradeRow)
    assert {f.name for f in spec} == {
        "ts_ns",
        "symbol",
        "regime",
        "pnl_usd",
        "fee_usd",
        "qty",
        "fill_price",
    }
    row = _row(regime=MacroRegime.RISK_ON, pnl_usd=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.pnl_usd = 0.0  # type: ignore[misc]
    assert not hasattr(row, "__dict__")  # slots=True


def test_regime_trade_row_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be >= 0"):
        _row(ts_ns=-1)


def test_regime_trade_row_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        RegimeTradeRow(
            ts_ns=True,  # type: ignore[arg-type]
            symbol="BTCUSDT",
            regime=MacroRegime.NEUTRAL,
            pnl_usd=0.0,
            fee_usd=0.0,
            qty=1.0,
            fill_price=100.0,
        )


def test_regime_trade_row_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        _row(symbol="")


def test_regime_trade_row_rejects_non_macro_regime() -> None:
    with pytest.raises(TypeError, match="regime must be MacroRegime"):
        RegimeTradeRow(
            ts_ns=0,
            symbol="BTCUSDT",
            regime="RISK_ON",  # type: ignore[arg-type]
            pnl_usd=0.0,
            fee_usd=0.0,
            qty=1.0,
            fill_price=100.0,
        )


def test_regime_trade_row_rejects_negative_fee() -> None:
    with pytest.raises(ValueError, match="fee_usd"):
        _row(fee_usd=-0.01)


def test_regime_trade_row_rejects_negative_qty() -> None:
    with pytest.raises(ValueError, match="qty"):
        _row(qty=-0.01)


def test_regime_trade_row_rejects_zero_fill_price() -> None:
    with pytest.raises(ValueError, match="fill_price"):
        _row(fill_price=0.0)


def test_regime_trade_row_rejects_negative_fill_price() -> None:
    with pytest.raises(ValueError, match="fill_price"):
        _row(fill_price=-1.0)


def test_regime_trade_row_rejects_nan_fields() -> None:
    nan = float("nan")
    with pytest.raises(ValueError, match="must not be NaN"):
        _row(pnl_usd=nan)
    with pytest.raises(ValueError, match="must not be NaN"):
        _row(qty=nan)


def test_regime_trade_row_rejects_bool_qty() -> None:
    with pytest.raises(TypeError, match="qty must be float"):
        RegimeTradeRow(
            ts_ns=0,
            symbol="BTCUSDT",
            regime=MacroRegime.NEUTRAL,
            pnl_usd=0.0,
            fee_usd=0.0,
            qty=True,  # type: ignore[arg-type]
            fill_price=100.0,
        )


# ----------------------------------------------------------------------
# RegimeStats validation
# ----------------------------------------------------------------------


def test_regime_stats_rejects_winners_exceeding_trades() -> None:
    with pytest.raises(ValueError, match="n_winners"):
        RegimeStats(
            regime=MacroRegime.NEUTRAL,
            n_trades=2,
            n_winners=3,
            win_rate=1.5,
            total_pnl_usd=0.0,
            avg_pnl_usd=0.0,
            pnl_std=0.0,
            total_fee_usd=0.0,
            total_notional_usd=0.0,
        )


def test_regime_stats_rejects_win_rate_out_of_unit() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        RegimeStats(
            regime=MacroRegime.NEUTRAL,
            n_trades=2,
            n_winners=1,
            win_rate=1.5,
            total_pnl_usd=0.0,
            avg_pnl_usd=0.0,
            pnl_std=0.0,
            total_fee_usd=0.0,
            total_notional_usd=0.0,
        )


def test_regime_stats_rejects_negative_pnl_std() -> None:
    with pytest.raises(ValueError, match="pnl_std"):
        RegimeStats(
            regime=MacroRegime.NEUTRAL,
            n_trades=1,
            n_winners=0,
            win_rate=0.0,
            total_pnl_usd=0.0,
            avg_pnl_usd=0.0,
            pnl_std=-0.1,
            total_fee_usd=0.0,
            total_notional_usd=0.0,
        )


def test_regime_stats_rejects_negative_total_fee() -> None:
    with pytest.raises(ValueError, match="total_fee_usd"):
        RegimeStats(
            regime=MacroRegime.NEUTRAL,
            n_trades=1,
            n_winners=0,
            win_rate=0.0,
            total_pnl_usd=0.0,
            avg_pnl_usd=0.0,
            pnl_std=0.0,
            total_fee_usd=-1.0,
            total_notional_usd=0.0,
        )


# ----------------------------------------------------------------------
# RegimeStatsReport validation
# ----------------------------------------------------------------------


def test_regime_stats_report_rejects_unsorted_by_regime() -> None:
    a = _stats(MacroRegime.RISK_ON)
    b = _stats(MacroRegime.NEUTRAL)
    with pytest.raises(ValueError, match="sorted"):
        RegimeStatsReport(
            by_regime=(a, b),  # NEUTRAL < RISK_ON alphabetically
            total_n_trades=2,
            overall_win_rate=0.5,
            overall_total_pnl_usd=0.0,
            overall_total_fee_usd=0.0,
            overall_total_notional_usd=0.0,
        )


def test_regime_stats_report_rejects_duplicate_regimes() -> None:
    a = _stats(MacroRegime.NEUTRAL)
    with pytest.raises(ValueError, match="unique"):
        RegimeStatsReport(
            by_regime=(a, a),
            total_n_trades=2,
            overall_win_rate=0.5,
            overall_total_pnl_usd=0.0,
            overall_total_fee_usd=0.0,
            overall_total_notional_usd=0.0,
        )


def test_regime_stats_report_rejects_non_tuple_by_regime() -> None:
    a = _stats(MacroRegime.NEUTRAL)
    with pytest.raises(TypeError, match="by_regime must be tuple"):
        RegimeStatsReport(
            by_regime=[a],  # type: ignore[arg-type]
            total_n_trades=1,
            overall_win_rate=0.0,
            overall_total_pnl_usd=0.0,
            overall_total_fee_usd=0.0,
            overall_total_notional_usd=0.0,
        )


def test_regime_stats_report_rejects_non_regime_stats_entries() -> None:
    with pytest.raises(TypeError, match="RegimeStats"):
        RegimeStatsReport(
            by_regime=("oops",),  # type: ignore[arg-type]
            total_n_trades=0,
            overall_win_rate=0.0,
            overall_total_pnl_usd=0.0,
            overall_total_fee_usd=0.0,
            overall_total_notional_usd=0.0,
        )


def test_regime_stats_report_rejects_overall_win_rate_out_of_unit() -> None:
    with pytest.raises(ValueError, match="overall_win_rate"):
        RegimeStatsReport(
            by_regime=(),
            total_n_trades=0,
            overall_win_rate=2.0,
            overall_total_pnl_usd=0.0,
            overall_total_fee_usd=0.0,
            overall_total_notional_usd=0.0,
        )


# ----------------------------------------------------------------------
# Functional aggregation
# ----------------------------------------------------------------------


def test_compute_regime_stats_empty_input() -> None:
    report = compute_regime_stats([])
    assert report.by_regime == ()
    assert report.total_n_trades == 0
    assert report.overall_win_rate == 0.0
    assert report.overall_total_pnl_usd == 0.0


def test_compute_regime_stats_single_winner() -> None:
    rows = [
        _row(
            ts_ns=1,
            symbol="BTCUSDT",
            regime=MacroRegime.RISK_ON,
            pnl_usd=42.0,
            fee_usd=0.5,
            qty=2.0,
            fill_price=100.0,
        )
    ]
    report = compute_regime_stats(rows)
    assert len(report.by_regime) == 1
    s = report.by_regime[0]
    assert s.regime is MacroRegime.RISK_ON
    assert s.n_trades == 1
    assert s.n_winners == 1
    assert s.win_rate == 1.0
    assert s.total_pnl_usd == pytest.approx(42.0)
    assert s.avg_pnl_usd == pytest.approx(42.0)
    assert s.pnl_std == pytest.approx(0.0)  # single sample => 0
    assert s.total_fee_usd == pytest.approx(0.5)
    assert s.total_notional_usd == pytest.approx(200.0)
    assert report.total_n_trades == 1
    assert report.overall_win_rate == 1.0
    assert report.overall_total_pnl_usd == pytest.approx(42.0)


def test_compute_regime_stats_loser_zero_winners() -> None:
    rows = [
        _row(
            regime=MacroRegime.RISK_OFF,
            pnl_usd=-5.0,
            qty=1.0,
            fill_price=10.0,
        )
    ]
    report = compute_regime_stats(rows)
    s = report.by_regime[0]
    assert s.n_winners == 0
    assert s.win_rate == 0.0
    assert s.total_pnl_usd == pytest.approx(-5.0)
    assert s.total_notional_usd == pytest.approx(10.0)


def test_compute_regime_stats_zero_pnl_is_not_winner() -> None:
    """Zero PnL must not count as a winner (strict ``> 0``)."""
    rows = [_row(regime=MacroRegime.NEUTRAL, pnl_usd=0.0, qty=1.0, fill_price=1.0)]
    report = compute_regime_stats(rows)
    assert report.by_regime[0].n_winners == 0
    assert report.by_regime[0].win_rate == 0.0


def test_compute_regime_stats_groups_by_regime_independently() -> None:
    rows = [
        _row(regime=MacroRegime.RISK_ON, pnl_usd=10.0, qty=1.0, fill_price=2.0),
        _row(regime=MacroRegime.RISK_ON, pnl_usd=-2.0, qty=1.0, fill_price=2.0),
        _row(regime=MacroRegime.RISK_OFF, pnl_usd=4.0, qty=1.0, fill_price=2.0),
    ]
    report = compute_regime_stats(rows)
    assert {s.regime for s in report.by_regime} == {
        MacroRegime.RISK_ON,
        MacroRegime.RISK_OFF,
    }
    by = {s.regime: s for s in report.by_regime}
    assert by[MacroRegime.RISK_ON].n_trades == 2
    assert by[MacroRegime.RISK_ON].n_winners == 1
    assert by[MacroRegime.RISK_ON].total_pnl_usd == pytest.approx(8.0)
    assert by[MacroRegime.RISK_OFF].n_trades == 1
    assert by[MacroRegime.RISK_OFF].n_winners == 1
    assert report.overall_total_pnl_usd == pytest.approx(12.0)


def test_compute_regime_stats_sorted_by_regime_value() -> None:
    rows = [
        _row(regime=r, pnl_usd=1.0, qty=1.0, fill_price=1.0)
        for r in (
            MacroRegime.RISK_OFF,
            MacroRegime.RISK_ON,
            MacroRegime.CRISIS,
            MacroRegime.NEUTRAL,
        )
    ]
    report = compute_regime_stats(rows)
    keys = [s.regime.value for s in report.by_regime]
    assert keys == sorted(keys)


def test_compute_regime_stats_pnl_std_population() -> None:
    """Population std-dev (ddof=0) over PnL within a regime."""
    rows = [
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=2.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=4.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=4.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=4.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=5.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=5.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=7.0, qty=1.0, fill_price=1.0),
        _row(regime=MacroRegime.NEUTRAL, pnl_usd=9.0, qty=1.0, fill_price=1.0),
    ]
    report = compute_regime_stats(rows)
    s = report.by_regime[0]
    # mean=5, var = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4 => std = 2
    assert s.pnl_std == pytest.approx(2.0, rel=1e-9)


def test_compute_regime_stats_rejects_non_row_input() -> None:
    with pytest.raises(TypeError, match="RegimeTradeRow"):
        compute_regime_stats(["not a row"])  # type: ignore[list-item]


# ----------------------------------------------------------------------
# INV-15 byte-identical replay
# ----------------------------------------------------------------------


def test_replay_byte_stable_across_three_runs() -> None:
    rows = [
        _row(
            ts_ns=i,
            symbol=f"SYM{i % 3}",
            regime=list(MacroRegime)[i % len(MacroRegime)],
            pnl_usd=(i % 7) - 3.0,
            fee_usd=(i % 5) * 0.1,
            qty=1.0 + (i % 4),
            fill_price=10.0 + (i % 11),
        )
        for i in range(50)
    ]
    a = compute_regime_stats(rows)
    b = compute_regime_stats(rows)
    c = compute_regime_stats(rows)
    assert a == b == c


def test_replay_permutation_invariant() -> None:
    rows = [
        _row(
            ts_ns=i * 1_000_000,
            symbol=f"SYM{i % 4}",
            regime=list(MacroRegime)[i % len(MacroRegime)],
            pnl_usd=((i * 13) % 19) - 9.0,
            fee_usd=(i % 3) * 0.05,
            qty=1.0 + (i % 5),
            fill_price=5.0 + (i % 17),
        )
        for i in range(40)
    ]
    a = compute_regime_stats(rows)
    b = compute_regime_stats(list(reversed(rows)))
    assert a == b


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _row(
    *,
    ts_ns: int = 0,
    symbol: str = "BTCUSDT",
    regime: MacroRegime = MacroRegime.NEUTRAL,
    pnl_usd: float = 0.0,
    fee_usd: float = 0.0,
    qty: float = 1.0,
    fill_price: float = 100.0,
) -> RegimeTradeRow:
    return RegimeTradeRow(
        ts_ns=ts_ns,
        symbol=symbol,
        regime=regime,
        pnl_usd=pnl_usd,
        fee_usd=fee_usd,
        qty=qty,
        fill_price=fill_price,
    )


def _stats(regime: MacroRegime) -> RegimeStats:
    return RegimeStats(
        regime=regime,
        n_trades=2,
        n_winners=1,
        win_rate=0.5,
        total_pnl_usd=0.0,
        avg_pnl_usd=0.0,
        pnl_std=0.0,
        total_fee_usd=0.0,
        total_notional_usd=0.0,
    )
