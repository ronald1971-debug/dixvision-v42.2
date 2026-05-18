"""Tests for learning_engine/analytics/pnl_attribution.py (S-10.1 polars)."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
from pathlib import Path

import pytest

# Module imports cleanly without polars installed (lazy-import contract).
from learning_engine.analytics.pnl_attribution import (
    NEW_PIP_DEPENDENCIES,
    PolarsPnLReport,
    SymbolAttribution,
    TradeRow,
    attribute_pnl_polars,
)

# Polars is required for actually running attribute_pnl_polars().
pl = pytest.importorskip("polars")  # noqa: F841

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "learning_engine" / "analytics" / "pnl_attribution.py"
)

_FORBIDDEN_TOPLEVEL_IMPORTS = (
    "datetime",
    "time",
    "asyncio",
    "threading",
    "subprocess",
    "socket",
    "logging",
    "polars",  # MUST be lazy-imported inside attribute_pnl_polars
    "pandas",
    "pyarrow",
    "numpy",
    "fsspec",
    "random",
    "secrets",
)


# ---------------------------------------------------------------------------
# Module metadata / authority lint
# ---------------------------------------------------------------------------


def test_module_declares_polars_pip_dependency() -> None:
    assert NEW_PIP_DEPENDENCIES == ("polars",)


def test_module_has_adapted_from_header() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: pola-rs/polars" in text


def test_module_has_no_forbidden_toplevel_imports() -> None:
    """Polars MUST be lazy-imported inside attribute_pnl_polars."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in tree.body:  # walk only top level (lazy imports are nested)
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_TOPLEVEL_IMPORTS:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN_TOPLEVEL_IMPORTS:
                bad.append(node.module)
    assert bad == [], f"Forbidden toplevel imports: {bad!r}"


def test_module_has_no_clock_substrings() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    for needle in (
        "time.time(",
        "time.monotonic(",
        "datetime.now(",
        "datetime.utcnow(",
    ):
        assert needle not in text, f"pnl_attribution.py contains {needle!r}"


def test_polars_lazy_import_lives_inside_attribute_pnl_polars() -> None:
    """The function body must contain `import polars` (lazy import contract)."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    target_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "attribute_pnl_polars":
            target_fn = node
            break
    assert target_fn is not None, "attribute_pnl_polars not found"
    has_lazy_polars = False
    for inner in ast.walk(target_fn):
        if isinstance(inner, ast.Import):
            for alias in inner.names:
                if alias.name == "polars":
                    has_lazy_polars = True
        elif isinstance(inner, ast.ImportFrom) and inner.module == "polars":
            has_lazy_polars = True
    assert has_lazy_polars, "polars must be lazy-imported inside attribute_pnl_polars"


def test_module_imports_without_polars_in_sys_modules() -> None:
    """Reimporting the module should not pull polars into sys.modules itself.

    polars is allowed to *already* be in sys.modules (the test process
    imported it at the top of this file via importorskip), but the
    module reload must not depend on it being there: we simulate
    "polars uninstalled" by stripping polars from sys.modules and
    asserting the module still imports.
    """
    saved = {k: v for k, v in sys.modules.items() if k.startswith("polars")}
    for k in list(sys.modules):
        if k.startswith("polars"):
            del sys.modules[k]
    if "learning_engine.analytics.pnl_attribution" in sys.modules:
        del sys.modules["learning_engine.analytics.pnl_attribution"]
    try:
        mod = importlib.import_module("learning_engine.analytics.pnl_attribution")
        assert mod.NEW_PIP_DEPENDENCIES == ("polars",)
        assert "polars" not in sys.modules, (
            "module import pulled polars in despite lazy-import contract"
        )
    finally:
        sys.modules.update(saved)
        importlib.import_module("learning_engine.analytics.pnl_attribution")


# ---------------------------------------------------------------------------
# TradeRow validation
# ---------------------------------------------------------------------------


def _row(
    *,
    ts_ns: int = 1_000,
    symbol: str = "btcusdt",
    side: str = "BUY",
    qty: float = 1.0,
    fill_price: float = 100.0,
    signal_price: float = 99.5,
    pnl_usd: float = 1.0,
    fee_usd: float = 0.05,
) -> TradeRow:
    return TradeRow(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        fill_price=fill_price,
        signal_price=signal_price,
        pnl_usd=pnl_usd,
        fee_usd=fee_usd,
    )


def test_trade_row_is_frozen() -> None:
    r = _row()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.symbol = "ethusdt"  # type: ignore[misc]


def test_trade_row_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError):
        _row(ts_ns=-1)


def test_trade_row_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError):
        _row(ts_ns=True)  # type: ignore[arg-type]


def test_trade_row_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        _row(symbol="")


def test_trade_row_rejects_invalid_side() -> None:
    with pytest.raises(ValueError):
        _row(side="HOLD")


def test_trade_row_rejects_negative_qty() -> None:
    with pytest.raises(ValueError):
        _row(qty=-1.0)


def test_trade_row_rejects_zero_fill_price() -> None:
    with pytest.raises(ValueError):
        _row(fill_price=0.0)


def test_trade_row_rejects_zero_signal_price() -> None:
    with pytest.raises(ValueError):
        _row(signal_price=0.0)


def test_trade_row_rejects_negative_fee() -> None:
    with pytest.raises(ValueError):
        _row(fee_usd=-0.01)


def test_trade_row_rejects_nan_qty() -> None:
    with pytest.raises(ValueError):
        _row(qty=float("nan"))


def test_trade_row_rejects_bool_qty() -> None:
    with pytest.raises(TypeError):
        _row(qty=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SymbolAttribution validation
# ---------------------------------------------------------------------------


def test_symbol_attribution_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        SymbolAttribution(
            symbol="",
            n_trades=1,
            notional_usd=1.0,
            realised_pnl_usd=0.0,
            signal_pnl_usd=0.0,
            slippage_pnl_usd=0.0,
            fee_pnl_usd=0.0,
        )


def test_symbol_attribution_rejects_negative_n_trades() -> None:
    with pytest.raises(ValueError):
        SymbolAttribution(
            symbol="x",
            n_trades=-1,
            notional_usd=0.0,
            realised_pnl_usd=0.0,
            signal_pnl_usd=0.0,
            slippage_pnl_usd=0.0,
            fee_pnl_usd=0.0,
        )


def test_symbol_attribution_rejects_negative_notional() -> None:
    with pytest.raises(ValueError):
        SymbolAttribution(
            symbol="x",
            n_trades=1,
            notional_usd=-1.0,
            realised_pnl_usd=0.0,
            signal_pnl_usd=0.0,
            slippage_pnl_usd=0.0,
            fee_pnl_usd=0.0,
        )


# ---------------------------------------------------------------------------
# PolarsPnLReport validation
# ---------------------------------------------------------------------------


def _sym(name: str) -> SymbolAttribution:
    return SymbolAttribution(
        symbol=name,
        n_trades=1,
        notional_usd=1.0,
        realised_pnl_usd=0.0,
        signal_pnl_usd=0.0,
        slippage_pnl_usd=0.0,
        fee_pnl_usd=0.0,
    )


def test_polars_pnl_report_rejects_unsorted_by_symbol() -> None:
    with pytest.raises(ValueError):
        PolarsPnLReport(
            by_symbol=(_sym("zzz"), _sym("aaa")),
            total_n_trades=2,
            total_notional_usd=2.0,
            total_realised_pnl_usd=0.0,
            total_signal_pnl_usd=0.0,
            total_slippage_pnl_usd=0.0,
            total_fee_pnl_usd=0.0,
        )


def test_polars_pnl_report_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError):
        PolarsPnLReport(
            by_symbol=(_sym("aaa"), _sym("aaa")),
            total_n_trades=2,
            total_notional_usd=2.0,
            total_realised_pnl_usd=0.0,
            total_signal_pnl_usd=0.0,
            total_slippage_pnl_usd=0.0,
            total_fee_pnl_usd=0.0,
        )


def test_polars_pnl_report_rejects_non_tuple() -> None:
    with pytest.raises(TypeError):
        PolarsPnLReport(
            by_symbol=[_sym("aaa")],  # type: ignore[arg-type]
            total_n_trades=1,
            total_notional_usd=1.0,
            total_realised_pnl_usd=0.0,
            total_signal_pnl_usd=0.0,
            total_slippage_pnl_usd=0.0,
            total_fee_pnl_usd=0.0,
        )


def test_polars_pnl_report_rejects_non_symbol_attribution_entries() -> None:
    with pytest.raises(TypeError):
        PolarsPnLReport(
            by_symbol=("not_a_row",),  # type: ignore[arg-type]
            total_n_trades=0,
            total_notional_usd=0.0,
            total_realised_pnl_usd=0.0,
            total_signal_pnl_usd=0.0,
            total_slippage_pnl_usd=0.0,
            total_fee_pnl_usd=0.0,
        )


def test_polars_pnl_report_rejects_negative_total_notional() -> None:
    with pytest.raises(ValueError):
        PolarsPnLReport(
            by_symbol=(),
            total_n_trades=0,
            total_notional_usd=-1.0,
            total_realised_pnl_usd=0.0,
            total_signal_pnl_usd=0.0,
            total_slippage_pnl_usd=0.0,
            total_fee_pnl_usd=0.0,
        )


# ---------------------------------------------------------------------------
# attribute_pnl_polars — empty / single-symbol / multi-symbol
# ---------------------------------------------------------------------------


def test_attribute_pnl_polars_empty() -> None:
    rep = attribute_pnl_polars(())
    assert rep.by_symbol == ()
    assert rep.total_n_trades == 0
    assert rep.total_notional_usd == 0.0
    assert rep.total_realised_pnl_usd == 0.0
    assert rep.total_signal_pnl_usd == 0.0
    assert rep.total_slippage_pnl_usd == 0.0
    assert rep.total_fee_pnl_usd == 0.0


def test_attribute_pnl_polars_single_buy_with_slippage_and_fee() -> None:
    """BUY at fill > signal → slippage_pnl < 0; fee_pnl < 0; realised stays."""
    r = _row(
        side="BUY",
        qty=2.0,
        fill_price=101.0,
        signal_price=100.0,
        pnl_usd=10.0,
        fee_usd=0.50,
    )
    rep = attribute_pnl_polars((r,))
    assert len(rep.by_symbol) == 1
    s = rep.by_symbol[0]
    assert s.symbol == "btcusdt"
    assert s.n_trades == 1
    assert s.notional_usd == pytest.approx(202.0)  # 2 * 101
    # slippage_pnl = -(101 - 100) * 2 * +1 = -2
    assert s.slippage_pnl_usd == pytest.approx(-2.0)
    # fee_pnl = -0.5
    assert s.fee_pnl_usd == pytest.approx(-0.5)
    # realised - slippage - fee = 10 - (-2) - (-0.5) = 12.5
    assert s.signal_pnl_usd == pytest.approx(12.5)
    assert s.realised_pnl_usd == pytest.approx(10.0)
    # Identity holds
    assert s.signal_pnl_usd + s.slippage_pnl_usd + s.fee_pnl_usd == pytest.approx(
        s.realised_pnl_usd
    )


def test_attribute_pnl_polars_single_sell_inverts_slippage_sign() -> None:
    """SELL at fill < signal → slippage_pnl < 0 (received less than expected)."""
    r = _row(
        side="SELL",
        qty=3.0,
        fill_price=99.0,
        signal_price=100.0,
        pnl_usd=-5.0,
        fee_usd=0.10,
    )
    rep = attribute_pnl_polars((r,))
    s = rep.by_symbol[0]
    # slippage_pnl = -(99 - 100) * 3 * -1 = -3
    assert s.slippage_pnl_usd == pytest.approx(-3.0)
    assert s.fee_pnl_usd == pytest.approx(-0.10)
    # realised - slippage - fee = -5 - (-3) - (-0.10) = -1.90
    assert s.signal_pnl_usd == pytest.approx(-1.90)


def test_attribute_pnl_polars_perfect_fill_zeroes_slippage() -> None:
    """fill_price == signal_price → slippage_pnl == 0."""
    r = _row(side="BUY", fill_price=100.0, signal_price=100.0, fee_usd=0.0, pnl_usd=2.5)
    rep = attribute_pnl_polars((r,))
    s = rep.by_symbol[0]
    assert s.slippage_pnl_usd == pytest.approx(0.0)
    assert s.fee_pnl_usd == pytest.approx(0.0)
    assert s.signal_pnl_usd == pytest.approx(2.5)


def test_attribute_pnl_polars_groups_by_symbol() -> None:
    rows = (
        _row(symbol="aaa", pnl_usd=1.0),
        _row(symbol="bbb", pnl_usd=2.0),
        _row(symbol="aaa", pnl_usd=3.0, ts_ns=2_000),
    )
    rep = attribute_pnl_polars(rows)
    assert [s.symbol for s in rep.by_symbol] == ["aaa", "bbb"]
    aaa = rep.by_symbol[0]
    bbb = rep.by_symbol[1]
    assert aaa.n_trades == 2
    assert bbb.n_trades == 1
    assert aaa.realised_pnl_usd == pytest.approx(4.0)
    assert bbb.realised_pnl_usd == pytest.approx(2.0)


def test_attribute_pnl_polars_sorts_symbols_ascending() -> None:
    rows = (
        _row(symbol="zzz"),
        _row(symbol="aaa"),
        _row(symbol="mmm"),
    )
    rep = attribute_pnl_polars(rows)
    assert [s.symbol for s in rep.by_symbol] == ["aaa", "mmm", "zzz"]


def test_attribute_pnl_polars_totals_match_sum_of_groups() -> None:
    rows = tuple(
        _row(symbol=s, pnl_usd=p, fee_usd=f, qty=q, fill_price=fp, signal_price=sp)
        for s, p, f, q, fp, sp in [
            ("aaa", 1.0, 0.10, 1.0, 100.0, 99.0),
            ("bbb", 2.0, 0.20, 2.0, 200.0, 199.5),
            ("aaa", -0.5, 0.05, 0.5, 101.0, 100.0),
        ]
    )
    rep = attribute_pnl_polars(rows)
    assert rep.total_n_trades == 3
    assert rep.total_realised_pnl_usd == pytest.approx(2.5)
    assert rep.total_realised_pnl_usd == pytest.approx(
        sum(s.realised_pnl_usd for s in rep.by_symbol)
    )
    assert rep.total_signal_pnl_usd == pytest.approx(sum(s.signal_pnl_usd for s in rep.by_symbol))
    assert rep.total_slippage_pnl_usd == pytest.approx(
        sum(s.slippage_pnl_usd for s in rep.by_symbol)
    )
    assert rep.total_fee_pnl_usd == pytest.approx(sum(s.fee_pnl_usd for s in rep.by_symbol))


def test_attribute_pnl_polars_identity_holds_per_symbol() -> None:
    """signal + slippage + fee == realised, per symbol."""
    rows = tuple(
        _row(
            symbol=s,
            side=side,
            qty=q,
            fill_price=fp,
            signal_price=sp,
            pnl_usd=p,
            fee_usd=f,
        )
        for s, side, q, fp, sp, p, f in [
            ("aaa", "BUY", 1.0, 100.0, 99.5, 0.5, 0.05),
            ("aaa", "SELL", 1.0, 100.0, 100.5, 1.0, 0.05),
            ("bbb", "BUY", 2.5, 50.0, 49.0, -1.0, 0.10),
        ]
    )
    rep = attribute_pnl_polars(rows)
    for s in rep.by_symbol:
        assert s.signal_pnl_usd + s.slippage_pnl_usd + s.fee_pnl_usd == pytest.approx(
            s.realised_pnl_usd, abs=1e-9
        )


def test_attribute_pnl_polars_rejects_non_trade_row() -> None:
    with pytest.raises(TypeError):
        attribute_pnl_polars(("not_a_row",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15 byte-stable replay determinism
# ---------------------------------------------------------------------------


def test_attribute_pnl_polars_is_byte_stable_across_three_runs() -> None:
    rows = tuple(
        _row(
            symbol=f"sym{(i % 5):02d}",
            ts_ns=1_000 + i,
            side="BUY" if i % 2 == 0 else "SELL",
            qty=float(i + 1),
            fill_price=100.0 + 0.5 * i,
            signal_price=100.0 + 0.25 * i,
            pnl_usd=0.1 * (i - 25),
            fee_usd=0.01 * (i % 7),
        )
        for i in range(50)
    )
    a = attribute_pnl_polars(rows)
    b = attribute_pnl_polars(rows)
    c = attribute_pnl_polars(rows)
    assert a == b == c


def test_attribute_pnl_polars_input_order_does_not_affect_output() -> None:
    """Sorting inputs internally → output is permutation-invariant."""
    rows = (
        _row(symbol="aaa", ts_ns=3_000, pnl_usd=1.0),
        _row(symbol="bbb", ts_ns=2_000, pnl_usd=2.0),
        _row(symbol="aaa", ts_ns=1_000, pnl_usd=3.0),
    )
    perm1 = attribute_pnl_polars(rows)
    perm2 = attribute_pnl_polars(tuple(reversed(rows)))
    perm3 = attribute_pnl_polars((rows[1], rows[0], rows[2]))
    assert perm1 == perm2 == perm3


# ---------------------------------------------------------------------------
# Defensive guards
# ---------------------------------------------------------------------------


def test_attribute_pnl_polars_handles_zero_qty() -> None:
    """qty=0 trades are allowed (closing pings) — they contribute zero pnl."""
    r = _row(qty=0.0, fill_price=100.0, signal_price=100.0, pnl_usd=0.0, fee_usd=0.0)
    rep = attribute_pnl_polars((r,))
    assert rep.by_symbol[0].notional_usd == pytest.approx(0.0)
    assert rep.by_symbol[0].slippage_pnl_usd == pytest.approx(0.0)


def test_attribute_pnl_polars_handles_many_symbols() -> None:
    rows = tuple(_row(symbol=f"sym{i:03d}") for i in range(100))
    rep = attribute_pnl_polars(rows)
    assert len(rep.by_symbol) == 100
    assert [s.symbol for s in rep.by_symbol] == sorted(s.symbol for s in rep.by_symbol)


def test_attribute_pnl_polars_no_polars_in_module_globals() -> None:
    import learning_engine.analytics.pnl_attribution as mod

    assert "polars" not in mod.__dict__, (
        "polars must not be a module-level binding (lazy-import contract)"
    )


def test_attribute_pnl_polars_returns_frozen_dataclass() -> None:
    rep = attribute_pnl_polars((_row(),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.total_n_trades = 999  # type: ignore[misc]
