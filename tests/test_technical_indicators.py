"""B-05 — pandas-ta technical indicators tests.

Covers:
    * AST authority pins (no pandas/polars/numpy/pandas-ta/clock/random
      top-level imports; no typed bus event constructors; no engine
      cross-imports; no langsmith).
    * NEW_PIP_DEPENDENCIES == ().
    * ``# ADAPTED FROM:`` headers.
    * Value-object validation for OHLCVBar / IndicatorSpec / IndicatorSeries
      / IndicatorBatch.
    * Happy-path computation for each indicator (RSI, MACD, ATR, ADX,
      BBANDS, STOCH, OBV, VWAP).
    * 3-run replay determinism for digests and series.
    * Dispatcher + registry symmetry.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensory.indicators import technical as tech  # noqa: E402
from sensory.indicators.technical import (  # noqa: E402
    CANONICAL_INDICATOR_ORDER,
    DEFAULT_ATR_PERIOD,
    DEFAULT_BBANDS_PERIOD,
    DEFAULT_BBANDS_STDEV,
    DEFAULT_MACD_FAST,
    DEFAULT_MACD_SIGNAL,
    DEFAULT_MACD_SLOW,
    DEFAULT_RSI_PERIOD,
    DEFAULT_STOCH_D,
    DEFAULT_STOCH_K,
    DEFAULT_STOCH_K_SMOOTH,
    INDICATOR_REGISTRY,
    MAX_BARS,
    MIN_PERIOD,
    NEW_PIP_DEPENDENCIES,
    IndicatorBatch,
    IndicatorError,
    IndicatorName,
    IndicatorSeries,
    IndicatorSpec,
    OHLCVBar,
    compute_adx,
    compute_atr,
    compute_bbands,
    compute_indicator,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_stoch,
    compute_vwap,
)

SOURCE_PATH = (REPO_ROOT / "sensory" / "indicators" / "technical.py").resolve()
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")
SOURCE_TREE = ast.parse(SOURCE_TEXT)


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


FORBIDDEN_TOP_LEVEL_IMPORTS = frozenset(
    {
        "pandas",
        "pandas_ta",
        "polars",
        "numpy",
        "scipy",
        "torch",
        "random",
        "time",
        "datetime",
        "os",
        "asyncio",
        "websockets",
        "langsmith",
        "secrets",
    }
)

FORBIDDEN_ENGINE_IMPORTS = frozenset(
    {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
    }
)

FORBIDDEN_TYPED_EVENT_CTORS = frozenset(
    {
        "PatchProposal",
        "SignalEvent",
        "GovernanceDecision",
        "ExecutionIntent",
    }
)


def _top_level_imports() -> set[str]:
    names: set[str] = set()
    for node in SOURCE_TREE.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_no_forbidden_top_level_imports() -> None:
    forbidden = _top_level_imports() & FORBIDDEN_TOP_LEVEL_IMPORTS
    assert forbidden == set(), f"forbidden top-level imports: {forbidden}"


def test_no_engine_cross_imports() -> None:
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")[0]
            assert head not in FORBIDDEN_ENGINE_IMPORTS, f"engine cross-import: {node.module}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                assert head not in FORBIDDEN_ENGINE_IMPORTS, f"engine cross-import: {alias.name}"


def test_does_not_construct_typed_bus_events() -> None:
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_TYPED_EVENT_CTORS, (
                f"forbidden typed-event ctor: {node.func.id}"
            )


def test_adapted_from_header_present() -> None:
    assert "# ADAPTED FROM: twopirllc/pandas-ta" in SOURCE_TEXT


def test_new_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_module_carries_offline_only_clause() -> None:
    assert "OFFLINE_ONLY" in SOURCE_TEXT


def test_canonical_indicator_order_matches_enum() -> None:
    assert CANONICAL_INDICATOR_ORDER == tuple(IndicatorName)
    assert len(CANONICAL_INDICATOR_ORDER) == 8


# ---------------------------------------------------------------------------
# OHLCVBar
# ---------------------------------------------------------------------------


def _bar(ts: int, o: float, h: float, lo: float, c: float, v: float = 1.0) -> OHLCVBar:
    return OHLCVBar(ts_ns=ts, open=o, high=h, low=lo, close=c, volume=v)


def test_ohlcvbar_basic() -> None:
    bar = _bar(1, 1.0, 2.0, 0.5, 1.5)
    assert bar.close == 1.5


def test_ohlcvbar_is_frozen() -> None:
    bar = _bar(1, 1.0, 2.0, 0.5, 1.5)
    with pytest.raises(FrozenInstanceError):
        bar.close = 9.0  # type: ignore[misc]


def test_ohlcvbar_rejects_negative_ts() -> None:
    with pytest.raises(IndicatorError):
        OHLCVBar(ts_ns=-1, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)


def test_ohlcvbar_rejects_bool_ts() -> None:
    with pytest.raises(IndicatorError):
        OHLCVBar(ts_ns=True, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)  # type: ignore[arg-type]


def test_ohlcvbar_rejects_high_below_low() -> None:
    with pytest.raises(IndicatorError):
        OHLCVBar(ts_ns=1, open=1.0, high=0.5, low=1.0, close=0.7, volume=1.0)


def test_ohlcvbar_rejects_negative_volume() -> None:
    with pytest.raises(IndicatorError):
        OHLCVBar(ts_ns=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=-0.1)


def test_ohlcvbar_rejects_nan() -> None:
    with pytest.raises(IndicatorError):
        OHLCVBar(ts_ns=1, open=float("nan"), high=2.0, low=0.5, close=1.5, volume=1.0)


# ---------------------------------------------------------------------------
# IndicatorSpec
# ---------------------------------------------------------------------------


def test_indicator_spec_sorts_params() -> None:
    spec = IndicatorSpec(name=IndicatorName.MACD, params={"slow": 26, "fast": 12, "signal": 9})
    assert list(spec.params) == ["fast", "signal", "slow"]
    assert spec.params["fast"] == 12.0


def test_indicator_spec_empty_params_ok() -> None:
    spec = IndicatorSpec(name=IndicatorName.RSI, params={})
    assert dict(spec.params) == {}


def test_indicator_spec_is_frozen() -> None:
    spec = IndicatorSpec(name=IndicatorName.RSI, params={"period": 14})
    with pytest.raises(FrozenInstanceError):
        spec.name = IndicatorName.MACD  # type: ignore[misc]


def test_indicator_spec_rejects_non_enum_name() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSpec(name="rsi", params={})  # type: ignore[arg-type]


def test_indicator_spec_rejects_non_mapping_params() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSpec(name=IndicatorName.RSI, params=[("period", 14)])  # type: ignore[arg-type]


def test_indicator_spec_rejects_blank_key() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSpec(name=IndicatorName.RSI, params={"": 14})


def test_indicator_spec_rejects_non_numeric_value() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSpec(name=IndicatorName.RSI, params={"period": "14"})  # type: ignore[dict-item]


def test_indicator_spec_rejects_bool_value() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSpec(name=IndicatorName.RSI, params={"period": True})


def test_indicator_spec_params_immutable() -> None:
    spec = IndicatorSpec(name=IndicatorName.RSI, params={"period": 14})
    with pytest.raises(TypeError):
        spec.params["period"] = 7  # type: ignore[index]


# ---------------------------------------------------------------------------
# IndicatorSeries
# ---------------------------------------------------------------------------


def test_indicator_series_basic() -> None:
    s = IndicatorSeries(
        name=IndicatorName.RSI,
        column="rsi",
        values=(None, 50.0),
        meta={"period": "14"},
    )
    assert s.values == (None, 50.0)
    assert list(s.meta) == ["period"]


def test_indicator_series_is_frozen() -> None:
    s = IndicatorSeries(
        name=IndicatorName.RSI,
        column="rsi",
        values=(None,),
        meta={},
    )
    with pytest.raises(FrozenInstanceError):
        s.column = "x"  # type: ignore[misc]


def test_indicator_series_rejects_blank_column() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSeries(name=IndicatorName.RSI, column="", values=(None,), meta={})


def test_indicator_series_rejects_non_float_value() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSeries(name=IndicatorName.RSI, column="rsi", values=("x",), meta={})  # type: ignore[arg-type]


def test_indicator_series_rejects_bool_value() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSeries(name=IndicatorName.RSI, column="rsi", values=(True,), meta={})


def test_indicator_series_rejects_nan() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSeries(
            name=IndicatorName.RSI,
            column="rsi",
            values=(float("nan"),),
            meta={},
        )


def test_indicator_series_rejects_non_str_meta_value() -> None:
    with pytest.raises(IndicatorError):
        IndicatorSeries(
            name=IndicatorName.RSI,
            column="rsi",
            values=(None,),
            meta={"period": 14},  # type: ignore[dict-item]
        )


def test_indicator_series_meta_immutable() -> None:
    s = IndicatorSeries(name=IndicatorName.RSI, column="rsi", values=(None,), meta={"a": "b"})
    with pytest.raises(TypeError):
        s.meta["a"] = "c"  # type: ignore[index]


# ---------------------------------------------------------------------------
# IndicatorBatch
# ---------------------------------------------------------------------------


def test_indicator_batch_basic() -> None:
    s = IndicatorSeries(name=IndicatorName.MACD, column="line", values=(None,), meta={})
    batch = IndicatorBatch(
        name=IndicatorName.MACD,
        series={"line": s},
        digest="00" * 16,
    )
    assert "line" in batch.series


def test_indicator_batch_is_frozen() -> None:
    s = IndicatorSeries(name=IndicatorName.MACD, column="line", values=(None,), meta={})
    batch = IndicatorBatch(name=IndicatorName.MACD, series={"line": s}, digest="00" * 16)
    with pytest.raises(FrozenInstanceError):
        batch.digest = "ff" * 16  # type: ignore[misc]


def test_indicator_batch_rejects_empty_series() -> None:
    with pytest.raises(IndicatorError):
        IndicatorBatch(name=IndicatorName.MACD, series={}, digest="00" * 16)


def test_indicator_batch_rejects_name_mismatch() -> None:
    s = IndicatorSeries(name=IndicatorName.RSI, column="line", values=(None,), meta={})
    with pytest.raises(IndicatorError):
        IndicatorBatch(name=IndicatorName.MACD, series={"line": s}, digest="00" * 16)


def test_indicator_batch_rejects_column_mismatch() -> None:
    s = IndicatorSeries(name=IndicatorName.MACD, column="line", values=(None,), meta={})
    with pytest.raises(IndicatorError):
        IndicatorBatch(name=IndicatorName.MACD, series={"signal": s}, digest="00" * 16)


def test_indicator_batch_rejects_bad_digest_length() -> None:
    s = IndicatorSeries(name=IndicatorName.MACD, column="line", values=(None,), meta={})
    with pytest.raises(IndicatorError):
        IndicatorBatch(name=IndicatorName.MACD, series={"line": s}, digest="abc")


def test_indicator_batch_rejects_non_hex_digest() -> None:
    s = IndicatorSeries(name=IndicatorName.MACD, column="line", values=(None,), meta={})
    with pytest.raises(IndicatorError):
        IndicatorBatch(name=IndicatorName.MACD, series={"line": s}, digest="z" * 32)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def _flat_bars(n: int, *, start: float = 100.0, step: float = 0.0) -> tuple[OHLCVBar, ...]:
    bars: list[OHLCVBar] = []
    for i in range(n):
        price = start + i * step
        bars.append(_bar(i * 10, price, price + 0.5, price - 0.5, price, 1.0 + i * 0.1))
    return tuple(bars)


def test_rsi_warmup_returns_none() -> None:
    bars = _flat_bars(10)
    series = compute_rsi(bars, period=14)
    assert all(v is None for v in series.values)


def test_rsi_flat_input_after_warmup_no_loss() -> None:
    bars = _flat_bars(30)
    series = compute_rsi(bars, period=14)
    later = [v for v in series.values if v is not None]
    assert later
    assert all(v == 100.0 for v in later)


def test_rsi_monotonic_uptrend_high() -> None:
    bars = _flat_bars(40, step=0.5)
    series = compute_rsi(bars, period=14)
    later = [v for v in series.values if v is not None]
    assert all(v == 100.0 for v in later)


def test_rsi_monotonic_downtrend_low() -> None:
    bars = _flat_bars(40, start=100.0, step=-0.5)
    series = compute_rsi(bars, period=14)
    later = [v for v in series.values if v is not None]
    assert all(v == 0.0 for v in later)


def test_rsi_default_period() -> None:
    bars = _flat_bars(40)
    series = compute_rsi(bars)
    assert series.meta["period"] == str(DEFAULT_RSI_PERIOD)


def test_rsi_invalid_period_rejected() -> None:
    bars = _flat_bars(40)
    with pytest.raises(IndicatorError):
        compute_rsi(bars, period=1)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def test_macd_returns_three_columns() -> None:
    bars = _flat_bars(60, step=0.1)
    batch = compute_macd(bars)
    assert set(batch.series) == {"line", "signal", "hist"}
    assert batch.series["line"].column == "line"


def test_macd_meta_records_periods() -> None:
    bars = _flat_bars(60, step=0.1)
    batch = compute_macd(bars)
    assert batch.series["line"].meta["fast"] == str(DEFAULT_MACD_FAST)
    assert batch.series["line"].meta["slow"] == str(DEFAULT_MACD_SLOW)
    assert batch.series["line"].meta["signal"] == str(DEFAULT_MACD_SIGNAL)


def test_macd_rejects_fast_ge_slow() -> None:
    bars = _flat_bars(60)
    with pytest.raises(IndicatorError):
        compute_macd(bars, fast=26, slow=26)


def test_macd_hist_equals_line_minus_signal() -> None:
    bars = _flat_bars(80, step=0.2)
    batch = compute_macd(bars)
    line = batch.series["line"].values
    sig = batch.series["signal"].values
    hist = batch.series["hist"].values
    for i in range(len(line)):
        if line[i] is None or sig[i] is None:
            assert hist[i] is None
        else:
            assert hist[i] is not None
            assert abs(hist[i] - (line[i] - sig[i])) < 1e-9


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def test_atr_basic() -> None:
    bars = _flat_bars(40)
    series = compute_atr(bars, period=14)
    later = [v for v in series.values if v is not None]
    assert later
    assert all(v >= 0.0 for v in later)


def test_atr_default_period() -> None:
    bars = _flat_bars(40)
    series = compute_atr(bars)
    assert series.meta["period"] == str(DEFAULT_ATR_PERIOD)


def test_atr_invalid_period_rejected() -> None:
    bars = _flat_bars(40)
    with pytest.raises(IndicatorError):
        compute_atr(bars, period=0)


def test_atr_increases_with_widening_range() -> None:
    bars = []
    for i in range(40):
        spread = 0.5 + i * 0.05
        bars.append(_bar(i * 10, 100.0, 100.0 + spread, 100.0 - spread, 100.0, 1.0))
    series = compute_atr(bars, period=14)
    later = [v for v in series.values if v is not None]
    assert later[-1] > later[0]


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def test_adx_returns_three_columns() -> None:
    bars = _flat_bars(80, step=0.2)
    batch = compute_adx(bars)
    assert set(batch.series) == {"adx", "plus_di", "minus_di"}


def test_adx_uptrend_plus_di_dominates() -> None:
    bars = _flat_bars(80, step=0.5)
    batch = compute_adx(bars)
    plus = [v for v in batch.series["plus_di"].values if v is not None]
    minus = [v for v in batch.series["minus_di"].values if v is not None]
    assert plus[-1] > minus[-1]


def test_adx_downtrend_minus_di_dominates() -> None:
    bars = _flat_bars(80, start=100.0, step=-0.5)
    batch = compute_adx(bars)
    plus = [v for v in batch.series["plus_di"].values if v is not None]
    minus = [v for v in batch.series["minus_di"].values if v is not None]
    assert minus[-1] > plus[-1]


def test_adx_default_period() -> None:
    bars = _flat_bars(40)
    batch = compute_adx(bars)
    assert batch.series["adx"].meta["period"] == "14"


# ---------------------------------------------------------------------------
# BBANDS
# ---------------------------------------------------------------------------


def test_bbands_lower_below_middle_below_upper() -> None:
    bars = []
    for i in range(40):
        bars.append(_bar(i * 10, 100.0, 101.0, 99.0, 100.0 + (1.0 if i % 2 == 0 else -1.0), 1.0))
    batch = compute_bbands(bars, period=20, stdev=2.0)
    for i in range(19, 40):
        lo = batch.series["lower"].values[i]
        mid = batch.series["middle"].values[i]
        hi = batch.series["upper"].values[i]
        assert lo is not None and mid is not None and hi is not None
        assert lo <= mid <= hi


def test_bbands_default_params() -> None:
    bars = _flat_bars(40)
    batch = compute_bbands(bars)
    assert batch.series["middle"].meta["period"] == str(DEFAULT_BBANDS_PERIOD)
    assert float(batch.series["middle"].meta["stdev"]) == DEFAULT_BBANDS_STDEV


def test_bbands_rejects_zero_stdev() -> None:
    bars = _flat_bars(40)
    with pytest.raises(IndicatorError):
        compute_bbands(bars, stdev=0.0)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


def test_stoch_returns_k_and_d() -> None:
    bars = _flat_bars(40)
    batch = compute_stoch(bars)
    assert set(batch.series) == {"k", "d"}


def test_stoch_default_meta() -> None:
    bars = _flat_bars(40)
    batch = compute_stoch(bars)
    assert batch.series["k"].meta["k_period"] == str(DEFAULT_STOCH_K)
    assert batch.series["k"].meta["k_smooth"] == str(DEFAULT_STOCH_K_SMOOTH)
    assert batch.series["k"].meta["d_period"] == str(DEFAULT_STOCH_D)


def test_stoch_values_in_zero_hundred_range() -> None:
    bars = []
    for i in range(40):
        close = 100.0 + (5.0 if i % 3 == 0 else -2.0)
        bars.append(_bar(i * 10, 100.0, close + 1.0, close - 1.0, close, 1.0))
    batch = compute_stoch(bars)
    for v in batch.series["k"].values:
        if v is None:
            continue
        assert 0.0 <= v <= 100.0
    for v in batch.series["d"].values:
        if v is None:
            continue
        assert 0.0 <= v <= 100.0


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


def test_obv_zero_on_flat_close() -> None:
    bars = _flat_bars(10)
    series = compute_obv(bars)
    assert series.values[-1] == 0.0


def test_obv_positive_on_uptrend() -> None:
    bars = _flat_bars(10, step=0.5)
    series = compute_obv(bars)
    assert series.values[-1] > 0.0


def test_obv_negative_on_downtrend() -> None:
    bars = _flat_bars(10, start=100.0, step=-0.5)
    series = compute_obv(bars)
    assert series.values[-1] < 0.0


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


def test_vwap_increases_under_higher_close() -> None:
    bars = _flat_bars(10, step=0.5)
    series = compute_vwap(bars)
    later = [v for v in series.values if v is not None]
    assert later[-1] > later[0]


def test_vwap_first_bar_is_typical_price() -> None:
    bars = (_bar(0, 100.0, 102.0, 98.0, 101.0, 5.0),)
    series = compute_vwap(bars)
    assert series.values[0] == pytest.approx((102.0 + 98.0 + 101.0) / 3.0)


# ---------------------------------------------------------------------------
# Dispatcher and registry
# ---------------------------------------------------------------------------


def test_compute_indicator_dispatches_rsi() -> None:
    bars = _flat_bars(40)
    out = compute_indicator(IndicatorSpec(name=IndicatorName.RSI, params={}), bars)
    assert isinstance(out, IndicatorSeries)
    assert out.name is IndicatorName.RSI


def test_compute_indicator_dispatches_macd() -> None:
    bars = _flat_bars(60)
    out = compute_indicator(IndicatorSpec(name=IndicatorName.MACD, params={}), bars)
    assert isinstance(out, IndicatorBatch)
    assert out.name is IndicatorName.MACD


def test_compute_indicator_respects_params() -> None:
    bars = _flat_bars(40)
    out = compute_indicator(IndicatorSpec(name=IndicatorName.RSI, params={"period": 7}), bars)
    assert isinstance(out, IndicatorSeries)
    assert out.meta["period"] == "7"


def test_compute_indicator_rejects_non_spec() -> None:
    bars = _flat_bars(40)
    with pytest.raises(IndicatorError):
        compute_indicator(IndicatorName.RSI, bars)  # type: ignore[arg-type]


def test_compute_indicator_rejects_fractional_period() -> None:
    bars = _flat_bars(40)
    with pytest.raises(IndicatorError):
        compute_indicator(IndicatorSpec(name=IndicatorName.RSI, params={"period": 3.5}), bars)


def test_indicator_registry_matches_dispatcher_output() -> None:
    bars = _flat_bars(60, step=0.1)
    for name in CANONICAL_INDICATOR_ORDER:
        out = compute_indicator(IndicatorSpec(name=name, params={}), bars)
        expected_cols = INDICATOR_REGISTRY[name]
        if isinstance(out, IndicatorBatch):
            assert tuple(sorted(out.series)) == expected_cols
        else:
            assert (out.column,) == expected_cols


def test_indicator_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        INDICATOR_REGISTRY[IndicatorName.RSI] = ("x",)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------


def _build_walk(n: int) -> tuple[OHLCVBar, ...]:
    bars = []
    price = 100.0
    for i in range(n):
        bump = 0.5 if i % 3 == 0 else (-0.4 if i % 5 == 0 else 0.1)
        price += bump
        bars.append(_bar(i * 10, price, price + 0.5, price - 0.5, price, 1.0 + (i % 7) * 0.2))
    return tuple(bars)


def test_rsi_three_run_replay_equality() -> None:
    bars = _build_walk(40)
    s1 = compute_rsi(bars, period=14)
    s2 = compute_rsi(bars, period=14)
    s3 = compute_rsi(bars, period=14)
    assert s1.values == s2.values == s3.values


def test_macd_digest_three_run_replay_equality() -> None:
    bars = _build_walk(80)
    d1 = compute_macd(bars).digest
    d2 = compute_macd(bars).digest
    d3 = compute_macd(bars).digest
    assert d1 == d2 == d3


def test_macd_digest_changes_with_close_value() -> None:
    bars = list(_build_walk(80))
    base = compute_macd(tuple(bars)).digest
    target = bars[40]
    bars[40] = _bar(
        target.ts_ns,
        target.open,
        target.high + 1.0,
        target.low,
        target.close + 1.0,
        target.volume,
    )
    perturbed = compute_macd(tuple(bars)).digest
    assert base != perturbed


def test_adx_digest_three_run_replay_equality() -> None:
    bars = _build_walk(80)
    d1 = compute_adx(bars).digest
    d2 = compute_adx(bars).digest
    d3 = compute_adx(bars).digest
    assert d1 == d2 == d3


def test_bbands_digest_three_run_replay_equality() -> None:
    bars = _build_walk(80)
    d1 = compute_bbands(bars).digest
    d2 = compute_bbands(bars).digest
    d3 = compute_bbands(bars).digest
    assert d1 == d2 == d3


def test_stoch_digest_three_run_replay_equality() -> None:
    bars = _build_walk(80)
    d1 = compute_stoch(bars).digest
    d2 = compute_stoch(bars).digest
    d3 = compute_stoch(bars).digest
    assert d1 == d2 == d3


def test_macd_digest_independent_of_param_dict_order() -> None:
    bars = _build_walk(60)
    spec_a = IndicatorSpec(name=IndicatorName.MACD, params={"fast": 12, "slow": 26, "signal": 9})
    spec_b = IndicatorSpec(name=IndicatorName.MACD, params={"signal": 9, "slow": 26, "fast": 12})
    out_a = compute_indicator(spec_a, bars)
    out_b = compute_indicator(spec_b, bars)
    assert isinstance(out_a, IndicatorBatch)
    assert isinstance(out_b, IndicatorBatch)
    assert out_a.digest == out_b.digest


# ---------------------------------------------------------------------------
# Validation: bars
# ---------------------------------------------------------------------------


def test_compute_rsi_rejects_empty() -> None:
    with pytest.raises(IndicatorError):
        compute_rsi((), period=14)


def test_compute_rsi_rejects_non_sequence() -> None:
    with pytest.raises(IndicatorError):
        compute_rsi("abc", period=14)  # type: ignore[arg-type]


def test_compute_rsi_rejects_unsorted_ts() -> None:
    bars = (
        _bar(2, 1.0, 2.0, 0.5, 1.5, 1.0),
        _bar(1, 1.0, 2.0, 0.5, 1.5, 1.0),
    )
    with pytest.raises(IndicatorError):
        compute_rsi(bars, period=2)


def test_compute_rsi_rejects_min_period_below_2() -> None:
    bars = _flat_bars(10)
    with pytest.raises(IndicatorError):
        compute_rsi(bars, period=MIN_PERIOD - 1)


# ---------------------------------------------------------------------------
# Module namespace
# ---------------------------------------------------------------------------


def test_module_namespace() -> None:
    assert hasattr(tech, "OHLCVBar")
    assert hasattr(tech, "IndicatorSpec")
    assert hasattr(tech, "IndicatorSeries")
    assert hasattr(tech, "IndicatorBatch")
    assert hasattr(tech, "IndicatorName")
    assert hasattr(tech, "compute_indicator")
    assert hasattr(tech, "INDICATOR_REGISTRY")


def test_max_bars_constant_present() -> None:
    assert MAX_BARS == 1_000_000
