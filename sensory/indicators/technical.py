"""Technical indicators (B-05 — pandas-ta canonical adaptation).

SCVS row: SRC-SENSORY-INDICATORS-001 (registry/data_source_registry.yaml).

# ADAPTED FROM: twopirllc/pandas-ta
# - pandas_ta/momentum/rsi.py         (Wilder RSI)
# - pandas_ta/momentum/macd.py        (MACD line / signal / histogram)
# - pandas_ta/momentum/stoch.py       (Fast %K, Slow %K, Slow %D)
# - pandas_ta/volatility/atr.py       (Wilder ATR)
# - pandas_ta/volatility/bbands.py    (Bollinger Bands)
# - pandas_ta/trend/adx.py            (Wilder ADX / +DI / -DI)
# - pandas_ta/volume/obv.py           (On-Balance Volume)
# - pandas_ta/volume/vwap.py          (Volume-Weighted Average Price)
#
# License: MIT (pandas-ta). This file does NOT import pandas-ta — it is a
# from-scratch pure-Python reimplementation behind frozen DIX contracts so the
# canonical adaptation works in offline analytics tiers without dragging pandas
# / numpy / polars into the dependency graph.

Tier: OFFLINE_ONLY — hard-banned from runtime engines.

This module re-implements the canonical pandas-ta indicator family on top of
DIX's :class:`OHLCVBar` value object. Outputs are frozen ``IndicatorSeries``
value objects of length == ``len(bars)`` with ``None`` filling the warm-up
window per Wilder convention. Each indicator is a pure function — no IO, no
clock, no random, no global state, no engine cross-imports.

Hard constraints:
    * INV-15 — deterministic; same OHLCV input yields the same indicator
      output across machines and runs; no clock, no PRNG, no asyncio, no
      filesystem.
    * INV-12 — advisory only; this module never constructs typed bus events
      (``PatchProposal`` / ``SignalEvent`` / ``GovernanceDecision`` /
      ``ExecutionIntent``) and never imports the producing engines.
    * B-POLARS / B1 — no pandas / polars / numpy / pandas-ta imports anywhere
      (pinned by AST tests). The dispatcher / indicators / registry are pure
      stdlib.
    * No top-level ``random`` / ``time`` / ``datetime`` / ``os`` / ``asyncio``
      / ``websockets`` / ``langsmith`` imports.

What survives from pandas-ta:
    * Wilder RMA smoothing kernel — ``rma_t = ((n-1) * rma_{t-1} + x_t) / n``
      after a SMA-seeded first sample.
    * Indicator parameter conventions (period defaults, MACD 12/26/9,
      BBands 20/2.0, Stochastic 14/3/3, ATR 14, ADX 14).
    * Output naming and column semantics (`MACD_line` / `MACD_signal` /
      `MACD_hist`, `BBANDS_lower` / `BBANDS_middle` / `BBANDS_upper`, etc.).
    * Registry-driven design — :data:`INDICATOR_REGISTRY` lets the operator
      drive indicator selection / configuration without hard-coding.

What is stripped from pandas-ta:
    * The actual pandas / numpy / polars runtime (this module operates on
      tuples of :class:`OHLCVBar`).
    * The talib fallback path.
    * The ``Strategy`` builder (a DIX feature flag, not a code-time thing).
    * Mutability — every result is frozen + slotted with sorted-key meta.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""B-05 reimplements pandas-ta in pure stdlib — no new pip deps."""

MIN_PERIOD: Final[int] = 2
MAX_PERIOD: Final[int] = 1024
MIN_BARS: Final[int] = 1
MAX_BARS: Final[int] = 1_000_000
MAX_VALUE_LEN: Final[int] = 128
MAX_META_KEYS: Final[int] = 32

DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_MACD_FAST: Final[int] = 12
DEFAULT_MACD_SLOW: Final[int] = 26
DEFAULT_MACD_SIGNAL: Final[int] = 9
DEFAULT_ATR_PERIOD: Final[int] = 14
DEFAULT_ADX_PERIOD: Final[int] = 14
DEFAULT_BBANDS_PERIOD: Final[int] = 20
DEFAULT_BBANDS_STDEV: Final[float] = 2.0
DEFAULT_STOCH_K: Final[int] = 14
DEFAULT_STOCH_K_SMOOTH: Final[int] = 3
DEFAULT_STOCH_D: Final[int] = 3


class IndicatorError(ValueError):
    """Raised when indicator inputs or parameters are invalid."""


# ---------------------------------------------------------------------------
# Canonical indicator names
# ---------------------------------------------------------------------------


class IndicatorName(StrEnum):
    """Canonical indicator identifiers supported by :func:`compute_indicator`."""

    RSI = "rsi"
    MACD = "macd"
    ATR = "atr"
    ADX = "adx"
    BBANDS = "bbands"
    STOCH = "stoch"
    OBV = "obv"
    VWAP = "vwap"


CANONICAL_INDICATOR_ORDER: tuple[IndicatorName, ...] = (
    IndicatorName.RSI,
    IndicatorName.MACD,
    IndicatorName.ATR,
    IndicatorName.ADX,
    IndicatorName.BBANDS,
    IndicatorName.STOCH,
    IndicatorName.OBV,
    IndicatorName.VWAP,
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    """One OHLCV bar.

    Attributes:
        ts_ns: Monotonic timestamp (TimeAuthority, T0-04).
        open: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        volume: Bar volume (non-negative).
    """

    ts_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise IndicatorError("OHLCVBar.ts_ns must be int")
        if self.ts_ns < 0:
            raise IndicatorError("OHLCVBar.ts_ns must be non-negative")
        for field_name in ("open", "high", "low", "close", "volume"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise IndicatorError(f"OHLCVBar.{field_name} must be a real number")
            if value != value:  # NaN guard (no math import needed)
                raise IndicatorError(f"OHLCVBar.{field_name} must not be NaN")
        if self.volume < 0.0:
            raise IndicatorError("OHLCVBar.volume must be non-negative")
        if self.high < self.low:
            raise IndicatorError("OHLCVBar.high must be >= low")


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    """Operator-facing indicator request.

    Attributes:
        name: One of :class:`IndicatorName`.
        params: Sorted-key mapping of integer / float parameter overrides.
            Empty mapping selects canonical defaults.
    """

    name: IndicatorName
    params: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, IndicatorName):
            raise IndicatorError("IndicatorSpec.name must be IndicatorName")
        if not isinstance(self.params, Mapping):
            raise IndicatorError("IndicatorSpec.params must be a Mapping")
        for key, value in self.params.items():
            if not isinstance(key, str) or not key:
                raise IndicatorError("IndicatorSpec.params keys must be non-empty str")
            if len(key) > MAX_VALUE_LEN:
                raise IndicatorError("IndicatorSpec.params key too long")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise IndicatorError("IndicatorSpec.params values must be int/float")
        if len(self.params) > MAX_META_KEYS:
            raise IndicatorError("IndicatorSpec.params has too many keys")
        sorted_params = MappingProxyType({k: float(self.params[k]) for k in sorted(self.params)})
        object.__setattr__(self, "params", sorted_params)


@dataclass(frozen=True, slots=True)
class IndicatorSeries:
    """Aligned indicator output.

    Attributes:
        name: Indicator name.
        column: Output column key (e.g. ``"line"``, ``"signal"``, ``"hist"``).
        values: One value per input bar, with ``None`` for warm-up samples.
        meta: Sorted-key mapping of contextual notes (period etc.) for audit.
    """

    name: IndicatorName
    column: str
    values: tuple[float | None, ...]
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, IndicatorName):
            raise IndicatorError("IndicatorSeries.name must be IndicatorName")
        if not isinstance(self.column, str) or not self.column:
            raise IndicatorError("IndicatorSeries.column must be non-empty str")
        if len(self.column) > MAX_VALUE_LEN:
            raise IndicatorError("IndicatorSeries.column too long")
        if not isinstance(self.values, tuple):
            raise IndicatorError("IndicatorSeries.values must be tuple")
        for v in self.values:
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, float):
                raise IndicatorError("IndicatorSeries values must be float or None")
            if v != v:
                raise IndicatorError("IndicatorSeries values must not be NaN")
        if not isinstance(self.meta, Mapping):
            raise IndicatorError("IndicatorSeries.meta must be a Mapping")
        for key, value in self.meta.items():
            if not isinstance(key, str) or not key:
                raise IndicatorError("IndicatorSeries.meta keys must be non-empty str")
            if not isinstance(value, str):
                raise IndicatorError("IndicatorSeries.meta values must be str")
            if len(key) > MAX_VALUE_LEN or len(value) > MAX_VALUE_LEN:
                raise IndicatorError("IndicatorSeries.meta entry too long")
        if len(self.meta) > MAX_META_KEYS:
            raise IndicatorError("IndicatorSeries.meta has too many keys")
        sorted_meta = {k: self.meta[k] for k in sorted(self.meta)}
        object.__setattr__(self, "meta", MappingProxyType(sorted_meta))


@dataclass(frozen=True, slots=True)
class IndicatorBatch:
    """A bundle of related indicator series (e.g. MACD line/signal/hist).

    Attributes:
        name: Indicator name.
        series: Sorted-key mapping of ``column → IndicatorSeries``.
        digest: BLAKE2b-16 hex digest over the canonical text projection.
    """

    name: IndicatorName
    series: Mapping[str, IndicatorSeries]
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, IndicatorName):
            raise IndicatorError("IndicatorBatch.name must be IndicatorName")
        if not isinstance(self.series, Mapping) or not self.series:
            raise IndicatorError("IndicatorBatch.series must be non-empty Mapping")
        for key, value in self.series.items():
            if not isinstance(key, str) or not key:
                raise IndicatorError("IndicatorBatch.series keys must be non-empty str")
            if not isinstance(value, IndicatorSeries):
                raise IndicatorError("IndicatorBatch.series values must be IndicatorSeries")
            if value.name is not self.name:
                raise IndicatorError("IndicatorBatch.series IndicatorSeries.name mismatch")
            if value.column != key:
                raise IndicatorError("IndicatorBatch.series column/key mismatch")
        if not isinstance(self.digest, str) or len(self.digest) != 32:
            raise IndicatorError("IndicatorBatch.digest must be 32-char hex str")
        try:
            int(self.digest, 16)
        except ValueError as exc:  # pragma: no cover - defensive
            raise IndicatorError("IndicatorBatch.digest must be hex") from exc
        sorted_series = {k: self.series[k] for k in sorted(self.series)}
        object.__setattr__(self, "series", MappingProxyType(sorted_series))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_period(value: float | int | None, *, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndicatorError(f"{name} must be int")
    period = int(value)
    if period != float(value):
        raise IndicatorError(f"{name} must be a whole number")
    if period < MIN_PERIOD:
        raise IndicatorError(f"{name} must be >= {MIN_PERIOD}")
    if period > MAX_PERIOD:
        raise IndicatorError(f"{name} must be <= {MAX_PERIOD}")
    return period


def _check_period(period: int, name: str) -> int:
    if isinstance(period, int) and not isinstance(period, bool):
        fallback: int = period
    else:
        fallback = MIN_PERIOD
    return _coerce_period(period, default=fallback, name=name)


def _check_stdev(stdev: float, name: str) -> float:
    if isinstance(stdev, (int, float)) and not isinstance(stdev, bool):
        fallback: float = float(stdev)
    else:
        fallback = 1.0
    return _coerce_float(stdev, default=fallback, name=name)


def _coerce_float(value: float | int | None, *, default: float, name: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndicatorError(f"{name} must be a real number")
    out = float(value)
    if out != out:
        raise IndicatorError(f"{name} must not be NaN")
    if out <= 0.0:
        raise IndicatorError(f"{name} must be > 0")
    return out


def _validate_bars(bars: Sequence[OHLCVBar]) -> tuple[OHLCVBar, ...]:
    if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)):
        raise IndicatorError("bars must be a Sequence of OHLCVBar")
    out = tuple(bars)
    if not out:
        raise IndicatorError("bars must be non-empty")
    if len(out) > MAX_BARS:
        raise IndicatorError("bars exceeds MAX_BARS")
    for i, bar in enumerate(out):
        if not isinstance(bar, OHLCVBar):
            raise IndicatorError(f"bars[{i}] must be OHLCVBar")
        if i > 0 and bar.ts_ns < out[i - 1].ts_ns:
            raise IndicatorError("bars must be non-decreasing by ts_ns")
    return out


def _sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average; first ``period - 1`` values are ``None``."""

    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def _ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average seeded by SMA at ``period - 1``."""

    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed_sum = sum(values[:period])
    prev = seed_sum / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = (values[i] - prev) * alpha + prev
        out[i] = prev
    return out


def _rma(values: Sequence[float], period: int) -> list[float | None]:
    """Wilder smoothing (RMA): SMA seed, then ``(prev * (n-1) + x) / n``."""

    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    seed_sum = sum(values[:period])
    prev = seed_sum / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# RSI (Wilder, 14)
# ---------------------------------------------------------------------------


def compute_rsi(bars: Sequence[OHLCVBar], *, period: int = DEFAULT_RSI_PERIOD) -> IndicatorSeries:
    """Wilder RSI.

    # ADAPTED FROM: pandas_ta/momentum/rsi.py
    """

    period = _check_period(period, "rsi.period")
    seq = _validate_bars(bars)
    n = len(seq)
    values: list[float | None] = [None] * n
    if n <= period:
        return IndicatorSeries(
            name=IndicatorName.RSI,
            column="rsi",
            values=tuple(values),
            meta={"period": str(period)},
        )
    gains: list[float] = [0.0] * n
    losses: list[float] = [0.0] * n
    for i in range(1, n):
        delta = seq[i].close - seq[i - 1].close
        if delta > 0.0:
            gains[i] = delta
        else:
            losses[i] = -delta
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    if avg_loss == 0.0:
        values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        values[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            values[i] = 100.0 - (100.0 / (1.0 + rs))
    return IndicatorSeries(
        name=IndicatorName.RSI,
        column="rsi",
        values=tuple(values),
        meta={"period": str(period)},
    )


# ---------------------------------------------------------------------------
# MACD (12, 26, 9)
# ---------------------------------------------------------------------------


def compute_macd(
    bars: Sequence[OHLCVBar],
    *,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> IndicatorBatch:
    """MACD line / signal / histogram.

    # ADAPTED FROM: pandas_ta/momentum/macd.py
    """

    fast = _check_period(fast, "macd.fast")
    slow = _check_period(slow, "macd.slow")
    signal = _check_period(signal, "macd.signal")
    if fast >= slow:
        raise IndicatorError("MACD fast must be < slow")
    seq = _validate_bars(bars)
    closes = [bar.close for bar in seq]
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    line: list[float | None] = [None] * len(seq)
    for i in range(len(seq)):
        f = ema_fast[i]
        s = ema_slow[i]
        if f is None or s is None:
            continue
        line[i] = f - s
    line_values_for_signal = [v if v is not None else 0.0 for v in line]
    valid_start = next(
        (i for i, v in enumerate(line) if v is not None),
        len(seq),
    )
    signal_raw = _ema(line_values_for_signal[valid_start:], signal)
    signal_series: list[float | None] = [None] * len(seq)
    for offset, value in enumerate(signal_raw):
        if value is None:
            continue
        signal_series[valid_start + offset] = value
    hist: list[float | None] = [None] * len(seq)
    for i in range(len(seq)):
        if line[i] is None or signal_series[i] is None:
            continue
        hist[i] = line[i] - signal_series[i]
    meta = {
        "fast": str(fast),
        "slow": str(slow),
        "signal": str(signal),
    }
    line_series = IndicatorSeries(
        name=IndicatorName.MACD,
        column="line",
        values=tuple(line),
        meta=meta,
    )
    signal_ind = IndicatorSeries(
        name=IndicatorName.MACD,
        column="signal",
        values=tuple(signal_series),
        meta=meta,
    )
    hist_series = IndicatorSeries(
        name=IndicatorName.MACD,
        column="hist",
        values=tuple(hist),
        meta=meta,
    )
    series_map = {
        "hist": hist_series,
        "line": line_series,
        "signal": signal_ind,
    }
    return IndicatorBatch(
        name=IndicatorName.MACD,
        series=series_map,
        digest=_batch_digest(IndicatorName.MACD, series_map),
    )


# ---------------------------------------------------------------------------
# ATR (Wilder, 14)
# ---------------------------------------------------------------------------


def compute_atr(bars: Sequence[OHLCVBar], *, period: int = DEFAULT_ATR_PERIOD) -> IndicatorSeries:
    """Wilder ATR.

    # ADAPTED FROM: pandas_ta/volatility/atr.py
    """

    period = _check_period(period, "atr.period")
    seq = _validate_bars(bars)
    n = len(seq)
    tr = [0.0] * n
    tr[0] = seq[0].high - seq[0].low
    for i in range(1, n):
        prev_close = seq[i - 1].close
        tr[i] = max(
            seq[i].high - seq[i].low,
            abs(seq[i].high - prev_close),
            abs(seq[i].low - prev_close),
        )
    rma = _rma(tr, period)
    return IndicatorSeries(
        name=IndicatorName.ATR,
        column="atr",
        values=tuple(rma),
        meta={"period": str(period)},
    )


# ---------------------------------------------------------------------------
# ADX (Wilder, 14)
# ---------------------------------------------------------------------------


def compute_adx(bars: Sequence[OHLCVBar], *, period: int = DEFAULT_ADX_PERIOD) -> IndicatorBatch:
    """Wilder ADX with +DI / -DI companions.

    # ADAPTED FROM: pandas_ta/trend/adx.py
    """

    period = _check_period(period, "adx.period")
    seq = _validate_bars(bars)
    n = len(seq)
    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    if n > 1:
        tr[0] = seq[0].high - seq[0].low
    for i in range(1, n):
        up_move = seq[i].high - seq[i - 1].high
        down_move = seq[i - 1].low - seq[i].low
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0
        prev_close = seq[i - 1].close
        tr[i] = max(
            seq[i].high - seq[i].low,
            abs(seq[i].high - prev_close),
            abs(seq[i].low - prev_close),
        )
    atr = _rma(tr, period)
    plus_dm_smoothed = _rma(plus_dm, period)
    minus_dm_smoothed = _rma(minus_dm, period)
    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n
    dx: list[float | None] = [None] * n
    for i in range(n):
        a = atr[i]
        pdm = plus_dm_smoothed[i]
        mdm = minus_dm_smoothed[i]
        if a is None or pdm is None or mdm is None or a == 0.0:
            continue
        pdi = 100.0 * pdm / a
        mdi = 100.0 * mdm / a
        plus_di[i] = pdi
        minus_di[i] = mdi
        denom = pdi + mdi
        if denom == 0.0:
            dx[i] = 0.0
        else:
            dx[i] = 100.0 * abs(pdi - mdi) / denom
    adx_seed_idx = None
    for i in range(n):
        if dx[i] is not None:
            adx_seed_idx = i
            break
    adx: list[float | None] = [None] * n
    if adx_seed_idx is not None and n - adx_seed_idx >= period:
        seed_sum = 0.0
        for j in range(period):
            value = dx[adx_seed_idx + j]
            if value is None:
                seed_sum = 0.0
                break
            seed_sum += value
        prev = seed_sum / period
        adx[adx_seed_idx + period - 1] = prev
        for i in range(adx_seed_idx + period, n):
            value = dx[i]
            if value is None:
                continue
            prev = (prev * (period - 1) + value) / period
            adx[i] = prev
    meta = {"period": str(period)}
    adx_series = IndicatorSeries(name=IndicatorName.ADX, column="adx", values=tuple(adx), meta=meta)
    plus_di_series = IndicatorSeries(
        name=IndicatorName.ADX, column="plus_di", values=tuple(plus_di), meta=meta
    )
    minus_di_series = IndicatorSeries(
        name=IndicatorName.ADX, column="minus_di", values=tuple(minus_di), meta=meta
    )
    series_map = {
        "adx": adx_series,
        "minus_di": minus_di_series,
        "plus_di": plus_di_series,
    }
    return IndicatorBatch(
        name=IndicatorName.ADX,
        series=series_map,
        digest=_batch_digest(IndicatorName.ADX, series_map),
    )


# ---------------------------------------------------------------------------
# Bollinger Bands (20, 2.0)
# ---------------------------------------------------------------------------


def compute_bbands(
    bars: Sequence[OHLCVBar],
    *,
    period: int = DEFAULT_BBANDS_PERIOD,
    stdev: float = DEFAULT_BBANDS_STDEV,
) -> IndicatorBatch:
    """Bollinger Bands (lower / middle / upper).

    # ADAPTED FROM: pandas_ta/volatility/bbands.py
    """

    period = _check_period(period, "bbands.period")
    stdev = _check_stdev(stdev, "bbands.stdev")
    seq = _validate_bars(bars)
    closes = [bar.close for bar in seq]
    n = len(closes)
    middle = _sma(closes, period)
    lower: list[float | None] = [None] * n
    upper: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        sd = variance**0.5
        lower[i] = mean - stdev * sd
        upper[i] = mean + stdev * sd
    meta = {"period": str(period), "stdev": _format_float(stdev)}
    lower_series = IndicatorSeries(
        name=IndicatorName.BBANDS, column="lower", values=tuple(lower), meta=meta
    )
    middle_series = IndicatorSeries(
        name=IndicatorName.BBANDS, column="middle", values=tuple(middle), meta=meta
    )
    upper_series = IndicatorSeries(
        name=IndicatorName.BBANDS, column="upper", values=tuple(upper), meta=meta
    )
    series_map = {
        "lower": lower_series,
        "middle": middle_series,
        "upper": upper_series,
    }
    return IndicatorBatch(
        name=IndicatorName.BBANDS,
        series=series_map,
        digest=_batch_digest(IndicatorName.BBANDS, series_map),
    )


# ---------------------------------------------------------------------------
# Stochastic (14, 3, 3)
# ---------------------------------------------------------------------------


def compute_stoch(
    bars: Sequence[OHLCVBar],
    *,
    k_period: int = DEFAULT_STOCH_K,
    k_smooth: int = DEFAULT_STOCH_K_SMOOTH,
    d_period: int = DEFAULT_STOCH_D,
) -> IndicatorBatch:
    """Slow Stochastic %K / %D.

    # ADAPTED FROM: pandas_ta/momentum/stoch.py
    """

    k_period = _check_period(k_period, "stoch.k_period")
    k_smooth = _check_period(k_smooth, "stoch.k_smooth")
    d_period = _check_period(d_period, "stoch.d_period")
    seq = _validate_bars(bars)
    n = len(seq)
    fast_k: list[float | None] = [None] * n
    for i in range(k_period - 1, n):
        window = seq[i - k_period + 1 : i + 1]
        high = max(bar.high for bar in window)
        low = min(bar.low for bar in window)
        denom = high - low
        if denom == 0.0:
            fast_k[i] = 50.0
        else:
            fast_k[i] = 100.0 * (seq[i].close - low) / denom
    slow_k: list[float | None] = [None] * n
    valid_start = next((i for i, v in enumerate(fast_k) if v is not None), n)
    fast_k_filled = [v if v is not None else 0.0 for v in fast_k]
    smoothed = _sma(fast_k_filled[valid_start:], k_smooth)
    for offset, value in enumerate(smoothed):
        if value is None:
            continue
        slow_k[valid_start + offset] = value
    slow_d: list[float | None] = [None] * n
    valid_start_d = next((i for i, v in enumerate(slow_k) if v is not None), n)
    slow_k_filled = [v if v is not None else 0.0 for v in slow_k]
    smoothed_d = _sma(slow_k_filled[valid_start_d:], d_period)
    for offset, value in enumerate(smoothed_d):
        if value is None:
            continue
        slow_d[valid_start_d + offset] = value
    meta = {
        "d_period": str(d_period),
        "k_period": str(k_period),
        "k_smooth": str(k_smooth),
    }
    k_series = IndicatorSeries(
        name=IndicatorName.STOCH, column="k", values=tuple(slow_k), meta=meta
    )
    d_series = IndicatorSeries(
        name=IndicatorName.STOCH, column="d", values=tuple(slow_d), meta=meta
    )
    series_map = {"d": d_series, "k": k_series}
    return IndicatorBatch(
        name=IndicatorName.STOCH,
        series=series_map,
        digest=_batch_digest(IndicatorName.STOCH, series_map),
    )


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


def compute_obv(bars: Sequence[OHLCVBar]) -> IndicatorSeries:
    """On-Balance Volume.

    # ADAPTED FROM: pandas_ta/volume/obv.py
    """

    seq = _validate_bars(bars)
    n = len(seq)
    out: list[float | None] = [None] * n
    if n == 0:
        return IndicatorSeries(name=IndicatorName.OBV, column="obv", values=tuple(out), meta={})
    cumulative = 0.0
    out[0] = 0.0
    for i in range(1, n):
        if seq[i].close > seq[i - 1].close:
            cumulative += seq[i].volume
        elif seq[i].close < seq[i - 1].close:
            cumulative -= seq[i].volume
        out[i] = cumulative
    return IndicatorSeries(
        name=IndicatorName.OBV,
        column="obv",
        values=tuple(out),
        meta={},
    )


# ---------------------------------------------------------------------------
# VWAP (session-anchored)
# ---------------------------------------------------------------------------


def compute_vwap(bars: Sequence[OHLCVBar]) -> IndicatorSeries:
    """Cumulative VWAP anchored at the first bar.

    # ADAPTED FROM: pandas_ta/volume/vwap.py
    """

    seq = _validate_bars(bars)
    n = len(seq)
    out: list[float | None] = [None] * n
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(n):
        tp = (seq[i].high + seq[i].low + seq[i].close) / 3.0
        cum_tpv += tp * seq[i].volume
        cum_vol += seq[i].volume
        if cum_vol > 0.0:
            out[i] = cum_tpv / cum_vol
    return IndicatorSeries(
        name=IndicatorName.VWAP,
        column="vwap",
        values=tuple(out),
        meta={},
    )


# ---------------------------------------------------------------------------
# Digest helpers
# ---------------------------------------------------------------------------


def _format_float(value: float) -> str:
    """Canonical float formatting for meta strings (17 significant digits)."""

    return repr(float(value))


def _batch_digest(name: IndicatorName, series_map: Mapping[str, IndicatorSeries]) -> str:
    parts: list[str] = [name.value]
    for column in sorted(series_map):
        series = series_map[column]
        parts.append(f"col={column}")
        for key in sorted(series.meta):
            parts.append(f"meta:{key}={series.meta[key]}")
        for value in series.values:
            parts.append("None" if value is None else _format_float(value))
    text = "|".join(parts)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Dispatcher and registry
# ---------------------------------------------------------------------------


def compute_indicator(
    spec: IndicatorSpec,
    bars: Sequence[OHLCVBar],
) -> IndicatorSeries | IndicatorBatch:
    """Operator-facing dispatcher. Routes ``spec.name`` to its compute_*.

    Per-indicator parameter keys (canonical):
        * RSI: ``period``.
        * MACD: ``fast``, ``slow``, ``signal``.
        * ATR: ``period``.
        * ADX: ``period``.
        * BBANDS: ``period``, ``stdev``.
        * STOCH: ``k_period``, ``k_smooth``, ``d_period``.
        * OBV / VWAP: no parameters.
    """

    if not isinstance(spec, IndicatorSpec):
        raise IndicatorError("spec must be IndicatorSpec")
    params = spec.params
    if spec.name is IndicatorName.RSI:
        return compute_rsi(
            bars,
            period=_coerce_period(
                params.get("period"), default=DEFAULT_RSI_PERIOD, name="rsi.period"
            ),
        )
    if spec.name is IndicatorName.MACD:
        return compute_macd(
            bars,
            fast=_coerce_period(params.get("fast"), default=DEFAULT_MACD_FAST, name="macd.fast"),
            slow=_coerce_period(params.get("slow"), default=DEFAULT_MACD_SLOW, name="macd.slow"),
            signal=_coerce_period(
                params.get("signal"), default=DEFAULT_MACD_SIGNAL, name="macd.signal"
            ),
        )
    if spec.name is IndicatorName.ATR:
        return compute_atr(
            bars,
            period=_coerce_period(
                params.get("period"), default=DEFAULT_ATR_PERIOD, name="atr.period"
            ),
        )
    if spec.name is IndicatorName.ADX:
        return compute_adx(
            bars,
            period=_coerce_period(
                params.get("period"), default=DEFAULT_ADX_PERIOD, name="adx.period"
            ),
        )
    if spec.name is IndicatorName.BBANDS:
        return compute_bbands(
            bars,
            period=_coerce_period(
                params.get("period"), default=DEFAULT_BBANDS_PERIOD, name="bbands.period"
            ),
            stdev=_coerce_float(
                params.get("stdev"), default=DEFAULT_BBANDS_STDEV, name="bbands.stdev"
            ),
        )
    if spec.name is IndicatorName.STOCH:
        return compute_stoch(
            bars,
            k_period=_coerce_period(
                params.get("k_period"), default=DEFAULT_STOCH_K, name="stoch.k_period"
            ),
            k_smooth=_coerce_period(
                params.get("k_smooth"), default=DEFAULT_STOCH_K_SMOOTH, name="stoch.k_smooth"
            ),
            d_period=_coerce_period(
                params.get("d_period"), default=DEFAULT_STOCH_D, name="stoch.d_period"
            ),
        )
    if spec.name is IndicatorName.OBV:
        return compute_obv(bars)
    if spec.name is IndicatorName.VWAP:
        return compute_vwap(bars)
    raise IndicatorError(f"unknown indicator name: {spec.name!r}")  # pragma: no cover


INDICATOR_REGISTRY: Mapping[IndicatorName, tuple[str, ...]] = MappingProxyType(
    {
        IndicatorName.RSI: ("rsi",),
        IndicatorName.MACD: ("hist", "line", "signal"),
        IndicatorName.ATR: ("atr",),
        IndicatorName.ADX: ("adx", "minus_di", "plus_di"),
        IndicatorName.BBANDS: ("lower", "middle", "upper"),
        IndicatorName.STOCH: ("d", "k"),
        IndicatorName.OBV: ("obv",),
        IndicatorName.VWAP: ("vwap",),
    }
)
"""Sorted-key registry of ``IndicatorName → output column tuple``.

Used by operators to discover the available indicator surface without
hard-coding per-indicator switch statements.
"""


__all__ = [
    "CANONICAL_INDICATOR_ORDER",
    "DEFAULT_ADX_PERIOD",
    "DEFAULT_ATR_PERIOD",
    "DEFAULT_BBANDS_PERIOD",
    "DEFAULT_BBANDS_STDEV",
    "DEFAULT_MACD_FAST",
    "DEFAULT_MACD_SIGNAL",
    "DEFAULT_MACD_SLOW",
    "DEFAULT_RSI_PERIOD",
    "DEFAULT_STOCH_D",
    "DEFAULT_STOCH_K",
    "DEFAULT_STOCH_K_SMOOTH",
    "INDICATOR_REGISTRY",
    "IndicatorBatch",
    "IndicatorError",
    "IndicatorName",
    "IndicatorSeries",
    "IndicatorSpec",
    "MAX_BARS",
    "MAX_META_KEYS",
    "MAX_PERIOD",
    "MAX_VALUE_LEN",
    "MIN_BARS",
    "MIN_PERIOD",
    "NEW_PIP_DEPENDENCIES",
    "OHLCVBar",
    "compute_adx",
    "compute_atr",
    "compute_bbands",
    "compute_indicator",
    "compute_macd",
    "compute_obv",
    "compute_rsi",
    "compute_stoch",
    "compute_vwap",
]
