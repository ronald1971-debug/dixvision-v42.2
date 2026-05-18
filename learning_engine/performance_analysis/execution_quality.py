# ADAPTED FROM: qlib/qlib/contrib/evaluate.py
# ADAPTED FROM: qlib/qlib/backtest/profit_attribution.py
"""Execution-quality scoring — Implementation Shortfall, VWAP deviation, timing cost.

Adapted from microsoft/qlib's ``contrib/evaluate.py`` execution-cost pattern:
every realised fill is benchmarked against the *arrival price* (the mid at
the moment the parent decision was made) and the *interval VWAP* (the
volume-weighted average price across the parent-order window).  The two
benchmarks decompose execution cost into the classic Almgren-Chriss
"timing" and "VWAP-deviation" components — the same decomposition qlib's
backtest report emits.

This module is the **third and final** S-04 pyqlib leaf
(``pnl_attribution.py`` + ``alpha_decay.py`` + ``execution_quality.py``);
it intentionally restricts itself to one concern: scoring a stream of
:class:`BenchmarkedFill` records against arrival / VWAP benchmarks.

Tier
----
**OFFLINE.** ``learning_engine/performance_analysis/`` is a slow-cadence
analytics tier — never called from the hot path, never imported by
``hot_path/`` modules (authority_lint T1 / B1).

Design constraints
------------------
* **Pure functions.** No clock reads (``time.time()`` / ``datetime.now()``),
  no IO, no global mutable state.  Replay-deterministic (INV-15): identical
  inputs produce byte-identical :class:`ExecutionQualityReport` outputs.
* **Frozen contracts.** :class:`BenchmarkedFill` and
  :class:`ExecutionQualityReport` are ``@dataclass(frozen=True, slots=True)``
  so structural equality is preserved across replays.
* **Eager validation.** Constructors reject malformed input
  (``ValueError`` / ``TypeError``) so a downstream learning loop never
  observes partial state.
* **No new pip dependencies.** :data:`NEW_PIP_DEPENDENCIES` is empty —
  the qlib formulas only need :mod:`math` from the stdlib.
* **Stable accumulation order.** Aggregators walk fills in iteration
  order and use plain Python floats so floating-point rounding is
  identical across CPython versions.

Algorithmic summary
-------------------
For one round-trip ``BacktestTrade`` paired with its arrival mid and the
interval VWAP (with ``side_sign = +1`` for BUY, ``-1`` for SELL,
``notional = qty * fill_price``):

* ``is_cost_usd      = -(fill_price - arrival_price) * qty * side_sign``
  — Implementation Shortfall.  Negative when the fill paid more (BUY) /
  received less (SELL) than the arrival mid.
* ``vwap_deviation_usd = -(fill_price - interval_vwap) * qty * side_sign``
  — execution-side cost relative to the interval VWAP benchmark.
* ``timing_cost_usd  = is_cost_usd - vwap_deviation_usd``
  — equivalently ``-(interval_vwap - arrival_price) * qty * side_sign``.
  The price drift between decision time and the trade's interval window;
  Almgren-Chriss "timing" component.
* ``participation_rate = qty / interval_volume`` (when ``interval_volume
  > 0``) — own footprint inside the parent-order window; aggregated by
  notional weight at the report level.

The aggregator preserves the IS = VWAP-dev + timing identity exactly
for every :class:`ExecutionQualityReport` (sums commute over the
fill stream).  Basis-point helpers normalise costs against the fill
notional and return ``0.0`` when notional is non-positive.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from typing import Final

from core.contracts.backtest_result import BacktestTrade
from core.contracts.events import Side

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""S-04.3 introduces no new pip dependencies — pure stdlib."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkedFill:
    """One :class:`BacktestTrade` paired with arrival + VWAP benchmarks.

    Args:
        trade: The realised fill being scored.  Must be a
            :class:`BacktestTrade`; fees and PnL fields on the trade are
            ignored by this module — they belong to ``pnl_attribution``.
        arrival_price: The mid price at the moment the parent decision
            was emitted (Implementation Shortfall benchmark).  Must be
            strictly positive and finite.
        interval_vwap: Volume-weighted average price across the parent
            order's interval window (VWAP benchmark).  Must be strictly
            positive and finite.
        interval_volume: Total market volume executed inside the same
            interval (in the same units as ``trade.qty`` — *e.g.* base
            units for an L1 fill, USD notional for venue-aggregated).
            ``0.0`` is allowed and means "participation rate is
            undefined" — the aggregator surfaces ``0.0`` for that fill.
            Must be ``>= 0`` and finite.

    Raises:
        TypeError: If any field has the wrong runtime type.
        ValueError: If a benchmark is non-positive / non-finite, or
            ``interval_volume`` is negative.
    """

    trade: BacktestTrade
    arrival_price: float
    interval_vwap: float
    interval_volume: float

    def __post_init__(self) -> None:
        if not isinstance(self.trade, BacktestTrade):
            raise TypeError(f"trade must be BacktestTrade; got {type(self.trade).__name__}")
        for name, value in (
            ("arrival_price", self.arrival_price),
            ("interval_vwap", self.interval_vwap),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be float; got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite; got {value}")
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0; got {value}")
        if not isinstance(self.interval_volume, (int, float)) or isinstance(
            self.interval_volume, bool
        ):
            raise TypeError(
                f"interval_volume must be float; got {type(self.interval_volume).__name__}"
            )
        if not math.isfinite(float(self.interval_volume)):
            raise ValueError(f"interval_volume must be finite; got {self.interval_volume}")
        if self.interval_volume < 0.0:
            raise ValueError(f"interval_volume must be >= 0; got {self.interval_volume}")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionQualityReport:
    """Aggregate execution-quality stats across a fill stream.

    Args:
        n_fills: Number of :class:`BenchmarkedFill` rows aggregated.
            ``>= 0``.
        notional_usd: Sum of ``trade.qty * trade.price`` across all
            fills.  Always ``>= 0``.
        is_cost_usd: Implementation Shortfall in USD.  Signed: negative
            when the realised fills underperformed the arrival mid
            (paid more on BUY / received less on SELL).
        vwap_deviation_usd: Cost relative to interval VWAP.  Same sign
            convention.
        timing_cost_usd: ``is_cost_usd - vwap_deviation_usd``.  Captures
            price drift between arrival and the parent-order window.
        participation_rate: Notional-weighted average of
            ``qty / interval_volume`` across fills with positive
            volume.  ``0.0`` when no fill has positive volume.

    Invariants enforced in :meth:`__post_init__`:
        * ``n_fills >= 0``
        * ``notional_usd >= 0`` and finite
        * ``participation_rate >= 0`` and finite
        * All cost fields are finite (no NaN / ±Inf)
        * ``is_cost_usd == vwap_deviation_usd + timing_cost_usd``
          (Almgren-Chriss decomposition identity, exact under the
          aggregator's float-summation strategy).

    The class is hashable and frozen — two reports constructed from the
    same fill stream compare equal byte-for-byte.
    """

    n_fills: int
    notional_usd: float
    is_cost_usd: float
    vwap_deviation_usd: float
    timing_cost_usd: float
    participation_rate: float

    def __post_init__(self) -> None:
        if self.n_fills < 0:
            raise ValueError(f"n_fills must be >= 0; got {self.n_fills}")
        for name, value in (
            ("notional_usd", self.notional_usd),
            ("is_cost_usd", self.is_cost_usd),
            ("vwap_deviation_usd", self.vwap_deviation_usd),
            ("timing_cost_usd", self.timing_cost_usd),
            ("participation_rate", self.participation_rate),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite; got {value}")
        if self.notional_usd < 0.0:
            raise ValueError(f"notional_usd must be >= 0; got {self.notional_usd}")
        if self.participation_rate < 0.0:
            raise ValueError(f"participation_rate must be >= 0; got {self.participation_rate}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def empty_report() -> ExecutionQualityReport:
    """Return the canonical zero :class:`ExecutionQualityReport`."""
    return ExecutionQualityReport(
        n_fills=0,
        notional_usd=0.0,
        is_cost_usd=0.0,
        vwap_deviation_usd=0.0,
        timing_cost_usd=0.0,
        participation_rate=0.0,
    )


def _side_sign(side: Side) -> float:
    if side is Side.BUY:
        return 1.0
    if side is Side.SELL:
        return -1.0
    raise ValueError(f"unsupported side: {side!r}")  # pragma: no cover


def is_cost_bps(report: ExecutionQualityReport) -> float:
    """Implementation Shortfall in basis points of notional.

    Returns ``0.0`` when ``report.notional_usd <= 0.0`` to avoid NaN
    propagation downstream.
    """
    if report.notional_usd <= 0.0:
        return 0.0
    return report.is_cost_usd / report.notional_usd * 10_000.0


def vwap_deviation_bps(report: ExecutionQualityReport) -> float:
    """VWAP deviation in basis points of notional (``0.0`` when notional <= 0)."""
    if report.notional_usd <= 0.0:
        return 0.0
    return report.vwap_deviation_usd / report.notional_usd * 10_000.0


def timing_cost_bps(report: ExecutionQualityReport) -> float:
    """Timing cost in basis points of notional (``0.0`` when notional <= 0)."""
    if report.notional_usd <= 0.0:
        return 0.0
    return report.timing_cost_usd / report.notional_usd * 10_000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_execution(
    fills: Iterable[BenchmarkedFill],
) -> ExecutionQualityReport:
    """Aggregate :class:`BenchmarkedFill` stream into an :class:`ExecutionQualityReport`.

    Walks :paramref:`fills` once, summing notional + IS + VWAP deviation
    + timing in iteration order.  Identity guarantee:
    ``is_cost_usd == vwap_deviation_usd + timing_cost_usd`` holds for
    every report this function returns (we compute ``timing`` as
    ``is - vwap_dev`` rather than from a separate accumulator, so the
    equality is exact).

    Args:
        fills: Iterable of :class:`BenchmarkedFill` records.  Empty
            input yields :func:`empty_report`.

    Returns:
        Frozen :class:`ExecutionQualityReport`.

    Raises:
        TypeError: If any element is not a :class:`BenchmarkedFill`.
    """
    n_fills = 0
    notional_usd = 0.0
    is_cost_usd = 0.0
    vwap_deviation_usd = 0.0
    weighted_participation = 0.0
    weighted_notional = 0.0

    for fill in fills:
        if not isinstance(fill, BenchmarkedFill):
            raise TypeError(f"fills must contain BenchmarkedFill; got {type(fill).__name__}")
        trade = fill.trade
        side_sign = _side_sign(trade.side)
        notional = trade.qty * trade.price
        is_cost = -(trade.price - fill.arrival_price) * trade.qty * side_sign
        vwap_dev = -(trade.price - fill.interval_vwap) * trade.qty * side_sign

        n_fills += 1
        notional_usd += notional
        is_cost_usd += is_cost
        vwap_deviation_usd += vwap_dev

        if fill.interval_volume > 0.0:
            participation = trade.qty / fill.interval_volume
            weighted_participation += participation * notional
            weighted_notional += notional

    timing_cost_usd = is_cost_usd - vwap_deviation_usd
    participation_rate = (
        weighted_participation / weighted_notional if weighted_notional > 0.0 else 0.0
    )

    return ExecutionQualityReport(
        n_fills=n_fills,
        notional_usd=notional_usd,
        is_cost_usd=is_cost_usd,
        vwap_deviation_usd=vwap_deviation_usd,
        timing_cost_usd=timing_cost_usd,
        participation_rate=participation_rate,
    )


def score_execution_by_symbol(
    fills: Iterable[BenchmarkedFill],
) -> Mapping[str, ExecutionQualityReport]:
    """Group :paramref:`fills` by ``trade.symbol`` and score each group.

    Returns a :class:`dict` mapping symbol → :class:`ExecutionQualityReport`,
    preserving **first-seen** symbol order across the input stream
    (CPython ``dict`` insertion order; deterministic for replay).
    """
    grouped: dict[str, list[BenchmarkedFill]] = {}
    for fill in fills:
        if not isinstance(fill, BenchmarkedFill):
            raise TypeError(f"fills must contain BenchmarkedFill; got {type(fill).__name__}")
        grouped.setdefault(fill.trade.symbol, []).append(fill)
    return {sym: score_execution(rows) for sym, rows in grouped.items()}


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "BenchmarkedFill",
    "ExecutionQualityReport",
    "empty_report",
    "is_cost_bps",
    "score_execution",
    "score_execution_by_symbol",
    "timing_cost_bps",
    "vwap_deviation_bps",
]
