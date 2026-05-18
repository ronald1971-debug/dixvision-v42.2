# ADAPTED FROM: pola-rs/polars py-polars/polars/lazyframe/frame.py
# (LazyFrame.with_columns / group_by / agg / sort / collect lazy-API
#  pattern; py-polars/polars/expr/expr.py — pl.col / pl.lit / pl.when
#  expression patterns; py-polars/polars/dataframe/frame.py —
#  iter_rows(named=True) materialisation.)
"""Polars LazyFrame-backed PnL attribution — OFFLINE batch tier.

This module is the **first** of the S-10 polars triple
(``pnl_attribution.py`` + ``regime_stats.py`` + ``feature_importance.py``);
it adapts polars' lazy ``with_columns -> group_by -> agg -> sort ->
collect`` pattern into DIX as a high-throughput batch reproduction of
the per-trade decomposition shipped under
``learning_engine/performance_analysis/pnl_attribution.py`` (S-04.1).

The S-04.1 module is the **canonical reference** — pure stdlib, one
:class:`AttributedTrade` at a time. This module is the **batch
companion**: same decomposition (``signal + slippage + fee ==
realised``), but column-oriented and groupable by symbol.

Tier
----
**OFFLINE_ONLY.** ``learning_engine/analytics/`` is the high-throughput
slow-cadence analytics tier. Polars must never be imported from
``execution_engine/``, ``governance_engine/``, ``system_engine/``,
``core/``, or ``intelligence_engine/meta_controller/hot_path.py`` —
that ban is enforced in ``tools/authority_lint.py`` (S-10 lint rule
ships in a follow-up sub-PR).

Design constraints
------------------
* **Lazy import.** ``import polars`` lives **inside** the function that
  needs it so this module imports cleanly in environments without
  ``polars`` installed (mirrors the S-01 ccxt and S-05 firecrawl
  precedents).
* **Pure data.** No wall-clock reads, no IO, no global mutable state,
  no PRNG. Polars is asked for an
  **eager** ``collect()`` each call — there is no shared session, no
  connection pool, and the resulting :class:`PolarsPnLReport` is
  fully serialisable (frozen + slotted).
* **Frozen contracts.** :class:`TradeRow`, :class:`SymbolAttribution`,
  and :class:`PolarsPnLReport` are ``@dataclass(frozen=True,
  slots=True)`` with eager validation in ``__post_init__``.
* **INV-15 byte-identical.** Inputs are sorted by ``(symbol, ts_ns,
  fill_price, qty)`` *before* the LazyFrame is constructed; the
  group-by output is post-sorted by ``symbol``; per-row aggregation
  uses polars' ``sum`` reduction (associative, but determinism is
  guaranteed by the explicit ``sort`` and by the absence of any
  parallel/streaming flag at ``collect`` time). Pinned by a 3-run
  byte-equality test.
* **No new pip deps in module-import time.** :data:`NEW_PIP_DEPENDENCIES`
  declares ``("polars",)`` so pip-dep audit picks it up, but the
  module body never imports polars at toplevel; calling
  :func:`attribute_pnl_polars` without polars installed raises a
  clean ``ImportError`` with an actionable hint.

Algorithmic summary
-------------------
For each trade row::

    side_sign      = +1 if side == "BUY" else -1
    notional       = qty * fill_price
    slippage_pnl   = -(fill_price - signal_price) * qty * side_sign
    fee_pnl        = -fee_usd
    signal_pnl     = realised_pnl - slippage_pnl - fee_pnl

The aggregator preserves the identity ``signal_pnl + slippage_pnl +
fee_pnl == realised_pnl`` per symbol and overall, within the bounds of
IEEE-754 floating-point summation (enforced by the test suite using
``math.fsum``-style equality up to a strict tolerance).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:  # pragma: no cover — imports for type checkers only
    pass

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("polars",)
"""S-10.1 introduces a single new pip dep: ``polars``.

Polars is **lazy-imported inside** :func:`attribute_pnl_polars` so this
module imports cleanly without it. ``tools/authority_lint.py`` will ban
``import polars`` from RUNTIME tiers in the S-10.4 follow-up sub-PR.
"""

_VALID_SIDES: Final[frozenset[str]] = frozenset({"BUY", "SELL"})


@dataclasses.dataclass(frozen=True, slots=True)
class TradeRow:
    """One realised trade row, structured for polars columnar batch.

    Mirrors :class:`learning_engine.performance_analysis.pnl_attribution.AttributedTrade`
    (S-04.1) but flattens the nested ``BacktestTrade`` into bare scalar
    fields so polars can construct a typed columnar LazyFrame in one
    pass without per-row attribute lookups.

    Args:
        ts_ns: Trade timestamp (ns since epoch). Must be ``>= 0``.
        symbol: Instrument symbol (non-empty, lowercase recommended).
        side: ``"BUY"`` or ``"SELL"``.
        qty: Filled quantity (``>= 0``, NaN-rejecting).
        fill_price: Realised fill price (``> 0``).
        signal_price: Strategy-side expected price at signal time
            (``> 0``).
        pnl_usd: Realised PnL in USD (signed).
        fee_usd: Fee paid in USD (``>= 0``).
    """

    ts_ns: int
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: float
    fill_price: float
    signal_price: float
    pnl_usd: float
    fee_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"ts_ns must be int; got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty str")
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side must be 'BUY' or 'SELL'; got {self.side!r}")
        for name in ("qty", "fill_price", "signal_price", "pnl_usd", "fee_usd"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be float; got {type(v).__name__}")
            if v != v:  # NaN check (IEEE-754)
                raise ValueError(f"{name} must not be NaN")
        if self.qty < 0.0:
            raise ValueError(f"qty must be >= 0; got {self.qty}")
        if self.fill_price <= 0.0:
            raise ValueError(f"fill_price must be > 0; got {self.fill_price}")
        if self.signal_price <= 0.0:
            raise ValueError(f"signal_price must be > 0; got {self.signal_price}")
        if self.fee_usd < 0.0:
            raise ValueError(f"fee_usd must be >= 0; got {self.fee_usd}")


@dataclasses.dataclass(frozen=True, slots=True)
class SymbolAttribution:
    """Per-symbol PnL attribution row from polars ``group_by`` aggregation.

    Invariant: ``signal_pnl_usd + slippage_pnl_usd + fee_pnl_usd ==
    realised_pnl_usd`` (within IEEE-754 floating-point tolerance).
    """

    symbol: str
    n_trades: int
    notional_usd: float
    realised_pnl_usd: float
    signal_pnl_usd: float
    slippage_pnl_usd: float
    fee_pnl_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty str")
        if not isinstance(self.n_trades, int) or isinstance(self.n_trades, bool):
            raise TypeError(f"n_trades must be int; got {type(self.n_trades).__name__}")
        if self.n_trades < 0:
            raise ValueError(f"n_trades must be >= 0; got {self.n_trades}")
        if self.notional_usd < 0.0:
            raise ValueError(f"notional_usd must be >= 0; got {self.notional_usd}")


@dataclasses.dataclass(frozen=True, slots=True)
class PolarsPnLReport:
    """Aggregate output of polars-backed PnL attribution.

    Args:
        by_symbol: Per-symbol attribution rows, sorted ascending by
            ``symbol`` for INV-15 byte-stable replay.
        total_n_trades: Sum of ``n_trades`` across symbols.
        total_notional_usd: Sum of ``notional_usd`` across symbols.
        total_realised_pnl_usd: Sum of ``realised_pnl_usd`` across symbols.
        total_signal_pnl_usd: Sum of ``signal_pnl_usd`` across symbols.
        total_slippage_pnl_usd: Sum of ``slippage_pnl_usd`` across symbols.
        total_fee_pnl_usd: Sum of ``fee_pnl_usd`` across symbols.
    """

    by_symbol: tuple[SymbolAttribution, ...]
    total_n_trades: int
    total_notional_usd: float
    total_realised_pnl_usd: float
    total_signal_pnl_usd: float
    total_slippage_pnl_usd: float
    total_fee_pnl_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.by_symbol, tuple):
            raise TypeError(f"by_symbol must be tuple; got {type(self.by_symbol).__name__}")
        for r in self.by_symbol:
            if not isinstance(r, SymbolAttribution):
                raise TypeError(
                    f"by_symbol entries must be SymbolAttribution; got {type(r).__name__}"
                )
        names = [r.symbol for r in self.by_symbol]
        if names != sorted(names):
            raise ValueError(
                "by_symbol must be sorted ascending by symbol for INV-15 replay determinism"
            )
        if len(set(names)) != len(names):
            raise ValueError("by_symbol must contain unique symbols")
        if not isinstance(self.total_n_trades, int) or isinstance(self.total_n_trades, bool):
            raise TypeError(f"total_n_trades must be int; got {type(self.total_n_trades).__name__}")
        if self.total_n_trades < 0:
            raise ValueError(f"total_n_trades must be >= 0; got {self.total_n_trades}")
        if self.total_notional_usd < 0.0:
            raise ValueError(f"total_notional_usd must be >= 0; got {self.total_notional_usd}")


def attribute_pnl_polars(
    trades: Sequence[TradeRow],
) -> PolarsPnLReport:
    """Compute per-symbol PnL attribution using a polars LazyFrame.

    The decomposition matches S-04.1 exactly (signal + slippage + fee
    == realised). The polars LazyFrame is materialised once via
    ``collect()`` per call and then dropped — there is no shared
    polars session, no streaming flag, and no concurrency, so
    floating-point summation is deterministic across runs.

    Args:
        trades: Iterable of :class:`TradeRow` records. May be empty.

    Returns:
        :class:`PolarsPnLReport` with per-symbol breakdown sorted
        ascending by ``symbol``.

    Raises:
        ImportError: if ``polars`` is not installed (lazy import).
        TypeError: if ``trades`` contains a non-:class:`TradeRow`.
    """
    try:
        import polars as pl  # noqa: PLC0415 — lazy import is intentional
    except ImportError as exc:  # pragma: no cover — env-specific
        raise ImportError(
            "polars is required for learning_engine.analytics.pnl_attribution; "
            "install it with `pip install polars` (S-10 OFFLINE_ONLY tier)"
        ) from exc

    rows = tuple(trades)
    for r in rows:
        if not isinstance(r, TradeRow):
            raise TypeError(f"trades entries must be TradeRow; got {type(r).__name__}")

    if not rows:
        return PolarsPnLReport(
            by_symbol=(),
            total_n_trades=0,
            total_notional_usd=0.0,
            total_realised_pnl_usd=0.0,
            total_signal_pnl_usd=0.0,
            total_slippage_pnl_usd=0.0,
            total_fee_pnl_usd=0.0,
        )

    # Sort inputs deterministically before building the LazyFrame.
    sorted_rows = sorted(rows, key=lambda t: (t.symbol, t.ts_ns, t.fill_price, t.qty, t.side))

    lf = pl.LazyFrame(
        {
            "symbol": [t.symbol for t in sorted_rows],
            "side": [t.side for t in sorted_rows],
            "qty": [float(t.qty) for t in sorted_rows],
            "fill_price": [float(t.fill_price) for t in sorted_rows],
            "signal_price": [float(t.signal_price) for t in sorted_rows],
            "pnl_usd": [float(t.pnl_usd) for t in sorted_rows],
            "fee_usd": [float(t.fee_usd) for t in sorted_rows],
        }
    )

    lf = (
        lf.with_columns(
            [
                (pl.col("qty") * pl.col("fill_price")).alias("notional"),
                pl.when(pl.col("side") == "BUY")
                .then(pl.lit(1.0))
                .otherwise(pl.lit(-1.0))
                .alias("side_sign"),
            ]
        )
        .with_columns(
            [
                (
                    -(
                        (pl.col("fill_price") - pl.col("signal_price"))
                        * pl.col("qty")
                        * pl.col("side_sign")
                    )
                ).alias("slippage_pnl"),
                (-pl.col("fee_usd")).alias("fee_pnl"),
            ]
        )
        .with_columns(
            [
                (pl.col("pnl_usd") - pl.col("slippage_pnl") - pl.col("fee_pnl")).alias(
                    "signal_pnl"
                ),
            ]
        )
    )

    by_symbol_lf = (
        lf.group_by("symbol")
        .agg(
            [
                pl.len().alias("n_trades"),
                pl.col("notional").sum().alias("notional_usd"),
                pl.col("pnl_usd").sum().alias("realised_pnl_usd"),
                pl.col("signal_pnl").sum().alias("signal_pnl_usd"),
                pl.col("slippage_pnl").sum().alias("slippage_pnl_usd"),
                pl.col("fee_pnl").sum().alias("fee_pnl_usd"),
            ]
        )
        .sort("symbol")
    )

    df = by_symbol_lf.collect()

    by_symbol_rows: list[SymbolAttribution] = []
    for r in df.iter_rows(named=True):
        by_symbol_rows.append(
            SymbolAttribution(
                symbol=str(r["symbol"]),
                n_trades=int(r["n_trades"]),
                notional_usd=float(r["notional_usd"]),
                realised_pnl_usd=float(r["realised_pnl_usd"]),
                signal_pnl_usd=float(r["signal_pnl_usd"]),
                slippage_pnl_usd=float(r["slippage_pnl_usd"]),
                fee_pnl_usd=float(r["fee_pnl_usd"]),
            )
        )

    return PolarsPnLReport(
        by_symbol=tuple(by_symbol_rows),
        total_n_trades=sum(r.n_trades for r in by_symbol_rows),
        total_notional_usd=math.fsum(r.notional_usd for r in by_symbol_rows),
        total_realised_pnl_usd=math.fsum(r.realised_pnl_usd for r in by_symbol_rows),
        total_signal_pnl_usd=math.fsum(r.signal_pnl_usd for r in by_symbol_rows),
        total_slippage_pnl_usd=math.fsum(r.slippage_pnl_usd for r in by_symbol_rows),
        total_fee_pnl_usd=math.fsum(r.fee_pnl_usd for r in by_symbol_rows),
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "PolarsPnLReport",
    "SymbolAttribution",
    "TradeRow",
    "attribute_pnl_polars",
)
