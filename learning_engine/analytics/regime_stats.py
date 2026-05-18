# ADAPTED FROM: pola-rs/polars py-polars/polars/lazyframe/frame.py
# (LazyFrame.group_by / agg / sort / collect lazy-API pattern;
#  py-polars/polars/expr/expr.py — pl.col / pl.when / pl.lit / pl.len /
#  pl.sum / pl.std expression patterns; py-polars/polars/dataframe/frame.py
#  — iter_rows(named=True) materialisation.)
"""Polars LazyFrame-backed per-regime trade statistics — OFFLINE batch tier.

This module is the **second** of the S-10 polars triple
(``pnl_attribution.py`` + ``regime_stats.py`` + ``feature_importance.py``).
It adapts polars' lazy ``group_by("regime") -> agg([...]) -> sort ->
collect`` pattern into DIX as a high-throughput aggregator that
breaks down a stream of realised trades by macro regime
(``UNKNOWN`` / ``RISK_ON`` / ``NEUTRAL`` / ``RISK_OFF`` / ``CRISIS``).

The output is a frozen :class:`RegimeStatsReport` with one
:class:`RegimeStats` row per regime: trade count, winner count, win-rate,
total PnL, average PnL per trade, population standard deviation of PnL
(volatility proxy), total fees paid, and total traded notional.

Tier
----
**OFFLINE_ONLY.** ``learning_engine/analytics/`` is the high-throughput
slow-cadence analytics tier. Polars must never be imported from
``execution_engine/``, ``governance_engine/``, ``system_engine/``,
``core/``, or ``intelligence_engine/meta_controller/hot_path.py`` —
that ban will be enforced in ``tools/authority_lint.py`` (S-10.4 lint
rule, follow-up sub-PR).

Design constraints
------------------
* **Lazy import.** ``import polars`` lives **inside**
  :func:`compute_regime_stats` so this module imports cleanly in
  environments without ``polars`` installed (mirrors S-10.1 and the
  S-01 ccxt / S-05 firecrawl precedents).
* **Pure data.** No wall-clock reads, no IO, no global mutable state,
  no PRNG. Polars is asked for an **eager** ``collect()`` each call —
  there is no shared session, no streaming flag, and the resulting
  :class:`RegimeStatsReport` is fully serialisable (frozen + slotted).
* **Frozen contracts.** :class:`RegimeTradeRow`, :class:`RegimeStats`,
  and :class:`RegimeStatsReport` are ``@dataclass(frozen=True,
  slots=True)`` with eager validation in ``__post_init__``.
* **INV-15 byte-identical.** Inputs are sorted by ``(regime.value,
  ts_ns, symbol, fill_price, qty)`` *before* the LazyFrame is
  constructed; the group-by output is post-sorted by ``regime.value``;
  per-row aggregation uses polars' associative reductions, and
  determinism is guaranteed by the explicit ``sort`` + absence of any
  parallel/streaming flag at ``collect`` time. Pinned by a 3-run
  byte-equality + permutation-invariance test.
* **No new pip deps in module-import time.** :data:`NEW_PIP_DEPENDENCIES`
  declares ``("polars",)`` so pip-dep audit picks it up, but the
  module body never imports polars at toplevel; calling
  :func:`compute_regime_stats` without polars installed raises a
  clean ``ImportError`` with an actionable hint.

Algorithmic summary
-------------------
For each :class:`RegimeTradeRow`::

    notional   = qty * fill_price
    is_winner  = 1 if pnl_usd > 0 else 0

Per regime aggregation::

    n_trades       = count
    n_winners      = sum(is_winner)
    win_rate       = n_winners / n_trades       (0.0 if n_trades == 0)
    total_pnl_usd  = sum(pnl_usd)
    avg_pnl_usd    = total_pnl_usd / n_trades   (0.0 if n_trades == 0)
    pnl_std        = population std-dev of pnl  (0.0 if n_trades < 2)
    total_fee_usd  = sum(fee_usd)
    total_notional = sum(notional)
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Final

from core.contracts.macro_regime import MacroRegime

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("polars",)
"""S-10.2 reuses the ``polars`` pip dep declared by S-10.1.

Polars is **lazy-imported inside** :func:`compute_regime_stats` so this
module imports cleanly without it. ``tools/authority_lint.py`` will ban
``import polars`` from RUNTIME tiers in the S-10.4 follow-up sub-PR.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class RegimeTradeRow:
    """One realised trade row tagged with the macro regime that produced it.

    Args:
        ts_ns: Trade timestamp (ns since epoch). Must be ``>= 0``.
        symbol: Instrument symbol (non-empty).
        regime: :class:`MacroRegime` classification active at trade time.
        pnl_usd: Realised PnL in USD (signed; > 0 ⇒ winner).
        fee_usd: Fee paid in USD (``>= 0``).
        qty: Filled quantity (``>= 0``, NaN-rejecting).
        fill_price: Realised fill price (``> 0``).
    """

    ts_ns: int
    symbol: str
    regime: MacroRegime
    pnl_usd: float
    fee_usd: float
    qty: float
    fill_price: float

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"ts_ns must be int; got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty str")
        if not isinstance(self.regime, MacroRegime):
            raise TypeError(f"regime must be MacroRegime; got {type(self.regime).__name__}")
        for name in ("pnl_usd", "fee_usd", "qty", "fill_price"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be float; got {type(v).__name__}")
            if v != v:  # NaN check (IEEE-754)
                raise ValueError(f"{name} must not be NaN")
        if self.fee_usd < 0.0:
            raise ValueError(f"fee_usd must be >= 0; got {self.fee_usd}")
        if self.qty < 0.0:
            raise ValueError(f"qty must be >= 0; got {self.qty}")
        if self.fill_price <= 0.0:
            raise ValueError(f"fill_price must be > 0; got {self.fill_price}")


@dataclasses.dataclass(frozen=True, slots=True)
class RegimeStats:
    """Aggregate statistics for one macro regime."""

    regime: MacroRegime
    n_trades: int
    n_winners: int
    win_rate: float
    total_pnl_usd: float
    avg_pnl_usd: float
    pnl_std: float
    total_fee_usd: float
    total_notional_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.regime, MacroRegime):
            raise TypeError(f"regime must be MacroRegime; got {type(self.regime).__name__}")
        for name in ("n_trades", "n_winners"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"{name} must be int; got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"{name} must be >= 0; got {v}")
        if self.n_winners > self.n_trades:
            raise ValueError(f"n_winners ({self.n_winners}) must be <= n_trades ({self.n_trades})")
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(f"win_rate must be in [0, 1]; got {self.win_rate}")
        if self.pnl_std < 0.0:
            raise ValueError(f"pnl_std must be >= 0; got {self.pnl_std}")
        if self.total_fee_usd < 0.0:
            raise ValueError(f"total_fee_usd must be >= 0; got {self.total_fee_usd}")
        if self.total_notional_usd < 0.0:
            raise ValueError(f"total_notional_usd must be >= 0; got {self.total_notional_usd}")


@dataclasses.dataclass(frozen=True, slots=True)
class RegimeStatsReport:
    """Aggregate per-regime stats over a stream of realised trades.

    Args:
        by_regime: Per-regime stats rows, sorted ascending by
            ``regime.value`` for INV-15 byte-stable replay.
        total_n_trades: Sum of ``n_trades`` across regimes.
        overall_win_rate: Total winners / total trades (0.0 if empty).
        overall_total_pnl_usd: Sum of ``total_pnl_usd`` across regimes.
        overall_total_fee_usd: Sum of ``total_fee_usd`` across regimes.
        overall_total_notional_usd: Sum of ``total_notional_usd``.
    """

    by_regime: tuple[RegimeStats, ...]
    total_n_trades: int
    overall_win_rate: float
    overall_total_pnl_usd: float
    overall_total_fee_usd: float
    overall_total_notional_usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.by_regime, tuple):
            raise TypeError(f"by_regime must be tuple; got {type(self.by_regime).__name__}")
        for r in self.by_regime:
            if not isinstance(r, RegimeStats):
                raise TypeError(f"by_regime entries must be RegimeStats; got {type(r).__name__}")
        keys = [r.regime.value for r in self.by_regime]
        if keys != sorted(keys):
            raise ValueError(
                "by_regime must be sorted ascending by regime.value for INV-15 replay determinism"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("by_regime must contain unique regimes")
        if not isinstance(self.total_n_trades, int) or isinstance(self.total_n_trades, bool):
            raise TypeError(f"total_n_trades must be int; got {type(self.total_n_trades).__name__}")
        if self.total_n_trades < 0:
            raise ValueError(f"total_n_trades must be >= 0; got {self.total_n_trades}")
        if not (0.0 <= self.overall_win_rate <= 1.0):
            raise ValueError(f"overall_win_rate must be in [0, 1]; got {self.overall_win_rate}")
        if self.overall_total_fee_usd < 0.0:
            raise ValueError(
                f"overall_total_fee_usd must be >= 0; got {self.overall_total_fee_usd}"
            )
        if self.overall_total_notional_usd < 0.0:
            raise ValueError(
                f"overall_total_notional_usd must be >= 0; got {self.overall_total_notional_usd}"
            )


def compute_regime_stats(
    trades: Sequence[RegimeTradeRow],
) -> RegimeStatsReport:
    """Compute per-regime aggregate stats using a polars LazyFrame.

    The aggregation builds one :class:`RegimeStats` row per macro
    regime present in the input and post-sorts the result by
    ``regime.value`` for INV-15 byte-stable replay.

    Args:
        trades: Iterable of :class:`RegimeTradeRow` records. May be empty.

    Returns:
        :class:`RegimeStatsReport` with per-regime breakdown sorted
        ascending by ``regime.value``.

    Raises:
        ImportError: if ``polars`` is not installed (lazy import).
        TypeError: if ``trades`` contains a non-:class:`RegimeTradeRow`.
    """
    try:
        import polars as pl  # noqa: PLC0415 — lazy import is intentional
    except ImportError as exc:  # pragma: no cover — env-specific
        raise ImportError(
            "polars is required for learning_engine.analytics.regime_stats; "
            "install it with `pip install polars` (S-10 OFFLINE_ONLY tier)"
        ) from exc

    rows = tuple(trades)
    for r in rows:
        if not isinstance(r, RegimeTradeRow):
            raise TypeError(f"trades entries must be RegimeTradeRow; got {type(r).__name__}")

    if not rows:
        return RegimeStatsReport(
            by_regime=(),
            total_n_trades=0,
            overall_win_rate=0.0,
            overall_total_pnl_usd=0.0,
            overall_total_fee_usd=0.0,
            overall_total_notional_usd=0.0,
        )

    sorted_rows = sorted(
        rows,
        key=lambda t: (
            t.regime.value,
            t.ts_ns,
            t.symbol,
            t.fill_price,
            t.qty,
        ),
    )

    lf = pl.LazyFrame(
        {
            "regime": [t.regime.value for t in sorted_rows],
            "pnl_usd": [float(t.pnl_usd) for t in sorted_rows],
            "fee_usd": [float(t.fee_usd) for t in sorted_rows],
            "qty": [float(t.qty) for t in sorted_rows],
            "fill_price": [float(t.fill_price) for t in sorted_rows],
        }
    )

    lf = lf.with_columns(
        [
            (pl.col("qty") * pl.col("fill_price")).alias("notional"),
            pl.when(pl.col("pnl_usd") > 0.0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_winner"),
        ]
    )

    by_regime_lf = (
        lf.group_by("regime")
        .agg(
            [
                pl.len().alias("n_trades"),
                pl.col("is_winner").sum().alias("n_winners"),
                pl.col("pnl_usd").sum().alias("total_pnl_usd"),
                pl.col("pnl_usd").std(ddof=0).alias("pnl_std"),
                pl.col("fee_usd").sum().alias("total_fee_usd"),
                pl.col("notional").sum().alias("total_notional_usd"),
            ]
        )
        .sort("regime")
    )

    df = by_regime_lf.collect()

    by_regime_rows: list[RegimeStats] = []
    for r in df.iter_rows(named=True):
        n_trades = int(r["n_trades"])
        n_winners = int(r["n_winners"])
        total_pnl = float(r["total_pnl_usd"])
        # polars returns null for std on a single-element group
        std_raw = r["pnl_std"]
        pnl_std = 0.0 if std_raw is None else float(std_raw)
        win_rate = (n_winners / n_trades) if n_trades > 0 else 0.0
        avg_pnl = (total_pnl / n_trades) if n_trades > 0 else 0.0
        by_regime_rows.append(
            RegimeStats(
                regime=MacroRegime(r["regime"]),
                n_trades=n_trades,
                n_winners=n_winners,
                win_rate=win_rate,
                total_pnl_usd=total_pnl,
                avg_pnl_usd=avg_pnl,
                pnl_std=pnl_std,
                total_fee_usd=float(r["total_fee_usd"]),
                total_notional_usd=float(r["total_notional_usd"]),
            )
        )

    total_n_trades = sum(r.n_trades for r in by_regime_rows)
    total_winners = sum(r.n_winners for r in by_regime_rows)
    overall_win_rate = (total_winners / total_n_trades) if total_n_trades > 0 else 0.0

    return RegimeStatsReport(
        by_regime=tuple(by_regime_rows),
        total_n_trades=total_n_trades,
        overall_win_rate=overall_win_rate,
        overall_total_pnl_usd=math.fsum(r.total_pnl_usd for r in by_regime_rows),
        overall_total_fee_usd=math.fsum(r.total_fee_usd for r in by_regime_rows),
        overall_total_notional_usd=math.fsum(r.total_notional_usd for r in by_regime_rows),
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "RegimeStats",
    "RegimeStatsReport",
    "RegimeTradeRow",
    "compute_regime_stats",
)
