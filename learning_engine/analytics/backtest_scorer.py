# ADAPTED FROM: polakowo/vectorbt (Fair Source 100 License — non-commercial).
# This module deliberately does NOT import vectorbt and does NOT copy
# vectorbt source code; the canonical performance ratios it computes
# (Sharpe, Sortino, Calmar, max-drawdown, win rate, profit factor,
# CAGR) are *textbook* formulas that pre-date vectorbt by decades —
# Sharpe (1966), Sortino (1980), Young (1991) for Calmar, Magdon-Ismail
# (2004) for the closed-form expected max-drawdown estimate. The only
# thing inherited from vectorbt is the *contract shape*: a single
# value-object ``BacktestScore`` aggregating the canonical ratio set so
# downstream learning-loop callers see one stable schema regardless of
# which scoring backend is wired in.
#
# Per the canonical spec (DIX_MASTER_CANONICAL.md B-11) the Fair Source
# license caveat is surfaced in the PR description; operator approval
# is required before any *runtime* use of vectorbt itself. This module
# is OFFLINE-ONLY and clean-room with respect to vectorbt source code.
"""Backtest scoring — OFFLINE batch tier.

Adapts the *contract shape* of ``vectorbt.portfolio.base.Portfolio.stats()``
(Sharpe / Sortino / Calmar / max-drawdown / win rate / total return /
profit factor / CAGR / volatility) into a single frozen, slotted
:class:`BacktestScore` value object that downstream learning-loop
callers consume.

The implementation is pure stdlib — no ``vectorbt``, no ``pandas``, no
``numpy``. All formulas are textbook:

* Sharpe ratio — Sharpe (1966):
  ``mean(excess_returns) / std(excess_returns) * sqrt(periods_per_year)``
* Sortino ratio — Sortino & Price (1994):
  ``mean(excess_returns) / std(downside_returns) * sqrt(periods_per_year)``
* Max drawdown — peak-to-trough on the equity curve:
  ``max((peak - trough) / peak)`` over time
* Calmar ratio — Young (1991):
  ``annualised_return / max_drawdown``
* Win rate, profit factor, CAGR, total return, volatility — standard
  finance definitions.

Tier
----
**OFFLINE_ONLY.** ``learning_engine/analytics/`` is the slow-cadence
batch analytics tier. This module is never imported from
``execution_engine/``, ``governance_engine/``, ``system_engine/``,
``core/``, or ``intelligence_engine/meta_controller/hot_path.py``.

Design constraints
------------------
* **No new pip deps.** :data:`NEW_PIP_DEPENDENCIES` is the empty tuple.
* **No clock / IO / PRNG / asyncio.** All inputs are caller-supplied
  ``BacktestResult`` instances; no wall-clock reads, no file IO, no
  network IO.
* **B27 / B28 / INV-71 authority symmetry.** Module never constructs
  ``SignalEvent`` / ``ExecutionIntent`` / ``HazardEvent`` /
  ``GovernanceDecision`` / ``PatchProposal``. Output is a frozen
  advisory value object.
* **B1 isolation.** No ``execution_engine`` / ``governance_engine`` /
  ``system_engine`` / ``evolution_engine`` / ``intelligence_engine``
  imports.
* **INV-15 byte-identical replay.** Returns are computed
  deterministically from the caller-supplied :class:`BacktestResult`;
  trade rows are sorted by ``(ts_ns, symbol, order_id)`` before any
  reduction; pinned by 3-run equality test.

Canonical contract shape (mirrors ``vectorbt.Portfolio.stats()``)
-----------------------------------------------------------------
``BacktestScore`` fields:

* ``total_return``      — decimal (``0.15`` == +15 %)
* ``cagr``              — decimal annualised return
* ``volatility``        — annualised stddev of period returns
* ``sharpe``            — annualised
* ``sortino``           — annualised
* ``calmar``            — ``cagr / max_drawdown`` (0.0 if ``max_drawdown == 0``)
* ``max_drawdown``      — absolute, in ``[0, 1]``
* ``win_rate``          — in ``[0, 1]``
* ``profit_factor``     — gross profit / abs(gross loss); ``inf`` if no losses
* ``n_trades``          — int
* ``avg_trade_pnl``     — average per-trade USD pnl
* ``best_trade_pnl``    — max per-trade USD pnl
* ``worst_trade_pnl``   — min per-trade USD pnl
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from core.contracts.backtest_result import BacktestResult

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""B-11 introduces **no** new pip dependencies.

The Fair Source 100 license on ``vectorbt`` is incompatible with
default DIX runtime licensing; we implement the canonical performance
ratios from textbook formulas instead.
"""

# Calendar conventions for annualisation. Callers may override
# ``periods_per_year`` at score() call time; these are sensible defaults.
DAILY_PERIODS_PER_YEAR: Final[int] = 252
HOURLY_PERIODS_PER_YEAR: Final[int] = 252 * 24
MINUTE_PERIODS_PER_YEAR: Final[int] = 252 * 24 * 60

# Soft cap on the size of any single batch to keep heap pressure
# bounded for offline jobs.
MAX_TRADES_PER_BATCH: Final[int] = 1_000_000
MAX_EQUITY_POINTS_PER_BATCH: Final[int] = 1_000_000

# Near-zero stddev threshold. IEEE-754 round-off on equity-curve
# division produces residual stddev around 1e-15 even for *exactly*
# constant period returns (e.g. 1.10 / 1.00 -> 1.10000000000000009).
# Below this threshold we treat the dispersion as zero so Sharpe /
# Sortino collapse to zero rather than blowing up to ~1e16.
_STDDEV_EPSILON: Final[float] = 1e-12


class BacktestScorerError(ValueError):
    """Raised on any malformed input to :func:`score_backtest`."""


@dataclass(frozen=True, slots=True)
class BacktestScore:
    """Aggregate score for a single backtest.

    Mirrors the contract shape of ``vectorbt.Portfolio.stats()`` so
    downstream consumers see a single stable schema regardless of which
    backtest engine produced the source rows.

    All fields are finite ``float`` / ``int``; ``profit_factor`` may be
    ``math.inf`` when the backtest never realised a losing trade. The
    score is **advisory** — projecting it into the learning loop must
    happen through the existing ``project_to_trade_outcomes`` /
    ``UpdateProposal`` machinery, never directly into governance.
    """

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float

    def __post_init__(self) -> None:
        for name in (
            "total_return",
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "win_rate",
            "avg_trade_pnl",
            "best_trade_pnl",
            "worst_trade_pnl",
        ):
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be float; got {type(value)!r}")
            if not (math.isfinite(value) or (name == "profit_factor")):
                raise BacktestScorerError(f"{name} must be finite; got {value!r}")
        if not isinstance(self.profit_factor, float):
            raise TypeError(f"profit_factor must be float; got {type(self.profit_factor)!r}")
        if math.isnan(self.profit_factor):
            raise BacktestScorerError("profit_factor must not be NaN")
        if self.profit_factor < 0.0:
            raise BacktestScorerError(f"profit_factor must be >= 0; got {self.profit_factor}")
        if not isinstance(self.n_trades, int) or isinstance(self.n_trades, bool):
            raise TypeError(f"n_trades must be int; got {type(self.n_trades)!r}")
        if self.n_trades < 0:
            raise BacktestScorerError(f"n_trades must be >= 0; got {self.n_trades}")
        if not (0.0 <= self.win_rate <= 1.0):
            raise BacktestScorerError(f"win_rate must be in [0, 1]; got {self.win_rate}")
        if not (0.0 <= self.max_drawdown <= 1.0):
            raise BacktestScorerError(f"max_drawdown must be in [0, 1]; got {self.max_drawdown}")
        if self.volatility < 0.0:
            raise BacktestScorerError(f"volatility must be >= 0; got {self.volatility}")


def _equity_returns(equity: tuple[float, ...]) -> tuple[float, ...]:
    """Simple period returns from an equity curve.

    ``r_i = equity_i / equity_{i-1} - 1``. Returns an empty tuple when
    ``len(equity) < 2``.
    """

    if len(equity) < 2:
        return ()
    out: list[float] = []
    prev = equity[0]
    if prev <= 0.0:
        raise BacktestScorerError(f"equity_curve must be strictly positive; got {prev}")
    for value in equity[1:]:
        if value <= 0.0:
            raise BacktestScorerError(f"equity_curve must be strictly positive; got {value}")
        out.append(value / prev - 1.0)
        prev = value
    return tuple(out)


def _mean(xs: tuple[float, ...]) -> float:
    if not xs:
        return 0.0
    return math.fsum(xs) / len(xs)


def _stddev(xs: tuple[float, ...], mean: float | None = None) -> float:
    """Sample standard deviation (ddof=1).

    Returns ``0.0`` when ``len(xs) < 2`` — matches the conservative
    convention used by ``vectorbt`` and ``pandas`` for short series.
    """

    if len(xs) < 2:
        return 0.0
    mu = _mean(xs) if mean is None else mean
    deviations = tuple((x - mu) ** 2 for x in xs)
    return math.sqrt(math.fsum(deviations) / (len(xs) - 1))


def _downside_stddev(xs: tuple[float, ...], target: float) -> float:
    """Sample stddev of returns *below* the target (Sortino denominator).

    Following Sortino & Price (1994): only deviations below the target
    contribute, but the denominator is the full count ``len(xs) - 1``
    (not just the count of below-target observations) so the metric
    stays comparable across different downside frequencies.
    """

    if len(xs) < 2:
        return 0.0
    below = tuple(min(0.0, x - target) ** 2 for x in xs)
    return math.sqrt(math.fsum(below) / (len(xs) - 1))


def _max_drawdown(equity: tuple[float, ...]) -> float:
    """Max drawdown ``(peak - trough) / peak`` on a pre-sorted curve.

    Returns a value in ``[0.0, 1.0]``; ``0.0`` for a flat or rising curve.
    """

    if not equity:
        return 0.0
    peak = equity[0]
    if peak <= 0.0:
        raise BacktestScorerError(f"equity_curve must be strictly positive; got {peak}")
    worst = 0.0
    for value in equity[1:]:
        if value > peak:
            peak = value
            continue
        dd = (peak - value) / peak
        if dd > worst:
            worst = dd
    if worst > 1.0:
        # Defensive clamp — equity strictly positive guarantees dd <= 1.0.
        worst = 1.0
    return worst


def _cagr(
    initial_equity: float,
    final_equity: float,
    n_periods: int,
    periods_per_year: int,
) -> float:
    """Compound annual growth rate.

    ``(final / initial) ** (periods_per_year / n_periods) - 1``.
    Returns ``0.0`` when there are zero periods or the curve is flat.
    """

    if n_periods <= 0:
        return 0.0
    if initial_equity <= 0.0:
        raise BacktestScorerError(f"initial_equity must be > 0; got {initial_equity}")
    if final_equity <= 0.0:
        return -1.0
    growth = final_equity / initial_equity
    exponent = periods_per_year / n_periods
    return growth**exponent - 1.0


def _sorted_trade_pnls(result: BacktestResult) -> tuple[float, ...]:
    """Per-trade pnl sorted by ``(ts_ns, symbol, order_id)`` for determinism."""

    keyed = sorted(
        result.trades,
        key=lambda t: (t.ts_ns, t.symbol, t.order_id),
    )
    return tuple(float(t.pnl_usd) for t in keyed)


def _equity_curve_tuple(result: BacktestResult) -> tuple[float, ...]:
    """Equity-curve values in monotone-ascending ts_ns order.

    ``BacktestResult.__post_init__`` already enforces ts ordering, so we
    project straight through. Returns an empty tuple when there is no
    equity curve, in which case Sharpe / Sortino / Calmar collapse to 0.
    """

    return tuple(float(pt.equity_usd) for pt in result.equity_curve)


def score_backtest(
    result: BacktestResult,
    *,
    risk_free_rate_per_period: float = 0.0,
    periods_per_year: int = DAILY_PERIODS_PER_YEAR,
) -> BacktestScore:
    """Score *result* into a frozen :class:`BacktestScore`.

    The function is **pure**: same input always yields a byte-identical
    output. ``risk_free_rate_per_period`` is the per-period (not
    annualised) risk-free rate used to compute excess returns;
    ``periods_per_year`` controls the annualisation factor on Sharpe /
    Sortino / Calmar / volatility.

    Args:
        result: Fully validated :class:`BacktestResult` payload.
        risk_free_rate_per_period: Per-period (i.e. matching the equity
            curve cadence) risk-free rate as a decimal. Defaults to
            ``0.0`` so callers without a benchmark see the raw return /
            volatility ratio.
        periods_per_year: Annualisation factor — ``252`` for daily,
            ``252 * 24`` for hourly, ``252 * 24 * 60`` for minute bars,
            etc. Must be > 0.

    Returns:
        :class:`BacktestScore` aggregating the textbook performance ratios.

    Raises:
        BacktestScorerError: When the equity curve is empty, contains
            non-positive values, exceeds :data:`MAX_EQUITY_POINTS_PER_BATCH`,
            or when ``periods_per_year`` is non-positive.
        TypeError: When ``result`` is not a :class:`BacktestResult`.
    """

    if not isinstance(result, BacktestResult):
        raise TypeError(f"result must be BacktestResult; got {type(result)!r}")
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool):
        raise TypeError(f"periods_per_year must be int; got {type(periods_per_year)!r}")
    if periods_per_year <= 0:
        raise BacktestScorerError(f"periods_per_year must be > 0; got {periods_per_year}")
    if not isinstance(risk_free_rate_per_period, float):
        raise TypeError(
            f"risk_free_rate_per_period must be float; got {type(risk_free_rate_per_period)!r}"
        )
    if not math.isfinite(risk_free_rate_per_period):
        raise BacktestScorerError(
            f"risk_free_rate_per_period must be finite; got {risk_free_rate_per_period}"
        )
    if len(result.trades) > MAX_TRADES_PER_BATCH:
        raise BacktestScorerError(f"too many trades: {len(result.trades)} > {MAX_TRADES_PER_BATCH}")
    if len(result.equity_curve) > MAX_EQUITY_POINTS_PER_BATCH:
        raise BacktestScorerError(
            f"too many equity points: {len(result.equity_curve)} > {MAX_EQUITY_POINTS_PER_BATCH}"
        )

    equity = _equity_curve_tuple(result)
    trade_pnls = _sorted_trade_pnls(result)
    n_trades = len(trade_pnls)

    # Equity-side metrics.
    if equity:
        returns = _equity_returns(equity)
        excess = tuple(r - risk_free_rate_per_period for r in returns)
        mu_excess = _mean(excess)
        sigma_excess = _stddev(excess, mean=mu_excess)
        downside = _downside_stddev(excess, target=0.0)
        ann_factor = math.sqrt(periods_per_year)
        sharpe = (mu_excess / sigma_excess) * ann_factor if sigma_excess > _STDDEV_EPSILON else 0.0
        sortino = (mu_excess / downside) * ann_factor if downside > _STDDEV_EPSILON else 0.0
        raw_vol = _stddev(returns) * ann_factor
        volatility = raw_vol if raw_vol > _STDDEV_EPSILON else 0.0
        max_dd = _max_drawdown(equity)
        total_return = equity[-1] / equity[0] - 1.0
        cagr = _cagr(
            initial_equity=equity[0],
            final_equity=equity[-1],
            n_periods=len(returns),
            periods_per_year=periods_per_year,
        )
        calmar = cagr / max_dd if max_dd > 0.0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0
        volatility = 0.0
        max_dd = 0.0
        total_return = 0.0
        cagr = 0.0
        calmar = 0.0

    # Trade-side metrics.
    if n_trades > 0:
        wins = sum(1 for p in trade_pnls if p > 0.0)
        win_rate = wins / n_trades
        gross_profit = math.fsum(p for p in trade_pnls if p > 0.0)
        gross_loss = math.fsum(-p for p in trade_pnls if p < 0.0)
        if gross_loss > 0.0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0.0:
            profit_factor = math.inf
        else:
            profit_factor = 0.0
        avg_trade = math.fsum(trade_pnls) / n_trades
        best_trade = max(trade_pnls)
        worst_trade = min(trade_pnls)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
        best_trade = 0.0
        worst_trade = 0.0

    return BacktestScore(
        total_return=float(total_return),
        cagr=float(cagr),
        volatility=float(volatility),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(max_dd),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        n_trades=int(n_trades),
        avg_trade_pnl=float(avg_trade),
        best_trade_pnl=float(best_trade),
        worst_trade_pnl=float(worst_trade),
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "DAILY_PERIODS_PER_YEAR",
    "HOURLY_PERIODS_PER_YEAR",
    "MINUTE_PERIODS_PER_YEAR",
    "MAX_TRADES_PER_BATCH",
    "MAX_EQUITY_POINTS_PER_BATCH",
    "BacktestScorerError",
    "BacktestScore",
    "score_backtest",
)
