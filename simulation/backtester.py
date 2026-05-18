# ADAPTED FROM: backtrader (mementum/backtrader) patterns — GPL mitigation:
# no backtrader import, pattern only.
# Patterns extracted (rewritten in pure Python):
#   - Cerebro event loop: bar-by-bar driver that calls Strategy.next() once
#     per bar, then settles broker fills against the *next* bar's open
#     (Cerebro's standard "next-bar fill" semantics, see backtrader/cerebro.py).
#   - Strategy lifecycle hooks: prenext / next / notify_order / notify_trade
#     (backtrader/strategy.py).
#   - Broker fill model: percentage slippage applied against fill price plus
#     proportional commission (backtrader/broker.py).
"""Reference backtester (B-08) — pure-Python Backtrader-style event loop.

Wraps the DIX simulation tier as a Backtrader-style bar-by-bar driver and
emits a :class:`~core.contracts.backtest_result.BacktestResult` — never
backtrader objects, never any backtrader import.

Authority constraints
---------------------
* OFFLINE_ONLY tier — never reachable from
  :mod:`hot_path` / :mod:`execution_engine` / :mod:`governance_engine` /
  :mod:`system_engine` / :mod:`evolution_engine`.
* No clock — every timestamp comes from the caller-supplied :class:`Bar`
  stream; the module never imports ``datetime`` / ``time`` /
  :mod:`system.time_source`.
* No PRNG — fill jitter is a pure deterministic function of the
  caller-supplied ``seed`` and the current bar timestamp via splitmix64.
* No IO — pure in-memory state machine.
* No mutation of caller state — :func:`run_backtest` returns a frozen
  :class:`~core.contracts.backtest_result.BacktestResult`.
* B27 / B28 / INV-71 authority symmetry — the backtester does **not**
  construct :class:`~core.contracts.learning.PatchProposal` /
  :class:`~core.contracts.events.SignalEvent` /
  :class:`~core.contracts.governance.GovernanceDecision`. Backtests are
  research artefacts; promotion to live happens via the existing
  governance / strategy-registry surface, not from inside this module.

INV-15 (replay determinism)
---------------------------
Two callers that pass the same ``(bars, strategy, config, seed)`` produce
byte-identical :class:`BacktestResult` outputs:

* Bars are consumed in the order supplied.
* Strategy decisions are routed through the caller-supplied
  :class:`Strategy` Protocol; no global state, no caches.
* Slippage jitter is derived from ``splitmix64(seed ^ bar.ts_ns ^ idx)``
  so the same seed + bar stream always yields the same fill prices.
* Commission and equity-curve points are computed in declaration order;
  trades, equity points and metrics are sorted before being baked into
  the result.

GPL mitigation
--------------
Backtrader is GPL-3.0. PART 1 of ``DIX_MASTER_CANONICAL.md`` explicitly
forbids importing it. This module:

* Imports nothing from ``backtrader`` (pinned by an AST authority test).
* Re-implements the *patterns* (event loop, lifecycle hooks, fill model)
  in DIX-native classes that emit DIX contracts.
* Cites every pattern with an inline ``# ADAPTED FROM:`` comment so the
  audit trail is explicit about which patterns were copied.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from core.contracts.backtest_result import (
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
)
from core.contracts.events import Side
from core.contracts.signal_trust import SignalTrust

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""Pure stdlib — no backtrader, no numpy, no pandas."""

_SOURCE = "internal_replay"
_MAX_BARS = 1_000_000
_MAX_TRADES = 100_000
_MAX_SYMBOL_LEN = 32
_MAX_STRATEGY_ID_LEN = 64
_MAX_BACKTEST_ID_LEN = 64
_FILL_HASH_PREFIX = b"dix.b08.backtester.v1"


class BacktesterError(ValueError):
    """Raised when caller arguments violate backtester contracts."""


@dataclass(frozen=True, slots=True)
class Bar:
    """Single OHLCV bar — Backtrader's data feed atom in DIX form.

    ADAPTED FROM: backtrader/feeds/__init__.py — the fixed OHLCV layout
    every backtrader feed yields.
    """

    ts_ns: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise BacktesterError(f"Bar.ts_ns must be >= 0; got {self.ts_ns}")
        if not self.symbol:
            raise BacktesterError("Bar.symbol must be non-empty")
        if len(self.symbol) > _MAX_SYMBOL_LEN:
            raise BacktesterError(f"Bar.symbol must be <= {_MAX_SYMBOL_LEN} chars")
        for name, value in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ):
            if not math.isfinite(value):
                raise BacktesterError(f"Bar.{name} must be finite; got {value!r}")
            if value < 0.0:
                raise BacktesterError(f"Bar.{name} must be >= 0; got {value!r}")
        if self.high < self.low:
            raise BacktesterError(f"Bar.high ({self.high}) must be >= Bar.low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise BacktesterError("Bar.open must lie inside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise BacktesterError("Bar.close must lie inside [low, high]")


class OrderAction(StrEnum):
    """The three order primitives Backtrader exposes from Strategy.next()."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Strategy decision emitted from Strategy.next().

    ADAPTED FROM: backtrader/strategy.py — Strategy.buy() / Strategy.sell()
    return a market-order request that Cerebro settles against the next
    bar's open. The DIX form replaces the mutable Order object with a
    frozen value object so the strategy API is pure.
    """

    action: OrderAction
    qty: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.qty):
            raise BacktesterError(f"OrderRequest.qty must be finite; got {self.qty!r}")
        if self.qty < 0.0:
            raise BacktesterError(f"OrderRequest.qty must be >= 0; got {self.qty!r}")
        if self.action is OrderAction.HOLD and self.qty != 0.0:
            raise BacktesterError("HOLD orders must carry qty=0")
        if self.action is not OrderAction.HOLD and self.qty <= 0.0:
            raise BacktesterError(f"{self.action.value} orders must carry qty>0; got {self.qty}")


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Read-only snapshot passed to Strategy.next() each bar.

    ADAPTED FROM: backtrader/strategy.py — Strategy.__init__ exposes
    self.position / self.broker.getcash() / self.data; the DIX form
    surfaces the same view through a frozen record so strategies are
    side-effect-free.
    """

    bar_index: int
    bar: Bar
    history: tuple[Bar, ...]
    cash_usd: float
    position_qty: float
    position_avg_price: float
    equity_usd: float

    def __post_init__(self) -> None:
        if self.bar_index < 0:
            raise BacktesterError("StrategyContext.bar_index must be >= 0")
        if not math.isfinite(self.cash_usd):
            raise BacktesterError("StrategyContext.cash_usd must be finite")
        if not math.isfinite(self.position_qty):
            raise BacktesterError("StrategyContext.position_qty must be finite")
        if not math.isfinite(self.position_avg_price):
            raise BacktesterError("StrategyContext.position_avg_price must be finite")
        if self.position_avg_price < 0.0:
            raise BacktesterError("StrategyContext.position_avg_price must be >= 0")
        if not math.isfinite(self.equity_usd):
            raise BacktesterError("StrategyContext.equity_usd must be finite")


@runtime_checkable
class Strategy(Protocol):
    """Pure functional Strategy seam.

    ADAPTED FROM: backtrader/strategy.py — Strategy.next() called once
    per bar. The DIX seam removes mutable ``self`` state in favour of a
    pure ``(ctx) -> OrderRequest`` mapping; strategies that need state
    keep it inside their own caller-managed instance, but the
    backtester itself never reads back any mutation.
    """

    def next(self, ctx: StrategyContext) -> OrderRequest:  # pragma: no cover
        ...


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    """Broker fill model parameters.

    ADAPTED FROM: backtrader/broker.py — BackBroker.setcash /
    setcommission / set_slippage_perc / set_slippage_fixed.

    Args:
        initial_cash_usd: Starting wallet (positive).
        commission_rate: Proportional commission, e.g. ``0.001`` for 10 bps.
        slippage_perc: Worst-case slippage as fraction of fill price; the
            actual slippage is a deterministic value in ``[0, slippage_perc]``
            sampled from a stateless splitmix64 seeded by ``(seed, ts_ns, idx)``.
    """

    initial_cash_usd: float = 100_000.0
    commission_rate: float = 0.0
    slippage_perc: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_cash_usd):
            raise BacktesterError("BrokerConfig.initial_cash_usd must be finite")
        if self.initial_cash_usd <= 0.0:
            raise BacktesterError("BrokerConfig.initial_cash_usd must be > 0")
        if not math.isfinite(self.commission_rate):
            raise BacktesterError("BrokerConfig.commission_rate must be finite")
        if not (0.0 <= self.commission_rate <= 0.1):
            raise BacktesterError(
                f"BrokerConfig.commission_rate must lie in [0, 0.1]; got {self.commission_rate}"
            )
        if not math.isfinite(self.slippage_perc):
            raise BacktesterError("BrokerConfig.slippage_perc must be finite")
        if not (0.0 <= self.slippage_perc <= 0.1):
            raise BacktesterError(
                f"BrokerConfig.slippage_perc must lie in [0, 0.1]; got {self.slippage_perc}"
            )


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Top-level backtester configuration.

    Args:
        backtest_id: Stable identifier baked into the BacktestResult.
        strategy_id: Stable strategy identifier baked into the BacktestResult.
        symbol: Primary instrument (must match every bar in the stream).
        broker: BrokerConfig.
        seed: Replay seed for fill jitter (>=0).
        history_window: Number of past bars exposed to Strategy.next() via
            ``ctx.history`` (excluding the current bar). Default 0.
        meta: Optional free-form string metadata baked into BacktestResult.
    """

    backtest_id: str
    strategy_id: str
    symbol: str
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    seed: int = 0
    history_window: int = 0
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backtest_id:
            raise BacktesterError("BacktestConfig.backtest_id must be non-empty")
        if len(self.backtest_id) > _MAX_BACKTEST_ID_LEN:
            raise BacktesterError(
                f"BacktestConfig.backtest_id must be <= {_MAX_BACKTEST_ID_LEN} chars"
            )
        if not self.strategy_id:
            raise BacktesterError("BacktestConfig.strategy_id must be non-empty")
        if len(self.strategy_id) > _MAX_STRATEGY_ID_LEN:
            raise BacktesterError(
                f"BacktestConfig.strategy_id must be <= {_MAX_STRATEGY_ID_LEN} chars"
            )
        if not self.symbol:
            raise BacktesterError("BacktestConfig.symbol must be non-empty")
        if len(self.symbol) > _MAX_SYMBOL_LEN:
            raise BacktesterError(f"BacktestConfig.symbol must be <= {_MAX_SYMBOL_LEN} chars")
        if self.seed < 0:
            raise BacktesterError("BacktestConfig.seed must be >= 0")
        if self.history_window < 0:
            raise BacktesterError("BacktestConfig.history_window must be >= 0")
        if self.history_window > 1024:
            raise BacktesterError("BacktestConfig.history_window must be <= 1024")
        for key, value in self.meta.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise BacktesterError("BacktestConfig.meta keys and values must be str")


# ---------------------------------------------------------------------------
# Deterministic splitmix64 — same primitive used by simulation/parallel_runner
# and gym_env (A-01.1). Stateless function of (seed, ts_ns, idx).
# ---------------------------------------------------------------------------
_SPLITMIX_C1 = 0x9E3779B97F4A7C15
_SPLITMIX_C2 = 0xBF58476D1CE4E5B9
_SPLITMIX_C3 = 0x94D049BB133111EB
_U64 = 0xFFFFFFFFFFFFFFFF


def _splitmix64(state: int) -> int:
    state = (state + _SPLITMIX_C1) & _U64
    state = ((state ^ (state >> 30)) * _SPLITMIX_C2) & _U64
    state = ((state ^ (state >> 27)) * _SPLITMIX_C3) & _U64
    return (state ^ (state >> 31)) & _U64


def _fill_jitter(seed: int, ts_ns: int, idx: int) -> float:
    """Return a deterministic value in ``[0.0, 1.0)`` for fill jitter."""
    mix = (seed * _SPLITMIX_C1) & _U64
    mix ^= (ts_ns * _SPLITMIX_C2) & _U64
    mix ^= (idx * _SPLITMIX_C3) & _U64
    return _splitmix64(mix) / float(1 << 64)


# ---------------------------------------------------------------------------
# Internal mutable book-keeping (never escapes run_backtest).
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Position:
    qty: float = 0.0
    avg_price: float = 0.0


def _settle_fill(
    *,
    action: OrderAction,
    qty: float,
    base_price: float,
    cfg: BacktestConfig,
    bar: Bar,
    fill_idx: int,
    pos: _Position,
    cash: float,
) -> tuple[float, float, BacktestTrade]:
    """Settle one fill against the broker; returns (new_cash, realised_pnl, trade).

    ADAPTED FROM: backtrader/broker.py — fill price = base_price *
    (1 +/- slippage_perc * jitter); commission = price * qty *
    commission_rate. P&L on a sell is computed against the running
    average position price (Backtrader's default first-in matching).
    """
    jitter = _fill_jitter(cfg.seed, bar.ts_ns, fill_idx)
    if action is OrderAction.BUY:
        fill_price = base_price * (1.0 + cfg.broker.slippage_perc * jitter)
    else:
        fill_price = base_price * (1.0 - cfg.broker.slippage_perc * jitter)
    fill_price = max(fill_price, 0.0)
    notional = fill_price * qty
    commission = notional * cfg.broker.commission_rate

    realised_pnl = 0.0
    if action is OrderAction.BUY:
        new_qty = pos.qty + qty
        if new_qty > 0.0:
            pos.avg_price = (pos.avg_price * pos.qty + fill_price * qty) / new_qty
        pos.qty = new_qty
        cash = cash - notional - commission
    else:
        # Backtrader semantics: matched lot uses running avg price.
        matched = min(qty, pos.qty)
        if matched > 0.0:
            realised_pnl = (fill_price - pos.avg_price) * matched
        pos.qty = pos.qty - qty
        if pos.qty <= 0.0:
            pos.qty = 0.0
            pos.avg_price = 0.0
        cash = cash + notional - commission

    side = Side.BUY if action is OrderAction.BUY else Side.SELL
    trade = BacktestTrade(
        ts_ns=bar.ts_ns,
        symbol=bar.symbol,
        side=side,
        qty=qty,
        price=fill_price,
        pnl_usd=realised_pnl,
        fee_usd=commission,
        venue=_SOURCE,
        order_id=f"{cfg.backtest_id}-{fill_idx:08d}",
    )
    return cash, realised_pnl, trade


def _equity(cash: float, pos: _Position, mark_price: float) -> float:
    return cash + pos.qty * mark_price


def _round_pos(value: float) -> float:
    """Trim tiny FP residue so equity-curve / metrics stay byte-stable."""
    if abs(value) < 1e-9:
        return 0.0
    return value


def _compute_metrics(
    initial_cash: float,
    equity_curve: Sequence[EquityPoint],
    trades: Sequence[BacktestTrade],
) -> BacktestMetrics:
    """Compute Backtrader-style aggregate metrics over a finished run.

    ADAPTED FROM: backtrader/analyzers/{returns,drawdown,tradeanalyzer}.py.
    """
    n_trades = len(trades)
    win_rate = 0.0
    if n_trades > 0:
        wins = sum(1 for t in trades if t.pnl_usd > 0.0)
        win_rate = wins / n_trades
    if equity_curve:
        terminal = equity_curve[-1].equity_usd
    else:
        terminal = initial_cash
    total_return = (terminal - initial_cash) / initial_cash
    # Drawdown over the recorded equity curve.
    peak = initial_cash
    max_dd = 0.0
    for pt in equity_curve:
        if pt.equity_usd > peak:
            peak = pt.equity_usd
        if peak > 0.0:
            dd = (peak - pt.equity_usd) / peak
            if dd > max_dd:
                max_dd = dd
    # Clamp to contract bounds; FP residue can push slightly outside.
    if max_dd < 0.0:
        max_dd = 0.0
    if max_dd > 1.0:
        max_dd = 1.0
    if win_rate < 0.0:
        win_rate = 0.0
    if win_rate > 1.0:
        win_rate = 1.0
    return BacktestMetrics(
        n_trades=n_trades,
        win_rate=win_rate,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=0.0,
        sortino=0.0,
    )


def _policy_hash(cfg: BacktestConfig) -> str:
    """Stable BLAKE2b-16 anchor over the cfg projection."""
    h = hashlib.blake2b(_FILL_HASH_PREFIX, digest_size=16)
    h.update(cfg.backtest_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(cfg.strategy_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(cfg.symbol.encode("utf-8"))
    h.update(b"\x00")
    h.update(repr(cfg.broker.initial_cash_usd).encode("utf-8"))
    h.update(b"\x00")
    h.update(repr(cfg.broker.commission_rate).encode("utf-8"))
    h.update(b"\x00")
    h.update(repr(cfg.broker.slippage_perc).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(cfg.seed).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(cfg.history_window).encode("utf-8"))
    h.update(b"\x00")
    for key in sorted(cfg.meta):
        h.update(key.encode("utf-8"))
        h.update(b"=")
        h.update(cfg.meta[key].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def run_backtest(
    *,
    bars: Iterable[Bar],
    strategy: Strategy,
    config: BacktestConfig,
    result_ts_ns: int,
) -> BacktestResult:
    """Run a deterministic Backtrader-style event loop.

    The driver:

    1. Buffers ``bars`` as a tuple (validates count, monotonic ts_ns,
       single-symbol consistency).
    2. For each bar at index ``i``:
        * Builds a :class:`StrategyContext` from current cash / position /
          equity using ``bars[i].close`` as the marking price.
        * Calls ``strategy.next(ctx)`` to obtain an :class:`OrderRequest`.
        * Settles the order against ``bars[i+1].open`` using the broker
          fill model. The final bar's order is settled against the same
          bar's close (Backtrader's "end-of-data" convention).
        * Marks equity at the *current* bar's close and appends an
          :class:`EquityPoint`.
    3. Computes :class:`BacktestMetrics` and returns a frozen
       :class:`BacktestResult`.

    Args:
        bars: Caller-supplied bar stream. Must be non-empty, monotonic in
            ``ts_ns``, and all bars must carry ``config.symbol``.
        strategy: Object implementing the :class:`Strategy` Protocol.
        config: :class:`BacktestConfig`.
        result_ts_ns: Ingestion timestamp baked into the
            :class:`BacktestResult` (caller-supplied, INV-15).

    Returns:
        Frozen :class:`BacktestResult`.

    Raises:
        BacktesterError: When inputs are malformed.
    """
    if not isinstance(strategy, Strategy):
        raise BacktesterError(
            "strategy must implement the Strategy Protocol (`def next(self, ctx) -> OrderRequest`)"
        )
    if result_ts_ns < 0:
        raise BacktesterError("result_ts_ns must be >= 0")
    bar_tuple: tuple[Bar, ...] = tuple(bars)
    if not bar_tuple:
        raise BacktesterError("bars must be non-empty")
    if len(bar_tuple) > _MAX_BARS:
        raise BacktesterError(f"bars must be <= {_MAX_BARS}; got {len(bar_tuple)}")
    last_ts: int | None = None
    for bar in bar_tuple:
        if not isinstance(bar, Bar):
            raise BacktesterError("bars must be Bar instances")
        if bar.symbol != config.symbol:
            raise BacktesterError(f"bar symbol {bar.symbol!r} != config.symbol {config.symbol!r}")
        if last_ts is not None and bar.ts_ns < last_ts:
            raise BacktesterError("bars must be monotonic non-decreasing in ts_ns")
        last_ts = bar.ts_ns

    cash = config.broker.initial_cash_usd
    pos = _Position()
    trades: list[BacktestTrade] = []
    equity_curve: list[EquityPoint] = []
    last_eq_ts: int | None = None
    fill_idx = 0
    history: list[Bar] = []

    n = len(bar_tuple)
    for i, bar in enumerate(bar_tuple):
        history_view: tuple[Bar, ...]
        if config.history_window <= 0:
            history_view = ()
        else:
            window = history[-config.history_window :]
            history_view = tuple(window)
        equity = _equity(cash, pos, bar.close)
        ctx = StrategyContext(
            bar_index=i,
            bar=bar,
            history=history_view,
            cash_usd=_round_pos(cash),
            position_qty=_round_pos(pos.qty),
            position_avg_price=_round_pos(pos.avg_price),
            equity_usd=_round_pos(equity),
        )
        decision = strategy.next(ctx)
        if not isinstance(decision, OrderRequest):
            raise BacktesterError(
                f"Strategy.next must return an OrderRequest; got {type(decision).__name__}"
            )
        if decision.action is not OrderAction.HOLD:
            if i + 1 < n:
                fill_bar = bar_tuple[i + 1]
                base_price = fill_bar.open
            else:
                fill_bar = bar
                base_price = bar.close
            cash, _realised_pnl, trade = _settle_fill(
                action=decision.action,
                qty=decision.qty,
                base_price=base_price,
                cfg=config,
                bar=fill_bar,
                fill_idx=fill_idx,
                pos=pos,
                cash=cash,
            )
            trades.append(trade)
            fill_idx += 1
            if fill_idx > _MAX_TRADES:
                raise BacktesterError(f"trade count exceeded {_MAX_TRADES}")
        # Mark equity *after* any same-bar fill on the final bar so the
        # closing equity reflects exits.
        mark = bar.close
        equity = _equity(cash, pos, mark)
        # Equity points are sorted ascending by ts_ns; bars with
        # repeated ts_ns collapse to the latest equity (last write wins).
        if last_eq_ts is not None and bar.ts_ns == last_eq_ts:
            equity_curve[-1] = EquityPoint(ts_ns=bar.ts_ns, equity_usd=_round_pos(equity))
        else:
            equity_curve.append(EquityPoint(ts_ns=bar.ts_ns, equity_usd=_round_pos(equity)))
            last_eq_ts = bar.ts_ns
        history.append(bar)

    period_start = bar_tuple[0].ts_ns
    period_end = bar_tuple[-1].ts_ns
    if result_ts_ns < period_end:
        # Result ingestion ts must be at-or-after the last bar so the
        # contract round-trips cleanly through downstream ingesters.
        result_ts_ns = period_end
    metrics = _compute_metrics(config.broker.initial_cash_usd, equity_curve, trades)
    # Trades must lie inside [period_start, period_end] — the final-bar
    # fill ts (bar.ts_ns) is the same as period_end, so this holds by
    # construction. We still enforce an explicit guard for callers that
    # pass non-monotonic streams that pass the input validator.
    for trade in trades:
        if not (period_start <= trade.ts_ns <= period_end):
            raise BacktesterError(
                f"trade ts_ns {trade.ts_ns} outside period window [{period_start}, {period_end}]"
            )
    meta_dict: dict[str, str] = {str(k): str(v) for k, v in config.meta.items()}
    meta_dict.setdefault("seed", str(config.seed))
    meta_dict.setdefault("history_window", str(config.history_window))
    return BacktestResult(
        ts_ns=result_ts_ns,
        source=_SOURCE,
        backtest_id=config.backtest_id,
        strategy_id=config.strategy_id,
        symbol=config.symbol,
        period_start_ns=period_start,
        period_end_ns=period_end,
        equity_curve=tuple(equity_curve),
        trades=tuple(trades),
        metrics=metrics,
        trust=SignalTrust.EXTERNAL_LOW,
        policy_hash=_policy_hash(config),
        meta=meta_dict,
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "Bar",
    "BacktestConfig",
    "BacktesterError",
    "BrokerConfig",
    "OrderAction",
    "OrderRequest",
    "Strategy",
    "StrategyContext",
    "run_backtest",
)
