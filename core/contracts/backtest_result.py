"""BacktestResult — frozen contract for ingesting external backtest output.

Paper-S3 closes the data-normalisation seam between **read-only**
external simulation sources (TradingView, QuantConnect, MT5
historical, future replay engines) and Indira's learning loop.

Design constraints
------------------
* **Frozen, slotted, hashable** — same shape as every other contract
  under :mod:`core.contracts`. No callables, no IO, no clocks.
* **Read-only ingestion only.** A :class:`BacktestResult` carries
  *historical* trade rows; the contract itself does not grant any
  execution authority. Projection into the learning loop happens
  through :func:`project_to_trade_outcomes`, which only emits
  :class:`~core.contracts.learning.TradeOutcome` records — never
  :class:`~core.contracts.execution_intent.ExecutionIntent`.
* **Replay-deterministic.** All fields are primitives (``int``,
  ``str``, ``float``) or tuples of primitive records. Equality is
  structural so a backtest projected twice yields byte-identical
  ``TradeOutcome`` streams (INV-15 / TEST-01).
* **Validation in ``__post_init__``.** Period window monotonicity,
  equity-curve ts ordering, trades inside the window, metric ranges.
  Bad payloads are rejected at construction so the learning loop
  never sees malformed state.

Trust class
-----------
External backtests are conservatively classified; the ingester sets
:class:`~core.contracts.signal_trust.SignalTrust` when projecting
backtests into :class:`~core.contracts.events.SignalEvent`. The
contract here defaults the *source-level* trust to
``EXTERNAL_LOW`` so an unattended ingester cannot accidentally
promote backtests to ``INTERNAL`` confidence.

INV-08, INV-11, INV-15 — only typed records cross domain boundaries,
no direct cross-engine method calls, all fields deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from core.contracts.events import Side
from core.contracts.learning import TradeOutcome
from core.contracts.signal_trust import SignalTrust

_SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_POLICY_HASH_PATTERN = re.compile(r"^[a-f0-9]{16,64}$")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Single point on a backtest equity curve (in USD)."""

    ts_ns: int
    equity_usd: float


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One realised trade inside a backtest run.

    Mirrors the structure of an :class:`ExecutionEvent` so the
    projection into :class:`TradeOutcome` is a 1-to-1 mapping —
    no inferred fields, no synthesis, no clocks.
    """

    ts_ns: int
    symbol: str
    side: Side
    qty: float
    price: float
    pnl_usd: float
    fee_usd: float = 0.0
    venue: str = ""
    order_id: str = ""

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.qty < 0.0:
            raise ValueError(f"qty must be >= 0; got {self.qty}")
        if self.price < 0.0:
            raise ValueError(f"price must be >= 0; got {self.price}")
        if self.fee_usd < 0.0:
            raise ValueError(f"fee_usd must be >= 0; got {self.fee_usd}")


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate metrics summarising a backtest run.

    Implementations are required to populate ``n_trades``, ``win_rate``,
    ``total_return`` and ``max_drawdown``. Sharpe / Sortino are
    optional (set to ``0.0`` when not computed).
    """

    n_trades: int
    win_rate: float        # in [0.0, 1.0]
    total_return: float    # decimal: 0.15 == +15 %
    max_drawdown: float    # absolute, in [0.0, 1.0]
    sharpe: float = 0.0
    sortino: float = 0.0

    def __post_init__(self) -> None:
        if self.n_trades < 0:
            raise ValueError(f"n_trades must be >= 0; got {self.n_trades}")
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(f"win_rate must be in [0,1]; got {self.win_rate}")
        if not (0.0 <= self.max_drawdown <= 1.0):
            raise ValueError(
                f"max_drawdown must be in [0,1]; got {self.max_drawdown}"
            )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete backtest result ingested from an external source.

    Args:
        ts_ns: Ingestion timestamp (ns since epoch).
        source: Lowercase platform identifier
            (``"tradingview"``, ``"quantconnect"``, ``"mt5"``,
            ``"internal_replay"``, …). Restricted to
            ``[a-z][a-z0-9_]{1,31}``.
        backtest_id: Stable identifier from the source platform.
        strategy_id: Strategy that was backtested.
        symbol: Primary instrument the backtest ran on.
        period_start_ns / period_end_ns: Closed window (start <= end).
        equity_curve: Sorted-ascending tuple of :class:`EquityPoint`.
        trades: Tuple of :class:`BacktestTrade` rows; each ``ts_ns``
            must lie inside the period window.
        metrics: Aggregate :class:`BacktestMetrics`.
        trust: Trust class for any signals projected from this backtest.
            Defaults to :data:`SignalTrust.EXTERNAL_LOW`.
        policy_hash: Optional SHA hex (16-64 chars) anchoring the
            backtest configuration / strategy version.
        meta: Free-form, all values must be ``str``.
    """

    ts_ns: int
    source: str
    backtest_id: str
    strategy_id: str
    symbol: str
    period_start_ns: int
    period_end_ns: int
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[BacktestTrade, ...]
    metrics: BacktestMetrics
    trust: SignalTrust = SignalTrust.EXTERNAL_LOW
    policy_hash: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError(f"ts_ns must be >= 0; got {self.ts_ns}")
        if not _SOURCE_PATTERN.fullmatch(self.source):
            raise ValueError(
                "source must match [a-z][a-z0-9_]{1,31}; "
                f"got {self.source!r}"
            )
        if not self.backtest_id:
            raise ValueError("backtest_id must be non-empty")
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.period_start_ns < 0 or self.period_end_ns < 0:
            raise ValueError("period_*_ns must be >= 0")
        if self.period_end_ns < self.period_start_ns:
            raise ValueError(
                "period_end_ns must be >= period_start_ns; got "
                f"{self.period_start_ns} > {self.period_end_ns}"
            )
        # Equity curve must be ts-ascending and inside the period.
        last_ts: int | None = None
        for pt in self.equity_curve:
            if not isinstance(pt, EquityPoint):
                raise TypeError(
                    f"equity_curve entries must be EquityPoint; got {type(pt)!r}"
                )
            if pt.ts_ns < self.period_start_ns or pt.ts_ns > self.period_end_ns:
                raise ValueError(
                    f"equity point ts_ns={pt.ts_ns} outside period window"
                )
            if last_ts is not None and pt.ts_ns < last_ts:
                raise ValueError(
                    "equity_curve must be sorted ascending by ts_ns"
                )
            last_ts = pt.ts_ns
        # Trades must lie inside the window (any order — multiple
        # symbols may interleave).
        for trade in self.trades:
            if not isinstance(trade, BacktestTrade):
                raise TypeError(
                    f"trades entries must be BacktestTrade; got {type(trade)!r}"
                )
            if (
                trade.ts_ns < self.period_start_ns
                or trade.ts_ns > self.period_end_ns
            ):
                raise ValueError(
                    f"trade ts_ns={trade.ts_ns} outside period window"
                )
        # n_trades in metrics should equal len(trades) when both
        # are non-empty; we enforce exact equality so a malformed
        # ingester cannot ship inconsistent stats.
        if self.metrics.n_trades != len(self.trades):
            raise ValueError(
                f"metrics.n_trades ({self.metrics.n_trades}) != "
                f"len(trades) ({len(self.trades)})"
            )
        if self.policy_hash and not _POLICY_HASH_PATTERN.fullmatch(
            self.policy_hash
        ):
            raise ValueError(
                "policy_hash must be 16-64 lowercase hex chars; "
                f"got {self.policy_hash!r}"
            )
        # Confirm meta values are all str so JSON / event-bus
        # round-trip stays trivial.
        for key, value in self.meta.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "meta keys and values must be str; "
                    f"got ({type(key)!r}, {type(value)!r})"
                )


def project_to_trade_outcomes(
    result: BacktestResult,
) -> tuple[TradeOutcome, ...]:
    """Project *result*'s trade log into a tuple of :class:`TradeOutcome`.

    Used by Paper-S4..S7 ingesters (TradingView / QuantConnect / MT5)
    to feed external backtests into the existing learning loop without
    granting them execution authority. The mapping is 1-to-1, lossless
    on the fields the learning loop reads, and pure (no clocks, no IO).

    The emitted ``TradeOutcome`` carries ``status=ExecutionStatus.FILLED``
    because the source platform considered the trade executed — even
    if internally we treat the result as low-trust historical data.

    The ``meta`` of every emitted outcome carries:

    * ``source`` — the platform identifier;
    * ``backtest_id`` — the source-side identifier;
    * ``trust`` — the :class:`SignalTrust` value, so downstream gates
      (governance cap, drift oracle) can apply the right policy.
    """

    from core.contracts.events import ExecutionStatus  # local: avoid cycle

    out: list[TradeOutcome] = []
    base_meta: dict[str, str] = {
        "source": result.source,
        "backtest_id": result.backtest_id,
        "trust": str(result.trust),
    }
    if result.policy_hash:
        base_meta["policy_hash"] = result.policy_hash
    for trade in result.trades:
        meta = dict(base_meta)
        if trade.fee_usd:
            meta["fee_usd"] = f"{trade.fee_usd:.10g}"
        meta["price"] = f"{trade.price:.10g}"
        meta["side"] = str(trade.side)
        out.append(
            TradeOutcome(
                ts_ns=trade.ts_ns,
                strategy_id=result.strategy_id,
                symbol=trade.symbol,
                qty=trade.qty,
                pnl=trade.pnl_usd,
                status=ExecutionStatus.FILLED,
                venue=trade.venue or result.source,
                order_id=trade.order_id,
                meta=meta,
            )
        )
    return tuple(out)


def build_equity_curve(
    points: Iterable[tuple[int, float]],
) -> tuple[EquityPoint, ...]:
    """Convenience: build a sorted, validated equity curve from raw pairs."""

    return tuple(
        EquityPoint(ts_ns=int(ts), equity_usd=float(eq))
        for ts, eq in sorted(points, key=lambda p: int(p[0]))
    )


__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "BacktestTrade",
    "EquityPoint",
    "build_equity_curve",
    "project_to_trade_outcomes",
]
