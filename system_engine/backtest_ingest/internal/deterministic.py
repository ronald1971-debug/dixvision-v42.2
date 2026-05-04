"""Deterministic in-process backtester used by ``POST /api/testing/backtest``.

This is the canonical server-side analogue of the seed-driven simulation
that the dashboard ``Backtester`` widget previously did entirely in the
browser. Pulling the algorithm into the harness gives the audit trail a
single source of truth — every backtest the operator runs is now hashed
into the in-memory event ring (keyed by request seed) and can be re-run
identically by replaying the same parameters.

The implementation mirrors the widget's prior algorithm bit-for-bit
(FNV-1a hash → linear congruential generator → 240-bar walk) so the
move from browser to server does not change visible output for any
operator who has bookmarked a particular ``(strategy, symbol, range,
fill, slippage)`` combination.

The endpoint is **read-only** — running a backtest never produces an
``ExecutionIntent``, never writes to the authority ledger, never feeds
the learning loop. The :class:`~core.contracts.backtest_result.BacktestResult`
ingestion seam (Paper-S3) remains the only path that grafts external
backtests into Indira's learning surface.

INV-15 (replay determinism): given the same ``BacktestRequest`` the
function returns byte-identical :class:`BacktestReport` records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC
from typing import Literal

Strategy = Literal[
    "ema_cross_20_50",
    "rsi_2_meanrev",
    "vwap_reversion",
    "breakout_channel",
    "microstructure_v1",
    "news_event_drift",
    "memecoin_copy",
    "memecoin_sniper",
]
FillModel = Literal["next_tick", "vwap_5min", "mid_price", "tob_aggress"]


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """Immutable request payload for :func:`run_deterministic_backtest`.

    Mirrors the form fields in ``dashboard2026/src/widgets/testing/Backtester.tsx``.
    """

    strategy: Strategy
    symbol: str
    start_iso: str
    end_iso: str
    fill_model: FillModel = "next_tick"
    slippage_bps: float = 8.0


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One trade row in the deterministic walk."""

    ts_iso: str
    side: Literal["BUY", "SELL"]
    pnl_pct: float
    bars_held: int


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate metrics summarising a backtest walk."""

    final_equity_pct: float
    cagr: float
    sharpe: float
    sortino: float
    max_dd_pct: float
    win_rate: float
    profit_factor: float
    avg_trade_pct: float
    longest_loss_streak: int
    n_trades: int


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Result returned by :func:`run_deterministic_backtest`.

    Equity / drawdown are sampled per bar; trades carry a per-step
    realised pnl. ``seed`` is the FNV-1a hash of the request fields so
    an operator can correlate two reports by their seed string in the
    audit trail.
    """

    seed: str
    request: BacktestRequest
    equity: tuple[float, ...]
    drawdown: tuple[float, ...]
    trades: tuple[BacktestTrade, ...]
    metrics: BacktestMetrics
    notes: tuple[str, ...] = field(default_factory=tuple)


_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619
_LCG_MUL = 1664525
_LCG_INC = 1013904223
_U32 = 0xFFFFFFFF


def _fnv1a(parts: tuple[str | int | float, ...]) -> int:
    h = _FNV_OFFSET
    for part in parts:
        for ch in str(part):
            h ^= ord(ch)
            h = (h * _FNV_PRIME) & _U32
    return h


def _seeded_rng(seed: int):
    state = seed & _U32

    def next_unit() -> float:
        nonlocal state
        state = (state * _LCG_MUL + _LCG_INC) & _U32
        return state / 4294967296.0

    return next_unit


def _parse_iso_ms(iso: str) -> int:
    """Parse a YYYY-MM-DD or full ISO-8601 string to epoch milliseconds.

    Accepts the trailing ``Z`` UTC marker the widget emits.
    """

    raw = iso.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    from datetime import datetime

    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def run_deterministic_backtest(req: BacktestRequest) -> BacktestReport:
    """Run the canonical deterministic backtest walk.

    The walk is identical to the prior browser-side simulation:
    240 bars, every fifth bar produces a trade, drift / volatility are
    keyed off the strategy. Slippage applies on trade bars only.
    """

    seed = _fnv1a(
        (
            req.strategy,
            req.symbol,
            req.start_iso,
            req.end_iso,
            req.fill_model,
            req.slippage_bps,
        )
    )
    rng = _seeded_rng(seed)
    n = 240

    equity: list[float] = [100.0]
    drawdown: list[float] = [0.0]
    trades: list[BacktestTrade] = []

    peak = 100.0
    losing_streak = 0
    longest_loss = 0
    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0

    drift = 0.06 if req.strategy == "memecoin_sniper" else 0.012
    vol = 0.95 if req.strategy == "memecoin_sniper" else 0.32
    slip_drag = req.slippage_bps / 10000.0

    start_ms = _parse_iso_ms(req.start_iso)

    for i in range(n):
        r = (rng() - 0.5) * vol + drift / 100.0
        step_pnl = r - (slip_drag if i % 5 == 0 else 0.0)
        if i % 5 == 0:
            pnl_pct = step_pnl * 100.0
            ts_ms = start_ms + i * 3_600_000
            from datetime import datetime

            ts_iso = (
                datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
                .isoformat()
                .replace("+00:00", "Z")
            )
            trades.append(
                BacktestTrade(
                    ts_iso=ts_iso,
                    side="BUY" if rng() > 0.5 else "SELL",
                    pnl_pct=pnl_pct,
                    bars_held=3 + int(rng() * 8),
                )
            )
            if pnl_pct >= 0.0:
                wins += 1
                gross_win += pnl_pct
                losing_streak = 0
            else:
                losses += 1
                gross_loss += abs(pnl_pct)
                losing_streak += 1
                if losing_streak > longest_loss:
                    longest_loss = losing_streak

        nxt = equity[-1] * (1.0 + step_pnl)
        equity.append(nxt)
        if nxt > peak:
            peak = nxt
        drawdown.append(((nxt - peak) / peak) * 100.0)

    final_pct = equity[-1] - 100.0
    end_ms = _parse_iso_ms(req.end_iso)
    days = max(1.0, (end_ms - start_ms) / (24 * 3_600_000))
    cagr = (math.pow(equity[-1] / 100.0, 365.0 / days) - 1.0) * 100.0

    rets: list[float] = []
    for j in range(1, len(equity)):
        rets.append((equity[j] - equity[j - 1]) / equity[j - 1])
    mean = sum(rets) / max(1, len(rets))
    var = sum((x - mean) ** 2 for x in rets) / max(1, len(rets) - 1)
    sharpe = (mean / math.sqrt(var + 1e-9)) * math.sqrt(252.0)
    downs = [x for x in rets if x < 0.0]
    down_var = sum(x * x for x in downs) / max(1, len(downs) - 1)
    sortino = (mean / math.sqrt(down_var + 1e-9)) * math.sqrt(252.0)
    max_dd = min(drawdown) if drawdown else 0.0
    total_trades = wins + losses
    profit_factor = gross_win / gross_loss if gross_loss > 0.0 else math.inf
    avg_trade = (
        (gross_win - gross_loss) / total_trades if total_trades > 0 else 0.0
    )

    metrics = BacktestMetrics(
        final_equity_pct=final_pct,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_dd_pct=abs(max_dd),
        win_rate=(wins / total_trades) if total_trades > 0 else 0.0,
        profit_factor=profit_factor,
        avg_trade_pct=avg_trade,
        longest_loss_streak=longest_loss,
        n_trades=total_trades,
    )

    return BacktestReport(
        seed=f"{seed:08x}",
        request=req,
        equity=tuple(equity),
        drawdown=tuple(drawdown),
        trades=tuple(trades),
        metrics=metrics,
        notes=(
            "deterministic-internal",
            "no-historical-data-source",
            "no-execution-authority",
        ),
    )


__all__ = [
    "BacktestMetrics",
    "BacktestReport",
    "BacktestRequest",
    "BacktestTrade",
    "FillModel",
    "Strategy",
    "run_deterministic_backtest",
]
