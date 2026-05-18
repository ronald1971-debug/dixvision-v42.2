# ADAPTED FROM: qlib/qlib/contrib/evaluate.py
# ADAPTED FROM: qlib/qlib/contrib/report/analysis_position/score_ic.py
"""PnL attribution — decompose realised PnL into signal alpha + execution slippage + fees.

Adapted from microsoft/qlib's ``contrib/evaluate.py`` PnL-attribution pattern:
the realised PnL of every round-trip trade is the sum of a theoretical
"signal" component (what the alpha would have produced at the signal-time
price) and an "execution" component (what slippage and fees took away).

This module is the **first** of the S-04 pyqlib triple
(``pnl_attribution.py`` + ``alpha_decay.py`` + ``execution_quality.py``);
it intentionally restricts itself to one concern: signal-vs-execution
attribution of a sequence of :class:`AttributedTrade` records.

Tier
----
**OFFLINE.** ``learning_engine/performance_analysis/`` is a slow-cadence
analytics tier — never called from the hot path, never imported by
``hot_path/`` modules (authority_lint T1 / B1).

Design constraints
------------------
* **Pure functions.** No clock reads (``time.time()``,
  ``datetime.now()``), no IO, no global mutable state. Replay-deterministic
  (INV-15): identical inputs produce byte-identical
  :class:`PnLAttribution` outputs.
* **Frozen contracts.** Both :class:`AttributedTrade` and
  :class:`PnLAttribution` are ``@dataclass(frozen=True, slots=True)`` so
  structural equality is preserved across replays.
* **Eager validation.** Constructors reject malformed input
  (``ValueError`` / ``TypeError``) so a downstream learning loop never
  observes partial state.
* **No new pip dependencies.** :data:`NEW_PIP_DEPENDENCIES` is empty —
  the qlib formulas only need :mod:`math` from the stdlib.
* **Stable accumulation order.** When aggregating, the implementation
  walks input trades in iteration order; sums use plain Python floats
  (no numpy reductions) so floating-point rounding is identical across
  CPython versions.

Algorithmic summary
-------------------
For a single round-trip ``BacktestTrade`` paired with a strategy-side
``signal_price`` (the price the alpha *expected* to transact at), the
decomposition is::

    realised_pnl = signal_pnl + slippage_pnl + fee_pnl

where (with ``side_sign = +1`` for BUY and ``-1`` for SELL):

* ``slippage_pnl = -(fill_price - signal_price) * qty * side_sign``
  — signed; **negative** when execution paid more (BUY) or received less
  (SELL) than the alpha expected.
* ``fee_pnl     = -fee_usd``
  — always ≤ 0. Fees are pure costs.
* ``signal_pnl  = realised_pnl - slippage_pnl - fee_pnl``
  — the theoretical alpha contribution if the trade had filled at
  ``signal_price`` with no fees.

The aggregator preserves this identity exactly: ``signal + slippage +
fee == realised`` for every :class:`PnLAttribution` instance.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Final

from core.contracts.backtest_result import BacktestTrade
from core.contracts.events import Side

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""S-04.1 introduces no new pip dependencies — pure stdlib."""


@dataclasses.dataclass(frozen=True, slots=True)
class AttributedTrade:
    """One :class:`BacktestTrade` paired with the signal-time price.

    The strategy-side ``signal_price`` is the price the alpha expected
    to transact at when it produced the signal — typically the mid or
    last-trade price observed at the moment the order was emitted.

    Args:
        trade: The realised :class:`BacktestTrade` row from a backtest.
        signal_price: Strategy-side expected price (USD, > 0).
    """

    trade: BacktestTrade
    signal_price: float

    def __post_init__(self) -> None:
        if not isinstance(self.trade, BacktestTrade):
            raise TypeError(f"trade must be BacktestTrade; got {type(self.trade).__name__}")
        if not isinstance(self.signal_price, (int, float)):
            raise TypeError(f"signal_price must be float; got {type(self.signal_price).__name__}")
        if self.signal_price != self.signal_price:  # NaN check (IEEE-754)
            raise ValueError("signal_price must not be NaN")
        if self.signal_price <= 0.0:
            raise ValueError(f"signal_price must be > 0; got {self.signal_price}")


@dataclasses.dataclass(frozen=True, slots=True)
class PnLAttribution:
    """Decomposition of realised PnL into alpha + execution + fees.

    Invariant: ``signal_pnl_usd + slippage_pnl_usd + fee_pnl_usd ==
    realised_pnl_usd`` (within floating-point precision).

    Args:
        n_trades: Number of underlying :class:`AttributedTrade` rows
            aggregated into this record.
        notional_usd: Sum of ``|qty * fill_price|`` across trades.
            Always ≥ 0.
        realised_pnl_usd: Sum of :attr:`BacktestTrade.pnl_usd` across
            trades. Signed.
        signal_pnl_usd: Theoretical alpha contribution at signal price.
            Signed.
        slippage_pnl_usd: Signed slippage component. Negative means
            slippage hurt the strategy; positive means execution
            beat the signal (rare).
        fee_pnl_usd: Sum of ``-fee_usd`` across trades. Always ≤ 0.
    """

    n_trades: int
    notional_usd: float
    realised_pnl_usd: float
    signal_pnl_usd: float
    slippage_pnl_usd: float
    fee_pnl_usd: float

    def __post_init__(self) -> None:
        if self.n_trades < 0:
            raise ValueError(f"n_trades must be >= 0; got {self.n_trades}")
        if self.notional_usd < 0.0:
            raise ValueError(f"notional_usd must be >= 0; got {self.notional_usd}")
        if self.fee_pnl_usd > 0.0:
            raise ValueError(f"fee_pnl_usd must be <= 0 (fees are costs); got {self.fee_pnl_usd}")

    def slippage_bps(self) -> float:
        """Slippage in basis points relative to notional.

        Positive bps means execution **cost** the strategy money
        (slippage_pnl_usd was negative). Returns ``0.0`` when notional
        is zero (no trades).
        """
        if self.notional_usd <= 0.0:
            return 0.0
        return -self.slippage_pnl_usd / self.notional_usd * 10_000.0

    def fee_bps(self) -> float:
        """Fee drag in basis points relative to notional.

        Positive bps. Returns ``0.0`` when notional is zero.
        """
        if self.notional_usd <= 0.0:
            return 0.0
        return -self.fee_pnl_usd / self.notional_usd * 10_000.0


_EMPTY: Final[PnLAttribution] = PnLAttribution(
    n_trades=0,
    notional_usd=0.0,
    realised_pnl_usd=0.0,
    signal_pnl_usd=0.0,
    slippage_pnl_usd=0.0,
    fee_pnl_usd=0.0,
)


def _side_sign(side: Side) -> int:
    """Return +1 for BUY, -1 for SELL."""
    if side is Side.BUY:
        return 1
    if side is Side.SELL:
        return -1
    raise ValueError(f"unsupported Side: {side!r}")


def attribute_pnl(trades: Iterable[AttributedTrade]) -> PnLAttribution:
    """Aggregate :class:`AttributedTrade` records into one :class:`PnLAttribution`.

    Walks ``trades`` in iteration order with deterministic float
    accumulation. An empty input returns the canonical zero record.

    Args:
        trades: Iterable of :class:`AttributedTrade` records.

    Returns:
        A :class:`PnLAttribution` whose components satisfy the identity
        ``signal_pnl + slippage_pnl + fee_pnl == realised_pnl``.

    Raises:
        TypeError: If any element is not an :class:`AttributedTrade`.
    """
    n: int = 0
    notional: float = 0.0
    realised: float = 0.0
    slippage: float = 0.0
    fee: float = 0.0

    for at in trades:
        if not isinstance(at, AttributedTrade):
            raise TypeError(f"trades must contain AttributedTrade; got {type(at).__name__}")
        t = at.trade
        sign = _side_sign(t.side)
        notional_i = abs(t.qty * t.price)
        slippage_i = -(t.price - at.signal_price) * t.qty * sign
        fee_i = -t.fee_usd

        n += 1
        notional += notional_i
        realised += t.pnl_usd
        slippage += slippage_i
        fee += fee_i

    signal = realised - slippage - fee
    return PnLAttribution(
        n_trades=n,
        notional_usd=notional,
        realised_pnl_usd=realised,
        signal_pnl_usd=signal,
        slippage_pnl_usd=slippage,
        fee_pnl_usd=fee,
    )


def attribute_pnl_by_symbol(
    trades: Iterable[AttributedTrade],
) -> Mapping[str, PnLAttribution]:
    """Group :class:`AttributedTrade` records by symbol then aggregate each group.

    Symbols are emitted in **first-seen** order to keep replay byte-
    identical (INV-15) — :class:`dict` preserves insertion order on
    CPython ≥ 3.7.

    Args:
        trades: Iterable of :class:`AttributedTrade` records.

    Returns:
        A read-only ``dict`` mapping each symbol to its
        :class:`PnLAttribution`. Empty input returns an empty dict.

    Raises:
        TypeError: If any element is not an :class:`AttributedTrade`.
    """
    buckets: dict[str, list[AttributedTrade]] = {}
    for at in trades:
        if not isinstance(at, AttributedTrade):
            raise TypeError(f"trades must contain AttributedTrade; got {type(at).__name__}")
        buckets.setdefault(at.trade.symbol, []).append(at)

    return {sym: attribute_pnl(rows) for sym, rows in buckets.items()}


def empty_attribution() -> PnLAttribution:
    """Return the canonical zero-trade :class:`PnLAttribution`."""
    return _EMPTY


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "AttributedTrade",
    "PnLAttribution",
    "attribute_pnl",
    "attribute_pnl_by_symbol",
    "empty_attribution",
]
