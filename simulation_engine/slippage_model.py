# ADAPTED FROM: hftbacktest/py-hftbacktest/hftbacktest/binding.pyi
# ADAPTED FROM: hftbacktest/hftbacktest/src/backtest/proc/local.rs (market-order book walk)
# ADAPTED FROM: hftbacktest/hftbacktest/src/backtest/models/queue.rs (queue model authority)
"""Slippage models (S-02.1) — adapted from ``hftbacktest``.

This is the price-impact half of the S-02 ``hftbacktest`` adaptation
pair. The latency / queue-time half (S-02.2) lives in
:mod:`simulation_engine.latency_model` and is shipped in a separate
PR per the strict canonical cadence.

What survives from upstream
---------------------------
* The level-by-level book walk used by ``hftbacktest``'s local processor
  to match a market order against opposite-side levels (in
  ``hftbacktest::backtest::proc::local::Local::submit_order``) is
  replicated verbatim in :class:`BookWalkSlippage._walk`.
* hftbacktest's separation between *price-impact* (slippage) and
  *time-impact* (latency / queue position) is preserved — this module
  only models the former. Latency lives in
  :mod:`simulation_engine.latency_model` (S-02.2).
* The square-root market-impact form (Kyle / Almgren-Chriss family),
  which hftbacktest exposes in its ``examples/sqrt_impact.py`` notebook,
  is the third available model here.

What is rewritten behind DIX contracts
--------------------------------------
* No numpy, no pandas, no torch — every model is plain-Python with
  frozen dataclasses, so the simulation tier stays leaf-pure
  (``simulation_engine/`` imports only from stdlib + ``core.contracts``).
* No clock, no PRNG, no global mutable state. Every model is a pure
  function of ``(side, qty, mark_price, book)`` plus its frozen
  config. INV-15 replay determinism is guaranteed — replaying the same
  inputs always yields the same output to the bit.
* No ``hftbacktest`` import (it's a Rust-backed pip dependency we
  haven't pulled in). The algorithms are reproduced from the upstream
  source without taking the runtime dependency. ``hftbacktest`` is
  flagged in :data:`NEW_PIP_DEPENDENCIES` as ``()`` — we do *not* take
  the dep here; if a future S-02.x adaptation needs the real Rust
  backtester it can flag the dep then.
* The :class:`SlippageModel` Protocol is the exclusive integration
  surface — every concrete model satisfies it, and the strategy arena
  / paper broker / SIM step functions consume the Protocol, not the
  classes. New models can be added without touching callers.

Tier classification
-------------------
``simulation_engine/`` is **OFFLINE tier** per the master canonical
PART 1: it can use ML in future modules, must never be called from
``hot_path/`` directly, and only ever emits structured outputs the
meta-controller's scoring layer reads asynchronously.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol, runtime_checkable

from core.contracts.events import Side

# This module reproduces hftbacktest's algorithms in plain Python, so
# we deliberately do not pull in the Rust-backed pip wheel.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Book snapshot (input to every slippage model)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BookLevel:
    """One price level on a side of the book.

    ``price`` is the level's quoted price (must be ``> 0``); ``qty`` is
    the resting size at that level (must be ``>= 0``). Both are floats
    to match every adapter we'll plug in (``ccxt`` / ``hftbacktest`` /
    ``nautilus_trader`` all return floats).
    """

    price: float
    qty: float

    def __post_init__(self) -> None:
        if not (self.price > 0.0):  # IEEE-754 NaN-safe (PR #234 pattern)
            raise ValueError(f"BookLevel.price must be > 0 (NaN-safe), got {self.price!r}")
        if not (self.qty >= 0.0):
            raise ValueError(f"BookLevel.qty must be >= 0 (NaN-safe), got {self.qty!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Frozen, sorted L2 book snapshot.

    ``bids`` are sorted in **descending** price order (best bid first);
    ``asks`` are sorted in **ascending** price order (best ask first).
    The :meth:`__post_init__` hook validates ordering so a malformed
    snapshot can't silently mis-fill a downstream order.
    """

    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    def __post_init__(self) -> None:
        for i in range(1, len(self.bids)):
            if self.bids[i].price > self.bids[i - 1].price:
                raise ValueError("BookSnapshot.bids must be sorted descending by price")
        for i in range(1, len(self.asks)):
            if self.asks[i].price < self.asks[i - 1].price:
                raise ValueError("BookSnapshot.asks must be sorted ascending by price")
        if self.bids and self.asks:
            if self.bids[0].price >= self.asks[0].price:
                raise ValueError(
                    "BookSnapshot.bids[0].price must be strictly less than "
                    f"asks[0].price (got bid={self.bids[0].price!r}, "
                    f"ask={self.asks[0].price!r}); a crossed book is invalid."
                )

    @property
    def mid_price(self) -> float | None:
        """Mid-price if both sides have at least one level, else ``None``."""
        if not self.bids or not self.asks:
            return None
        return 0.5 * (self.bids[0].price + self.asks[0].price)


# ---------------------------------------------------------------------------
# Protocol (the exclusive integration surface)
# ---------------------------------------------------------------------------


@runtime_checkable
class SlippageModel(Protocol):
    """Slippage Protocol — every concrete model implements this.

    Attributes:
        name: Short, stable identifier (logged into
            ``ExecutionEvent.meta["slippage_model"]`` so audit replays
            can identify which model produced a given fill price).

    The :meth:`apply` method is a *pure function* of its inputs plus
    the model's frozen config — no clock reads, no PRNG, no IO. This
    is what guarantees INV-15 replay determinism for synthetic fills
    that flow through the strategy arena and SIM step functions.
    """

    name: str

    def apply(
        self,
        side: Side,
        qty: float,
        mark_price: float,
        book: BookSnapshot | None = None,
    ) -> float:
        """Return the per-unit fill price for ``qty`` units on ``side``.

        ``mark_price`` is the reference / arrival price; ``book`` is an
        optional L2 snapshot used by depth-aware models. Models that
        ignore the book (e.g. constant-bps) accept any value (or
        ``None``).

        Side semantics:
        * ``BUY``  — buyer crosses the spread; fill price ≥ mark.
        * ``SELL`` — seller crosses the spread; fill price ≤ mark.
        * ``HOLD`` — pass-through; returns ``mark_price`` unchanged.
        """
        ...


# ---------------------------------------------------------------------------
# 1. Constant-bps slippage (the simplest reference; matches PaperBroker)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantBpsSlippage:
    """Linear-bps slippage applied on the same side as the order.

    Output: ``mark_price * (1 + sign * bps / 1e4)``.

    This is the deterministic reference model: the same shape the
    existing :class:`execution_engine.adapters.paper.PaperBroker`
    already uses for its slippage_bps field. It's quantity- and
    book-independent, so it's the right default for unit tests and
    the smoke-test arena (TEST-01 / INV-15).
    """

    bps: float = 0.0
    name: str = "constant_bps"

    def __post_init__(self) -> None:
        if not (self.bps >= 0.0):
            raise ValueError(f"ConstantBpsSlippage.bps must be >= 0 (NaN-safe), got {self.bps!r}")

    def apply(
        self,
        side: Side,
        qty: float,
        mark_price: float,
        book: BookSnapshot | None = None,
    ) -> float:
        del qty, book  # ignored by design
        if side is Side.HOLD or mark_price <= 0.0:
            return mark_price
        sign = 1.0 if side is Side.BUY else -1.0
        return mark_price * (1.0 + sign * self.bps / 1e4)


# ---------------------------------------------------------------------------
# 2. Book-walk slippage (adapted from hftbacktest market-order matching)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BookWalkSlippage:
    """Walk the opposite-side levels until ``qty`` is filled.

    Algorithm (verbatim from hftbacktest's local processor matching for
    market orders, modulo the Rust-vs-Python idiom): iterate the
    opposite-side levels in price priority, consume up to
    ``min(remaining, level.qty)`` at each level, accumulate
    ``cost = sum(level.price * fill_at_level)`` and ``filled = sum
    fills``. Return ``cost / filled`` as the volume-weighted average
    fill price.

    If the book runs out of depth before ``qty`` is reached and
    :attr:`fallback_to_mark_on_empty` is ``True`` (the default), the
    *unfilled* remainder is priced at ``mark_price`` and folded into
    the average — this matches hftbacktest's ``trade_back`` recovery
    when book depth is exhausted by historical replay. Set the flag
    to ``False`` to instead raise :class:`InsufficientLiquidity`.

    Notes:
    * If ``qty <= 0`` or ``side is HOLD``, returns ``mark_price``.
    * If ``book is None`` or has no opposite-side levels, returns
      ``mark_price`` (degenerate fallback — the caller should usually
      pair this with :class:`ConstantBpsSlippage` for empty-book
      scenarios).
    """

    fallback_to_mark_on_empty: bool = True
    name: str = "book_walk"

    def apply(
        self,
        side: Side,
        qty: float,
        mark_price: float,
        book: BookSnapshot | None = None,
    ) -> float:
        if side is Side.HOLD or qty <= 0.0 or mark_price <= 0.0:
            return mark_price
        if book is None:
            return mark_price
        levels = book.asks if side is Side.BUY else book.bids
        if not levels:
            return mark_price

        remaining = float(qty)
        cost = 0.0
        filled = 0.0
        for lvl in levels:
            if remaining <= 0.0:
                break
            take = min(remaining, lvl.qty)
            if take <= 0.0:
                continue
            cost += take * lvl.price
            filled += take
            remaining -= take

        if remaining > 0.0:
            # Book exhausted before the order is filled.
            if not self.fallback_to_mark_on_empty:
                raise InsufficientLiquidity(
                    f"BookWalkSlippage: book exhausted with "
                    f"{remaining!r} units unfilled (side={side.name}, "
                    f"qty={qty!r})"
                )
            cost += remaining * mark_price
            filled += remaining

        if filled <= 0.0:
            # All requested levels had zero qty + we're not falling
            # back — return mark to avoid a divide-by-zero. This path
            # is unreachable when :attr:`fallback_to_mark_on_empty` is
            # left at its default.
            return mark_price
        return cost / filled


class InsufficientLiquidity(RuntimeError):
    """Raised by :class:`BookWalkSlippage` when book depth runs out.

    Only raised when :attr:`BookWalkSlippage.fallback_to_mark_on_empty`
    is set to ``False``.
    """


# ---------------------------------------------------------------------------
# 3. Square-root impact (Kyle / Almgren-Chriss style)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SquareRootImpactSlippage:
    """Square-root market-impact slippage.

    Output: ``mark_price * (1 + sign * eta * sqrt(qty / adv))`` where
    ``eta`` is the impact coefficient (dimensionless) and ``adv`` is
    the average daily volume baseline (in base-asset units). This is
    the standard Kyle / Almgren-Chriss closed-form impact model used
    by hftbacktest's ``examples/sqrt_impact.py`` notebook.

    With sane parameters (``eta ≈ 0.1``, ``adv`` calibrated to the
    instrument's true ADV) it reproduces the empirical
    ``slippage ∝ sqrt(participation)`` regularity that comes out of
    the Bouchaud / Almgren impact literature.

    All inputs are validated to be strictly positive (NaN-safe via
    the IEEE-754 ``not (x > 0)`` pattern from PR #234).
    """

    eta: float = 0.1
    adv: float = 1.0
    name: str = "sqrt_impact"

    def __post_init__(self) -> None:
        if not (self.eta >= 0.0):
            raise ValueError(
                f"SquareRootImpactSlippage.eta must be >= 0 (NaN-safe), got {self.eta!r}"
            )
        if not (self.adv > 0.0):
            raise ValueError(
                f"SquareRootImpactSlippage.adv must be > 0 (NaN-safe), got {self.adv!r}"
            )

    def apply(
        self,
        side: Side,
        qty: float,
        mark_price: float,
        book: BookSnapshot | None = None,
    ) -> float:
        del book  # ignored by design
        if side is Side.HOLD or qty <= 0.0 or mark_price <= 0.0:
            return mark_price
        sign = 1.0 if side is Side.BUY else -1.0
        impact = self.eta * math.sqrt(qty / self.adv)
        return mark_price * (1.0 + sign * impact)


__all__ = [
    "BookLevel",
    "BookSnapshot",
    "BookWalkSlippage",
    "ConstantBpsSlippage",
    "InsufficientLiquidity",
    "NEW_PIP_DEPENDENCIES",
    "SlippageModel",
    "SquareRootImpactSlippage",
]
