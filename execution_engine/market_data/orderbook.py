# ADAPTED FROM: grantjenks/python-sortedcontainers
# (sortedcontainers/sorteddict.py — SortedDict with custom key
#  function; bisect-backed SortedKeyList for O(log n) insert and
#  O(1) best-of-side via peekitem(0) / peekitem(-1).)
"""A-18.1 sortedcontainers → ``execution_engine/market_data/orderbook.py``.

A stateful L2 order book that maintains per-side **price level maps**
behind a small ``PriceLevelMap`` Protocol seam. Two concrete
implementations are shipped:

* :class:`PurePyPriceLevelMap` — the default. A plain ``dict[float,
  float]`` of ``price -> qty`` with a sorted-views helper. ``O(n log
  n)`` per ``peekitem`` (cheap for the L2-depth-bounded books DIX
  consumes; typical depth ≤ 200 levels per side) but pulls **no**
  external pip dependency, so callers can use the order book on a
  bare-stdlib install.
* ``_SortedContainersPriceLevelMap`` — the sortedcontainers-backed
  variant. ``O(log n)`` insert / delete and ``O(1)`` best-of-side via
  ``SortedDict.peekitem`` (bids use a negative-key projection so the
  highest price sorts to position 0). Constructed only inside
  :func:`sortedcontainers_orderbook_factory`, which lazy-imports
  ``sortedcontainers.SortedDict`` so this module imports cleanly on a
  bare-stdlib install.

The book consumes the ``OrderBookSnapshot`` / ``BookDelta`` value
objects from the S-11 :mod:`execution_engine.market_data.aggregator`
adapter and projects each apply-step into a fresh frozen
``OrderBookSnapshot`` for downstream consumers (DIX never mutates a
published snapshot in place — INV-15).

Tier discipline — RUNTIME_SAFE (per A-18 spec line 1442):

* ``ts_ns`` is **always** caller-supplied (B-CLOCK / TimeAuthority
  chokepoint). The book never reads a clock.
* No ``random`` / ``time`` / ``datetime`` / ``asyncio`` / ``os`` import.
* No ``numpy`` / ``torch`` / ``polars`` / ``pandas`` import.
* No engine cross-import (``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``evolution_engine``).
* No typed-event construction (``SignalEvent`` / ``ExecutionEvent`` /
  ``SystemEvent`` / ``HazardEvent`` — B27 / B28 / INV-71). The book
  surfaces a frozen :class:`GapDetection` advisory record on a
  sequence gap; the caller (S-11 ingest layer) projects it into a
  ``HazardEvent``.
* All AST guards pinned by :mod:`tests.test_orderbook`.

Two replays with byte-identical input frames produce byte-identical
``OrderBookSnapshot`` tuples — pinned by 3-run determinism tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from execution_engine.market_data.aggregator import (
    BookDelta,
    OrderBookLevel,
    OrderBookSnapshot,
)

#: External pip dependencies introduced by this module.
#:
#: ``sortedcontainers`` is the canonical A-18 dependency but is
#: lazy-imported only inside :func:`sortedcontainers_orderbook_factory`.
#: Callers that stay on :class:`PurePyPriceLevelMap` need zero new
#: third-party packages.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("sortedcontainers",)


# ---------------------------------------------------------------------------
# Side discriminator
# ---------------------------------------------------------------------------


_BID = "bid"
_ASK = "ask"


# ---------------------------------------------------------------------------
# Gap-detection advisory record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GapDetection:
    """Frozen advisory record surfaced on an L2 sequence gap.

    ``apply_delta`` returns one of these (instead of an updated
    snapshot) when the delta's ``first_update_id`` is not the
    contiguous next chunk after ``last_update_id``. The caller
    (S-11 / ingest layer) projects this into a ``HazardEvent``
    (``HAZARD_KIND_BOOK_GAP``) — the order book itself never
    constructs typed bus events (B27 / B28 / INV-71).
    """

    ts_ns: int
    symbol: str
    venue: str
    last_known_update_id: int
    delta_first_update_id: int
    delta_final_update_id: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("GapDetection.symbol must be non-empty")
        if self.last_known_update_id < 0:
            raise ValueError(
                f"GapDetection.last_known_update_id must be >= 0, got {self.last_known_update_id!r}"
            )
        if self.delta_first_update_id < 0:
            raise ValueError(
                "GapDetection.delta_first_update_id must be >= 0, "
                f"got {self.delta_first_update_id!r}"
            )
        if self.delta_final_update_id < self.delta_first_update_id:
            raise ValueError("GapDetection.delta_final_update_id must be >= delta_first_update_id")


# ---------------------------------------------------------------------------
# PriceLevelMap Protocol seam
# ---------------------------------------------------------------------------


@runtime_checkable
class PriceLevelMap(Protocol):
    """Minimal sorted-by-price level map.

    The L2 order book holds two of these — one per side. The bid map
    sorts price *descending* (best price at position 0) and the ask
    map sorts price *ascending*. Implementations decide whether to
    use a sorted-keys helper (``PurePyPriceLevelMap``) or a real
    SortedDict (``_SortedContainersPriceLevelMap``).
    """

    def set(self, price: float, qty: float) -> None: ...

    def remove(self, price: float) -> None: ...

    def get(self, price: float) -> float: ...

    def has(self, price: float) -> bool: ...

    def peek_best(self) -> tuple[float, float] | None: ...

    def items_sorted(self) -> tuple[tuple[float, float], ...]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# Pure-Python default implementation
# ---------------------------------------------------------------------------


class PurePyPriceLevelMap:
    """``dict[float, float]`` backed sorted-view price-level map.

    Per-side sort direction is fixed at construction time. ``set`` /
    ``remove`` are ``O(1)`` average; ``peek_best`` and
    ``items_sorted`` are ``O(n log n)`` over the level count.

    For DIX L2 depth-N books (N is the configured cap, typically
    ``50 / 100 / 200``) this is more than fast enough. The
    :func:`sortedcontainers_orderbook_factory` variant is an
    asymptotic-only win on books with thousands of levels.
    """

    __slots__ = ("_levels", "_descending")

    def __init__(self, *, descending: bool) -> None:
        self._levels: dict[float, float] = {}
        self._descending: bool = bool(descending)

    def set(self, price: float, qty: float) -> None:
        self._validate_price(price)
        self._validate_qty(qty)
        self._levels[float(price)] = float(qty)

    def remove(self, price: float) -> None:
        self._validate_price(price)
        self._levels.pop(float(price), None)

    def get(self, price: float) -> float:
        self._validate_price(price)
        return float(self._levels.get(float(price), 0.0))

    def has(self, price: float) -> bool:
        self._validate_price(price)
        return float(price) in self._levels

    def peek_best(self) -> tuple[float, float] | None:
        if not self._levels:
            return None
        if self._descending:
            best_price = max(self._levels)
        else:
            best_price = min(self._levels)
        return (best_price, self._levels[best_price])

    def items_sorted(self) -> tuple[tuple[float, float], ...]:
        prices = sorted(self._levels, reverse=self._descending)
        return tuple((p, self._levels[p]) for p in prices)

    def clear(self) -> None:
        self._levels.clear()

    def __len__(self) -> int:
        return len(self._levels)

    @staticmethod
    def _validate_price(price: float) -> None:
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise TypeError(f"PriceLevelMap price must be int|float, got {type(price)!r}")
        if float(price) <= 0.0:
            raise ValueError(f"PriceLevelMap.price must be > 0, got {price!r}")

    @staticmethod
    def _validate_qty(qty: float) -> None:
        if isinstance(qty, bool) or not isinstance(qty, (int, float)):
            raise TypeError(f"PriceLevelMap qty must be int|float, got {type(qty)!r}")
        if float(qty) < 0.0:
            raise ValueError(f"PriceLevelMap.qty must be >= 0, got {qty!r}")


# ---------------------------------------------------------------------------
# sortedcontainers-backed implementation (lazy)
# ---------------------------------------------------------------------------


class _SortedContainersPriceLevelMap:
    """``sortedcontainers.SortedDict`` backed price-level map.

    Uses ``SortedDict`` directly: the bid variant stores entries
    keyed by ``-price`` so the largest price sorts to position 0 and
    ``peekitem(0)`` returns the best bid in ``O(1)``; the ask variant
    stores ``+price`` directly so the smallest sorts to position 0.

    Constructed only via :func:`sortedcontainers_orderbook_factory`.
    """

    __slots__ = ("_sd", "_descending")

    def __init__(self, *, sd: Any, descending: bool) -> None:
        # ``sd`` is a ``sortedcontainers.SortedDict`` instance (lazy).
        self._sd = sd
        self._descending: bool = bool(descending)

    def set(self, price: float, qty: float) -> None:
        PurePyPriceLevelMap._validate_price(price)
        PurePyPriceLevelMap._validate_qty(qty)
        self._sd[self._key(float(price))] = float(qty)

    def remove(self, price: float) -> None:
        PurePyPriceLevelMap._validate_price(price)
        self._sd.pop(self._key(float(price)), None)

    def get(self, price: float) -> float:
        PurePyPriceLevelMap._validate_price(price)
        return float(self._sd.get(self._key(float(price)), 0.0))

    def has(self, price: float) -> bool:
        PurePyPriceLevelMap._validate_price(price)
        return self._key(float(price)) in self._sd

    def peek_best(self) -> tuple[float, float] | None:
        if not self._sd:
            return None
        key, qty = self._sd.peekitem(0)
        price = -key if self._descending else key
        return (float(price), float(qty))

    def items_sorted(self) -> tuple[tuple[float, float], ...]:
        if self._descending:
            return tuple((float(-k), float(v)) for k, v in self._sd.items())
        return tuple((float(k), float(v)) for k, v in self._sd.items())

    def clear(self) -> None:
        self._sd.clear()

    def __len__(self) -> int:
        return len(self._sd)

    def _key(self, price: float) -> float:
        return -price if self._descending else price


# ---------------------------------------------------------------------------
# L2 order book coordinator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ApplyOutcome:
    """Internal — result of one ``apply_*`` step."""

    snapshot: OrderBookSnapshot | None
    gap: GapDetection | None


@dataclass(frozen=True, slots=True)
class OrderBookApplyResult:
    """Public — result of one ``apply_delta`` step.

    Exactly one of ``snapshot`` / ``gap`` is non-``None``. Callers
    branch on ``gap is not None`` for the hazard projection path.
    """

    snapshot: OrderBookSnapshot | None
    gap: GapDetection | None

    def __post_init__(self) -> None:
        if (self.snapshot is None) == (self.gap is None):
            raise ValueError(
                "OrderBookApplyResult must have exactly one of snapshot / gap populated"
            )


class L2OrderBook:
    """Stateful L2 order book.

    Apply a :class:`OrderBookSnapshot` to seed the book, then fold
    each contiguous :class:`BookDelta` to advance ``last_update_id``.
    Each successful apply step returns a fresh frozen
    :class:`OrderBookSnapshot`. On a sequence gap, :meth:`apply_delta`
    returns a :class:`GapDetection` advisory record (not a snapshot)
    and leaves the book state unchanged — the caller is expected to
    re-seed with a fresh snapshot.

    The two per-side :class:`PriceLevelMap` instances are injected so
    callers pick the backend (pure-Python default or sortedcontainers
    via the factory).
    """

    __slots__ = (
        "_bids",
        "_asks",
        "_symbol",
        "_venue",
        "_last_update_id",
        "_max_depth",
    )

    def __init__(
        self,
        *,
        bids: PriceLevelMap,
        asks: PriceLevelMap,
        symbol: str,
        venue: str,
        max_depth: int = 200,
    ) -> None:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("L2OrderBook.symbol must be non-empty string")
        if not isinstance(venue, str) or not venue:
            raise ValueError("L2OrderBook.venue must be non-empty string")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise TypeError("L2OrderBook.max_depth must be int")
        if max_depth <= 0:
            raise ValueError(f"L2OrderBook.max_depth must be > 0, got {max_depth!r}")
        self._bids: PriceLevelMap = bids
        self._asks: PriceLevelMap = asks
        self._symbol: str = symbol
        self._venue: str = venue
        self._last_update_id: int = -1
        self._max_depth: int = max_depth

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def venue(self) -> str:
        return self._venue

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def __len__(self) -> int:
        return len(self._bids) + len(self._asks)

    # -- best-of-side --------------------------------------------------

    def best_bid(self) -> OrderBookLevel | None:
        item = self._bids.peek_best()
        if item is None:
            return None
        return OrderBookLevel(price=item[0], qty=item[1])

    def best_ask(self) -> OrderBookLevel | None:
        item = self._asks.peek_best()
        if item is None:
            return None
        return OrderBookLevel(price=item[0], qty=item[1])

    def mid(self) -> float | None:
        bid = self._bids.peek_best()
        ask = self._asks.peek_best()
        if bid is None or ask is None:
            return None
        return (bid[0] + ask[0]) / 2.0

    def spread(self) -> float | None:
        bid = self._bids.peek_best()
        ask = self._asks.peek_best()
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    # -- seed / apply --------------------------------------------------

    def apply_snapshot(
        self,
        snapshot: OrderBookSnapshot,
    ) -> OrderBookSnapshot:
        """Re-seed the book from a full L2 snapshot.

        Clears existing state, loads every level, and returns a fresh
        snapshot capped to ``max_depth`` per side. Symbol / venue must
        match the book's configured pair; mismatches raise
        ``ValueError``.
        """
        if snapshot.symbol != self._symbol:
            raise ValueError(
                "apply_snapshot.symbol mismatch: book="
                f"{self._symbol!r} snapshot={snapshot.symbol!r}"
            )
        if snapshot.venue != self._venue:
            raise ValueError(
                f"apply_snapshot.venue mismatch: book={self._venue!r} snapshot={snapshot.venue!r}"
            )
        self._bids.clear()
        self._asks.clear()
        for lvl in snapshot.bids:
            if lvl.qty > 0.0:
                self._bids.set(lvl.price, lvl.qty)
        for lvl in snapshot.asks:
            if lvl.qty > 0.0:
                self._asks.set(lvl.price, lvl.qty)
        self._last_update_id = snapshot.last_update_id
        return self._project(snapshot.ts_ns)

    def apply_delta(
        self,
        delta: BookDelta,
    ) -> OrderBookApplyResult:
        """Fold one contiguous delta into the book.

        Binance L2 contract: ``first_update_id <= last_update_id + 1
        <= final_update_id``. On gap detection the book is left
        unchanged and a :class:`GapDetection` is returned; otherwise
        the book is updated and a fresh snapshot is returned.
        """
        if delta.symbol != self._symbol:
            raise ValueError(
                f"apply_delta.symbol mismatch: book={self._symbol!r} delta={delta.symbol!r}"
            )
        if delta.venue != self._venue:
            raise ValueError(
                f"apply_delta.venue mismatch: book={self._venue!r} delta={delta.venue!r}"
            )
        if self._last_update_id < 0:
            raise RuntimeError(
                "L2OrderBook.apply_delta requires apply_snapshot first "
                "(no seed snapshot has been applied)"
            )
        expected = self._last_update_id + 1
        if not (delta.first_update_id <= expected <= delta.final_update_id):
            return OrderBookApplyResult(
                snapshot=None,
                gap=GapDetection(
                    ts_ns=delta.ts_ns,
                    symbol=self._symbol,
                    venue=self._venue,
                    last_known_update_id=self._last_update_id,
                    delta_first_update_id=delta.first_update_id,
                    delta_final_update_id=delta.final_update_id,
                ),
            )
        for lvl in delta.bid_updates:
            self._apply_level(self._bids, lvl)
        for lvl in delta.ask_updates:
            self._apply_level(self._asks, lvl)
        self._last_update_id = delta.final_update_id
        return OrderBookApplyResult(
            snapshot=self._project(delta.ts_ns),
            gap=None,
        )

    # -- projection ----------------------------------------------------

    def project_snapshot(self, ts_ns: int) -> OrderBookSnapshot:
        """Project a fresh frozen snapshot at the caller's ``ts_ns``."""
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError("project_snapshot.ts_ns must be int")
        if ts_ns < 0:
            raise ValueError(f"project_snapshot.ts_ns must be >= 0, got {ts_ns!r}")
        return self._project(ts_ns)

    def top_n_bids(self, n: int) -> tuple[OrderBookLevel, ...]:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("top_n_bids.n must be int")
        if n < 0:
            raise ValueError(f"top_n_bids.n must be >= 0, got {n!r}")
        return self._truncate_levels(self._bids, n)

    def top_n_asks(self, n: int) -> tuple[OrderBookLevel, ...]:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("top_n_asks.n must be int")
        if n < 0:
            raise ValueError(f"top_n_asks.n must be >= 0, got {n!r}")
        return self._truncate_levels(self._asks, n)

    # -- internal ------------------------------------------------------

    @staticmethod
    def _apply_level(side: PriceLevelMap, lvl: OrderBookLevel) -> None:
        if lvl.qty == 0.0:
            side.remove(lvl.price)
        else:
            side.set(lvl.price, lvl.qty)

    def _project(self, ts_ns: int) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            ts_ns=ts_ns,
            symbol=self._symbol,
            last_update_id=max(self._last_update_id, 0),
            bids=self._truncate_levels(self._bids, self._max_depth),
            asks=self._truncate_levels(self._asks, self._max_depth),
            venue=self._venue,
        )

    @staticmethod
    def _truncate_levels(
        side: PriceLevelMap,
        n: int,
    ) -> tuple[OrderBookLevel, ...]:
        if n <= 0:
            return ()
        out: list[OrderBookLevel] = []
        for price, qty in side.items_sorted():
            if qty <= 0.0:
                continue
            out.append(OrderBookLevel(price=price, qty=qty))
            if len(out) >= n:
                break
        return tuple(out)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def pure_python_orderbook_factory(
    *,
    symbol: str,
    venue: str,
    max_depth: int = 200,
) -> L2OrderBook:
    """Construct a :class:`L2OrderBook` over :class:`PurePyPriceLevelMap`.

    Pure-stdlib path. No external imports.
    """
    return L2OrderBook(
        bids=PurePyPriceLevelMap(descending=True),
        asks=PurePyPriceLevelMap(descending=False),
        symbol=symbol,
        venue=venue,
        max_depth=max_depth,
    )


def sortedcontainers_orderbook_factory(
    *,
    symbol: str,
    venue: str,
    max_depth: int = 200,
) -> L2OrderBook:
    """Construct a :class:`L2OrderBook` over the sortedcontainers backend.

    Lazy-imports ``sortedcontainers.SortedDict`` only inside this
    function body. Raises :class:`RuntimeError` (never propagating
    the underlying :class:`ImportError`) when the package is not
    installed so callers can fail-soft to
    :func:`pure_python_orderbook_factory`.
    """
    try:
        from sortedcontainers import SortedDict  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "sortedcontainers_orderbook_factory: sortedcontainers "
            "package is not installed; install via "
            "`pip install sortedcontainers` or fall back to "
            "pure_python_orderbook_factory()"
        ) from exc
    return L2OrderBook(
        bids=_SortedContainersPriceLevelMap(
            sd=SortedDict(),
            descending=True,
        ),
        asks=_SortedContainersPriceLevelMap(
            sd=SortedDict(),
            descending=False,
        ),
        symbol=symbol,
        venue=venue,
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Convenience replay helper (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Pure summary of a deterministic snapshot+delta replay."""

    snapshots: tuple[OrderBookSnapshot, ...] = field(default_factory=tuple)
    gaps: tuple[GapDetection, ...] = field(default_factory=tuple)


def replay_l2(
    *,
    seed: OrderBookSnapshot,
    deltas: Iterable[BookDelta],
    symbol: str,
    venue: str,
    max_depth: int = 200,
) -> ReplaySummary:
    """Replay a snapshot + delta stream into a :class:`ReplaySummary`.

    Pure helper used by the determinism tests. Returns the full
    sequence of intermediate snapshots / gaps so two byte-identical
    inputs produce byte-identical outputs (INV-15).
    """
    book = pure_python_orderbook_factory(
        symbol=symbol,
        venue=venue,
        max_depth=max_depth,
    )
    snapshots: list[OrderBookSnapshot] = [book.apply_snapshot(seed)]
    gaps: list[GapDetection] = []
    for delta in _consume(deltas):
        result = book.apply_delta(delta)
        if result.snapshot is not None:
            snapshots.append(result.snapshot)
        if result.gap is not None:
            gaps.append(result.gap)
    return ReplaySummary(
        snapshots=tuple(snapshots),
        gaps=tuple(gaps),
    )


def _consume(deltas: Iterable[BookDelta]) -> Iterator[BookDelta]:
    """Yield deltas in caller order (no internal sort).

    Order is part of the input contract — the book is sequence-tagged
    by ``first_update_id`` / ``final_update_id`` so re-ordering here
    would mask gap detection.
    """
    for delta in deltas:
        if not isinstance(delta, BookDelta):
            raise TypeError(f"replay_l2.deltas must yield BookDelta, got {type(delta)!r}")
        yield delta
