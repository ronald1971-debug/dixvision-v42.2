# ADAPTED FROM: bmoscon/cryptofeed
# (cryptofeed/exchange/binance.py — _trade() handler, _book() L2/L3
#  snapshot+delta gap-detection pattern; cryptofeed/connection.py —
#  AsyncConnection exponential-backoff reconnect; cryptofeed/types.py —
#  Trade / OrderBook / Ticker frozen value-object shapes.)
"""S-11 cryptofeed → ``execution_engine/market_data/aggregator.py``.

Adapts cryptofeed's market-data ingestion patterns into DIX:

* **Trade / OrderBook value objects** — frozen+slotted dataclasses
  modeled on ``cryptofeed/types.py``. Bid/ask levels are kept as
  ``tuple[OrderBookLevel, ...]`` rather than dicts so two replays with
  identical inputs produce byte-identical snapshots (INV-15).
* **Pure parsers** — ``parse_binance_trade`` /
  ``parse_binance_book_snapshot`` / ``parse_binance_book_delta`` map
  Binance public-WS frames into the value objects. Each returns
  ``None`` (never raises) on a malformed / non-data frame so a pump
  loop can silently skip subscription acks, heartbeats, etc.
* **L2 ``OrderBookAggregator``** — applies a snapshot, then folds
  delta updates while enforcing Binance's published L2 contract
  (``U <= last_update_id + 1 <= u``). On gap detection the aggregator
  surfaces a frozen :class:`BookGap` record so the caller can emit a
  ``HazardEvent`` (``HAZARD_KIND_BOOK_GAP``) onto the typed bus and
  trigger a snapshot resync.
* **Reconnect backoff** — ``next_reconnect_delay_s`` is a pure
  exponential-backoff calculator (no clock, no PRNG, no I/O) so a
  caller's reconnect loop stays ledger-replayable.

This file contains **no asyncio I/O, no network code, no clock
reads**. Network pumps live in the sibling ``ui/feeds/`` layer and
inject the parsers + aggregator from this module so the parsing /
aggregation surface is RUNTIME_SAFE per the S-11 spec.

INV-15 (replay determinism): every public function is pure.
``ts_ns`` is supplied by the caller (TimeAuthority chokepoint, B-CLOCK).
``OrderBookSnapshot.bids`` is sorted ``price DESC, qty DESC``;
``OrderBookSnapshot.asks`` is sorted ``price ASC, qty DESC`` so two
inputs that differ only in level-insertion order produce identical
snapshots.

Authority constraints (S-11 RUNTIME_SAFE tier):

* No clock reads (caller-supplied ``ts_ns``).
* No polars import (B-POLARS-banned in execution_engine).
* No numpy import in the flat path (Python floats only).
* No external pip dependency: ``NEW_PIP_DEPENDENCIES = ()``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: New non-stdlib pip dependencies introduced by this module.
#:
#: Empty tuple — cryptofeed itself is **not** imported (XStamper
#: License flag); only its parsing / aggregation patterns are
#: adapted in pure Python. The optional async pump that callers
#: layer on top will pull in ``websockets`` lazily, but that is the
#: pump's dependency, not this module's.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Value objects (frozen, slotted, Mapping-typed metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trade:
    """One trade print from a venue.

    Mirrors ``cryptofeed/types.py:Trade`` minus the runtime callback
    glue. ``ts_ns`` is caller-supplied (B-CLOCK) — never derived from
    the payload's exchange timestamp because exchanges return ms
    precision and that would alias replay-determinism.
    """

    ts_ns: int
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    qty: float
    trade_id: str
    venue: str

    def __post_init__(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Trade.side must be 'BUY' or 'SELL', got {self.side!r}")
        if self.price <= 0:
            raise ValueError(f"Trade.price must be > 0, got {self.price!r}")
        if self.qty <= 0:
            raise ValueError(f"Trade.qty must be > 0, got {self.qty!r}")
        if not self.symbol:
            raise ValueError("Trade.symbol must be non-empty")
        if not self.trade_id:
            raise ValueError("Trade.trade_id must be non-empty")


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """One price level on either side of an L2 book."""

    price: float
    qty: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"OrderBookLevel.price must be > 0, got {self.price!r}")
        if self.qty < 0:
            raise ValueError(f"OrderBookLevel.qty must be >= 0, got {self.qty!r}")


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Full L2 order book state at one instant.

    Bids are sorted ``price DESC`` (best bid first); asks are sorted
    ``price ASC`` (best ask first). The aggregator maintains this
    invariant so callers can index ``bids[0]`` / ``asks[0]`` for the
    top-of-book without a fresh sort.
    """

    ts_ns: int
    symbol: str
    last_update_id: int
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    venue: str

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("OrderBookSnapshot.symbol must be non-empty")
        if self.last_update_id < 0:
            raise ValueError(
                f"OrderBookSnapshot.last_update_id must be >= 0, got {self.last_update_id!r}"
            )
        # Sort invariants — checked, not auto-sorted, so the caller
        # explicitly funnels through ``_make_snapshot`` (or apply_*
        # which both go through it).
        for i in range(1, len(self.bids)):
            if self.bids[i - 1].price < self.bids[i].price:
                raise ValueError("OrderBookSnapshot.bids must be sorted price DESC")
        for i in range(1, len(self.asks)):
            if self.asks[i - 1].price > self.asks[i].price:
                raise ValueError("OrderBookSnapshot.asks must be sorted price ASC")

    def best_bid(self) -> OrderBookLevel | None:
        """Return the highest-price bid (or ``None`` if the bid side is empty)."""
        return self.bids[0] if self.bids else None

    def best_ask(self) -> OrderBookLevel | None:
        """Return the lowest-price ask (or ``None`` if the ask side is empty)."""
        return self.asks[0] if self.asks else None

    def mid(self) -> float | None:
        """Return ``(best_bid + best_ask) / 2`` (or ``None`` if either side is empty)."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid.price + ask.price) / 2.0


@dataclass(frozen=True, slots=True)
class BookDelta:
    """One ``depthUpdate`` payload — sequence-tagged level changes.

    ``first_update_id`` (Binance's ``U``) and ``final_update_id``
    (Binance's ``u``) bound the contiguous sequence range covered by
    this delta. ``bid_updates`` / ``ask_updates`` are absolute level
    states — qty=0 means "remove this level" per the Binance L2 spec.
    """

    ts_ns: int
    symbol: str
    first_update_id: int
    final_update_id: int
    bid_updates: tuple[OrderBookLevel, ...]
    ask_updates: tuple[OrderBookLevel, ...]
    venue: str

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("BookDelta.symbol must be non-empty")
        if self.first_update_id < 0:
            raise ValueError(
                f"BookDelta.first_update_id must be >= 0, got {self.first_update_id!r}"
            )
        if self.final_update_id < self.first_update_id:
            raise ValueError("BookDelta.final_update_id must be >= first_update_id")


@dataclass(frozen=True, slots=True)
class BookGap:
    """Detected sequence-gap on the L2 book.

    The aggregator surfaces this when the next delta's
    ``first_update_id`` does not satisfy
    ``first_update_id <= last_update_id + 1 <= final_update_id``,
    i.e. either the delta starts after the snapshot's tail or it
    straddles a missing chunk. Callers project this into a
    :class:`HazardEvent` (``HAZARD_KIND_BOOK_GAP``) so SystemEngine
    can throttle execution while the snapshot is resynced.
    """

    ts_ns: int
    symbol: str
    venue: str
    last_known_update_id: int
    delta_first_update_id: int
    delta_final_update_id: int


# ---------------------------------------------------------------------------
# Pure parsers (Binance public WS frame shapes)
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_inner(payload: Any) -> Mapping[str, Any] | None:
    """Unwrap Binance's combined-stream envelope ``{"stream":..,"data":..}``.

    Returns the inner data mapping when present; otherwise returns the
    payload unchanged when it already looks like a mapping; otherwise
    ``None``.
    """
    if not isinstance(payload, Mapping):
        return None
    inner = payload.get("data")
    if isinstance(inner, Mapping):
        return inner
    return payload


def _parse_levels(
    raw: Any,
) -> tuple[OrderBookLevel, ...] | None:
    """Project a Binance ``[["price","qty"], ...]`` array into levels.

    Returns ``None`` if the array is malformed (so the parent parser
    can emit ``None`` rather than raise on a stray frame). qty=0 is
    accepted — it represents a level removal and is filtered later by
    the aggregator.
    """
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    out: list[OrderBookLevel] = []
    for entry in raw:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            return None
        price = _safe_float(entry[0])
        qty = _safe_float(entry[1])
        if price is None or qty is None:
            return None
        if price <= 0 or qty < 0:
            return None
        out.append(OrderBookLevel(price=price, qty=qty))
    return tuple(out)


def parse_binance_trade(payload: Any, *, ts_ns: int, venue: str = "BINANCE") -> Trade | None:
    """Project one Binance ``trade`` WS frame into a :class:`Trade`.

    Frame shape (per Binance spot docs)::

        {"e":"trade","s":"BTCUSDT","t":12345,
         "p":"50000.0","q":"0.001","m":false}

    ``m`` (Binance's ``isBuyerMaker``) is inverted to derive the
    aggressor side: ``m=False`` ⇒ taker is BUY (lifted ask),
    ``m=True`` ⇒ taker is SELL (hit bid). Returns ``None`` on any
    missing / malformed field rather than raising — the pump loop
    silently skips malformed frames.
    """
    inner = _coerce_inner(payload)
    if inner is None or inner.get("e") != "trade":
        return None
    symbol_raw = inner.get("s")
    trade_id_raw = inner.get("t")
    price = _safe_float(inner.get("p"))
    qty = _safe_float(inner.get("q"))
    is_buyer_maker = inner.get("m")
    if not isinstance(symbol_raw, str) or not symbol_raw:
        return None
    if trade_id_raw is None:
        return None
    if price is None or price <= 0:
        return None
    if qty is None or qty <= 0:
        return None
    if not isinstance(is_buyer_maker, bool):
        return None
    side = "SELL" if is_buyer_maker else "BUY"
    return Trade(
        ts_ns=ts_ns,
        symbol=symbol_raw,
        side=side,
        price=price,
        qty=qty,
        trade_id=str(trade_id_raw),
        venue=venue,
    )


def parse_binance_book_snapshot(
    payload: Any, *, ts_ns: int, symbol: str, venue: str = "BINANCE"
) -> OrderBookSnapshot | None:
    """Project a Binance REST ``/api/v3/depth`` snapshot into a snapshot.

    Snapshot shape::

        {"lastUpdateId": 12345, "bids":[["p","q"],...], "asks":[...]}

    ``symbol`` is supplied by the caller because the REST snapshot
    payload itself omits the symbol field. Returns ``None`` on any
    missing / malformed field.
    """
    if not isinstance(payload, Mapping):
        return None
    last_update_id = _safe_int(payload.get("lastUpdateId"))
    if last_update_id is None or last_update_id < 0:
        return None
    bids = _parse_levels(payload.get("bids"))
    asks = _parse_levels(payload.get("asks"))
    if bids is None or asks is None:
        return None
    if not symbol:
        return None
    return _make_snapshot(
        ts_ns=ts_ns,
        symbol=symbol,
        last_update_id=last_update_id,
        bid_levels=bids,
        ask_levels=asks,
        venue=venue,
    )


def parse_binance_book_delta(
    payload: Any, *, ts_ns: int, venue: str = "BINANCE"
) -> BookDelta | None:
    """Project one Binance ``depthUpdate`` WS frame into a :class:`BookDelta`.

    Frame shape::

        {"e":"depthUpdate","s":"BTCUSDT",
         "U":1, "u":3, "b":[["p","q"],...], "a":[...]}

    Returns ``None`` on any missing / malformed field.
    """
    inner = _coerce_inner(payload)
    if inner is None or inner.get("e") != "depthUpdate":
        return None
    symbol_raw = inner.get("s")
    first_id = _safe_int(inner.get("U"))
    final_id = _safe_int(inner.get("u"))
    if not isinstance(symbol_raw, str) or not symbol_raw:
        return None
    if first_id is None or final_id is None:
        return None
    if first_id < 0 or final_id < first_id:
        return None
    bid_updates = _parse_levels(inner.get("b"))
    ask_updates = _parse_levels(inner.get("a"))
    if bid_updates is None or ask_updates is None:
        return None
    return BookDelta(
        ts_ns=ts_ns,
        symbol=symbol_raw,
        first_update_id=first_id,
        final_update_id=final_id,
        bid_updates=bid_updates,
        ask_updates=ask_updates,
        venue=venue,
    )


# ---------------------------------------------------------------------------
# L2 OrderBookAggregator (snapshot + delta with sequence-gap detection)
# ---------------------------------------------------------------------------


def _merge_levels(
    side: tuple[OrderBookLevel, ...],
    updates: Iterable[OrderBookLevel],
    *,
    descending: bool,
) -> tuple[OrderBookLevel, ...]:
    """Merge ``updates`` into ``side`` and return the new sorted side.

    Per the Binance L2 spec, an update with ``qty=0`` removes the
    level. Updates with ``qty>0`` either insert a new level or
    replace the existing one at that price. The result is sorted
    ``price DESC`` if ``descending=True`` (bids), else ``price ASC``
    (asks). Equal-price ties are broken by ``qty DESC`` so two
    inputs that differ only by insertion order produce identical
    output (INV-15).
    """
    book: dict[float, float] = {lvl.price: lvl.qty for lvl in side}
    for upd in updates:
        if upd.qty == 0:
            book.pop(upd.price, None)
        else:
            book[upd.price] = upd.qty
    levels = [OrderBookLevel(price=price, qty=qty) for price, qty in book.items()]
    levels.sort(
        key=lambda lvl: (lvl.price, lvl.qty),
        reverse=descending,
    )
    if not descending:
        # Asks: ascending price, descending qty within ties.
        levels.sort(key=lambda lvl: lvl.qty, reverse=True)
        levels.sort(key=lambda lvl: lvl.price)
    return tuple(levels)


def _make_snapshot(
    *,
    ts_ns: int,
    symbol: str,
    last_update_id: int,
    bid_levels: Iterable[OrderBookLevel],
    ask_levels: Iterable[OrderBookLevel],
    venue: str,
) -> OrderBookSnapshot:
    """Construct a snapshot with the canonical sort applied to both sides."""
    bids = _merge_levels((), bid_levels, descending=True)
    asks = _merge_levels((), ask_levels, descending=False)
    return OrderBookSnapshot(
        ts_ns=ts_ns,
        symbol=symbol,
        last_update_id=last_update_id,
        bids=bids,
        asks=asks,
        venue=venue,
    )


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Outcome of :meth:`OrderBookAggregator.apply_delta`.

    Exactly one of ``snapshot`` / ``gap`` / ``stale`` is populated:

    * ``snapshot`` — delta applied cleanly; the new top-of-book is
      reflected in ``snapshot``.
    * ``gap`` — sequence gap detected; the aggregator's internal
      state is **frozen** at the last known good snapshot, and the
      caller must resync from a fresh REST snapshot before applying
      further deltas.
    * ``stale`` — delta was older than the current snapshot's
      ``last_update_id`` (``final_update_id <= last_update_id``).
      Common during the initial catch-up window between snapshot
      fetch and delta subscription. The caller can safely ignore.
    """

    snapshot: OrderBookSnapshot | None
    gap: BookGap | None
    stale: bool

    def is_ok(self) -> bool:
        return self.snapshot is not None and self.gap is None and not self.stale


class OrderBookAggregator:
    """L2 order book — snapshot + sequence of deltas → current snapshot.

    Lifecycle:

    1. Construct with ``OrderBookAggregator(symbol="BTCUSDT")`` — no
       book state yet, all accessors return ``None`` /
       ``ApplyResult(stale=True, ...)``.
    2. Caller fetches REST snapshot, passes it to
       :meth:`apply_snapshot`. State is now initialised.
    3. Caller streams ``BookDelta`` frames from the WS pump and
       passes each to :meth:`apply_delta`. The aggregator validates
       the Binance L2 contract and returns either an updated
       :class:`OrderBookSnapshot` (success), a :class:`BookGap`
       (resync required), or ``stale=True`` (delta predates
       snapshot — ignore).
    4. If a gap is reported, the caller emits a hazard, fetches a
       fresh REST snapshot, and calls :meth:`apply_snapshot` again.

    The aggregator owns no I/O and reads no clock — every input is
    caller-supplied.
    """

    def __init__(self, *, symbol: str) -> None:
        if not symbol:
            raise ValueError("OrderBookAggregator.symbol must be non-empty")
        self._symbol = symbol
        self._snapshot: OrderBookSnapshot | None = None

    @property
    def symbol(self) -> str:
        return self._symbol

    def current(self) -> OrderBookSnapshot | None:
        """Return the current snapshot (or ``None`` if uninitialised)."""
        return self._snapshot

    def apply_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Initialise / resync the aggregator from a REST snapshot.

        Raises :class:`ValueError` if ``snapshot.symbol`` does not
        match the aggregator's bound symbol — silently switching
        symbols is a class of bug we want to catch loudly.
        """
        if snapshot.symbol != self._symbol:
            raise ValueError(
                f"OrderBookAggregator({self._symbol!r}).apply_snapshot: "
                f"received symbol {snapshot.symbol!r}"
            )
        # Re-canonicalise sort order even though the dataclass
        # validator already enforces it — this guards against future
        # callers passing a hand-rolled snapshot from another path.
        self._snapshot = _make_snapshot(
            ts_ns=snapshot.ts_ns,
            symbol=snapshot.symbol,
            last_update_id=snapshot.last_update_id,
            bid_levels=snapshot.bids,
            ask_levels=snapshot.asks,
            venue=snapshot.venue,
        )

    def apply_delta(self, delta: BookDelta) -> ApplyResult:
        """Fold one delta into the book; report sequence-gap if detected.

        Binance L2 contract:

        * **Stale**: ``delta.final_update_id <= last_update_id``  ⇒
          the delta predates the snapshot — drop silently.
        * **Clean**: ``delta.first_update_id <= last_update_id + 1
          <= delta.final_update_id`` ⇒ apply.
        * **Gap**: anything else ⇒ surface :class:`BookGap`, freeze
          state, await caller resync.

        Raises :class:`RuntimeError` if called before
        :meth:`apply_snapshot` (the aggregator has no book state to
        fold into).

        Raises :class:`ValueError` on a symbol-mismatch (same
        rationale as :meth:`apply_snapshot`).
        """
        if self._snapshot is None:
            raise RuntimeError("OrderBookAggregator.apply_delta: must apply_snapshot first")
        if delta.symbol != self._symbol:
            raise ValueError(
                f"OrderBookAggregator({self._symbol!r}).apply_delta: "
                f"received symbol {delta.symbol!r}"
            )
        last_id = self._snapshot.last_update_id
        if delta.final_update_id <= last_id:
            return ApplyResult(snapshot=None, gap=None, stale=True)
        # Clean-window check: the *expected* next id (``last_id+1``)
        # must lie inside ``[first_update_id, final_update_id]``.
        if not (delta.first_update_id <= last_id + 1 <= delta.final_update_id):
            gap = BookGap(
                ts_ns=delta.ts_ns,
                symbol=delta.symbol,
                venue=delta.venue,
                last_known_update_id=last_id,
                delta_first_update_id=delta.first_update_id,
                delta_final_update_id=delta.final_update_id,
            )
            return ApplyResult(snapshot=None, gap=gap, stale=False)
        new_bids = _merge_levels(self._snapshot.bids, delta.bid_updates, descending=True)
        new_asks = _merge_levels(self._snapshot.asks, delta.ask_updates, descending=False)
        new_snapshot = OrderBookSnapshot(
            ts_ns=delta.ts_ns,
            symbol=self._symbol,
            last_update_id=delta.final_update_id,
            bids=new_bids,
            asks=new_asks,
            venue=self._snapshot.venue,
        )
        self._snapshot = new_snapshot
        return ApplyResult(snapshot=new_snapshot, gap=None, stale=False)


# ---------------------------------------------------------------------------
# Reconnect backoff helper (pure)
# ---------------------------------------------------------------------------


#: Default reconnect backoff floor (seconds) — matches
#: ``ui/feeds/binance_public_ws.py`` so existing harness configuration
#: aligns.
DEFAULT_RECONNECT_DELAY_FLOOR_S: float = 5.0

#: Default reconnect backoff ceiling (seconds).
DEFAULT_RECONNECT_DELAY_CEILING_S: float = 60.0

#: Default exponential factor.
DEFAULT_RECONNECT_BACKOFF_FACTOR: float = 2.0


def next_reconnect_delay_s(
    *,
    attempt: int,
    floor_s: float = DEFAULT_RECONNECT_DELAY_FLOOR_S,
    ceiling_s: float = DEFAULT_RECONNECT_DELAY_CEILING_S,
    factor: float = DEFAULT_RECONNECT_BACKOFF_FACTOR,
) -> float:
    """Compute the reconnect delay for the given (zero-indexed) attempt.

    Pure exponential backoff: ``floor_s * factor**attempt`` clamped
    to ``[floor_s, ceiling_s]``. ``attempt=0`` returns ``floor_s``.
    No clock, no PRNG — caller-side jitter (if any) lives outside
    this helper so the replay path stays deterministic.

    Raises :class:`ValueError` on non-positive floor / factor or a
    ceiling below the floor — a misconfigured backoff is a bug we
    want to catch loudly rather than silently degrade to a hot loop.
    """
    if attempt < 0:
        raise ValueError(f"next_reconnect_delay_s: attempt must be >= 0, got {attempt!r}")
    if floor_s <= 0:
        raise ValueError(f"next_reconnect_delay_s: floor_s must be > 0, got {floor_s!r}")
    if ceiling_s < floor_s:
        raise ValueError("next_reconnect_delay_s: ceiling_s must be >= floor_s")
    if factor <= 1.0:
        raise ValueError(f"next_reconnect_delay_s: factor must be > 1.0, got {factor!r}")
    delay = floor_s * (factor**attempt)
    if delay > ceiling_s:
        return ceiling_s
    if delay < floor_s:  # never tripped for attempt >= 0; defensive
        return floor_s
    return delay


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "Trade",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "BookDelta",
    "BookGap",
    "ApplyResult",
    "OrderBookAggregator",
    "parse_binance_trade",
    "parse_binance_book_snapshot",
    "parse_binance_book_delta",
    "DEFAULT_RECONNECT_DELAY_FLOOR_S",
    "DEFAULT_RECONNECT_DELAY_CEILING_S",
    "DEFAULT_RECONNECT_BACKOFF_FACTOR",
    "next_reconnect_delay_s",
]
