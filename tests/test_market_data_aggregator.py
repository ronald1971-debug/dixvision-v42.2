"""Tests for ``execution_engine/market_data/aggregator.py`` (S-11)."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from execution_engine.market_data import aggregator as agg_mod
from execution_engine.market_data.aggregator import (
    ApplyResult,
    BookDelta,
    BookGap,
    OrderBookAggregator,
    OrderBookLevel,
    OrderBookSnapshot,
    Trade,
    next_reconnect_delay_s,
    parse_binance_book_delta,
    parse_binance_book_snapshot,
    parse_binance_trade,
)

# ---------------------------------------------------------------------------
# Module metadata + AST authority pins
# ---------------------------------------------------------------------------


_MOD_PATH = Path(agg_mod.__file__)


def test_new_pip_dependencies_is_empty_tuple() -> None:
    assert agg_mod.NEW_PIP_DEPENDENCIES == ()


def test_adapted_from_header_present() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    assert "ADAPTED FROM: bmoscon/cryptofeed" in src


def _all_imports(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_module_does_not_import_cryptofeed_or_polars_or_numpy() -> None:
    """RUNTIME_SAFE: no cryptofeed (license), no polars (B-POLARS), no numpy."""
    tree = ast.parse(_MOD_PATH.read_text(encoding="utf-8"))
    imported = set(_all_imports(tree))
    assert not any(name.startswith("cryptofeed") for name in imported)
    assert not any(name.startswith("polars") for name in imported)
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imported)


def test_module_does_not_import_clock_modules() -> None:
    """B-CLOCK: caller-supplied ``ts_ns`` only, never derived in-module."""
    tree = ast.parse(_MOD_PATH.read_text(encoding="utf-8"))
    imported = set(_all_imports(tree))
    forbidden = {"time", "datetime", "system.time_source", "system_engine.time"}
    assert not (imported & forbidden), f"forbidden clock imports: {imported & forbidden}"


def test_module_does_not_import_asyncio_or_websockets() -> None:
    """No I/O — pump lives elsewhere; this module is pure aggregation."""
    tree = ast.parse(_MOD_PATH.read_text(encoding="utf-8"))
    imported = set(_all_imports(tree))
    assert "asyncio" not in imported
    assert "websockets" not in imported


# ---------------------------------------------------------------------------
# Trade value object
# ---------------------------------------------------------------------------


def test_trade_frozen_and_slotted() -> None:
    t = Trade(
        ts_ns=1,
        symbol="BTCUSDT",
        side="BUY",
        price=1.0,
        qty=2.0,
        trade_id="x",
        venue="BINANCE",
    )
    with pytest.raises((AttributeError, TypeError)):
        t.price = 99.0  # type: ignore[misc]


@pytest.mark.parametrize("side", ["", "buy", "sell", "LIFT"])
def test_trade_rejects_invalid_side(side: str) -> None:
    with pytest.raises(ValueError, match="side"):
        Trade(
            ts_ns=1,
            symbol="BTCUSDT",
            side=side,
            price=1.0,
            qty=1.0,
            trade_id="x",
            venue="BINANCE",
        )


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_trade_rejects_non_positive_price(price: float) -> None:
    with pytest.raises(ValueError, match="price"):
        Trade(
            ts_ns=1,
            symbol="BTCUSDT",
            side="BUY",
            price=price,
            qty=1.0,
            trade_id="x",
            venue="BINANCE",
        )


@pytest.mark.parametrize("qty", [0.0, -0.001])
def test_trade_rejects_non_positive_qty(qty: float) -> None:
    with pytest.raises(ValueError, match="qty"):
        Trade(
            ts_ns=1,
            symbol="BTCUSDT",
            side="BUY",
            price=1.0,
            qty=qty,
            trade_id="x",
            venue="BINANCE",
        )


def test_trade_rejects_empty_symbol_and_trade_id() -> None:
    with pytest.raises(ValueError, match="symbol"):
        Trade(
            ts_ns=1,
            symbol="",
            side="BUY",
            price=1.0,
            qty=1.0,
            trade_id="x",
            venue="BINANCE",
        )
    with pytest.raises(ValueError, match="trade_id"):
        Trade(
            ts_ns=1,
            symbol="BTCUSDT",
            side="BUY",
            price=1.0,
            qty=1.0,
            trade_id="",
            venue="BINANCE",
        )


# ---------------------------------------------------------------------------
# OrderBookLevel / OrderBookSnapshot validators
# ---------------------------------------------------------------------------


def test_order_book_level_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError, match="price"):
        OrderBookLevel(price=0.0, qty=1.0)


def test_order_book_level_allows_zero_qty_for_removal() -> None:
    OrderBookLevel(price=1.0, qty=0.0)


def test_order_book_snapshot_rejects_unsorted_bids() -> None:
    with pytest.raises(ValueError, match="bids must be sorted"):
        OrderBookSnapshot(
            ts_ns=1,
            symbol="BTCUSDT",
            last_update_id=1,
            venue="BINANCE",
            bids=(
                OrderBookLevel(price=10.0, qty=1.0),
                OrderBookLevel(price=11.0, qty=1.0),  # ascending — illegal for bids
            ),
            asks=(),
        )


def test_order_book_snapshot_rejects_unsorted_asks() -> None:
    with pytest.raises(ValueError, match="asks must be sorted"):
        OrderBookSnapshot(
            ts_ns=1,
            symbol="BTCUSDT",
            last_update_id=1,
            venue="BINANCE",
            bids=(),
            asks=(
                OrderBookLevel(price=11.0, qty=1.0),
                OrderBookLevel(price=10.0, qty=1.0),  # descending — illegal for asks
            ),
        )


def test_order_book_snapshot_top_of_book_helpers() -> None:
    snap = OrderBookSnapshot(
        ts_ns=1,
        symbol="BTCUSDT",
        last_update_id=1,
        venue="BINANCE",
        bids=(
            OrderBookLevel(price=100.0, qty=1.0),
            OrderBookLevel(price=99.0, qty=2.0),
        ),
        asks=(
            OrderBookLevel(price=101.0, qty=1.0),
            OrderBookLevel(price=102.0, qty=2.0),
        ),
    )
    assert snap.best_bid() == OrderBookLevel(price=100.0, qty=1.0)
    assert snap.best_ask() == OrderBookLevel(price=101.0, qty=1.0)
    assert snap.mid() == 100.5


def test_order_book_snapshot_mid_none_on_empty_side() -> None:
    snap = OrderBookSnapshot(
        ts_ns=1,
        symbol="BTCUSDT",
        last_update_id=1,
        venue="BINANCE",
        bids=(),
        asks=(OrderBookLevel(price=101.0, qty=1.0),),
    )
    assert snap.best_bid() is None
    assert snap.mid() is None


# ---------------------------------------------------------------------------
# BookDelta validators
# ---------------------------------------------------------------------------


def test_book_delta_rejects_inverted_window() -> None:
    with pytest.raises(ValueError, match="final_update_id"):
        BookDelta(
            ts_ns=1,
            symbol="BTCUSDT",
            venue="BINANCE",
            first_update_id=10,
            final_update_id=5,
            bid_updates=(),
            ask_updates=(),
        )


# ---------------------------------------------------------------------------
# Pure parsers — Binance frame shapes
# ---------------------------------------------------------------------------


def _trade_frame(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "e": "trade",
        "s": "BTCUSDT",
        "t": 12345,
        "p": "50000.0",
        "q": "0.001",
        "m": False,
    }
    base.update(overrides)
    return base


def test_parse_binance_trade_buy_side() -> None:
    t = parse_binance_trade(_trade_frame(m=False), ts_ns=42)
    assert t == Trade(
        ts_ns=42,
        symbol="BTCUSDT",
        side="BUY",
        price=50000.0,
        qty=0.001,
        trade_id="12345",
        venue="BINANCE",
    )


def test_parse_binance_trade_sell_side_when_buyer_is_maker() -> None:
    t = parse_binance_trade(_trade_frame(m=True), ts_ns=42)
    assert t is not None
    assert t.side == "SELL"


def test_parse_binance_trade_unwraps_combined_stream_envelope() -> None:
    payload = {"stream": "btcusdt@trade", "data": _trade_frame()}
    t = parse_binance_trade(payload, ts_ns=42)
    assert t is not None
    assert t.symbol == "BTCUSDT"


@pytest.mark.parametrize(
    "broken",
    [
        None,
        [],
        "not a mapping",
        {},  # missing "e"
        {"e": "subscribe"},  # not a trade frame
        _trade_frame(p="not a number"),
        _trade_frame(q="not a number"),
        _trade_frame(p="-1.0"),
        _trade_frame(q="0.0"),
        _trade_frame(s=""),
        _trade_frame(s=None),
        _trade_frame(t=None),
        _trade_frame(m="true"),  # m must be bool, not str
    ],
)
def test_parse_binance_trade_returns_none_on_malformed(broken: object) -> None:
    assert parse_binance_trade(broken, ts_ns=42) is None


def test_parse_binance_trade_uses_caller_supplied_ts_ns() -> None:
    """INV-15: ts_ns must come from the caller, not the payload."""
    t = parse_binance_trade(_trade_frame(), ts_ns=99_999)
    assert t is not None
    assert t.ts_ns == 99_999


def test_parse_binance_book_snapshot_canonical_shape() -> None:
    payload = {
        "lastUpdateId": 100,
        "bids": [["100.0", "1.0"], ["99.0", "2.0"]],
        "asks": [["101.0", "1.0"], ["102.0", "2.0"]],
    }
    snap = parse_binance_book_snapshot(payload, ts_ns=1, symbol="BTCUSDT")
    assert snap is not None
    assert snap.last_update_id == 100
    assert snap.bids == (
        OrderBookLevel(price=100.0, qty=1.0),
        OrderBookLevel(price=99.0, qty=2.0),
    )
    assert snap.asks == (
        OrderBookLevel(price=101.0, qty=1.0),
        OrderBookLevel(price=102.0, qty=2.0),
    )


def test_parse_binance_book_snapshot_canonicalises_arbitrary_input_order() -> None:
    """Two inputs that differ only by level-insertion order must produce identical snapshots."""
    a = parse_binance_book_snapshot(
        {
            "lastUpdateId": 1,
            "bids": [["100.0", "1.0"], ["99.0", "2.0"]],
            "asks": [["101.0", "1.0"], ["102.0", "2.0"]],
        },
        ts_ns=1,
        symbol="BTCUSDT",
    )
    b = parse_binance_book_snapshot(
        {
            "lastUpdateId": 1,
            "bids": [["99.0", "2.0"], ["100.0", "1.0"]],
            "asks": [["102.0", "2.0"], ["101.0", "1.0"]],
        },
        ts_ns=1,
        symbol="BTCUSDT",
    )
    assert a == b


@pytest.mark.parametrize(
    "broken",
    [
        None,
        "not a mapping",
        {},
        {"lastUpdateId": -1, "bids": [], "asks": []},
        {"lastUpdateId": 1, "bids": [["bad"]], "asks": []},
        {"lastUpdateId": 1, "bids": [["100.0", "-1.0"]], "asks": []},
        {"lastUpdateId": 1, "bids": [["0.0", "1.0"]], "asks": []},
    ],
)
def test_parse_binance_book_snapshot_returns_none_on_malformed(
    broken: object,
) -> None:
    assert parse_binance_book_snapshot(broken, ts_ns=1, symbol="BTCUSDT") is None


def test_parse_binance_book_snapshot_rejects_empty_symbol() -> None:
    payload = {"lastUpdateId": 1, "bids": [], "asks": []}
    assert parse_binance_book_snapshot(payload, ts_ns=1, symbol="") is None


def _delta_frame(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "e": "depthUpdate",
        "s": "BTCUSDT",
        "U": 11,
        "u": 13,
        "b": [["100.0", "0.5"]],
        "a": [["101.0", "0.5"]],
    }
    base.update(overrides)
    return base


def test_parse_binance_book_delta_canonical_shape() -> None:
    d = parse_binance_book_delta(_delta_frame(), ts_ns=42)
    assert d == BookDelta(
        ts_ns=42,
        symbol="BTCUSDT",
        venue="BINANCE",
        first_update_id=11,
        final_update_id=13,
        bid_updates=(OrderBookLevel(price=100.0, qty=0.5),),
        ask_updates=(OrderBookLevel(price=101.0, qty=0.5),),
    )


@pytest.mark.parametrize(
    "broken",
    [
        None,
        {"e": "trade"},  # wrong event kind
        _delta_frame(s=""),
        _delta_frame(U=None),
        _delta_frame(u=None),
        _delta_frame(U=10, u=5),  # final < first
        _delta_frame(b=[["bad"]]),
    ],
)
def test_parse_binance_book_delta_returns_none_on_malformed(
    broken: object,
) -> None:
    assert parse_binance_book_delta(broken, ts_ns=1) is None


def test_parsers_are_pure_under_json_round_trip() -> None:
    """Parsers must accept JSON-decoded payloads (tests ride dict/list)."""
    payload_str = json.dumps(_trade_frame())
    payload = json.loads(payload_str)
    assert parse_binance_trade(payload, ts_ns=1) is not None


# ---------------------------------------------------------------------------
# OrderBookAggregator — snapshot + delta + sequence-gap detection
# ---------------------------------------------------------------------------


def _snapshot(
    last_update_id: int = 100,
    bids: tuple[tuple[float, float], ...] = (),
    asks: tuple[tuple[float, float], ...] = (),
    *,
    ts_ns: int = 1,
    symbol: str = "BTCUSDT",
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        ts_ns=ts_ns,
        symbol=symbol,
        last_update_id=last_update_id,
        venue="BINANCE",
        bids=tuple(OrderBookLevel(price=p, qty=q) for p, q in bids),
        asks=tuple(OrderBookLevel(price=p, qty=q) for p, q in asks),
    )


def _delta(
    first: int,
    final: int,
    bids: tuple[tuple[float, float], ...] = (),
    asks: tuple[tuple[float, float], ...] = (),
    *,
    ts_ns: int = 2,
    symbol: str = "BTCUSDT",
) -> BookDelta:
    return BookDelta(
        ts_ns=ts_ns,
        symbol=symbol,
        venue="BINANCE",
        first_update_id=first,
        final_update_id=final,
        bid_updates=tuple(OrderBookLevel(price=p, qty=q) for p, q in bids),
        ask_updates=tuple(OrderBookLevel(price=p, qty=q) for p, q in asks),
    )


def test_aggregator_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        OrderBookAggregator(symbol="")


def test_aggregator_apply_delta_before_snapshot_raises() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    with pytest.raises(RuntimeError, match="apply_snapshot first"):
        a.apply_delta(_delta(first=101, final=102))


def test_aggregator_apply_snapshot_rejects_symbol_mismatch() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    with pytest.raises(ValueError, match="received symbol"):
        a.apply_snapshot(_snapshot(symbol="ETHUSDT"))


def test_aggregator_apply_delta_rejects_symbol_mismatch() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(_snapshot())
    with pytest.raises(ValueError, match="received symbol"):
        a.apply_delta(_delta(first=101, final=102, symbol="ETHUSDT"))


def test_aggregator_clean_delta_updates_top_of_book() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(
        _snapshot(
            last_update_id=100,
            bids=((99.0, 1.0),),
            asks=((101.0, 1.0),),
        )
    )
    res = a.apply_delta(
        _delta(
            first=101,
            final=101,
            bids=((100.0, 0.5),),
            asks=((100.5, 0.5),),
        )
    )
    assert res.is_ok()
    assert res.snapshot is not None
    assert res.snapshot.last_update_id == 101
    assert res.snapshot.best_bid() == OrderBookLevel(price=100.0, qty=0.5)
    assert res.snapshot.best_ask() == OrderBookLevel(price=100.5, qty=0.5)


def test_aggregator_window_delta_advances_sequence() -> None:
    """Binance L2 contract: a delta with U <= last+1 <= u is clean."""
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(_snapshot(last_update_id=100))
    res = a.apply_delta(_delta(first=98, final=105))  # window straddles last+1=101
    assert res.is_ok()
    assert res.snapshot is not None
    assert res.snapshot.last_update_id == 105


def test_aggregator_zero_qty_removes_level() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(
        _snapshot(
            last_update_id=100,
            bids=((100.0, 1.0), (99.0, 2.0)),
        )
    )
    res = a.apply_delta(
        _delta(
            first=101,
            final=101,
            bids=((100.0, 0.0),),  # remove top bid
        )
    )
    assert res.is_ok()
    assert res.snapshot is not None
    assert res.snapshot.best_bid() == OrderBookLevel(price=99.0, qty=2.0)


def test_aggregator_stale_delta_silently_ignored() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(_snapshot(last_update_id=100))
    res = a.apply_delta(_delta(first=50, final=60))  # final <= last
    assert res.stale is True
    assert res.snapshot is None
    assert res.gap is None
    # State unchanged.
    cur = a.current()
    assert cur is not None
    assert cur.last_update_id == 100


def test_aggregator_gap_freezes_state() -> None:
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(_snapshot(last_update_id=100))
    res = a.apply_delta(_delta(first=200, final=210))  # well past last+1
    assert res.gap == BookGap(
        ts_ns=2,
        symbol="BTCUSDT",
        venue="BINANCE",
        last_known_update_id=100,
        delta_first_update_id=200,
        delta_final_update_id=210,
    )
    assert res.snapshot is None
    # State frozen at last good snapshot.
    cur = a.current()
    assert cur is not None
    assert cur.last_update_id == 100


def test_aggregator_resync_after_gap() -> None:
    """Caller pattern: gap → emit hazard → re-fetch snapshot → resume."""
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(_snapshot(last_update_id=100))
    gap_res = a.apply_delta(_delta(first=200, final=210))
    assert gap_res.gap is not None
    # Resync.
    a.apply_snapshot(_snapshot(last_update_id=300))
    res = a.apply_delta(_delta(first=300, final=305))
    assert res.is_ok()
    assert res.snapshot is not None
    assert res.snapshot.last_update_id == 305


def test_aggregator_replay_determinism() -> None:
    """INV-15: same snapshot + delta sequence → byte-identical final state."""
    deltas = [
        _delta(first=101, final=101, bids=((99.5, 0.7),), asks=((100.5, 0.7),)),
        _delta(first=102, final=102, bids=((99.0, 0.0),), asks=((101.0, 0.0),)),
        _delta(first=103, final=103, asks=((100.6, 0.3),)),
    ]
    finals: list[OrderBookSnapshot] = []
    for _ in range(3):
        a = OrderBookAggregator(symbol="BTCUSDT")
        a.apply_snapshot(
            _snapshot(
                last_update_id=100,
                bids=((100.0, 1.0), (99.0, 2.0)),
                asks=((101.0, 1.0), (102.0, 2.0)),
            )
        )
        for d in deltas:
            a.apply_delta(d)
        cur = a.current()
        assert cur is not None
        finals.append(cur)
    assert finals[0] == finals[1] == finals[2]


def test_aggregator_canonical_sort_breaks_input_order_dependence() -> None:
    """Delta updates passed in different orders produce identical snapshots."""
    a = OrderBookAggregator(symbol="BTCUSDT")
    a.apply_snapshot(_snapshot(last_update_id=100))
    res_a = a.apply_delta(
        _delta(
            first=101,
            final=101,
            bids=((100.0, 1.0), (99.0, 2.0)),
            asks=((101.0, 1.0), (102.0, 2.0)),
        )
    )
    b = OrderBookAggregator(symbol="BTCUSDT")
    b.apply_snapshot(_snapshot(last_update_id=100))
    res_b = b.apply_delta(
        _delta(
            first=101,
            final=101,
            bids=((99.0, 2.0), (100.0, 1.0)),
            asks=((102.0, 2.0), (101.0, 1.0)),
        )
    )
    assert res_a.snapshot == res_b.snapshot


# ---------------------------------------------------------------------------
# ApplyResult helpers
# ---------------------------------------------------------------------------


def test_apply_result_is_ok_only_on_clean_delta() -> None:
    snap = _snapshot()
    assert ApplyResult(snapshot=snap, gap=None, stale=False).is_ok()
    assert not ApplyResult(snapshot=None, gap=None, stale=True).is_ok()
    gap = BookGap(
        ts_ns=1,
        symbol="BTCUSDT",
        venue="BINANCE",
        last_known_update_id=1,
        delta_first_update_id=10,
        delta_final_update_id=11,
    )
    assert not ApplyResult(snapshot=None, gap=gap, stale=False).is_ok()


# ---------------------------------------------------------------------------
# next_reconnect_delay_s — pure exponential backoff
# ---------------------------------------------------------------------------


def test_next_reconnect_delay_s_zero_attempt_returns_floor() -> None:
    assert next_reconnect_delay_s(attempt=0, floor_s=5.0, ceiling_s=60.0) == 5.0


def test_next_reconnect_delay_s_doubles_each_attempt() -> None:
    assert next_reconnect_delay_s(attempt=1, floor_s=5.0, ceiling_s=60.0) == 10.0
    assert next_reconnect_delay_s(attempt=2, floor_s=5.0, ceiling_s=60.0) == 20.0
    assert next_reconnect_delay_s(attempt=3, floor_s=5.0, ceiling_s=60.0) == 40.0


def test_next_reconnect_delay_s_clamps_to_ceiling() -> None:
    assert next_reconnect_delay_s(attempt=10, floor_s=5.0, ceiling_s=60.0) == 60.0


def test_next_reconnect_delay_s_rejects_negative_attempt() -> None:
    with pytest.raises(ValueError, match="attempt"):
        next_reconnect_delay_s(attempt=-1)


def test_next_reconnect_delay_s_rejects_non_positive_floor() -> None:
    with pytest.raises(ValueError, match="floor_s"):
        next_reconnect_delay_s(attempt=0, floor_s=0.0)


def test_next_reconnect_delay_s_rejects_ceiling_below_floor() -> None:
    with pytest.raises(ValueError, match="ceiling_s"):
        next_reconnect_delay_s(attempt=0, floor_s=10.0, ceiling_s=5.0)


def test_next_reconnect_delay_s_rejects_factor_le_one() -> None:
    with pytest.raises(ValueError, match="factor"):
        next_reconnect_delay_s(attempt=1, factor=1.0)


def test_next_reconnect_delay_s_is_pure_replay_deterministic() -> None:
    runs = [[next_reconnect_delay_s(attempt=i) for i in range(8)] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
