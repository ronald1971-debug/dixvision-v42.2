"""A-18.1 — Tests for L2 order book (sortedcontainers adaptation)."""

from __future__ import annotations

import ast
import builtins
import dataclasses
from collections.abc import Mapping
from pathlib import Path

import pytest

from execution_engine.market_data.aggregator import (
    BookDelta,
    OrderBookLevel,
    OrderBookSnapshot,
)
from execution_engine.market_data.orderbook import (
    NEW_PIP_DEPENDENCIES,
    GapDetection,
    L2OrderBook,
    OrderBookApplyResult,
    PriceLevelMap,
    PurePyPriceLevelMap,
    ReplaySummary,
    pure_python_orderbook_factory,
    replay_l2,
    sortedcontainers_orderbook_factory,
)

_MODULE_PATH = Path("execution_engine/market_data/orderbook.py")


def _level(price: float, qty: float) -> OrderBookLevel:
    return OrderBookLevel(price=price, qty=qty)


def _seed_snapshot(
    *,
    ts_ns: int = 1_000,
    last_update_id: int = 100,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        ts_ns=ts_ns,
        symbol="BTCUSDT",
        last_update_id=last_update_id,
        bids=(_level(99.0, 1.0), _level(98.0, 2.0), _level(97.0, 3.0)),
        asks=(_level(100.0, 1.0), _level(101.0, 2.0), _level(102.0, 3.0)),
        venue="binance",
    )


# ----------------------------------------------------------------------
# PurePyPriceLevelMap
# ----------------------------------------------------------------------


class TestPurePyPriceLevelMap:
    def test_bid_descending(self) -> None:
        m = PurePyPriceLevelMap(descending=True)
        m.set(100.0, 1.0)
        m.set(101.0, 2.0)
        m.set(99.0, 3.0)
        best = m.peek_best()
        assert best == (101.0, 2.0)
        assert m.items_sorted() == (
            (101.0, 2.0),
            (100.0, 1.0),
            (99.0, 3.0),
        )

    def test_ask_ascending(self) -> None:
        m = PurePyPriceLevelMap(descending=False)
        m.set(100.0, 1.0)
        m.set(99.0, 3.0)
        m.set(101.0, 2.0)
        assert m.peek_best() == (99.0, 3.0)
        assert m.items_sorted() == (
            (99.0, 3.0),
            (100.0, 1.0),
            (101.0, 2.0),
        )

    def test_remove_and_get(self) -> None:
        m = PurePyPriceLevelMap(descending=True)
        m.set(100.0, 1.0)
        assert m.has(100.0)
        assert m.get(100.0) == 1.0
        m.remove(100.0)
        assert not m.has(100.0)
        assert m.get(100.0) == 0.0
        m.remove(999.0)  # tolerant

    def test_empty_peek_returns_none(self) -> None:
        m = PurePyPriceLevelMap(descending=True)
        assert m.peek_best() is None
        assert m.items_sorted() == ()

    def test_clear(self) -> None:
        m = PurePyPriceLevelMap(descending=False)
        m.set(1.0, 1.0)
        m.set(2.0, 2.0)
        assert len(m) == 2
        m.clear()
        assert len(m) == 0

    @pytest.mark.parametrize("bad_price", [0.0, -1.0])
    def test_invalid_price(self, bad_price: float) -> None:
        m = PurePyPriceLevelMap(descending=True)
        with pytest.raises(ValueError):
            m.set(bad_price, 1.0)

    @pytest.mark.parametrize("bad_qty", [-0.5, -1.0])
    def test_invalid_qty(self, bad_qty: float) -> None:
        m = PurePyPriceLevelMap(descending=True)
        with pytest.raises(ValueError):
            m.set(1.0, bad_qty)

    def test_bool_rejected(self) -> None:
        m = PurePyPriceLevelMap(descending=True)
        with pytest.raises(TypeError):
            m.set(True, 1.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            m.set(1.0, False)  # type: ignore[arg-type]

    def test_protocol_runtime_check(self) -> None:
        m = PurePyPriceLevelMap(descending=True)
        assert isinstance(m, PriceLevelMap)


# ----------------------------------------------------------------------
# L2OrderBook — construction
# ----------------------------------------------------------------------


class TestL2OrderBookConstruction:
    def test_factory_defaults(self) -> None:
        book = pure_python_orderbook_factory(
            symbol="BTCUSDT",
            venue="binance",
        )
        assert book.symbol == "BTCUSDT"
        assert book.venue == "binance"
        assert book.max_depth == 200
        assert book.last_update_id == -1
        assert len(book) == 0

    def test_factory_custom_depth(self) -> None:
        book = pure_python_orderbook_factory(
            symbol="ETHUSDT",
            venue="binance",
            max_depth=10,
        )
        assert book.max_depth == 10

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_factory_invalid_depth(self, bad: int) -> None:
        with pytest.raises(ValueError):
            pure_python_orderbook_factory(symbol="X", venue="Y", max_depth=bad)

    def test_factory_invalid_symbol(self) -> None:
        with pytest.raises(ValueError):
            pure_python_orderbook_factory(symbol="", venue="Y")

    def test_factory_invalid_venue(self) -> None:
        with pytest.raises(ValueError):
            pure_python_orderbook_factory(symbol="X", venue="")


# ----------------------------------------------------------------------
# L2OrderBook — apply_snapshot
# ----------------------------------------------------------------------


class TestApplySnapshot:
    def test_seed_basic(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        snap = book.apply_snapshot(_seed_snapshot())
        assert snap.last_update_id == 100
        assert snap.bids == (
            _level(99.0, 1.0),
            _level(98.0, 2.0),
            _level(97.0, 3.0),
        )
        assert snap.asks == (
            _level(100.0, 1.0),
            _level(101.0, 2.0),
            _level(102.0, 3.0),
        )
        assert book.best_bid() == _level(99.0, 1.0)
        assert book.best_ask() == _level(100.0, 1.0)
        assert book.mid() == pytest.approx(99.5)
        assert book.spread() == pytest.approx(1.0)

    def test_seed_truncates_by_max_depth(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance", max_depth=2)
        snap = book.apply_snapshot(_seed_snapshot())
        assert len(snap.bids) == 2
        assert len(snap.asks) == 2

    def test_seed_clears_zero_qty_levels(self) -> None:
        seed = OrderBookSnapshot(
            ts_ns=1_000,
            symbol="BTCUSDT",
            last_update_id=10,
            bids=(_level(99.0, 1.0), _level(98.0, 0.0)),
            asks=(_level(100.0, 0.0), _level(101.0, 2.0)),
            venue="binance",
        )
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        snap = book.apply_snapshot(seed)
        assert snap.bids == (_level(99.0, 1.0),)
        assert snap.asks == (_level(101.0, 2.0),)

    def test_seed_symbol_mismatch(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        seed = _seed_snapshot()
        seed = dataclasses.replace(seed, symbol="ETHUSDT")
        with pytest.raises(ValueError):
            book.apply_snapshot(seed)

    def test_seed_venue_mismatch(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        seed = _seed_snapshot()
        seed = dataclasses.replace(seed, venue="bybit")
        with pytest.raises(ValueError):
            book.apply_snapshot(seed)


# ----------------------------------------------------------------------
# L2OrderBook — apply_delta
# ----------------------------------------------------------------------


class TestApplyDelta:
    def _book(self) -> L2OrderBook:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        book.apply_snapshot(_seed_snapshot())
        return book

    def test_contiguous_delta_advances(self) -> None:
        book = self._book()
        delta = BookDelta(
            ts_ns=2_000,
            symbol="BTCUSDT",
            first_update_id=101,
            final_update_id=101,
            bid_updates=(_level(99.5, 5.0),),
            ask_updates=(_level(100.5, 5.0),),
            venue="binance",
        )
        result = book.apply_delta(delta)
        assert result.gap is None
        assert result.snapshot is not None
        assert result.snapshot.last_update_id == 101
        assert book.best_bid() == _level(99.5, 5.0)
        # best ask still 100.0 (added 100.5 is worse than existing 100.0)
        assert book.best_ask() == _level(100.0, 1.0)

    def test_delta_qty_zero_removes_level(self) -> None:
        book = self._book()
        delta = BookDelta(
            ts_ns=2_000,
            symbol="BTCUSDT",
            first_update_id=101,
            final_update_id=101,
            bid_updates=(_level(99.0, 0.0),),
            ask_updates=(),
            venue="binance",
        )
        result = book.apply_delta(delta)
        assert result.snapshot is not None
        assert result.snapshot.bids == (
            _level(98.0, 2.0),
            _level(97.0, 3.0),
        )

    def test_gap_detection_below_expected(self) -> None:
        book = self._book()
        # expected next = 101; delta starts at 110 -> gap
        delta = BookDelta(
            ts_ns=2_000,
            symbol="BTCUSDT",
            first_update_id=110,
            final_update_id=115,
            bid_updates=(),
            ask_updates=(),
            venue="binance",
        )
        result = book.apply_delta(delta)
        assert result.snapshot is None
        assert result.gap is not None
        assert result.gap.last_known_update_id == 100
        assert result.gap.delta_first_update_id == 110

    def test_gap_detection_stale_delta(self) -> None:
        book = self._book()
        delta = BookDelta(
            ts_ns=2_000,
            symbol="BTCUSDT",
            first_update_id=50,
            final_update_id=80,
            bid_updates=(),
            ask_updates=(),
            venue="binance",
        )
        result = book.apply_delta(delta)
        assert result.snapshot is None
        assert result.gap is not None

    def test_delta_spanning_expected(self) -> None:
        # first <= expected <= final  →  accept
        book = self._book()
        delta = BookDelta(
            ts_ns=2_000,
            symbol="BTCUSDT",
            first_update_id=95,
            final_update_id=105,
            bid_updates=(_level(96.0, 4.0),),
            ask_updates=(),
            venue="binance",
        )
        result = book.apply_delta(delta)
        assert result.gap is None
        assert result.snapshot is not None
        assert book.last_update_id == 105

    def test_apply_delta_without_seed_raises(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        delta = BookDelta(
            ts_ns=2_000,
            symbol="BTCUSDT",
            first_update_id=1,
            final_update_id=1,
            bid_updates=(),
            ask_updates=(),
            venue="binance",
        )
        with pytest.raises(RuntimeError):
            book.apply_delta(delta)


# ----------------------------------------------------------------------
# top_n + projection
# ----------------------------------------------------------------------


class TestProjections:
    def test_top_n_bids(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        book.apply_snapshot(_seed_snapshot())
        assert book.top_n_bids(2) == (
            _level(99.0, 1.0),
            _level(98.0, 2.0),
        )

    def test_top_n_asks(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        book.apply_snapshot(_seed_snapshot())
        assert book.top_n_asks(1) == (_level(100.0, 1.0),)

    def test_top_n_zero(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        book.apply_snapshot(_seed_snapshot())
        assert book.top_n_bids(0) == ()
        assert book.top_n_asks(0) == ()

    @pytest.mark.parametrize("bad", [-1, -10])
    def test_top_n_negative_rejected(self, bad: int) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        book.apply_snapshot(_seed_snapshot())
        with pytest.raises(ValueError):
            book.top_n_bids(bad)
        with pytest.raises(ValueError):
            book.top_n_asks(bad)

    def test_project_snapshot(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        book.apply_snapshot(_seed_snapshot())
        snap = book.project_snapshot(ts_ns=999_999)
        assert snap.ts_ns == 999_999
        assert snap.symbol == "BTCUSDT"

    def test_empty_book_projections(self) -> None:
        book = pure_python_orderbook_factory(symbol="BTCUSDT", venue="binance")
        assert book.best_bid() is None
        assert book.best_ask() is None
        assert book.mid() is None
        assert book.spread() is None


# ----------------------------------------------------------------------
# Determinism — INV-15
# ----------------------------------------------------------------------


class TestDeterminism:
    def _build_replay(self) -> ReplaySummary:
        seed = _seed_snapshot(ts_ns=1_000, last_update_id=100)
        deltas = (
            BookDelta(
                ts_ns=2_000,
                symbol="BTCUSDT",
                first_update_id=101,
                final_update_id=101,
                bid_updates=(_level(99.5, 5.0),),
                ask_updates=(),
                venue="binance",
            ),
            BookDelta(
                ts_ns=3_000,
                symbol="BTCUSDT",
                first_update_id=102,
                final_update_id=102,
                bid_updates=(),
                ask_updates=(_level(100.5, 5.0),),
                venue="binance",
            ),
            BookDelta(
                ts_ns=4_000,
                symbol="BTCUSDT",
                first_update_id=200,
                final_update_id=210,
                bid_updates=(),
                ask_updates=(),
                venue="binance",
            ),
        )
        return replay_l2(
            seed=seed,
            deltas=deltas,
            symbol="BTCUSDT",
            venue="binance",
        )

    def test_3_run_equality(self) -> None:
        runs = [self._build_replay() for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]
        assert len(runs[0].snapshots) == 3
        assert len(runs[0].gaps) == 1

    def test_replay_produces_byte_identical_snapshots(self) -> None:
        r1 = self._build_replay()
        r2 = self._build_replay()
        for s1, s2 in zip(r1.snapshots, r2.snapshots, strict=True):
            assert s1 == s2
        for g1, g2 in zip(r1.gaps, r2.gaps, strict=True):
            assert g1 == g2


# ----------------------------------------------------------------------
# GapDetection
# ----------------------------------------------------------------------


class TestGapDetection:
    def test_round_trip(self) -> None:
        g = GapDetection(
            ts_ns=1_000,
            symbol="BTCUSDT",
            venue="binance",
            last_known_update_id=100,
            delta_first_update_id=110,
            delta_final_update_id=115,
        )
        assert g.symbol == "BTCUSDT"

    def test_frozen(self) -> None:
        g = GapDetection(
            ts_ns=1_000,
            symbol="BTCUSDT",
            venue="binance",
            last_known_update_id=100,
            delta_first_update_id=110,
            delta_final_update_id=115,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            g.symbol = "X"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kw",
        [
            {"symbol": ""},
            {"last_known_update_id": -1},
            {"delta_first_update_id": -1},
            {"delta_final_update_id": -1, "delta_first_update_id": 0},
        ],
    )
    def test_invalid(self, kw: Mapping[str, object]) -> None:
        base = dict(
            ts_ns=1_000,
            symbol="BTCUSDT",
            venue="binance",
            last_known_update_id=100,
            delta_first_update_id=110,
            delta_final_update_id=115,
        )
        base.update(kw)
        with pytest.raises(ValueError):
            GapDetection(**base)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# OrderBookApplyResult
# ----------------------------------------------------------------------


class TestApplyResult:
    def test_exactly_one_field(self) -> None:
        # both populated → reject
        snap = _seed_snapshot()
        gap = GapDetection(
            ts_ns=1_000,
            symbol="BTCUSDT",
            venue="binance",
            last_known_update_id=100,
            delta_first_update_id=110,
            delta_final_update_id=115,
        )
        with pytest.raises(ValueError):
            OrderBookApplyResult(snapshot=snap, gap=gap)
        # neither populated → reject
        with pytest.raises(ValueError):
            OrderBookApplyResult(snapshot=None, gap=None)


# ----------------------------------------------------------------------
# sortedcontainers factory
# ----------------------------------------------------------------------


class TestSortedContainersFactory:
    def test_factory_raises_without_sortedcontainers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_import = builtins.__import__

        def _blocking_import(
            name: str,
            globals_: object = None,
            locals_: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "sortedcontainers":
                raise ImportError("simulated missing sortedcontainers")
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        with pytest.raises(RuntimeError, match="sortedcontainers"):
            sortedcontainers_orderbook_factory(symbol="BTCUSDT", venue="binance")


# ----------------------------------------------------------------------
# AST guards — INV-15 / B27 / B28 / B1
# ----------------------------------------------------------------------


def _parse_module() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


class TestASTGuards:
    def test_new_pip_dependencies_declared(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ("sortedcontainers",)

    def test_no_top_level_sortedcontainers_import(self) -> None:
        tree = _parse_module()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("sortedcontainers")
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or not (node.module.startswith("sortedcontainers"))

    def test_sortedcontainers_only_imported_in_factory(self) -> None:
        tree = _parse_module()
        found_factory = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "sortedcontainers_orderbook_factory"
            ):
                found_factory = True
                src = ast.unparse(node)
                assert "from sortedcontainers import" in src
        assert found_factory, "sortedcontainers_orderbook_factory must exist"

    def test_no_clock_imports(self) -> None:
        tree = _parse_module()
        banned = {"time", "datetime", "random", "asyncio", "os"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned

    def test_no_engine_imports(self) -> None:
        tree = _parse_module()
        banned_engines = {
            "governance_engine",
            "system_engine",
            "intelligence_engine",
            "evolution_engine",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned_engines

    def test_no_typed_event_construction(self) -> None:
        tree = _parse_module()
        banned_ctors = {
            "SignalEvent",
            "ExecutionEvent",
            "SystemEvent",
            "HazardEvent",
            "PatchProposal",
            "GovernanceDecision",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in banned_ctors
                elif isinstance(func, ast.Attribute):
                    assert func.attr not in banned_ctors

    def test_no_numpy_polars_torch(self) -> None:
        tree = _parse_module()
        banned = {"numpy", "polars", "pandas", "torch", "langsmith"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned

    def test_adapted_from_header(self) -> None:
        src = _MODULE_PATH.read_text(encoding="utf-8")
        assert "# ADAPTED FROM: grantjenks/python-sortedcontainers" in src
