# ADAPTED FROM: https://github.com/numba/numba  (BSD-2-Clause)
#
# Benchmarking-only JIT harness for the slippage inner loop (I-13).
#
# NEW_PIP_DEPENDENCIES = ("numba",)
#
# Authority constraints:
#
#   * BENCH_ONLY — this file lives under ``tests/bench/`` and MUST NOT be
#     imported by any production module. ``numba`` is the lazy seam: it is
#     only imported via ``pytest.importorskip("numba")`` inside the
#     numba-only tests. The reference pure-Python inner loop produces
#     byte-identical floats to :class:`BookWalkSlippage.apply` so the
#     JIT-compiled variant can be verified for correctness without
#     touching production hot-path code.
#   * INV-15 — the inner loop is a pure function of its inputs (no clock
#     reads, no PRNG, no IO). Three independent calls with identical
#     arguments produce byte-identical output, both in pure-Python and
#     numba-jit modes.
#   * B1 — no runtime engine imports (no ``execution_engine`` /
#     ``intelligence_engine`` / ``governance_engine`` here). The
#     production ``simulation_engine.slippage_model`` import is used
#     only as the equivalence oracle for correctness assertions.
"""I-13 numba bench: slippage inner loop (pure-Python ref + optional JIT)."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from simulation_engine.slippage_model import (
    BookLevel,
    BookSnapshot,
    BookWalkSlippage,
    Side,
)

# ---------------------------------------------------------------------------
# Reference inner loop (pure Python — verbatim from BookWalkSlippage.apply)
# ---------------------------------------------------------------------------


def book_walk_vwap_pure(
    prices: tuple[float, ...],
    qtys: tuple[float, ...],
    qty: float,
    mark_price: float,
    fallback: bool,
) -> float:
    """Pure-Python BookWalk VWAP — the inner loop alone.

    Operates on parallel ``prices`` / ``qtys`` tuples (the JIT-friendly
    shape) rather than :class:`BookLevel` objects, so the same function
    can be JIT-compiled by numba with the typed-list/array signature
    without touching frozen dataclasses.
    """

    if qty <= 0.0 or mark_price <= 0.0 or len(prices) == 0:
        return mark_price
    remaining = float(qty)
    cost = 0.0
    filled = 0.0
    for i in range(len(prices)):
        if remaining <= 0.0:
            break
        take = remaining if remaining < qtys[i] else qtys[i]
        if take <= 0.0:
            continue
        cost += take * prices[i]
        filled += take
        remaining -= take
    if remaining > 0.0:
        if not fallback:
            return float("nan")  # sentinel — caller checks fallback flag
        cost += remaining * mark_price
        filled += remaining
    if filled <= 0.0:
        return mark_price
    return cost / filled


# ---------------------------------------------------------------------------
# Helpers — synthetic books with deterministic structure (no PRNG)
# ---------------------------------------------------------------------------


def _ask_book(depth: int, base: float = 100.0, step: float = 0.01) -> BookSnapshot:
    """Build an ascending ask book — N levels, +0.01 step, qty=1.0 each."""

    asks = tuple(BookLevel(price=base + i * step, qty=1.0) for i in range(depth))
    bids = (BookLevel(price=base - 0.01, qty=1.0),)
    return BookSnapshot(bids=bids, asks=asks)


def _split(levels: Iterable[BookLevel]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Split book levels into parallel ``(prices, qtys)`` tuples."""

    prices_list: list[float] = []
    qtys_list: list[float] = []
    for lvl in levels:
        prices_list.append(lvl.price)
        qtys_list.append(lvl.qty)
    return tuple(prices_list), tuple(qtys_list)


# ---------------------------------------------------------------------------
# Pure-Python reference vs production oracle
# ---------------------------------------------------------------------------


def test_pure_ref_matches_book_walk_slippage_oracle() -> None:
    book = _ask_book(depth=64)
    prices, qtys = _split(book.asks)
    model = BookWalkSlippage()
    expected = model.apply(Side.BUY, qty=10.0, mark_price=100.0, book=book)
    got = book_walk_vwap_pure(prices, qtys, 10.0, 100.0, True)
    assert got == expected


def test_pure_ref_byte_identical_three_runs() -> None:
    """INV-15 — three independent calls produce byte-identical output."""

    prices, qtys = _split(_ask_book(depth=128).asks)
    a = book_walk_vwap_pure(prices, qtys, 7.5, 100.0, True)
    b = book_walk_vwap_pure(prices, qtys, 7.5, 100.0, True)
    c = book_walk_vwap_pure(prices, qtys, 7.5, 100.0, True)
    assert a == b == c


def test_pure_ref_fallback_to_mark_on_empty_depth() -> None:
    """qty exceeds available depth ⇒ remainder priced at mark_price."""

    prices = (100.0, 101.0)
    qtys = (1.0, 1.0)
    got = book_walk_vwap_pure(prices, qtys, 5.0, 200.0, True)
    # 1*100 + 1*101 + 3*200 = 801 ; / 5 = 160.2
    assert got == pytest.approx(160.2, rel=0.0, abs=0.0)


def test_pure_ref_no_fallback_returns_nan_sentinel() -> None:
    prices = (100.0,)
    qtys = (0.5,)
    got = book_walk_vwap_pure(prices, qtys, 2.0, 200.0, False)
    assert got != got  # NaN comparison


def test_pure_ref_handles_zero_qty_levels() -> None:
    prices = (100.0, 100.5, 101.0)
    qtys = (0.0, 2.0, 1.0)
    got = book_walk_vwap_pure(prices, qtys, 2.0, 100.0, True)
    # First level skipped (qty=0), 2 units consumed at 100.5 ⇒ vwap = 100.5
    assert got == 100.5


def test_pure_ref_empty_book_returns_mark() -> None:
    got = book_walk_vwap_pure((), (), 1.0, 100.0, True)
    assert got == 100.0


def test_pure_ref_zero_qty_request_returns_mark() -> None:
    prices = (100.0, 101.0)
    qtys = (1.0, 1.0)
    got = book_walk_vwap_pure(prices, qtys, 0.0, 100.0, True)
    assert got == 100.0


# ---------------------------------------------------------------------------
# Numba JIT correctness — only runs if numba is installed
# ---------------------------------------------------------------------------


def test_numba_jit_matches_pure_reference() -> None:
    """JIT-compiled inner loop produces identical output to pure Python."""

    numba = pytest.importorskip("numba")
    jit_walk = numba.njit(cache=False)(book_walk_vwap_pure)
    prices, qtys = _split(_ask_book(depth=32).asks)
    expected = book_walk_vwap_pure(prices, qtys, 8.0, 100.0, True)
    got = jit_walk(prices, qtys, 8.0, 100.0, True)
    assert got == expected


def test_numba_jit_byte_identical_three_runs() -> None:
    """INV-15 — JIT three-run determinism."""

    numba = pytest.importorskip("numba")
    jit_walk = numba.njit(cache=False)(book_walk_vwap_pure)
    prices, qtys = _split(_ask_book(depth=16).asks)
    a = jit_walk(prices, qtys, 5.0, 100.0, True)
    b = jit_walk(prices, qtys, 5.0, 100.0, True)
    c = jit_walk(prices, qtys, 5.0, 100.0, True)
    assert a == b == c


# ---------------------------------------------------------------------------
# Bench harness — calls inner loop N times to gather a timing sample.
# Asserts only on correctness (not timing) so CI never flakes.
# ---------------------------------------------------------------------------


def test_pure_ref_bench_executes_many_iterations() -> None:
    prices, qtys = _split(_ask_book(depth=256).asks)
    last = 0.0
    for _ in range(2000):
        last = book_walk_vwap_pure(prices, qtys, 12.5, 100.0, True)
    expected = book_walk_vwap_pure(prices, qtys, 12.5, 100.0, True)
    assert last == expected


# ---------------------------------------------------------------------------
# AST guardrails — no top-level numba; bench-only contract
# ---------------------------------------------------------------------------


def test_no_top_level_numba_import_in_this_file() -> None:
    """``numba`` must never be imported at module scope (lazy seam)."""

    import ast as _ast
    import inspect as _inspect

    src = _inspect.getsource(__import__("tests.bench.test_slippage_jit_bench", fromlist=["*"]))
    tree = _ast.parse(src)
    for node in tree.body:
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("numba"), (
                    "I-13 bench-only contract: numba must not be a top-level import; "
                    "use pytest.importorskip('numba') inside the test body"
                )
        if isinstance(node, _ast.ImportFrom):
            assert node.module is None or not node.module.startswith("numba"), (
                "I-13 bench-only contract: numba must not be a top-level import"
            )
