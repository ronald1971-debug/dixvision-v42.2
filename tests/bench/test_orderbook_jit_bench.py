# ADAPTED FROM: https://github.com/numba/numba  (BSD-2-Clause)
#
# Benchmarking-only JIT harness for the L2 order-book inner loops (I-13).
#
# NEW_PIP_DEPENDENCIES = ("numba",)
#
# Authority constraints:
#
#   * BENCH_ONLY — this file lives under ``tests/bench/`` and MUST NOT be
#     imported by any production module. ``numba`` is the lazy seam: it
#     is only imported via ``pytest.importorskip("numba")`` inside the
#     numba-only tests. The reference pure-Python kernels operate on
#     parallel float tuples (the JIT-friendly shape) and are byte-stable
#     under three-run replay (INV-15).
#   * INV-15 — every kernel is a pure function of its inputs; three
#     independent calls with identical arguments produce byte-identical
#     output, both in pure-Python and numba-jit modes.
#   * B1 — no runtime engine imports (no ``execution_engine`` /
#     ``intelligence_engine`` / ``governance_engine`` here).
"""I-13 numba bench: L2 order-book kernels (pure-Python ref + optional JIT)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Reference kernels (pure Python) — parallel-tuple shape, JIT-friendly
# ---------------------------------------------------------------------------


def book_depth_within(prices: tuple[float, ...], qtys: tuple[float, ...], max_levels: int) -> float:
    """Total resting quantity within the top ``max_levels`` levels."""

    if max_levels <= 0:
        return 0.0
    n = len(prices) if len(prices) < max_levels else max_levels
    total = 0.0
    for i in range(n):
        if prices[i] > 0.0 and qtys[i] > 0.0:
            total += qtys[i]
    return total


def book_imbalance(
    bid_qtys: tuple[float, ...], ask_qtys: tuple[float, ...], max_levels: int
) -> float:
    """Top-N imbalance: ``(bidQ - askQ) / (bidQ + askQ)`` (or 0.0 when flat)."""

    if max_levels <= 0:
        return 0.0
    bidn = len(bid_qtys) if len(bid_qtys) < max_levels else max_levels
    askn = len(ask_qtys) if len(ask_qtys) < max_levels else max_levels
    bid_total = 0.0
    ask_total = 0.0
    for i in range(bidn):
        if bid_qtys[i] > 0.0:
            bid_total += bid_qtys[i]
    for i in range(askn):
        if ask_qtys[i] > 0.0:
            ask_total += ask_qtys[i]
    denom = bid_total + ask_total
    if denom <= 0.0:
        return 0.0
    return (bid_total - ask_total) / denom


def book_microprice(
    best_bid_price: float,
    best_bid_qty: float,
    best_ask_price: float,
    best_ask_qty: float,
) -> float:
    """Stoll's microprice — qty-weighted average of best bid/ask."""

    if best_bid_qty < 0.0 or best_ask_qty < 0.0:
        return 0.0
    denom = best_bid_qty + best_ask_qty
    if denom <= 0.0:
        return 0.0
    return (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / denom


# ---------------------------------------------------------------------------
# Synthetic books (deterministic, no PRNG)
# ---------------------------------------------------------------------------


def _ladder(n: int, base: float, step: float, sign: float) -> tuple[float, ...]:
    """Build N prices walking ``base + sign*i*step``."""

    out: list[float] = []
    for i in range(n):
        out.append(base + sign * i * step)
    return tuple(out)


def _qtys(n: int, q: float = 1.0) -> tuple[float, ...]:
    return tuple(q for _ in range(n))


# ---------------------------------------------------------------------------
# Pure-Python kernel correctness
# ---------------------------------------------------------------------------


def test_book_depth_within_top10() -> None:
    prices = _ladder(20, 100.0, 0.01, +1.0)
    qtys = _qtys(20, 2.0)
    assert book_depth_within(prices, qtys, 10) == 20.0


def test_book_depth_within_zero_levels_is_zero() -> None:
    assert book_depth_within((100.0,), (1.0,), 0) == 0.0


def test_book_depth_skips_zero_qty_levels() -> None:
    prices = (100.0, 100.5, 101.0)
    qtys = (0.0, 2.0, 1.0)
    assert book_depth_within(prices, qtys, 3) == 3.0


def test_book_imbalance_balanced_is_zero() -> None:
    assert book_imbalance(_qtys(5, 1.0), _qtys(5, 1.0), 5) == 0.0


def test_book_imbalance_bid_heavy_positive() -> None:
    got = book_imbalance(_qtys(5, 2.0), _qtys(5, 1.0), 5)
    assert got == pytest.approx((10.0 - 5.0) / 15.0, rel=0.0, abs=0.0)


def test_book_imbalance_empty_returns_zero() -> None:
    assert book_imbalance((), (), 5) == 0.0


def test_book_microprice_balanced_is_midpoint() -> None:
    got = book_microprice(99.99, 1.0, 100.01, 1.0)
    assert got == pytest.approx(100.0, rel=0.0, abs=0.0)


def test_book_microprice_ask_heavy_skews_toward_bid() -> None:
    got = book_microprice(99.0, 1.0, 101.0, 99.0)
    assert got < 100.0


def test_book_microprice_zero_qtys_returns_zero() -> None:
    assert book_microprice(99.0, 0.0, 101.0, 0.0) == 0.0


def test_kernels_byte_identical_three_runs() -> None:
    """INV-15 — every kernel is byte-stable across three independent calls."""

    bids = _qtys(8, 1.5)
    asks = _qtys(8, 2.5)
    a = book_depth_within(_ladder(8, 100.0, 0.01, -1.0), bids, 5)
    b = book_depth_within(_ladder(8, 100.0, 0.01, -1.0), bids, 5)
    c = book_depth_within(_ladder(8, 100.0, 0.01, -1.0), bids, 5)
    assert a == b == c
    a = book_imbalance(bids, asks, 5)
    b = book_imbalance(bids, asks, 5)
    c = book_imbalance(bids, asks, 5)
    assert a == b == c
    a = book_microprice(99.99, 1.5, 100.01, 2.5)
    b = book_microprice(99.99, 1.5, 100.01, 2.5)
    c = book_microprice(99.99, 1.5, 100.01, 2.5)
    assert a == b == c


# ---------------------------------------------------------------------------
# Numba JIT correctness — only runs when numba is installed
# ---------------------------------------------------------------------------


def test_numba_jit_depth_matches_pure_reference() -> None:
    numba = pytest.importorskip("numba")
    jit_depth = numba.njit(cache=False)(book_depth_within)
    prices = _ladder(64, 100.0, 0.01, +1.0)
    qtys = _qtys(64, 1.25)
    expected = book_depth_within(prices, qtys, 32)
    got = jit_depth(prices, qtys, 32)
    assert got == expected


def test_numba_jit_imbalance_matches_pure_reference() -> None:
    numba = pytest.importorskip("numba")
    jit_imb = numba.njit(cache=False)(book_imbalance)
    bids = _qtys(16, 2.0)
    asks = _qtys(16, 3.0)
    expected = book_imbalance(bids, asks, 10)
    got = jit_imb(bids, asks, 10)
    assert got == expected


def test_numba_jit_microprice_matches_pure_reference() -> None:
    numba = pytest.importorskip("numba")
    jit_mp = numba.njit(cache=False)(book_microprice)
    expected = book_microprice(99.5, 1.5, 100.5, 2.5)
    got = jit_mp(99.5, 1.5, 100.5, 2.5)
    assert got == expected


def test_numba_jit_byte_identical_three_runs() -> None:
    numba = pytest.importorskip("numba")
    jit_depth = numba.njit(cache=False)(book_depth_within)
    prices = _ladder(32, 100.0, 0.01, +1.0)
    qtys = _qtys(32, 1.0)
    a = jit_depth(prices, qtys, 20)
    b = jit_depth(prices, qtys, 20)
    c = jit_depth(prices, qtys, 20)
    assert a == b == c


# ---------------------------------------------------------------------------
# Bench harness — many iterations, correctness-only assertion
# ---------------------------------------------------------------------------


def test_kernels_bench_executes_many_iterations() -> None:
    bids = _qtys(64, 1.5)
    asks = _qtys(64, 2.5)
    last_imb = 0.0
    last_depth = 0.0
    last_mp = 0.0
    for _ in range(2000):
        last_imb = book_imbalance(bids, asks, 32)
        last_depth = book_depth_within(_ladder(64, 100.0, 0.01, -1.0), bids, 32)
        last_mp = book_microprice(99.99, 1.5, 100.01, 2.5)
    assert last_imb == book_imbalance(bids, asks, 32)
    assert last_depth == book_depth_within(_ladder(64, 100.0, 0.01, -1.0), bids, 32)
    assert last_mp == book_microprice(99.99, 1.5, 100.01, 2.5)


# ---------------------------------------------------------------------------
# AST guardrail — no top-level numba in this file (lazy seam contract)
# ---------------------------------------------------------------------------


def test_no_top_level_numba_import_in_this_file() -> None:
    import ast as _ast
    import inspect as _inspect

    src = _inspect.getsource(__import__("tests.bench.test_orderbook_jit_bench", fromlist=["*"]))
    tree = _ast.parse(src)
    for node in tree.body:
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("numba"), (
                    "I-13 bench-only contract: numba must not be a top-level import"
                )
        if isinstance(node, _ast.ImportFrom):
            assert node.module is None or not node.module.startswith("numba"), (
                "I-13 bench-only contract: numba must not be a top-level import"
            )
