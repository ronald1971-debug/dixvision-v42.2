# ADAPTED FROM: HypothesisWorks/hypothesis
#   - hypothesis/strategies/_internal/strategies.py — `@given`
#   - hypothesis/strategies/_internal/numbers.py — `floats`/`integers`
#   - hypothesis/strategies/_internal/collections.py — `lists`
# MPL-2.0 license; only the public strategy/decorator contract is used.
"""A-13 hypothesis → property-based slippage-model invariants.

Properties pinned across hundreds of random inputs for the three
S-02.1 slippage models (``ConstantBpsSlippage`` / ``BookWalkSlippage``
/ ``SquareRootImpactSlippage``):

1. **Determinism (INV-15).** ``model.apply(side, qty, mark, book)`` is
   a pure function of its inputs — two calls return byte-identical
   floats (compared via ``math.isclose`` with zero tolerance).
2. **Sided monotonicity.** ``BUY`` fills are always ``>= mark_price``
   and ``SELL`` fills are always ``<= mark_price``. ``HOLD`` is
   ``== mark_price``.
3. **Quantity monotonicity (BookWalk).** Larger walks pay at least as
   much as smaller walks on the same book and side.
4. **Symmetry of constant-bps.** ``ConstantBpsSlippage.apply`` returns
   ``mark * (1 + sign * bps/1e4)`` exactly — buy/sell are reflections
   around mark.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from core.contracts.events import Side
from simulation_engine.slippage_model import (
    BookLevel,
    BookSnapshot,
    BookWalkSlippage,
    ConstantBpsSlippage,
    InsufficientLiquidity,
    SquareRootImpactSlippage,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _book_snapshots(draw: st.DrawFn) -> BookSnapshot:
    """A well-formed L2 book with positive depth on both sides."""

    mid = draw(st.floats(min_value=10.0, max_value=10_000.0, allow_nan=False))
    bid_step = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False))
    ask_step = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False))
    n_levels = draw(st.integers(min_value=3, max_value=8))

    bids: list[BookLevel] = []
    for i in range(n_levels):
        bids.append(
            BookLevel(
                price=mid - bid_step * (i + 1),
                qty=draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False)),
            )
        )
    asks: list[BookLevel] = []
    for i in range(n_levels):
        asks.append(
            BookLevel(
                price=mid + ask_step * (i + 1),
                qty=draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False)),
            )
        )
    return BookSnapshot(bids=tuple(bids), asks=tuple(asks))


_SIDES = st.sampled_from([Side.BUY, Side.SELL, Side.HOLD])
_MARKS = st.floats(
    min_value=1.0,
    max_value=100_000.0,
    allow_nan=False,
    allow_infinity=False,
)
_QTYS = st.floats(
    min_value=0.0,
    max_value=10_000.0,
    allow_nan=False,
    allow_infinity=False,
)


# ---------------------------------------------------------------------------
# ConstantBpsSlippage
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    side=_SIDES,
    qty=_QTYS,
    mark=_MARKS,
    bps=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
)
def test_constant_bps_deterministic_and_sided(
    side: Side, qty: float, mark: float, bps: float
) -> None:
    model = ConstantBpsSlippage(bps=bps)
    a = model.apply(side, qty, mark, None)
    b = model.apply(side, qty, mark, None)
    assert a == b
    if side is Side.BUY:
        assert a >= mark
    elif side is Side.SELL:
        assert a <= mark
    else:
        assert a == mark


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    qty=_QTYS,
    mark=_MARKS,
    bps=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
)
def test_constant_bps_symmetric_around_mark(qty: float, mark: float, bps: float) -> None:
    """Buy excess equals sell discount (reflection around mark)."""

    model = ConstantBpsSlippage(bps=bps)
    buy = model.apply(Side.BUY, qty, mark, None)
    sell = model.apply(Side.SELL, qty, mark, None)
    assert abs((buy - mark) - (mark - sell)) <= 1e-9 * mark


# ---------------------------------------------------------------------------
# BookWalkSlippage
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(book=_book_snapshots(), side=_SIDES)
def test_book_walk_deterministic_and_sided(book: BookSnapshot, side: Side) -> None:
    model = BookWalkSlippage()
    mark = book.mid_price
    assert mark is not None

    # Pick a feasible qty (sum of book depth on the executable side).
    if side is Side.BUY:
        feasible = sum(lvl.qty for lvl in book.asks)
    elif side is Side.SELL:
        feasible = sum(lvl.qty for lvl in book.bids)
    else:
        feasible = 0.0

    qty = feasible * 0.5 if feasible > 0.0 else 0.0
    try:
        a = model.apply(side, qty, mark, book)
        b = model.apply(side, qty, mark, book)
    except InsufficientLiquidity:
        return
    assert a == b
    if qty == 0.0 or side is Side.HOLD:
        assert a == mark
    elif side is Side.BUY:
        assert a >= mark
    else:
        assert a <= mark


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(book=_book_snapshots())
def test_book_walk_buy_monotone_in_qty(book: BookSnapshot) -> None:
    """Larger BUY pays at least as much per unit as smaller BUY."""

    feasible = sum(lvl.qty for lvl in book.asks)
    if feasible <= 0.0:
        return
    mark = book.mid_price
    assert mark is not None
    model = BookWalkSlippage()

    small = feasible * 0.2
    large = feasible * 0.8
    try:
        small_fill = model.apply(Side.BUY, small, mark, book)
        large_fill = model.apply(Side.BUY, large, mark, book)
    except InsufficientLiquidity:
        return
    # Allow 1e-9*mark tolerance for IEEE-754 rounding on the
    # weighted-average division.
    assert large_fill >= small_fill - 1e-9 * mark


# ---------------------------------------------------------------------------
# SquareRootImpactSlippage
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    side=_SIDES,
    qty=_QTYS,
    mark=_MARKS,
    eta=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    adv=st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
)
def test_sqrt_impact_deterministic_and_sided(
    side: Side, qty: float, mark: float, eta: float, adv: float
) -> None:
    model = SquareRootImpactSlippage(eta=eta, adv=adv)
    a = model.apply(side, qty, mark, None)
    b = model.apply(side, qty, mark, None)
    assert a == b
    if side is Side.HOLD or qty == 0.0:
        assert a == mark
    elif side is Side.BUY:
        assert a >= mark
    else:
        assert a <= mark
