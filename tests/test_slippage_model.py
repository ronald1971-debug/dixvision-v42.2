"""Unit tests for `simulation_engine.slippage_model` (S-02.1, hftbacktest).

Pure pytest, no UI. Verifies Protocol satisfaction, IEEE-754 NaN
safety, book-walk algorithm correctness, square-root impact closed
form, and INV-15 replay determinism.
"""

from __future__ import annotations

import math

import pytest

from core.contracts.events import Side
from simulation_engine.slippage_model import (
    NEW_PIP_DEPENDENCIES,
    BookLevel,
    BookSnapshot,
    BookWalkSlippage,
    ConstantBpsSlippage,
    InsufficientLiquidity,
    SlippageModel,
    SquareRootImpactSlippage,
)

# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_no_pip_dependency_added() -> None:
    """We reproduce the algorithms in pure Python; no new pip dep."""
    assert NEW_PIP_DEPENDENCIES == ()


# ---------------------------------------------------------------------------
# BookLevel / BookSnapshot
# ---------------------------------------------------------------------------


def test_book_level_validates_positive_price() -> None:
    with pytest.raises(ValueError, match="price must be > 0"):
        BookLevel(price=0.0, qty=1.0)
    with pytest.raises(ValueError, match="price must be > 0"):
        BookLevel(price=-1.0, qty=1.0)
    with pytest.raises(ValueError, match="price must be > 0"):
        BookLevel(price=float("nan"), qty=1.0)


def test_book_level_validates_non_negative_qty() -> None:
    with pytest.raises(ValueError, match="qty must be >= 0"):
        BookLevel(price=100.0, qty=-0.1)
    with pytest.raises(ValueError, match="qty must be >= 0"):
        BookLevel(price=100.0, qty=float("nan"))


def test_book_snapshot_rejects_unsorted_bids() -> None:
    with pytest.raises(ValueError, match="bids must be sorted descending"):
        BookSnapshot(
            bids=(BookLevel(99.0, 1.0), BookLevel(100.0, 1.0)),
            asks=(BookLevel(101.0, 1.0),),
        )


def test_book_snapshot_rejects_unsorted_asks() -> None:
    with pytest.raises(ValueError, match="asks must be sorted ascending"):
        BookSnapshot(
            bids=(BookLevel(99.0, 1.0),),
            asks=(BookLevel(102.0, 1.0), BookLevel(101.0, 1.0)),
        )


def test_book_snapshot_rejects_crossed_book() -> None:
    with pytest.raises(ValueError, match="crossed book is invalid"):
        BookSnapshot(
            bids=(BookLevel(101.0, 1.0),),
            asks=(BookLevel(101.0, 1.0),),
        )


def test_book_snapshot_mid_price() -> None:
    book = BookSnapshot(
        bids=(BookLevel(99.0, 1.0),),
        asks=(BookLevel(101.0, 1.0),),
    )
    assert book.mid_price == 100.0


def test_book_snapshot_mid_price_none_when_one_side_empty() -> None:
    only_bids = BookSnapshot(bids=(BookLevel(99.0, 1.0),), asks=())
    only_asks = BookSnapshot(bids=(), asks=(BookLevel(101.0, 1.0),))
    assert only_bids.mid_price is None
    assert only_asks.mid_price is None


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        ConstantBpsSlippage(),
        BookWalkSlippage(),
        SquareRootImpactSlippage(),
    ],
)
def test_all_models_satisfy_protocol(model: SlippageModel) -> None:
    assert isinstance(model, SlippageModel)
    assert isinstance(model.name, str) and model.name


# ---------------------------------------------------------------------------
# ConstantBpsSlippage
# ---------------------------------------------------------------------------


def test_constant_bps_zero_is_identity_for_buy_and_sell() -> None:
    m = ConstantBpsSlippage(bps=0.0)
    assert m.apply(Side.BUY, 1.0, 100.0) == 100.0
    assert m.apply(Side.SELL, 1.0, 100.0) == 100.0


def test_constant_bps_buy_pays_above_mark() -> None:
    # 10 bps = 0.1% → 100.0 → 100.10
    m = ConstantBpsSlippage(bps=10.0)
    assert m.apply(Side.BUY, 1.0, 100.0) == pytest.approx(100.10)


def test_constant_bps_sell_receives_below_mark() -> None:
    m = ConstantBpsSlippage(bps=10.0)
    assert m.apply(Side.SELL, 1.0, 100.0) == pytest.approx(99.90)


def test_constant_bps_hold_passes_mark_through() -> None:
    m = ConstantBpsSlippage(bps=10.0)
    assert m.apply(Side.HOLD, 1.0, 100.0) == 100.0


def test_constant_bps_non_positive_mark_passes_through() -> None:
    m = ConstantBpsSlippage(bps=10.0)
    assert m.apply(Side.BUY, 1.0, 0.0) == 0.0
    assert m.apply(Side.BUY, 1.0, -5.0) == -5.0


def test_constant_bps_rejects_nan_at_construction() -> None:
    with pytest.raises(ValueError, match="bps must be >= 0"):
        ConstantBpsSlippage(bps=float("nan"))
    with pytest.raises(ValueError, match="bps must be >= 0"):
        ConstantBpsSlippage(bps=-1.0)


def test_constant_bps_ignores_qty_and_book() -> None:
    m = ConstantBpsSlippage(bps=5.0)
    book = BookSnapshot(bids=(BookLevel(99.0, 1.0),), asks=(BookLevel(101.0, 1.0),))
    assert m.apply(Side.BUY, 0.0001, 100.0, book) == m.apply(Side.BUY, 1_000_000.0, 100.0, None)


# ---------------------------------------------------------------------------
# BookWalkSlippage
# ---------------------------------------------------------------------------


def _make_book() -> BookSnapshot:
    """5-level / side toy book around mid 100.0."""
    return BookSnapshot(
        bids=(
            BookLevel(99.99, 1.0),
            BookLevel(99.98, 2.0),
            BookLevel(99.97, 3.0),
            BookLevel(99.96, 4.0),
            BookLevel(99.95, 5.0),
        ),
        asks=(
            BookLevel(100.01, 1.0),
            BookLevel(100.02, 2.0),
            BookLevel(100.03, 3.0),
            BookLevel(100.04, 4.0),
            BookLevel(100.05, 5.0),
        ),
    )


def test_book_walk_buy_one_unit_takes_top_of_book() -> None:
    m = BookWalkSlippage()
    book = _make_book()
    assert m.apply(Side.BUY, 1.0, 100.0, book) == pytest.approx(100.01)


def test_book_walk_sell_one_unit_takes_top_of_book() -> None:
    m = BookWalkSlippage()
    book = _make_book()
    assert m.apply(Side.SELL, 1.0, 100.0, book) == pytest.approx(99.99)


def test_book_walk_buy_three_units_walks_two_levels_vwap() -> None:
    m = BookWalkSlippage()
    book = _make_book()
    # Take all 1 @ 100.01, then 2 @ 100.02 → cost = 1*100.01 + 2*100.02
    # VWAP = (100.01 + 200.04) / 3
    expected = (100.01 + 2 * 100.02) / 3.0
    assert m.apply(Side.BUY, 3.0, 100.0, book) == pytest.approx(expected)


def test_book_walk_sell_six_units_walks_three_levels_vwap() -> None:
    m = BookWalkSlippage()
    book = _make_book()
    # Take 1 @ 99.99 + 2 @ 99.98 + 3 @ 99.97 = 6 units
    expected = (1 * 99.99 + 2 * 99.98 + 3 * 99.97) / 6.0
    assert m.apply(Side.SELL, 6.0, 100.0, book) == pytest.approx(expected)


def test_book_walk_falls_back_to_mark_when_book_exhausted() -> None:
    m = BookWalkSlippage(fallback_to_mark_on_empty=True)
    book = _make_book()
    # Total ask depth = 1+2+3+4+5 = 15. Request 16 → 1 unit fills at mark.
    cost_from_book = 1 * 100.01 + 2 * 100.02 + 3 * 100.03 + 4 * 100.04 + 5 * 100.05
    expected = (cost_from_book + 1 * 100.0) / 16.0
    assert m.apply(Side.BUY, 16.0, 100.0, book) == pytest.approx(expected)


def test_book_walk_raises_when_fallback_disabled_and_book_exhausted() -> None:
    m = BookWalkSlippage(fallback_to_mark_on_empty=False)
    book = _make_book()
    with pytest.raises(InsufficientLiquidity):
        m.apply(Side.BUY, 16.0, 100.0, book)


def test_book_walk_returns_mark_when_book_is_none() -> None:
    m = BookWalkSlippage()
    assert m.apply(Side.BUY, 5.0, 100.0, None) == 100.0


def test_book_walk_returns_mark_when_opposite_side_empty() -> None:
    m = BookWalkSlippage()
    only_bids = BookSnapshot(bids=(BookLevel(99.0, 1.0),), asks=())
    only_asks = BookSnapshot(bids=(), asks=(BookLevel(101.0, 1.0),))
    assert m.apply(Side.BUY, 5.0, 100.0, only_bids) == 100.0
    assert m.apply(Side.SELL, 5.0, 100.0, only_asks) == 100.0


def test_book_walk_returns_mark_for_hold_or_zero_qty() -> None:
    m = BookWalkSlippage()
    book = _make_book()
    assert m.apply(Side.HOLD, 1.0, 100.0, book) == 100.0
    assert m.apply(Side.BUY, 0.0, 100.0, book) == 100.0


def test_book_walk_skips_zero_qty_levels() -> None:
    m = BookWalkSlippage()
    # First ask level is empty → walker should skip past it.
    book = BookSnapshot(
        bids=(BookLevel(99.0, 1.0),),
        asks=(
            BookLevel(100.01, 0.0),
            BookLevel(100.02, 1.0),
        ),
    )
    assert m.apply(Side.BUY, 1.0, 100.0, book) == pytest.approx(100.02)


# ---------------------------------------------------------------------------
# SquareRootImpactSlippage
# ---------------------------------------------------------------------------


def test_sqrt_impact_zero_qty_is_mark() -> None:
    m = SquareRootImpactSlippage(eta=0.1, adv=1.0)
    assert m.apply(Side.BUY, 0.0, 100.0) == 100.0


def test_sqrt_impact_buy_closed_form() -> None:
    m = SquareRootImpactSlippage(eta=0.1, adv=4.0)
    # qty/adv = 1.0 → sqrt = 1.0 → impact = 0.1 → fill = 100*(1+0.1)
    assert m.apply(Side.BUY, 4.0, 100.0) == pytest.approx(110.0)


def test_sqrt_impact_sell_closed_form() -> None:
    m = SquareRootImpactSlippage(eta=0.1, adv=4.0)
    assert m.apply(Side.SELL, 4.0, 100.0) == pytest.approx(90.0)


def test_sqrt_impact_scales_as_sqrt_of_qty() -> None:
    m = SquareRootImpactSlippage(eta=0.1, adv=1.0)
    fill_1 = m.apply(Side.BUY, 1.0, 100.0)
    fill_4 = m.apply(Side.BUY, 4.0, 100.0)
    fill_9 = m.apply(Side.BUY, 9.0, 100.0)
    impact_1 = fill_1 - 100.0
    impact_4 = fill_4 - 100.0
    impact_9 = fill_9 - 100.0
    # impact_4 should be 2x impact_1; impact_9 should be 3x impact_1.
    assert math.isclose(impact_4 / impact_1, 2.0, rel_tol=1e-9)
    assert math.isclose(impact_9 / impact_1, 3.0, rel_tol=1e-9)


def test_sqrt_impact_validates_eta_and_adv() -> None:
    with pytest.raises(ValueError, match="eta must be >= 0"):
        SquareRootImpactSlippage(eta=-0.1, adv=1.0)
    with pytest.raises(ValueError, match="eta must be >= 0"):
        SquareRootImpactSlippage(eta=float("nan"), adv=1.0)
    with pytest.raises(ValueError, match="adv must be > 0"):
        SquareRootImpactSlippage(eta=0.1, adv=0.0)
    with pytest.raises(ValueError, match="adv must be > 0"):
        SquareRootImpactSlippage(eta=0.1, adv=float("nan"))


def test_sqrt_impact_hold_passes_mark_through() -> None:
    m = SquareRootImpactSlippage(eta=0.1, adv=1.0)
    assert m.apply(Side.HOLD, 1.0, 100.0) == 100.0


def test_sqrt_impact_ignores_book() -> None:
    m = SquareRootImpactSlippage(eta=0.1, adv=4.0)
    book = _make_book()
    assert m.apply(Side.BUY, 4.0, 100.0, book) == m.apply(Side.BUY, 4.0, 100.0, None)


# ---------------------------------------------------------------------------
# INV-15 replay determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        ConstantBpsSlippage(bps=12.5),
        BookWalkSlippage(),
        SquareRootImpactSlippage(eta=0.13, adv=4.2),
    ],
)
def test_replay_determinism_same_inputs_same_output(
    model: SlippageModel,
) -> None:
    book = _make_book()
    a = model.apply(Side.BUY, 3.0, 100.0, book)
    b = model.apply(Side.BUY, 3.0, 100.0, book)
    c = model.apply(Side.BUY, 3.0, 100.0, book)
    assert a == b == c


def test_book_walk_byte_equal_across_two_independent_instances() -> None:
    """Two independently-constructed models, identical inputs → equal."""
    book = _make_book()
    a = BookWalkSlippage().apply(Side.BUY, 7.5, 100.0, book)
    b = BookWalkSlippage().apply(Side.BUY, 7.5, 100.0, book)
    assert a == b


# ---------------------------------------------------------------------------
# Exported surface
# ---------------------------------------------------------------------------


def test_public_api_surface() -> None:
    import simulation_engine.slippage_model as mod

    expected = {
        "BookLevel",
        "BookSnapshot",
        "BookWalkSlippage",
        "ConstantBpsSlippage",
        "InsufficientLiquidity",
        "NEW_PIP_DEPENDENCIES",
        "SlippageModel",
        "SquareRootImpactSlippage",
    }
    assert expected.issubset(set(mod.__all__))
