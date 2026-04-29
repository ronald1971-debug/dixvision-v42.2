"""Tests for the Almgren-Chriss strategic-execution scheduler."""

from __future__ import annotations

import math

import pytest

from execution_engine.strategic import (
    ExecutionSchedule,
    ExecutionSlice,
    solve_almgren_chriss,
)


def _solve(**overrides) -> ExecutionSchedule:
    base = dict(
        quantity=1000.0,
        horizon_seconds=600.0,
        n_slices=10,
        sigma=0.02,
        eta=1e-4,
        gamma=0.0,
        risk_aversion=0.0,
    )
    base.update(overrides)
    return solve_almgren_chriss(**base)


# ---------------------------------------------------------------------------
# Schedule shape
# ---------------------------------------------------------------------------


def test_solve_returns_schedule_with_n_slices():
    s = _solve()
    assert isinstance(s, ExecutionSchedule)
    assert len(s.slices) == 10


def test_slices_preserve_total_quantity_long():
    s = _solve(quantity=1000.0)
    total = sum(sl.quantity for sl in s.slices)
    assert math.isclose(total, 1000.0, abs_tol=1e-9)


def test_slices_preserve_total_quantity_short():
    s = _solve(quantity=-500.0)
    total = sum(sl.quantity for sl in s.slices)
    assert math.isclose(total, -500.0, abs_tol=1e-9)


def test_slice_time_offsets_are_strictly_increasing():
    s = _solve()
    offsets = [sl.time_offset_seconds for sl in s.slices]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)
    assert math.isclose(offsets[-1], s.horizon_seconds, abs_tol=1e-9)


def test_holdings_after_final_slice_is_zero():
    s = _solve()
    assert math.isclose(s.slices[-1].holdings_after, 0.0, abs_tol=1e-9)


def test_holdings_after_monotone_for_long_liquidation():
    s = _solve(quantity=1000.0, sigma=0.05, risk_aversion=1e-3)
    holdings = [s.quantity, *[sl.holdings_after for sl in s.slices]]
    for a, b in zip(holdings[:-1], holdings[1:], strict=True):
        assert b <= a + 1e-9


def test_holdings_after_monotone_for_short_acquisition():
    s = _solve(quantity=-1000.0, sigma=0.05, risk_aversion=1e-3)
    holdings = [s.quantity, *[sl.holdings_after for sl in s.slices]]
    for a, b in zip(holdings[:-1], holdings[1:], strict=True):
        # Acquisitions: holdings move toward zero from below.
        assert b >= a - 1e-9


def test_slice_quantities_share_parent_sign_for_long():
    s = _solve(quantity=1000.0, sigma=0.05, risk_aversion=1e-3)
    assert all(sl.quantity > 0 for sl in s.slices)


def test_slice_quantities_share_parent_sign_for_short():
    s = _solve(quantity=-1000.0, sigma=0.05, risk_aversion=1e-3)
    assert all(sl.quantity < 0 for sl in s.slices)


# ---------------------------------------------------------------------------
# Limiting cases
# ---------------------------------------------------------------------------


def test_zero_risk_aversion_gives_twap():
    s = _solve(risk_aversion=0.0)
    expected = 1000.0 / 10
    for sl in s.slices:
        assert math.isclose(sl.quantity, expected, abs_tol=1e-9)
    assert s.is_twap()
    assert s.kappa == 0.0


def test_zero_sigma_gives_twap_even_with_risk_aversion():
    s = _solve(sigma=0.0, risk_aversion=10.0)
    expected = 1000.0 / 10
    for sl in s.slices:
        assert math.isclose(sl.quantity, expected, abs_tol=1e-9)
    assert s.is_twap()


def test_high_risk_aversion_front_loads_schedule():
    s = _solve(sigma=0.5, risk_aversion=100.0)
    # First slice strictly larger than last for any positive kappa.
    first = s.slices[0].quantity
    last = s.slices[-1].quantity
    assert first > last
    # And first slice is bigger than the TWAP slice.
    assert first > 1000.0 / 10


def test_low_risk_aversion_close_to_twap():
    s = _solve(sigma=0.01, risk_aversion=1e-12)
    twap_slice = 1000.0 / 10
    for sl in s.slices:
        assert math.isclose(sl.quantity, twap_slice, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_same_inputs_produce_identical_slices():
    s1 = _solve(sigma=0.05, risk_aversion=2e-4)
    s2 = _solve(sigma=0.05, risk_aversion=2e-4)
    assert s1.slices == s2.slices
    assert s1.kappa == s2.kappa


def test_kappa_grows_monotonically_in_risk_aversion():
    a = _solve(sigma=0.05, risk_aversion=1e-5).kappa
    b = _solve(sigma=0.05, risk_aversion=1e-3).kappa
    c = _solve(sigma=0.05, risk_aversion=1e-1).kappa
    assert 0 <= a < b < c


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"horizon_seconds": 0.0}, "horizon_seconds"),
        ({"horizon_seconds": -10.0}, "horizon_seconds"),
        ({"n_slices": 0}, "n_slices"),
        ({"n_slices": -1}, "n_slices"),
        ({"sigma": -0.1}, "sigma"),
        ({"eta": 0.0}, "eta"),
        ({"eta": -1.0}, "eta"),
        ({"gamma": -1.0}, "gamma"),
        ({"risk_aversion": -1.0}, "risk_aversion"),
        ({"quantity": float("nan")}, "quantity"),
        ({"horizon_seconds": float("inf")}, "horizon_seconds"),
        ({"sigma": float("nan")}, "sigma"),
    ],
)
def test_validation_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _solve(**kwargs)


def test_permanent_impact_too_large_rejected():
    # gamma * tau / 2 must be < eta
    with pytest.raises(ValueError, match="eta - gamma"):
        solve_almgren_chriss(
            quantity=1000.0,
            horizon_seconds=600.0,
            n_slices=2,
            sigma=0.02,
            eta=1e-4,
            gamma=10.0,  # gamma*tau/2 = 1500 >> eta
            risk_aversion=0.0,
        )


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------


def test_kappa_squared_matches_definition():
    sigma = 0.05
    eta = 1e-4
    gamma = 1e-6
    horizon = 300.0
    n = 30
    lam = 5e-4
    s = solve_almgren_chriss(
        quantity=10_000.0,
        horizon_seconds=horizon,
        n_slices=n,
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        risk_aversion=lam,
    )
    tau = horizon / n
    eta_tilde = eta - 0.5 * gamma * tau
    expected_kappa = math.sqrt(lam * sigma * sigma / eta_tilde)
    assert math.isclose(s.kappa, expected_kappa, rel_tol=1e-12)


def test_holdings_match_closed_form_at_midpoint():
    sigma = 0.1
    eta = 1e-3
    horizon = 100.0
    n = 10
    lam = 1e-2
    quantity = 500.0
    s = solve_almgren_chriss(
        quantity=quantity,
        horizon_seconds=horizon,
        n_slices=n,
        sigma=sigma,
        eta=eta,
        gamma=0.0,
        risk_aversion=lam,
    )
    kappa = s.kappa
    # x_5 should equal sinh(kappa * (T - 5*tau)) / sinh(kappa*T) * X
    tau = horizon / n
    expected = math.sinh(kappa * (horizon - 5 * tau)) / math.sinh(kappa * horizon) * quantity
    actual = s.slices[4].holdings_after  # after slice index 4 -> after k=5
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_zero_quantity_returns_all_zero_slices():
    s = _solve(quantity=0.0)
    assert all(sl.quantity == 0.0 for sl in s.slices)
    assert all(sl.holdings_after == 0.0 for sl in s.slices)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_schedule_is_frozen():
    s = _solve()
    with pytest.raises((AttributeError, TypeError)):
        s.kappa = 999  # type: ignore[misc]


def test_slice_is_frozen():
    s = _solve()
    with pytest.raises((AttributeError, TypeError)):
        s.slices[0].quantity = 999  # type: ignore[misc]


def test_slice_post_init_rejects_negative_index():
    with pytest.raises(ValueError, match="slice.index"):
        ExecutionSlice(index=-1, time_offset_seconds=0.0, quantity=0.0, holdings_after=0.0)


def test_slice_post_init_rejects_negative_time():
    with pytest.raises(ValueError, match="time_offset"):
        ExecutionSlice(index=0, time_offset_seconds=-1.0, quantity=0.0, holdings_after=0.0)
