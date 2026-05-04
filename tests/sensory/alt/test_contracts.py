"""Unit tests for sensory.alt.contracts."""

from __future__ import annotations

import math

import pytest

from sensory.alt.contracts import PredictionMarket


def _ok(**overrides: object) -> PredictionMarket:
    kwargs = {
        "ts_ns": 1,
        "source": "POLYMARKET",
        "market_id": "0xabc",
        "question": "Will X happen by date Y?",
        "outcome": "YES",
        "probability": 0.42,
    }
    kwargs.update(overrides)
    return PredictionMarket(**kwargs)  # type: ignore[arg-type]


def test_minimal_construct() -> None:
    p = _ok()
    assert p.volume_usd is None
    assert p.observed_ts_ns is None
    assert dict(p.meta) == {}


def test_full_construct() -> None:
    p = _ok(
        volume_usd=12_345.0,
        observed_ts_ns=2,
        meta={"slug": "btc-100k-2025"},
    )
    assert p.volume_usd == 12_345.0
    assert p.observed_ts_ns == 2


def test_frozen_and_slotted() -> None:
    p = _ok()
    with pytest.raises(AttributeError):
        p.probability = 0.99  # type: ignore[misc]


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", ""),
        ("market_id", ""),
        ("question", ""),
        ("outcome", ""),
        ("probability", -0.01),
        ("probability", 1.01),
        ("volume_usd", -1.0),
        ("observed_ts_ns", 0),
        ("observed_ts_ns", -1),
    ],
)
def test_validation_rejects(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _ok(**{field: value})


def test_probability_bounds_inclusive() -> None:
    assert _ok(probability=0.0).probability == 0.0
    assert _ok(probability=1.0).probability == 1.0


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf")],
)
def test_volume_usd_rejects_nan_and_infinity(bad: float) -> None:
    """volume_usd must reject NaN and -Inf — INV-15 contract.

    Regression: ``< 0`` evaluates to ``False`` for NaN under IEEE 754,
    so a buggy upstream (e.g. a divide-by-zero in a Polymarket adapter)
    would have silently produced an invalid PredictionMarket. The
    validator now uses ``not (>= 0)`` so NaN and -inf fail; +inf still
    passes the ``>= 0`` predicate (tighten to ``math.isfinite`` if a
    future product requirement needs to reject +inf as well).
    """
    if math.isnan(bad) or bad < 0:
        with pytest.raises(ValueError, match="volume_usd"):
            _ok(volume_usd=bad)
    else:
        # +inf passes the >= 0 predicate but is still nonsensical for a
        # USD volume. Document current behaviour: +inf is accepted.
        # If product later wants to reject +inf, tighten to math.isfinite.
        p = _ok(volume_usd=bad)
        assert p.volume_usd == bad


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_probability_rejects_nan_and_infinity(bad: float) -> None:
    """probability already rejects NaN via the bounds check; assert it."""
    with pytest.raises(ValueError, match="probability"):
        _ok(probability=bad)
