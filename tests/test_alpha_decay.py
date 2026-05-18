"""Tests for ``learning_engine/performance_analysis/alpha_decay.py`` (S-04.2)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from learning_engine.performance_analysis.alpha_decay import (
    NEW_PIP_DEPENDENCIES,
    AlphaDecayCurve,
    HorizonIC,
    ScoredObservation,
    compute_alpha_decay,
    compute_ic,
    empty_horizon_ic,
)


def _obs(
    *,
    ts_ns: int,
    symbol: str,
    score: float,
    future_return: float,
) -> ScoredObservation:
    return ScoredObservation(ts_ns=ts_ns, symbol=symbol, score=score, future_return=future_return)


# ---------------------------------------------------------------------------
# Sanity / surface
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_empty_horizon_ic_is_zero() -> None:
    h = empty_horizon_ic(5)
    assert h.horizon_steps == 5
    assert h.n_buckets == 0
    assert h.n_observations == 0
    assert h.ic_mean == 0.0
    assert h.ic_std == 0.0
    assert h.icir == 0.0
    assert h.rank_ic_mean == 0.0
    assert h.rank_ic_std == 0.0
    assert h.rank_icir == 0.0


def test_compute_ic_empty_returns_canonical_zero() -> None:
    assert compute_ic([], horizon_steps=1) == empty_horizon_ic(1)


def test_compute_alpha_decay_empty_returns_empty_curve() -> None:
    curve = compute_alpha_decay({})
    assert curve == AlphaDecayCurve(horizons=())
    assert curve.horizons_steps() == ()
    assert curve.best_horizon() is None


# ---------------------------------------------------------------------------
# ScoredObservation validation
# ---------------------------------------------------------------------------


def test_scored_observation_rejects_non_int_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        _obs(ts_ns=1.0, symbol="A", score=1.0, future_return=0.0)  # type: ignore[arg-type]


def test_scored_observation_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        _obs(ts_ns=True, symbol="A", score=1.0, future_return=0.0)  # type: ignore[arg-type]


def test_scored_observation_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be >= 0"):
        _obs(ts_ns=-1, symbol="A", score=1.0, future_return=0.0)


def test_scored_observation_rejects_non_string_symbol() -> None:
    with pytest.raises(TypeError, match="symbol must be str"):
        _obs(ts_ns=0, symbol=1, score=1.0, future_return=0.0)  # type: ignore[arg-type]


def test_scored_observation_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol must be non-empty"):
        _obs(ts_ns=0, symbol="", score=1.0, future_return=0.0)


def test_scored_observation_rejects_non_finite_score() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="score must be finite"):
            _obs(ts_ns=0, symbol="A", score=bad, future_return=0.0)


def test_scored_observation_rejects_non_finite_return() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="future_return must be finite"):
            _obs(ts_ns=0, symbol="A", score=1.0, future_return=bad)


def test_scored_observation_is_frozen() -> None:
    o = _obs(ts_ns=0, symbol="A", score=1.0, future_return=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.score = 2.0  # type: ignore[misc]


def test_scored_observation_is_hashable() -> None:
    o = _obs(ts_ns=0, symbol="A", score=1.0, future_return=0.0)
    assert len({o, o}) == 1


# ---------------------------------------------------------------------------
# HorizonIC validation
# ---------------------------------------------------------------------------


def _hic(**overrides) -> HorizonIC:
    base: dict[str, object] = dict(
        horizon_steps=1,
        n_buckets=0,
        n_observations=0,
        ic_mean=0.0,
        ic_std=0.0,
        icir=0.0,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )
    base.update(overrides)
    return HorizonIC(**base)  # type: ignore[arg-type]


def test_horizon_ic_rejects_zero_horizon_steps() -> None:
    with pytest.raises(ValueError, match="horizon_steps must be > 0"):
        _hic(horizon_steps=0)


def test_horizon_ic_rejects_negative_horizon_steps() -> None:
    with pytest.raises(ValueError, match="horizon_steps must be > 0"):
        _hic(horizon_steps=-1)


def test_horizon_ic_rejects_negative_n_buckets() -> None:
    with pytest.raises(ValueError, match="n_buckets must be >= 0"):
        _hic(n_buckets=-1)


def test_horizon_ic_rejects_negative_n_observations() -> None:
    with pytest.raises(ValueError, match="n_observations must be >= 0"):
        _hic(n_observations=-1)


def test_horizon_ic_rejects_nan_metric() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _hic(ic_mean=float("nan"))


def test_horizon_ic_rejects_negative_ic_std() -> None:
    with pytest.raises(ValueError, match="ic_std must be >= 0"):
        _hic(ic_std=-0.1)


def test_horizon_ic_rejects_negative_rank_ic_std() -> None:
    with pytest.raises(ValueError, match="rank_ic_std must be >= 0"):
        _hic(rank_ic_std=-0.1)


# ---------------------------------------------------------------------------
# AlphaDecayCurve validation
# ---------------------------------------------------------------------------


def test_alpha_decay_curve_rejects_non_tuple_horizons() -> None:
    with pytest.raises(TypeError, match="horizons must be tuple"):
        AlphaDecayCurve(horizons=[empty_horizon_ic(1)])  # type: ignore[arg-type]


def test_alpha_decay_curve_rejects_non_horizon_ic_element() -> None:
    with pytest.raises(TypeError, match="horizons must contain HorizonIC"):
        AlphaDecayCurve(horizons=("not-an-hic",))  # type: ignore[arg-type]


def test_alpha_decay_curve_rejects_unsorted_horizons() -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        AlphaDecayCurve(
            horizons=(empty_horizon_ic(5), empty_horizon_ic(1)),
        )


def test_alpha_decay_curve_rejects_duplicate_horizons() -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        AlphaDecayCurve(
            horizons=(empty_horizon_ic(5), empty_horizon_ic(5)),
        )


# ---------------------------------------------------------------------------
# compute_ic core formulas
# ---------------------------------------------------------------------------


def test_perfectly_correlated_bucket_yields_ic_one() -> None:
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
        _obs(ts_ns=1_000, symbol="B", score=2.0, future_return=0.02),
        _obs(ts_ns=1_000, symbol="C", score=3.0, future_return=0.03),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.n_buckets == 1
    assert h.n_observations == 3
    assert h.ic_mean == pytest.approx(1.0)
    # Single bucket → population std = 0 → ICIR convention is 0.0
    assert h.ic_std == 0.0
    assert h.icir == 0.0
    assert h.rank_ic_mean == pytest.approx(1.0)


def test_perfectly_anti_correlated_bucket_yields_ic_minus_one() -> None:
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.03),
        _obs(ts_ns=1_000, symbol="B", score=2.0, future_return=0.02),
        _obs(ts_ns=1_000, symbol="C", score=3.0, future_return=0.01),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.ic_mean == pytest.approx(-1.0)
    assert h.rank_ic_mean == pytest.approx(-1.0)


def test_known_pearson_value() -> None:
    # Hand-computed: scores [1,2,3,4], returns [1,3,2,5]
    # mean(s)=2.5, mean(r)=2.75
    # cov = (-1.5)(-1.75)+(-0.5)(0.25)+(0.5)(-0.75)+(1.5)(2.25) = 2.625-0.125-0.375+3.375 = 5.5
    # var(s) = 1.5²+0.5²+0.5²+1.5² = 5.0
    # var(r) = 1.75²+0.25²+0.75²+2.25² = 8.75
    # pearson = 5.5 / sqrt(5*8.75) = 5.5 / sqrt(43.75)
    expected = 5.5 / math.sqrt(43.75)
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=1.0),
        _obs(ts_ns=1_000, symbol="B", score=2.0, future_return=3.0),
        _obs(ts_ns=1_000, symbol="C", score=3.0, future_return=2.0),
        _obs(ts_ns=1_000, symbol="D", score=4.0, future_return=5.0),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.ic_mean == pytest.approx(expected)


def test_rank_ic_uses_average_rank_for_ties() -> None:
    # scores [1,1,2], returns [1,2,3]
    # ranks(scores) = [1.5, 1.5, 3];  ranks(returns) = [1, 2, 3]
    # Pearson of ranks ≈ correlation of [1.5,1.5,3] vs [1,2,3]
    # mean_x=2, mean_y=2  → cov = (-0.5)(-1)+(-0.5)(0)+(1)(1) = 1.5
    # var_x = 0.25+0.25+1 = 1.5;  var_y = 1+0+1 = 2
    # rho = 1.5 / sqrt(3.0) = 0.866...
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=1.0),
        _obs(ts_ns=1_000, symbol="B", score=1.0, future_return=2.0),
        _obs(ts_ns=1_000, symbol="C", score=2.0, future_return=3.0),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.rank_ic_mean == pytest.approx(1.5 / math.sqrt(3.0))


# ---------------------------------------------------------------------------
# Bucketing / aggregation
# ---------------------------------------------------------------------------


def test_buckets_are_grouped_by_ts_ns_independently() -> None:
    # bucket 1 perfectly correlated, bucket 2 perfectly anti-correlated
    obs = [
        # bucket 1
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
        _obs(ts_ns=1_000, symbol="B", score=2.0, future_return=0.02),
        # bucket 2
        _obs(ts_ns=2_000, symbol="A", score=1.0, future_return=0.02),
        _obs(ts_ns=2_000, symbol="B", score=2.0, future_return=0.01),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.n_buckets == 2
    assert h.n_observations == 4
    assert h.ic_mean == pytest.approx(0.0)
    # ICs are [+1, -1] → population std = 1.0 → ICIR = 0.0/1.0 = 0.0
    assert h.ic_std == pytest.approx(1.0)
    assert h.icir == 0.0


def test_singleton_buckets_are_skipped() -> None:
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
        # bucket 2 has only one row → degenerate, should be skipped
        _obs(ts_ns=2_000, symbol="A", score=1.0, future_return=0.01),
    ]
    h = compute_ic(obs, horizon_steps=1)
    # 2 observations, but 0 surviving buckets (each bucket has < 2 rows)
    assert h.n_observations == 2
    assert h.n_buckets == 0
    assert h.ic_mean == 0.0
    assert h.ic_std == 0.0


def test_zero_variance_buckets_are_skipped() -> None:
    # constant scores → score variance = 0 → bucket dropped
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
        _obs(ts_ns=1_000, symbol="B", score=1.0, future_return=0.02),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.n_buckets == 0
    assert h.ic_mean == 0.0


def test_n_observations_counts_input_rows_even_if_dropped() -> None:
    obs = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
    ]
    h = compute_ic(obs, horizon_steps=1)
    assert h.n_observations == 1
    assert h.n_buckets == 0


def test_compute_ic_rejects_non_observation() -> None:
    with pytest.raises(TypeError, match="observations must contain"):
        compute_ic([object()], horizon_steps=1)  # type: ignore[list-item]


def test_compute_ic_rejects_non_int_horizon() -> None:
    with pytest.raises(TypeError, match="horizon_steps must be int"):
        compute_ic([], horizon_steps=1.0)  # type: ignore[arg-type]


def test_compute_ic_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_steps must be > 0"):
        compute_ic([], horizon_steps=0)


def test_compute_ic_consumes_iterator_once() -> None:
    obs_list = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
        _obs(ts_ns=1_000, symbol="B", score=2.0, future_return=0.02),
    ]

    def gen():
        yield from obs_list

    h = compute_ic(gen(), horizon_steps=1)
    assert h.n_observations == 2
    assert h.ic_mean == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AlphaDecayCurve / compute_alpha_decay
# ---------------------------------------------------------------------------


def _strong_obs(ts_ns: int) -> list[ScoredObservation]:
    return [
        _obs(ts_ns=ts_ns, symbol="A", score=1.0, future_return=0.01),
        _obs(ts_ns=ts_ns, symbol="B", score=2.0, future_return=0.02),
        _obs(ts_ns=ts_ns, symbol="C", score=3.0, future_return=0.03),
    ]


def _weak_obs(ts_ns: int) -> list[ScoredObservation]:
    return [
        _obs(ts_ns=ts_ns, symbol="A", score=1.0, future_return=0.03),
        _obs(ts_ns=ts_ns, symbol="B", score=2.0, future_return=0.02),
        _obs(ts_ns=ts_ns, symbol="C", score=3.0, future_return=0.01),
    ]


def test_compute_alpha_decay_orders_horizons_strictly_ascending() -> None:
    # Insert in random key order; output must be sorted ascending.
    by_h = {
        20: _strong_obs(2_000) + _strong_obs(3_000),
        1: _strong_obs(1_000) + _strong_obs(2_000),
        5: _strong_obs(1_000) + _strong_obs(2_000),
    }
    curve = compute_alpha_decay(by_h)
    assert curve.horizons_steps() == (1, 5, 20)


def test_compute_alpha_decay_records_each_horizon_independently() -> None:
    by_h = {
        1: _strong_obs(1_000) + _strong_obs(2_000),
        5: _weak_obs(1_000) + _weak_obs(2_000),
    }
    curve = compute_alpha_decay(by_h)
    by_horizon = {h.horizon_steps: h for h in curve.horizons}
    assert by_horizon[1].ic_mean == pytest.approx(1.0)
    assert by_horizon[5].ic_mean == pytest.approx(-1.0)


def test_best_horizon_picks_largest_icir() -> None:
    # Hand-craft three horizons with distinct ic_std so ICIR differs.
    hic_short = HorizonIC(
        horizon_steps=1,
        n_buckets=2,
        n_observations=4,
        ic_mean=0.20,
        ic_std=0.10,
        icir=2.0,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )
    hic_mid = HorizonIC(
        horizon_steps=5,
        n_buckets=2,
        n_observations=4,
        ic_mean=0.30,
        ic_std=0.10,
        icir=3.0,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )
    hic_long = HorizonIC(
        horizon_steps=20,
        n_buckets=2,
        n_observations=4,
        ic_mean=0.05,
        ic_std=0.10,
        icir=0.5,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )
    curve = AlphaDecayCurve(horizons=(hic_short, hic_mid, hic_long))
    assert curve.best_horizon() is hic_mid


def test_best_horizon_breaks_ties_with_smaller_horizon() -> None:
    a = HorizonIC(
        horizon_steps=5,
        n_buckets=2,
        n_observations=4,
        ic_mean=0.10,
        ic_std=0.05,
        icir=2.0,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )
    b = HorizonIC(
        horizon_steps=10,
        n_buckets=2,
        n_observations=4,
        ic_mean=0.10,
        ic_std=0.05,
        icir=2.0,
        rank_ic_mean=0.0,
        rank_ic_std=0.0,
        rank_icir=0.0,
    )
    curve = AlphaDecayCurve(horizons=(a, b))
    assert curve.best_horizon() is a  # smaller horizon wins on tie


def test_compute_alpha_decay_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="must be Mapping"):
        compute_alpha_decay([1, 2, 3])  # type: ignore[arg-type]


def test_compute_alpha_decay_rejects_non_int_key() -> None:
    with pytest.raises(TypeError, match="keys must be int"):
        compute_alpha_decay({"1": []})  # type: ignore[dict-item]


def test_compute_alpha_decay_rejects_non_positive_key() -> None:
    with pytest.raises(ValueError, match="keys must be > 0"):
        compute_alpha_decay({0: []})


# ---------------------------------------------------------------------------
# Replay determinism (INV-15)
# ---------------------------------------------------------------------------


def test_replay_determinism_across_three_runs() -> None:
    obs = []
    for t in range(10):
        for i, sym in enumerate(("Z", "Y", "X", "W", "V")):
            obs.append(
                _obs(
                    ts_ns=t * 1_000,
                    symbol=sym,
                    score=float(i + t * 0.1),
                    future_return=0.001 * (i - 2) + 0.0001 * t,
                )
            )
    a = compute_ic(obs, horizon_steps=3)
    b = compute_ic(obs, horizon_steps=3)
    c = compute_ic(obs, horizon_steps=3)
    assert a == b == c

    by_h = {1: obs, 5: obs, 20: obs}
    curve_a = compute_alpha_decay(by_h)
    curve_b = compute_alpha_decay(by_h)
    assert curve_a == curve_b
    assert curve_a.horizons_steps() == (1, 5, 20)


def test_symbol_insertion_order_does_not_change_result() -> None:
    a = [
        _obs(ts_ns=1_000, symbol="A", score=1.0, future_return=0.01),
        _obs(ts_ns=1_000, symbol="B", score=2.0, future_return=0.02),
        _obs(ts_ns=1_000, symbol="C", score=3.0, future_return=0.03),
    ]
    b = list(reversed(a))
    assert compute_ic(a, horizon_steps=1) == compute_ic(b, horizon_steps=1)
