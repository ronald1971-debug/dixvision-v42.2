"""Tests for ``simulation_engine/latency_model.py`` (S-02.2 hftbacktest)."""

from __future__ import annotations

import dataclasses

import pytest

from simulation_engine.latency_model import (
    NEW_PIP_DEPENDENCIES,
    ConstantLatency,
    InterpolatedLatency,
    JitteredLatency,
    LatencyModel,
    LatencySample,
)

# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_no_pip_dependency_added() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_public_surface_is_documented() -> None:
    import simulation_engine.latency_model as m

    assert set(m.__all__) == {
        "ConstantLatency",
        "InterpolatedLatency",
        "JitteredLatency",
        "LatencyModel",
        "LatencySample",
        "NEW_PIP_DEPENDENCIES",
    }


# ---------------------------------------------------------------------------
# LatencySample
# ---------------------------------------------------------------------------


def test_latency_sample_basic() -> None:
    s = LatencySample(entry_latency_ns=100, response_latency_ns=200)
    assert s.entry_latency_ns == 100
    assert s.response_latency_ns == 200
    assert s.round_trip_ns == 300


def test_latency_sample_zero_is_allowed() -> None:
    s = LatencySample(entry_latency_ns=0, response_latency_ns=0)
    assert s.round_trip_ns == 0


def test_latency_sample_rejects_negative_entry() -> None:
    with pytest.raises(ValueError, match="entry_latency_ns"):
        LatencySample(entry_latency_ns=-1, response_latency_ns=0)


def test_latency_sample_rejects_negative_response() -> None:
    with pytest.raises(ValueError, match="response_latency_ns"):
        LatencySample(entry_latency_ns=0, response_latency_ns=-1)


def test_latency_sample_rejects_float_entry() -> None:
    with pytest.raises(TypeError, match="entry_latency_ns"):
        LatencySample(
            entry_latency_ns=1.0,  # type: ignore[arg-type]
            response_latency_ns=0,
        )


def test_latency_sample_rejects_float_response() -> None:
    with pytest.raises(TypeError, match="response_latency_ns"):
        LatencySample(
            entry_latency_ns=0,
            response_latency_ns=1.5,  # type: ignore[arg-type]
        )


def test_latency_sample_is_frozen() -> None:
    s = LatencySample(entry_latency_ns=1, response_latency_ns=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.entry_latency_ns = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_constant_latency_satisfies_protocol() -> None:
    m = ConstantLatency()
    assert isinstance(m, LatencyModel)


def test_interpolated_latency_satisfies_protocol() -> None:
    m = InterpolatedLatency(samples=((0, 100, 200),))
    assert isinstance(m, LatencyModel)


def test_jittered_latency_satisfies_protocol() -> None:
    m = JitteredLatency(base=ConstantLatency(100, 200))
    assert isinstance(m, LatencyModel)


# ---------------------------------------------------------------------------
# ConstantLatency
# ---------------------------------------------------------------------------


def test_constant_latency_default_is_zero() -> None:
    m = ConstantLatency()
    s = m.sample(ts_ns=0)
    assert s.entry_latency_ns == 0
    assert s.response_latency_ns == 0


def test_constant_latency_returns_fixed_pair() -> None:
    m = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    s = m.sample(ts_ns=0)
    assert s.entry_latency_ns == 500
    assert s.response_latency_ns == 1500


def test_constant_latency_ignores_ts_ns() -> None:
    m = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    s_a = m.sample(ts_ns=0)
    s_b = m.sample(ts_ns=10**18)
    assert s_a == s_b


def test_constant_latency_ignores_seed() -> None:
    m = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    s_a = m.sample(ts_ns=0, seed=0)
    s_b = m.sample(ts_ns=0, seed=999_999)
    assert s_a == s_b


def test_constant_latency_has_name() -> None:
    assert ConstantLatency().name == "constant_latency"


def test_constant_latency_rejects_negative_entry() -> None:
    with pytest.raises(ValueError, match="entry_latency_ns"):
        ConstantLatency(entry_latency_ns=-1, response_latency_ns=0)


def test_constant_latency_rejects_negative_response() -> None:
    with pytest.raises(ValueError, match="response_latency_ns"):
        ConstantLatency(entry_latency_ns=0, response_latency_ns=-1)


# ---------------------------------------------------------------------------
# InterpolatedLatency
# ---------------------------------------------------------------------------


def test_interpolated_latency_single_row_is_constant() -> None:
    m = InterpolatedLatency(samples=((0, 100, 200),))
    assert m.sample(ts_ns=0) == LatencySample(100, 200)
    assert m.sample(ts_ns=10**12) == LatencySample(100, 200)
    assert m.sample(ts_ns=-(10**12)) == LatencySample(100, 200)


def test_interpolated_latency_clamps_left() -> None:
    m = InterpolatedLatency(
        samples=((1000, 100, 200), (2000, 300, 400)),
    )
    s = m.sample(ts_ns=500)
    assert s == LatencySample(100, 200)


def test_interpolated_latency_clamps_right() -> None:
    m = InterpolatedLatency(
        samples=((1000, 100, 200), (2000, 300, 400)),
    )
    s = m.sample(ts_ns=5000)
    assert s == LatencySample(300, 400)


def test_interpolated_latency_endpoint_match_left() -> None:
    m = InterpolatedLatency(
        samples=((1000, 100, 200), (2000, 300, 400)),
    )
    assert m.sample(ts_ns=1000) == LatencySample(100, 200)


def test_interpolated_latency_endpoint_match_right() -> None:
    m = InterpolatedLatency(
        samples=((1000, 100, 200), (2000, 300, 400)),
    )
    assert m.sample(ts_ns=2000) == LatencySample(300, 400)


def test_interpolated_latency_midpoint_linear() -> None:
    m = InterpolatedLatency(
        samples=((1000, 100, 200), (2000, 300, 400)),
    )
    s = m.sample(ts_ns=1500)
    assert s.entry_latency_ns == 200
    assert s.response_latency_ns == 300


def test_interpolated_latency_quarter_point_linear() -> None:
    m = InterpolatedLatency(
        samples=((0, 0, 0), (4000, 400, 800)),
    )
    s = m.sample(ts_ns=1000)
    assert s.entry_latency_ns == 100
    assert s.response_latency_ns == 200


def test_interpolated_latency_three_segment_walk() -> None:
    m = InterpolatedLatency(
        samples=(
            (0, 0, 0),
            (1000, 100, 200),
            (2000, 50, 150),
        ),
    )
    # Mid of first segment.
    a = m.sample(ts_ns=500)
    assert a.entry_latency_ns == 50
    assert a.response_latency_ns == 100
    # Mid of second segment (latencies *decrease* — must still
    # interpolate linearly, never clamp early).
    b = m.sample(ts_ns=1500)
    assert b.entry_latency_ns == 75
    assert b.response_latency_ns == 175


def test_interpolated_latency_ignores_seed() -> None:
    m = InterpolatedLatency(
        samples=((1000, 100, 200), (2000, 300, 400)),
    )
    s_a = m.sample(ts_ns=1500, seed=0)
    s_b = m.sample(ts_ns=1500, seed=10**9)
    assert s_a == s_b


def test_interpolated_latency_has_name() -> None:
    m = InterpolatedLatency(samples=((0, 0, 0),))
    assert m.name == "interpolated_latency"


def test_interpolated_latency_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        InterpolatedLatency(samples=())


def test_interpolated_latency_rejects_non_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        InterpolatedLatency(
            samples=((1000, 100, 200), (1000, 300, 400)),
        )


def test_interpolated_latency_rejects_decreasing_ts() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        InterpolatedLatency(
            samples=((1000, 100, 200), (500, 300, 400)),
        )


def test_interpolated_latency_rejects_negative_entry() -> None:
    with pytest.raises(ValueError, match="latencies must"):
        InterpolatedLatency(samples=((0, -1, 200),))


def test_interpolated_latency_rejects_negative_response() -> None:
    with pytest.raises(ValueError, match="latencies must"):
        InterpolatedLatency(samples=((0, 100, -1),))


def test_interpolated_latency_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="3-tuple"):
        InterpolatedLatency(
            samples=((0, 100),),  # type: ignore[arg-type]
        )


def test_interpolated_latency_rejects_non_int_ts() -> None:
    with pytest.raises(TypeError, match="all int"):
        InterpolatedLatency(
            samples=((0.0, 100, 200),),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# JitteredLatency
# ---------------------------------------------------------------------------


def test_jittered_latency_with_zero_caps_recovers_constant() -> None:
    base = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    m = JitteredLatency(base=base)
    s_a = m.sample(ts_ns=0, seed=0)
    s_b = m.sample(ts_ns=10**12, seed=10**9)
    assert s_a == s_b == LatencySample(500, 1500)


def test_jittered_latency_within_bounds() -> None:
    base = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    m = JitteredLatency(
        base=base,
        max_jitter_entry_ns=100,
        max_jitter_response_ns=200,
    )
    for ts_ns in range(0, 10_000, 137):
        for seed in range(0, 50):
            s = m.sample(ts_ns=ts_ns, seed=seed)
            assert 500 <= s.entry_latency_ns <= 600
            assert 1500 <= s.response_latency_ns <= 1700


def test_jittered_latency_is_replay_deterministic() -> None:
    base = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    m = JitteredLatency(
        base=base,
        max_jitter_entry_ns=100,
        max_jitter_response_ns=200,
    )
    a = m.sample(ts_ns=12345, seed=42)
    b = m.sample(ts_ns=12345, seed=42)
    c = m.sample(ts_ns=12345, seed=42)
    assert a == b == c


def test_jittered_latency_seed_changes_sample() -> None:
    base = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    m = JitteredLatency(
        base=base,
        max_jitter_entry_ns=10_000,
        max_jitter_response_ns=10_000,
    )
    seen: set[LatencySample] = set()
    for seed in range(50):
        seen.add(m.sample(ts_ns=0, seed=seed))
    # 50 different seeds should produce more than one distinct draw —
    # if they don't, the seed isn't actually being mixed in.
    assert len(seen) > 1


def test_jittered_latency_ts_ns_changes_sample() -> None:
    base = ConstantLatency(entry_latency_ns=500, response_latency_ns=1500)
    m = JitteredLatency(
        base=base,
        max_jitter_entry_ns=10_000,
        max_jitter_response_ns=10_000,
    )
    seen: set[LatencySample] = set()
    for ts_ns in range(0, 50_000, 1000):
        seen.add(m.sample(ts_ns=ts_ns, seed=0))
    assert len(seen) > 1


def test_jittered_latency_entry_and_response_independent() -> None:
    base = ConstantLatency(entry_latency_ns=0, response_latency_ns=0)
    m = JitteredLatency(
        base=base,
        max_jitter_entry_ns=10_000,
        max_jitter_response_ns=10_000,
    )
    # If the two legs were perfectly correlated, every sample would
    # have entry == response. Across many seeds we expect at least
    # one sample where they differ.
    assert any(
        m.sample(ts_ns=0, seed=seed).entry_latency_ns
        != m.sample(ts_ns=0, seed=seed).response_latency_ns
        for seed in range(100)
    )


def test_jittered_latency_has_name() -> None:
    assert JitteredLatency(base=ConstantLatency()).name == "jittered_latency"


def test_jittered_latency_rejects_negative_jitter_entry() -> None:
    with pytest.raises(ValueError, match="max_jitter_entry_ns"):
        JitteredLatency(base=ConstantLatency(), max_jitter_entry_ns=-1)


def test_jittered_latency_rejects_negative_jitter_response() -> None:
    with pytest.raises(ValueError, match="max_jitter_response_ns"):
        JitteredLatency(base=ConstantLatency(), max_jitter_response_ns=-1)


# ---------------------------------------------------------------------------
# Replay determinism (INV-15) across all three models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        ConstantLatency(entry_latency_ns=500, response_latency_ns=1500),
        InterpolatedLatency(
            samples=(
                (0, 100, 200),
                (1000, 200, 400),
                (2000, 100, 300),
            ),
        ),
        JitteredLatency(
            base=ConstantLatency(entry_latency_ns=500, response_latency_ns=1500),
            max_jitter_entry_ns=100,
            max_jitter_response_ns=200,
        ),
    ],
    ids=["constant", "interpolated", "jittered"],
)
def test_replay_determinism_inv_15(model: LatencyModel) -> None:
    a = model.sample(ts_ns=1500, seed=42)
    b = model.sample(ts_ns=1500, seed=42)
    c = model.sample(ts_ns=1500, seed=42)
    assert a == b == c
