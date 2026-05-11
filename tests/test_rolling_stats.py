# ADAPTED FROM: https://github.com/pydata/bottleneck (BSD-2-Clause).
#
# Bench-style + correctness tests for the I-14 bottleneck-shape rolling-window
# statistics module.
"""I-14 tests: bottleneck-shape rolling-window statistics."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from learning_engine.analytics import rolling_stats as rs
from learning_engine.analytics.rolling_stats import (
    NEW_PIP_DEPENDENCIES,
    RollingStatsError,
    RollingWindowSpec,
    enable_bottleneck_kernel_factory,
    move_max,
    move_mean,
    move_min,
    move_std,
    move_sum,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("bottleneck",)


def test_rolling_stats_error_is_value_error_subclass() -> None:
    assert issubclass(RollingStatsError, ValueError)


# ---------------------------------------------------------------------------
# RollingWindowSpec value-object validation
# ---------------------------------------------------------------------------


def test_window_spec_happy_path() -> None:
    spec = RollingWindowSpec(window=5, min_count=3)
    assert spec.window == 5
    assert spec.min_count == 3


def test_window_spec_is_frozen() -> None:
    spec = RollingWindowSpec(window=5, min_count=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.window = 6  # type: ignore[misc]


@pytest.mark.parametrize("bad", [0, -1, 1.5, "5"])
def test_window_spec_rejects_bad_window(bad: object) -> None:
    with pytest.raises(RollingStatsError):
        RollingWindowSpec(window=bad, min_count=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1"])
def test_window_spec_rejects_bad_min_count(bad: object) -> None:
    with pytest.raises(RollingStatsError):
        RollingWindowSpec(window=5, min_count=bad)  # type: ignore[arg-type]


def test_window_spec_rejects_min_count_exceeding_window() -> None:
    with pytest.raises(RollingStatsError):
        RollingWindowSpec(window=3, min_count=4)


# ---------------------------------------------------------------------------
# move_sum
# ---------------------------------------------------------------------------


def test_move_sum_window_one_is_identity() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(move_sum(arr, window=1), arr)


def test_move_sum_full_window() -> None:
    out = move_sum([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
    expected = np.array([np.nan, np.nan, 6.0, 9.0, 12.0])
    np.testing.assert_array_equal(out, expected)


def test_move_sum_min_count_below_window_fills_early_positions() -> None:
    out = move_sum([1.0, 2.0, 3.0, 4.0], window=3, min_count=1)
    expected = np.array([1.0, 3.0, 6.0, 9.0])
    np.testing.assert_array_equal(out, expected)


def test_move_sum_handles_nan_with_min_count_one() -> None:
    out = move_sum([1.0, np.nan, 3.0, 4.0], window=3, min_count=1)
    expected = np.array([1.0, 1.0, 4.0, 7.0])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# move_mean
# ---------------------------------------------------------------------------


def test_move_mean_full_window() -> None:
    out = move_mean([2.0, 4.0, 6.0, 8.0], window=2)
    expected = np.array([np.nan, 3.0, 5.0, 7.0])
    np.testing.assert_array_equal(out, expected)


def test_move_mean_with_nan_and_min_count_one() -> None:
    out = move_mean([1.0, np.nan, 3.0, 5.0], window=2, min_count=1)
    expected = np.array([1.0, 1.0, 3.0, 4.0])
    np.testing.assert_array_equal(out, expected)


def test_move_mean_returns_nan_when_window_all_nan() -> None:
    out = move_mean([np.nan, np.nan, 1.0], window=2)
    expected = np.array([np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# move_std
# ---------------------------------------------------------------------------


def test_move_std_population_default_ddof_zero() -> None:
    out = move_std([1.0, 2.0, 3.0, 4.0], window=2)
    expected = np.array([np.nan, 0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(out, expected)


def test_move_std_sample_ddof_one() -> None:
    out = move_std([1.0, 2.0, 3.0, 4.0], window=3, ddof=1)
    expected_value = float(np.std([1.0, 2.0, 3.0], ddof=1))
    assert math_isnan(out[0]) and math_isnan(out[1])
    assert out[2] == pytest.approx(expected_value)


def test_move_std_rejects_negative_ddof() -> None:
    with pytest.raises(RollingStatsError):
        move_std([1.0, 2.0, 3.0], window=2, ddof=-1)


# ---------------------------------------------------------------------------
# move_min / move_max
# ---------------------------------------------------------------------------


def test_move_min_basic() -> None:
    out = move_min([3.0, 1.0, 4.0, 1.0, 5.0], window=3)
    expected = np.array([np.nan, np.nan, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out, expected)


def test_move_max_basic() -> None:
    out = move_max([3.0, 1.0, 4.0, 1.0, 5.0], window=3)
    expected = np.array([np.nan, np.nan, 4.0, 4.0, 5.0])
    np.testing.assert_array_equal(out, expected)


def test_move_min_max_handle_nan_with_min_count_one() -> None:
    out_min = move_min([2.0, np.nan, 1.0], window=2, min_count=1)
    out_max = move_max([2.0, np.nan, 1.0], window=2, min_count=1)
    np.testing.assert_array_equal(out_min, [2.0, 2.0, 1.0])
    np.testing.assert_array_equal(out_max, [2.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# General behaviour: empty input, ndim guard, contiguous output
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_output() -> None:
    out = move_mean([], window=3)
    assert out.shape == (0,)


def test_rejects_non_1d_input() -> None:
    with pytest.raises(RollingStatsError):
        move_sum(np.array([[1.0, 2.0], [3.0, 4.0]]), window=2)


def test_output_dtype_is_float64() -> None:
    out = move_sum([1, 2, 3, 4], window=2, min_count=1)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism (three independent calls)
# ---------------------------------------------------------------------------


def test_inv15_move_mean_three_run_byte_identical() -> None:
    data = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    a = move_mean(data, window=3)
    b = move_mean(data, window=3)
    c = move_mean(data, window=3)
    assert a.tobytes() == b.tobytes() == c.tobytes()


def test_inv15_move_std_three_run_byte_identical() -> None:
    data = [1.0, 2.0, 4.0, 8.0, 16.0]
    a = move_std(data, window=2, ddof=1)
    b = move_std(data, window=2, ddof=1)
    c = move_std(data, window=2, ddof=1)
    assert a.tobytes() == b.tobytes() == c.tobytes()


def test_inv15_move_extrema_three_run_byte_identical() -> None:
    data = [9.0, 1.0, 8.0, 2.0, 7.0]
    assert (
        move_min(data, window=3).tobytes()
        == move_min(data, window=3).tobytes()
        == move_min(data, window=3).tobytes()
    )
    assert (
        move_max(data, window=3).tobytes()
        == move_max(data, window=3).tobytes()
        == move_max(data, window=3).tobytes()
    )


# ---------------------------------------------------------------------------
# Bottleneck equivalence — production numpy backend must produce the same
# output (within float tolerance) as bottleneck when available
# ---------------------------------------------------------------------------


def test_bottleneck_factory_matches_numpy_backend_when_installed() -> None:
    bn = pytest.importorskip("bottleneck")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    factory = enable_bottleneck_kernel_factory()
    for kind, ref_fn in (
        ("sum", move_sum),
        ("mean", move_mean),
        ("min", move_min),
        ("max", move_max),
    ):
        ours = ref_fn(data, window=3)
        theirs = factory(kind)(data, window=3)
        np.testing.assert_allclose(ours, theirs, equal_nan=True)
    ours_std = move_std(data, window=3)
    theirs_std = bn.move_std(data, window=3)
    np.testing.assert_allclose(ours_std, theirs_std, equal_nan=True)


def test_bottleneck_factory_rejects_unknown_kernel_when_installed() -> None:
    pytest.importorskip("bottleneck")
    factory = enable_bottleneck_kernel_factory()
    with pytest.raises(RollingStatsError):
        factory("variance")  # not in the table


# ---------------------------------------------------------------------------
# AST guardrails — no forbidden top-level imports + no typed-event ctors
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return inspect.getsource(rs)


def test_no_top_level_bottleneck_import() -> None:
    src = _module_source().splitlines()
    for line in src:
        if line.startswith(("import bottleneck", "from bottleneck")):
            raise AssertionError(
                "bottleneck must be a lazy seam — no top-level import: "
                f"{line!r}"
            )


def test_no_forbidden_top_level_imports() -> None:
    forbidden = (
        "import time",
        "from time",
        "import datetime",
        "from datetime",
        "import random",
        "from random",
        "import asyncio",
        "from asyncio",
        "import torch",
        "from torch",
        "import polars",
        "from polars",
        "import requests",
        "from requests",
    )
    src = _module_source().splitlines()
    for line in src:
        for bad in forbidden:
            if line.startswith(bad):
                raise AssertionError(
                    f"INV-15 / B1 — forbidden top-level import {bad!r} "
                    f"in rolling_stats: {line!r}"
                )


def test_no_typed_event_constructors_b27_b28_inv71() -> None:
    src = _module_source()
    forbidden_ctors = (
        "SignalEvent(",
        "ExecutionEvent(",
        "HazardEvent(",
        "LearningUpdate(",
        "PatchProposal(",
    )
    for ctor in forbidden_ctors:
        assert ctor not in src, f"B27/B28/INV-71 violation: {ctor} present"


def test_no_runtime_engine_imports_b1() -> None:
    src = _module_source().splitlines()
    forbidden = (
        "execution_engine",
        "intelligence_engine",
        "governance_engine",
        "system_engine",
    )
    for line in src:
        if line.startswith(("import ", "from ")):
            for tier in forbidden:
                assert tier not in line, (
                    f"B1 violation — rolling_stats must not import "
                    f"runtime tier {tier!r}: {line!r}"
                )


# ---------------------------------------------------------------------------
# Small helper — math.isnan without re-importing math in the test fixtures
# ---------------------------------------------------------------------------


def math_isnan(x: float) -> bool:
    return x != x
