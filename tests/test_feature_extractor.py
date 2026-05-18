"""B-06 — tsfresh canonical-adaptation feature-extractor tests.

Covers:

* AST authority pins — no banned imports, no typed bus event ctors,
  ADAPTED FROM headers present.
* Value-object validation — :class:`TimeSeries`, :class:`FeatureSpec`,
  :class:`FeatureVector` bounds + frozen semantics.
* Happy path — every calculator computes the expected closed-form
  value on small fixtures.
* Quantile correctness — linear-interpolation quantile matches the
  textbook formula on a 5-point series.
* 3-run determinism — same input → byte-identical digest.
* Sorted-key independence — same values inserted in different orders
  → same digest.
* Spec/registry symmetry — every preset name maps to a registered
  calculator; every quantile spec resolves.
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning_engine.trader_abstraction.extractor import (  # noqa: E402
    EFFICIENT_FEATURE_NAMES,
    FEATURE_CALCULATORS,
    MINIMAL_FEATURE_NAMES,
    NEW_PIP_DEPENDENCIES,
    FeatureExtractionError,
    FeatureSet,
    FeatureSpec,
    FeatureVector,
    TimeSeries,
    calculate_feature,
    extract_features,
)

SOURCE_PATH = (REPO_ROOT / "learning_engine" / "trader_abstraction" / "extractor.py").resolve()
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")
SOURCE_TREE = ast.parse(SOURCE_TEXT)


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            out.add(node.module.split(".")[0])
    return out


def test_no_banned_top_level_imports() -> None:
    banned = {
        "tsfresh",
        "pandas",
        "polars",
        "numpy",
        "scipy",
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "websockets",
        "langsmith",
        "requests",
        "httpx",
    }
    found = _imported_modules(SOURCE_TREE)
    assert not (banned & found), f"banned imports: {banned & found}"


def test_no_typed_event_ctors() -> None:
    """B27 / B28 / INV-71 authority symmetry: no typed bus events."""

    forbidden_names = {
        "PatchProposal",
        "SignalEvent",
        "GovernanceDecision",
        "SystemEvent",
        "ExecutionIntent",
        "FillEvent",
    }
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in forbidden_names:
                pytest.fail(f"forbidden ctor: {target.id}")
            if isinstance(target, ast.Attribute) and target.attr in forbidden_names:
                pytest.fail(f"forbidden ctor: {target.attr}")


def test_no_engine_cross_imports() -> None:
    """B1 engine isolation."""

    forbidden = {
        "governance_engine",
        "system_engine",
        "execution_engine",
        "evolution_engine",
        "intelligence_engine",
    }
    for node in ast.walk(SOURCE_TREE):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            assert top not in forbidden, f"forbidden engine import: {node.module}"


def test_adapted_from_header_present() -> None:
    assert "# ADAPTED FROM: blue-yonder/tsfresh" in SOURCE_TEXT


def test_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_no_top_level_io_side_effects() -> None:
    """Module body must be pure: only imports / constants / defs / classes."""

    for node in SOURCE_TREE.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,  # docstrings
            ),
        ):
            continue
        if isinstance(node, ast.If) and (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            continue
        pytest.fail(f"unexpected top-level node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# TimeSeries validation
# ---------------------------------------------------------------------------


def _series(values: tuple[float, ...], *, name: str = "close", ts_ns: int = 100) -> TimeSeries:
    return TimeSeries(name=name, values=values, ts_ns=ts_ns)


def test_time_series_rejects_empty_values() -> None:
    with pytest.raises(FeatureExtractionError):
        TimeSeries(name="close", values=(), ts_ns=0)


def test_time_series_rejects_empty_name() -> None:
    with pytest.raises(FeatureExtractionError):
        TimeSeries(name="", values=(1.0,), ts_ns=0)


def test_time_series_rejects_nan() -> None:
    with pytest.raises(FeatureExtractionError):
        TimeSeries(name="close", values=(1.0, math.nan), ts_ns=0)


def test_time_series_rejects_inf() -> None:
    with pytest.raises(FeatureExtractionError):
        TimeSeries(name="close", values=(1.0, math.inf), ts_ns=0)


def test_time_series_rejects_bool_value() -> None:
    with pytest.raises(FeatureExtractionError):
        TimeSeries(name="close", values=(True,), ts_ns=0)  # type: ignore[arg-type]


def test_time_series_rejects_negative_ts() -> None:
    with pytest.raises(FeatureExtractionError):
        TimeSeries(name="close", values=(1.0,), ts_ns=-1)


def test_time_series_accepts_int_values_coerced_to_float() -> None:
    ts = TimeSeries(name="close", values=(1, 2, 3), ts_ns=0)  # type: ignore[arg-type]
    assert ts.values == (1.0, 2.0, 3.0)
    assert all(isinstance(v, float) for v in ts.values)


def test_time_series_frozen() -> None:
    ts = _series((1.0, 2.0, 3.0))
    with pytest.raises((AttributeError, FrozenInstanceError)):
        ts.values = (4.0,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureSpec validation
# ---------------------------------------------------------------------------


def test_feature_spec_default_minimal() -> None:
    spec = FeatureSpec(series_name="close")
    assert spec.feature_set is FeatureSet.MINIMAL
    assert spec.quantile_levels == ()


def test_feature_spec_rejects_quantile_out_of_range() -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureSpec(series_name="close", quantile_levels=(0.0,))
    with pytest.raises(FeatureExtractionError):
        FeatureSpec(series_name="close", quantile_levels=(1.0,))
    with pytest.raises(FeatureExtractionError):
        FeatureSpec(series_name="close", quantile_levels=(1.5,))


def test_feature_spec_sorts_quantile_levels() -> None:
    spec = FeatureSpec(series_name="close", quantile_levels=(0.9, 0.1, 0.5))
    assert spec.quantile_levels == (0.1, 0.5, 0.9)


def test_feature_spec_rejects_empty_name() -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureSpec(series_name="")


# ---------------------------------------------------------------------------
# Calculator correctness — closed-form fixtures
# ---------------------------------------------------------------------------


# A 5-point series with mean = 3.0, variance = 2.0, std = sqrt(2).
_FIVE = _series((1.0, 2.0, 3.0, 4.0, 5.0))


def test_length() -> None:
    assert calculate_feature("length", _FIVE) == 5.0


def test_sum_values() -> None:
    assert calculate_feature("sum_values", _FIVE) == 15.0


def test_mean() -> None:
    assert calculate_feature("mean", _FIVE) == 3.0


def test_median_odd_length() -> None:
    assert calculate_feature("median", _FIVE) == 3.0


def test_median_even_length() -> None:
    ts = _series((1.0, 2.0, 3.0, 4.0))
    assert calculate_feature("median", ts) == 2.5


def test_maximum_and_minimum() -> None:
    assert calculate_feature("maximum", _FIVE) == 5.0
    assert calculate_feature("minimum", _FIVE) == 1.0


def test_absolute_maximum() -> None:
    ts = _series((-7.0, 2.0, 3.0))
    assert calculate_feature("absolute_maximum", ts) == 7.0


def test_variance_population() -> None:
    assert calculate_feature("variance", _FIVE) == pytest.approx(2.0)


def test_standard_deviation_population() -> None:
    assert calculate_feature("standard_deviation", _FIVE) == pytest.approx(math.sqrt(2.0))


def test_root_mean_square() -> None:
    # rms = sqrt((1+4+9+16+25)/5) = sqrt(11)
    assert calculate_feature("root_mean_square", _FIVE) == pytest.approx(math.sqrt(11.0))


def test_abs_energy() -> None:
    assert calculate_feature("abs_energy", _FIVE) == 1.0 + 4.0 + 9.0 + 16.0 + 25.0


def test_mean_change() -> None:
    assert calculate_feature("mean_change", _FIVE) == 1.0


def test_mean_change_requires_two_points() -> None:
    with pytest.raises(FeatureExtractionError):
        calculate_feature("mean_change", _series((1.0,)))


def test_mean_abs_change() -> None:
    assert calculate_feature("mean_abs_change", _FIVE) == 1.0


def test_count_above_and_below_mean() -> None:
    # mean=3; strictly above: 4, 5 → 2; strictly below: 1, 2 → 2
    assert calculate_feature("count_above_mean", _FIVE) == 2.0
    assert calculate_feature("count_below_mean", _FIVE) == 2.0


def test_longest_strike_above_mean() -> None:
    ts = _series((1.0, 5.0, 6.0, 7.0, 2.0, 8.0, 9.0))
    # mean = 38/7 ≈ 5.43; above-mean run lengths: [6,7]=2, [8,9]=2 → 2
    assert calculate_feature("longest_strike_above_mean", ts) == 2.0


def test_longest_strike_below_mean() -> None:
    ts = _series((1.0, 2.0, 3.0, 9.0, 1.0, 2.0))
    # mean = 18/6 = 3.0; below-mean run lengths: [1,2]=2, [1,2]=2 → 2
    assert calculate_feature("longest_strike_below_mean", ts) == 2.0


def test_first_location_of_maximum() -> None:
    ts = _series((1.0, 5.0, 3.0, 5.0))
    # first index 1 of 4 → 1/4 = 0.25
    assert calculate_feature("first_location_of_maximum", ts) == 0.25


def test_last_location_of_maximum() -> None:
    ts = _series((1.0, 5.0, 3.0, 5.0))
    # last index 3 of 4 → (3+1)/4 = 1.0
    assert calculate_feature("last_location_of_maximum", ts) == 1.0


def test_first_location_of_minimum() -> None:
    ts = _series((3.0, 1.0, 4.0, 1.0))
    assert calculate_feature("first_location_of_minimum", ts) == 0.25


def test_last_location_of_minimum() -> None:
    ts = _series((3.0, 1.0, 4.0, 1.0))
    assert calculate_feature("last_location_of_minimum", ts) == 1.0


def test_skewness_symmetric_is_zero() -> None:
    assert calculate_feature("skewness", _FIVE) == pytest.approx(0.0)


def test_skewness_constant_input_is_zero() -> None:
    ts = _series((3.0, 3.0, 3.0))
    assert calculate_feature("skewness", ts) == 0.0


def test_kurtosis_constant_input_is_minus_three() -> None:
    ts = _series((3.0, 3.0, 3.0))
    assert calculate_feature("kurtosis", ts) == -3.0


def test_kurtosis_symmetric_finite() -> None:
    # Excess kurtosis of a 5-point linear ramp is -1.3 (matches numpy).
    assert calculate_feature("kurtosis", _FIVE) == pytest.approx(-1.3)


def test_quantile_p50_matches_median() -> None:
    spec = FeatureSpec(series_name="close", quantile_levels=(0.5,))
    fv = extract_features(_FIVE, spec)
    assert fv.values[f"quantile_{repr(0.5)}"] == 3.0


def test_quantile_p25_linear_interpolation() -> None:
    spec = FeatureSpec(series_name="close", quantile_levels=(0.25,))
    fv = extract_features(_FIVE, spec)
    # pos = 0.25 * 4 = 1.0 → exact index 1 → value 2.0
    assert fv.values[f"quantile_{repr(0.25)}"] == 2.0


def test_quantile_p10_linear_interpolation() -> None:
    spec = FeatureSpec(series_name="close", quantile_levels=(0.1,))
    fv = extract_features(_FIVE, spec)
    # pos = 0.1 * 4 = 0.4 → between 1.0 and 2.0 with frac 0.4 → 1.4
    assert fv.values[f"quantile_{repr(0.1)}"] == pytest.approx(1.4)


def test_quantile_rejects_out_of_range_via_calculate() -> None:
    with pytest.raises(FeatureExtractionError):
        calculate_feature("quantile_0.0", _FIVE)


def test_calculate_unknown_feature_rejected() -> None:
    with pytest.raises(FeatureExtractionError):
        calculate_feature("does_not_exist", _FIVE)


def test_calculate_unparseable_quantile_rejected() -> None:
    with pytest.raises(FeatureExtractionError):
        calculate_feature("quantile_abc", _FIVE)


def test_calculate_rejects_non_time_series() -> None:
    with pytest.raises(FeatureExtractionError):
        calculate_feature("mean", [1.0, 2.0, 3.0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Preset / registry symmetry
# ---------------------------------------------------------------------------


def test_minimal_preset_all_registered() -> None:
    for name in MINIMAL_FEATURE_NAMES:
        assert name in FEATURE_CALCULATORS, name


def test_efficient_preset_all_registered() -> None:
    for name in EFFICIENT_FEATURE_NAMES:
        assert name in FEATURE_CALCULATORS, name


def test_efficient_superset_of_minimal() -> None:
    assert set(MINIMAL_FEATURE_NAMES).issubset(set(EFFICIENT_FEATURE_NAMES))


def test_minimal_preset_sorted() -> None:
    assert list(MINIMAL_FEATURE_NAMES) == sorted(MINIMAL_FEATURE_NAMES)


def test_efficient_preset_sorted() -> None:
    assert list(EFFICIENT_FEATURE_NAMES) == sorted(EFFICIENT_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# extract_features happy path
# ---------------------------------------------------------------------------


def test_extract_minimal_features_returns_all_preset() -> None:
    spec = FeatureSpec(series_name="close")
    fv = extract_features(_FIVE, spec)
    assert set(fv.values) == set(MINIMAL_FEATURE_NAMES)


def test_extract_efficient_features_returns_all_preset() -> None:
    spec = FeatureSpec(series_name="close", feature_set=FeatureSet.EFFICIENT)
    fv = extract_features(_FIVE, spec)
    assert set(fv.values) == set(EFFICIENT_FEATURE_NAMES)


def test_extract_features_with_quantiles() -> None:
    spec = FeatureSpec(series_name="close", quantile_levels=(0.25, 0.75))
    fv = extract_features(_FIVE, spec)
    expected = set(MINIMAL_FEATURE_NAMES) | {
        f"quantile_{repr(0.25)}",
        f"quantile_{repr(0.75)}",
    }
    assert set(fv.values) == expected


def test_extract_features_forwards_ts_ns() -> None:
    spec = FeatureSpec(series_name="close")
    fv = extract_features(_series((1.0, 2.0, 3.0), ts_ns=12345), spec)
    assert fv.ts_ns == 12345


def test_extract_features_rejects_spec_mismatch() -> None:
    spec = FeatureSpec(series_name="volume")
    with pytest.raises(FeatureExtractionError):
        extract_features(_FIVE, spec)


def test_extract_features_rejects_non_time_series() -> None:
    spec = FeatureSpec(series_name="close")
    with pytest.raises(FeatureExtractionError):
        extract_features([1.0, 2.0], spec)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FeatureVector validation
# ---------------------------------------------------------------------------


def test_feature_vector_empty_values_rejected() -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureVector(
            series_name="close",
            feature_set=FeatureSet.MINIMAL,
            values={},
            ts_ns=0,
        )


def test_feature_vector_nan_value_rejected() -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureVector(
            series_name="close",
            feature_set=FeatureSet.MINIMAL,
            values={"mean": math.nan},
            ts_ns=0,
        )


def test_feature_vector_keys_sorted_after_init() -> None:
    fv = FeatureVector(
        series_name="close",
        feature_set=FeatureSet.MINIMAL,
        values={"b": 2.0, "a": 1.0, "c": 3.0},
        ts_ns=0,
    )
    assert list(fv.values) == ["a", "b", "c"]


def test_feature_vector_invalid_digest_rejected() -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureVector(
            series_name="close",
            feature_set=FeatureSet.MINIMAL,
            values={"mean": 1.0},
            ts_ns=0,
            digest="not-hex",
        )


def test_feature_vector_supplied_digest_accepted() -> None:
    # Round-trip: extract, take digest, rebuild with that explicit digest.
    spec = FeatureSpec(series_name="close")
    fv1 = extract_features(_FIVE, spec)
    fv2 = FeatureVector(
        series_name=fv1.series_name,
        feature_set=fv1.feature_set,
        values=dict(fv1.values),
        ts_ns=fv1.ts_ns,
        digest=fv1.digest,
    )
    assert fv2.digest == fv1.digest


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def test_digest_3_run_identical() -> None:
    spec = FeatureSpec(
        series_name="close",
        feature_set=FeatureSet.EFFICIENT,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    digests = {extract_features(_FIVE, spec).digest for _ in range(3)}
    assert len(digests) == 1


def test_digest_dict_order_independence() -> None:
    """Building the vector twice with values inserted in different orders
    must produce the same digest."""

    values_a = {"a": 1.0, "b": 2.0, "c": 3.0}
    values_b = {"c": 3.0, "a": 1.0, "b": 2.0}
    fv_a = FeatureVector(
        series_name="close",
        feature_set=FeatureSet.MINIMAL,
        values=values_a,
        ts_ns=0,
    )
    fv_b = FeatureVector(
        series_name="close",
        feature_set=FeatureSet.MINIMAL,
        values=values_b,
        ts_ns=0,
    )
    assert fv_a.digest == fv_b.digest


def test_digest_sensitive_to_value_change() -> None:
    spec = FeatureSpec(series_name="close")
    base = extract_features(_FIVE, spec).digest
    perturbed = extract_features(_series((1.0, 2.0, 3.0, 4.0, 5.5)), spec).digest
    assert base != perturbed


def test_digest_sensitive_to_ts_ns() -> None:
    spec = FeatureSpec(series_name="close")
    a = extract_features(_series((1.0, 2.0, 3.0), ts_ns=100), spec).digest
    b = extract_features(_series((1.0, 2.0, 3.0), ts_ns=200), spec).digest
    assert a != b


def test_digest_sensitive_to_feature_set() -> None:
    spec_min = FeatureSpec(series_name="close", feature_set=FeatureSet.MINIMAL)
    spec_eff = FeatureSpec(series_name="close", feature_set=FeatureSet.EFFICIENT)
    a = extract_features(_FIVE, spec_min).digest
    b = extract_features(_FIVE, spec_eff).digest
    assert a != b


def test_digest_hex_format() -> None:
    spec = FeatureSpec(series_name="close")
    fv = extract_features(_FIVE, spec)
    assert len(fv.digest) == 32
    int(fv.digest, 16)  # must parse as hex


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_extract_supports_single_point_minimal_preset() -> None:
    # Minimal preset excludes mean_change / skewness; should succeed.
    spec = FeatureSpec(series_name="close")
    fv = extract_features(_series((42.0,)), spec)
    assert fv.values["length"] == 1.0
    assert fv.values["mean"] == 42.0
    assert fv.values["variance"] == 0.0
    assert fv.values["standard_deviation"] == 0.0


def test_extract_single_point_efficient_preset_raises() -> None:
    # Efficient preset includes mean_change → requires len >= 2.
    spec = FeatureSpec(series_name="close", feature_set=FeatureSet.EFFICIENT)
    with pytest.raises(FeatureExtractionError):
        extract_features(_series((42.0,)), spec)
