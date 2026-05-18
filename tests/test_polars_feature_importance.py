"""Tests for ``learning_engine.analytics.feature_importance`` (S-10.3 polars).

Covers:

* Module metadata (ADAPTED-FROM header, ``NEW_PIP_DEPENDENCIES``).
* Lazy-import contract — polars is imported inside
  :func:`compute_feature_importance`, never at module top-level.
* No clock / no engine cross-imports / no global mutable state.
* Frozen+slotted dataclass validation
  (:class:`FeatureObservation`, :class:`FeatureImportance`,
  :class:`FeatureImportanceReport`).
* Functional correctness against hand-rolled Pearson reference.
* Spearman rank correlation reaches 1.0 on monotone-but-non-linear
  data while Pearson stays below 1.0.
* INV-15 byte-identical replay (3-run + permutation invariance).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import io
import math
import sys
import tokenize
from pathlib import Path

import pytest

from learning_engine.analytics import feature_importance as fi
from learning_engine.analytics.feature_importance import (
    NEW_PIP_DEPENDENCIES,
    FeatureImportance,
    FeatureImportanceReport,
    FeatureObservation,
    compute_feature_importance,
)

# Polars is required to actually exercise compute_feature_importance().
# Skip the file when polars is missing (matches S-10.1 / S-10.2 patterns).
pl = pytest.importorskip("polars")  # noqa: F841

MODULE_PATH = Path(fi.__file__)


# ----------------------------------------------------------------------
# Module metadata
# ----------------------------------------------------------------------


def test_adapted_from_header_present() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    first_lines = src.splitlines()[:6]
    joined = "\n".join(first_lines)
    assert "ADAPTED FROM:" in joined
    assert "polars" in joined.lower()


def test_new_pip_dependencies_declares_polars() -> None:
    assert NEW_PIP_DEPENDENCIES == ("polars",)


def test_module_has_no_forbidden_top_level_imports() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {
        "polars",
        "numpy",
        "pandas",
        "torch",
        "datetime",
        "time",
        "ccxt",
        "river",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden, (
                    f"top-level import of {alias.name} forbidden in OFFLINE module"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".", 1)[0]
            assert mod not in forbidden, f"top-level from-import of {node.module} forbidden"


def test_module_has_no_clock_calls() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    name_tokens = [
        t.string
        for t in tokenize.generate_tokens(io.StringIO(src).readline)
        if t.type == tokenize.NAME
    ]
    joined = " ".join(name_tokens)
    assert "datetime now" not in joined
    assert "time time" not in joined
    assert "time monotonic" not in joined
    assert "time perf_counter" not in joined


def test_module_has_no_engine_cross_imports() -> None:
    """No imports from runtime/hot-path tiers (AST-only — docstring mentions OK)."""
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden_roots, f"OFFLINE module must not import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            assert root not in forbidden_roots, f"OFFLINE module must not import from {mod}"
            if mod.startswith("intelligence_engine.meta_controller.hot_path"):
                raise AssertionError(f"OFFLINE module must not import from {mod}")


def test_module_has_no_random_or_prng() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or (node.names[0].name if node.names else "")
            mod_root = (mod or "").split(".", 1)[0]
            assert mod_root != "random", "no PRNG in deterministic OFFLINE module"


# ----------------------------------------------------------------------
# Lazy-import contract
# ----------------------------------------------------------------------


def test_polars_lazy_import_lives_inside_compute_feature_importance() -> None:
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    func_imports: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            inner = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        inner.append(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    inner.append(sub.module or "")
            func_imports[node.name] = inner

    assert "polars" in func_imports.get("compute_feature_importance", []), (
        "polars must be lazy-imported inside compute_feature_importance"
    )


def test_module_globals_do_not_leak_polars() -> None:
    assert "polars" not in vars(fi), "polars must not leak into module globals after lazy import"
    assert "pl" not in vars(fi)


def test_module_imports_without_polars_in_sys_modules() -> None:
    """Reimporting the module should not pull polars into sys.modules itself."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("polars")}
    for k in list(sys.modules):
        if k.startswith("polars"):
            del sys.modules[k]
    if "learning_engine.analytics.feature_importance" in sys.modules:
        del sys.modules["learning_engine.analytics.feature_importance"]
    try:
        mod = importlib.import_module("learning_engine.analytics.feature_importance")
        assert mod.NEW_PIP_DEPENDENCIES == ("polars",)
        assert "polars" not in sys.modules, (
            "module import pulled polars in despite lazy-import contract"
        )
    finally:
        sys.modules.update(saved)
        importlib.import_module("learning_engine.analytics.feature_importance")


# ----------------------------------------------------------------------
# FeatureObservation validation
# ----------------------------------------------------------------------


def test_feature_observation_is_frozen_and_slotted() -> None:
    assert dataclasses.is_dataclass(FeatureObservation)
    spec = dataclasses.fields(FeatureObservation)
    assert {f.name for f in spec} == {
        "ts_ns",
        "feature_name",
        "feature_value",
        "target_value",
    }
    obs = _obs(feature_value=1.0, target_value=2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.feature_value = 0.0  # type: ignore[misc]
    assert not hasattr(obs, "__dict__")  # slots=True


def test_feature_observation_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be >= 0"):
        _obs(ts_ns=-1)


def test_feature_observation_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        FeatureObservation(
            ts_ns=True,  # type: ignore[arg-type]
            feature_name="f",
            feature_value=1.0,
            target_value=2.0,
        )


def test_feature_observation_rejects_empty_feature_name() -> None:
    with pytest.raises(ValueError, match="feature_name"):
        _obs(feature_name="")


def test_feature_observation_rejects_non_str_feature_name() -> None:
    with pytest.raises(TypeError, match="feature_name must be str"):
        FeatureObservation(
            ts_ns=0,
            feature_name=123,  # type: ignore[arg-type]
            feature_value=1.0,
            target_value=2.0,
        )


def test_feature_observation_rejects_nan_values() -> None:
    nan = float("nan")
    with pytest.raises(ValueError, match="must not be NaN"):
        _obs(feature_value=nan)
    with pytest.raises(ValueError, match="must not be NaN"):
        _obs(target_value=nan)


def test_feature_observation_rejects_inf_values() -> None:
    inf = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        _obs(feature_value=inf)
    with pytest.raises(ValueError, match="must be finite"):
        _obs(target_value=-inf)


def test_feature_observation_rejects_bool_values() -> None:
    with pytest.raises(TypeError, match="feature_value must be float"):
        FeatureObservation(
            ts_ns=0,
            feature_name="f",
            feature_value=True,  # type: ignore[arg-type]
            target_value=1.0,
        )


# ----------------------------------------------------------------------
# FeatureImportance validation
# ----------------------------------------------------------------------


def test_feature_importance_rejects_pearson_out_of_unit_interval() -> None:
    with pytest.raises(ValueError, match="pearson_corr"):
        FeatureImportance(
            feature_name="f",
            n_obs=2,
            mean_feature=0.0,
            mean_target=0.0,
            pearson_corr=1.5,
            rank_corr=0.0,
            abs_score=1.5,
        )


def test_feature_importance_rejects_abs_score_out_of_unit() -> None:
    with pytest.raises(ValueError, match="abs_score"):
        FeatureImportance(
            feature_name="f",
            n_obs=2,
            mean_feature=0.0,
            mean_target=0.0,
            pearson_corr=0.5,
            rank_corr=0.5,
            abs_score=1.5,
        )


def test_feature_importance_rejects_negative_n_obs() -> None:
    with pytest.raises(ValueError, match="n_obs must be >= 0"):
        FeatureImportance(
            feature_name="f",
            n_obs=-1,
            mean_feature=0.0,
            mean_target=0.0,
            pearson_corr=0.0,
            rank_corr=0.0,
            abs_score=0.0,
        )


def test_feature_importance_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="feature_name"):
        FeatureImportance(
            feature_name="",
            n_obs=1,
            mean_feature=0.0,
            mean_target=0.0,
            pearson_corr=0.0,
            rank_corr=0.0,
            abs_score=0.0,
        )


# ----------------------------------------------------------------------
# FeatureImportanceReport validation
# ----------------------------------------------------------------------


def test_report_rejects_unsorted_by_feature() -> None:
    a = _imp("a", abs_score=0.3)
    b = _imp("b", abs_score=0.9)
    with pytest.raises(ValueError, match="sorted descending"):
        FeatureImportanceReport(
            by_feature=(a, b),
            total_n_obs=2,
            mean_abs_score=0.6,
        )


def test_report_rejects_tie_breaking_in_wrong_direction() -> None:
    a = _imp("z", abs_score=0.5)
    b = _imp("a", abs_score=0.5)
    with pytest.raises(ValueError, match="ties .* feature_name ascending"):
        FeatureImportanceReport(
            by_feature=(a, b),
            total_n_obs=2,
            mean_abs_score=0.5,
        )


def test_report_rejects_duplicate_feature_names() -> None:
    a = _imp("dup", abs_score=0.5)
    with pytest.raises(ValueError, match="unique"):
        FeatureImportanceReport(
            by_feature=(a, a),
            total_n_obs=2,
            mean_abs_score=0.5,
        )


def test_report_rejects_non_tuple_by_feature() -> None:
    a = _imp("a", abs_score=0.5)
    with pytest.raises(TypeError, match="by_feature must be tuple"):
        FeatureImportanceReport(
            by_feature=[a],  # type: ignore[arg-type]
            total_n_obs=1,
            mean_abs_score=0.5,
        )


def test_report_rejects_non_feature_importance_entries() -> None:
    with pytest.raises(TypeError, match="FeatureImportance"):
        FeatureImportanceReport(
            by_feature=("oops",),  # type: ignore[arg-type]
            total_n_obs=0,
            mean_abs_score=0.0,
        )


def test_report_rejects_mean_abs_score_out_of_unit() -> None:
    with pytest.raises(ValueError, match="mean_abs_score"):
        FeatureImportanceReport(
            by_feature=(),
            total_n_obs=0,
            mean_abs_score=2.0,
        )


# ----------------------------------------------------------------------
# Functional aggregation
# ----------------------------------------------------------------------


def test_compute_feature_importance_empty_input() -> None:
    report = compute_feature_importance([])
    assert report.by_feature == ()
    assert report.total_n_obs == 0
    assert report.mean_abs_score == 0.0


def test_compute_feature_importance_perfect_positive_correlation() -> None:
    # y = 2x + 1 => Pearson = 1.0, Spearman = 1.0
    obs = [_obs(ts_ns=i, feature_value=float(i), target_value=2.0 * i + 1.0) for i in range(1, 11)]
    report = compute_feature_importance(obs)
    assert len(report.by_feature) == 1
    f = report.by_feature[0]
    assert f.n_obs == 10
    assert f.pearson_corr == pytest.approx(1.0, abs=1e-9)
    assert f.rank_corr == pytest.approx(1.0, abs=1e-9)
    assert f.abs_score == pytest.approx(1.0, abs=1e-9)


def test_compute_feature_importance_perfect_negative_correlation() -> None:
    obs = [_obs(ts_ns=i, feature_value=float(i), target_value=-3.0 * i) for i in range(1, 9)]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    assert f.pearson_corr == pytest.approx(-1.0, abs=1e-9)
    assert f.rank_corr == pytest.approx(-1.0, abs=1e-9)
    assert f.abs_score == pytest.approx(1.0, abs=1e-9)


def test_compute_feature_importance_zero_correlation_zero_variance_target() -> None:
    """Zero-variance target ⇒ corr is NaN; module clamps to 0.0."""
    obs = [_obs(ts_ns=i, feature_value=float(i), target_value=5.0) for i in range(5)]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    assert f.pearson_corr == 0.0
    assert f.rank_corr == 0.0
    assert f.abs_score == 0.0


def test_compute_feature_importance_zero_variance_feature() -> None:
    obs = [_obs(ts_ns=i, feature_value=7.0, target_value=float(i)) for i in range(5)]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    assert f.pearson_corr == 0.0
    assert f.rank_corr == 0.0
    assert f.abs_score == 0.0


def test_compute_feature_importance_single_observation() -> None:
    """Single sample ⇒ corr undefined ⇒ 0.0."""
    obs = [_obs(feature_value=1.0, target_value=2.0)]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    assert f.n_obs == 1
    assert f.pearson_corr == 0.0
    assert f.rank_corr == 0.0
    assert f.abs_score == 0.0


def test_compute_feature_importance_spearman_beats_pearson_on_monotone_nonlinear() -> None:
    """y = x^3 over positive x: Spearman == 1.0 but Pearson < 1.0."""
    obs = [_obs(ts_ns=i, feature_value=float(i), target_value=float(i) ** 3) for i in range(1, 11)]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    assert f.rank_corr == pytest.approx(1.0, abs=1e-9)
    assert f.pearson_corr < 1.0
    assert f.pearson_corr > 0.9
    assert f.abs_score == pytest.approx(1.0, abs=1e-9)


def test_compute_feature_importance_groups_features_independently() -> None:
    obs = []
    # Feature A: perfectly positive
    obs += [
        _obs(ts_ns=i, feature_name="a", feature_value=float(i), target_value=float(i) * 2.0)
        for i in range(1, 6)
    ]
    # Feature B: perfectly negative
    obs += [
        _obs(ts_ns=i, feature_name="b", feature_value=float(i), target_value=-float(i))
        for i in range(1, 6)
    ]
    # Feature C: zero variance feature
    obs += [
        _obs(ts_ns=i, feature_name="c", feature_value=1.0, target_value=float(i)) for i in range(5)
    ]
    report = compute_feature_importance(obs)
    assert len(report.by_feature) == 3
    by_name = {f.feature_name: f for f in report.by_feature}
    assert by_name["a"].abs_score == pytest.approx(1.0, abs=1e-9)
    assert by_name["b"].abs_score == pytest.approx(1.0, abs=1e-9)
    assert by_name["c"].abs_score == 0.0


def test_compute_feature_importance_sorted_descending_by_abs_score() -> None:
    """Output order: abs_score DESC, feature_name ASC tiebreak."""
    obs = []
    # weak: a (Pearson ≈ 0.0)
    obs += [
        _obs(
            ts_ns=i, feature_name="weak", feature_value=float(i), target_value=float(i % 2) * 0.0001
        )
        for i in range(1, 6)
    ]
    # strong: y = x
    obs += [
        _obs(ts_ns=i, feature_name="strong", feature_value=float(i), target_value=float(i))
        for i in range(1, 6)
    ]
    # medium: y = x but with tiny noise
    obs += [
        _obs(
            ts_ns=i,
            feature_name="medium",
            feature_value=float(i),
            target_value=float(i) + (0.5 if i == 3 else 0.0),
        )
        for i in range(1, 6)
    ]
    report = compute_feature_importance(obs)
    scores = [f.abs_score for f in report.by_feature]
    assert scores == sorted(scores, reverse=True)


def test_compute_feature_importance_tie_break_alphabetical() -> None:
    """When two features get identical abs_score, name asc breaks the tie."""
    obs = []
    for name in ("zeta", "alpha", "mu"):
        obs += [
            _obs(ts_ns=i, feature_name=name, feature_value=float(i), target_value=float(i))
            for i in range(1, 5)
        ]
    report = compute_feature_importance(obs)
    names = [f.feature_name for f in report.by_feature]
    # All have abs_score == 1.0; tiebreak by name ascending.
    assert names == ["alpha", "mu", "zeta"]


def test_compute_feature_importance_n_obs_and_means() -> None:
    obs = [_obs(ts_ns=i, feature_value=float(i), target_value=float(i + 5)) for i in range(1, 6)]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    assert f.n_obs == 5
    assert f.mean_feature == pytest.approx(3.0)
    assert f.mean_target == pytest.approx(8.0)
    assert report.total_n_obs == 5


def test_compute_feature_importance_pearson_matches_reference() -> None:
    """Hand-rolled Pearson on a small, known sample."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 5.0, 4.0, 5.0]
    obs = [
        _obs(ts_ns=i, feature_value=x, target_value=y)
        for i, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]
    report = compute_feature_importance(obs)
    f = report.by_feature[0]
    expected = _pearson(xs, ys)
    assert f.pearson_corr == pytest.approx(expected, abs=1e-9)


def test_compute_feature_importance_rejects_non_observation_input() -> None:
    with pytest.raises(TypeError, match="FeatureObservation"):
        compute_feature_importance(["not an obs"])  # type: ignore[list-item]


# ----------------------------------------------------------------------
# INV-15 byte-identical replay
# ----------------------------------------------------------------------


def test_replay_byte_stable_across_three_runs() -> None:
    obs = [
        _obs(
            ts_ns=i,
            feature_name=f"f{i % 4}",
            feature_value=float((i * 7) % 13),
            target_value=float(((i * 11) % 17) - 5),
        )
        for i in range(50)
    ]
    a = compute_feature_importance(obs)
    b = compute_feature_importance(obs)
    c = compute_feature_importance(obs)
    assert a == b == c


def test_replay_permutation_invariant() -> None:
    obs = [
        _obs(
            ts_ns=i * 1_000_000,
            feature_name=f"feat_{i % 3}",
            feature_value=float(((i * 13) % 19) - 9),
            target_value=float(((i * 17) % 23) - 11),
        )
        for i in range(40)
    ]
    a = compute_feature_importance(obs)
    b = compute_feature_importance(list(reversed(obs)))
    assert a == b


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _obs(
    *,
    ts_ns: int = 0,
    feature_name: str = "f",
    feature_value: float = 0.0,
    target_value: float = 0.0,
) -> FeatureObservation:
    return FeatureObservation(
        ts_ns=ts_ns,
        feature_name=feature_name,
        feature_value=feature_value,
        target_value=target_value,
    )


def _imp(name: str, *, abs_score: float) -> FeatureImportance:
    return FeatureImportance(
        feature_name=name,
        n_obs=1,
        mean_feature=0.0,
        mean_target=0.0,
        pearson_corr=abs_score,
        rank_corr=abs_score,
        abs_score=abs_score,
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y)
