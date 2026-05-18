"""C-37 — tests for the econml HTE analyser surface."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.hte_econml import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_CONFIDENCE_LEVEL,
    MAX_DATA_DIGEST_LEN,
    MAX_N_SAMPLES,
    MAX_POINTS,
    MIN_CONFIDENCE_LEVEL,
    MIN_N_SAMPLES,
    NEW_PIP_DEPENDENCIES,
    EconMLHteAnalyser,
    HteAnalyserConfigError,
    HteAnalysisCallback,
    HteAnalysisRecord,
    HteAnalysisResult,
    HteArguments,
    HteEffectPoint,
    HteEstimand,
    HteEstimatorKind,
    econml_dml_estimator,
    null_hte_analysis_callback,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == (
        "econml",
        "pandas",
        "numpy",
        "scikit-learn",
    )


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.hte_econml"


def test_bounds_constants() -> None:
    assert MIN_N_SAMPLES == 1
    assert MAX_N_SAMPLES == 10_000_000
    assert MAX_POINTS == 4096
    assert MIN_CONFIDENCE_LEVEL == 0.5
    assert MAX_CONFIDENCE_LEVEL == 0.9999
    assert MAX_ANALYSIS_ID_LEN == 256
    assert MAX_DATA_DIGEST_LEN == 64


# ---------------------------------------------------------------------------
# HteEstimatorKind
# ---------------------------------------------------------------------------


def test_estimator_kind_values() -> None:
    assert HteEstimatorKind.DML.value == "DML"
    assert HteEstimatorKind.SPARSE_LINEAR_DML.value == "SparseLinearDML"
    assert HteEstimatorKind.CAUSAL_FOREST_DML.value == "CausalForestDML"
    assert HteEstimatorKind.DR_LEARNER.value == "DRLearner"
    assert HteEstimatorKind.ORTHO_FOREST.value == "DMLOrthoForest"
    assert HteEstimatorKind.META_LEARNER_S.value == "SLearner"
    assert HteEstimatorKind.META_LEARNER_T.value == "TLearner"
    assert HteEstimatorKind.META_LEARNER_X.value == "XLearner"
    assert HteEstimatorKind.META_LEARNER_DR.value == ("DRLearner.MetaLearner")
    assert HteEstimatorKind.DEEP_IV.value == "DeepIV"


def test_estimator_kind_count() -> None:
    assert len(list(HteEstimatorKind)) == 10


# ---------------------------------------------------------------------------
# HteEstimand
# ---------------------------------------------------------------------------


def _valid_estimand(**overrides: Any) -> HteEstimand:
    base: dict[str, Any] = {
        "features": ("regime_id", "session_id"),
        "treatment": "strategy_arm",
        "outcome": "pnl_usd",
        "effect_modifiers": ("trader_archetype",),
        "data_digest": "abc1234567",
    }
    base.update(overrides)
    return HteEstimand(**base)


def test_estimand_constructs_with_defaults() -> None:
    e = _valid_estimand()
    assert e.outcome == "pnl_usd"


def test_estimand_is_frozen_and_slotted() -> None:
    e = _valid_estimand()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.treatment = "x"  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


def test_estimand_rejects_non_tuple_features() -> None:
    with pytest.raises(TypeError):
        _valid_estimand(features=["a", "b"])  # type: ignore[arg-type]


def test_estimand_rejects_empty_features() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(features=())


def test_estimand_rejects_empty_feature_entry() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(features=("",))


def test_estimand_rejects_empty_treatment() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(treatment="")


def test_estimand_rejects_empty_outcome() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(outcome="")


def test_estimand_rejects_non_tuple_effect_modifiers() -> None:
    with pytest.raises(TypeError):
        _valid_estimand(effect_modifiers=["a"])  # type: ignore[arg-type]


def test_estimand_accepts_empty_effect_modifiers() -> None:
    e = _valid_estimand(effect_modifiers=())
    assert e.effect_modifiers == ()


def test_estimand_rejects_empty_modifier_entry() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(effect_modifiers=("",))


def test_estimand_rejects_empty_data_digest() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(data_digest="")


def test_estimand_rejects_oversize_data_digest() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(data_digest="x" * (MAX_DATA_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# HteArguments
# ---------------------------------------------------------------------------


def _valid_arguments(**overrides: Any) -> HteArguments:
    base: dict[str, Any] = {
        "estimator_kind": HteEstimatorKind.DML,
        "random_seed": 0,
        "n_samples": 1000,
        "confidence_level": 0.95,
        "n_points": 100,
    }
    base.update(overrides)
    return HteArguments(**base)


def test_arguments_is_frozen_and_slotted() -> None:
    a = _valid_arguments()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.random_seed = 99  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_arguments_rejects_non_enum_estimator_kind() -> None:
    with pytest.raises(TypeError):
        HteArguments(  # type: ignore[arg-type]
            estimator_kind="DML",
            random_seed=0,
        )


def test_arguments_rejects_bool_random_seed() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(random_seed=True)


def test_arguments_rejects_negative_random_seed() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(random_seed=-1)


def test_arguments_rejects_below_min_n_samples() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_samples=0)


def test_arguments_rejects_above_max_n_samples() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_samples=MAX_N_SAMPLES + 1)


def test_arguments_rejects_nan_confidence_level() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(confidence_level=float("nan"))


def test_arguments_rejects_low_confidence_level() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(confidence_level=0.4)


def test_arguments_rejects_high_confidence_level() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(confidence_level=1.0)


def test_arguments_rejects_below_one_n_points() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_points=0)


def test_arguments_rejects_above_max_n_points() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_points=MAX_POINTS + 1)


# ---------------------------------------------------------------------------
# HteEffectPoint
# ---------------------------------------------------------------------------


def _valid_point(**overrides: Any) -> HteEffectPoint:
    base: dict[str, Any] = {
        "point_id": 0,
        "point_estimate": 0.5,
        "ci_lower": 0.3,
        "ci_upper": 0.7,
        "std_error": 0.05,
    }
    base.update(overrides)
    return HteEffectPoint(**base)


def test_point_is_frozen_and_slotted() -> None:
    p = _valid_point()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.point_estimate = 0.0  # type: ignore[misc]
    assert not hasattr(p, "__dict__")


def test_point_rejects_bool_point_id() -> None:
    with pytest.raises(TypeError):
        _valid_point(point_id=True)


def test_point_rejects_negative_point_id() -> None:
    with pytest.raises(ValueError):
        _valid_point(point_id=-1)


def test_point_rejects_nan_point_estimate() -> None:
    with pytest.raises(ValueError):
        _valid_point(point_estimate=float("nan"))


def test_point_rejects_inf_ci_lower() -> None:
    with pytest.raises(ValueError):
        _valid_point(ci_lower=float("inf"))


def test_point_rejects_inf_ci_upper() -> None:
    with pytest.raises(ValueError):
        _valid_point(ci_upper=float("inf"))


def test_point_rejects_ci_lower_above_upper() -> None:
    with pytest.raises(ValueError):
        _valid_point(ci_lower=0.8, ci_upper=0.7)


def test_point_rejects_inf_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_point(std_error=float("inf"))


def test_point_rejects_negative_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_point(std_error=-0.1)


# ---------------------------------------------------------------------------
# HteAnalysisResult
# ---------------------------------------------------------------------------


def _valid_result(**overrides: Any) -> HteAnalysisResult:
    base: dict[str, Any] = {
        "average_treatment_effect": 0.42,
        "ate_std_error": 0.06,
        "points": (_valid_point(),),
    }
    base.update(overrides)
    return HteAnalysisResult(**base)


def test_result_is_frozen_and_slotted() -> None:
    r = _valid_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.average_treatment_effect = 0.0  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_nan_ate() -> None:
    with pytest.raises(ValueError):
        _valid_result(average_treatment_effect=float("nan"))


def test_result_rejects_inf_ate_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_result(ate_std_error=float("inf"))


def test_result_rejects_negative_ate_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_result(ate_std_error=-0.1)


def test_result_rejects_non_tuple_points() -> None:
    with pytest.raises(TypeError):
        _valid_result(points=[_valid_point()])  # type: ignore[arg-type]


def test_result_rejects_non_point_entry() -> None:
    with pytest.raises(TypeError):
        _valid_result(points=("bad",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HteAnalysisRecord
# ---------------------------------------------------------------------------


def _valid_record(**overrides: Any) -> HteAnalysisRecord:
    base: dict[str, Any] = {
        "ts_ns": 100,
        "analysis_id": "test_analysis",
        "source": ANALYSIS_SOURCE,
        "estimand": _valid_estimand(),
        "result": _valid_result(),
        "analysis_digest": "0123456789abcdef",
        "meta": {},
    }
    base.update(overrides)
    return HteAnalysisRecord(**base)


def test_record_is_frozen_and_slotted() -> None:
    r = _valid_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.ts_ns = 1  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_record_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError):
        _valid_record(ts_ns=True)


def test_record_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError):
        _valid_record(ts_ns=-1)


def test_record_rejects_empty_analysis_id() -> None:
    with pytest.raises(ValueError):
        _valid_record(analysis_id="")


def test_record_rejects_oversize_analysis_id() -> None:
    with pytest.raises(ValueError):
        _valid_record(analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1))


def test_record_rejects_empty_source() -> None:
    with pytest.raises(ValueError):
        _valid_record(source="")


def test_record_rejects_non_estimand() -> None:
    with pytest.raises(TypeError):
        _valid_record(estimand="bad")  # type: ignore[arg-type]


def test_record_rejects_non_result() -> None:
    with pytest.raises(TypeError):
        _valid_record(result="bad")  # type: ignore[arg-type]


def test_record_rejects_wrong_digest_length() -> None:
    with pytest.raises(ValueError):
        _valid_record(analysis_digest="short")


def test_record_rejects_non_hex_digest() -> None:
    with pytest.raises(ValueError):
        _valid_record(analysis_digest="ZZZZZZZZZZZZZZZZ")


# ---------------------------------------------------------------------------
# Deterministic fake estimator
# ---------------------------------------------------------------------------


class _FakeEstimator:
    """Deterministic :class:`HteEffectEstimator` fake."""

    __slots__ = ("_result",)

    def __init__(self, *, result: HteAnalysisResult) -> None:
        self._result = result

    def estimate(
        self,
        *,
        estimand: HteEstimand,
        arguments: HteArguments,
        ts_ns: int,
        callback: HteAnalysisCallback,
    ) -> HteAnalysisResult:
        callback.on_analysis_start(ts_ns=ts_ns, estimand=estimand, arguments=arguments)
        for p in self._result.points:
            callback.on_point_ready(ts_ns=ts_ns, point=p)
        return self._result


# ---------------------------------------------------------------------------
# EconMLHteAnalyser.analyse end-to-end
# ---------------------------------------------------------------------------


def _analyser_inputs() -> tuple[HteEstimand, HteArguments, _FakeEstimator]:
    return (
        _valid_estimand(),
        _valid_arguments(),
        _FakeEstimator(result=_valid_result()),
    )


def test_analyser_is_frozen_and_slotted() -> None:
    _, _, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(dataclasses.FrozenInstanceError):
        analyser.estimator = None  # type: ignore[misc]
    assert not hasattr(analyser, "__dict__")


def test_analyser_rejects_non_estimator() -> None:
    with pytest.raises(TypeError):
        EconMLHteAnalyser(estimator="bad")  # type: ignore[arg-type]


def test_analyse_emits_record() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=12345,
        analysis_id="analysis_0001",
    )
    assert isinstance(record, HteAnalysisRecord)
    assert record.source == ANALYSIS_SOURCE
    assert record.analysis_id == "analysis_0001"
    assert record.ts_ns == 12345


def test_analyse_meta_includes_provenance() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["estimator_kind"] == "DML"
    assert record.meta["random_seed"] == "0"
    assert record.meta["point_count"] == str(len(record.result.points))


def test_analyse_caller_meta_does_not_override_provenance() -> None:
    estimand, _, estimator = _analyser_inputs()
    args = _valid_arguments(meta={"analysis_digest": "ZZZZ", "k": "v"})
    analyser = EconMLHteAnalyser(estimator=estimator)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["k"] == "v"


def test_analyse_rejects_non_estimand() -> None:
    _, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand="bad",  # type: ignore[arg-type]
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_non_arguments() -> None:
    estimand, _, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments="bad",  # type: ignore[arg-type]
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_bool_ts_ns() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=True,  # type: ignore[arg-type]
            analysis_id="a",
        )


def test_analyse_rejects_negative_ts_ns() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(HteAnalyserConfigError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=-1,
            analysis_id="a",
        )


def test_analyse_rejects_empty_analysis_id() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(HteAnalyserConfigError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="",
        )


def test_analyse_rejects_oversize_analysis_id() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(HteAnalyserConfigError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyse_uses_null_callback_by_default() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert isinstance(record, HteAnalysisRecord)


def test_analyse_rejects_non_protocol_callback() -> None:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
            callback="bad",  # type: ignore[arg-type]
        )


def test_analyse_rejects_estimator_returning_wrong_type() -> None:
    class _BadEstimator:
        def estimate(
            self,
            *,
            estimand: HteEstimand,
            arguments: HteArguments,
            ts_ns: int,
            callback: HteAnalysisCallback,
        ) -> HteAnalysisResult:
            return "bad"  # type: ignore[return-value]

    estimand, args, _ = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=_BadEstimator())
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _run_once() -> HteAnalysisRecord:
    estimand, args, estimator = _analyser_inputs()
    analyser = EconMLHteAnalyser(estimator=estimator)
    return analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=42,
        analysis_id="canonical_analysis",
    )


def test_inv15_three_run_byte_identical_replay() -> None:
    r1 = _run_once()
    r2 = _run_once()
    r3 = _run_once()
    assert r1.analysis_digest == r2.analysis_digest == r3.analysis_digest
    assert r1.result == r2.result == r3.result
    assert r1.estimand == r2.estimand == r3.estimand


def test_inv15_digest_changes_when_seed_changes() -> None:
    estimand = _valid_estimand()
    args_a = _valid_arguments(random_seed=0)
    args_b = _valid_arguments(random_seed=1)
    analyser = EconMLHteAnalyser(
        estimator=_FakeEstimator(result=_valid_result()),
    )
    r0 = analyser.analyse(
        estimand=estimand,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = analyser.analyse(
        estimand=estimand,
        arguments=args_b,
        ts_ns=1,
        analysis_id="a",
    )
    assert r0.analysis_digest != r1.analysis_digest


def test_inv15_digest_changes_when_estimator_kind_changes() -> None:
    estimand = _valid_estimand()
    args_a = _valid_arguments(estimator_kind=HteEstimatorKind.DML)
    args_b = _valid_arguments(estimator_kind=HteEstimatorKind.CAUSAL_FOREST_DML)
    analyser = EconMLHteAnalyser(
        estimator=_FakeEstimator(result=_valid_result()),
    )
    r0 = analyser.analyse(
        estimand=estimand,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = analyser.analyse(
        estimand=estimand,
        arguments=args_b,
        ts_ns=1,
        analysis_id="a",
    )
    assert r0.analysis_digest != r1.analysis_digest


def test_inv15_digest_is_blake2b_16_hex() -> None:
    r = _run_once()
    assert len(r.analysis_digest) == 16
    assert re.fullmatch(r"[0-9a-f]{16}", r.analysis_digest)
    h = hashlib.blake2b(b"smoke", digest_size=8).hexdigest()
    assert len(h) == 16


# ---------------------------------------------------------------------------
# null_hte_analysis_callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_hte_analysis_callback()
    assert isinstance(cb, HteAnalysisCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_hte_analysis_callback()
    estimand = _valid_estimand()
    args = _valid_arguments()
    result = _valid_result()
    assert cb.on_analysis_start(ts_ns=0, estimand=estimand, arguments=args) is None
    assert cb.on_point_ready(ts_ns=0, point=_valid_point()) is None
    assert cb.on_analysis_end(ts_ns=0, result=result) is None


# ---------------------------------------------------------------------------
# Convenience factory raises when econml missing
# ---------------------------------------------------------------------------


def test_econml_estimator_factory_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("econml")
    except ImportError:
        with pytest.raises(ImportError, match="econml"):
            econml_dml_estimator()
    else:
        pytest.skip("econml installed — production seam smoke skipped")


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "intelligence_engine" / "hte_econml.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_econml_import() -> None:
    assert all(not name.startswith("econml") for name in _top_level_imports(_module_ast()))


def test_no_top_level_pandas_import() -> None:
    assert all(not name.startswith("pandas") for name in _top_level_imports(_module_ast()))


def test_no_top_level_numpy_import() -> None:
    assert all(not name.startswith("numpy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_sklearn_import() -> None:
    assert all(not name.startswith("sklearn") for name in _top_level_imports(_module_ast()))


def test_no_top_level_io_imports() -> None:
    banned = {
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
    }
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_engine_cross_imports_at_top_level() -> None:
    banned_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "registry.",
        "ui.",
    )
    for name in _top_level_imports(_module_ast()):
        for prefix in banned_prefixes:
            assert not name.startswith(prefix), name


def test_no_engine_cross_imports_in_code() -> None:
    tree = _module_ast()
    code_only_segments: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Name)):
            code_only_segments.append(ast.dump(node))
    blob = "\n".join(code_only_segments)
    for needle in (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "registry",
    ):
        assert needle not in blob, needle


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


def test_econml_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith(("econml", "pandas", "numpy", "sklearn")):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside econml_dml_estimator factory"
                    )
                    assert parent.name == "econml_dml_estimator", (
                        f"{name} imported in {parent.name!r} — must be inside econml_dml_estimator"
                    )


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.hte_econml as mod1

    importlib.reload(mod1)
    import intelligence_engine.hte_econml as mod2

    assert mod1.ANALYSIS_SOURCE == mod2.ANALYSIS_SOURCE
    assert mod1.MAX_N_SAMPLES == mod2.MAX_N_SAMPLES
    assert mod1.HteEstimatorKind.DML is mod2.HteEstimatorKind.DML
