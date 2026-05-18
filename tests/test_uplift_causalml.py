"""C-36 — tests for the causalml uplift-analyser surface."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.uplift_causalml import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_DATA_DIGEST_LEN,
    MAX_N_SAMPLES,
    MAX_N_TREATMENT,
    MAX_SEGMENTS,
    MIN_N_SAMPLES,
    MIN_N_TREATMENT,
    NEW_PIP_DEPENDENCIES,
    CausalMLUpliftAnalyser,
    UpliftAnalyserConfigError,
    UpliftAnalysisCallback,
    UpliftAnalysisRecord,
    UpliftAnalysisResult,
    UpliftArguments,
    UpliftEstimand,
    UpliftLearnerKind,
    UpliftSegmentResult,
    causalml_s_learner_estimator,
    null_uplift_analysis_callback,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == (
        "causalml",
        "pandas",
        "numpy",
        "scikit-learn",
    )


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.uplift_causalml"


def test_bounds_constants() -> None:
    assert MIN_N_SAMPLES == 1
    assert MAX_N_SAMPLES == 10_000_000
    assert MIN_N_TREATMENT == 1
    assert MAX_N_TREATMENT == 10
    assert MAX_SEGMENTS == 1024
    assert MAX_ANALYSIS_ID_LEN == 256
    assert MAX_DATA_DIGEST_LEN == 64


# ---------------------------------------------------------------------------
# UpliftLearnerKind
# ---------------------------------------------------------------------------


def test_learner_kind_values() -> None:
    assert UpliftLearnerKind.S_LEARNER.value == "BaseSRegressor"
    assert UpliftLearnerKind.T_LEARNER.value == "BaseTRegressor"
    assert UpliftLearnerKind.X_LEARNER.value == "BaseXRegressor"
    assert UpliftLearnerKind.R_LEARNER.value == "BaseRRegressor"
    assert UpliftLearnerKind.CAUSAL_FOREST.value == ("CausalRandomForestRegressor")


def test_learner_kind_count() -> None:
    assert len(list(UpliftLearnerKind)) == 5


# ---------------------------------------------------------------------------
# UpliftEstimand
# ---------------------------------------------------------------------------


def _valid_estimand(**overrides: Any) -> UpliftEstimand:
    base: dict[str, Any] = {
        "features": ("regime_id", "session_id", "trader_archetype"),
        "treatment": "strategy_arm",
        "outcome": "pnl_usd",
        "n_treatment_arms": 2,
        "data_digest": "abc1234567",
    }
    base.update(overrides)
    return UpliftEstimand(**base)


def test_estimand_constructs_with_defaults() -> None:
    estimand = _valid_estimand()
    assert estimand.outcome == "pnl_usd"


def test_estimand_is_frozen_and_slotted() -> None:
    estimand = _valid_estimand()
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimand.treatment = "x"  # type: ignore[misc]
    assert not hasattr(estimand, "__dict__")


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


def test_estimand_rejects_bool_n_treatment_arms() -> None:
    with pytest.raises(TypeError):
        _valid_estimand(n_treatment_arms=True)


def test_estimand_rejects_below_min_n_treatment_arms() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(n_treatment_arms=0)


def test_estimand_rejects_above_max_n_treatment_arms() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(n_treatment_arms=MAX_N_TREATMENT + 1)


def test_estimand_rejects_empty_data_digest() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(data_digest="")


def test_estimand_rejects_oversize_data_digest() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(data_digest="x" * (MAX_DATA_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# UpliftArguments
# ---------------------------------------------------------------------------


def _valid_arguments(**overrides: Any) -> UpliftArguments:
    base: dict[str, Any] = {
        "learner_kind": UpliftLearnerKind.S_LEARNER,
        "random_seed": 0,
        "n_samples": 1000,
        "n_segments": 10,
    }
    base.update(overrides)
    return UpliftArguments(**base)


def test_arguments_is_frozen_and_slotted() -> None:
    args = _valid_arguments()
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.random_seed = 99  # type: ignore[misc]
    assert not hasattr(args, "__dict__")


def test_arguments_rejects_non_enum_learner_kind() -> None:
    with pytest.raises(TypeError):
        UpliftArguments(  # type: ignore[arg-type]
            learner_kind="S",
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


def test_arguments_rejects_below_one_n_segments() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_segments=0)


def test_arguments_rejects_above_max_n_segments() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_segments=MAX_SEGMENTS + 1)


# ---------------------------------------------------------------------------
# UpliftSegmentResult
# ---------------------------------------------------------------------------


def _valid_segment(**overrides: Any) -> UpliftSegmentResult:
    base: dict[str, Any] = {
        "segment_id": 0,
        "segment_size": 100,
        "segment_ate": 0.42,
        "segment_p_value": 0.03,
    }
    base.update(overrides)
    return UpliftSegmentResult(**base)


def test_segment_is_frozen_and_slotted() -> None:
    s = _valid_segment()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.segment_ate = 1.0  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_segment_rejects_bool_segment_id() -> None:
    with pytest.raises(TypeError):
        _valid_segment(segment_id=True)


def test_segment_rejects_negative_segment_id() -> None:
    with pytest.raises(ValueError):
        _valid_segment(segment_id=-1)


def test_segment_rejects_bool_segment_size() -> None:
    with pytest.raises(TypeError):
        _valid_segment(segment_size=True)


def test_segment_rejects_negative_segment_size() -> None:
    with pytest.raises(ValueError):
        _valid_segment(segment_size=-1)


def test_segment_rejects_nan_segment_ate() -> None:
    with pytest.raises(ValueError):
        _valid_segment(segment_ate=float("nan"))


def test_segment_rejects_inf_p_value() -> None:
    with pytest.raises(ValueError):
        _valid_segment(segment_p_value=float("inf"))


def test_segment_rejects_p_value_above_one() -> None:
    with pytest.raises(ValueError):
        _valid_segment(segment_p_value=1.5)


def test_segment_rejects_p_value_below_zero() -> None:
    with pytest.raises(ValueError):
        _valid_segment(segment_p_value=-0.1)


# ---------------------------------------------------------------------------
# UpliftAnalysisResult
# ---------------------------------------------------------------------------


def _valid_result(**overrides: Any) -> UpliftAnalysisResult:
    base: dict[str, Any] = {
        "overall_ate": 0.35,
        "overall_std_error": 0.04,
        "segments": (_valid_segment(),),
    }
    base.update(overrides)
    return UpliftAnalysisResult(**base)


def test_result_is_frozen_and_slotted() -> None:
    r = _valid_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.overall_ate = 0.0  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_nan_overall_ate() -> None:
    with pytest.raises(ValueError):
        _valid_result(overall_ate=float("nan"))


def test_result_rejects_inf_overall_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_result(overall_std_error=float("inf"))


def test_result_rejects_negative_overall_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_result(overall_std_error=-0.1)


def test_result_rejects_non_tuple_segments() -> None:
    with pytest.raises(TypeError):
        _valid_result(segments=[_valid_segment()])  # type: ignore[arg-type]


def test_result_rejects_non_segment_entry() -> None:
    with pytest.raises(TypeError):
        _valid_result(segments=("bad",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UpliftAnalysisRecord
# ---------------------------------------------------------------------------


def _valid_record(**overrides: Any) -> UpliftAnalysisRecord:
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
    return UpliftAnalysisRecord(**base)


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
# Deterministic fake learner
# ---------------------------------------------------------------------------


class _FakeLearner:
    """Deterministic :class:`UpliftLearner` fake — returns canned result."""

    __slots__ = ("_result",)

    def __init__(self, *, result: UpliftAnalysisResult) -> None:
        self._result = result

    def estimate(
        self,
        *,
        estimand: UpliftEstimand,
        arguments: UpliftArguments,
        ts_ns: int,
        callback: UpliftAnalysisCallback,
    ) -> UpliftAnalysisResult:
        callback.on_analysis_start(ts_ns=ts_ns, estimand=estimand, arguments=arguments)
        for s in self._result.segments:
            callback.on_segment_ready(ts_ns=ts_ns, segment=s)
        return self._result


# ---------------------------------------------------------------------------
# CausalMLUpliftAnalyser.analyse end-to-end
# ---------------------------------------------------------------------------


def _analyser_inputs() -> tuple[UpliftEstimand, UpliftArguments, _FakeLearner]:
    return (
        _valid_estimand(),
        _valid_arguments(),
        _FakeLearner(result=_valid_result()),
    )


def test_analyser_is_frozen_and_slotted() -> None:
    _, _, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(dataclasses.FrozenInstanceError):
        analyser.learner = None  # type: ignore[misc]
    assert not hasattr(analyser, "__dict__")


def test_analyser_rejects_non_learner() -> None:
    with pytest.raises(TypeError):
        CausalMLUpliftAnalyser(learner="bad")  # type: ignore[arg-type]


def test_analyse_emits_record() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=12345,
        analysis_id="analysis_0001",
    )
    assert isinstance(record, UpliftAnalysisRecord)
    assert record.source == ANALYSIS_SOURCE
    assert record.analysis_id == "analysis_0001"
    assert record.ts_ns == 12345


def test_analyse_meta_includes_provenance() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["learner_kind"] == "BaseSRegressor"
    assert record.meta["random_seed"] == "0"
    assert record.meta["segment_count"] == str(len(record.result.segments))


def test_analyse_caller_meta_does_not_override_provenance() -> None:
    estimand, _, learner = _analyser_inputs()
    args = _valid_arguments(meta={"analysis_digest": "ZZZZ", "k": "v"})
    analyser = CausalMLUpliftAnalyser(learner=learner)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["k"] == "v"


def test_analyse_rejects_non_estimand() -> None:
    _, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand="bad",  # type: ignore[arg-type]
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_non_arguments() -> None:
    estimand, _, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments="bad",  # type: ignore[arg-type]
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_bool_ts_ns() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=True,  # type: ignore[arg-type]
            analysis_id="a",
        )


def test_analyse_rejects_negative_ts_ns() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(UpliftAnalyserConfigError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=-1,
            analysis_id="a",
        )


def test_analyse_rejects_empty_analysis_id() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(UpliftAnalyserConfigError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="",
        )


def test_analyse_rejects_oversize_analysis_id() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(UpliftAnalyserConfigError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyse_uses_null_callback_by_default() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    record = analyser.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert isinstance(record, UpliftAnalysisRecord)


def test_analyse_rejects_non_protocol_callback() -> None:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
    with pytest.raises(TypeError):
        analyser.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
            callback="bad",  # type: ignore[arg-type]
        )


def test_analyse_rejects_learner_returning_wrong_type() -> None:
    class _BadLearner:
        def estimate(
            self,
            *,
            estimand: UpliftEstimand,
            arguments: UpliftArguments,
            ts_ns: int,
            callback: UpliftAnalysisCallback,
        ) -> UpliftAnalysisResult:
            return "bad"  # type: ignore[return-value]

    estimand, args, _ = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=_BadLearner())
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


def _run_once() -> UpliftAnalysisRecord:
    estimand, args, learner = _analyser_inputs()
    analyser = CausalMLUpliftAnalyser(learner=learner)
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
    analyser = CausalMLUpliftAnalyser(
        learner=_FakeLearner(result=_valid_result()),
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


def test_inv15_digest_changes_when_learner_kind_changes() -> None:
    estimand = _valid_estimand()
    args_a = _valid_arguments(learner_kind=UpliftLearnerKind.S_LEARNER)
    args_b = _valid_arguments(learner_kind=UpliftLearnerKind.T_LEARNER)
    analyser = CausalMLUpliftAnalyser(
        learner=_FakeLearner(result=_valid_result()),
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
# null_uplift_analysis_callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_uplift_analysis_callback()
    assert isinstance(cb, UpliftAnalysisCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_uplift_analysis_callback()
    estimand = _valid_estimand()
    args = _valid_arguments()
    result = _valid_result()
    assert cb.on_analysis_start(ts_ns=0, estimand=estimand, arguments=args) is None
    assert cb.on_segment_ready(ts_ns=0, segment=_valid_segment()) is None
    assert cb.on_analysis_end(ts_ns=0, result=result) is None


# ---------------------------------------------------------------------------
# Convenience factory raises when causalml missing
# ---------------------------------------------------------------------------


def test_causalml_estimator_factory_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("causalml")
    except ImportError:
        with pytest.raises(ImportError, match="causalml"):
            causalml_s_learner_estimator()
    else:
        pytest.skip("causalml installed — production seam smoke skipped")


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "intelligence_engine" / "uplift_causalml.py"


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


def test_no_top_level_causalml_import() -> None:
    assert all(not name.startswith("causalml") for name in _top_level_imports(_module_ast()))


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


def test_causalml_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith(("causalml", "pandas", "numpy", "sklearn")):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside "
                        "causalml_s_learner_estimator factory"
                    )
                    assert parent.name == "causalml_s_learner_estimator", (
                        f"{name} imported in {parent.name!r} — must be "
                        "inside causalml_s_learner_estimator"
                    )


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.uplift_causalml as mod1

    importlib.reload(mod1)
    import intelligence_engine.uplift_causalml as mod2

    assert mod1.ANALYSIS_SOURCE == mod2.ANALYSIS_SOURCE
    assert mod1.MAX_N_SAMPLES == mod2.MAX_N_SAMPLES
    assert mod1.UpliftLearnerKind.S_LEARNER is mod2.UpliftLearnerKind.S_LEARNER
