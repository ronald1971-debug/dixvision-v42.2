"""C-35 — tests for the dowhy causal-reasoner surface.

Mirrors the test-shape of :mod:`tests.test_sample_factory_sandbox`
(C-34) — frozen+slotted validators, deterministic Protocol-injected
fake, end-to-end advisory record, INV-15 byte-identical replay, AST
guards.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.causal_dowhy import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_BOOTSTRAP_ROUNDS,
    MAX_CONFIDENCE_LEVEL,
    MAX_DATA_DIGEST_LEN,
    MAX_N_SAMPLES,
    MIN_BOOTSTRAP_ROUNDS,
    MIN_CONFIDENCE_LEVEL,
    MIN_N_SAMPLES,
    NEW_PIP_DEPENDENCIES,
    CausalAnalysisCallback,
    CausalAnalysisRecord,
    CausalArguments,
    CausalEstimand,
    CausalEstimateResult,
    CausalEstimatorKind,
    CausalReasonerConfigError,
    CausalRefutationResult,
    CausalRefuterKind,
    DoWhyCausalReasoner,
    dowhy_linear_regression_estimator,
    null_causal_analysis_callback,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("dowhy", "pandas", "numpy", "scipy")


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.causal_dowhy"


def test_n_samples_bounds() -> None:
    assert MIN_N_SAMPLES == 1
    assert MAX_N_SAMPLES == 10_000_000


def test_bootstrap_rounds_bounds() -> None:
    assert MIN_BOOTSTRAP_ROUNDS == 0
    assert MAX_BOOTSTRAP_ROUNDS == 1024


def test_confidence_level_bounds() -> None:
    assert MIN_CONFIDENCE_LEVEL == 0.5
    assert MAX_CONFIDENCE_LEVEL == 0.9999


def test_max_analysis_id_len_bound() -> None:
    assert MAX_ANALYSIS_ID_LEN == 256


def test_max_data_digest_len_bound() -> None:
    assert MAX_DATA_DIGEST_LEN == 64


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_estimator_kind_values_match_dowhy_methods() -> None:
    assert CausalEstimatorKind.LINEAR_REGRESSION.value == ("backdoor.linear_regression")
    assert CausalEstimatorKind.PROPENSITY_SCORE_STRATIFICATION.value == (
        "backdoor.propensity_score_stratification"
    )
    assert CausalEstimatorKind.INSTRUMENTAL_VARIABLE.value == ("iv.instrumental_variable")
    assert CausalEstimatorKind.REGRESSION_DISCONTINUITY.value == ("iv.regression_discontinuity")


def test_refuter_kind_values_match_dowhy_refuters() -> None:
    assert CausalRefuterKind.RANDOM_COMMON_CAUSE.value == ("random_common_cause")
    assert CausalRefuterKind.PLACEBO_TREATMENT_REFUTER.value == ("placebo_treatment_refuter")
    assert CausalRefuterKind.DATA_SUBSET_REFUTER.value == ("data_subset_refuter")
    assert CausalRefuterKind.BOOTSTRAP_REFUTER.value == "bootstrap_refuter"


def test_estimator_kind_count() -> None:
    assert len(list(CausalEstimatorKind)) == 4


def test_refuter_kind_count() -> None:
    assert len(list(CausalRefuterKind)) == 4


# ---------------------------------------------------------------------------
# CausalEstimand
# ---------------------------------------------------------------------------


def _valid_estimand(**overrides: Any) -> CausalEstimand:
    base: dict[str, Any] = {
        "treatment": "intervention_A",
        "outcome": "pnl_usd",
        "common_causes": ("regime_id", "session_id"),
        "data_digest": "abc1234567",
    }
    base.update(overrides)
    return CausalEstimand(**base)


def test_estimand_constructs_with_defaults() -> None:
    estimand = _valid_estimand()
    assert estimand.treatment == "intervention_A"


def test_estimand_is_frozen_and_slotted() -> None:
    estimand = _valid_estimand()
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimand.treatment = "x"  # type: ignore[misc]
    assert not hasattr(estimand, "__dict__")


def test_estimand_rejects_empty_treatment() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(treatment="")


def test_estimand_rejects_empty_outcome() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(outcome="")


def test_estimand_rejects_non_tuple_common_causes() -> None:
    with pytest.raises(TypeError):
        _valid_estimand(common_causes=["a", "b"])  # type: ignore[arg-type]


def test_estimand_rejects_empty_common_cause_entry() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(common_causes=("",))


def test_estimand_rejects_empty_data_digest() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(data_digest="")


def test_estimand_rejects_oversized_data_digest() -> None:
    with pytest.raises(ValueError):
        _valid_estimand(data_digest="x" * (MAX_DATA_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# CausalArguments
# ---------------------------------------------------------------------------


def _valid_arguments(**overrides: Any) -> CausalArguments:
    base: dict[str, Any] = {
        "estimator_kind": CausalEstimatorKind.LINEAR_REGRESSION,
        "refuters": (
            CausalRefuterKind.RANDOM_COMMON_CAUSE,
            CausalRefuterKind.PLACEBO_TREATMENT_REFUTER,
        ),
        "random_seed": 0,
        "n_samples": 1000,
        "confidence_level": 0.95,
        "bootstrap_rounds": 100,
    }
    base.update(overrides)
    return CausalArguments(**base)


def test_arguments_constructs_with_defaults() -> None:
    args = _valid_arguments()
    assert args.estimator_kind is CausalEstimatorKind.LINEAR_REGRESSION


def test_arguments_is_frozen_and_slotted() -> None:
    args = _valid_arguments()
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.random_seed = 99  # type: ignore[misc]
    assert not hasattr(args, "__dict__")


def test_arguments_rejects_non_enum_estimator_kind() -> None:
    with pytest.raises(TypeError):
        CausalArguments(  # type: ignore[arg-type]
            estimator_kind="linear",
            refuters=(),
            random_seed=0,
        )


def test_arguments_rejects_non_tuple_refuters() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(refuters=[CausalRefuterKind.RANDOM_COMMON_CAUSE])  # type: ignore[arg-type]


def test_arguments_rejects_non_enum_refuter_entry() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(refuters=("random_common_cause",))  # type: ignore[arg-type]


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


def test_arguments_rejects_below_min_confidence_level() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(confidence_level=0.1)


def test_arguments_rejects_above_max_confidence_level() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(confidence_level=1.0)


def test_arguments_rejects_nan_confidence_level() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(confidence_level=float("nan"))


def test_arguments_rejects_below_min_bootstrap_rounds() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(bootstrap_rounds=-1)


def test_arguments_rejects_above_max_bootstrap_rounds() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(bootstrap_rounds=MAX_BOOTSTRAP_ROUNDS + 1)


# ---------------------------------------------------------------------------
# CausalRefutationResult
# ---------------------------------------------------------------------------


def _valid_refutation(**overrides: Any) -> CausalRefutationResult:
    base: dict[str, Any] = {
        "refuter": CausalRefuterKind.RANDOM_COMMON_CAUSE,
        "new_estimate": 0.42,
        "p_value": 0.05,
        "passed": True,
    }
    base.update(overrides)
    return CausalRefutationResult(**base)


def test_refutation_is_frozen_and_slotted() -> None:
    r = _valid_refutation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.passed = False  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_refutation_rejects_non_enum_refuter() -> None:
    with pytest.raises(TypeError):
        CausalRefutationResult(
            refuter="rcc",  # type: ignore[arg-type]
            new_estimate=0.42,
            p_value=0.05,
            passed=True,
        )


def test_refutation_rejects_nan_new_estimate() -> None:
    with pytest.raises(ValueError):
        _valid_refutation(new_estimate=float("nan"))


def test_refutation_rejects_inf_p_value() -> None:
    with pytest.raises(ValueError):
        _valid_refutation(p_value=float("inf"))


def test_refutation_rejects_p_value_above_one() -> None:
    with pytest.raises(ValueError):
        _valid_refutation(p_value=1.5)


def test_refutation_rejects_p_value_below_zero() -> None:
    with pytest.raises(ValueError):
        _valid_refutation(p_value=-0.1)


# ---------------------------------------------------------------------------
# CausalEstimateResult
# ---------------------------------------------------------------------------


def _valid_estimate(**overrides: Any) -> CausalEstimateResult:
    base: dict[str, Any] = {
        "point_estimate": 0.42,
        "std_error": 0.05,
        "confidence_interval_lower": 0.30,
        "confidence_interval_upper": 0.55,
        "refutations": (_valid_refutation(),),
    }
    base.update(overrides)
    return CausalEstimateResult(**base)


def test_estimate_is_frozen_and_slotted() -> None:
    e = _valid_estimate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.point_estimate = 1.0  # type: ignore[misc]
    assert not hasattr(e, "__dict__")


def test_estimate_rejects_nan_point_estimate() -> None:
    with pytest.raises(ValueError):
        _valid_estimate(point_estimate=float("nan"))


def test_estimate_rejects_inf_ci_lower() -> None:
    with pytest.raises(ValueError):
        _valid_estimate(confidence_interval_lower=float("inf"))


def test_estimate_rejects_negative_std_error() -> None:
    with pytest.raises(ValueError):
        _valid_estimate(std_error=-0.1)


def test_estimate_rejects_inverted_confidence_interval() -> None:
    with pytest.raises(ValueError):
        _valid_estimate(
            confidence_interval_lower=0.9,
            confidence_interval_upper=0.1,
        )


def test_estimate_rejects_non_tuple_refutations() -> None:
    with pytest.raises(TypeError):
        _valid_estimate(refutations=[_valid_refutation()])  # type: ignore[arg-type]


def test_estimate_rejects_non_refutation_entry() -> None:
    with pytest.raises(TypeError):
        _valid_estimate(refutations=("bad",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CausalAnalysisRecord
# ---------------------------------------------------------------------------


def _valid_record(**overrides: Any) -> CausalAnalysisRecord:
    base: dict[str, Any] = {
        "ts_ns": 100,
        "analysis_id": "test_analysis",
        "source": ANALYSIS_SOURCE,
        "estimand": _valid_estimand(),
        "estimate": _valid_estimate(),
        "analysis_digest": "0123456789abcdef",
        "meta": {},
    }
    base.update(overrides)
    return CausalAnalysisRecord(**base)


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


def test_record_rejects_oversized_analysis_id() -> None:
    with pytest.raises(ValueError):
        _valid_record(analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1))


def test_record_rejects_empty_source() -> None:
    with pytest.raises(ValueError):
        _valid_record(source="")


def test_record_rejects_non_estimand() -> None:
    with pytest.raises(TypeError):
        _valid_record(estimand="not an estimand")  # type: ignore[arg-type]


def test_record_rejects_non_estimate() -> None:
    with pytest.raises(TypeError):
        _valid_record(estimate="not an estimate")  # type: ignore[arg-type]


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
    """Deterministic :class:`CausalEffectEstimator` fake — returns canned estimate."""

    __slots__ = ("_estimate",)

    def __init__(self, *, estimate: CausalEstimateResult) -> None:
        self._estimate = estimate

    def estimate(
        self,
        *,
        estimand: CausalEstimand,
        arguments: CausalArguments,
        ts_ns: int,
        callback: CausalAnalysisCallback,
    ) -> CausalEstimateResult:
        callback.on_analysis_start(ts_ns=ts_ns, estimand=estimand, arguments=arguments)
        callback.on_estimate_ready(
            ts_ns=ts_ns,
            point_estimate=self._estimate.point_estimate,
            std_error=self._estimate.std_error,
        )
        for r in self._estimate.refutations:
            callback.on_refutation(ts_ns=ts_ns, refutation=r)
        return self._estimate


# ---------------------------------------------------------------------------
# DoWhyCausalReasoner.analyse end-to-end
# ---------------------------------------------------------------------------


def _reasoner_inputs() -> tuple[CausalEstimand, CausalArguments, _FakeEstimator]:
    return (
        _valid_estimand(),
        _valid_arguments(),
        _FakeEstimator(estimate=_valid_estimate()),
    )


def test_reasoner_is_frozen_and_slotted() -> None:
    _, _, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reasoner.estimator = None  # type: ignore[misc]
    assert not hasattr(reasoner, "__dict__")


def test_reasoner_rejects_non_estimator() -> None:
    with pytest.raises(TypeError):
        DoWhyCausalReasoner(estimator="not an estimator")  # type: ignore[arg-type]


def test_analyse_emits_analysis_record() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    record = reasoner.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=12345,
        analysis_id="analysis_0001",
    )
    assert isinstance(record, CausalAnalysisRecord)
    assert record.source == ANALYSIS_SOURCE
    assert record.analysis_id == "analysis_0001"
    assert record.ts_ns == 12345


def test_analyse_meta_includes_estimator_kind_and_digest() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    record = reasoner.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["estimator_kind"] == "backdoor.linear_regression"
    assert record.meta["random_seed"] == "0"


def test_analyse_caller_meta_does_not_override_provenance() -> None:
    estimand, _, est = _reasoner_inputs()
    args = _valid_arguments(meta={"analysis_digest": "ZZZZ", "key": "v"})
    reasoner = DoWhyCausalReasoner(estimator=est)
    record = reasoner.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["key"] == "v"


def test_analyse_rejects_non_estimand() -> None:
    _, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(TypeError):
        reasoner.analyse(
            estimand="bad",  # type: ignore[arg-type]
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_non_arguments() -> None:
    estimand, _, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(TypeError):
        reasoner.analyse(
            estimand=estimand,
            arguments="bad",  # type: ignore[arg-type]
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_bool_ts_ns() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(TypeError):
        reasoner.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=True,  # type: ignore[arg-type]
            analysis_id="a",
        )


def test_analyse_rejects_negative_ts_ns() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(CausalReasonerConfigError):
        reasoner.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=-1,
            analysis_id="a",
        )


def test_analyse_rejects_empty_analysis_id() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(CausalReasonerConfigError):
        reasoner.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="",
        )


def test_analyse_rejects_oversize_analysis_id() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(CausalReasonerConfigError):
        reasoner.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyse_uses_null_callback_by_default() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    record = reasoner.analyse(
        estimand=estimand,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert isinstance(record, CausalAnalysisRecord)


def test_analyse_rejects_non_protocol_callback() -> None:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    with pytest.raises(TypeError):
        reasoner.analyse(
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
            estimand: CausalEstimand,
            arguments: CausalArguments,
            ts_ns: int,
            callback: CausalAnalysisCallback,
        ) -> CausalEstimateResult:
            return "bad"  # type: ignore[return-value]

    estimand, args, _ = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=_BadEstimator())
    with pytest.raises(TypeError):
        reasoner.analyse(
            estimand=estimand,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _run_once() -> CausalAnalysisRecord:
    estimand, args, est = _reasoner_inputs()
    reasoner = DoWhyCausalReasoner(estimator=est)
    return reasoner.analyse(
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
    assert r1.estimate == r2.estimate == r3.estimate
    assert r1.estimand == r2.estimand == r3.estimand


def test_inv15_digest_changes_when_seed_changes() -> None:
    estimand, _, _ = _reasoner_inputs()
    args_a = _valid_arguments(random_seed=0)
    args_b = _valid_arguments(random_seed=1)
    reasoner = DoWhyCausalReasoner(
        estimator=_FakeEstimator(estimate=_valid_estimate()),
    )
    r0 = reasoner.analyse(
        estimand=estimand,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = reasoner.analyse(
        estimand=estimand,
        arguments=args_b,
        ts_ns=1,
        analysis_id="a",
    )
    assert r0.analysis_digest != r1.analysis_digest


def test_inv15_digest_changes_when_estimand_changes() -> None:
    estimand_a = _valid_estimand(treatment="A")
    estimand_b = _valid_estimand(treatment="B")
    reasoner = DoWhyCausalReasoner(
        estimator=_FakeEstimator(estimate=_valid_estimate()),
    )
    r0 = reasoner.analyse(
        estimand=estimand_a,
        arguments=_valid_arguments(),
        ts_ns=1,
        analysis_id="a",
    )
    r1 = reasoner.analyse(
        estimand=estimand_b,
        arguments=_valid_arguments(),
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
# null_causal_analysis_callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_causal_analysis_callback()
    assert isinstance(cb, CausalAnalysisCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_causal_analysis_callback()
    estimand = _valid_estimand()
    args = _valid_arguments()
    estimate = _valid_estimate()
    assert cb.on_analysis_start(ts_ns=0, estimand=estimand, arguments=args) is None
    assert cb.on_estimate_ready(ts_ns=0, point_estimate=0.0, std_error=0.0) is None
    assert cb.on_refutation(ts_ns=0, refutation=_valid_refutation()) is None
    assert cb.on_analysis_end(ts_ns=0, estimate=estimate) is None


# ---------------------------------------------------------------------------
# Convenience factory raises when dowhy missing
# ---------------------------------------------------------------------------


def test_dowhy_estimator_factory_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("dowhy")
    except ImportError:
        with pytest.raises(ImportError, match="dowhy"):
            dowhy_linear_regression_estimator()
    else:
        pytest.skip("dowhy installed — production seam smoke skipped")


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "intelligence_engine" / "causal_dowhy.py"


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


def test_no_top_level_dowhy_import() -> None:
    assert all(not name.startswith("dowhy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_pandas_import() -> None:
    assert all(not name.startswith("pandas") for name in _top_level_imports(_module_ast()))


def test_no_top_level_numpy_import() -> None:
    assert all(not name.startswith("numpy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_scipy_import() -> None:
    assert all(not name.startswith("scipy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_io_imports() -> None:
    banned = {"subprocess", "socket", "urllib", "requests", "httpx", "aiohttp"}
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


def test_dowhy_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith(("dowhy", "pandas", "numpy", "scipy")):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside "
                        "dowhy_linear_regression_estimator factory"
                    )
                    assert parent.name == "dowhy_linear_regression_estimator", (
                        f"{name} imported in {parent.name!r} — must be "
                        "inside dowhy_linear_regression_estimator"
                    )


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.causal_dowhy as mod1

    importlib.reload(mod1)
    import intelligence_engine.causal_dowhy as mod2

    assert mod1.ANALYSIS_SOURCE == mod2.ANALYSIS_SOURCE
    assert mod1.MAX_N_SAMPLES == mod2.MAX_N_SAMPLES
    assert mod1.CausalEstimatorKind.LINEAR_REGRESSION is mod2.CausalEstimatorKind.LINEAR_REGRESSION
