"""C-42 — tests for the arviz post-inference diagnostics analyser."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.diag_arviz import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_MODEL_DIGEST_LEN,
    MAX_NUM_CHAINS,
    MAX_NUM_DRAWS,
    MAX_NUM_VARS,
    MAX_SAMPLE_LEN,
    MAX_VAR_NAME_LEN,
    MIN_NUM_CHAINS,
    MIN_NUM_DRAWS,
    MIN_NUM_VARS,
    MIN_SAMPLE_LEN,
    NEW_PIP_DEPENDENCIES,
    ArviZAnalyserConfigError,
    ArviZDiagnosticAnalyser,
    ArviZDiagnosticArguments,
    ArviZDiagnosticCallback,
    ArviZDiagnosticEngine,
    ArviZDiagnosticKind,
    ArviZDiagnosticRecord,
    ArviZDiagnosticResult,
    ArviZPosteriorSpec,
    ArviZVariableSummary,
    arviz_diagnostic_engine,
    null_arviz_diagnostic_callback,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("arviz", "numpy", "xarray")


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.diag_arviz"


def test_bounds_constants() -> None:
    assert MIN_NUM_VARS == 1
    assert MAX_NUM_VARS == 1024
    assert MIN_NUM_CHAINS == 1
    assert MAX_NUM_CHAINS == 64
    assert MIN_NUM_DRAWS == 1
    assert MAX_NUM_DRAWS == 100_000
    assert MIN_SAMPLE_LEN == 0
    assert MAX_SAMPLE_LEN == 1_000_000
    assert MAX_ANALYSIS_ID_LEN == 256
    assert MAX_MODEL_DIGEST_LEN == 64
    assert MAX_VAR_NAME_LEN == 128


# ---------------------------------------------------------------------------
# ArviZDiagnosticKind
# ---------------------------------------------------------------------------


def test_diagnostic_kind_values() -> None:
    assert ArviZDiagnosticKind.SUMMARY.value == "summary"
    assert ArviZDiagnosticKind.RHAT.value == "rhat"
    assert ArviZDiagnosticKind.ESS.value == "ess"
    assert ArviZDiagnosticKind.MCSE.value == "mcse"


def test_diagnostic_kind_count() -> None:
    assert len(list(ArviZDiagnosticKind)) == 4


# ---------------------------------------------------------------------------
# ArviZPosteriorSpec
# ---------------------------------------------------------------------------


def _valid_spec(**overrides: Any) -> ArviZPosteriorSpec:
    base: dict[str, Any] = {
        "num_chains": 4,
        "num_draws": 500,
        "num_vars": 2,
        "model_digest": "model_abcdef",
    }
    base.update(overrides)
    return ArviZPosteriorSpec(**base)


def test_spec_constructs_with_defaults() -> None:
    s = _valid_spec()
    assert s.num_chains == 4
    assert s.num_draws == 500
    assert s.num_vars == 2
    assert s.model_digest == "model_abcdef"


def test_spec_is_frozen_and_slotted() -> None:
    s = _valid_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.num_chains = 8  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_spec_rejects_num_chains_below_min() -> None:
    with pytest.raises(ValueError, match="num_chains must be >="):
        _valid_spec(num_chains=0)


def test_spec_rejects_num_chains_above_max() -> None:
    with pytest.raises(ValueError, match="num_chains must be <="):
        _valid_spec(num_chains=MAX_NUM_CHAINS + 1)


def test_spec_rejects_num_chains_non_int() -> None:
    with pytest.raises(TypeError, match="num_chains must be int"):
        _valid_spec(num_chains=4.0)  # type: ignore[arg-type]


def test_spec_rejects_num_chains_bool() -> None:
    with pytest.raises(TypeError, match="num_chains must be int"):
        _valid_spec(num_chains=True)  # type: ignore[arg-type]


def test_spec_rejects_num_draws_below_min() -> None:
    with pytest.raises(ValueError, match="num_draws must be >="):
        _valid_spec(num_draws=0)


def test_spec_rejects_num_draws_above_max() -> None:
    with pytest.raises(ValueError, match="num_draws must be <="):
        _valid_spec(num_draws=MAX_NUM_DRAWS + 1)


def test_spec_rejects_num_draws_non_int() -> None:
    with pytest.raises(TypeError, match="num_draws must be int"):
        _valid_spec(num_draws=500.0)  # type: ignore[arg-type]


def test_spec_rejects_num_draws_bool() -> None:
    with pytest.raises(TypeError, match="num_draws must be int"):
        _valid_spec(num_draws=False)  # type: ignore[arg-type]


def test_spec_rejects_num_vars_below_min() -> None:
    with pytest.raises(ValueError, match="num_vars must be >="):
        _valid_spec(num_vars=0)


def test_spec_rejects_num_vars_above_max() -> None:
    with pytest.raises(ValueError, match="num_vars must be <="):
        _valid_spec(num_vars=MAX_NUM_VARS + 1)


def test_spec_rejects_num_vars_non_int() -> None:
    with pytest.raises(TypeError, match="num_vars must be int"):
        _valid_spec(num_vars=2.0)  # type: ignore[arg-type]


def test_spec_rejects_num_vars_bool() -> None:
    with pytest.raises(TypeError, match="num_vars must be int"):
        _valid_spec(num_vars=True)  # type: ignore[arg-type]


def test_spec_rejects_empty_model_digest() -> None:
    with pytest.raises(ValueError, match="model_digest must be non-empty"):
        _valid_spec(model_digest="")


def test_spec_rejects_long_model_digest() -> None:
    with pytest.raises(ValueError, match="model_digest must be <="):
        _valid_spec(model_digest="x" * (MAX_MODEL_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# ArviZDiagnosticArguments
# ---------------------------------------------------------------------------


def _valid_args(**overrides: Any) -> ArviZDiagnosticArguments:
    base: dict[str, Any] = {
        "diagnostic_kind": ArviZDiagnosticKind.SUMMARY,
        "random_seed": 42,
        "hdi_prob": 0.94,
        "samples": (1.0, 2.0, 3.0, 4.0),
        "meta": {},
    }
    base.update(overrides)
    return ArviZDiagnosticArguments(**base)


def test_args_constructs_with_defaults() -> None:
    a = _valid_args()
    assert a.diagnostic_kind is ArviZDiagnosticKind.SUMMARY
    assert a.random_seed == 42
    assert a.hdi_prob == 0.94
    assert a.samples == (1.0, 2.0, 3.0, 4.0)
    assert a.meta == {}


def test_args_is_frozen_and_slotted() -> None:
    a = _valid_args()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.random_seed = 999  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_args_rejects_non_enum_diagnostic_kind() -> None:
    with pytest.raises(TypeError, match="diagnostic_kind must be ArviZDiagnosticKind"):
        _valid_args(diagnostic_kind="summary")  # type: ignore[arg-type]


def test_args_rejects_negative_random_seed() -> None:
    with pytest.raises(ValueError, match="random_seed must be non-negative"):
        _valid_args(random_seed=-1)


def test_args_rejects_random_seed_non_int() -> None:
    with pytest.raises(TypeError, match="random_seed must be int"):
        _valid_args(random_seed=1.5)  # type: ignore[arg-type]


def test_args_rejects_random_seed_bool() -> None:
    with pytest.raises(TypeError, match="random_seed must be int"):
        _valid_args(random_seed=True)  # type: ignore[arg-type]


def test_args_rejects_non_float_hdi_prob() -> None:
    with pytest.raises(TypeError, match="hdi_prob must be float"):
        _valid_args(hdi_prob="x")  # type: ignore[arg-type]


def test_args_rejects_bool_hdi_prob() -> None:
    with pytest.raises(TypeError, match="hdi_prob must be float"):
        _valid_args(hdi_prob=True)  # type: ignore[arg-type]


def test_args_rejects_nan_hdi_prob() -> None:
    with pytest.raises(ValueError, match="hdi_prob must be finite"):
        _valid_args(hdi_prob=float("nan"))


def test_args_rejects_inf_hdi_prob() -> None:
    with pytest.raises(ValueError, match="hdi_prob must be finite"):
        _valid_args(hdi_prob=float("inf"))


def test_args_rejects_zero_hdi_prob() -> None:
    with pytest.raises(ValueError, match=r"hdi_prob must be in \(0.0, 1.0\)"):
        _valid_args(hdi_prob=0.0)


def test_args_rejects_one_hdi_prob() -> None:
    with pytest.raises(ValueError, match=r"hdi_prob must be in \(0.0, 1.0\)"):
        _valid_args(hdi_prob=1.0)


def test_args_rejects_negative_hdi_prob() -> None:
    with pytest.raises(ValueError, match=r"hdi_prob must be in \(0.0, 1.0\)"):
        _valid_args(hdi_prob=-0.1)


def test_args_rejects_above_one_hdi_prob() -> None:
    with pytest.raises(ValueError, match=r"hdi_prob must be in \(0.0, 1.0\)"):
        _valid_args(hdi_prob=1.1)


def test_args_rejects_samples_non_tuple() -> None:
    with pytest.raises(TypeError, match="samples must be a tuple"):
        _valid_args(samples=[1.0, 2.0])  # type: ignore[arg-type]


def test_args_accepts_empty_samples() -> None:
    a = _valid_args(samples=())
    assert a.samples == ()


def test_args_rejects_samples_above_max() -> None:
    too_many = tuple([1.0] * (MAX_SAMPLE_LEN + 1))
    with pytest.raises(ValueError, match="samples must have <="):
        _valid_args(samples=too_many)


def test_args_rejects_non_float_sample() -> None:
    with pytest.raises(TypeError, match="samples values must be float"):
        _valid_args(samples=(1.0, "x"))  # type: ignore[arg-type]


def test_args_rejects_bool_sample() -> None:
    with pytest.raises(TypeError, match="samples values must be float"):
        _valid_args(samples=(1.0, True))  # type: ignore[arg-type]


def test_args_rejects_nan_sample() -> None:
    with pytest.raises(ValueError, match="samples values must be finite"):
        _valid_args(samples=(1.0, float("nan")))


def test_args_rejects_inf_sample() -> None:
    with pytest.raises(ValueError, match="samples values must be finite"):
        _valid_args(samples=(1.0, float("inf")))


def test_args_rejects_empty_meta_key() -> None:
    with pytest.raises(ValueError, match="meta keys must be non-empty"):
        _valid_args(meta={"": "v"})


def test_args_rejects_non_str_meta_key() -> None:
    with pytest.raises(ValueError, match="meta keys must be non-empty"):
        _valid_args(meta={1: "v"})  # type: ignore[dict-item]


def test_args_rejects_empty_meta_value() -> None:
    with pytest.raises(ValueError, match="meta values must be non-empty"):
        _valid_args(meta={"k": ""})


def test_args_rejects_non_str_meta_value() -> None:
    with pytest.raises(ValueError, match="meta values must be non-empty"):
        _valid_args(meta={"k": 1})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# ArviZVariableSummary
# ---------------------------------------------------------------------------


def _valid_summary(**overrides: Any) -> ArviZVariableSummary:
    base: dict[str, Any] = {
        "name": "mu",
        "mean": 0.5,
        "sd": 0.1,
        "hdi_3": 0.3,
        "hdi_97": 0.7,
        "ess_bulk": 200.0,
        "ess_tail": 180.0,
        "r_hat": 1.01,
    }
    base.update(overrides)
    return ArviZVariableSummary(**base)


def test_summary_constructs_with_defaults() -> None:
    s = _valid_summary()
    assert s.name == "mu"
    assert s.mean == 0.5
    assert s.sd == 0.1
    assert s.hdi_3 == 0.3
    assert s.hdi_97 == 0.7
    assert s.ess_bulk == 200.0
    assert s.ess_tail == 180.0
    assert s.r_hat == 1.01


def test_summary_is_frozen_and_slotted() -> None:
    s = _valid_summary()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.mean = 1.0  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_summary_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        _valid_summary(name="")


def test_summary_rejects_long_name() -> None:
    with pytest.raises(ValueError, match="name must be <="):
        _valid_summary(name="x" * (MAX_VAR_NAME_LEN + 1))


def test_summary_rejects_non_str_name() -> None:
    with pytest.raises(TypeError, match="name must be str"):
        _valid_summary(name=1)  # type: ignore[arg-type]


def test_summary_rejects_non_float_mean() -> None:
    with pytest.raises(TypeError, match="mean must be float"):
        _valid_summary(mean="x")  # type: ignore[arg-type]


def test_summary_rejects_bool_mean() -> None:
    with pytest.raises(TypeError, match="mean must be float"):
        _valid_summary(mean=True)  # type: ignore[arg-type]


def test_summary_rejects_nan_mean() -> None:
    with pytest.raises(ValueError, match="mean must be finite"):
        _valid_summary(mean=float("nan"))


def test_summary_rejects_inf_mean() -> None:
    with pytest.raises(ValueError, match="mean must be finite"):
        _valid_summary(mean=float("inf"))


def test_summary_rejects_negative_sd() -> None:
    with pytest.raises(ValueError, match="sd must be non-negative"):
        _valid_summary(sd=-0.1)


def test_summary_accepts_zero_sd() -> None:
    s = _valid_summary(sd=0.0)
    assert s.sd == 0.0


def test_summary_rejects_hdi_3_above_hdi_97() -> None:
    with pytest.raises(ValueError, match="hdi_3 must be <= hdi_97"):
        _valid_summary(hdi_3=0.8, hdi_97=0.2)


def test_summary_accepts_equal_hdi_bounds() -> None:
    s = _valid_summary(hdi_3=0.5, hdi_97=0.5)
    assert s.hdi_3 == s.hdi_97 == 0.5


def test_summary_rejects_negative_ess_bulk() -> None:
    with pytest.raises(ValueError, match="ess_bulk must be non-negative"):
        _valid_summary(ess_bulk=-1.0)


def test_summary_rejects_negative_ess_tail() -> None:
    with pytest.raises(ValueError, match="ess_tail must be non-negative"):
        _valid_summary(ess_tail=-1.0)


def test_summary_rejects_negative_r_hat() -> None:
    with pytest.raises(ValueError, match="r_hat must be non-negative"):
        _valid_summary(r_hat=-1.0)


def test_summary_rejects_nan_hdi_3() -> None:
    with pytest.raises(ValueError, match="hdi_3 must be finite"):
        _valid_summary(hdi_3=float("nan"))


def test_summary_rejects_inf_hdi_97() -> None:
    with pytest.raises(ValueError, match="hdi_97 must be finite"):
        _valid_summary(hdi_97=float("inf"))


# ---------------------------------------------------------------------------
# ArviZDiagnosticResult
# ---------------------------------------------------------------------------


def _valid_result(**overrides: Any) -> ArviZDiagnosticResult:
    base: dict[str, Any] = {
        "variable_summaries": (
            _valid_summary(name="mu"),
            _valid_summary(name="sigma"),
        ),
        "num_divergences": 0,
    }
    base.update(overrides)
    return ArviZDiagnosticResult(**base)


def test_result_constructs_with_defaults() -> None:
    r = _valid_result()
    assert len(r.variable_summaries) == 2
    assert r.num_divergences == 0


def test_result_is_frozen_and_slotted() -> None:
    r = _valid_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.num_divergences = 5  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_non_tuple_summaries() -> None:
    with pytest.raises(TypeError, match="variable_summaries must be a tuple"):
        _valid_result(variable_summaries=[_valid_summary()])  # type: ignore[arg-type]


def test_result_rejects_empty_summaries() -> None:
    with pytest.raises(ValueError, match="variable_summaries must be non-empty"):
        _valid_result(variable_summaries=())


def test_result_rejects_too_many_summaries() -> None:
    too_many = tuple(_valid_summary(name=f"v{i}") for i in range(MAX_NUM_VARS + 1))
    with pytest.raises(ValueError, match="variable_summaries must have <="):
        _valid_result(variable_summaries=too_many)


def test_result_rejects_non_summary_entries() -> None:
    with pytest.raises(
        TypeError,
        match="variable_summaries entries must be ArviZVariableSummary",
    ):
        _valid_result(variable_summaries=(_valid_summary(), "x"))  # type: ignore[arg-type]


def test_result_rejects_duplicate_variable_names() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        _valid_result(
            variable_summaries=(
                _valid_summary(name="mu"),
                _valid_summary(name="mu"),
            )
        )


def test_result_rejects_non_int_num_divergences() -> None:
    with pytest.raises(TypeError, match="num_divergences must be int"):
        _valid_result(num_divergences=1.0)  # type: ignore[arg-type]


def test_result_rejects_bool_num_divergences() -> None:
    with pytest.raises(TypeError, match="num_divergences must be int"):
        _valid_result(num_divergences=True)  # type: ignore[arg-type]


def test_result_rejects_negative_num_divergences() -> None:
    with pytest.raises(ValueError, match="num_divergences must be non-negative"):
        _valid_result(num_divergences=-1)


# ---------------------------------------------------------------------------
# ArviZDiagnosticRecord
# ---------------------------------------------------------------------------


def _hex16() -> str:
    return hashlib.blake2b(b"x", digest_size=8).hexdigest()


def _valid_record(**overrides: Any) -> ArviZDiagnosticRecord:
    base: dict[str, Any] = {
        "ts_ns": 1_000_000_000,
        "analysis_id": "aid-1",
        "source": ANALYSIS_SOURCE,
        "spec": _valid_spec(),
        "result": _valid_result(),
        "analysis_digest": _hex16(),
        "meta": {"k": "v"},
    }
    base.update(overrides)
    return ArviZDiagnosticRecord(**base)


def test_record_constructs_with_defaults() -> None:
    r = _valid_record()
    assert r.ts_ns == 1_000_000_000
    assert r.analysis_id == "aid-1"
    assert r.source == ANALYSIS_SOURCE


def test_record_is_frozen_and_slotted() -> None:
    r = _valid_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.ts_ns = 2  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_record_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be non-negative"):
        _valid_record(ts_ns=-1)


def test_record_rejects_non_int_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        _valid_record(ts_ns=1.0)  # type: ignore[arg-type]


def test_record_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError, match="ts_ns must be int"):
        _valid_record(ts_ns=True)  # type: ignore[arg-type]


def test_record_rejects_empty_analysis_id() -> None:
    with pytest.raises(ValueError, match="analysis_id must be non-empty"):
        _valid_record(analysis_id="")


def test_record_rejects_long_analysis_id() -> None:
    with pytest.raises(ValueError, match="analysis_id must be <="):
        _valid_record(analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1))


def test_record_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source must be non-empty"):
        _valid_record(source="")


def test_record_rejects_non_spec() -> None:
    with pytest.raises(TypeError, match="spec must be ArviZPosteriorSpec"):
        _valid_record(spec="x")  # type: ignore[arg-type]


def test_record_rejects_non_result() -> None:
    with pytest.raises(TypeError, match="result must be ArviZDiagnosticResult"):
        _valid_record(result="x")  # type: ignore[arg-type]


def test_record_rejects_wrong_digest_length() -> None:
    with pytest.raises(ValueError, match="16-hex-char digest"):
        _valid_record(analysis_digest="abc")


def test_record_rejects_non_hex_digest() -> None:
    with pytest.raises(ValueError, match="lowercase hex"):
        _valid_record(analysis_digest="g" * 16)


# ---------------------------------------------------------------------------
# Null callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_arviz_diagnostic_callback()
    assert isinstance(cb, ArviZDiagnosticCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_arviz_diagnostic_callback()
    spec = _valid_spec()
    args = _valid_args()
    summary = _valid_summary()
    result = _valid_result()
    assert cb.on_diagnostic_start(ts_ns=0, spec=spec, arguments=args) is None
    assert cb.on_variable_summary(ts_ns=0, summary=summary) is None
    assert cb.on_diagnostic_end(ts_ns=0, result=result) is None


# ---------------------------------------------------------------------------
# Deterministic fake engine
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeEngine:
    """Deterministic arviz engine for testing."""

    num_divergences: int = 0
    delta: float = 0.0

    def diagnose(
        self,
        *,
        spec: ArviZPosteriorSpec,
        arguments: ArviZDiagnosticArguments,
        ts_ns: int,
        callback: ArviZDiagnosticCallback,
    ) -> ArviZDiagnosticResult:
        summaries = []
        for i in range(spec.num_vars):
            base = float(arguments.random_seed + i) + self.delta
            s = ArviZVariableSummary(
                name=f"var_{i}",
                mean=base,
                sd=0.1 * (i + 1),
                hdi_3=base - 0.5,
                hdi_97=base + 0.5,
                ess_bulk=float(spec.num_draws * (i + 1)),
                ess_tail=float(spec.num_draws * (i + 1)) * 0.9,
                r_hat=1.0 + 0.001 * i,
            )
            callback.on_variable_summary(ts_ns=ts_ns, summary=s)
            summaries.append(s)
        return ArviZDiagnosticResult(
            variable_summaries=tuple(summaries),
            num_divergences=self.num_divergences,
        )


# ---------------------------------------------------------------------------
# ArviZDiagnosticAnalyser end-to-end
# ---------------------------------------------------------------------------


def test_analyser_constructs_with_valid_engine() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    assert isinstance(a, ArviZDiagnosticAnalyser)


def test_analyser_is_frozen_and_slotted() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.engine = _FakeEngine()  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_analyser_rejects_non_protocol_engine() -> None:
    class _NotEngine:
        pass

    with pytest.raises(TypeError, match="engine must implement the ArviZDiagnosticEngine"):
        ArviZDiagnosticAnalyser(engine=_NotEngine())  # type: ignore[arg-type]


def test_analyser_emits_canonical_record() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    record = a.analyse(
        spec=_valid_spec(num_vars=3),
        arguments=_valid_args(),
        ts_ns=1234,
        analysis_id="aid-x",
    )
    assert isinstance(record, ArviZDiagnosticRecord)
    assert record.ts_ns == 1234
    assert record.analysis_id == "aid-x"
    assert record.source == ANALYSIS_SOURCE
    assert len(record.result.variable_summaries) == 3
    assert len(record.analysis_digest) == 16
    assert all(c in "0123456789abcdef" for c in record.analysis_digest)


def test_analyser_meta_contains_canonical_keys() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    record = a.analyse(
        spec=_valid_spec(num_vars=2),
        arguments=_valid_args(meta={"strategy": "alpha"}),
        ts_ns=10,
        analysis_id="aid",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["diagnostic_kind"] == "summary"
    assert record.meta["random_seed"] == "42"
    assert record.meta["hdi_prob"] == repr(0.94)
    assert record.meta["num_chains"] == "4"
    assert record.meta["num_draws"] == "500"
    assert record.meta["num_vars"] == "2"
    assert record.meta["sample_count"] == "4"
    assert record.meta["variable_summary_count"] == "2"
    assert record.meta["num_divergences"] == "0"
    assert record.meta["strategy"] == "alpha"


def test_analyser_rejects_non_spec() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(TypeError, match="spec must be ArviZPosteriorSpec"):
        a.analyse(
            spec="x",  # type: ignore[arg-type]
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_non_args() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(TypeError, match="arguments must be ArviZDiagnosticArguments"):
        a.analyse(
            spec=_valid_spec(),
            arguments="x",  # type: ignore[arg-type]
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_non_int_ts_ns() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(TypeError, match="ts_ns must be int"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=1.5,  # type: ignore[arg-type]
            analysis_id="aid",
        )


def test_analyser_rejects_negative_ts_ns() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(ArviZAnalyserConfigError, match="ts_ns must be non-negative"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=-1,
            analysis_id="aid",
        )


def test_analyser_rejects_empty_analysis_id() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(ArviZAnalyserConfigError, match="analysis_id must be non-empty"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="",
        )


def test_analyser_rejects_long_analysis_id() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    with pytest.raises(ArviZAnalyserConfigError, match="analysis_id must be <="):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyser_rejects_engine_returning_wrong_type() -> None:
    class _BadEngine:
        def diagnose(
            self,
            *,
            spec: ArviZPosteriorSpec,
            arguments: ArviZDiagnosticArguments,
            ts_ns: int,
            callback: ArviZDiagnosticCallback,
        ) -> ArviZDiagnosticResult:
            return "not a result"  # type: ignore[return-value]

    a = ArviZDiagnosticAnalyser(engine=_BadEngine())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return ArviZDiagnosticResult"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_mismatched_variable_count() -> None:
    class _MismatchEngine:
        def diagnose(
            self,
            *,
            spec: ArviZPosteriorSpec,
            arguments: ArviZDiagnosticArguments,
            ts_ns: int,
            callback: ArviZDiagnosticCallback,
        ) -> ArviZDiagnosticResult:
            return ArviZDiagnosticResult(
                variable_summaries=(_valid_summary(name="only_one"),),
                num_divergences=0,
            )

    a = ArviZDiagnosticAnalyser(engine=_MismatchEngine())  # type: ignore[arg-type]
    with pytest.raises(
        ArviZAnalyserConfigError,
        match="variable_summaries length",
    ):
        a.analyse(
            spec=_valid_spec(num_vars=3),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_non_callback() -> None:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())

    class _NotCB:
        pass

    with pytest.raises(
        TypeError,
        match="callback must implement the ArviZDiagnosticCallback",
    ):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
            callback=_NotCB(),  # type: ignore[arg-type]
        )


def test_analyser_threads_custom_callback() -> None:
    summaries_seen: list[ArviZVariableSummary] = []
    start_seen: list[ArviZPosteriorSpec] = []
    end_seen: list[ArviZDiagnosticResult] = []

    class _Recorder:
        def on_diagnostic_start(
            self,
            *,
            ts_ns: int,
            spec: ArviZPosteriorSpec,
            arguments: ArviZDiagnosticArguments,
        ) -> None:
            start_seen.append(spec)

        def on_variable_summary(
            self,
            *,
            ts_ns: int,
            summary: ArviZVariableSummary,
        ) -> None:
            summaries_seen.append(summary)

        def on_diagnostic_end(
            self,
            *,
            ts_ns: int,
            result: ArviZDiagnosticResult,
        ) -> None:
            end_seen.append(result)

    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    a.analyse(
        spec=_valid_spec(num_vars=2),
        arguments=_valid_args(),
        ts_ns=10,
        analysis_id="aid",
        callback=_Recorder(),
    )
    assert len(start_seen) == 1
    assert len(summaries_seen) == 2
    assert len(end_seen) == 1


# ---------------------------------------------------------------------------
# INV-15: three-run byte-identical replay
# ---------------------------------------------------------------------------


def _run_once(
    *,
    seed: int = 42,
    num_vars: int = 2,
    hdi_prob: float = 0.94,
    kind: ArviZDiagnosticKind = ArviZDiagnosticKind.SUMMARY,
) -> ArviZDiagnosticRecord:
    a = ArviZDiagnosticAnalyser(engine=_FakeEngine())
    return a.analyse(
        spec=_valid_spec(num_vars=num_vars),
        arguments=_valid_args(
            diagnostic_kind=kind,
            random_seed=seed,
            hdi_prob=hdi_prob,
        ),
        ts_ns=999,
        analysis_id="inv15",
    )


def test_inv15_three_run_byte_identical_digest() -> None:
    a = _run_once()
    b = _run_once()
    c = _run_once()
    assert a.analysis_digest == b.analysis_digest == c.analysis_digest


def test_inv15_three_run_byte_identical_record() -> None:
    a = _run_once()
    b = _run_once()
    c = _run_once()
    assert a == b == c


def test_inv15_digest_changes_with_seed() -> None:
    a = _run_once(seed=1)
    b = _run_once(seed=2)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_changes_with_kind() -> None:
    a = _run_once(kind=ArviZDiagnosticKind.SUMMARY)
    b = _run_once(kind=ArviZDiagnosticKind.RHAT)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_changes_with_num_vars() -> None:
    a = _run_once(num_vars=2)
    b = _run_once(num_vars=3)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_changes_with_hdi_prob() -> None:
    a = _run_once(hdi_prob=0.90)
    b = _run_once(hdi_prob=0.94)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_is_16_hex_chars() -> None:
    r = _run_once()
    assert len(r.analysis_digest) == 16
    assert all(c in "0123456789abcdef" for c in r.analysis_digest)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_arviz_diagnostic_engine_factory_signature() -> None:
    """Factory exists and can be called when packages installed.

    If arviz is not present, ImportError is raised with a helpful
    pip install hint that references NEW_PIP_DEPENDENCIES.
    """

    try:
        engine = arviz_diagnostic_engine()
    except ImportError as exc:
        msg = str(exc)
        assert "pip install" in msg
        assert "arviz" in msg
        return
    assert isinstance(engine, ArviZDiagnosticEngine)


# ---------------------------------------------------------------------------
# AST guards: OFFLINE_ONLY tier — no top-level vendor imports
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parent.parent / "intelligence_engine" / "diag_arviz.py"


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports() -> set[str]:
    tree = _module_ast()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_ast_no_top_level_arviz_import() -> None:
    assert "arviz" not in _top_level_imports()


def test_ast_no_top_level_numpy_import() -> None:
    assert "numpy" not in _top_level_imports()


def test_ast_no_top_level_xarray_import() -> None:
    assert "xarray" not in _top_level_imports()


def test_ast_no_top_level_io_imports() -> None:
    forbidden = {
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "asyncio",
    }
    assert forbidden.isdisjoint(_top_level_imports())


def test_ast_no_cross_engine_imports() -> None:
    """No imports from execution_engine / governance_engine / system_engine."""

    tree = _module_ast()
    forbidden = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "registry",
        "ui",
    }
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                seen.add(node.module.split(".")[0])
    assert forbidden.isdisjoint(seen), f"forbidden imports in diag_arviz.py: {forbidden & seen!r}"


def test_ast_vendor_imports_confined_to_factory() -> None:
    """All vendor imports must live inside ``arviz_diagnostic_engine``
    body."""

    tree = _module_ast()
    factory_imports: set[str] = set()
    other_imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "arviz_diagnostic_engine":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        factory_imports.add(alias.name.split(".")[0])

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                other_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                other_imports.add(node.module.split(".")[0])

    assert "arviz" in factory_imports
    assert "numpy" in factory_imports
    assert "xarray" in factory_imports
    assert {"arviz", "numpy", "xarray"}.isdisjoint(other_imports)


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.diag_arviz as m1

    importlib.reload(m1)
    import intelligence_engine.diag_arviz as m2

    assert m1.ANALYSIS_SOURCE == m2.ANALYSIS_SOURCE
    assert m1.NEW_PIP_DEPENDENCIES == m2.NEW_PIP_DEPENDENCIES
    assert tuple(m1.__all__) == tuple(m2.__all__)
