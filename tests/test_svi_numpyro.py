"""C-41 — tests for the numpyro SVI/MCMC inference analyser surface."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.svi_numpyro import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_MODEL_DIGEST_LEN,
    MAX_NUM_SAMPLES,
    MAX_NUM_SITES,
    MAX_NUM_WARMUP,
    MAX_OBSERVATION_LEN,
    MAX_SITE_NAME_LEN,
    MIN_NUM_SAMPLES,
    MIN_NUM_SITES,
    MIN_NUM_WARMUP,
    MIN_OBSERVATION_LEN,
    NEW_PIP_DEPENDENCIES,
    NumpyroAnalyserConfigError,
    NumpyroInferenceArguments,
    NumpyroInferenceCallback,
    NumpyroInferenceEngine,
    NumpyroInferenceKind,
    NumpyroInferenceRecord,
    NumpyroInferenceResult,
    NumpyroModelSpec,
    NumpyroSiteSummary,
    NumpyroSVIAnalyser,
    null_numpyro_inference_callback,
    numpyro_svi_engine,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("numpyro", "jax", "numpy")


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.svi_numpyro"


def test_bounds_constants() -> None:
    assert MIN_NUM_SITES == 1
    assert MAX_NUM_SITES == 1024
    assert MIN_NUM_SAMPLES == 1
    assert MAX_NUM_SAMPLES == 100_000
    assert MIN_NUM_WARMUP == 0
    assert MAX_NUM_WARMUP == 100_000
    assert MIN_OBSERVATION_LEN == 0
    assert MAX_OBSERVATION_LEN == 100_000
    assert MAX_ANALYSIS_ID_LEN == 256
    assert MAX_MODEL_DIGEST_LEN == 64
    assert MAX_SITE_NAME_LEN == 128


# ---------------------------------------------------------------------------
# NumpyroInferenceKind
# ---------------------------------------------------------------------------


def test_inference_kind_values() -> None:
    assert NumpyroInferenceKind.SVI.value == "SVI"
    assert NumpyroInferenceKind.NUTS.value == "NUTS"
    assert NumpyroInferenceKind.HMC.value == "HMC"
    assert NumpyroInferenceKind.SA.value == "SA"


def test_inference_kind_count() -> None:
    assert len(list(NumpyroInferenceKind)) == 4


# ---------------------------------------------------------------------------
# NumpyroModelSpec
# ---------------------------------------------------------------------------


def _valid_spec(**overrides: Any) -> NumpyroModelSpec:
    base: dict[str, Any] = {
        "num_sites": 2,
        "num_observations": 10,
        "model_digest": "model_abcdef",
    }
    base.update(overrides)
    return NumpyroModelSpec(**base)


def test_spec_constructs_with_defaults() -> None:
    s = _valid_spec()
    assert s.num_sites == 2
    assert s.num_observations == 10
    assert s.model_digest == "model_abcdef"


def test_spec_is_frozen_and_slotted() -> None:
    s = _valid_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.num_sites = 4  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_spec_rejects_num_sites_below_min() -> None:
    with pytest.raises(ValueError, match="num_sites must be >="):
        _valid_spec(num_sites=0)


def test_spec_rejects_num_sites_above_max() -> None:
    with pytest.raises(ValueError, match="num_sites must be <="):
        _valid_spec(num_sites=MAX_NUM_SITES + 1)


def test_spec_rejects_num_sites_non_int() -> None:
    with pytest.raises(TypeError, match="num_sites must be int"):
        _valid_spec(num_sites=2.0)  # type: ignore[arg-type]


def test_spec_rejects_num_sites_bool() -> None:
    with pytest.raises(TypeError, match="num_sites must be int"):
        _valid_spec(num_sites=True)  # type: ignore[arg-type]


def test_spec_rejects_num_observations_negative() -> None:
    with pytest.raises(ValueError, match="num_observations must be non-negative"):
        _valid_spec(num_observations=-1)


def test_spec_rejects_num_observations_above_max() -> None:
    with pytest.raises(ValueError, match="num_observations must be <="):
        _valid_spec(num_observations=MAX_OBSERVATION_LEN + 1)


def test_spec_rejects_num_observations_non_int() -> None:
    with pytest.raises(TypeError, match="num_observations must be int"):
        _valid_spec(num_observations=1.0)  # type: ignore[arg-type]


def test_spec_rejects_num_observations_bool() -> None:
    with pytest.raises(TypeError, match="num_observations must be int"):
        _valid_spec(num_observations=False)  # type: ignore[arg-type]


def test_spec_accepts_zero_observations() -> None:
    s = _valid_spec(num_observations=0)
    assert s.num_observations == 0


def test_spec_rejects_empty_model_digest() -> None:
    with pytest.raises(ValueError, match="model_digest must be non-empty"):
        _valid_spec(model_digest="")


def test_spec_rejects_long_model_digest() -> None:
    with pytest.raises(ValueError, match="model_digest must be <="):
        _valid_spec(model_digest="x" * (MAX_MODEL_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# NumpyroInferenceArguments
# ---------------------------------------------------------------------------


def _valid_args(**overrides: Any) -> NumpyroInferenceArguments:
    base: dict[str, Any] = {
        "inference_kind": NumpyroInferenceKind.NUTS,
        "random_seed": 42,
        "num_samples": 100,
        "num_warmup": 50,
        "observations": (1.0, 2.0, 3.0),
        "meta": {},
    }
    base.update(overrides)
    return NumpyroInferenceArguments(**base)


def test_args_constructs_with_defaults() -> None:
    a = _valid_args()
    assert a.inference_kind is NumpyroInferenceKind.NUTS
    assert a.random_seed == 42
    assert a.num_samples == 100
    assert a.num_warmup == 50
    assert a.observations == (1.0, 2.0, 3.0)
    assert a.meta == {}


def test_args_is_frozen_and_slotted() -> None:
    a = _valid_args()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.random_seed = 999  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_args_rejects_non_enum_inference_kind() -> None:
    with pytest.raises(TypeError, match="inference_kind must be NumpyroInferenceKind"):
        _valid_args(inference_kind="NUTS")  # type: ignore[arg-type]


def test_args_rejects_negative_random_seed() -> None:
    with pytest.raises(ValueError, match="random_seed must be non-negative"):
        _valid_args(random_seed=-1)


def test_args_rejects_random_seed_non_int() -> None:
    with pytest.raises(TypeError, match="random_seed must be int"):
        _valid_args(random_seed=1.5)  # type: ignore[arg-type]


def test_args_rejects_random_seed_bool() -> None:
    with pytest.raises(TypeError, match="random_seed must be int"):
        _valid_args(random_seed=True)  # type: ignore[arg-type]


def test_args_rejects_num_samples_below_min() -> None:
    with pytest.raises(ValueError, match="num_samples must be >="):
        _valid_args(num_samples=0)


def test_args_rejects_num_samples_above_max() -> None:
    with pytest.raises(ValueError, match="num_samples must be <="):
        _valid_args(num_samples=MAX_NUM_SAMPLES + 1)


def test_args_rejects_num_samples_non_int() -> None:
    with pytest.raises(TypeError, match="num_samples must be int"):
        _valid_args(num_samples=100.0)  # type: ignore[arg-type]


def test_args_rejects_num_samples_bool() -> None:
    with pytest.raises(TypeError, match="num_samples must be int"):
        _valid_args(num_samples=True)  # type: ignore[arg-type]


def test_args_accepts_zero_warmup() -> None:
    a = _valid_args(num_warmup=0)
    assert a.num_warmup == 0


def test_args_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError, match="num_warmup must be >="):
        _valid_args(num_warmup=-1)


def test_args_rejects_warmup_above_max() -> None:
    with pytest.raises(ValueError, match="num_warmup must be <="):
        _valid_args(num_warmup=MAX_NUM_WARMUP + 1)


def test_args_rejects_warmup_non_int() -> None:
    with pytest.raises(TypeError, match="num_warmup must be int"):
        _valid_args(num_warmup=10.0)  # type: ignore[arg-type]


def test_args_rejects_warmup_bool() -> None:
    with pytest.raises(TypeError, match="num_warmup must be int"):
        _valid_args(num_warmup=True)  # type: ignore[arg-type]


def test_args_rejects_observations_non_tuple() -> None:
    with pytest.raises(TypeError, match="observations must be a tuple"):
        _valid_args(observations=[1.0, 2.0])  # type: ignore[arg-type]


def test_args_accepts_empty_observations() -> None:
    a = _valid_args(observations=())
    assert a.observations == ()


def test_args_rejects_observations_above_max() -> None:
    with pytest.raises(ValueError, match="observations must have <="):
        _valid_args(observations=tuple([1.0] * (MAX_OBSERVATION_LEN + 1)))


def test_args_rejects_non_float_observation() -> None:
    with pytest.raises(TypeError, match="observations values must be float"):
        _valid_args(observations=(1.0, "x"))  # type: ignore[arg-type]


def test_args_rejects_bool_observation() -> None:
    with pytest.raises(TypeError, match="observations values must be float"):
        _valid_args(observations=(1.0, True))  # type: ignore[arg-type]


def test_args_rejects_nan_observation() -> None:
    with pytest.raises(ValueError, match="observations values must be finite"):
        _valid_args(observations=(1.0, float("nan")))


def test_args_rejects_inf_observation() -> None:
    with pytest.raises(ValueError, match="observations values must be finite"):
        _valid_args(observations=(1.0, float("inf")))


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
# NumpyroSiteSummary
# ---------------------------------------------------------------------------


def _valid_summary(**overrides: Any) -> NumpyroSiteSummary:
    base: dict[str, Any] = {
        "name": "mu",
        "mean": 0.5,
        "std": 0.1,
        "effective_sample_size": 50.0,
        "r_hat": 1.01,
        "divergences": 0,
    }
    base.update(overrides)
    return NumpyroSiteSummary(**base)


def test_summary_constructs_with_defaults() -> None:
    s = _valid_summary()
    assert s.name == "mu"
    assert s.mean == 0.5
    assert s.std == 0.1
    assert s.effective_sample_size == 50.0
    assert s.r_hat == 1.01
    assert s.divergences == 0


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
        _valid_summary(name="x" * (MAX_SITE_NAME_LEN + 1))


def test_summary_rejects_non_str_name() -> None:
    with pytest.raises(TypeError, match="name must be str"):
        _valid_summary(name=1)  # type: ignore[arg-type]


def test_summary_rejects_non_float_mean() -> None:
    with pytest.raises(TypeError, match="mean must be float"):
        _valid_summary(mean="x")  # type: ignore[arg-type]


def test_summary_rejects_nan_mean() -> None:
    with pytest.raises(ValueError, match="mean must be finite"):
        _valid_summary(mean=float("nan"))


def test_summary_rejects_inf_mean() -> None:
    with pytest.raises(ValueError, match="mean must be finite"):
        _valid_summary(mean=float("inf"))


def test_summary_rejects_negative_std() -> None:
    with pytest.raises(ValueError, match="std must be non-negative"):
        _valid_summary(std=-0.1)


def test_summary_accepts_zero_std() -> None:
    s = _valid_summary(std=0.0)
    assert s.std == 0.0


def test_summary_rejects_negative_ess() -> None:
    with pytest.raises(
        ValueError,
        match="effective_sample_size must be non-negative",
    ):
        _valid_summary(effective_sample_size=-1.0)


def test_summary_rejects_negative_r_hat() -> None:
    with pytest.raises(ValueError, match="r_hat must be non-negative"):
        _valid_summary(r_hat=-1.0)


def test_summary_rejects_bool_mean() -> None:
    with pytest.raises(TypeError, match="mean must be float"):
        _valid_summary(mean=True)  # type: ignore[arg-type]


def test_summary_rejects_divergences_non_int() -> None:
    with pytest.raises(TypeError, match="divergences must be int"):
        _valid_summary(divergences=1.0)  # type: ignore[arg-type]


def test_summary_rejects_divergences_bool() -> None:
    with pytest.raises(TypeError, match="divergences must be int"):
        _valid_summary(divergences=True)  # type: ignore[arg-type]


def test_summary_rejects_negative_divergences() -> None:
    with pytest.raises(ValueError, match="divergences must be non-negative"):
        _valid_summary(divergences=-1)


# ---------------------------------------------------------------------------
# NumpyroInferenceResult
# ---------------------------------------------------------------------------


def _valid_result(**overrides: Any) -> NumpyroInferenceResult:
    base: dict[str, Any] = {
        "site_summaries": (
            _valid_summary(name="mu"),
            _valid_summary(name="sigma"),
        ),
        "log_evidence": -10.0,
    }
    base.update(overrides)
    return NumpyroInferenceResult(**base)


def test_result_constructs_with_defaults() -> None:
    r = _valid_result()
    assert len(r.site_summaries) == 2
    assert r.log_evidence == -10.0


def test_result_is_frozen_and_slotted() -> None:
    r = _valid_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.log_evidence = -5.0  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_non_tuple_summaries() -> None:
    with pytest.raises(TypeError, match="site_summaries must be a tuple"):
        _valid_result(site_summaries=[_valid_summary()])  # type: ignore[arg-type]


def test_result_rejects_empty_summaries() -> None:
    with pytest.raises(ValueError, match="site_summaries must be non-empty"):
        _valid_result(site_summaries=())


def test_result_rejects_too_many_summaries() -> None:
    too_many = tuple(_valid_summary(name=f"s{i}") for i in range(MAX_NUM_SITES + 1))
    with pytest.raises(ValueError, match="site_summaries must have <="):
        _valid_result(site_summaries=too_many)


def test_result_rejects_non_summary_entries() -> None:
    with pytest.raises(
        TypeError,
        match="site_summaries entries must be NumpyroSiteSummary",
    ):
        _valid_result(site_summaries=(_valid_summary(), "x"))  # type: ignore[arg-type]


def test_result_rejects_duplicate_site_names() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        _valid_result(
            site_summaries=(
                _valid_summary(name="mu"),
                _valid_summary(name="mu"),
            )
        )


def test_result_rejects_non_float_log_evidence() -> None:
    with pytest.raises(TypeError, match="log_evidence must be float"):
        _valid_result(log_evidence="x")  # type: ignore[arg-type]


def test_result_rejects_bool_log_evidence() -> None:
    with pytest.raises(TypeError, match="log_evidence must be float"):
        _valid_result(log_evidence=True)  # type: ignore[arg-type]


def test_result_rejects_nan_log_evidence() -> None:
    with pytest.raises(ValueError, match="log_evidence must be finite"):
        _valid_result(log_evidence=float("nan"))


def test_result_rejects_inf_log_evidence() -> None:
    with pytest.raises(ValueError, match="log_evidence must be finite"):
        _valid_result(log_evidence=float("inf"))


def test_result_accepts_positive_log_evidence() -> None:
    r = _valid_result(log_evidence=5.0)
    assert r.log_evidence == 5.0


# ---------------------------------------------------------------------------
# NumpyroInferenceRecord
# ---------------------------------------------------------------------------


def _hex16() -> str:
    return hashlib.blake2b(b"x", digest_size=8).hexdigest()


def _valid_record(**overrides: Any) -> NumpyroInferenceRecord:
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
    return NumpyroInferenceRecord(**base)


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
    with pytest.raises(TypeError, match="spec must be NumpyroModelSpec"):
        _valid_record(spec="x")  # type: ignore[arg-type]


def test_record_rejects_non_result() -> None:
    with pytest.raises(TypeError, match="result must be NumpyroInferenceResult"):
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
    cb = null_numpyro_inference_callback()
    assert isinstance(cb, NumpyroInferenceCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_numpyro_inference_callback()
    spec = _valid_spec()
    args = _valid_args()
    summary = _valid_summary()
    result = _valid_result()
    assert cb.on_inference_start(ts_ns=0, spec=spec, arguments=args) is None
    assert cb.on_site_summary(ts_ns=0, summary=summary) is None
    assert cb.on_inference_end(ts_ns=0, result=result) is None


# ---------------------------------------------------------------------------
# Deterministic fake engine
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeEngine:
    """Deterministic numpyro engine for testing."""

    log_evidence: float = -10.0
    delta: float = 0.0

    def infer(
        self,
        *,
        spec: NumpyroModelSpec,
        arguments: NumpyroInferenceArguments,
        ts_ns: int,
        callback: NumpyroInferenceCallback,
    ) -> NumpyroInferenceResult:
        summaries = []
        for i in range(spec.num_sites):
            s = NumpyroSiteSummary(
                name=f"site_{i}",
                mean=float(arguments.random_seed + i) + self.delta,
                std=0.1 * (i + 1),
                effective_sample_size=float(arguments.num_samples),
                r_hat=1.0 + 0.001 * i,
                divergences=i,
            )
            callback.on_site_summary(ts_ns=ts_ns, summary=s)
            summaries.append(s)
        return NumpyroInferenceResult(
            site_summaries=tuple(summaries),
            log_evidence=self.log_evidence,
        )


# ---------------------------------------------------------------------------
# NumpyroSVIAnalyser end-to-end
# ---------------------------------------------------------------------------


def test_analyser_constructs_with_valid_engine() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    assert isinstance(a, NumpyroSVIAnalyser)


def test_analyser_is_frozen_and_slotted() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.engine = _FakeEngine()  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_analyser_rejects_non_protocol_engine() -> None:
    class _NotEngine:
        pass

    with pytest.raises(TypeError, match="engine must implement the NumpyroInferenceEngine"):
        NumpyroSVIAnalyser(engine=_NotEngine())  # type: ignore[arg-type]


def test_analyser_emits_canonical_record() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    record = a.analyse(
        spec=_valid_spec(num_sites=3),
        arguments=_valid_args(),
        ts_ns=1234,
        analysis_id="aid-x",
    )
    assert isinstance(record, NumpyroInferenceRecord)
    assert record.ts_ns == 1234
    assert record.analysis_id == "aid-x"
    assert record.source == ANALYSIS_SOURCE
    assert len(record.result.site_summaries) == 3
    assert len(record.analysis_digest) == 16
    assert all(c in "0123456789abcdef" for c in record.analysis_digest)


def test_analyser_meta_contains_canonical_keys() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    record = a.analyse(
        spec=_valid_spec(num_sites=2),
        arguments=_valid_args(meta={"strategy": "alpha"}),
        ts_ns=10,
        analysis_id="aid",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["inference_kind"] == "NUTS"
    assert record.meta["random_seed"] == "42"
    assert record.meta["num_samples"] == "100"
    assert record.meta["num_warmup"] == "50"
    assert record.meta["num_sites"] == "2"
    assert record.meta["num_observations"] == "10"
    assert record.meta["observation_count"] == "3"
    assert record.meta["site_summary_count"] == "2"
    assert record.meta["strategy"] == "alpha"


def test_analyser_rejects_non_spec() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(TypeError, match="spec must be NumpyroModelSpec"):
        a.analyse(
            spec="x",  # type: ignore[arg-type]
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_non_args() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(TypeError, match="arguments must be NumpyroInferenceArguments"):
        a.analyse(
            spec=_valid_spec(),
            arguments="x",  # type: ignore[arg-type]
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_non_int_ts_ns() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(TypeError, match="ts_ns must be int"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=1.5,  # type: ignore[arg-type]
            analysis_id="aid",
        )


def test_analyser_rejects_negative_ts_ns() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(NumpyroAnalyserConfigError, match="ts_ns must be non-negative"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=-1,
            analysis_id="aid",
        )


def test_analyser_rejects_empty_analysis_id() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(NumpyroAnalyserConfigError, match="analysis_id must be non-empty"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="",
        )


def test_analyser_rejects_long_analysis_id() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    with pytest.raises(NumpyroAnalyserConfigError, match="analysis_id must be <="):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyser_rejects_engine_returning_wrong_type() -> None:
    class _BadEngine:
        def infer(
            self,
            *,
            spec: NumpyroModelSpec,
            arguments: NumpyroInferenceArguments,
            ts_ns: int,
            callback: NumpyroInferenceCallback,
        ) -> NumpyroInferenceResult:
            return "not a result"  # type: ignore[return-value]

    a = NumpyroSVIAnalyser(engine=_BadEngine())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return NumpyroInferenceResult"):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_mismatched_site_count() -> None:
    class _MismatchEngine:
        def infer(
            self,
            *,
            spec: NumpyroModelSpec,
            arguments: NumpyroInferenceArguments,
            ts_ns: int,
            callback: NumpyroInferenceCallback,
        ) -> NumpyroInferenceResult:
            return NumpyroInferenceResult(
                site_summaries=(_valid_summary(name="only_one"),),
                log_evidence=-1.0,
            )

    a = NumpyroSVIAnalyser(engine=_MismatchEngine())  # type: ignore[arg-type]
    with pytest.raises(
        NumpyroAnalyserConfigError,
        match="site_summaries length",
    ):
        a.analyse(
            spec=_valid_spec(num_sites=3),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
        )


def test_analyser_rejects_non_callback() -> None:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())

    class _NotCB:
        pass

    with pytest.raises(
        TypeError,
        match="callback must implement the NumpyroInferenceCallback",
    ):
        a.analyse(
            spec=_valid_spec(),
            arguments=_valid_args(),
            ts_ns=0,
            analysis_id="aid",
            callback=_NotCB(),  # type: ignore[arg-type]
        )


def test_analyser_threads_custom_callback() -> None:
    summaries_seen: list[NumpyroSiteSummary] = []
    start_seen: list[NumpyroModelSpec] = []
    end_seen: list[NumpyroInferenceResult] = []

    class _Recorder:
        def on_inference_start(
            self,
            *,
            ts_ns: int,
            spec: NumpyroModelSpec,
            arguments: NumpyroInferenceArguments,
        ) -> None:
            start_seen.append(spec)

        def on_site_summary(
            self,
            *,
            ts_ns: int,
            summary: NumpyroSiteSummary,
        ) -> None:
            summaries_seen.append(summary)

        def on_inference_end(
            self,
            *,
            ts_ns: int,
            result: NumpyroInferenceResult,
        ) -> None:
            end_seen.append(result)

    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    a.analyse(
        spec=_valid_spec(num_sites=2),
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
    num_sites: int = 2,
    num_warmup: int = 50,
    kind: NumpyroInferenceKind = NumpyroInferenceKind.NUTS,
) -> NumpyroInferenceRecord:
    a = NumpyroSVIAnalyser(engine=_FakeEngine())
    return a.analyse(
        spec=_valid_spec(num_sites=num_sites),
        arguments=_valid_args(
            inference_kind=kind,
            random_seed=seed,
            num_warmup=num_warmup,
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
    a = _run_once(kind=NumpyroInferenceKind.SVI)
    b = _run_once(kind=NumpyroInferenceKind.NUTS)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_changes_with_num_sites() -> None:
    a = _run_once(num_sites=2)
    b = _run_once(num_sites=3)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_changes_with_num_warmup() -> None:
    a = _run_once(num_warmup=50)
    b = _run_once(num_warmup=100)
    assert a.analysis_digest != b.analysis_digest


def test_inv15_digest_is_16_hex_chars() -> None:
    r = _run_once()
    assert len(r.analysis_digest) == 16
    assert all(c in "0123456789abcdef" for c in r.analysis_digest)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_numpyro_svi_engine_factory_signature() -> None:
    """Factory exists and can be called when packages installed.

    If numpyro is not present, ImportError is raised with a
    helpful pip install hint that references
    NEW_PIP_DEPENDENCIES.
    """

    try:
        engine = numpyro_svi_engine()
    except ImportError as exc:
        msg = str(exc)
        assert "pip install" in msg
        assert "numpyro" in msg
        return
    assert isinstance(engine, NumpyroInferenceEngine)


# ---------------------------------------------------------------------------
# AST guards: OFFLINE_ONLY tier — no top-level vendor imports
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parent.parent / "intelligence_engine" / "svi_numpyro.py"


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


def test_ast_no_top_level_numpyro_import() -> None:
    assert "numpyro" not in _top_level_imports()


def test_ast_no_top_level_jax_import() -> None:
    assert "jax" not in _top_level_imports()


def test_ast_no_top_level_numpy_import() -> None:
    assert "numpy" not in _top_level_imports()


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
    assert forbidden.isdisjoint(seen), f"forbidden imports in svi_numpyro.py: {forbidden & seen!r}"


def test_ast_vendor_imports_confined_to_factory() -> None:
    """All vendor imports must live inside ``numpyro_svi_engine``
    body."""

    tree = _module_ast()
    factory_imports: set[str] = set()
    other_imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "numpyro_svi_engine":
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

    assert "numpyro" in factory_imports
    assert "jax" in factory_imports
    assert "numpy" in factory_imports
    assert {"numpyro", "jax", "numpy"}.isdisjoint(other_imports)


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.svi_numpyro as m1

    importlib.reload(m1)
    import intelligence_engine.svi_numpyro as m2

    assert m1.ANALYSIS_SOURCE == m2.ANALYSIS_SOURCE
    assert m1.NEW_PIP_DEPENDENCIES == m2.NEW_PIP_DEPENDENCIES
    assert tuple(m1.__all__) == tuple(m2.__all__)
