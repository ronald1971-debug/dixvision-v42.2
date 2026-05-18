"""C-39 — tests for the hmmlearn HMM-inference analyser surface."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.hmm_hmmlearn import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_MODEL_DIGEST_LEN,
    MAX_N_COMPONENTS,
    MAX_N_FEATURES,
    MAX_OBSERVATION_LEN,
    MIN_N_COMPONENTS,
    MIN_N_FEATURES,
    MIN_OBSERVATION_LEN,
    NEW_PIP_DEPENDENCIES,
    HMMAnalyserConfigError,
    HMMInferenceArguments,
    HMMInferenceCallback,
    HMMInferenceRecord,
    HMMInferenceResult,
    HmmlearnAnalyser,
    HMMModelKind,
    HMMSpec,
    HMMStatePosterior,
    hmmlearn_gaussian_engine,
    null_hmm_inference_callback,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == (
        "hmmlearn",
        "numpy",
        "scipy",
        "scikit-learn",
    )


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.hmm_hmmlearn"


def test_bounds_constants() -> None:
    assert MIN_N_COMPONENTS == 1
    assert MAX_N_COMPONENTS == 256
    assert MIN_N_FEATURES == 1
    assert MAX_N_FEATURES == 256
    assert MIN_OBSERVATION_LEN == 1
    assert MAX_OBSERVATION_LEN == 100_000
    assert MAX_ANALYSIS_ID_LEN == 256
    assert MAX_MODEL_DIGEST_LEN == 64


# ---------------------------------------------------------------------------
# HMMModelKind
# ---------------------------------------------------------------------------


def test_model_kind_values() -> None:
    assert HMMModelKind.GAUSSIAN.value == "GaussianHMM"
    assert HMMModelKind.GMM.value == "GMMHMM"
    assert HMMModelKind.MULTINOMIAL.value == "MultinomialHMM"
    assert HMMModelKind.CATEGORICAL.value == "CategoricalHMM"


def test_model_kind_count() -> None:
    assert len(list(HMMModelKind)) == 4


# ---------------------------------------------------------------------------
# HMMSpec
# ---------------------------------------------------------------------------


def _valid_spec(**overrides: Any) -> HMMSpec:
    base: dict[str, Any] = {
        "n_components": 3,
        "n_features": 2,
        "model_digest": "model_abcdef",
    }
    base.update(overrides)
    return HMMSpec(**base)


def test_spec_constructs_with_defaults() -> None:
    s = _valid_spec()
    assert s.n_components == 3
    assert s.n_features == 2


def test_spec_is_frozen_and_slotted() -> None:
    s = _valid_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n_components = 4  # type: ignore[misc]
    assert not hasattr(s, "__dict__")


def test_spec_rejects_bool_n_components() -> None:
    with pytest.raises(TypeError):
        _valid_spec(n_components=True)


def test_spec_rejects_below_min_n_components() -> None:
    with pytest.raises(ValueError):
        _valid_spec(n_components=0)


def test_spec_rejects_above_max_n_components() -> None:
    with pytest.raises(ValueError):
        _valid_spec(n_components=MAX_N_COMPONENTS + 1)


def test_spec_rejects_bool_n_features() -> None:
    with pytest.raises(TypeError):
        _valid_spec(n_features=True)


def test_spec_rejects_below_min_n_features() -> None:
    with pytest.raises(ValueError):
        _valid_spec(n_features=0)


def test_spec_rejects_above_max_n_features() -> None:
    with pytest.raises(ValueError):
        _valid_spec(n_features=MAX_N_FEATURES + 1)


def test_spec_rejects_empty_model_digest() -> None:
    with pytest.raises(ValueError):
        _valid_spec(model_digest="")


def test_spec_rejects_oversize_model_digest() -> None:
    with pytest.raises(ValueError):
        _valid_spec(model_digest="x" * (MAX_MODEL_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# HMMInferenceArguments
# ---------------------------------------------------------------------------


def _valid_arguments(**overrides: Any) -> HMMInferenceArguments:
    base: dict[str, Any] = {
        "model_kind": HMMModelKind.GAUSSIAN,
        "random_seed": 0,
        "observations": (
            (0.1, 0.2),
            (0.3, 0.4),
            (0.5, 0.6),
        ),
        "meta": {},
    }
    base.update(overrides)
    return HMMInferenceArguments(**base)


def test_arguments_is_frozen_and_slotted() -> None:
    a = _valid_arguments()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.random_seed = 1  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_arguments_rejects_non_enum_kind() -> None:
    with pytest.raises(TypeError):
        HMMInferenceArguments(  # type: ignore[arg-type]
            model_kind="GaussianHMM",
            random_seed=0,
            observations=((0.0,),),
        )


def test_arguments_rejects_bool_seed() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(random_seed=True)


def test_arguments_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(random_seed=-1)


def test_arguments_rejects_non_tuple_observations() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(observations=[(0.1,)])  # type: ignore[arg-type]


def test_arguments_rejects_empty_observations() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(observations=())


def test_arguments_rejects_non_tuple_observation_step() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(observations=([0.1, 0.2],))  # type: ignore[arg-type]


def test_arguments_rejects_empty_observation_step() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(observations=((),))


def test_arguments_rejects_dimension_mismatch_across_steps() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(observations=((0.1, 0.2), (0.3,)))


def test_arguments_rejects_bool_observation_value() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(observations=((True, 0.2),))


def test_arguments_rejects_string_observation_value() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(
            observations=(("x", 0.2),)  # type: ignore[arg-type]
        )


def test_arguments_rejects_nan_observation_value() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(observations=((float("nan"), 0.2),))


def test_arguments_rejects_inf_observation_value() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(observations=((float("inf"), 0.2),))


def test_arguments_rejects_above_max_observation_len() -> None:
    obs = tuple((float(i),) for i in range(MAX_OBSERVATION_LEN + 1))
    with pytest.raises(ValueError):
        _valid_arguments(observations=obs)


def test_arguments_rejects_empty_meta_key() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(meta={"": "v"})


def test_arguments_rejects_empty_meta_value() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(meta={"k": ""})


# ---------------------------------------------------------------------------
# HMMStatePosterior
# ---------------------------------------------------------------------------


def _valid_posterior(**overrides: Any) -> HMMStatePosterior:
    base: dict[str, Any] = {
        "step_index": 0,
        "state_probabilities": (0.5, 0.3, 0.2),
    }
    base.update(overrides)
    return HMMStatePosterior(**base)


def test_posterior_is_frozen_and_slotted() -> None:
    p = _valid_posterior()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.step_index = 1  # type: ignore[misc]
    assert not hasattr(p, "__dict__")


def test_posterior_rejects_bool_step_index() -> None:
    with pytest.raises(TypeError):
        _valid_posterior(step_index=True)


def test_posterior_rejects_negative_step_index() -> None:
    with pytest.raises(ValueError):
        _valid_posterior(step_index=-1)


def test_posterior_rejects_non_tuple_probabilities() -> None:
    with pytest.raises(TypeError):
        _valid_posterior(state_probabilities=[0.5, 0.5])  # type: ignore[arg-type]


def test_posterior_rejects_empty_probabilities() -> None:
    with pytest.raises(ValueError):
        _valid_posterior(state_probabilities=())


def test_posterior_rejects_above_max_components() -> None:
    probs = tuple([1.0 / (MAX_N_COMPONENTS + 1)] * (MAX_N_COMPONENTS + 1))
    with pytest.raises(ValueError):
        _valid_posterior(state_probabilities=probs)


def test_posterior_rejects_bool_probability() -> None:
    with pytest.raises(TypeError):
        _valid_posterior(state_probabilities=(True, False))


def test_posterior_rejects_nan_probability() -> None:
    with pytest.raises(ValueError):
        _valid_posterior(state_probabilities=(float("nan"), 0.5, 0.5))


def test_posterior_rejects_negative_probability() -> None:
    with pytest.raises(ValueError):
        _valid_posterior(state_probabilities=(-0.1, 0.6, 0.5))


def test_posterior_rejects_probability_above_one() -> None:
    with pytest.raises(ValueError):
        _valid_posterior(state_probabilities=(1.5, -0.3, -0.2))


def test_posterior_rejects_probabilities_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        _valid_posterior(state_probabilities=(0.5, 0.3, 0.1))


# ---------------------------------------------------------------------------
# HMMInferenceResult
# ---------------------------------------------------------------------------


def _valid_result(**overrides: Any) -> HMMInferenceResult:
    base: dict[str, Any] = {
        "viterbi_path": (0, 1, 2),
        "posteriors": (
            HMMStatePosterior(
                step_index=0,
                state_probabilities=(0.6, 0.3, 0.1),
            ),
            HMMStatePosterior(
                step_index=1,
                state_probabilities=(0.2, 0.7, 0.1),
            ),
            HMMStatePosterior(
                step_index=2,
                state_probabilities=(0.1, 0.2, 0.7),
            ),
        ),
        "log_likelihood": -5.5,
    }
    base.update(overrides)
    return HMMInferenceResult(**base)


def test_result_is_frozen_and_slotted() -> None:
    r = _valid_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.log_likelihood = -1.0  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_non_tuple_viterbi_path() -> None:
    with pytest.raises(TypeError):
        _valid_result(viterbi_path=[0, 1, 2])  # type: ignore[arg-type]


def test_result_rejects_empty_viterbi_path() -> None:
    with pytest.raises(ValueError):
        _valid_result(viterbi_path=(), posteriors=())


def test_result_rejects_bool_viterbi_state() -> None:
    with pytest.raises(TypeError):
        _valid_result(
            viterbi_path=(True, False, True),
            posteriors=(
                _valid_posterior(step_index=0),
                _valid_posterior(step_index=1),
                _valid_posterior(step_index=2),
            ),
        )


def test_result_rejects_negative_viterbi_state() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            viterbi_path=(-1, 0, 1),
            posteriors=(
                _valid_posterior(step_index=0),
                _valid_posterior(step_index=1),
                _valid_posterior(step_index=2),
            ),
        )


def test_result_rejects_viterbi_state_above_max_components() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            viterbi_path=(MAX_N_COMPONENTS, 0, 1),
            posteriors=(
                _valid_posterior(step_index=0),
                _valid_posterior(step_index=1),
                _valid_posterior(step_index=2),
            ),
        )


def test_result_rejects_non_tuple_posteriors() -> None:
    with pytest.raises(TypeError):
        _valid_result(posteriors=[_valid_posterior()])  # type: ignore[arg-type]


def test_result_rejects_viterbi_posterior_length_mismatch() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            viterbi_path=(0, 1),
            posteriors=(_valid_posterior(),),
        )


def test_result_rejects_non_posterior_entry() -> None:
    with pytest.raises(TypeError):
        _valid_result(
            viterbi_path=(0, 1, 2),
            posteriors=("bad", "bad", "bad"),  # type: ignore[arg-type]
        )


def test_result_rejects_posterior_step_index_mismatch() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            viterbi_path=(0, 1, 2),
            posteriors=(
                _valid_posterior(step_index=99),
                _valid_posterior(step_index=1),
                _valid_posterior(step_index=2),
            ),
        )


def test_result_rejects_posterior_components_mismatch() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            viterbi_path=(0, 1, 2),
            posteriors=(
                HMMStatePosterior(step_index=0, state_probabilities=(0.5, 0.5)),
                _valid_posterior(step_index=1),
                _valid_posterior(step_index=2),
            ),
        )


def test_result_rejects_viterbi_state_above_posterior_components() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            viterbi_path=(0, 1, 5),
            posteriors=(
                _valid_posterior(step_index=0),
                _valid_posterior(step_index=1),
                _valid_posterior(step_index=2),
            ),
        )


def test_result_rejects_bool_log_likelihood() -> None:
    with pytest.raises(TypeError):
        _valid_result(log_likelihood=True)


def test_result_rejects_string_log_likelihood() -> None:
    with pytest.raises(TypeError):
        _valid_result(log_likelihood="bad")  # type: ignore[arg-type]


def test_result_rejects_nan_log_likelihood() -> None:
    with pytest.raises(ValueError):
        _valid_result(log_likelihood=float("nan"))


def test_result_rejects_positive_log_likelihood() -> None:
    with pytest.raises(ValueError):
        _valid_result(log_likelihood=1.0)


# ---------------------------------------------------------------------------
# HMMInferenceRecord
# ---------------------------------------------------------------------------


def _valid_record(**overrides: Any) -> HMMInferenceRecord:
    base: dict[str, Any] = {
        "ts_ns": 100,
        "analysis_id": "test_analysis",
        "source": ANALYSIS_SOURCE,
        "spec": _valid_spec(),
        "result": _valid_result(),
        "analysis_digest": "0123456789abcdef",
        "meta": {},
    }
    base.update(overrides)
    return HMMInferenceRecord(**base)


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


def test_record_rejects_non_spec() -> None:
    with pytest.raises(TypeError):
        _valid_record(spec="bad")  # type: ignore[arg-type]


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
# Deterministic fake engine
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Deterministic :class:`HMMInferenceEngine` fake."""

    __slots__ = ("_result",)

    def __init__(self, *, result: HMMInferenceResult) -> None:
        self._result = result

    def infer(
        self,
        *,
        spec: HMMSpec,
        arguments: HMMInferenceArguments,
        ts_ns: int,
        callback: HMMInferenceCallback,
    ) -> HMMInferenceResult:
        callback.on_inference_start(ts_ns=ts_ns, spec=spec, arguments=arguments)
        for post in self._result.posteriors:
            callback.on_step_posterior(ts_ns=ts_ns, posterior=post)
        return self._result


# ---------------------------------------------------------------------------
# HmmlearnAnalyser.analyse end-to-end
# ---------------------------------------------------------------------------


def _analyser_inputs() -> tuple[HMMSpec, HMMInferenceArguments, _FakeEngine]:
    return (
        _valid_spec(),
        _valid_arguments(),
        _FakeEngine(result=_valid_result()),
    )


def test_analyser_is_frozen_and_slotted() -> None:
    _, _, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(dataclasses.FrozenInstanceError):
        analyser.engine = None  # type: ignore[misc]
    assert not hasattr(analyser, "__dict__")


def test_analyser_rejects_non_engine() -> None:
    with pytest.raises(TypeError):
        HmmlearnAnalyser(engine="bad")  # type: ignore[arg-type]


def test_analyse_emits_record() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    record = analyser.analyse(
        spec=spec,
        arguments=args,
        ts_ns=12345,
        analysis_id="analysis_0001",
    )
    assert isinstance(record, HMMInferenceRecord)
    assert record.source == ANALYSIS_SOURCE
    assert record.analysis_id == "analysis_0001"
    assert record.ts_ns == 12345


def test_analyse_meta_includes_provenance() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    record = analyser.analyse(
        spec=spec,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["model_kind"] == "GaussianHMM"
    assert record.meta["random_seed"] == "0"
    assert record.meta["n_components"] == str(spec.n_components)
    assert record.meta["n_features"] == str(spec.n_features)
    assert record.meta["observation_count"] == str(len(args.observations))
    assert record.meta["viterbi_length"] == str(len(record.result.viterbi_path))
    assert record.meta["posterior_count"] == str(len(record.result.posteriors))


def test_analyse_caller_meta_does_not_override_provenance() -> None:
    spec, _, engine = _analyser_inputs()
    args = _valid_arguments(meta={"analysis_digest": "ZZ", "k": "v"})
    analyser = HmmlearnAnalyser(engine=engine)
    record = analyser.analyse(
        spec=spec,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["k"] == "v"


def test_analyse_rejects_non_spec() -> None:
    _, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            spec="bad",  # type: ignore[arg-type]
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_non_arguments() -> None:
    spec, _, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            spec=spec,
            arguments="bad",  # type: ignore[arg-type]
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_bool_ts_ns() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=True,  # type: ignore[arg-type]
            analysis_id="a",
        )


def test_analyse_rejects_negative_ts_ns() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(HMMAnalyserConfigError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=-1,
            analysis_id="a",
        )


def test_analyse_rejects_empty_analysis_id() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(HMMAnalyserConfigError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="",
        )


def test_analyse_rejects_oversize_analysis_id() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(HMMAnalyserConfigError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyse_rejects_dimension_mismatch_between_spec_and_observations() -> None:
    spec = _valid_spec(n_features=5)
    args = _valid_arguments()  # 2-d observations
    analyser = HmmlearnAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    with pytest.raises(HMMAnalyserConfigError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_engine_viterbi_path_length_mismatch() -> None:
    spec = _valid_spec()
    args = _valid_arguments(
        observations=(
            (0.1, 0.2),
            (0.3, 0.4),
        ),
    )
    analyser = HmmlearnAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    with pytest.raises(HMMAnalyserConfigError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_engine_state_count_mismatch() -> None:
    spec = _valid_spec(n_components=7)  # fake produces 3 states
    args = _valid_arguments()
    analyser = HmmlearnAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    with pytest.raises(HMMAnalyserConfigError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_uses_null_callback_by_default() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    record = analyser.analyse(
        spec=spec,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert isinstance(record, HMMInferenceRecord)


def test_analyse_rejects_non_protocol_callback() -> None:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
            callback="bad",  # type: ignore[arg-type]
        )


def test_analyse_rejects_engine_returning_wrong_type() -> None:
    class _BadEngine:
        def infer(
            self,
            *,
            spec: HMMSpec,
            arguments: HMMInferenceArguments,
            ts_ns: int,
            callback: HMMInferenceCallback,
        ) -> HMMInferenceResult:
            return "bad"  # type: ignore[return-value]

    spec, args, _ = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=_BadEngine())
    with pytest.raises(TypeError):
        analyser.analyse(
            spec=spec,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _run_once() -> HMMInferenceRecord:
    spec, args, engine = _analyser_inputs()
    analyser = HmmlearnAnalyser(engine=engine)
    return analyser.analyse(
        spec=spec,
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
    assert r1.spec == r2.spec == r3.spec


def test_inv15_digest_changes_when_seed_changes() -> None:
    spec = _valid_spec()
    args_a = _valid_arguments(random_seed=0)
    args_b = _valid_arguments(random_seed=1)
    analyser = HmmlearnAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    r0 = analyser.analyse(
        spec=spec,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = analyser.analyse(
        spec=spec,
        arguments=args_b,
        ts_ns=1,
        analysis_id="a",
    )
    assert r0.analysis_digest != r1.analysis_digest


def test_inv15_digest_changes_when_model_kind_changes() -> None:
    spec = _valid_spec()
    args_a = _valid_arguments(model_kind=HMMModelKind.GAUSSIAN)
    args_b = _valid_arguments(model_kind=HMMModelKind.MULTINOMIAL)
    analyser = HmmlearnAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    r0 = analyser.analyse(
        spec=spec,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = analyser.analyse(
        spec=spec,
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
# null_hmm_inference_callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_hmm_inference_callback()
    assert isinstance(cb, HMMInferenceCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_hmm_inference_callback()
    spec = _valid_spec()
    args = _valid_arguments()
    result = _valid_result()
    assert cb.on_inference_start(ts_ns=0, spec=spec, arguments=args) is None
    assert cb.on_step_posterior(ts_ns=0, posterior=_valid_posterior()) is None
    assert cb.on_inference_end(ts_ns=0, result=result) is None


# ---------------------------------------------------------------------------
# Convenience factory raises when hmmlearn missing
# ---------------------------------------------------------------------------


def test_hmmlearn_engine_factory_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("hmmlearn")
    except ImportError:
        with pytest.raises(ImportError, match="hmmlearn"):
            hmmlearn_gaussian_engine()
    else:
        pytest.skip("hmmlearn installed — production seam smoke skipped")


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "intelligence_engine" / "hmm_hmmlearn.py"


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


def test_no_top_level_hmmlearn_import() -> None:
    assert all(not name.startswith("hmmlearn") for name in _top_level_imports(_module_ast()))


def test_no_top_level_numpy_import() -> None:
    assert all(not name.startswith("numpy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_scipy_import() -> None:
    assert all(not name.startswith("scipy") for name in _top_level_imports(_module_ast()))


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


def test_hmmlearn_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith(("hmmlearn", "numpy", "scipy", "sklearn")):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside hmmlearn_gaussian_engine factory"
                    )
                    assert parent.name == "hmmlearn_gaussian_engine", (
                        f"{name} imported in {parent.name!r} — must "
                        "be inside hmmlearn_gaussian_engine"
                    )


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.hmm_hmmlearn as mod1

    importlib.reload(mod1)
    import intelligence_engine.hmm_hmmlearn as mod2

    assert mod1.ANALYSIS_SOURCE == mod2.ANALYSIS_SOURCE
    assert mod1.MAX_N_COMPONENTS == mod2.MAX_N_COMPONENTS
    assert mod1.HMMModelKind.GAUSSIAN is mod2.HMMModelKind.GAUSSIAN
