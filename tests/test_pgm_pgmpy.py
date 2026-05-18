"""C-38 — tests for the pgmpy Bayesian-inference analyser surface."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.pgm_pgmpy import (
    ANALYSIS_SOURCE,
    MAX_ANALYSIS_ID_LEN,
    MAX_CPD_DIGEST_LEN,
    MAX_EDGES,
    MAX_N_SAMPLES,
    MAX_NODES,
    MAX_STATES,
    MIN_N_SAMPLES,
    NEW_PIP_DEPENDENCIES,
    BayesianAnalyserConfigError,
    BayesianInferenceArguments,
    BayesianInferenceCallback,
    BayesianInferenceKind,
    BayesianInferenceRecord,
    BayesianInferenceResult,
    BayesianMarginalResult,
    BayesianNetworkSpec,
    PgmpyBayesianAnalyser,
    null_bayesian_inference_callback,
    pgmpy_variable_elimination_engine,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == (
        "pgmpy",
        "numpy",
        "pandas",
        "networkx",
    )


def test_analysis_source_is_canonical_module_path() -> None:
    assert ANALYSIS_SOURCE == "intelligence_engine.pgm_pgmpy"


def test_bounds_constants() -> None:
    assert MIN_N_SAMPLES == 1
    assert MAX_N_SAMPLES == 10_000_000
    assert MAX_NODES == 1024
    assert MAX_EDGES == 4096
    assert MAX_STATES == 256
    assert MAX_ANALYSIS_ID_LEN == 256
    assert MAX_CPD_DIGEST_LEN == 64


# ---------------------------------------------------------------------------
# BayesianInferenceKind
# ---------------------------------------------------------------------------


def test_inference_kind_values() -> None:
    assert BayesianInferenceKind.VARIABLE_ELIMINATION.value == "VariableElimination"
    assert BayesianInferenceKind.BELIEF_PROPAGATION.value == "BeliefPropagation"
    assert BayesianInferenceKind.GIBBS_SAMPLING.value == "GibbsSampling"
    assert BayesianInferenceKind.LIKELIHOOD_WEIGHTING.value == "LikelihoodWeighting"


def test_inference_kind_count() -> None:
    assert len(list(BayesianInferenceKind)) == 4


# ---------------------------------------------------------------------------
# BayesianNetworkSpec
# ---------------------------------------------------------------------------


def _valid_network(**overrides: Any) -> BayesianNetworkSpec:
    base: dict[str, Any] = {
        "nodes": ("regime", "signal", "outcome"),
        "edges": (("regime", "signal"), ("signal", "outcome")),
        "cpd_digest": "cpd1234567",
    }
    base.update(overrides)
    return BayesianNetworkSpec(**base)


def test_network_constructs_with_defaults() -> None:
    n = _valid_network()
    assert n.nodes == ("regime", "signal", "outcome")


def test_network_is_frozen_and_slotted() -> None:
    n = _valid_network()
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.nodes = ()  # type: ignore[misc]
    assert not hasattr(n, "__dict__")


def test_network_rejects_non_tuple_nodes() -> None:
    with pytest.raises(TypeError):
        _valid_network(nodes=["a"])  # type: ignore[arg-type]


def test_network_rejects_empty_nodes() -> None:
    with pytest.raises(ValueError):
        _valid_network(nodes=())


def test_network_rejects_empty_node_name() -> None:
    with pytest.raises(ValueError):
        _valid_network(nodes=("",))


def test_network_rejects_duplicate_nodes() -> None:
    with pytest.raises(ValueError):
        _valid_network(nodes=("a", "a"))


def test_network_rejects_above_max_nodes() -> None:
    nodes = tuple(f"n{i}" for i in range(MAX_NODES + 1))
    with pytest.raises(ValueError):
        _valid_network(nodes=nodes, edges=())


def test_network_rejects_non_tuple_edges() -> None:
    with pytest.raises(TypeError):
        _valid_network(edges=[("regime", "signal")])  # type: ignore[arg-type]


def test_network_accepts_empty_edges() -> None:
    n = _valid_network(edges=())
    assert n.edges == ()


def test_network_rejects_malformed_edge() -> None:
    with pytest.raises(TypeError):
        _valid_network(edges=(("only_one",),))  # type: ignore[arg-type]


def test_network_rejects_edge_parent_not_in_nodes() -> None:
    with pytest.raises(ValueError):
        _valid_network(edges=(("missing", "signal"),))


def test_network_rejects_edge_child_not_in_nodes() -> None:
    with pytest.raises(ValueError):
        _valid_network(edges=(("regime", "missing"),))


def test_network_rejects_self_loop_edge() -> None:
    with pytest.raises(ValueError):
        _valid_network(edges=(("regime", "regime"),))


def test_network_rejects_empty_cpd_digest() -> None:
    with pytest.raises(ValueError):
        _valid_network(cpd_digest="")


def test_network_rejects_oversize_cpd_digest() -> None:
    with pytest.raises(ValueError):
        _valid_network(cpd_digest="x" * (MAX_CPD_DIGEST_LEN + 1))


# ---------------------------------------------------------------------------
# BayesianInferenceArguments
# ---------------------------------------------------------------------------


def _valid_arguments(**overrides: Any) -> BayesianInferenceArguments:
    base: dict[str, Any] = {
        "inference_kind": BayesianInferenceKind.VARIABLE_ELIMINATION,
        "random_seed": 0,
        "query_variables": ("outcome",),
        "evidence": {"regime": "bull"},
        "n_samples": 1000,
    }
    base.update(overrides)
    return BayesianInferenceArguments(**base)


def test_arguments_is_frozen_and_slotted() -> None:
    a = _valid_arguments()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.random_seed = 1  # type: ignore[misc]
    assert not hasattr(a, "__dict__")


def test_arguments_rejects_non_enum_kind() -> None:
    with pytest.raises(TypeError):
        BayesianInferenceArguments(  # type: ignore[arg-type]
            inference_kind="VariableElimination",
            random_seed=0,
            query_variables=("a",),
        )


def test_arguments_rejects_bool_seed() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(random_seed=True)


def test_arguments_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(random_seed=-1)


def test_arguments_rejects_non_tuple_query_variables() -> None:
    with pytest.raises(TypeError):
        _valid_arguments(query_variables=["outcome"])  # type: ignore[arg-type]


def test_arguments_rejects_empty_query_variables() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(query_variables=())


def test_arguments_rejects_empty_query_var() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(query_variables=("",))


def test_arguments_rejects_duplicate_query_vars() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(query_variables=("outcome", "outcome"))


def test_arguments_rejects_empty_evidence_key() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(evidence={"": "v"})


def test_arguments_rejects_empty_evidence_value() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(evidence={"k": ""})


def test_arguments_rejects_below_min_samples() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_samples=0)


def test_arguments_rejects_above_max_samples() -> None:
    with pytest.raises(ValueError):
        _valid_arguments(n_samples=MAX_N_SAMPLES + 1)


# ---------------------------------------------------------------------------
# BayesianMarginalResult
# ---------------------------------------------------------------------------


def _valid_marginal(**overrides: Any) -> BayesianMarginalResult:
    base: dict[str, Any] = {
        "variable": "outcome",
        "states": ("up", "down"),
        "probabilities": (0.6, 0.4),
    }
    base.update(overrides)
    return BayesianMarginalResult(**base)


def test_marginal_is_frozen_and_slotted() -> None:
    m = _valid_marginal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.variable = "x"  # type: ignore[misc]
    assert not hasattr(m, "__dict__")


def test_marginal_rejects_empty_variable() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(variable="")


def test_marginal_rejects_non_tuple_states() -> None:
    with pytest.raises(TypeError):
        _valid_marginal(states=["up"])  # type: ignore[arg-type]


def test_marginal_rejects_empty_states() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(states=(), probabilities=())


def test_marginal_rejects_duplicate_states() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(states=("a", "a"), probabilities=(0.5, 0.5))


def test_marginal_rejects_above_max_states() -> None:
    states = tuple(f"s{i}" for i in range(MAX_STATES + 1))
    probs = tuple([1.0 / (MAX_STATES + 1)] * (MAX_STATES + 1))
    with pytest.raises(ValueError):
        _valid_marginal(states=states, probabilities=probs)


def test_marginal_rejects_states_probability_length_mismatch() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(states=("a", "b"), probabilities=(1.0,))


def test_marginal_rejects_nan_probability() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(
            states=("a", "b"),
            probabilities=(float("nan"), 0.5),
        )


def test_marginal_rejects_negative_probability() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(
            states=("a", "b"),
            probabilities=(-0.1, 1.1),
        )


def test_marginal_rejects_probability_above_one() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(
            states=("a", "b"),
            probabilities=(1.5, -0.5),
        )


def test_marginal_rejects_probabilities_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        _valid_marginal(
            states=("a", "b"),
            probabilities=(0.5, 0.4),
        )


# ---------------------------------------------------------------------------
# BayesianInferenceResult
# ---------------------------------------------------------------------------


def _valid_result(**overrides: Any) -> BayesianInferenceResult:
    base: dict[str, Any] = {"marginals": (_valid_marginal(),)}
    base.update(overrides)
    return BayesianInferenceResult(**base)


def test_result_is_frozen_and_slotted() -> None:
    r = _valid_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.marginals = ()  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


def test_result_rejects_non_tuple_marginals() -> None:
    with pytest.raises(TypeError):
        _valid_result(marginals=[_valid_marginal()])  # type: ignore[arg-type]


def test_result_rejects_empty_marginals() -> None:
    with pytest.raises(ValueError):
        _valid_result(marginals=())


def test_result_rejects_non_marginal_entry() -> None:
    with pytest.raises(TypeError):
        _valid_result(marginals=("bad",))  # type: ignore[arg-type]


def test_result_rejects_duplicate_variable_marginals() -> None:
    with pytest.raises(ValueError):
        _valid_result(
            marginals=(_valid_marginal(), _valid_marginal()),
        )


# ---------------------------------------------------------------------------
# BayesianInferenceRecord
# ---------------------------------------------------------------------------


def _valid_record(**overrides: Any) -> BayesianInferenceRecord:
    base: dict[str, Any] = {
        "ts_ns": 100,
        "analysis_id": "test_analysis",
        "source": ANALYSIS_SOURCE,
        "network": _valid_network(),
        "result": _valid_result(),
        "analysis_digest": "0123456789abcdef",
        "meta": {},
    }
    base.update(overrides)
    return BayesianInferenceRecord(**base)


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


def test_record_rejects_non_network() -> None:
    with pytest.raises(TypeError):
        _valid_record(network="bad")  # type: ignore[arg-type]


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
    """Deterministic :class:`BayesianInferenceEngine` fake."""

    __slots__ = ("_result",)

    def __init__(self, *, result: BayesianInferenceResult) -> None:
        self._result = result

    def infer(
        self,
        *,
        network: BayesianNetworkSpec,
        arguments: BayesianInferenceArguments,
        ts_ns: int,
        callback: BayesianInferenceCallback,
    ) -> BayesianInferenceResult:
        callback.on_inference_start(ts_ns=ts_ns, network=network, arguments=arguments)
        for m in self._result.marginals:
            callback.on_marginal_ready(ts_ns=ts_ns, marginal=m)
        return self._result


# ---------------------------------------------------------------------------
# PgmpyBayesianAnalyser.analyse end-to-end
# ---------------------------------------------------------------------------


def _analyser_inputs() -> tuple[BayesianNetworkSpec, BayesianInferenceArguments, _FakeEngine]:
    return (
        _valid_network(),
        _valid_arguments(),
        _FakeEngine(result=_valid_result()),
    )


def test_analyser_is_frozen_and_slotted() -> None:
    _, _, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(dataclasses.FrozenInstanceError):
        analyser.engine = None  # type: ignore[misc]
    assert not hasattr(analyser, "__dict__")


def test_analyser_rejects_non_engine() -> None:
    with pytest.raises(TypeError):
        PgmpyBayesianAnalyser(engine="bad")  # type: ignore[arg-type]


def test_analyse_emits_record() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    record = analyser.analyse(
        network=network,
        arguments=args,
        ts_ns=12345,
        analysis_id="analysis_0001",
    )
    assert isinstance(record, BayesianInferenceRecord)
    assert record.source == ANALYSIS_SOURCE
    assert record.analysis_id == "analysis_0001"
    assert record.ts_ns == 12345


def test_analyse_meta_includes_provenance() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    record = analyser.analyse(
        network=network,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["inference_kind"] == "VariableElimination"
    assert record.meta["random_seed"] == "0"
    assert record.meta["marginal_count"] == str(len(record.result.marginals))
    assert record.meta["node_count"] == str(len(network.nodes))
    assert record.meta["edge_count"] == str(len(network.edges))


def test_analyse_caller_meta_does_not_override_provenance() -> None:
    network, _, engine = _analyser_inputs()
    args = _valid_arguments(meta={"analysis_digest": "ZZ", "k": "v"})
    analyser = PgmpyBayesianAnalyser(engine=engine)
    record = analyser.analyse(
        network=network,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert record.meta["analysis_digest"] == record.analysis_digest
    assert record.meta["k"] == "v"


def test_analyse_rejects_non_network() -> None:
    _, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            network="bad",  # type: ignore[arg-type]
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_non_arguments() -> None:
    network, _, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            network=network,
            arguments="bad",  # type: ignore[arg-type]
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_bool_ts_ns() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=True,  # type: ignore[arg-type]
            analysis_id="a",
        )


def test_analyse_rejects_negative_ts_ns() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(BayesianAnalyserConfigError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=-1,
            analysis_id="a",
        )


def test_analyse_rejects_empty_analysis_id() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(BayesianAnalyserConfigError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=1,
            analysis_id="",
        )


def test_analyse_rejects_oversize_analysis_id() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(BayesianAnalyserConfigError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=1,
            analysis_id="x" * (MAX_ANALYSIS_ID_LEN + 1),
        )


def test_analyse_rejects_query_variable_not_in_network() -> None:
    network = _valid_network()
    args = _valid_arguments(query_variables=("missing",))
    analyser = PgmpyBayesianAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    with pytest.raises(BayesianAnalyserConfigError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_evidence_variable_not_in_network() -> None:
    network = _valid_network()
    args = _valid_arguments(evidence={"missing": "v"})
    analyser = PgmpyBayesianAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    with pytest.raises(BayesianAnalyserConfigError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_rejects_query_evidence_overlap() -> None:
    network = _valid_network()
    args = _valid_arguments(
        query_variables=("outcome",),
        evidence={"outcome": "up"},
    )
    analyser = PgmpyBayesianAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    with pytest.raises(BayesianAnalyserConfigError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


def test_analyse_uses_null_callback_by_default() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    record = analyser.analyse(
        network=network,
        arguments=args,
        ts_ns=1,
        analysis_id="a",
    )
    assert isinstance(record, BayesianInferenceRecord)


def test_analyse_rejects_non_protocol_callback() -> None:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    with pytest.raises(TypeError):
        analyser.analyse(
            network=network,
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
            network: BayesianNetworkSpec,
            arguments: BayesianInferenceArguments,
            ts_ns: int,
            callback: BayesianInferenceCallback,
        ) -> BayesianInferenceResult:
            return "bad"  # type: ignore[return-value]

    network, args, _ = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=_BadEngine())
    with pytest.raises(TypeError):
        analyser.analyse(
            network=network,
            arguments=args,
            ts_ns=1,
            analysis_id="a",
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _run_once() -> BayesianInferenceRecord:
    network, args, engine = _analyser_inputs()
    analyser = PgmpyBayesianAnalyser(engine=engine)
    return analyser.analyse(
        network=network,
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
    assert r1.network == r2.network == r3.network


def test_inv15_digest_changes_when_seed_changes() -> None:
    network = _valid_network()
    args_a = _valid_arguments(random_seed=0)
    args_b = _valid_arguments(random_seed=1)
    analyser = PgmpyBayesianAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    r0 = analyser.analyse(
        network=network,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = analyser.analyse(
        network=network,
        arguments=args_b,
        ts_ns=1,
        analysis_id="a",
    )
    assert r0.analysis_digest != r1.analysis_digest


def test_inv15_digest_changes_when_inference_kind_changes() -> None:
    network = _valid_network()
    args_a = _valid_arguments(inference_kind=BayesianInferenceKind.VARIABLE_ELIMINATION)
    args_b = _valid_arguments(inference_kind=BayesianInferenceKind.GIBBS_SAMPLING)
    analyser = PgmpyBayesianAnalyser(
        engine=_FakeEngine(result=_valid_result()),
    )
    r0 = analyser.analyse(
        network=network,
        arguments=args_a,
        ts_ns=1,
        analysis_id="a",
    )
    r1 = analyser.analyse(
        network=network,
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
# null_bayesian_inference_callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_bayesian_inference_callback()
    assert isinstance(cb, BayesianInferenceCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_bayesian_inference_callback()
    network = _valid_network()
    args = _valid_arguments()
    result = _valid_result()
    assert cb.on_inference_start(ts_ns=0, network=network, arguments=args) is None
    assert cb.on_marginal_ready(ts_ns=0, marginal=_valid_marginal()) is None
    assert cb.on_inference_end(ts_ns=0, result=result) is None


# ---------------------------------------------------------------------------
# Convenience factory raises when pgmpy missing
# ---------------------------------------------------------------------------


def test_pgmpy_engine_factory_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("pgmpy")
    except ImportError:
        with pytest.raises(ImportError, match="pgmpy"):
            pgmpy_variable_elimination_engine()
    else:
        pytest.skip("pgmpy installed — production seam smoke skipped")


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(__file__).resolve().parents[1] / "intelligence_engine" / "pgm_pgmpy.py"


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


def test_no_top_level_pgmpy_import() -> None:
    assert all(not name.startswith("pgmpy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_numpy_import() -> None:
    assert all(not name.startswith("numpy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_pandas_import() -> None:
    assert all(not name.startswith("pandas") for name in _top_level_imports(_module_ast()))


def test_no_top_level_networkx_import() -> None:
    assert all(not name.startswith("networkx") for name in _top_level_imports(_module_ast()))


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


def test_pgmpy_import_only_inside_factory() -> None:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith(("pgmpy", "networkx", "numpy", "pandas")):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside "
                        "pgmpy_variable_elimination_engine factory"
                    )
                    assert parent.name == ("pgmpy_variable_elimination_engine"), (
                        f"{name} imported in {parent.name!r} — must "
                        "be inside pgmpy_variable_elimination_engine"
                    )


# ---------------------------------------------------------------------------
# Module reload idempotency
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import intelligence_engine.pgm_pgmpy as mod1

    importlib.reload(mod1)
    import intelligence_engine.pgm_pgmpy as mod2

    assert mod1.ANALYSIS_SOURCE == mod2.ANALYSIS_SOURCE
    assert mod1.MAX_NODES == mod2.MAX_NODES
    assert (
        mod1.BayesianInferenceKind.VARIABLE_ELIMINATION
        is mod2.BayesianInferenceKind.VARIABLE_ELIMINATION
    )
