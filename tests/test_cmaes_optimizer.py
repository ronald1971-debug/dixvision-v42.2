"""Tests for evolution_engine/genetic/cmaes_optimizer.py (A-02.2).

Pinned constraints:
- OFFLINE-tier: no clock, no IO, no PRNG, no engine cross-imports.
- INV-13/14: governance isolation — produces PatchProposal, does not deploy.
- INV-15: byte-identical replay across 3 runs with identical inputs.
- AST authority pins: no top-level evotorch / numpy / torch imports.
- Frozen + slotted dataclasses; no __dict__.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from dataclasses import FrozenInstanceError

import pytest

from core.contracts.learning import PatchProposal
from evolution_engine.genetic.cmaes_optimizer import (
    DEFAULT_DRAWDOWN_PENALTY,
    DEFAULT_SIGMA_INIT,
    MAX_GENERATIONS,
    MAX_POPULATION_SIZE,
    MAX_PROPOSAL_ID_LEN,
    MAX_RATIONALE_LEN,
    MAX_TOTAL_EVALUATIONS,
    MIN_POPULATION_SIZE,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    CMAESCallback,
    CMAESConfig,
    CMAESConfigError,
    CMAESEvaluationError,
    CMAESOptimizer,
    CMAESResult,
    FitnessReport,
    GenerationReport,
    IndividualResult,
    null_cmaes_callback,
)
from evolution_engine.genetic.strategy_chromosome import (
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
)

MODULE_PATH = pathlib.Path("evolution_engine/genetic/cmaes_optimizer.py").resolve()
MODULE_TEXT = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _two_d_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(name="x", kind=ParameterKind.CONTINUOUS, low=-5.0, high=5.0),
        ParameterSpec(name="y", kind=ParameterKind.CONTINUOUS, low=-5.0, high=5.0),
    )


def _mixed_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(name="lr", kind=ParameterKind.LOG_CONTINUOUS, low=1e-4, high=1e-1),
        ParameterSpec(name="window", kind=ParameterKind.INTEGER, low=1.0, high=100.0),
        ParameterSpec(name="z", kind=ParameterKind.CONTINUOUS, low=-1.0, high=1.0),
    )


class SphereEvaluator:
    """Fitness = -(x^2 + y^2). Optimum at origin."""

    def evaluate(self, *, chromosome: StrategyChromosome, seed: int, ts_ns: int) -> FitnessReport:
        m = chromosome.to_mapping()
        loss = sum(v * v for v in m.values())
        return FitnessReport(pnl_mean=-loss, max_drawdown=0.0, n_samples=1)


class ConstantEvaluator:
    def __init__(self, scalar: float) -> None:
        self.scalar = scalar

    def evaluate(self, *, chromosome: StrategyChromosome, seed: int, ts_ns: int) -> FitnessReport:
        return FitnessReport(pnl_mean=self.scalar, max_drawdown=0.0, n_samples=1)


class CallCountingEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def evaluate(self, *, chromosome: StrategyChromosome, seed: int, ts_ns: int) -> FitnessReport:
        self.calls.append((chromosome.strategy_id, seed, ts_ns))
        return FitnessReport(pnl_mean=0.5, max_drawdown=0.1, n_samples=1)


class BadReturnEvaluator:
    def evaluate(self, *, chromosome: StrategyChromosome, seed: int, ts_ns: int) -> object:
        return "not a fitness report"


class NonFiniteEvaluator:
    """Evaluator returning a non-finite fitness scalar (via huge
    drawdown penalty interaction). FitnessReport itself is finite."""

    def evaluate(self, *, chromosome: StrategyChromosome, seed: int, ts_ns: int) -> FitnessReport:
        return FitnessReport(
            pnl_mean=float("1e308"),
            max_drawdown=float("1e308"),
            n_samples=1,
        )


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


def test_module_NEW_PIP_DEPENDENCIES_is_empty():
    assert NEW_PIP_DEPENDENCIES == ()


def test_module_PROPOSAL_SOURCE_namespace():
    assert PROPOSAL_SOURCE == "evolution_engine.genetic.cmaes_optimizer"


def test_module_constants_are_in_range():
    assert MIN_POPULATION_SIZE == 4
    assert MAX_POPULATION_SIZE >= MIN_POPULATION_SIZE
    assert MAX_GENERATIONS >= 1
    assert MAX_TOTAL_EVALUATIONS >= MIN_POPULATION_SIZE
    assert MAX_PROPOSAL_ID_LEN > 0
    assert MAX_RATIONALE_LEN > 0
    assert DEFAULT_SIGMA_INIT > 0.0
    assert DEFAULT_DRAWDOWN_PENALTY >= 0.0


def test_module_has_adapted_from_header():
    assert MODULE_TEXT.startswith("# ADAPTED FROM:")
    assert "evotorch" in MODULE_TEXT.split("\n")[0]


def test_module_has_a02_2_label():
    assert "A-02.2" in MODULE_TEXT


def test_module_has_inv_13_14_15_pins():
    assert "INV-13" in MODULE_TEXT or "INV-13/14" in MODULE_TEXT
    assert "INV-15" in MODULE_TEXT


# ---------------------------------------------------------------------------
# AST authority pins
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    return ast.parse(MODULE_TEXT)


def _iter_imports(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_ast_no_evotorch_import():
    for name in _iter_imports(_module_ast()):
        assert "evotorch" not in name, f"forbidden evotorch import: {name}"


def test_ast_no_numpy_import():
    for name in _iter_imports(_module_ast()):
        assert "numpy" not in name, f"forbidden numpy import: {name}"


def test_ast_no_torch_import():
    for name in _iter_imports(_module_ast()):
        assert "torch" not in name, f"forbidden torch import: {name}"


def test_ast_no_clock_imports():
    forbidden = {"time", "datetime", "calendar"}
    for name in _iter_imports(_module_ast()):
        head = name.split(".")[0]
        assert head not in forbidden, f"forbidden clock import: {name}"


def test_ast_no_random_imports():
    for name in _iter_imports(_module_ast()):
        head = name.split(".")[0]
        assert head not in {"random", "secrets", "os"}, f"forbidden PRNG/IO import: {name}"


def test_ast_no_engine_cross_imports():
    forbidden_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
        "cockpit.",
        "dashboard.",
    )
    for name in _iter_imports(_module_ast()):
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix.rstrip(".")), f"forbidden engine import: {name}"
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), f"forbidden engine import: {name}"


def test_ast_no_clock_text_in_module():
    forbidden = (
        "time.time",
        "time.monotonic",
        "datetime.now",
        "datetime.utcnow",
        "perf_counter",
        "monotonic_ns",
        "time_ns",
    )
    for token in forbidden:
        assert token not in MODULE_TEXT, f"forbidden clock call: {token}"


def test_ast_imports_only_stdlib_plus_strategy_chromosome():
    permitted_thirdparty: set[str] = set()  # zero non-stdlib deps
    permitted_local = {
        "core.contracts.learning",
        "evolution_engine.genetic.strategy_chromosome",
    }
    stdlib = {
        "__future__",
        "dataclasses",
        "hashlib",
        "math",
        "collections.abc",
        "typing",
    }
    for name in _iter_imports(_module_ast()):
        if not name:
            continue
        head = name.split(".")[0]
        if name in permitted_local:
            continue
        if name in stdlib or head in {
            "dataclasses",
            "hashlib",
            "math",
            "collections",
            "typing",
            "__future__",
        }:
            continue
        if head in permitted_thirdparty:
            continue
        raise AssertionError(f"unexpected import: {name}")


# ---------------------------------------------------------------------------
# CMAESConfig validation
# ---------------------------------------------------------------------------


def test_config_happy_path():
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=10,
    )
    assert cfg.target_strategy_id == "alpha"
    assert cfg.population_size == 8
    assert cfg.max_generations == 10
    assert cfg.sigma_init == DEFAULT_SIGMA_INIT
    assert cfg.fitness_drawdown_weight == DEFAULT_DRAWDOWN_PENALTY


def test_config_is_frozen():
    cfg = CMAESConfig(target_strategy_id="alpha", population_size=8, max_generations=10)
    with pytest.raises(FrozenInstanceError):
        cfg.population_size = 16  # type: ignore[misc]


def test_config_is_slotted():
    cfg = CMAESConfig(target_strategy_id="alpha", population_size=8, max_generations=10)
    assert not hasattr(cfg, "__dict__")


def test_config_rejects_non_str_target_strategy_id():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id=123,  # type: ignore[arg-type]
            population_size=8,
            max_generations=10,
        )


def test_config_rejects_empty_target_strategy_id():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(target_strategy_id="", population_size=8, max_generations=10)


def test_config_rejects_population_below_min():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=MIN_POPULATION_SIZE - 1,
            max_generations=10,
        )


def test_config_rejects_population_above_max():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=MAX_POPULATION_SIZE + 1,
            max_generations=1,
        )


def test_config_rejects_bool_population():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=True,  # type: ignore[arg-type]
            max_generations=10,
        )


def test_config_rejects_max_generations_zero():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=0,
        )


def test_config_rejects_max_generations_above_cap():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=MAX_GENERATIONS + 1,
        )


def test_config_rejects_total_evaluations_blowout():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=MAX_POPULATION_SIZE,
            max_generations=MAX_GENERATIONS,
        )


def test_config_rejects_non_positive_sigma_init():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=10,
            sigma_init=0.0,
        )


def test_config_rejects_non_finite_sigma_init():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=10,
            sigma_init=float("inf"),
        )


def test_config_rejects_negative_drawdown_weight():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=10,
            fitness_drawdown_weight=-0.1,
        )


def test_config_rejects_non_finite_drawdown_weight():
    with pytest.raises(CMAESConfigError):
        CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=10,
            fitness_drawdown_weight=float("nan"),
        )


# ---------------------------------------------------------------------------
# FitnessReport validation
# ---------------------------------------------------------------------------


def test_fitness_report_happy_path():
    r = FitnessReport(pnl_mean=1.0, max_drawdown=0.5, n_samples=10)
    assert r.pnl_mean == 1.0
    assert r.fitness(0.5) == 1.0 - 0.25


def test_fitness_report_is_frozen():
    r = FitnessReport(pnl_mean=1.0, max_drawdown=0.0, n_samples=1)
    with pytest.raises(FrozenInstanceError):
        r.pnl_mean = 2.0  # type: ignore[misc]


def test_fitness_report_is_slotted():
    r = FitnessReport(pnl_mean=1.0, max_drawdown=0.0, n_samples=1)
    assert not hasattr(r, "__dict__")


def test_fitness_report_rejects_non_finite_pnl():
    with pytest.raises(CMAESEvaluationError):
        FitnessReport(pnl_mean=float("nan"), max_drawdown=0.0, n_samples=1)


def test_fitness_report_rejects_negative_drawdown():
    with pytest.raises(CMAESEvaluationError):
        FitnessReport(pnl_mean=1.0, max_drawdown=-0.1, n_samples=1)


def test_fitness_report_rejects_non_finite_drawdown():
    with pytest.raises(CMAESEvaluationError):
        FitnessReport(pnl_mean=1.0, max_drawdown=float("inf"), n_samples=1)


def test_fitness_report_rejects_zero_n_samples():
    with pytest.raises(CMAESEvaluationError):
        FitnessReport(pnl_mean=1.0, max_drawdown=0.0, n_samples=0)


def test_fitness_report_rejects_bool_n_samples():
    with pytest.raises(CMAESEvaluationError):
        FitnessReport(
            pnl_mean=1.0,
            max_drawdown=0.0,
            n_samples=True,  # type: ignore[arg-type]
        )


def test_fitness_report_rejects_non_str_meta_keys():
    with pytest.raises(CMAESEvaluationError):
        FitnessReport(
            pnl_mean=1.0,
            max_drawdown=0.0,
            n_samples=1,
            meta={1: "v"},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# IndividualResult / GenerationReport validation
# ---------------------------------------------------------------------------


def _toy_individual(scalar: float = 0.0, gen: int = 0) -> IndividualResult:
    specs = _two_d_specs()
    chrom = StrategyChromosome(
        strategy_id="alpha",
        specs=specs,
        values=(0.0, 0.0),
        version=gen,
    )
    return IndividualResult(
        chromosome=chrom,
        fitness_report=FitnessReport(pnl_mean=scalar, max_drawdown=0.0, n_samples=1),
        fitness_scalar=scalar,
        generation_idx=gen,
    )


def test_individual_result_is_frozen_and_slotted():
    ind = _toy_individual()
    with pytest.raises(FrozenInstanceError):
        ind.fitness_scalar = 9.0  # type: ignore[misc]
    assert not hasattr(ind, "__dict__")


def test_individual_result_rejects_non_finite_scalar():
    specs = _two_d_specs()
    chrom = StrategyChromosome(strategy_id="alpha", specs=specs, values=(0.0, 0.0), version=0)
    with pytest.raises(ValueError):
        IndividualResult(
            chromosome=chrom,
            fitness_report=FitnessReport(pnl_mean=0.0, max_drawdown=0.0, n_samples=1),
            fitness_scalar=float("nan"),
            generation_idx=0,
        )


def test_individual_result_rejects_negative_generation():
    specs = _two_d_specs()
    chrom = StrategyChromosome(strategy_id="alpha", specs=specs, values=(0.0, 0.0), version=0)
    with pytest.raises(ValueError):
        IndividualResult(
            chromosome=chrom,
            fitness_report=FitnessReport(pnl_mean=0.0, max_drawdown=0.0, n_samples=1),
            fitness_scalar=0.0,
            generation_idx=-1,
        )


def test_generation_report_rejects_empty_individuals():
    ind = _toy_individual(scalar=1.0)
    with pytest.raises(ValueError):
        GenerationReport(
            generation_idx=0,
            individuals=(),
            best_fitness=1.0,
            mean_fitness=1.0,
            best_individual=ind,
        )


def test_generation_report_is_frozen():
    ind = _toy_individual(scalar=1.0)
    rep = GenerationReport(
        generation_idx=0,
        individuals=(ind,),
        best_fitness=1.0,
        mean_fitness=1.0,
        best_individual=ind,
    )
    with pytest.raises(FrozenInstanceError):
        rep.generation_idx = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CMAESResult validation
# ---------------------------------------------------------------------------


def _build_minimal_result() -> CMAESResult:
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    return opt.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )


def test_result_is_frozen_and_slotted():
    res = _build_minimal_result()
    with pytest.raises(FrozenInstanceError):
        res.policy_digest = "x" * 16  # type: ignore[misc]
    assert not hasattr(res, "__dict__")


def test_result_policy_digest_is_16_hex_chars():
    res = _build_minimal_result()
    assert len(res.policy_digest) == 16
    assert all(c in "0123456789abcdef" for c in res.policy_digest)


def test_result_proposal_is_patch_proposal():
    res = _build_minimal_result()
    assert isinstance(res.proposal, PatchProposal)
    assert res.proposal.source == PROPOSAL_SOURCE
    assert res.proposal.target_strategy == "alpha"
    assert res.proposal.patch_id == "p"


def test_result_proposal_touchpoints_match_specs():
    res = _build_minimal_result()
    assert res.proposal.touchpoints == ("x", "y")


def test_result_proposal_meta_carries_digest_and_seed():
    res = _build_minimal_result()
    assert res.proposal.meta["policy_digest"] == res.policy_digest
    assert res.proposal.meta["seed"] == "1"
    assert res.proposal.meta["proposal_id"] == "p"


def test_result_generations_count_matches_config():
    res = _build_minimal_result()
    assert len(res.generations) == 2


# ---------------------------------------------------------------------------
# CMAESOptimizer validation
# ---------------------------------------------------------------------------


def test_optimizer_is_frozen_and_slotted():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    with pytest.raises(FrozenInstanceError):
        opt.evaluator = SphereEvaluator()  # type: ignore[misc]
    assert not hasattr(opt, "__dict__")


def test_optimizer_rejects_non_evaluator():
    class NotEvaluator:
        pass

    with pytest.raises(TypeError):
        CMAESOptimizer(evaluator=NotEvaluator())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# evolve() argument validation
# ---------------------------------------------------------------------------


def _basic_kwargs():
    return dict(
        specs=_two_d_specs(),
        config=CMAESConfig(
            target_strategy_id="alpha",
            population_size=4,
            max_generations=2,
        ),
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )


def test_evolve_rejects_non_tuple_specs():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["specs"] = list(_two_d_specs())  # type: ignore[assignment]
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_empty_specs():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["specs"] = ()
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_non_int_seed():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["seed"] = "1"  # type: ignore[assignment]
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_negative_seed():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["seed"] = -1
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_bool_seed():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["seed"] = True  # type: ignore[assignment]
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_negative_ts_ns():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["ts_ns"] = -1
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_empty_proposal_id():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["proposal_id"] = ""
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_oversized_proposal_id():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    kwargs = _basic_kwargs()
    kwargs["proposal_id"] = "x" * (MAX_PROPOSAL_ID_LEN + 1)
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_initial_chromosome_with_wrong_specs():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    other_specs = (ParameterSpec(name="q", kind=ParameterKind.CONTINUOUS, low=0.0, high=1.0),)
    bad = StrategyChromosome(
        strategy_id="alpha",
        specs=other_specs,
        values=(0.5,),
        version=0,
    )
    kwargs = _basic_kwargs()
    kwargs["initial_chromosome"] = bad
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_rejects_initial_chromosome_with_wrong_strategy_id():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    specs = _two_d_specs()
    bad = StrategyChromosome(
        strategy_id="other",
        specs=specs,
        values=(0.0, 0.0),
        version=0,
    )
    kwargs = _basic_kwargs()
    kwargs["initial_chromosome"] = bad
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


def test_evolve_accepts_initial_chromosome_aligned_with_config():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    specs = _two_d_specs()
    init = StrategyChromosome(
        strategy_id="alpha",
        specs=specs,
        values=(1.5, -2.0),
        version=0,
    )
    kwargs = _basic_kwargs()
    kwargs["initial_chromosome"] = init
    res = opt.evolve(**kwargs)
    assert isinstance(res, CMAESResult)


def test_evolve_rejects_non_callback_object():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())

    class Bogus:
        pass

    kwargs = _basic_kwargs()
    kwargs["callback"] = Bogus()  # type: ignore[arg-type]
    with pytest.raises(CMAESConfigError):
        opt.evolve(**kwargs)


# ---------------------------------------------------------------------------
# Convergence + correctness
# ---------------------------------------------------------------------------


def test_evolve_converges_on_sphere():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=12,
        max_generations=30,
        sigma_init=2.0,
    )
    res = opt.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=42,
        ts_ns=1,
        proposal_id="run-1",
    )
    best = res.best_individual.chromosome.to_mapping()
    # Tolerant convergence threshold: after 30 generations from
    # sigma=2 on a 2D sphere, the best ought to be within 0.1 of
    # the optimum.
    assert abs(best["x"]) < 0.1
    assert abs(best["y"]) < 0.1


def test_evolve_individuals_in_each_generation_match_population_size():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=3,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen in res.generations:
        assert len(gen.individuals) == 8


def test_evolve_individuals_sorted_fittest_first():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=3,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen in res.generations:
        scalars = [ind.fitness_scalar for ind in gen.individuals]
        assert scalars == sorted(scalars, reverse=True)


def test_evolve_running_best_is_monotonic_non_decreasing():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=10,
        sigma_init=2.0,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    bests = [gen.best_individual.fitness_scalar for gen in res.generations]
    for prev, curr in zip(bests, bests[1:], strict=False):
        assert curr >= prev


def test_evolve_constant_evaluator_finishes():
    opt = CMAESOptimizer(evaluator=ConstantEvaluator(scalar=0.7))
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=3,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen in res.generations:
        for ind in gen.individuals:
            assert ind.fitness_scalar == 0.7


def test_evolve_evaluator_seed_is_unique_per_individual():
    eval_ = CallCountingEvaluator()
    opt = CMAESOptimizer(evaluator=eval_)
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=3,
    )
    opt.evolve(specs=_two_d_specs(), config=cfg, seed=42, ts_ns=1, proposal_id="p")
    seeds = [seed for _, seed, _ in eval_.calls]
    assert len(set(seeds)) == len(seeds)


def test_evolve_evaluator_ts_ns_is_passed_through():
    eval_ = CallCountingEvaluator()
    opt = CMAESOptimizer(evaluator=eval_)
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    opt.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=987654321,
        proposal_id="p",
    )
    for _, _, ts in eval_.calls:
        assert ts == 987654321


def test_evolve_rejects_non_fitness_report_return():
    opt = CMAESOptimizer(evaluator=BadReturnEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    with pytest.raises(CMAESEvaluationError):
        opt.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=1,
            ts_ns=1,
            proposal_id="p",
        )


def test_evolve_rejects_non_finite_fitness_scalar_after_drawdown_combo():
    opt = CMAESOptimizer(evaluator=NonFiniteEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
        fitness_drawdown_weight=1e308,
    )
    with pytest.raises(CMAESEvaluationError):
        opt.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=1,
            ts_ns=1,
            proposal_id="p",
        )


# ---------------------------------------------------------------------------
# Mixed-kind + bounds handling
# ---------------------------------------------------------------------------


def test_evolve_mixed_kinds_yields_feasible_chromosomes():
    opt = CMAESOptimizer(evaluator=ConstantEvaluator(scalar=1.0))
    specs = _mixed_specs()
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=4,
        sigma_init=2.0,
    )
    res = opt.evolve(specs=specs, config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen in res.generations:
        for ind in gen.individuals:
            m = ind.chromosome.to_mapping()
            assert 1e-4 <= m["lr"] <= 1e-1
            assert 1.0 <= m["window"] <= 100.0
            assert int(m["window"]) == m["window"]
            assert -1.0 <= m["z"] <= 1.0


def test_evolve_with_initial_chromosome_starts_at_centroid():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    specs = _two_d_specs()
    init = StrategyChromosome(
        strategy_id="alpha",
        specs=specs,
        values=(2.0, -3.0),
        version=0,
    )
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=1,
        sigma_init=0.001,  # tiny — barely move
    )
    res = opt.evolve(
        specs=specs,
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
        initial_chromosome=init,
    )
    # First-generation samples should cluster near (2, -3).
    for ind in res.generations[0].individuals:
        m = ind.chromosome.to_mapping()
        assert abs(m["x"] - 2.0) < 0.05
        assert abs(m["y"] - (-3.0)) < 0.05


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def test_inv15_three_run_identical_digest():
    digests = []
    for _ in range(3):
        opt = CMAESOptimizer(evaluator=SphereEvaluator())
        cfg = CMAESConfig(
            target_strategy_id="alpha",
            population_size=8,
            max_generations=5,
            sigma_init=1.0,
        )
        res = opt.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=99,
            ts_ns=42,
            proposal_id="p",
        )
        digests.append(res.policy_digest)
    assert len(set(digests)) == 1


def test_inv15_three_run_identical_proposal_meta():
    metas = []
    for _ in range(3):
        opt = CMAESOptimizer(evaluator=SphereEvaluator())
        cfg = CMAESConfig(
            target_strategy_id="alpha",
            population_size=6,
            max_generations=4,
        )
        res = opt.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=7,
            ts_ns=11,
            proposal_id="p",
        )
        metas.append(dict(res.proposal.meta))
    assert metas[0] == metas[1] == metas[2]


def test_inv15_three_run_identical_chromosome_digests():
    chains = []
    for _ in range(3):
        opt = CMAESOptimizer(evaluator=SphereEvaluator())
        cfg = CMAESConfig(
            target_strategy_id="alpha",
            population_size=6,
            max_generations=4,
        )
        res = opt.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=7,
            ts_ns=11,
            proposal_id="p",
        )
        chains.append(
            tuple(
                tuple(chromosome_digest(ind.chromosome) for ind in gen.individuals)
                for gen in res.generations
            )
        )
    assert chains[0] == chains[1] == chains[2]


def test_inv15_different_seeds_yield_different_digests():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=8,
        max_generations=5,
    )
    res_a = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    res_b = opt.evolve(specs=_two_d_specs(), config=cfg, seed=2, ts_ns=1, proposal_id="p")
    assert res_a.policy_digest != res_b.policy_digest


def test_inv15_different_ts_ns_change_digest():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    res_a = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    res_b = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=2, proposal_id="p")
    assert res_a.policy_digest != res_b.policy_digest


def test_inv15_different_proposal_id_change_digest():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    res_a = opt.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p1",
    )
    res_b = opt.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p2",
    )
    assert res_a.policy_digest != res_b.policy_digest


# ---------------------------------------------------------------------------
# Callback Protocol behavior
# ---------------------------------------------------------------------------


def test_null_callback_is_a_callback():
    cb = null_cmaes_callback()
    assert isinstance(cb, CMAESCallback)


def test_null_callback_methods_are_no_ops():
    cb = null_cmaes_callback()
    cfg = CMAESConfig(target_strategy_id="alpha", population_size=4, max_generations=1)
    cb.on_evolution_start(ts_ns=1, config=cfg, dimensionality=2)
    # Should not raise; that's the whole contract.


class RecordingCallback:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_evolution_start(self, *, ts_ns: int, config: CMAESConfig, dimensionality: int) -> None:
        self.events.append(f"start:{dimensionality}")

    def on_individual_evaluated(
        self,
        *,
        ts_ns: int,
        generation_idx: int,
        individual_idx: int,
        chromosome: StrategyChromosome,
        fitness_report: FitnessReport,
    ) -> None:
        self.events.append(f"ind:{generation_idx}:{individual_idx}")

    def on_generation_end(self, *, ts_ns: int, report: GenerationReport) -> None:
        self.events.append(f"gen:{report.generation_idx}")

    def on_evolution_end(self, *, ts_ns: int, result: CMAESResult) -> None:
        self.events.append("end")


def test_callback_lifecycle_is_called_in_order():
    cb = RecordingCallback()
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    opt.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
        callback=cb,
    )
    assert cb.events[0] == "start:2"
    assert cb.events[-1] == "end"
    # 4 individuals × 2 generations = 8 individual events
    assert sum(1 for e in cb.events if e.startswith("ind:")) == 8
    assert sum(1 for e in cb.events if e.startswith("gen:")) == 2


# ---------------------------------------------------------------------------
# StrategyChromosome shape inside generations
# ---------------------------------------------------------------------------


def test_chromosomes_carry_target_strategy_id():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen in res.generations:
        for ind in gen.individuals:
            assert ind.chromosome.strategy_id == "alpha"


def test_chromosomes_version_matches_generation_idx():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=3,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen_idx, gen in enumerate(res.generations):
        for ind in gen.individuals:
            assert ind.chromosome.version == gen_idx


def test_chromosomes_meta_carries_generation_and_individual():
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=2,
    )
    res = opt.evolve(specs=_two_d_specs(), config=cfg, seed=1, ts_ns=1, proposal_id="p")
    for gen in res.generations:
        for ind in gen.individuals:
            assert "generation" in ind.chromosome.meta
            assert "individual" in ind.chromosome.meta


# ---------------------------------------------------------------------------
# Proposal rationale + meta
# ---------------------------------------------------------------------------


def test_proposal_rationale_within_cap():
    res = _build_minimal_result()
    assert len(res.proposal.rationale) <= MAX_RATIONALE_LEN


def test_proposal_rationale_mentions_chromosome_digest():
    res = _build_minimal_result()
    assert chromosome_digest(res.best_individual.chromosome) in (res.proposal.rationale)


def test_proposal_meta_keys_are_strs():
    res = _build_minimal_result()
    for k, v in res.proposal.meta.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_proposal_meta_contains_required_audit_fields():
    res = _build_minimal_result()
    expected = {
        "policy_digest",
        "seed",
        "proposal_id",
        "population_size",
        "max_generations",
        "generation_count",
        "best_chromosome_digest",
        "best_fitness",
        "best_pnl_mean",
        "best_max_drawdown",
    }
    assert expected <= set(res.proposal.meta.keys())


# ---------------------------------------------------------------------------
# OFFLINE_ONLY tier — module imports cleanly without optional deps
# ---------------------------------------------------------------------------


def test_module_imports_without_evotorch_or_numpy(monkeypatch: pytest.MonkeyPatch):
    """Re-import the module after blocking evotorch/numpy/torch from
    sys.modules. Must succeed (lazy / never imports them)."""

    import sys

    saved = {}
    for blocked in ("evotorch", "numpy", "torch"):
        saved[blocked] = sys.modules.pop(blocked, None)
        sys.modules[blocked] = None  # type: ignore[assignment]
    try:
        # Force reimport.
        if "evolution_engine.genetic.cmaes_optimizer" in sys.modules:
            del sys.modules["evolution_engine.genetic.cmaes_optimizer"]
        importlib.import_module("evolution_engine.genetic.cmaes_optimizer")
    finally:
        for blocked, prev in saved.items():
            if prev is None:
                sys.modules.pop(blocked, None)
            else:
                sys.modules[blocked] = prev


# ---------------------------------------------------------------------------
# Boundary / extreme inputs
# ---------------------------------------------------------------------------


def test_evolve_handles_high_dimensional_specs():
    specs = tuple(
        ParameterSpec(
            name=f"p{i}",
            kind=ParameterKind.CONTINUOUS,
            low=-1.0,
            high=1.0,
        )
        for i in range(20)
    )
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=12,
        max_generations=3,
    )
    res = opt.evolve(specs=specs, config=cfg, seed=1, ts_ns=1, proposal_id="p")
    assert res.proposal.touchpoints == tuple(f"p{i}" for i in range(20))


def test_evolve_single_axis_specs_works():
    specs = (
        ParameterSpec(
            name="solo",
            kind=ParameterKind.CONTINUOUS,
            low=-3.0,
            high=3.0,
        ),
    )
    opt = CMAESOptimizer(evaluator=SphereEvaluator())
    cfg = CMAESConfig(
        target_strategy_id="alpha",
        population_size=4,
        max_generations=20,
        sigma_init=2.0,
    )
    res = opt.evolve(specs=specs, config=cfg, seed=1, ts_ns=1, proposal_id="p")
    assert abs(res.best_individual.chromosome.to_mapping()["solo"]) < 0.1
