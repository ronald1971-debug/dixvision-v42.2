"""Tests for evolution_engine/pipeline.py (A-04.2).

Pinned constraints:
- OFFLINE-tier: no clock, no IO, no engine cross-imports, no top-level
  nevergrad / numpy / torch.
- INV-13/14: governance isolation — emits PatchProposal, never deploys.
- INV-15: byte-identical replay across 3 runs with identical inputs.
- AST authority pins.
- Frozen + slotted dataclasses; no __dict__.
- B27 / B28 / INV-71: only ``evolution_engine.*`` may construct
  PatchProposal — and this module IS in ``evolution_engine.*``.
"""

from __future__ import annotations

import ast
import importlib
import math
import pathlib
from dataclasses import FrozenInstanceError, dataclass, field

import pytest

from core.contracts.learning import PatchProposal
from evolution_engine.genetic.strategy_chromosome import (
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
    pack,
    unpack,
)
from evolution_engine.pipeline import (
    DEFAULT_CR,
    DEFAULT_DRAWDOWN_PENALTY,
    DEFAULT_F,
    DEFAULT_INIT_SIGMA,
    MAX_GENERATIONS,
    MAX_POPULATION_SIZE,
    MAX_PROPOSAL_ID_LEN,
    MAX_RATIONALE_LEN,
    MAX_TOTAL_EVALUATIONS,
    MIN_POPULATION_SIZE,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    EvolutionPipeline,
    EvolutionPipelineCallback,
    EvolutionPipelineConfig,
    EvolutionPipelineError,
    EvolutionPipelineEvaluationError,
    EvolutionPipelineResult,
    FitnessEvaluator,  # noqa: F401  (re-exported, surface-pinned below)
    FitnessReport,
    GenerationReport,
    IndividualResult,
    MutationStrategy,
    null_evolution_pipeline_callback,
)

MODULE_PATH = pathlib.Path("evolution_engine/pipeline.py").resolve()
MODULE_TEXT = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers + fakes
# ---------------------------------------------------------------------------


def _two_d_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(name="x", kind=ParameterKind.CONTINUOUS, low=-5.0, high=5.0),
        ParameterSpec(name="y", kind=ParameterKind.CONTINUOUS, low=-5.0, high=5.0),
    )


def _mixed_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(
            name="lr",
            kind=ParameterKind.LOG_CONTINUOUS,
            low=1e-4,
            high=1e-1,
        ),
        ParameterSpec(
            name="window",
            kind=ParameterKind.INTEGER,
            low=1.0,
            high=100.0,
        ),
        ParameterSpec(
            name="threshold",
            kind=ParameterKind.CONTINUOUS,
            low=-1.0,
            high=1.0,
        ),
    )


def _seed_chromosome(
    *,
    target_strategy_id: str = "strat-001",
    specs: tuple[ParameterSpec, ...] | None = None,
    values: tuple[float, ...] | None = None,
) -> StrategyChromosome:
    use_specs = specs if specs is not None else _two_d_specs()
    if values is None:
        out: list[float] = []
        for s in use_specs:
            mid = 0.5 * (s.low + s.high)
            if s.kind is ParameterKind.INTEGER:
                mid = float(round(mid))
            out.append(mid)
        values = tuple(out)
    return StrategyChromosome(
        strategy_id=target_strategy_id,
        specs=use_specs,
        values=values,
        version=0,
        meta={},
    )


@dataclass(frozen=True, slots=True)
class _SphereEvaluator:
    """Fitness = -||encoded_values||^2. Optimum is the all-zero
    encoded vector (i.e. encoded midpoint, decoded back via
    :func:`unpack`)."""

    drawdown_floor: float = 0.0

    def evaluate(
        self,
        *,
        chromosome: StrategyChromosome,
        seed: int,
        ts_ns: int,
    ) -> FitnessReport:
        encoded = pack(chromosome.specs, chromosome.to_mapping())
        sq = 0.0
        for v in encoded:
            sq += v * v
        return FitnessReport(
            pnl_mean=-sq,
            max_drawdown=self.drawdown_floor,
            n_samples=10,
            meta={"squared_norm": repr(sq)},
        )


@dataclass(frozen=True, slots=True)
class _ConstantEvaluator:
    pnl: float = 0.0
    drawdown: float = 0.0

    def evaluate(
        self,
        *,
        chromosome: StrategyChromosome,
        seed: int,
        ts_ns: int,
    ) -> FitnessReport:
        return FitnessReport(
            pnl_mean=self.pnl,
            max_drawdown=self.drawdown,
            n_samples=1,
            meta={},
        )


@dataclass(frozen=True, slots=True)
class _BadEvaluatorReturnType:
    def evaluate(self, *, chromosome, seed, ts_ns):  # type: ignore[no-untyped-def]
        return "not a fitness report"


@dataclass(frozen=True, slots=True)
class _RecordingEvaluator:
    """Wraps a sphere evaluator and records every call key.

    The recording list itself is mutable but the dataclass is frozen,
    so we expose it via a default_factory list that callers append
    to. This is fine for tests — replay determinism is provided by
    the deterministic call order, not by evaluator immutability.
    """

    inner: _SphereEvaluator
    calls: list[tuple[int, int, str]] = field(default_factory=list)

    def evaluate(
        self,
        *,
        chromosome: StrategyChromosome,
        seed: int,
        ts_ns: int,
    ) -> FitnessReport:
        self.calls.append((seed, ts_ns, chromosome_digest(chromosome)))
        return self.inner.evaluate(chromosome=chromosome, seed=seed, ts_ns=ts_ns)


def _make_config(
    *,
    target_strategy_id: str = "strat-001",
    population_size: int = 6,
    max_generations: int = 4,
    mutation_strategy: MutationStrategy = MutationStrategy.DE_RAND_1,
    F: float = DEFAULT_F,
    CR: float = DEFAULT_CR,
    init_sigma: float = DEFAULT_INIT_SIGMA,
    fitness_drawdown_weight: float = DEFAULT_DRAWDOWN_PENALTY,
) -> EvolutionPipelineConfig:
    return EvolutionPipelineConfig(
        target_strategy_id=target_strategy_id,
        population_size=population_size,
        max_generations=max_generations,
        mutation_strategy=mutation_strategy,
        F=F,
        CR=CR,
        init_sigma=init_sigma,
        fitness_drawdown_weight=fitness_drawdown_weight,
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_proposal_source_pinned() -> None:
    assert PROPOSAL_SOURCE == "evolution_engine.pipeline"


def test_min_population_size_is_four() -> None:
    assert MIN_POPULATION_SIZE == 4


def test_max_population_size_is_256() -> None:
    assert MAX_POPULATION_SIZE == 256


def test_max_generations_is_1000() -> None:
    assert MAX_GENERATIONS == 1000


def test_max_total_evaluations_is_100_000() -> None:
    assert MAX_TOTAL_EVALUATIONS == 100_000


def test_max_proposal_id_len_is_256() -> None:
    assert MAX_PROPOSAL_ID_LEN == 256


def test_max_rationale_len_is_1024() -> None:
    assert MAX_RATIONALE_LEN == 1024


def test_default_constants_are_sane() -> None:
    assert 0.0 < DEFAULT_F <= 2.0
    assert 0.0 <= DEFAULT_CR <= 1.0
    assert 0.0 < DEFAULT_INIT_SIGMA <= 1.0
    assert 0.0 <= DEFAULT_DRAWDOWN_PENALTY <= 10.0


# ---------------------------------------------------------------------------
# AST pins
# ---------------------------------------------------------------------------


def test_ast_no_top_level_runtime_imports() -> None:
    forbidden = {
        "nevergrad",
        "numpy",
        "torch",
        "scipy",
        "evotorch",
        "deap",
        "gymnasium",
        "stable_baselines3",
        "polars",
        "datetime",
        "time",
        "os",
        "random",
        "asyncio",
        "websockets",
        "litellm",
    }
    tree = ast.parse(MODULE_TEXT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"Forbidden top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            assert root not in forbidden, f"Forbidden from-import: {node.module}"


def test_ast_no_engine_cross_imports() -> None:
    forbidden = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
        "ui",
        "dashboard_backend",
        "learning_engine",
        "simulation",
        "sensory",
        "agents",
    )
    tree = ast.parse(MODULE_TEXT)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.split(".")[0]
            assert root not in forbidden, f"Forbidden engine cross-import: {node.module}"


def test_ast_adapted_from_header_present() -> None:
    head = MODULE_TEXT.splitlines()[:6]
    assert any("ADAPTED FROM" in line for line in head), head
    assert any("nevergrad" in line for line in head), head


def test_ast_no_clock_or_random_calls() -> None:
    """No raw clock / no module-level random.* / no os.urandom."""

    forbidden = {
        ("time", "time"),
        ("time", "monotonic"),
        ("time", "perf_counter"),
        ("time", "time_ns"),
        ("time", "monotonic_ns"),
        ("random", "random"),
        ("random", "uniform"),
        ("random", "gauss"),
        ("random", "randint"),
        ("os", "urandom"),
    }
    tree = ast.parse(MODULE_TEXT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in forbidden, (
                f"Forbidden raw-clock/random call: "
                f"{node.value.id}.{node.attr} on line {node.lineno}"
            )


# ---------------------------------------------------------------------------
# Frozen / slotted dataclasses
# ---------------------------------------------------------------------------


def test_config_is_frozen_and_slotted() -> None:
    cfg = _make_config()
    assert not hasattr(cfg, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cfg.population_size = 8  # type: ignore[misc]


def test_fitness_report_is_frozen_and_slotted() -> None:
    fr = FitnessReport(pnl_mean=1.0, max_drawdown=0.0, n_samples=1)
    assert not hasattr(fr, "__dict__")
    with pytest.raises(FrozenInstanceError):
        fr.pnl_mean = 2.0  # type: ignore[misc]


def test_individual_result_is_frozen_and_slotted() -> None:
    fr = FitnessReport(pnl_mean=0.0, max_drawdown=0.0, n_samples=1)
    ind = IndividualResult(
        chromosome=_seed_chromosome(),
        fitness_report=fr,
        fitness_scalar=0.0,
        generation_idx=0,
    )
    assert not hasattr(ind, "__dict__")
    with pytest.raises(FrozenInstanceError):
        ind.fitness_scalar = 1.0  # type: ignore[misc]


def test_generation_report_is_frozen_and_slotted() -> None:
    fr = FitnessReport(pnl_mean=0.0, max_drawdown=0.0, n_samples=1)
    ind = IndividualResult(
        chromosome=_seed_chromosome(),
        fitness_report=fr,
        fitness_scalar=0.0,
        generation_idx=0,
    )
    rep = GenerationReport(
        generation_idx=0,
        individuals=(ind,),
        best_fitness=0.0,
        mean_fitness=0.0,
        best_individual=ind,
        n_replacements=0,
    )
    assert not hasattr(rep, "__dict__")
    with pytest.raises(FrozenInstanceError):
        rep.best_fitness = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_target_strategy_id_must_be_str() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id=42,  # type: ignore[arg-type]
            population_size=6,
            max_generations=2,
        )


def test_config_target_strategy_id_must_be_non_empty() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="",
            population_size=6,
            max_generations=2,
        )


def test_config_population_size_must_be_int() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=6.0,  # type: ignore[arg-type]
            max_generations=2,
        )


def test_config_population_size_below_min_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=MIN_POPULATION_SIZE - 1,
            max_generations=2,
        )


def test_config_population_size_above_max_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=MAX_POPULATION_SIZE + 1,
            max_generations=2,
        )


def test_config_max_generations_below_one_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=6,
            max_generations=0,
        )


def test_config_max_generations_above_max_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=6,
            max_generations=MAX_GENERATIONS + 1,
        )


def test_config_total_evaluations_above_cap_raises() -> None:
    # population_size * (max_generations + 1) must be <= cap.
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=200,
            max_generations=999,
        )


def test_config_F_must_be_positive() -> None:
    with pytest.raises(EvolutionPipelineError):
        _make_config(F=0.0)
    with pytest.raises(EvolutionPipelineError):
        _make_config(F=-0.5)


def test_config_F_must_be_finite() -> None:
    with pytest.raises(EvolutionPipelineError):
        _make_config(F=math.inf)


def test_config_CR_out_of_range_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        _make_config(CR=-0.1)
    with pytest.raises(EvolutionPipelineError):
        _make_config(CR=1.1)


def test_config_init_sigma_non_positive_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        _make_config(init_sigma=0.0)
    with pytest.raises(EvolutionPipelineError):
        _make_config(init_sigma=-0.1)


def test_config_drawdown_weight_negative_raises() -> None:
    with pytest.raises(EvolutionPipelineError):
        _make_config(fitness_drawdown_weight=-0.5)


def test_config_drawdown_weight_zero_allowed() -> None:
    cfg = _make_config(fitness_drawdown_weight=0.0)
    assert cfg.fitness_drawdown_weight == 0.0


def test_config_mutation_strategy_must_be_enum() -> None:
    with pytest.raises(EvolutionPipelineError):
        EvolutionPipelineConfig(
            target_strategy_id="s",
            population_size=6,
            max_generations=2,
            mutation_strategy="DE_RAND_1",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# FitnessReport validation
# ---------------------------------------------------------------------------


def test_fitness_report_pnl_must_be_finite() -> None:
    with pytest.raises(EvolutionPipelineEvaluationError):
        FitnessReport(pnl_mean=math.nan, max_drawdown=0.0, n_samples=1)


def test_fitness_report_drawdown_must_be_non_negative() -> None:
    with pytest.raises(EvolutionPipelineEvaluationError):
        FitnessReport(pnl_mean=0.0, max_drawdown=-1.0, n_samples=1)


def test_fitness_report_n_samples_must_be_positive() -> None:
    with pytest.raises(EvolutionPipelineEvaluationError):
        FitnessReport(pnl_mean=0.0, max_drawdown=0.0, n_samples=0)


def test_fitness_report_meta_must_map_str_str() -> None:
    with pytest.raises(EvolutionPipelineEvaluationError):
        FitnessReport(
            pnl_mean=0.0,
            max_drawdown=0.0,
            n_samples=1,
            meta={"k": 42},  # type: ignore[dict-item]
        )


def test_fitness_report_scalar_collapse() -> None:
    fr = FitnessReport(pnl_mean=10.0, max_drawdown=4.0, n_samples=1)
    assert fr.fitness(0.5) == 10.0 - 0.5 * 4.0


# ---------------------------------------------------------------------------
# Pipeline construction validation
# ---------------------------------------------------------------------------


def test_pipeline_evaluator_must_implement_protocol() -> None:
    with pytest.raises(TypeError):
        EvolutionPipeline(evaluator=object())  # type: ignore[arg-type]


def test_pipeline_is_frozen_and_slotted() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    assert not hasattr(pipe, "__dict__")
    with pytest.raises(FrozenInstanceError):
        pipe.evaluator = _ConstantEvaluator()  # type: ignore[misc]


def test_evolve_specs_must_be_tuple() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=list(_two_d_specs()),  # type: ignore[arg-type]
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="p-1",
        )


def test_evolve_specs_must_be_non_empty() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=(),
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="p-1",
        )


def test_evolve_seed_must_be_non_negative_int() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=-1,
            ts_ns=1,
            proposal_id="p-1",
        )
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1.0,  # type: ignore[arg-type]
            ts_ns=1,
            proposal_id="p-1",
        )


def test_evolve_ts_ns_must_be_non_negative_int() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1,
            ts_ns=-1,
            proposal_id="p-1",
        )


def test_evolve_proposal_id_must_be_non_empty_str() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="",
        )


def test_evolve_proposal_id_max_length_enforced() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="x" * (MAX_PROPOSAL_ID_LEN + 1),
        )


def test_evolve_initial_chromosome_must_match_specs() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    seed_chrom = _seed_chromosome(specs=_mixed_specs())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="p-1",
            initial_chromosome=seed_chrom,
        )


def test_evolve_initial_chromosome_must_match_target_strategy() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    seed_chrom = _seed_chromosome(target_strategy_id="other-strat")
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(target_strategy_id="strat-001"),
            seed=1,
            ts_ns=1,
            proposal_id="p-1",
            initial_chromosome=seed_chrom,
        )


def test_evolve_callback_must_implement_protocol() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    with pytest.raises(EvolutionPipelineError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="p-1",
            callback=object(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Smoke run + result invariants
# ---------------------------------------------------------------------------


def test_evolve_returns_evolution_pipeline_result() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=_make_config(),
        seed=42,
        ts_ns=1_700_000_000_000_000_000,
        proposal_id="prop-001",
    )
    assert isinstance(result, EvolutionPipelineResult)
    assert isinstance(result.proposal, PatchProposal)
    assert isinstance(result.best_individual, IndividualResult)
    assert isinstance(result.generations, tuple)
    assert isinstance(result.initial_population, tuple)


def test_evolve_proposal_carries_canonical_fields() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(target_strategy_id="strat-001", max_generations=3)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=42,
        ts_ns=1_700_000_000_000_000_000,
        proposal_id="prop-001",
    )
    proposal = result.proposal
    assert proposal.patch_id == "prop-001"
    assert proposal.source == PROPOSAL_SOURCE
    assert proposal.target_strategy == "strat-001"
    assert proposal.touchpoints == ("x", "y")
    assert proposal.ts_ns == 1_700_000_000_000_000_000


def test_evolve_proposal_meta_keys_sorted_with_required_audit_fields() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=_make_config(),
        seed=42,
        ts_ns=1,
        proposal_id="prop-001",
    )
    keys = list(result.proposal.meta.keys())
    assert keys == sorted(keys)
    expected = {
        "best_chromosome_digest",
        "best_fitness",
        "best_max_drawdown",
        "best_pnl_mean",
        "generation_count",
        "mutation_strategy",
        "n_evaluations",
        "policy_digest",
        "population_size",
        "seed",
        "source_module",
    }
    assert set(keys) == expected


def test_evolve_proposal_rationale_under_cap() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=_make_config(),
        seed=42,
        ts_ns=1,
        proposal_id="prop-001",
    )
    assert 0 < len(result.proposal.rationale) <= MAX_RATIONALE_LEN


def test_evolve_initial_population_size_matches_config() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=8, max_generations=2)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    assert len(result.initial_population) == 8


def test_evolve_generation_count_matches_max_plus_initial() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=6, max_generations=5)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    # Includes initial generation (idx=0) + max_generations subsequent.
    assert len(result.generations) == 6
    assert [g.generation_idx for g in result.generations] == [0, 1, 2, 3, 4, 5]


def test_evolve_n_evaluations_is_population_times_generation_plus_one() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=6, max_generations=4)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    assert result.n_evaluations == 6 * (4 + 1)


def test_evolve_each_generation_individuals_match_population_size() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=8, max_generations=3)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    for gen in result.generations:
        assert len(gen.individuals) == 8


def test_evolve_each_generation_is_fittest_first() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=_make_config(),
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    for gen in result.generations:
        scalars = [ind.fitness_scalar for ind in gen.individuals]
        assert scalars == sorted(scalars, reverse=True)


def test_evolve_best_individual_is_max_fitness_across_run() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=_make_config(),
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    best_scalar = result.best_individual.fitness_scalar
    for gen in result.generations:
        for ind in gen.individuals:
            assert ind.fitness_scalar <= best_scalar


def test_evolve_policy_digest_is_16_hex_chars() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=_make_config(),
        seed=1,
        ts_ns=1,
        proposal_id="p-1",
    )
    assert len(result.policy_digest) == 16
    assert all(c in "0123456789abcdef" for c in result.policy_digest)
    # Echoed in proposal meta.
    assert result.proposal.meta["policy_digest"] == result.policy_digest


# ---------------------------------------------------------------------------
# INV-15 byte-identical replays
# ---------------------------------------------------------------------------


def _digest_result(result: EvolutionPipelineResult) -> tuple:
    """Reduce a result to a hashable canonical projection used for
    INV-15 equality checks."""

    return (
        result.policy_digest,
        result.n_evaluations,
        result.proposal.patch_id,
        result.proposal.source,
        result.proposal.target_strategy,
        result.proposal.touchpoints,
        tuple(sorted(result.proposal.meta.items())),
        chromosome_digest(result.best_individual.chromosome),
        result.best_individual.fitness_scalar,
        tuple(
            (
                gen.generation_idx,
                gen.best_fitness,
                gen.mean_fitness,
                gen.n_replacements,
                tuple(
                    (
                        chromosome_digest(ind.chromosome),
                        ind.fitness_scalar,
                        ind.generation_idx,
                    )
                    for ind in gen.individuals
                ),
            )
            for gen in result.generations
        ),
    )


def test_inv15_three_run_equality_de_rand_1() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(mutation_strategy=MutationStrategy.DE_RAND_1)
    digests = []
    for _ in range(3):
        result = pipe.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=42,
            ts_ns=1_700_000_000_000_000_000,
            proposal_id="prop-001",
        )
        digests.append(_digest_result(result))
    assert digests[0] == digests[1] == digests[2]


def test_inv15_three_run_equality_de_current_to_best_1() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(mutation_strategy=MutationStrategy.DE_CURRENT_TO_BEST_1)
    digests = []
    for _ in range(3):
        result = pipe.evolve(
            specs=_two_d_specs(),
            config=cfg,
            seed=42,
            ts_ns=1_700_000_000_000_000_000,
            proposal_id="prop-001",
        )
        digests.append(_digest_result(result))
    assert digests[0] == digests[1] == digests[2]


def test_inv15_three_run_equality_mixed_specs() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator(drawdown_floor=0.5))
    cfg = _make_config(population_size=6, max_generations=4)
    digests = []
    for _ in range(3):
        result = pipe.evolve(
            specs=_mixed_specs(),
            config=cfg,
            seed=7,
            ts_ns=1,
            proposal_id="prop-mixed",
        )
        digests.append(_digest_result(result))
    assert digests[0] == digests[1] == digests[2]


def test_inv15_seed_change_produces_different_run() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config()
    a = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    b = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=2,
        ts_ns=1,
        proposal_id="p",
    )
    assert _digest_result(a) != _digest_result(b)


def test_inv15_proposal_id_change_changes_policy_digest() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config()
    a = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="prop-A",
    )
    b = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="prop-B",
    )
    # Search result identical, but policy_digest folds in proposal_id.
    assert a.best_individual.fitness_scalar == b.best_individual.fitness_scalar
    assert a.policy_digest != b.policy_digest


def test_inv15_ts_ns_change_changes_proposal_ts() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config()
    a = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    b = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=2,
        proposal_id="p",
    )
    assert a.proposal.ts_ns == 1
    assert b.proposal.ts_ns == 2
    assert a.policy_digest != b.policy_digest


# ---------------------------------------------------------------------------
# Convergence (sphere)
# ---------------------------------------------------------------------------


def _shifted_specs() -> tuple[ParameterSpec, ...]:
    # Midpoint is (1.0, 1.0); sphere optimum is at (0, 0). The seed
    # individual is therefore meaningfully sub-optimal so DE has room
    # to make progress.
    return (
        ParameterSpec(name="x", kind=ParameterKind.CONTINUOUS, low=-3.0, high=5.0),
        ParameterSpec(name="y", kind=ParameterKind.CONTINUOUS, low=-3.0, high=5.0),
    )


def test_sphere_converges_de_rand_1() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(
        population_size=12,
        max_generations=40,
        mutation_strategy=MutationStrategy.DE_RAND_1,
        F=0.7,
        CR=0.9,
        init_sigma=0.5,
    )
    result = pipe.evolve(
        specs=_shifted_specs(),
        config=cfg,
        seed=123,
        ts_ns=1,
        proposal_id="p",
    )
    initial_best = result.generations[0].best_fitness
    final_best = result.generations[-1].best_fitness
    # Strict improvement over the initial population.
    assert final_best > initial_best
    # Sphere optimum is 0; we should be within a few units.
    assert final_best > -3.0


def test_sphere_converges_de_current_to_best_1() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(
        population_size=12,
        max_generations=30,
        mutation_strategy=MutationStrategy.DE_CURRENT_TO_BEST_1,
        F=0.5,
        CR=0.9,
        init_sigma=0.5,
    )
    result = pipe.evolve(
        specs=_shifted_specs(),
        config=cfg,
        seed=321,
        ts_ns=1,
        proposal_id="p",
    )
    initial_best = result.generations[0].best_fitness
    final_best = result.generations[-1].best_fitness
    assert final_best > initial_best
    assert final_best > -3.0


def test_running_best_is_monotonic_non_decreasing() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=8, max_generations=10)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=11,
        ts_ns=1,
        proposal_id="p",
    )
    last = -math.inf
    for gen in result.generations:
        assert gen.best_fitness >= last
        last = gen.best_fitness


# ---------------------------------------------------------------------------
# DE selection rule
# ---------------------------------------------------------------------------


def test_constant_evaluator_replacements_are_bounded_by_population_size() -> None:
    """Under a constant fitness evaluator every replacement is
    purely digest-driven (lexicographically-smaller chromosome digest
    wins). The number of replacements per generation is therefore
    bounded by ``population_size``, and the initial generation always
    reports zero replacements by construction."""

    pipe = EvolutionPipeline(evaluator=_ConstantEvaluator(pnl=0.0))
    cfg = _make_config(population_size=6, max_generations=3)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    assert result.generations[0].n_replacements == 0
    for gen in result.generations[1:]:
        assert 0 <= gen.n_replacements <= cfg.population_size


def test_constant_evaluator_best_fitness_is_constant_across_generations() -> None:
    pipe = EvolutionPipeline(evaluator=_ConstantEvaluator(pnl=1.5))
    cfg = _make_config(population_size=6, max_generations=4)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=2,
        ts_ns=1,
        proposal_id="p",
    )
    for gen in result.generations:
        assert gen.best_fitness == 1.5
        assert gen.mean_fitness == 1.5


def test_replacements_are_observed_when_search_makes_progress() -> None:
    """Sphere search must produce some replacements over many
    generations."""

    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=8, max_generations=15)
    result = pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=99,
        ts_ns=1,
        proposal_id="p",
    )
    total = sum(g.n_replacements for g in result.generations[1:])
    assert total > 0


# ---------------------------------------------------------------------------
# Initial chromosome handling
# ---------------------------------------------------------------------------


def test_evolve_default_initial_is_midpoint_chromosome() -> None:
    """When no initial_chromosome is supplied, the seed individual
    (population[0] of the initial generation) is the per-spec midpoint
    chromosome."""

    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config()
    specs = _two_d_specs()
    result = pipe.evolve(
        specs=specs,
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    # Find the seed: it has the parent-chromosome digest in
    # initial_population[0].chromosome.
    seed_chrom = result.initial_population[0].chromosome
    expected = tuple(0.5 * (s.low + s.high) for s in specs)
    assert seed_chrom.values == expected


def test_evolve_explicit_initial_chromosome_seeds_population() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(target_strategy_id="strat-explicit")
    specs = _two_d_specs()
    seed_chrom = StrategyChromosome(
        strategy_id="strat-explicit",
        specs=specs,
        values=(1.0, -1.0),
        version=0,
        meta={},
    )
    result = pipe.evolve(
        specs=specs,
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
        initial_chromosome=seed_chrom,
    )
    assert result.initial_population[0].chromosome.values == (1.0, -1.0)


# ---------------------------------------------------------------------------
# Evaluator contract
# ---------------------------------------------------------------------------


def test_bad_evaluator_return_type_raises() -> None:
    pipe = EvolutionPipeline(evaluator=_BadEvaluatorReturnType())
    with pytest.raises(EvolutionPipelineEvaluationError):
        pipe.evolve(
            specs=_two_d_specs(),
            config=_make_config(),
            seed=1,
            ts_ns=1,
            proposal_id="p",
        )


def test_evaluator_is_called_once_per_individual_per_generation() -> None:
    rec = _RecordingEvaluator(inner=_SphereEvaluator())
    pipe = EvolutionPipeline(evaluator=rec)
    cfg = _make_config(population_size=6, max_generations=3)
    pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    expected = 6 * (3 + 1)
    assert len(rec.calls) == expected  # type: ignore[attr-defined]


def test_evaluator_seed_keys_are_distinct_across_calls() -> None:
    """Per-individual evaluator seeds are derived deterministically
    via splitmix64, so distinct (g, i) keys produce distinct evaluator
    seeds with overwhelming probability."""

    rec = _RecordingEvaluator(inner=_SphereEvaluator())
    pipe = EvolutionPipeline(evaluator=rec)
    cfg = _make_config(population_size=6, max_generations=3)
    pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    seeds = [c[0] for c in rec.calls]  # type: ignore[attr-defined]
    # Allow some collisions in principle but never zero distinct keys.
    assert len(set(seeds)) >= 0.9 * len(seeds)


# ---------------------------------------------------------------------------
# Callback wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RecordingCallback:
    events: list[tuple] = field(default_factory=list)

    def on_evolution_start(
        self,
        *,
        ts_ns: int,
        config: EvolutionPipelineConfig,
        dimensionality: int,
    ) -> None:
        self.events.append(("start", ts_ns, dimensionality))

    def on_individual_evaluated(
        self,
        *,
        ts_ns: int,
        generation_idx: int,
        individual_idx: int,
        chromosome: StrategyChromosome,
        fitness_report: FitnessReport,
    ) -> None:
        self.events.append(("eval", generation_idx, individual_idx))

    def on_generation_end(
        self,
        *,
        ts_ns: int,
        report: GenerationReport,
    ) -> None:
        self.events.append(("gen_end", report.generation_idx, report.n_replacements))

    def on_evolution_end(
        self,
        *,
        ts_ns: int,
        result: EvolutionPipelineResult,
    ) -> None:
        self.events.append(("end", result.n_evaluations))


def test_null_callback_is_runtime_checkable() -> None:
    cb = null_evolution_pipeline_callback()
    assert isinstance(cb, EvolutionPipelineCallback)


def test_callback_lifecycle_is_invoked() -> None:
    cb = _RecordingCallback()
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    cfg = _make_config(population_size=6, max_generations=2)
    pipe.evolve(
        specs=_two_d_specs(),
        config=cfg,
        seed=1,
        ts_ns=42,
        proposal_id="p",
        callback=cb,
    )
    events = cb.events  # type: ignore[attr-defined]
    assert events[0][0] == "start"
    assert events[-1][0] == "end"
    assert any(e[0] == "gen_end" for e in events)
    assert any(e[0] == "eval" for e in events)


# ---------------------------------------------------------------------------
# Authority symmetry pin
# ---------------------------------------------------------------------------


def test_module_lives_on_evolution_engine_side_of_authority_boundary() -> None:
    """The module path begins with ``evolution_engine.`` so that the
    B28 PatchProposal-construction allowlist accepts it. This pin
    fails fast if the module is ever moved out of the
    ``evolution_engine.*`` namespace."""

    mod = importlib.import_module("evolution_engine.pipeline")
    assert mod.__name__.startswith("evolution_engine.")


def test_module_actually_constructs_patchproposal() -> None:
    """The presence of a literal ``PatchProposal(...)`` call expression
    is the authoritative producer signal for B28. This test pins it."""

    tree = ast.parse(MODULE_TEXT)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "PatchProposal":
                found = True
                break
    assert found, "EvolutionPipeline must construct PatchProposal"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_is_complete() -> None:
    import evolution_engine.pipeline as mod

    expected = {
        "DEFAULT_CR",
        "DEFAULT_DRAWDOWN_PENALTY",
        "DEFAULT_F",
        "DEFAULT_INIT_SIGMA",
        "EvolutionPipeline",
        "EvolutionPipelineCallback",
        "EvolutionPipelineConfig",
        "EvolutionPipelineError",
        "EvolutionPipelineEvaluationError",
        "EvolutionPipelineResult",
        "FitnessEvaluator",
        "FitnessReport",
        "GenerationReport",
        "IndividualResult",
        "MAX_GENERATIONS",
        "MAX_POPULATION_SIZE",
        "MAX_PROPOSAL_ID_LEN",
        "MAX_RATIONALE_LEN",
        "MAX_TOTAL_EVALUATIONS",
        "MIN_POPULATION_SIZE",
        "MutationStrategy",
        "NEW_PIP_DEPENDENCIES",
        "OPERATOR_DE_BINOMIAL_CROSSOVER",
        "OPERATOR_DE_CURRENT_TO_BEST_1",
        "OPERATOR_DE_RAND_1",
        "OPERATOR_GAUSSIAN",
        "PROPOSAL_SOURCE",
        "null_evolution_pipeline_callback",
    }
    assert set(mod.__all__) == expected


# ---------------------------------------------------------------------------
# Touchpoints / mixed-specs round-trip
# ---------------------------------------------------------------------------


def test_touchpoints_match_spec_names_in_order() -> None:
    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    specs = _mixed_specs()
    result = pipe.evolve(
        specs=specs,
        config=_make_config(),
        seed=1,
        ts_ns=1,
        proposal_id="p",
    )
    assert result.proposal.touchpoints == tuple(s.name for s in specs)


def test_best_chromosome_is_unpack_feasible_for_mixed_specs() -> None:
    """Mixed-spec runs (LOG_CONTINUOUS / INTEGER / CONTINUOUS) must
    return a best chromosome that round-trips through
    ``pack`` / ``unpack`` without raising — i.e. every dimension
    remained feasible despite the donor / crossover sequence."""

    pipe = EvolutionPipeline(evaluator=_SphereEvaluator())
    specs = _mixed_specs()
    result = pipe.evolve(
        specs=specs,
        config=_make_config(),
        seed=5,
        ts_ns=1,
        proposal_id="p",
    )
    chrom = result.best_individual.chromosome
    encoded = pack(specs, chrom.to_mapping())
    decoded = unpack(specs, encoded)
    assert len(decoded) == len(specs)
