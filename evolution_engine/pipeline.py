# ADAPTED FROM: facebookresearch/nevergrad
# (nevergrad/optimization/base.py — Optimizer.ask() / tell() ask/tell loop;
#  nevergrad/optimization/differentialevolution.py — DE/rand/1, DE/current-to-best/1,
#  binomial crossover; nevergrad/optimization/optimizerlib.py — DE generation
#  schedule, NGOpt initialisation pattern.)
"""A-04.2 — EvolutionPipeline: DE ask/tell orchestrator emitting PatchProposal.

nevergrad's :class:`Optimizer` walks an *ask / tell* loop: ``ask()`` draws
a candidate from the search distribution, the caller evaluates it on a
black-box objective, ``tell(candidate, loss)`` folds the loss back in.
Differential Evolution (``DE``, ``TwoPointsDE``) realises this loop with
three- or four-parent recombination + binomial crossover. The DIX
analogue sits behind the same ``ask / tell`` shape but is wrapped in a
single, stateless :meth:`EvolutionPipeline.evolve` entry-point so that
the pipeline is INV-15-deterministic by construction (no internal mutable
optimizer state survives the call).

What this module is
-------------------

* Pure-stdlib coordinator + frozen value objects. No nevergrad import.
  ``NEW_PIP_DEPENDENCIES = ()``. Mutation primitives come from the
  A-04.1 leaf (:mod:`evolution_engine.genetic.mutation_operators`); the
  parameter encoding from A-02.1
  (:mod:`evolution_engine.genetic.strategy_chromosome`).
* OFFLINE_ONLY tier. No clock, no IO, no environment reads, no engine
  cross-imports. The pipeline runs a search, captures the best
  individual, and emits a typed
  :class:`~core.contracts.learning.PatchProposal` for governance
  approval. INV-13/14: Evolution NEVER deploys directly.
* INV-15 byte-identical replays. ``EvolutionPipeline.evolve(...)`` with
  identical ``config`` / ``specs`` / ``initial_chromosome`` / ``seed`` /
  ``ts_ns`` / ``proposal_id`` / ``evaluator`` returns identical
  :class:`EvolutionPipelineResult` records. The PRNG is a stateless
  :func:`_splitmix64` / :func:`_uniform01` stream keyed off the
  caller-supplied ``seed``; selection ties (``trial_fitness ==
  target_fitness``) are broken by
  :func:`~evolution_engine.genetic.strategy_chromosome.chromosome_digest`
  ascending.
* Authority symmetry — INV-71 / B27 / B28. The mutation operators in
  :mod:`evolution_engine.genetic.mutation_operators` are pure
  functions and never construct :class:`PatchProposal`; this module
  lives on the evolution-engine side of the authority boundary, so it
  IS allowed to construct :class:`PatchProposal` (mirrors
  :mod:`evolution_engine.sandbox` and
  :mod:`evolution_engine.genetic.cmaes_optimizer`).

Algorithm
---------

For each :meth:`evolve` call:

1. **Initial population** is constructed from the caller-supplied
   ``initial_chromosome`` (defaulting to a midpoint chromosome over
   ``specs``) by per-individual Gaussian perturbation in encoded
   space::

       P[0] = initial
       P[i] = gaussian_mutate(initial, sigma=init_sigma,
                              seed=seed, generation=0,
                              individual=i)        for i in [1, N)

   Each ``P[i]`` is evaluated once via the
   :class:`FitnessEvaluator` Protocol; the running best is taken from
   the initial population (ties broken by ``chromosome_digest``).

2. **Per generation** ``g in [1, max_generations]``, for each target
   index ``i in [0, N)``:

   a. Three (DE/rand/1) or two (DE/current-to-best/1) distinct
      indices ``a, b, c`` (or ``a, b``) are sampled deterministically
      from ``[0, N) \\ {i}`` via :func:`_splitmix64`.
   b. The donor chromosome ``v`` is produced by either
      :func:`~evolution_engine.genetic.mutation_operators.de_rand_1`
      or
      :func:`~evolution_engine.genetic.mutation_operators.de_current_to_best_1`
      (selected by :class:`MutationStrategy`) with the running best as
      the elite anchor in the latter case.
   c. The trial ``u`` is produced by
      :func:`~evolution_engine.genetic.mutation_operators.de_binomial_crossover`
      between target ``P[i]`` and donor ``v``. The forced-mutant index
      guarantees ``u != P[i]`` even at ``CR = 0``.
   d. The trial is evaluated via the :class:`FitnessEvaluator`. The
      per-individual seed forwarded to the evaluator is
      ``_splitmix64(seed, g, i, EVAL_SALT)`` so the evaluator can
      downstream the determinism contract without sharing state with
      the pipeline.
   e. **Selection** (DE-style "replace if not worse"): if
      ``trial_fitness >= P[i].fitness_scalar`` the target slot is
      replaced by the trial; the running best is updated on
      ``trial_fitness > best.fitness_scalar`` (strictly greater) or
      on tie + lexicographically-smaller chromosome digest.

3. After ``max_generations`` generations the running best is
   collapsed into a typed :class:`PatchProposal`: ``patch_id =
   proposal_id``, ``source = PROPOSAL_SOURCE``, ``target_strategy =
   config.target_strategy_id``, ``touchpoints = tuple of spec names``,
   ``rationale = `` auto-generated, ``meta`` carries the audit
   fingerprints (best chromosome digest, policy digest, generation
   count, evaluation count, fitness scalar / pnl / drawdown / etc.).

The :class:`EvolutionPipelineResult` carries the proposal plus the
per-generation :class:`GenerationReport` tuple and a 16-hex
:attr:`policy_digest` so dashboards can render the audit trail at
review time.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE tier — no IO, no clock, no global state. AST tests pin the
  import contract.
* No engine cross-imports — AST test pins no ``execution_engine.`` /
  ``governance_engine.`` / ``system_engine.`` /
  ``intelligence_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-13/14 — :meth:`EvolutionPipeline.evolve` returns one
  :class:`EvolutionPipelineResult` containing one
  :class:`PatchProposal`; it does **not** mutate any external
  registry or governance ledger.
* INV-15 — the result is a pure function of its inputs. The
  fingerprint :attr:`EvolutionPipelineResult.policy_digest` is a
  16-hex-char BLAKE2b-8 hash of a canonical text projection of the
  optimizer state at the best individual.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from dataclasses import field
from typing import Protocol, runtime_checkable

from core.contracts.learning import PatchProposal
from evolution_engine.genetic.mutation_operators import (
    OPERATOR_DE_BINOMIAL_CROSSOVER,
    OPERATOR_DE_CURRENT_TO_BEST_1,
    OPERATOR_DE_RAND_1,
    OPERATOR_GAUSSIAN,
    de_binomial_crossover,
    de_current_to_best_1,
    de_rand_1,
    gaussian_mutate,
)
from evolution_engine.genetic.strategy_chromosome import (
    ChromosomeError,
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No new pip deps — pure-stdlib adaptation of nevergrad's DE loop."""

MIN_POPULATION_SIZE: int = 4
"""DE/rand/1 needs at least three parents distinct from the target, so
the minimum feasible population is four."""

MAX_POPULATION_SIZE: int = 256
"""Mirrors :data:`evolution_engine.genetic.cmaes_optimizer.MAX_POPULATION_SIZE`."""

MAX_GENERATIONS: int = 1000
"""Mirrors :data:`evolution_engine.genetic.cmaes_optimizer.MAX_GENERATIONS`."""

MAX_TOTAL_EVALUATIONS: int = 100_000
"""Hard ceiling on ``(population_size * (max_generations + 1))``: the
``+ 1`` covers the initial population evaluation."""

MAX_PROPOSAL_ID_LEN: int = 256
"""Mirrors :data:`evolution_engine.genetic.cmaes_optimizer.MAX_PROPOSAL_ID_LEN`."""

MAX_RATIONALE_LEN: int = 1024
"""Mirrors :data:`evolution_engine.genetic.cmaes_optimizer.MAX_RATIONALE_LEN`."""

PROPOSAL_SOURCE: str = "evolution_engine.pipeline"
"""Identifies this adapter as the producer of the proposal."""

DEFAULT_INIT_SIGMA: float = 0.2
"""Default Gaussian-mutation sigma (fraction of encoded span) used for
seeding the initial population from the caller-supplied initial
chromosome. ``0.2`` matches nevergrad's default
``DifferentialEvolution(initialization='gaussian', scale=0.2)``."""

DEFAULT_F: float = 0.5
"""Default DE differential weight. nevergrad's
``DifferentialEvolution`` ships ``F1 = 0.5`` by default; canonical DE
literature places ``F`` in ``[0.4, 1.0]``."""

DEFAULT_CR: float = 0.9
"""Default DE binomial-crossover rate. nevergrad's
``DifferentialEvolution`` ships ``CR = 0.9`` by default."""

DEFAULT_DRAWDOWN_PENALTY: float = 0.5
"""Mirrors :data:`evolution_engine.genetic.cmaes_optimizer.DEFAULT_DRAWDOWN_PENALTY`."""

_EVAL_SALT: int = 0xEA1_BAD_C0DE
"""Mixed into the per-individual evaluator seed to keep it disjoint
from the operator seed stream."""

_INDEX_SALT: int = 0xDE_1ABE1_C0DE
"""Mixed into the parent-index sampling stream to keep it disjoint
from the operator + evaluator seed streams."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EvolutionPipelineError(ValueError):
    """Raised when the caller passes an invalid combination of args to
    :class:`EvolutionPipelineConfig` /
    :meth:`EvolutionPipeline.evolve`."""


class EvolutionPipelineEvaluationError(RuntimeError):
    """Raised when the injected :class:`FitnessEvaluator` returns an
    invalid :class:`FitnessReport`. Fail-fast — a bad evaluation is
    never silently dropped because that would hide replay
    divergence."""


# ---------------------------------------------------------------------------
# Mutation strategy
# ---------------------------------------------------------------------------


class MutationStrategy(enum.StrEnum):
    """Which donor strategy the pipeline uses on each generation step.

    ``DE_RAND_1`` is the classical DE/rand/1 variant — robust, easy to
    tune, slow to converge. ``DE_CURRENT_TO_BEST_1`` adds elite
    guidance (``best - target`` term) — faster convergence, more prone
    to premature convergence on multimodal landscapes.
    """

    DE_RAND_1 = "DE_RAND_1"
    DE_CURRENT_TO_BEST_1 = "DE_CURRENT_TO_BEST_1"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EvolutionPipelineConfig:
    """Frozen configuration for one :meth:`EvolutionPipeline.evolve` call.

    ``target_strategy_id`` is the parent strategy whose parameters this
    run is searching over; it is stamped into the resulting
    :attr:`PatchProposal.target_strategy` and into every individual's
    :attr:`StrategyChromosome.strategy_id`.
    """

    target_strategy_id: str
    population_size: int
    max_generations: int
    mutation_strategy: MutationStrategy = MutationStrategy.DE_RAND_1
    F: float = DEFAULT_F
    CR: float = DEFAULT_CR
    init_sigma: float = DEFAULT_INIT_SIGMA
    fitness_drawdown_weight: float = DEFAULT_DRAWDOWN_PENALTY

    def __post_init__(self) -> None:
        if not isinstance(self.target_strategy_id, str):
            raise EvolutionPipelineError("EvolutionPipelineConfig.target_strategy_id must be str")
        if not self.target_strategy_id:
            raise EvolutionPipelineError(
                "EvolutionPipelineConfig.target_strategy_id must be non-empty"
            )
        if not isinstance(self.population_size, int) or isinstance(self.population_size, bool):
            raise EvolutionPipelineError("EvolutionPipelineConfig.population_size must be int")
        if self.population_size < MIN_POPULATION_SIZE or self.population_size > MAX_POPULATION_SIZE:
            raise EvolutionPipelineError(
                f"EvolutionPipelineConfig.population_size must be in "
                f"[{MIN_POPULATION_SIZE}, {MAX_POPULATION_SIZE}], got "
                f"{self.population_size!r}"
            )
        if not isinstance(self.max_generations, int) or isinstance(self.max_generations, bool):
            raise EvolutionPipelineError("EvolutionPipelineConfig.max_generations must be int")
        if self.max_generations < 1 or self.max_generations > MAX_GENERATIONS:
            raise EvolutionPipelineError(
                f"EvolutionPipelineConfig.max_generations must be in "
                f"[1, {MAX_GENERATIONS}], got {self.max_generations!r}"
            )
        # +1 for initial population evaluation
        total = self.population_size * (self.max_generations + 1)
        if total > MAX_TOTAL_EVALUATIONS:
            raise EvolutionPipelineError(
                f"EvolutionPipelineConfig: population_size * "
                f"(max_generations + 1) = {total} > {MAX_TOTAL_EVALUATIONS}"
            )
        if not isinstance(self.mutation_strategy, MutationStrategy):
            raise EvolutionPipelineError(
                "EvolutionPipelineConfig.mutation_strategy must be MutationStrategy"
            )
        for fname in ("F", "CR", "init_sigma", "fitness_drawdown_weight"):
            v = getattr(self, fname)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise EvolutionPipelineError(f"EvolutionPipelineConfig.{fname} must be int|float")
            if not math.isfinite(v):
                raise EvolutionPipelineError(
                    f"EvolutionPipelineConfig.{fname} must be finite, got {v!r}"
                )
        if self.F <= 0.0:
            raise EvolutionPipelineError(f"EvolutionPipelineConfig.F must be > 0, got {self.F!r}")
        if self.CR < 0.0 or self.CR > 1.0:
            raise EvolutionPipelineError(
                f"EvolutionPipelineConfig.CR must be in [0, 1], got {self.CR!r}"
            )
        if self.init_sigma <= 0.0:
            raise EvolutionPipelineError(
                f"EvolutionPipelineConfig.init_sigma must be > 0, got {self.init_sigma!r}"
            )
        if self.fitness_drawdown_weight < 0.0:
            raise EvolutionPipelineError(
                f"EvolutionPipelineConfig.fitness_drawdown_weight must be "
                f">= 0, got {self.fitness_drawdown_weight!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class FitnessReport:
    """Result of evaluating one chromosome on the simulation harness.

    Mirrors
    :class:`evolution_engine.genetic.cmaes_optimizer.FitnessReport`. The
    pipeline collapses the report to a single scalar via
    ``fitness = pnl_mean - drawdown_weight * max_drawdown``.
    """

    pnl_mean: float
    max_drawdown: float
    n_samples: int
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pnl_mean, (int, float)) or isinstance(self.pnl_mean, bool):
            raise EvolutionPipelineEvaluationError("FitnessReport.pnl_mean must be int|float")
        if not math.isfinite(self.pnl_mean):
            raise EvolutionPipelineEvaluationError(
                f"FitnessReport.pnl_mean must be finite, got {self.pnl_mean!r}"
            )
        if not isinstance(self.max_drawdown, (int, float)) or isinstance(self.max_drawdown, bool):
            raise EvolutionPipelineEvaluationError("FitnessReport.max_drawdown must be int|float")
        if not math.isfinite(self.max_drawdown) or self.max_drawdown < 0.0:
            raise EvolutionPipelineEvaluationError(
                f"FitnessReport.max_drawdown must be a non-negative "
                f"finite float, got {self.max_drawdown!r}"
            )
        if not isinstance(self.n_samples, int) or isinstance(self.n_samples, bool):
            raise EvolutionPipelineEvaluationError("FitnessReport.n_samples must be int")
        if self.n_samples < 1:
            raise EvolutionPipelineEvaluationError(
                f"FitnessReport.n_samples must be >= 1, got {self.n_samples!r}"
            )
        if not isinstance(self.meta, Mapping):
            raise EvolutionPipelineEvaluationError("FitnessReport.meta must be Mapping")
        for mk, mv in self.meta.items():
            if not isinstance(mk, str) or not isinstance(mv, str):
                raise EvolutionPipelineEvaluationError("FitnessReport.meta must map str -> str")

    def fitness(self, drawdown_weight: float) -> float:
        """Collapse to a scalar: ``pnl_mean - w * max_drawdown``."""

        return self.pnl_mean - drawdown_weight * self.max_drawdown


@dataclasses.dataclass(frozen=True, slots=True)
class IndividualResult:
    """One sample paired with its fitness.

    Stored in :class:`GenerationReport.individuals` in selection order
    (fittest first; ties broken by chromosome digest ascending).
    """

    chromosome: StrategyChromosome
    fitness_report: FitnessReport
    fitness_scalar: float
    generation_idx: int

    def __post_init__(self) -> None:
        if not isinstance(self.chromosome, StrategyChromosome):
            raise TypeError(
                "IndividualResult.chromosome must be StrategyChromosome, "
                f"got {type(self.chromosome).__name__}"
            )
        if not isinstance(self.fitness_report, FitnessReport):
            raise TypeError(
                "IndividualResult.fitness_report must be FitnessReport, "
                f"got {type(self.fitness_report).__name__}"
            )
        if not isinstance(self.fitness_scalar, (int, float)) or isinstance(
            self.fitness_scalar, bool
        ):
            raise TypeError(
                "IndividualResult.fitness_scalar must be int|float, got "
                f"{type(self.fitness_scalar).__name__}"
            )
        if not math.isfinite(self.fitness_scalar):
            raise ValueError(
                f"IndividualResult.fitness_scalar must be finite, got {self.fitness_scalar!r}"
            )
        if not isinstance(self.generation_idx, int) or isinstance(self.generation_idx, bool):
            raise TypeError(
                "IndividualResult.generation_idx must be int, got "
                f"{type(self.generation_idx).__name__}"
            )
        if self.generation_idx < 0:
            raise ValueError(
                f"IndividualResult.generation_idx must be >= 0, got {self.generation_idx!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationReport:
    """Snapshot of one generation: post-selection individuals (sorted
    fittest-first), headline stats, and the running best at this point.
    """

    generation_idx: int
    individuals: tuple[IndividualResult, ...]
    best_fitness: float
    mean_fitness: float
    best_individual: IndividualResult
    n_replacements: int

    def __post_init__(self) -> None:
        if not isinstance(self.generation_idx, int) or isinstance(self.generation_idx, bool):
            raise TypeError(
                "GenerationReport.generation_idx must be int, got "
                f"{type(self.generation_idx).__name__}"
            )
        if self.generation_idx < 0:
            raise ValueError(
                f"GenerationReport.generation_idx must be >= 0, got {self.generation_idx!r}"
            )
        if not isinstance(self.individuals, tuple):
            raise TypeError(
                f"GenerationReport.individuals must be tuple, got {type(self.individuals).__name__}"
            )
        if not self.individuals:
            raise ValueError("GenerationReport.individuals must be non-empty")
        for idx, ind in enumerate(self.individuals):
            if not isinstance(ind, IndividualResult):
                raise TypeError(f"GenerationReport.individuals[{idx}] must be IndividualResult")
        if not isinstance(self.best_fitness, (int, float)) or isinstance(self.best_fitness, bool):
            raise TypeError("GenerationReport.best_fitness must be int|float")
        if not math.isfinite(self.best_fitness):
            raise ValueError(
                f"GenerationReport.best_fitness must be finite, got {self.best_fitness!r}"
            )
        if not isinstance(self.mean_fitness, (int, float)) or isinstance(self.mean_fitness, bool):
            raise TypeError("GenerationReport.mean_fitness must be int|float")
        if not math.isfinite(self.mean_fitness):
            raise ValueError(
                f"GenerationReport.mean_fitness must be finite, got {self.mean_fitness!r}"
            )
        if not isinstance(self.best_individual, IndividualResult):
            raise TypeError("GenerationReport.best_individual must be IndividualResult")
        if not isinstance(self.n_replacements, int) or isinstance(self.n_replacements, bool):
            raise TypeError("GenerationReport.n_replacements must be int")
        if self.n_replacements < 0:
            raise ValueError(
                f"GenerationReport.n_replacements must be >= 0, got {self.n_replacements!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class EvolutionPipelineResult:
    """Output of :meth:`EvolutionPipeline.evolve`.

    The :class:`PatchProposal` carries the governance-shaped payload;
    the :class:`GenerationReport` tuple and :attr:`policy_digest` carry
    the audit metadata operators consult when reviewing the proposal.
    """

    proposal: PatchProposal
    generations: tuple[GenerationReport, ...]
    initial_population: tuple[IndividualResult, ...]
    best_individual: IndividualResult
    policy_digest: str
    n_evaluations: int

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, PatchProposal):
            raise TypeError(
                "EvolutionPipelineResult.proposal must be PatchProposal, "
                f"got {type(self.proposal).__name__}"
            )
        if not isinstance(self.generations, tuple):
            raise TypeError("EvolutionPipelineResult.generations must be tuple")
        if not self.generations:
            raise ValueError("EvolutionPipelineResult.generations must be non-empty")
        for idx, gen in enumerate(self.generations):
            if not isinstance(gen, GenerationReport):
                raise TypeError(
                    f"EvolutionPipelineResult.generations[{idx}] must be GenerationReport"
                )
        if not isinstance(self.initial_population, tuple):
            raise TypeError("EvolutionPipelineResult.initial_population must be tuple")
        if not self.initial_population:
            raise ValueError("EvolutionPipelineResult.initial_population must be non-empty")
        for idx, ind in enumerate(self.initial_population):
            if not isinstance(ind, IndividualResult):
                raise TypeError(
                    f"EvolutionPipelineResult.initial_population[{idx}] must be IndividualResult"
                )
        if not isinstance(self.best_individual, IndividualResult):
            raise TypeError("EvolutionPipelineResult.best_individual must be IndividualResult")
        if not isinstance(self.policy_digest, str):
            raise TypeError("EvolutionPipelineResult.policy_digest must be str")
        if len(self.policy_digest) != 16:
            raise ValueError(
                f"EvolutionPipelineResult.policy_digest must be 16 hex "
                f"chars, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                f"EvolutionPipelineResult.policy_digest must be lowercase "
                f"hex, got {self.policy_digest!r}"
            )
        if not isinstance(self.n_evaluations, int) or isinstance(self.n_evaluations, bool):
            raise TypeError("EvolutionPipelineResult.n_evaluations must be int")
        if self.n_evaluations < 1:
            raise ValueError(
                f"EvolutionPipelineResult.n_evaluations must be >= 1, got {self.n_evaluations!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class FitnessEvaluator(Protocol):
    """Caller-supplied fitness evaluator.

    The only seam between this module and the trading simulation.
    Production wires a thin adapter onto
    :mod:`simulation.parallel_runner` (SIM-07); tests inject a
    deterministic fake. The contract is single-shot: the evaluator
    fully consumes one chromosome and returns one
    :class:`FitnessReport`.

    Determinism: the evaluator MUST be a pure function of
    ``(chromosome, seed, ts_ns)`` for INV-15 to hold.
    """

    def evaluate(
        self,
        *,
        chromosome: StrategyChromosome,
        seed: int,
        ts_ns: int,
    ) -> FitnessReport: ...


@runtime_checkable
class EvolutionPipelineCallback(Protocol):
    """Optional lifecycle callback. The default is
    :func:`null_evolution_pipeline_callback`."""

    def on_evolution_start(
        self,
        *,
        ts_ns: int,
        config: EvolutionPipelineConfig,
        dimensionality: int,
    ) -> None: ...

    def on_individual_evaluated(
        self,
        *,
        ts_ns: int,
        generation_idx: int,
        individual_idx: int,
        chromosome: StrategyChromosome,
        fitness_report: FitnessReport,
    ) -> None: ...

    def on_generation_end(
        self,
        *,
        ts_ns: int,
        report: GenerationReport,
    ) -> None: ...

    def on_evolution_end(
        self,
        *,
        ts_ns: int,
        result: EvolutionPipelineResult,
    ) -> None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class _NullCallback:
    def on_evolution_start(
        self,
        *,
        ts_ns: int,
        config: EvolutionPipelineConfig,
        dimensionality: int,
    ) -> None:
        return None

    def on_individual_evaluated(
        self,
        *,
        ts_ns: int,
        generation_idx: int,
        individual_idx: int,
        chromosome: StrategyChromosome,
        fitness_report: FitnessReport,
    ) -> None:
        return None

    def on_generation_end(
        self,
        *,
        ts_ns: int,
        report: GenerationReport,
    ) -> None:
        return None

    def on_evolution_end(
        self,
        *,
        ts_ns: int,
        result: EvolutionPipelineResult,
    ) -> None:
        return None


def null_evolution_pipeline_callback() -> EvolutionPipelineCallback:
    """Default no-op :class:`EvolutionPipelineCallback`."""

    return _NullCallback()


# ---------------------------------------------------------------------------
# Stateless deterministic PRNG (mirrors gym_env / cmaes_optimizer)
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    """Stateless 64-bit hash (mirrors :mod:`evolution_engine.gym_env` /
    :mod:`evolution_engine.genetic.cmaes_optimizer`)."""

    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _seed_int(*key: int) -> int:
    """Fold a key tuple into a non-negative 63-bit int."""

    h = 0
    for k in key:
        h = _splitmix64(h ^ (k & 0xFFFFFFFFFFFFFFFF))
    return h & 0x7FFFFFFFFFFFFFFF


def _sample_distinct(*, n: int, exclude: int, count: int, key: int) -> tuple[int, ...]:
    """Deterministically sample ``count`` distinct indices from ``[0, n)
    \\ {exclude}`` via :func:`_splitmix64`. ``count`` must be ``<=
    n - 1``.

    Uses partial Fisher-Yates over the ``[0, n) \\ {exclude}`` domain;
    the swap stream is a deterministic function of ``key``.
    """

    if count >= n:
        raise EvolutionPipelineError(f"_sample_distinct: count {count} must be < n {n}")
    if count > n - 1:
        raise EvolutionPipelineError(f"_sample_distinct: count {count} must be <= n - 1 = {n - 1}")
    pool = [i for i in range(n) if i != exclude]
    h = _splitmix64(key & 0xFFFFFFFFFFFFFFFF)
    last = len(pool) - 1
    for swap_step in range(count):
        h = _splitmix64(h ^ (swap_step + 1))
        j = swap_step + (h % (last - swap_step + 1))
        pool[swap_step], pool[j] = pool[j], pool[swap_step]
    return tuple(pool[:count])


# ---------------------------------------------------------------------------
# Initial chromosome construction
# ---------------------------------------------------------------------------


def _midpoint_value(spec: ParameterSpec) -> float:
    """Pick a deterministic feasible midpoint for a spec."""

    if spec.kind is ParameterKind.LOG_CONTINUOUS:
        # Log-space midpoint, then exponentiate.
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        return math.exp(0.5 * (log_low + log_high))
    if spec.kind is ParameterKind.INTEGER:
        # Integer midpoint via banker's rounding (matches
        # StrategyChromosome.unpack).
        midpoint = 0.5 * (spec.low + spec.high)
        return float(round(midpoint))
    return 0.5 * (spec.low + spec.high)


def _midpoint_chromosome(
    *,
    target_strategy_id: str,
    specs: tuple[ParameterSpec, ...],
) -> StrategyChromosome:
    """Build a midpoint chromosome over ``specs``."""

    values = tuple(_midpoint_value(s) for s in specs)
    try:
        return StrategyChromosome(
            strategy_id=target_strategy_id,
            specs=specs,
            values=values,
            version=0,
            meta={},
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise EvolutionPipelineError(f"_midpoint_chromosome: {exc}") from exc


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _is_better(*, candidate: IndividualResult, incumbent: IndividualResult) -> bool:
    """``candidate`` strictly preferred over ``incumbent``: higher
    fitness, or equal fitness + lexicographically-smaller chromosome
    digest. Tie-breaking by digest pins INV-15 across runs.
    """

    if candidate.fitness_scalar > incumbent.fitness_scalar:
        return True
    if candidate.fitness_scalar < incumbent.fitness_scalar:
        return False
    return chromosome_digest(candidate.chromosome) < chromosome_digest(incumbent.chromosome)


def _is_at_least_as_good(
    *,
    candidate: IndividualResult,
    incumbent: IndividualResult,
) -> bool:
    """``candidate`` accepted by DE selection: strictly higher fitness
    OR equal fitness + lexicographically-smaller chromosome digest.

    The plain ``>=`` selection rule used by canonical DE would accept
    *any* tie, which is replay-stable as long as the tie-break never
    flips on identical inputs. Going through the digest tie-break
    keeps the rule symmetric with :func:`_is_better` so both call
    sites use the same INV-15 ordering.
    """

    return _is_better(candidate=candidate, incumbent=incumbent) or (
        candidate.fitness_scalar == incumbent.fitness_scalar
        and chromosome_digest(candidate.chromosome) == chromosome_digest(incumbent.chromosome)
    )


# ---------------------------------------------------------------------------
# Policy digest + rationale
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    config: EvolutionPipelineConfig,
    best: IndividualResult,
    generation_count: int,
    n_evaluations: int,
    seed: int,
    ts_ns: int,
    proposal_id: str,
) -> str:
    """16-hex BLAKE2b-8 over a canonical text projection of the
    pipeline state at the best individual. INV-15."""

    parts: list[str] = [
        "evolution_pipeline/v1",
        f"target={config.target_strategy_id}",
        f"pop={config.population_size}",
        f"gens={config.max_generations}",
        f"strategy={config.mutation_strategy.value}",
        f"F={config.F!r}",
        f"CR={config.CR!r}",
        f"init_sigma={config.init_sigma!r}",
        f"dd_weight={config.fitness_drawdown_weight!r}",
        f"seed={seed}",
        f"ts_ns={ts_ns}",
        f"proposal_id={proposal_id}",
        f"generation_count={generation_count}",
        f"n_evaluations={n_evaluations}",
        f"best_chromosome={chromosome_digest(best.chromosome)}",
        f"best_fitness={best.fitness_scalar!r}",
        f"best_pnl={best.fitness_report.pnl_mean!r}",
        f"best_drawdown={best.fitness_report.max_drawdown!r}",
        f"best_n_samples={best.fitness_report.n_samples}",
    ]
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _build_rationale(
    *,
    best: IndividualResult,
    generation_count: int,
    n_evaluations: int,
    config: EvolutionPipelineConfig,
) -> str:
    """Build the auto-generated :attr:`PatchProposal.rationale`."""

    text = (
        f"DE search ({config.mutation_strategy.value}) over "
        f"{best.chromosome.dimensionality} parameters: "
        f"{generation_count} generations of pop="
        f"{config.population_size} (n_evaluations={n_evaluations}), "
        f"F={config.F:.4f}, CR={config.CR:.4f}, "
        f"init_sigma={config.init_sigma:.4f}, "
        f"best_fitness={best.fitness_scalar:.6f} "
        f"(pnl_mean={best.fitness_report.pnl_mean:.6f}, "
        f"max_drawdown={best.fitness_report.max_drawdown:.6f}, "
        f"drawdown_weight={config.fitness_drawdown_weight:.4f}), "
        f"chromosome_digest={chromosome_digest(best.chromosome)}"
    )
    if len(text) > MAX_RATIONALE_LEN:
        text = text[: MAX_RATIONALE_LEN - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# EvolutionPipeline
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EvolutionPipeline:
    """Frozen coordinator. Holds the :class:`FitnessEvaluator` seam;
    every call to :meth:`evolve` is a pure function of its arguments.
    """

    evaluator: FitnessEvaluator

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, FitnessEvaluator):
            raise TypeError(
                "EvolutionPipeline.evaluator must implement the "
                "FitnessEvaluator Protocol, got "
                f"{type(self.evaluator).__name__}"
            )

    def evolve(
        self,
        *,
        specs: tuple[ParameterSpec, ...],
        config: EvolutionPipelineConfig,
        seed: int,
        ts_ns: int,
        proposal_id: str,
        initial_chromosome: StrategyChromosome | None = None,
        callback: EvolutionPipelineCallback | None = None,
    ) -> EvolutionPipelineResult:
        """Run a DE search and emit one :class:`EvolutionPipelineResult`.

        INV-13/14: this never deploys. The returned
        :attr:`EvolutionPipelineResult.proposal` is a typed
        :class:`PatchProposal` ready to be enqueued onto the bus by
        the operator (see :mod:`evolution_engine.patch_pipeline`).
        """

        # ---- Argument validation --------------------------------------
        self._validate_evolve_args(
            specs=specs,
            config=config,
            seed=seed,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
            initial_chromosome=initial_chromosome,
            callback=callback,
        )
        cb: EvolutionPipelineCallback = (
            callback if callback is not None else null_evolution_pipeline_callback()
        )

        # ---- Initial population ---------------------------------------
        seed_initial = (
            initial_chromosome
            if initial_chromosome is not None
            else _midpoint_chromosome(
                target_strategy_id=config.target_strategy_id,
                specs=specs,
            )
        )
        cb.on_evolution_start(
            ts_ns=ts_ns,
            config=config,
            dimensionality=len(specs),
        )

        n_evaluations = 0
        initial_individuals: list[IndividualResult] = []
        for i in range(config.population_size):
            if i == 0:
                chromosome = seed_initial
            else:
                chromosome = gaussian_mutate(
                    chromosome=seed_initial,
                    sigma=config.init_sigma,
                    seed=seed,
                    generation=0,
                    individual=i,
                )
            report = self._evaluate(
                chromosome=chromosome,
                seed=seed,
                ts_ns=ts_ns,
                generation_idx=0,
                individual_idx=i,
                cb=cb,
            )
            n_evaluations += 1
            scalar = report.fitness(config.fitness_drawdown_weight)
            ind = IndividualResult(
                chromosome=chromosome,
                fitness_report=report,
                fitness_scalar=scalar,
                generation_idx=0,
            )
            initial_individuals.append(ind)
        initial_population = tuple(initial_individuals)

        # Track running best (initial best, ties → smallest digest)
        running_best = initial_individuals[0]
        for ind in initial_individuals[1:]:
            if _is_better(candidate=ind, incumbent=running_best):
                running_best = ind

        # Build initial generation report (g = 0, n_replacements = 0)
        sorted_initial = self._sort_fittest_first(initial_individuals)
        initial_gen_report = GenerationReport(
            generation_idx=0,
            individuals=tuple(sorted_initial),
            best_fitness=running_best.fitness_scalar,
            mean_fitness=self._mean_fitness(initial_individuals),
            best_individual=running_best,
            n_replacements=0,
        )
        cb.on_generation_end(ts_ns=ts_ns, report=initial_gen_report)
        generations: list[GenerationReport] = [initial_gen_report]

        # ---- Generation loop ------------------------------------------
        population: list[IndividualResult] = list(initial_individuals)
        for g in range(1, config.max_generations + 1):
            n_replacements = 0
            for i in range(config.population_size):
                trial = self._build_trial(
                    population=population,
                    target_idx=i,
                    config=config,
                    seed=seed,
                    generation=g,
                    running_best=running_best,
                )
                report = self._evaluate(
                    chromosome=trial,
                    seed=seed,
                    ts_ns=ts_ns,
                    generation_idx=g,
                    individual_idx=i,
                    cb=cb,
                )
                n_evaluations += 1
                scalar = report.fitness(config.fitness_drawdown_weight)
                trial_ind = IndividualResult(
                    chromosome=trial,
                    fitness_report=report,
                    fitness_scalar=scalar,
                    generation_idx=g,
                )
                if _is_at_least_as_good(candidate=trial_ind, incumbent=population[i]):
                    if chromosome_digest(trial_ind.chromosome) != chromosome_digest(
                        population[i].chromosome
                    ):
                        n_replacements += 1
                    population[i] = trial_ind
                    if _is_better(candidate=trial_ind, incumbent=running_best):
                        running_best = trial_ind

            sorted_pop = self._sort_fittest_first(population)
            gen_report = GenerationReport(
                generation_idx=g,
                individuals=tuple(sorted_pop),
                best_fitness=running_best.fitness_scalar,
                mean_fitness=self._mean_fitness(population),
                best_individual=running_best,
                n_replacements=n_replacements,
            )
            cb.on_generation_end(ts_ns=ts_ns, report=gen_report)
            generations.append(gen_report)

        # ---- Build proposal ------------------------------------------
        generation_count = config.max_generations
        policy_digest = _compute_policy_digest(
            config=config,
            best=running_best,
            generation_count=generation_count,
            n_evaluations=n_evaluations,
            seed=seed,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
        )
        rationale = _build_rationale(
            best=running_best,
            generation_count=generation_count,
            n_evaluations=n_evaluations,
            config=config,
        )
        touchpoints = tuple(s.name for s in specs)

        # Canonical proposal_meta (sorted-key construction). User
        # cannot overlay these — they are provenance, not config.
        proposal_meta: dict[str, str] = {
            "best_chromosome_digest": chromosome_digest(running_best.chromosome),
            "best_fitness": repr(running_best.fitness_scalar),
            "best_max_drawdown": repr(running_best.fitness_report.max_drawdown),
            "best_pnl_mean": repr(running_best.fitness_report.pnl_mean),
            "generation_count": str(generation_count),
            "mutation_strategy": config.mutation_strategy.value,
            "n_evaluations": str(n_evaluations),
            "policy_digest": policy_digest,
            "population_size": str(config.population_size),
            "seed": str(seed),
            "source_module": PROPOSAL_SOURCE,
        }
        proposal = PatchProposal(
            ts_ns=ts_ns,
            patch_id=proposal_id,
            source=PROPOSAL_SOURCE,
            target_strategy=config.target_strategy_id,
            touchpoints=touchpoints,
            rationale=rationale,
            meta={k: proposal_meta[k] for k in sorted(proposal_meta)},
        )
        result = EvolutionPipelineResult(
            proposal=proposal,
            generations=tuple(generations),
            initial_population=initial_population,
            best_individual=running_best,
            policy_digest=policy_digest,
            n_evaluations=n_evaluations,
        )
        cb.on_evolution_end(ts_ns=ts_ns, result=result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_evolve_args(
        *,
        specs: tuple[ParameterSpec, ...],
        config: EvolutionPipelineConfig,
        seed: int,
        ts_ns: int,
        proposal_id: str,
        initial_chromosome: StrategyChromosome | None,
        callback: EvolutionPipelineCallback | None,
    ) -> None:
        if not isinstance(specs, tuple):
            raise EvolutionPipelineError("EvolutionPipeline.evolve.specs must be tuple")
        if not specs:
            raise EvolutionPipelineError("EvolutionPipeline.evolve.specs must be non-empty")
        for idx, spec in enumerate(specs):
            if not isinstance(spec, ParameterSpec):
                raise EvolutionPipelineError(
                    f"EvolutionPipeline.evolve.specs[{idx}] must be ParameterSpec"
                )
        if not isinstance(config, EvolutionPipelineConfig):
            raise EvolutionPipelineError(
                "EvolutionPipeline.evolve.config must be EvolutionPipelineConfig"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise EvolutionPipelineError("EvolutionPipeline.evolve.seed must be int")
        if seed < 0:
            raise EvolutionPipelineError(
                f"EvolutionPipeline.evolve.seed must be >= 0, got {seed!r}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise EvolutionPipelineError("EvolutionPipeline.evolve.ts_ns must be int")
        if ts_ns < 0:
            raise EvolutionPipelineError(
                f"EvolutionPipeline.evolve.ts_ns must be >= 0, got {ts_ns!r}"
            )
        if not isinstance(proposal_id, str):
            raise EvolutionPipelineError("EvolutionPipeline.evolve.proposal_id must be str")
        if not proposal_id:
            raise EvolutionPipelineError("EvolutionPipeline.evolve.proposal_id must be non-empty")
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise EvolutionPipelineError(
                f"EvolutionPipeline.evolve.proposal_id length "
                f"{len(proposal_id)} > {MAX_PROPOSAL_ID_LEN}"
            )
        if initial_chromosome is not None:
            if not isinstance(initial_chromosome, StrategyChromosome):
                raise EvolutionPipelineError(
                    "EvolutionPipeline.evolve.initial_chromosome must be StrategyChromosome | None"
                )
            if initial_chromosome.specs != specs:
                raise EvolutionPipelineError(
                    "EvolutionPipeline.evolve.initial_chromosome.specs must equal specs"
                )
            if initial_chromosome.strategy_id != config.target_strategy_id:
                raise EvolutionPipelineError(
                    "EvolutionPipeline.evolve.initial_chromosome.strategy_id "
                    "must equal config.target_strategy_id"
                )
        if callback is not None and not isinstance(callback, EvolutionPipelineCallback):
            raise EvolutionPipelineError(
                "EvolutionPipeline.evolve.callback must implement "
                "EvolutionPipelineCallback Protocol"
            )

    def _evaluate(
        self,
        *,
        chromosome: StrategyChromosome,
        seed: int,
        ts_ns: int,
        generation_idx: int,
        individual_idx: int,
        cb: EvolutionPipelineCallback,
    ) -> FitnessReport:
        """Evaluate one chromosome through the injected
        :class:`FitnessEvaluator`. Fail-fast on bad return type."""

        eval_seed = _seed_int(seed, generation_idx, individual_idx, _EVAL_SALT)
        report = self.evaluator.evaluate(
            chromosome=chromosome,
            seed=eval_seed,
            ts_ns=ts_ns,
        )
        if not isinstance(report, FitnessReport):
            raise EvolutionPipelineEvaluationError(
                f"FitnessEvaluator.evaluate must return FitnessReport, got {type(report).__name__}"
            )
        cb.on_individual_evaluated(
            ts_ns=ts_ns,
            generation_idx=generation_idx,
            individual_idx=individual_idx,
            chromosome=chromosome,
            fitness_report=report,
        )
        return report

    @staticmethod
    def _build_trial(
        *,
        population: list[IndividualResult],
        target_idx: int,
        config: EvolutionPipelineConfig,
        seed: int,
        generation: int,
        running_best: IndividualResult,
    ) -> StrategyChromosome:
        """Produce one trial chromosome via mutation + binomial crossover."""

        n = len(population)
        target = population[target_idx].chromosome

        if config.mutation_strategy is MutationStrategy.DE_RAND_1:
            indices = _sample_distinct(
                n=n,
                exclude=target_idx,
                count=3,
                key=_seed_int(seed, generation, target_idx, _INDEX_SALT),
            )
            a = population[indices[0]].chromosome
            b = population[indices[1]].chromosome
            c = population[indices[2]].chromosome
            donor = de_rand_1(
                a=a,
                b=b,
                c=c,
                F=config.F,
                seed=seed,
                generation=generation,
                individual=target_idx,
            )
        elif config.mutation_strategy is MutationStrategy.DE_CURRENT_TO_BEST_1:
            indices = _sample_distinct(
                n=n,
                exclude=target_idx,
                count=2,
                key=_seed_int(seed, generation, target_idx, _INDEX_SALT),
            )
            a = population[indices[0]].chromosome
            b = population[indices[1]].chromosome
            donor = de_current_to_best_1(
                target=target,
                best=running_best.chromosome,
                a=a,
                b=b,
                F=config.F,
                seed=seed,
                generation=generation,
                individual=target_idx,
            )
        else:  # pragma: no cover - enum exhaustiveness
            raise EvolutionPipelineError(f"Unknown mutation strategy: {config.mutation_strategy!r}")

        # Binomial crossover always wraps the donor.
        return de_binomial_crossover(
            target=target,
            donor=donor,
            CR=config.CR,
            seed=seed,
            generation=generation,
            individual=target_idx,
        )

    @staticmethod
    def _sort_fittest_first(
        individuals: list[IndividualResult],
    ) -> list[IndividualResult]:
        """Sort individuals by ``-fitness_scalar`` then by chromosome
        digest ascending (INV-15 tie-break)."""

        return sorted(
            individuals,
            key=lambda ind: (
                -ind.fitness_scalar,
                chromosome_digest(ind.chromosome),
            ),
        )

    @staticmethod
    def _mean_fitness(individuals: list[IndividualResult]) -> float:
        """Arithmetic mean of fitness scalars (deterministic float
        accumulation in iteration order)."""

        if not individuals:
            return 0.0
        total = 0.0
        for ind in individuals:
            total += ind.fitness_scalar
        return total / float(len(individuals))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
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
]
