# ADAPTED FROM: nnaisense/evotorch
# (evotorch/algorithms/cmaes.py — CMA-ES (Hansen & Ostermeier 2001) sampler,
#  separable variant per Ros & Hansen 2008; evotorch/core.py — Problem /
#  SolutionBatch fitness contract; evotorch/algorithms/searchalgorithm.py —
#  generation-loop / callback shape.)
"""A-02.2 — CMAESOptimizer: governance-gated CMA-ES strategy search.

evotorch's :class:`CMAES` walks a Gaussian search distribution
``N(mean, sigma**2 * C)`` over the parameter vector of a
:class:`Problem`. Trading strategy evolution does **not** call it
directly: a candidate strategy is a *structural mutation* of the
running config and that path goes through
:class:`evolution_engine.patch_pipeline`. This module is the offline
harness that runs the CMA-ES sampler against a caller-supplied
:class:`FitnessEvaluator`, captures the best individual, and emits a
typed :class:`~core.contracts.learning.PatchProposal` for governance
approval. INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-stdlib coordinator + frozen value objects. No numpy / no
  evotorch / no torch import — the sampler implements the
  separable-CMA-ES (sep-CMA-ES) variant from Ros & Hansen 2008 in
  pure Python (diagonal covariance, ``O(n)`` per generation, no
  eigendecomposition). ``NEW_PIP_DEPENDENCIES = ()``.
* OFFLINE_ONLY tier. The optimizer reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. It produces one
  :class:`CMAESResult` record and stops.
* INV-15 byte-identical replays. ``CMAESOptimizer.evolve(...)``
  with identical ``config`` / ``specs`` / ``initial_chromosome`` /
  ``seed`` / ``ts_ns`` / ``proposal_id`` / ``evaluator`` returns
  identical :class:`CMAESResult` records. The Gaussian sampler is
  seeded by a stateless splitmix64 / Box-Muller stream keyed off the
  caller-supplied ``seed`` (mirrors the gym_env / S-02.2 latency
  pattern). Selection ties are broken deterministically by
  :func:`~evolution_engine.genetic.strategy_chromosome.chromosome_digest`.
* No clock reads. Caller supplies ``ts_ns`` (mirrors the S-06 typed
  agent + S-12 LiteLLM router + A-01.2 sandbox pattern).

What survives from upstream
---------------------------

* The ``(mu, lambda)`` weighted-recombination ES from Hansen &
  Ostermeier 2001 — selection of top ``mu = lambda // 2`` solutions
  with logarithmic weights ``w_i = log((mu+1)/(i+1))`` normalised to
  unit sum, and the ``mu_eff = 1 / sum(w_i**2)`` effective parent
  count.
* The cumulative step-size adaptation (CSA) path ``p_sigma`` and
  exponential update ``sigma *= exp((c_sigma/d_sigma)*
  (||p_sigma||/E[||N(0,I)||] - 1))``.
* The rank-one covariance update via the path ``p_c`` and the
  Heaviside-style ``h_sigma`` indicator that prevents covariance
  blow-up early in optimisation.
* The separable variant from Ros & Hansen 2008: covariance is
  restricted to the diagonal so the per-generation cost is ``O(n)``
  not ``O(n^3)`` (the eigendecomposition path is the only place
  upstream pulls numpy / torch). Learning rates ``c_1`` and ``c_mu``
  are scaled by ``(n+2)/3`` per the sep-CMA-ES recipe.

What we replaced
----------------

* numpy / torch tensor algebra → pure Python ``list[float]`` /
  ``math``-based loops. Per-generation cost is ``O(lambda * n)`` for
  sampling, ``O(lambda * log lambda)`` for selection, and ``O(n)``
  for the path / sigma / diag-C updates.
* numpy.random.RandomState / torch.Generator → stateless splitmix64
  hash + Box-Muller (mirrors :mod:`evolution_engine.gym_env`). No
  global PRNG state, no clock seeding.
* CMAES checkpoint files → :attr:`CMAESResult.policy_digest` (a
  16-hex-char content hash of the optimizer state at the best
  individual). Operators wire :class:`PatchProposal` onto the bus
  via :mod:`evolution_engine.patch_pipeline` — the optimizer never
  writes a file or mutates a registry.
* CMAES dynamic re-sampling on bound rejection → soft clipping in
  encoded space combined with kind-aware re-clip in decoded space
  (bounded by :func:`~evolution_engine.genetic.strategy_chromosome.unpack`).
  This keeps replay determinism: every sample is feasible by
  construction, so the population size never depends on a
  caller-supplied retry budget.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE tier — no IO, no clock, no global state, no PRNG (the
  sampler's PRNG is seeded by caller-supplied seed and never reads
  the wall clock). AST tests pin the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``intelligence_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-13/14 — :meth:`CMAESOptimizer.evolve` returns one
  :class:`CMAESResult` containing one :class:`PatchProposal`; it
  does **not** mutate any external registry or governance ledger.
  Wiring the proposal onto the bus is the operator's job (mirrors
  how :mod:`learning_engine.lanes` emits ``LearningUpdate`` records
  without applying them).
* INV-15 — :attr:`CMAESResult.policy_digest` is a deterministic
  function of the inputs (BLAKE2b over a canonical text projection
  of the best chromosome digest, the per-generation stats, and the
  config). 3-run identical-input replay equality is pinned in tests.
* Defensive caps:
  - :data:`MAX_POPULATION_SIZE` 256 hard ceiling on
    ``CMAESConfig.population_size``.
  - :data:`MAX_GENERATIONS` 1000 hard ceiling on
    ``CMAESConfig.max_generations``.
  - :data:`MAX_TOTAL_EVALUATIONS` 100_000 hard ceiling on
    ``population_size * max_generations``.
  - :data:`MAX_PROPOSAL_ID_LEN` 256 chars on the caller-supplied
    ``proposal_id`` (mirrors A-01.2 sandbox).

Refs:
- ``DIX_MASTER_CANONICAL.md`` lines 811–849 (A-02 evotorch spec).
- ``evolution_engine/genetic/strategy_chromosome.py`` (PR #294 —
  StrategyChromosome / pack / unpack / chromosome_digest).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping
from dataclasses import field
from typing import Protocol, runtime_checkable

from core.contracts.learning import PatchProposal
from evolution_engine.genetic.strategy_chromosome import (
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
    pack,
    unpack,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No new pip dependencies — sep-CMA-ES is reproduced in pure stdlib."""

MIN_POPULATION_SIZE: int = 4
"""Lowest population size that yields ``mu = lambda // 2 >= 2``
(weighted recombination requires at least two parents)."""

MAX_POPULATION_SIZE: int = 256
"""Upper bound on population size — sep-CMA-ES is ``O(lambda * n)`` per
generation, so this caps per-step work even for the
:data:`MAX_PARAMETERS_PER_CHROMOSOME` worst case."""

MAX_GENERATIONS: int = 1000
"""Upper bound on generation count — combined with population size, this
caps total evaluations against the evaluator."""

MAX_TOTAL_EVALUATIONS: int = 100_000
"""Hard ceiling on ``population_size * max_generations`` regardless of
the per-axis caps above."""

MAX_PROPOSAL_ID_LEN: int = 256
"""Mirrors A-01.2 sandbox cap on caller-supplied
:attr:`PatchProposal.patch_id`."""

MAX_RATIONALE_LEN: int = 1024
"""Cap on the auto-generated proposal rationale string."""

PROPOSAL_SOURCE: str = "evolution_engine.genetic.cmaes_optimizer"
"""Identifies this adapter as the producer of the proposal."""

DEFAULT_SIGMA_INIT: float = 0.5
"""Default initial step size in encoded space — half a unit on each
axis. Per Hansen 2016, this matches the recommended starting sigma when
parameters are normalised to ``[0, 1]``-ish ranges; for DIX the encoded
bounds are typically of unit-ish width too."""

DEFAULT_DRAWDOWN_PENALTY: float = 0.5
"""Default coefficient on the drawdown penalty: ``fitness = pnl_mean -
drawdown_penalty * max_drawdown``. The same weighting is used by
:mod:`learning_engine.analytics.regime_stats` for the per-regime
sharpe-proxy, so the two surfaces stay comparable."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CMAESConfigError(ValueError):
    """Raised when the caller passes an invalid combination of args to
    :class:`CMAESConfig` / :meth:`CMAESOptimizer.evolve`."""


class CMAESEvaluationError(RuntimeError):
    """Raised when the injected :class:`FitnessEvaluator` returns an
    invalid :class:`FitnessReport` (wrong type, non-finite scalar,
    etc.). The optimizer is fail-fast: a bad evaluation is never
    silently dropped because that would hide replay divergence."""


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CMAESConfig:
    """Frozen configuration for one :meth:`CMAESOptimizer.evolve` call.

    ``target_strategy_id`` is the parent strategy whose parameters this
    run is searching over; it is stamped into the resulting
    :attr:`PatchProposal.target_strategy` and into the
    :attr:`StrategyChromosome.strategy_id` of every individual.
    """

    target_strategy_id: str
    population_size: int
    max_generations: int
    sigma_init: float = DEFAULT_SIGMA_INIT
    fitness_drawdown_weight: float = DEFAULT_DRAWDOWN_PENALTY

    def __post_init__(self) -> None:
        if not isinstance(self.target_strategy_id, str):
            raise CMAESConfigError("CMAESConfig.target_strategy_id must be str")
        if not self.target_strategy_id:
            raise CMAESConfigError("CMAESConfig.target_strategy_id must be non-empty")
        if not isinstance(self.population_size, int) or isinstance(self.population_size, bool):
            raise CMAESConfigError("CMAESConfig.population_size must be int")
        if self.population_size < MIN_POPULATION_SIZE or self.population_size > MAX_POPULATION_SIZE:
            raise CMAESConfigError(
                f"CMAESConfig.population_size must be in "
                f"[{MIN_POPULATION_SIZE}, {MAX_POPULATION_SIZE}], "
                f"got {self.population_size!r}"
            )
        if not isinstance(self.max_generations, int) or isinstance(self.max_generations, bool):
            raise CMAESConfigError("CMAESConfig.max_generations must be int")
        if self.max_generations < 1 or self.max_generations > MAX_GENERATIONS:
            raise CMAESConfigError(
                f"CMAESConfig.max_generations must be in "
                f"[1, {MAX_GENERATIONS}], got {self.max_generations!r}"
            )
        if self.population_size * self.max_generations > MAX_TOTAL_EVALUATIONS:
            raise CMAESConfigError(
                f"CMAESConfig: population_size * max_generations = "
                f"{self.population_size * self.max_generations} > "
                f"{MAX_TOTAL_EVALUATIONS}"
            )
        if not isinstance(self.sigma_init, (int, float)) or isinstance(self.sigma_init, bool):
            raise CMAESConfigError("CMAESConfig.sigma_init must be int|float")
        if not math.isfinite(self.sigma_init) or self.sigma_init <= 0.0:
            raise CMAESConfigError(
                f"CMAESConfig.sigma_init must be a positive finite float, got {self.sigma_init!r}"
            )
        if not isinstance(self.fitness_drawdown_weight, (int, float)) or isinstance(
            self.fitness_drawdown_weight, bool
        ):
            raise CMAESConfigError("CMAESConfig.fitness_drawdown_weight must be int|float")
        if not math.isfinite(self.fitness_drawdown_weight) or self.fitness_drawdown_weight < 0.0:
            raise CMAESConfigError(
                f"CMAESConfig.fitness_drawdown_weight must be a "
                f"non-negative finite float, got "
                f"{self.fitness_drawdown_weight!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class FitnessReport:
    """Result of evaluating one chromosome on the simulation harness.

    The evaluator is the only seam between this module and the trading
    simulation (or any other backtest path). The optimizer collapses
    the report to a single scalar via
    ``fitness = pnl_mean - drawdown_weight * max_drawdown`` (mirrors
    the per-regime sharpe-proxy weighting used by S-10.2
    :mod:`learning_engine.analytics.regime_stats`).
    """

    pnl_mean: float
    max_drawdown: float
    n_samples: int
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pnl_mean, (int, float)) or isinstance(self.pnl_mean, bool):
            raise CMAESEvaluationError("FitnessReport.pnl_mean must be int|float")
        if not math.isfinite(self.pnl_mean):
            raise CMAESEvaluationError(
                f"FitnessReport.pnl_mean must be finite, got {self.pnl_mean!r}"
            )
        if not isinstance(self.max_drawdown, (int, float)) or isinstance(self.max_drawdown, bool):
            raise CMAESEvaluationError("FitnessReport.max_drawdown must be int|float")
        if not math.isfinite(self.max_drawdown) or self.max_drawdown < 0.0:
            raise CMAESEvaluationError(
                f"FitnessReport.max_drawdown must be a non-negative "
                f"finite float, got {self.max_drawdown!r}"
            )
        if not isinstance(self.n_samples, int) or isinstance(self.n_samples, bool):
            raise CMAESEvaluationError("FitnessReport.n_samples must be int")
        if self.n_samples < 1:
            raise CMAESEvaluationError(
                f"FitnessReport.n_samples must be >= 1, got {self.n_samples!r}"
            )
        if not isinstance(self.meta, Mapping):
            raise CMAESEvaluationError("FitnessReport.meta must be Mapping")
        for mk, mv in self.meta.items():
            if not isinstance(mk, str) or not isinstance(mv, str):
                raise CMAESEvaluationError("FitnessReport.meta must map str -> str")

    def fitness(self, drawdown_weight: float) -> float:
        """Collapse to a scalar: ``pnl_mean - w * max_drawdown``."""

        return self.pnl_mean - drawdown_weight * self.max_drawdown


@dataclasses.dataclass(frozen=True, slots=True)
class IndividualResult:
    """One sample from the search distribution paired with its fitness.

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
    """Snapshot of one generation: individuals (sorted fittest-first),
    headline stats, and the running best individual at this point.
    """

    generation_idx: int
    individuals: tuple[IndividualResult, ...]
    best_fitness: float
    mean_fitness: float
    best_individual: IndividualResult

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
        if not math.isfinite(self.best_fitness):
            raise ValueError(
                f"GenerationReport.best_fitness must be finite, got {self.best_fitness!r}"
            )
        if not math.isfinite(self.mean_fitness):
            raise ValueError(
                f"GenerationReport.mean_fitness must be finite, got {self.mean_fitness!r}"
            )
        if not isinstance(self.best_individual, IndividualResult):
            raise TypeError("GenerationReport.best_individual must be IndividualResult")


@dataclasses.dataclass(frozen=True, slots=True)
class CMAESResult:
    """Output of :meth:`CMAESOptimizer.evolve`.

    The :class:`PatchProposal` carries the governance-shaped payload
    (``patch_id``, ``source``, ``target_strategy``, ``touchpoints``,
    ``rationale``, ``meta``); the :class:`GenerationReport` tuple and
    :attr:`policy_digest` carry the audit metadata operators consult
    when reviewing the proposal in the dashboard.
    """

    proposal: PatchProposal
    generations: tuple[GenerationReport, ...]
    best_individual: IndividualResult
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, PatchProposal):
            raise TypeError(
                f"CMAESResult.proposal must be PatchProposal, got {type(self.proposal).__name__}"
            )
        if not isinstance(self.generations, tuple):
            raise TypeError(
                f"CMAESResult.generations must be tuple, got {type(self.generations).__name__}"
            )
        if not self.generations:
            raise ValueError("CMAESResult.generations must be non-empty")
        for idx, gen in enumerate(self.generations):
            if not isinstance(gen, GenerationReport):
                raise TypeError(f"CMAESResult.generations[{idx}] must be GenerationReport")
        if not isinstance(self.best_individual, IndividualResult):
            raise TypeError("CMAESResult.best_individual must be IndividualResult")
        if not isinstance(self.policy_digest, str):
            raise TypeError("CMAESResult.policy_digest must be str")
        if len(self.policy_digest) != 16:
            raise ValueError(
                f"CMAESResult.policy_digest must be 16 hex chars, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                f"CMAESResult.policy_digest must be lowercase hex, got {self.policy_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams (the only place the optimizer touches the outside)
# ---------------------------------------------------------------------------


@runtime_checkable
class FitnessEvaluator(Protocol):
    """Caller-supplied fitness evaluator.

    The Protocol is the only seam between this module and the trading
    simulation. Production wires a thin adapter onto
    :mod:`simulation.parallel_runner` (SIM-07); tests inject a
    deterministic fake. The contract is single-shot: the evaluator
    fully consumes one chromosome and returns one
    :class:`FitnessReport`. Anything richer (live eval, intermediate
    checkpoints) is the evaluator's concern.

    Determinism: the evaluator MUST be a pure function of
    ``(chromosome, seed, ts_ns)`` for INV-15 to hold. The optimizer
    forwards a :func:`_splitmix64`-derived per-individual seed so the
    evaluator can downstream the determinism contract without sharing
    state with the optimizer.
    """

    def evaluate(
        self,
        *,
        chromosome: StrategyChromosome,
        seed: int,
        ts_ns: int,
    ) -> FitnessReport: ...


@runtime_checkable
class CMAESCallback(Protocol):
    """Optional lifecycle callback (collapsed into one Protocol so the
    AST tests can pin "no top-level evotorch import"). The default is
    :func:`null_cmaes_callback`."""

    def on_evolution_start(
        self,
        *,
        ts_ns: int,
        config: CMAESConfig,
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
        result: CMAESResult,
    ) -> None: ...


def null_cmaes_callback() -> CMAESCallback:
    """Default no-op :class:`CMAESCallback`. Pure stdlib; safe to use
    as the default in :meth:`CMAESOptimizer.evolve`."""

    return _NullCallback()


@dataclasses.dataclass(frozen=True, slots=True)
class _NullCallback:
    def on_evolution_start(self, *, ts_ns: int, config: CMAESConfig, dimensionality: int) -> None:
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

    def on_generation_end(self, *, ts_ns: int, report: GenerationReport) -> None:
        return None

    def on_evolution_end(self, *, ts_ns: int, result: CMAESResult) -> None:
        return None


# ---------------------------------------------------------------------------
# Stateless deterministic PRNG (mirrors gym_env / S-02.2 latency)
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    """Stateless 64-bit hash. Used to derive every per-sample seed and
    every Box-Muller uniform. Pure stdlib; identical implementation to
    :func:`evolution_engine.gym_env._splitmix64`."""

    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _uniform01(*key: int) -> float:
    """Deterministic uniform in ``(0, 1]`` keyed off an integer tuple.

    Folds the key through splitmix64 and rescales the bottom 53 bits
    to a double. The result is strictly positive (offset by ``1`` /
    ``2**53``) so callers can pass it through ``math.log`` without
    needing to guard against zero — required by the Box-Muller
    transform.
    """

    h = 0
    for k in key:
        h = _splitmix64(h ^ (k & 0xFFFFFFFFFFFFFFFF))
    # 53-bit mantissa range [1, 2**53] -> (0, 1]
    return ((h >> 11) + 1) / (1 << 53)


def _gauss_pair(
    *,
    seed: int,
    generation: int,
    individual: int,
    dimension: int,
) -> tuple[float, float]:
    """One Box-Muller pair seeded by ``(seed, generation, individual,
    dimension)``. Returns two independent N(0, 1) samples; callers
    consume them in pairs for efficiency."""

    u1 = _uniform01(seed, generation, individual, dimension, 0)
    u2 = _uniform01(seed, generation, individual, dimension, 1)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return radius * math.cos(angle), radius * math.sin(angle)


def _gauss_vector(
    *,
    seed: int,
    generation: int,
    individual: int,
    dimensionality: int,
) -> tuple[float, ...]:
    """Sample one length-``n`` vector of i.i.d. N(0, 1) values.

    Box-Muller produces samples in pairs, so for odd ``n`` we discard
    the last sin component (this is standard, mirrors numpy's
    ``standard_normal``)."""

    out: list[float] = []
    paired = dimensionality // 2
    for j in range(paired):
        a, b = _gauss_pair(
            seed=seed,
            generation=generation,
            individual=individual,
            dimension=j,
        )
        out.append(a)
        out.append(b)
    if dimensionality % 2 == 1:
        a, _ = _gauss_pair(
            seed=seed,
            generation=generation,
            individual=individual,
            dimension=paired,
        )
        out.append(a)
    return tuple(out)


# ---------------------------------------------------------------------------
# Encoded-space helpers
# ---------------------------------------------------------------------------


def _encoded_bounds(
    specs: tuple[ParameterSpec, ...],
) -> tuple[tuple[float, float], ...]:
    """Per-spec ``(low_enc, high_enc)`` bounds in encoded (CMA-ES)
    space. ``CONTINUOUS`` and ``INTEGER`` pass through; ``LOG_CONTINUOUS``
    is on a ``log10`` scale (mirrors :func:`pack`'s LOG_CONTINUOUS
    branch).
    """

    bounds: list[tuple[float, float]] = []
    for spec in specs:
        if spec.kind is ParameterKind.LOG_CONTINUOUS:
            bounds.append((math.log10(spec.low), math.log10(spec.high)))
        else:
            bounds.append((spec.low, spec.high))
    return tuple(bounds)


def _clip_encoded(
    vec: list[float],
    bounds: tuple[tuple[float, float], ...],
) -> list[float]:
    """Per-axis clip into encoded bounds. In-place on ``vec`` for
    speed (the optimizer only ever calls this on freshly-allocated
    sample buffers)."""

    for i in range(len(vec)):
        lo, hi = bounds[i]
        if vec[i] < lo:
            vec[i] = lo
        elif vec[i] > hi:
            vec[i] = hi
    return vec


def _encoded_to_decoded(
    specs: tuple[ParameterSpec, ...],
    encoded: tuple[float, ...],
) -> tuple[float, ...]:
    """Materialise the decoded ``StrategyChromosome.values`` tuple from
    an encoded sample. Goes through :func:`unpack` so the integer
    rounding + bounds-clip path is the canonical one (no parallel
    decode logic to keep in sync)."""

    decoded_map = unpack(specs, encoded)
    return tuple(decoded_map[s.name] for s in specs)


def _initial_mean_encoded(
    specs: tuple[ParameterSpec, ...],
    initial_chromosome: StrategyChromosome | None,
) -> list[float]:
    """Pick the centroid of the initial search distribution.

    If the caller supplies an ``initial_chromosome`` we project its
    decoded values through :func:`pack` to get the encoded mean. Else
    we use the midpoint of each axis in encoded space (mirrors
    evotorch's ``Problem.initial_solution`` default when no candidate
    is provided)."""

    if initial_chromosome is not None:
        encoded = pack(specs, initial_chromosome.to_mapping())
        return list(encoded)
    mean: list[float] = []
    for lo, hi in _encoded_bounds(specs):
        mean.append(0.5 * (lo + hi))
    return mean


# ---------------------------------------------------------------------------
# Digest + rationale helpers
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    config: CMAESConfig,
    best: IndividualResult,
    generation_count: int,
    seed: int,
    ts_ns: int,
    proposal_id: str,
) -> str:
    """16-hex BLAKE2b-8 over a canonical text projection of the
    optimizer state. Deterministic by construction — two runs with
    identical inputs produce the same digest. INV-15."""

    parts: list[str] = [
        "cmaes_optimizer/v1",
        f"target={config.target_strategy_id}",
        f"pop={config.population_size}",
        f"gens={config.max_generations}",
        f"sigma_init={config.sigma_init!r}",
        f"dd_weight={config.fitness_drawdown_weight!r}",
        f"seed={seed}",
        f"ts_ns={ts_ns}",
        f"proposal_id={proposal_id}",
        f"generation_count={generation_count}",
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
    config: CMAESConfig,
) -> str:
    """Build the auto-generated :attr:`PatchProposal.rationale`. Length
    is bounded by :data:`MAX_RATIONALE_LEN`; if the formatted string
    exceeds that bound it is truncated with a trailing marker so the
    digest still distinguishes truncated runs."""

    text = (
        f"sep-CMA-ES search over {best.chromosome.dimensionality} "
        f"parameters: "
        f"{generation_count} generations of pop="
        f"{config.population_size}, "
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
# CMAESOptimizer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CMAESOptimizer:
    """Frozen coordinator. Holds the :class:`FitnessEvaluator` seam;
    every call to :meth:`evolve` is a pure function of its arguments."""

    evaluator: FitnessEvaluator

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, FitnessEvaluator):
            raise TypeError(
                "CMAESOptimizer.evaluator must implement the "
                "FitnessEvaluator Protocol, got "
                f"{type(self.evaluator).__name__}"
            )

    def evolve(
        self,
        *,
        specs: tuple[ParameterSpec, ...],
        config: CMAESConfig,
        seed: int,
        ts_ns: int,
        proposal_id: str,
        initial_chromosome: StrategyChromosome | None = None,
        callback: CMAESCallback | None = None,
    ) -> CMAESResult:
        """Run a sep-CMA-ES search and emit one :class:`CMAESResult`.

        INV-13/14: this never deploys. The returned
        :attr:`CMAESResult.proposal` is a typed
        :class:`PatchProposal` ready to be enqueued onto the bus by
        the operator (see :mod:`evolution_engine.patch_pipeline`).
        """

        # ---- Argument validation --------------------------------------
        if not isinstance(specs, tuple):
            raise CMAESConfigError("CMAESOptimizer.evolve.specs must be tuple")
        if not specs:
            raise CMAESConfigError("CMAESOptimizer.evolve.specs must be non-empty")
        for idx, spec in enumerate(specs):
            if not isinstance(spec, ParameterSpec):
                raise CMAESConfigError(f"CMAESOptimizer.evolve.specs[{idx}] must be ParameterSpec")
        if not isinstance(config, CMAESConfig):
            raise CMAESConfigError("CMAESOptimizer.evolve.config must be CMAESConfig")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise CMAESConfigError("CMAESOptimizer.evolve.seed must be int")
        if seed < 0:
            raise CMAESConfigError(f"CMAESOptimizer.evolve.seed must be >= 0, got {seed!r}")
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise CMAESConfigError("CMAESOptimizer.evolve.ts_ns must be int")
        if ts_ns < 0:
            raise CMAESConfigError(f"CMAESOptimizer.evolve.ts_ns must be >= 0, got {ts_ns!r}")
        if not isinstance(proposal_id, str):
            raise CMAESConfigError("CMAESOptimizer.evolve.proposal_id must be str")
        if not proposal_id:
            raise CMAESConfigError("CMAESOptimizer.evolve.proposal_id must be non-empty")
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise CMAESConfigError(
                f"CMAESOptimizer.evolve.proposal_id must be <= "
                f"{MAX_PROPOSAL_ID_LEN} chars, got {len(proposal_id)}"
            )
        if initial_chromosome is not None:
            if not isinstance(initial_chromosome, StrategyChromosome):
                raise CMAESConfigError(
                    "CMAESOptimizer.evolve.initial_chromosome must be StrategyChromosome | None"
                )
            if initial_chromosome.specs != specs:
                raise CMAESConfigError(
                    "CMAESOptimizer.evolve.initial_chromosome.specs must match specs"
                )
            if initial_chromosome.strategy_id != config.target_strategy_id:
                raise CMAESConfigError(
                    "CMAESOptimizer.evolve.initial_chromosome.strategy_id "
                    "must match config.target_strategy_id"
                )

        cb = callback if callback is not None else null_cmaes_callback()
        if not isinstance(cb, CMAESCallback):
            raise CMAESConfigError(
                "CMAESOptimizer.evolve.callback must implement the CMAESCallback Protocol"
            )

        # ---- sep-CMA-ES setup -----------------------------------------
        n = len(specs)
        bounds = _encoded_bounds(specs)
        lam = config.population_size
        mu = lam // 2

        # Logarithmic positive recombination weights, normalised.
        raw_w = [math.log(mu + 1.0) - math.log(i + 1.0) for i in range(mu)]
        w_sum = math.fsum(raw_w)
        weights = [w / w_sum for w in raw_w]
        mu_eff = 1.0 / math.fsum(w * w for w in weights)

        # Sep-CMA-ES learning rates (Hansen 2016 + Ros & Hansen 2008).
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1_full = 2.0 / ((n + 1.3) * (n + 1.3) + mu_eff)
        c_mu_full = min(
            1.0 - c_1_full,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) * (n + 2.0) + mu_eff),
        )
        # Sep-CMA-ES scaling (Ros & Hansen 2008 §3): boost rates by
        # (n+2)/3 because the diagonal model has fewer DoF than the
        # full covariance.
        sep_scale = (n + 2.0) / 3.0
        c_1 = min(1.0, c_1_full * sep_scale)
        c_mu = min(1.0 - c_1, c_mu_full * sep_scale)

        # Expected length of N(0, I) approximation.
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # State vectors.
        mean = _initial_mean_encoded(specs, initial_chromosome)
        # Project the caller's mean into encoded bounds in case the
        # initial chromosome sat exactly on the boundary and floating
        # point pushed it a hair past.
        _clip_encoded(mean, bounds)
        diag_C = [1.0] * n  # diagonal of covariance
        p_sigma = [0.0] * n
        p_c = [0.0] * n
        sigma = float(config.sigma_init)

        cb.on_evolution_start(ts_ns=ts_ns, config=config, dimensionality=n)

        # ---- Generation loop ------------------------------------------
        generation_reports: list[GenerationReport] = []
        running_best: IndividualResult | None = None

        for gen_idx in range(config.max_generations):
            # 1) Sample lambda offspring.
            zs: list[tuple[float, ...]] = []
            xs_decoded_values: list[tuple[float, ...]] = []
            xs_chromosomes: list[StrategyChromosome] = []
            for k in range(lam):
                z = _gauss_vector(
                    seed=seed,
                    generation=gen_idx,
                    individual=k,
                    dimensionality=n,
                )
                zs.append(z)
                # x = m + sigma * sqrt(diag_C) * z
                x_enc: list[float] = [
                    mean[i] + sigma * math.sqrt(diag_C[i]) * z[i] for i in range(n)
                ]
                _clip_encoded(x_enc, bounds)
                decoded_values = _encoded_to_decoded(specs, tuple(x_enc))
                chrom = StrategyChromosome(
                    strategy_id=config.target_strategy_id,
                    specs=specs,
                    values=decoded_values,
                    version=gen_idx,
                    meta={
                        "generation": str(gen_idx),
                        "individual": str(k),
                    },
                )
                xs_decoded_values.append(decoded_values)
                xs_chromosomes.append(chrom)

            # 2) Evaluate every individual against the caller's fitness
            #    surface. Per-individual seed forwards the PRNG family
            #    so the evaluator can stay deterministic without
            #    sharing state with the optimizer.
            individuals: list[IndividualResult] = []
            for k in range(lam):
                ind_seed = _splitmix64(_splitmix64(seed) ^ ((gen_idx << 32) | k))
                report = self.evaluator.evaluate(
                    chromosome=xs_chromosomes[k],
                    seed=ind_seed,
                    ts_ns=ts_ns,
                )
                if not isinstance(report, FitnessReport):
                    raise CMAESEvaluationError(
                        "FitnessEvaluator.evaluate must return a "
                        "FitnessReport, got "
                        f"{type(report).__name__}"
                    )
                scalar = report.fitness(config.fitness_drawdown_weight)
                if not math.isfinite(scalar):
                    raise CMAESEvaluationError(
                        f"FitnessEvaluator returned non-finite scalar "
                        f"{scalar!r} for individual {k} at generation "
                        f"{gen_idx}"
                    )
                ind = IndividualResult(
                    chromosome=xs_chromosomes[k],
                    fitness_report=report,
                    fitness_scalar=scalar,
                    generation_idx=gen_idx,
                )
                individuals.append(ind)
                cb.on_individual_evaluated(
                    ts_ns=ts_ns,
                    generation_idx=gen_idx,
                    individual_idx=k,
                    chromosome=ind.chromosome,
                    fitness_report=report,
                )

            # 3) Sort fittest-first; tie-break by chromosome digest
            #    ascending so two runs with identical inputs always
            #    rank ties in the same order (INV-15).
            order = sorted(
                range(lam),
                key=lambda k: (
                    -individuals[k].fitness_scalar,
                    chromosome_digest(individuals[k].chromosome),
                ),
            )
            sorted_individuals = tuple(individuals[k] for k in order)
            mean_fitness = math.fsum(ind.fitness_scalar for ind in individuals) / float(lam)
            best_in_gen = sorted_individuals[0]

            if (
                running_best is None
                or best_in_gen.fitness_scalar > running_best.fitness_scalar
                or (
                    best_in_gen.fitness_scalar == running_best.fitness_scalar
                    and chromosome_digest(best_in_gen.chromosome)
                    < chromosome_digest(running_best.chromosome)
                )
            ):
                running_best = best_in_gen

            report = GenerationReport(
                generation_idx=gen_idx,
                individuals=sorted_individuals,
                best_fitness=best_in_gen.fitness_scalar,
                mean_fitness=mean_fitness,
                best_individual=running_best,
            )
            generation_reports.append(report)
            cb.on_generation_end(ts_ns=ts_ns, report=report)

            # 4) sep-CMA-ES update. Compute the weighted mean of the
            #    selected mu offspring in z-space (the standard
            #    Mahalanobis-normalised step).
            selected_indices = order[:mu]
            z_w = [
                math.fsum(weights[i] * zs[selected_indices[i]][j] for i in range(mu))
                for j in range(n)
            ]

            # New mean: m + sigma * sqrt(diag_C) * z_w (= weighted mean
            # of the selected x's, by linearity of the encoding).
            new_mean = [mean[j] + sigma * math.sqrt(diag_C[j]) * z_w[j] for j in range(n)]
            _clip_encoded(new_mean, bounds)
            mean = new_mean

            # Update p_sigma in z-space (sep version: no eigenmatrix
            # multiply, just per-axis scaling already applied to z).
            p_sigma = [
                (1.0 - c_sigma) * p_sigma[j]
                + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_w[j]
                for j in range(n)
            ]
            norm_p_sigma = math.sqrt(math.fsum(v * v for v in p_sigma))

            # Heaviside h_sigma: damp covariance update if p_sigma is
            # too long (early-iteration safety per Hansen 2016).
            denom = math.sqrt(max(0.0, 1.0 - math.pow(1.0 - c_sigma, 2.0 * (gen_idx + 1))))
            if denom > 0.0:
                h_threshold = (1.4 + 2.0 / (n + 1.0)) * chi_n
                h_sigma = 1.0 if (norm_p_sigma / denom) < h_threshold else 0.0
            else:
                h_sigma = 0.0

            # Update p_c: tracks the weighted offspring step in C-space
            # (sep variant: scale by sqrt(diag_C) instead of full
            # eigenmatrix; the math is the same on the diagonal).
            p_c = [
                (1.0 - c_c) * p_c[j]
                + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * math.sqrt(diag_C[j]) * z_w[j]
                for j in range(n)
            ]

            # Rank-mu term contribution to diag_C (scaled offspring
            # squared, sep variant).
            rank_mu_diag: list[float] = []
            for j in range(n):
                acc = math.fsum(
                    weights[i]
                    * math.pow(
                        math.sqrt(diag_C[j]) * zs[selected_indices[i]][j],
                        2.0,
                    )
                    for i in range(mu)
                )
                rank_mu_diag.append(acc)

            # Heaviside damping of rank-1 contribution.
            delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c)
            new_diag_C: list[float] = []
            for j in range(n):
                term = (
                    (1.0 - c_1 - c_mu + delta_h * c_1) * diag_C[j]
                    + c_1 * p_c[j] * p_c[j]
                    + c_mu * rank_mu_diag[j]
                )
                # Numerical guard: covariance entries must stay
                # strictly positive; clip floor at a small epsilon.
                new_diag_C.append(max(term, 1e-300))
            diag_C = new_diag_C

            # Update sigma.
            sigma_log_step = (c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1.0)
            # Clamp the log step so a single bad generation can't
            # blow sigma to inf — same defensive cap as evotorch's
            # CMAES default (max log step = ~5).
            sigma_log_step = max(-5.0, min(5.0, sigma_log_step))
            sigma = sigma * math.exp(sigma_log_step)
            if not math.isfinite(sigma) or sigma <= 0.0:
                sigma = config.sigma_init

        # ---- Build proposal -------------------------------------------
        if running_best is None:  # pragma: no cover - guarded by max_gen >= 1
            raise CMAESEvaluationError("CMA-ES produced no individuals; check max_generations")

        digest = _compute_policy_digest(
            config=config,
            best=running_best,
            generation_count=len(generation_reports),
            seed=seed,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
        )
        rationale = _build_rationale(
            best=running_best,
            generation_count=len(generation_reports),
            config=config,
        )
        proposal_meta: dict[str, str] = {
            "policy_digest": digest,
            "seed": str(seed),
            "proposal_id": proposal_id,
            "population_size": str(config.population_size),
            "max_generations": str(config.max_generations),
            "generation_count": str(len(generation_reports)),
            "best_chromosome_digest": chromosome_digest(running_best.chromosome),
            "best_fitness": repr(running_best.fitness_scalar),
            "best_pnl_mean": repr(running_best.fitness_report.pnl_mean),
            "best_max_drawdown": repr(running_best.fitness_report.max_drawdown),
        }
        touchpoints = tuple(s.name for s in specs)
        proposal = PatchProposal(
            ts_ns=ts_ns,
            patch_id=proposal_id,
            source=PROPOSAL_SOURCE,
            target_strategy=config.target_strategy_id,
            touchpoints=touchpoints,
            rationale=rationale,
            meta=proposal_meta,
        )
        result = CMAESResult(
            proposal=proposal,
            generations=tuple(generation_reports),
            best_individual=running_best,
            policy_digest=digest,
        )
        cb.on_evolution_end(ts_ns=ts_ns, result=result)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "CMAESCallback",
    "CMAESConfig",
    "CMAESConfigError",
    "CMAESEvaluationError",
    "CMAESOptimizer",
    "CMAESResult",
    "DEFAULT_DRAWDOWN_PENALTY",
    "DEFAULT_SIGMA_INIT",
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
    "NEW_PIP_DEPENDENCIES",
    "PROPOSAL_SOURCE",
    "null_cmaes_callback",
]
