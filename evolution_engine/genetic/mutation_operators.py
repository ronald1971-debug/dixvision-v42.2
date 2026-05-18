# ADAPTED FROM: facebookresearch/nevergrad
# (nevergrad/optimization/optimizerlib.py — DE / NGOpt / TBPSA / CMA
#  ask()/tell() interface; nevergrad/parametrization/_layering.py —
#  bounded Gaussian + polynomial mutation; nevergrad/optimization/
#  differentialevolution.py — DE/rand/1, DE/current-to-best/1, binomial
#  crossover.)
"""A-04.1 — mutation operators for the genetic strategy-evolution loop.

nevergrad's :class:`Optimizer` walks an ``ask()`` / ``tell()`` loop over
a :class:`Parametrization` — every step asks the optimizer for a
candidate vector, evaluates it, and tells the optimizer the loss. The
canonical operators are the differential-evolution mutants
``DE/rand/1`` and ``DE/current-to-best/1`` plus binomial crossover, the
NSGA-II polynomial-bounded mutation, and a plain Gaussian perturbation.
Trading-strategy evolution does **not** call any of these directly: a
candidate strategy is a *structural mutation* of the running config,
and that path goes through :class:`evolution_engine.patch_pipeline`.
This module is the offline pure-function layer that the orchestrator
in :mod:`evolution_engine.genetic.pipeline` (A-04.2) composes into
``ask()`` / ``tell()`` cycles.

What this module is
-------------------

* Pure-stdlib operators over :class:`StrategyChromosome` (A-02.1). No
  numpy / no nevergrad / no torch import. ``NEW_PIP_DEPENDENCIES = ()``.
* OFFLINE_ONLY tier. The operators read no environment variables,
  perform no IO, never import ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. They produce one new
  :class:`StrategyChromosome` per call and stop.
* INV-15 byte-identical replays. Every operator is a *pure function*
  of its arguments and the caller-supplied ``(seed, generation,
  individual)`` triple. Box-Muller draws are seeded by stateless
  splitmix64 (mirrors :mod:`evolution_engine.gym_env` and
  :mod:`evolution_engine.genetic.cmaes_optimizer`).
* No clock reads. The caller supplies ``ts_ns`` if it wants to stamp
  the chromosome's ``meta`` map; the operators never call
  :func:`time.time_ns` themselves.

What survives from upstream
---------------------------

* The five canonical DE/NSGA-II/Gaussian mutation kernels from
  nevergrad and pymoo, faithful to the standard literature recipes:

  * :func:`gaussian_mutate` — per-dimension N(0, sigma_i) perturbation
    in *encoded* space (mirrors nevergrad's
    ``parametrization.Array.set_mutation(sigma=...)``).
  * :func:`polynomial_mutate` — NSGA-II bounded polynomial mutation
    (Deb & Goyal 1996), distribution index ``eta``. Used by nevergrad
    via ``parametrization.Scalar.with_distribution_indexed_mutation``.
  * :func:`de_rand_1` — DE/rand/1: ``v = a + F * (b - c)``.
  * :func:`de_current_to_best_1` — DE/current-to-best/1:
    ``v = target + F * (best - target) + F * (a - b)``.
  * :func:`de_binomial_crossover` — DE binomial crossover with
    crossover rate ``CR`` and a guaranteed-mutant index ``j*`` so the
    output never coincides with the target.

What we replaced
----------------

* numpy / torch tensor algebra → pure Python ``list[float]`` /
  ``math``-based loops. Per-call cost is ``O(n)`` in chromosome
  dimensionality.
* numpy.random.RandomState / torch.Generator → stateless splitmix64
  hash + Box-Muller. No global PRNG state, no clock seeding.
* nevergrad ``Optimizer.ask()`` / ``tell()`` state machine →
  pure-function operators. Statefulness lives one tier up in the
  pipeline orchestrator (A-04.2).
* nevergrad ``Parametrization.set_bounds(...)`` → kind-aware re-clip
  in *decoded* space via
  :func:`evolution_engine.genetic.strategy_chromosome.unpack` (which
  also re-rounds ``INTEGER`` kinds half-to-even). Every mutant is
  feasible by construction.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE tier — no IO, no clock, no global state, no PRNG (the
  operators' PRNG is seeded by caller-supplied seed and never reads
  the wall clock). AST tests pin the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``intelligence_engine.`` / ``registry.`` /
  ``ui.`` references.
* INV-13 / INV-14 — operators are pure functions and never mutate any
  external registry or governance ledger; the orchestrator (A-04.2)
  is responsible for routing accepted offspring through governance
  via :class:`PatchProposal`.
* INV-71 / B27 / B28 — this module never constructs a
  :class:`PatchProposal`; only ``evolution_engine.patch_pipeline`` /
  ``evolution_engine.genetic.pipeline`` (A-04.2) does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from evolution_engine.genetic.strategy_chromosome import (
    ChromosomeError,
    ParameterKind,
    ParameterSpec,
    StrategyChromosome,
    chromosome_digest,
    unpack,
)

# --- Constants -------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No new pip dependencies — the operators are pure-stdlib."""

OPERATOR_GAUSSIAN: str = "gaussian"
"""Operator tag stamped into the offspring's ``meta['operator']``."""

OPERATOR_POLYNOMIAL: str = "polynomial"
"""Operator tag for :func:`polynomial_mutate`."""

OPERATOR_DE_RAND_1: str = "de_rand_1"
"""Operator tag for :func:`de_rand_1`."""

OPERATOR_DE_CURRENT_TO_BEST_1: str = "de_current_to_best_1"
"""Operator tag for :func:`de_current_to_best_1`."""

OPERATOR_DE_BINOMIAL_CROSSOVER: str = "de_binomial_crossover"
"""Operator tag for :func:`de_binomial_crossover`."""

MAX_META_KEYS: int = 16
"""Hard cap on the number of caller-supplied ``extra_meta`` keys."""

MAX_META_KEY_LEN: int = 64
"""Hard cap on caller-supplied ``extra_meta`` key length."""

MAX_META_VALUE_LEN: int = 256
"""Hard cap on caller-supplied ``extra_meta`` value length."""

_RESERVED_META_KEYS: frozenset[str] = frozenset(
    {
        "operator",
        "parent_digest",
        "parent_a_digest",
        "parent_b_digest",
        "parent_c_digest",
        "best_digest",
        "target_digest",
        "donor_digest",
        "seed",
        "generation",
        "individual",
    }
)
"""Meta keys that operators populate themselves; callers cannot
override them via ``extra_meta``."""

_LOG_BASE: float = 10.0


# --- Errors ----------------------------------------------------------------


class MutationOperatorError(ValueError):
    """Raised when a mutation-operator invariant is violated."""


# ---------------------------------------------------------------------------
# Stateless deterministic PRNG (mirrors gym_env / cmaes_optimizer)
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    """Stateless 64-bit hash. Used to derive every per-sample seed and
    every Box-Muller uniform. Pure stdlib; identical implementation to
    :func:`evolution_engine.gym_env._splitmix64` and
    :func:`evolution_engine.genetic.cmaes_optimizer._splitmix64`."""

    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _uniform01(*key: int) -> float:
    """Deterministic uniform in ``(0, 1]`` keyed off an integer tuple.

    Folds the key through splitmix64 and rescales the bottom 53 bits
    to a double. Strictly positive (offset by ``1 / 2**53``) so callers
    can pass it through ``math.log`` without zero guards.
    """

    h = 0
    for k in key:
        h = _splitmix64(h ^ (k & 0xFFFFFFFFFFFFFFFF))
    return ((h >> 11) + 1) / (1 << 53)


def _gauss_pair(*, seed: int, generation: int, individual: int, dim: int) -> tuple[float, float]:
    """One Box-Muller pair seeded by ``(seed, generation, individual,
    dim)``. Returns two independent N(0, 1) samples."""

    u1 = _uniform01(seed, generation, individual, dim, 0)
    u2 = _uniform01(seed, generation, individual, dim, 1)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return radius * math.cos(angle), radius * math.sin(angle)


def _gauss_value(*, seed: int, generation: int, individual: int, dim: int) -> float:
    """One N(0, 1) draw seeded by the same key tuple as
    :func:`_gauss_pair`. Discards the second of the Box-Muller pair to
    keep per-dimension key independence."""

    a, _ = _gauss_pair(seed=seed, generation=generation, individual=individual, dim=dim)
    return a


# ---------------------------------------------------------------------------
# Encoded-space helpers (mirror cmaes_optimizer._encoded_bounds)
# ---------------------------------------------------------------------------


def _encoded_bounds(specs: tuple[ParameterSpec, ...]) -> tuple[tuple[float, float], ...]:
    """Per-spec ``(low_enc, high_enc)`` bounds in encoded (mutation)
    space. ``CONTINUOUS`` and ``INTEGER`` pass through; ``LOG_CONTINUOUS``
    becomes ``log10(low) .. log10(high)``."""

    out: list[tuple[float, float]] = []
    for spec in specs:
        if spec.kind is ParameterKind.LOG_CONTINUOUS:
            low_enc = math.log10(spec.low)
            high_enc = math.log10(spec.high)
        else:
            low_enc = float(spec.low)
            high_enc = float(spec.high)
        out.append((low_enc, high_enc))
    return tuple(out)


def _encode(spec: ParameterSpec, value: float) -> float:
    """Decoded → encoded for one dimension."""

    if spec.kind is ParameterKind.LOG_CONTINUOUS:
        # spec.low > 0 enforced upstream; value >= low > 0 by construction
        return math.log10(value) / math.log10(_LOG_BASE)
    return float(value)


def _encode_vector(
    specs: tuple[ParameterSpec, ...], values: tuple[float, ...]
) -> tuple[float, ...]:
    """Decoded → encoded for the full chromosome vector."""

    return tuple(_encode(s, v) for s, v in zip(specs, values, strict=True))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_chromosome(chromosome: StrategyChromosome, name: str) -> None:
    if not isinstance(chromosome, StrategyChromosome):
        raise MutationOperatorError(f"{name}: must be StrategyChromosome")


def _validate_pair_compatible(
    a: StrategyChromosome, b: StrategyChromosome, *, name_a: str, name_b: str
) -> None:
    """Two chromosomes share the same ``strategy_id`` + ``specs`` (by
    identity-of-content). Without this check, mixing parents from
    different search spaces is silently accepted."""

    if a.strategy_id != b.strategy_id:
        raise MutationOperatorError(
            f"{name_a} / {name_b}: strategy_id mismatch ({a.strategy_id!r} vs {b.strategy_id!r})"
        )
    if a.specs != b.specs:
        raise MutationOperatorError(
            f"{name_a} / {name_b}: specs mismatch (different search spaces)"
        )


def _validate_int(value: object, name: str, *, allow_negative: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MutationOperatorError(f"{name} must be int")
    if not allow_negative and value < 0:
        raise MutationOperatorError(f"{name} must be >= 0")
    return value


def _validate_unit_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MutationOperatorError(f"{name} must be int|float")
    v = float(value)
    if not math.isfinite(v):
        raise MutationOperatorError(f"{name} must be finite")
    if v < 0.0 or v > 1.0:
        raise MutationOperatorError(f"{name} must be in [0, 1]")
    return v


def _validate_positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MutationOperatorError(f"{name} must be int|float")
    v = float(value)
    if not math.isfinite(v):
        raise MutationOperatorError(f"{name} must be finite")
    if v <= 0.0:
        raise MutationOperatorError(f"{name} must be > 0")
    return v


def _validate_extra_meta(extra_meta: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Validate caller-supplied ``extra_meta`` and return a deterministic
    tuple of (key, value) pairs sorted by key (INV-15)."""

    if extra_meta is None:
        return ()
    if not isinstance(extra_meta, Mapping):
        raise MutationOperatorError("extra_meta must be Mapping or None")
    if len(extra_meta) > MAX_META_KEYS:
        raise MutationOperatorError(f"extra_meta has {len(extra_meta)} keys > {MAX_META_KEYS}")
    pairs: list[tuple[str, str]] = []
    for k, v in extra_meta.items():
        if not isinstance(k, str):
            raise MutationOperatorError("extra_meta keys must be str")
        if not k:
            raise MutationOperatorError("extra_meta keys must be non-empty")
        if len(k) > MAX_META_KEY_LEN:
            raise MutationOperatorError(f"extra_meta key length {len(k)} > {MAX_META_KEY_LEN}")
        if k in _RESERVED_META_KEYS:
            raise MutationOperatorError(f"extra_meta key {k!r} is reserved (operators set it)")
        if not isinstance(v, str):
            raise MutationOperatorError(f"extra_meta value for {k!r} must be str")
        if len(v) > MAX_META_VALUE_LEN:
            raise MutationOperatorError(
                f"extra_meta value length for {k!r}: {len(v)} > {MAX_META_VALUE_LEN}"
            )
        pairs.append((k, v))
    pairs.sort(key=lambda kv: kv[0])
    return tuple(pairs)


def _build_meta(
    *,
    operator: str,
    seed: int,
    generation: int,
    individual: int,
    parent_digests: Mapping[str, str],
    extra_meta_pairs: tuple[tuple[str, str], ...],
) -> Mapping[str, str]:
    """Compose the offspring chromosome's ``meta`` mapping, enforcing
    deterministic key ordering (INV-15)."""

    out: dict[str, str] = {
        "operator": operator,
        "seed": str(seed),
        "generation": str(generation),
        "individual": str(individual),
    }
    for key in sorted(parent_digests):
        out[key] = parent_digests[key]
    for k, v in extra_meta_pairs:
        out[k] = v
    return MappingProxyType({k: out[k] for k in sorted(out)})


# ---------------------------------------------------------------------------
# Operator: Gaussian mutation
# ---------------------------------------------------------------------------


def gaussian_mutate(
    *,
    chromosome: StrategyChromosome,
    sigma: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> StrategyChromosome:
    """Per-dimension Gaussian perturbation in encoded space.

    For each dimension ``i`` with encoded bounds ``(low_i, high_i)`` and
    span ``s_i = high_i - low_i``, the mutant is::

        v_enc[i] = clip(parent_enc[i] + sigma * s_i * z_i, low_i, high_i)

    where ``z_i`` is one Box-Muller N(0, 1) draw seeded by
    ``(seed, generation, individual, i)``. The mutant is then decoded
    via :func:`unpack`, which re-clips and re-rounds ``INTEGER`` kinds.

    The ``sigma`` argument is interpreted as a *fraction of the encoded
    span*: ``sigma=0.1`` means each dimension is perturbed by N(0,
    (0.1 * span)^2). This matches nevergrad's
    ``Array.set_mutation(sigma=...)``-as-fraction convention used by
    NGOpt for bounded continuous arrays.
    """

    _validate_chromosome(chromosome, "gaussian_mutate.chromosome")
    s = _validate_positive_float(sigma, "gaussian_mutate.sigma")
    seed_i = _validate_int(seed, "gaussian_mutate.seed")
    gen_i = _validate_int(generation, "gaussian_mutate.generation")
    ind_i = _validate_int(individual, "gaussian_mutate.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    bounds = _encoded_bounds(chromosome.specs)
    encoded = _encode_vector(chromosome.specs, chromosome.values)
    mutant_enc: list[float] = []
    for dim, (parent_v, (lo, hi)) in enumerate(zip(encoded, bounds, strict=True)):
        span = hi - lo
        z = _gauss_value(seed=seed_i, generation=gen_i, individual=ind_i, dim=dim)
        v = parent_v + s * span * z
        mutant_enc.append(min(hi, max(lo, v)))

    decoded = unpack(chromosome.specs, tuple(mutant_enc))
    new_values = tuple(decoded[spec.name] for spec in chromosome.specs)
    meta = _build_meta(
        operator=OPERATOR_GAUSSIAN,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_digests={"parent_digest": chromosome_digest(chromosome)},
        extra_meta_pairs=extra_pairs,
    )
    try:
        return StrategyChromosome(
            strategy_id=chromosome.strategy_id,
            specs=chromosome.specs,
            values=new_values,
            version=chromosome.version + 1,
            meta=meta,
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise MutationOperatorError(f"gaussian_mutate: {exc}") from exc


# ---------------------------------------------------------------------------
# Operator: NSGA-II polynomial-bounded mutation
# ---------------------------------------------------------------------------


def polynomial_mutate(
    *,
    chromosome: StrategyChromosome,
    eta: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> StrategyChromosome:
    """NSGA-II polynomial-bounded mutation (Deb & Goyal 1996).

    Per dimension with encoded bounds ``(low, high)`` and parent value
    ``y`` the mutation draws ``u ~ U(0, 1)`` and applies::

        delta1 = (y - low) / (high - low)
        delta2 = (high - y) / (high - low)
        if u < 0.5:
            xy     = 1 - delta1
            val    = 2*u + (1 - 2*u) * xy ** (eta + 1)
            deltaq = val ** (1 / (eta + 1)) - 1
        else:
            xy     = 1 - delta2
            val    = 2*(1 - u) + 2*(u - 0.5) * xy ** (eta + 1)
            deltaq = 1 - val ** (1 / (eta + 1))
        y_new = clip(y + deltaq * (high - low), low, high)

    The distribution index ``eta`` must be > 0; larger ``eta`` produces
    smaller perturbations (concentrated near ``y``). Standard NSGA-II
    default is ``eta = 20``.
    """

    _validate_chromosome(chromosome, "polynomial_mutate.chromosome")
    e = _validate_positive_float(eta, "polynomial_mutate.eta")
    seed_i = _validate_int(seed, "polynomial_mutate.seed")
    gen_i = _validate_int(generation, "polynomial_mutate.generation")
    ind_i = _validate_int(individual, "polynomial_mutate.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    bounds = _encoded_bounds(chromosome.specs)
    encoded = _encode_vector(chromosome.specs, chromosome.values)
    inv_exp = 1.0 / (e + 1.0)

    mutant_enc: list[float] = []
    for dim, (y, (lo, hi)) in enumerate(zip(encoded, bounds, strict=True)):
        span = hi - lo
        if span <= 0.0:  # pragma: no cover - guarded by ParameterSpec
            mutant_enc.append(y)
            continue
        u = _uniform01(seed_i, gen_i, ind_i, dim, 2)
        delta1 = (y - lo) / span
        delta2 = (hi - y) / span
        if u < 0.5:
            xy = max(0.0, 1.0 - delta1)
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (e + 1.0))
            deltaq = math.pow(val, inv_exp) - 1.0
        else:
            xy = max(0.0, 1.0 - delta2)
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (e + 1.0))
            deltaq = 1.0 - math.pow(val, inv_exp)
        v = y + deltaq * span
        mutant_enc.append(min(hi, max(lo, v)))

    decoded = unpack(chromosome.specs, tuple(mutant_enc))
    new_values = tuple(decoded[spec.name] for spec in chromosome.specs)
    meta = _build_meta(
        operator=OPERATOR_POLYNOMIAL,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_digests={"parent_digest": chromosome_digest(chromosome)},
        extra_meta_pairs=extra_pairs,
    )
    try:
        return StrategyChromosome(
            strategy_id=chromosome.strategy_id,
            specs=chromosome.specs,
            values=new_values,
            version=chromosome.version + 1,
            meta=meta,
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise MutationOperatorError(f"polynomial_mutate: {exc}") from exc


# ---------------------------------------------------------------------------
# Operator: Differential Evolution — DE/rand/1
# ---------------------------------------------------------------------------


def de_rand_1(
    *,
    a: StrategyChromosome,
    b: StrategyChromosome,
    c: StrategyChromosome,
    F: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> StrategyChromosome:
    """DE/rand/1: ``v = a + F * (b - c)`` in encoded space.

    The three parents must share ``strategy_id`` + ``specs``. The
    differential weight ``F`` is typically in ``[0.4, 1.0]``; it is
    only required to be strictly positive (nevergrad's NGOpt sometimes
    walks larger values during the early phase).

    The mutant is re-clipped to the encoded bounds and decoded via
    :func:`unpack`, which re-rounds ``INTEGER`` kinds.
    """

    _validate_chromosome(a, "de_rand_1.a")
    _validate_chromosome(b, "de_rand_1.b")
    _validate_chromosome(c, "de_rand_1.c")
    _validate_pair_compatible(a, b, name_a="de_rand_1.a", name_b="de_rand_1.b")
    _validate_pair_compatible(a, c, name_a="de_rand_1.a", name_b="de_rand_1.c")
    f = _validate_positive_float(F, "de_rand_1.F")
    seed_i = _validate_int(seed, "de_rand_1.seed")
    gen_i = _validate_int(generation, "de_rand_1.generation")
    ind_i = _validate_int(individual, "de_rand_1.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    bounds = _encoded_bounds(a.specs)
    enc_a = _encode_vector(a.specs, a.values)
    enc_b = _encode_vector(b.specs, b.values)
    enc_c = _encode_vector(c.specs, c.values)

    mutant_enc: list[float] = []
    for (va, vb, vc), (lo, hi) in zip(zip(enc_a, enc_b, enc_c, strict=True), bounds, strict=True):
        v = va + f * (vb - vc)
        mutant_enc.append(min(hi, max(lo, v)))

    decoded = unpack(a.specs, tuple(mutant_enc))
    new_values = tuple(decoded[spec.name] for spec in a.specs)
    meta = _build_meta(
        operator=OPERATOR_DE_RAND_1,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_digests={
            "parent_a_digest": chromosome_digest(a),
            "parent_b_digest": chromosome_digest(b),
            "parent_c_digest": chromosome_digest(c),
        },
        extra_meta_pairs=extra_pairs,
    )
    base_version = max(a.version, b.version, c.version)
    try:
        return StrategyChromosome(
            strategy_id=a.strategy_id,
            specs=a.specs,
            values=new_values,
            version=base_version + 1,
            meta=meta,
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise MutationOperatorError(f"de_rand_1: {exc}") from exc


# ---------------------------------------------------------------------------
# Operator: Differential Evolution — DE/current-to-best/1
# ---------------------------------------------------------------------------


def de_current_to_best_1(
    *,
    target: StrategyChromosome,
    best: StrategyChromosome,
    a: StrategyChromosome,
    b: StrategyChromosome,
    F: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> StrategyChromosome:
    """DE/current-to-best/1:
    ``v = target + F * (best - target) + F * (a - b)`` in encoded space.

    All four parents must share ``strategy_id`` + ``specs``. The
    differential weight ``F`` is typically in ``[0.4, 1.0]``. The
    mutant is re-clipped to the encoded bounds and decoded via
    :func:`unpack`.
    """

    _validate_chromosome(target, "de_current_to_best_1.target")
    _validate_chromosome(best, "de_current_to_best_1.best")
    _validate_chromosome(a, "de_current_to_best_1.a")
    _validate_chromosome(b, "de_current_to_best_1.b")
    _validate_pair_compatible(
        target, best, name_a="de_current_to_best_1.target", name_b="de_current_to_best_1.best"
    )
    _validate_pair_compatible(
        target, a, name_a="de_current_to_best_1.target", name_b="de_current_to_best_1.a"
    )
    _validate_pair_compatible(
        target, b, name_a="de_current_to_best_1.target", name_b="de_current_to_best_1.b"
    )
    f = _validate_positive_float(F, "de_current_to_best_1.F")
    seed_i = _validate_int(seed, "de_current_to_best_1.seed")
    gen_i = _validate_int(generation, "de_current_to_best_1.generation")
    ind_i = _validate_int(individual, "de_current_to_best_1.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    bounds = _encoded_bounds(target.specs)
    enc_t = _encode_vector(target.specs, target.values)
    enc_best = _encode_vector(best.specs, best.values)
    enc_a = _encode_vector(a.specs, a.values)
    enc_b = _encode_vector(b.specs, b.values)

    mutant_enc: list[float] = []
    for (vt, vbest, va, vb), (lo, hi) in zip(
        zip(enc_t, enc_best, enc_a, enc_b, strict=True), bounds, strict=True
    ):
        v = vt + f * (vbest - vt) + f * (va - vb)
        mutant_enc.append(min(hi, max(lo, v)))

    decoded = unpack(target.specs, tuple(mutant_enc))
    new_values = tuple(decoded[spec.name] for spec in target.specs)
    meta = _build_meta(
        operator=OPERATOR_DE_CURRENT_TO_BEST_1,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_digests={
            "target_digest": chromosome_digest(target),
            "best_digest": chromosome_digest(best),
            "parent_a_digest": chromosome_digest(a),
            "parent_b_digest": chromosome_digest(b),
        },
        extra_meta_pairs=extra_pairs,
    )
    base_version = max(target.version, best.version, a.version, b.version)
    try:
        return StrategyChromosome(
            strategy_id=target.strategy_id,
            specs=target.specs,
            values=new_values,
            version=base_version + 1,
            meta=meta,
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise MutationOperatorError(f"de_current_to_best_1: {exc}") from exc


# ---------------------------------------------------------------------------
# Operator: Differential Evolution — binomial crossover
# ---------------------------------------------------------------------------


def de_binomial_crossover(
    *,
    target: StrategyChromosome,
    donor: StrategyChromosome,
    CR: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> StrategyChromosome:
    """DE binomial crossover.

    For each dimension ``i`` independently draw ``u_i ~ U(0, 1)``;
    take ``donor[i]`` if ``u_i < CR`` else ``target[i]``. To guarantee
    the offspring differs from the target in at least one dimension,
    one *forced* index ``j*`` (deterministically derived from the
    seed/gen/individual triple) is always taken from the donor.

    The two parents must share ``strategy_id`` + ``specs``. Crossover
    is performed in *decoded* (post-``unpack``) space because per-dim
    selection is structure-preserving — there is no need to re-clip.
    """

    _validate_chromosome(target, "de_binomial_crossover.target")
    _validate_chromosome(donor, "de_binomial_crossover.donor")
    _validate_pair_compatible(
        target, donor, name_a="de_binomial_crossover.target", name_b="de_binomial_crossover.donor"
    )
    cr = _validate_unit_float(CR, "de_binomial_crossover.CR")
    seed_i = _validate_int(seed, "de_binomial_crossover.seed")
    gen_i = _validate_int(generation, "de_binomial_crossover.generation")
    ind_i = _validate_int(individual, "de_binomial_crossover.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    n = len(target.specs)
    forced_index = _splitmix64(_splitmix64(seed_i ^ ((gen_i << 32) | ind_i)) ^ 0xDC0DE_BCC0) % n

    out_values: list[float] = []
    for dim, (vt, vd) in enumerate(zip(target.values, donor.values, strict=True)):
        if dim == forced_index:
            out_values.append(vd)
            continue
        u = _uniform01(seed_i, gen_i, ind_i, dim, 3)
        out_values.append(vd if u < cr else vt)

    meta = _build_meta(
        operator=OPERATOR_DE_BINOMIAL_CROSSOVER,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_digests={
            "target_digest": chromosome_digest(target),
            "donor_digest": chromosome_digest(donor),
        },
        extra_meta_pairs=extra_pairs,
    )
    base_version = max(target.version, donor.version)
    try:
        return StrategyChromosome(
            strategy_id=target.strategy_id,
            specs=target.specs,
            values=tuple(out_values),
            version=base_version + 1,
            meta=meta,
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise MutationOperatorError(f"de_binomial_crossover: {exc}") from exc


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "MAX_META_KEYS",
    "MAX_META_KEY_LEN",
    "MAX_META_VALUE_LEN",
    "MutationOperatorError",
    "NEW_PIP_DEPENDENCIES",
    "OPERATOR_DE_BINOMIAL_CROSSOVER",
    "OPERATOR_DE_CURRENT_TO_BEST_1",
    "OPERATOR_DE_RAND_1",
    "OPERATOR_GAUSSIAN",
    "OPERATOR_POLYNOMIAL",
    "de_binomial_crossover",
    "de_current_to_best_1",
    "de_rand_1",
    "gaussian_mutate",
    "polynomial_mutate",
]
