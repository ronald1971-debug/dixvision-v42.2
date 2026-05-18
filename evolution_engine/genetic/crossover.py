# ADAPTED FROM: DEAP/deap
# (deap/tools/crossover.py — cxSimulatedBinary / cxSimulatedBinaryBounded /
#  cxBlend / cxTwoPoint reference implementations.)
"""B-02 — DEAP crossover operators (complement to A-03 / A-04.1).

DEAP's :mod:`deap.tools.crossover` ships canonical genetic-algorithm
crossover kernels. The three this module rewrites in pure Python are:

* **Simulated Binary Crossover** (Deb & Agrawal 1995, ``cxSimulatedBinary`` +
  ``cxSimulatedBinaryBounded``). The dominant NSGA-II real-valued
  crossover. We adapt the *bounded* variant: every offspring dimension
  is clipped to the spec's encoded bounds before being decoded.
* **Blend (BLX-alpha) crossover** (Eshelman & Schaffer 1993,
  ``cxBlend``). The canonical real-valued blend that samples each
  offspring dimension uniformly from
  ``[min - alpha*d, max + alpha*d]``.
* **Two-point crossover** (Holland 1975, ``cxTwoPoint``). The classic
  discrete-recombination kernel that swaps the segment between two
  cut-points. Used for ``INTEGER`` parameter chromosomes and for
  chromosomes with mixed-kind specs where blending the integer dims
  would lose round-trip stability.

This module deliberately does **not** re-implement DEAP's polynomial
or Gaussian mutation — those live in :mod:`evolution_engine.genetic.\
mutation_operators` (A-04.1, adapted from nevergrad / pymoo). The
crossover/mutation split mirrors DEAP's own module layout.

What this module is
-------------------

* Pure-stdlib operators over :class:`StrategyChromosome` (A-02.1). No
  numpy / no deap / no torch import. ``NEW_PIP_DEPENDENCIES = ()`` —
  the DEAP formulas are extracted and re-implemented from scratch
  (LGPL-3.0 mitigation: rewrite math, do not import DEAP).
* OFFLINE_ONLY tier. The operators read no environment variables,
  perform no IO, never import ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry`` / ``ui``. They produce
  exactly two child :class:`StrategyChromosome` instances per call
  and stop.
* INV-15 byte-identical replays. Every operator is a *pure function*
  of its arguments and the caller-supplied
  ``(seed, generation, individual)`` triple. Uniform draws are seeded
  by stateless splitmix64 (mirrors :mod:`evolution_engine.gym_env`
  and :mod:`evolution_engine.genetic.cmaes_optimizer`).
* No clock reads. The caller supplies ``ts_ns`` only if it wants to
  stamp something into ``extra_meta`` — the operators never call
  :func:`time.time_ns` themselves.

What survives from upstream
---------------------------

* **SBX kernel** (DEAP ``cxSimulatedBinaryBounded``). The Deb-Agrawal
  contraction-expansion formula::

      if u <= 0.5: beta_q = (2*u)^(1/(eta+1))
      else:        beta_q = (1/(2*(1-u)))^(1/(eta+1))
      c1 = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)
      c2 = 0.5 * ((1 - beta_q) * y1 + (1 + beta_q) * y2)

  with ``eta`` the distribution index (higher eta → children closer
  to parents). We use the *symmetric simplified* form per DEAP's
  ``cxSimulatedBinary``; the bounded variant differs only in the
  per-dimension clip we already apply via :func:`unpack`.
* **Blend (BLX-alpha) kernel** (DEAP ``cxBlend``). For each dimension::

      gamma = (1 + 2*alpha) * u - alpha
      c1 = (1 - gamma) * y1 + gamma * y2
      c2 = gamma * y1 + (1 - gamma) * y2

  This is DEAP's *symmetric* parameterisation (one ``u`` per
  dimension yields two offspring); identical in distribution to the
  Eshelman-Schaffer min/max range sample.
* **Two-point kernel** (DEAP ``cxTwoPoint``). Pick two distinct
  cut-points ``i <= j`` in ``[0, n]``, swap ``[i:j]`` between
  parents. Per-bracket independence in dimension order means the
  result is a deterministic function of the seed.

What we replaced
----------------

* numpy / DEAP ``Individual`` mutation in place → pure Python tuples
  on :class:`StrategyChromosome`. Two new immutable chromosomes per
  call; no parent mutation.
* numpy.random.RandomState / Python ``random.Random`` → stateless
  splitmix64. No global PRNG state, no clock seeding.
* DEAP ``creator.Individual`` typed inheritance → frozen+slotted
  :class:`StrategyChromosome` value object (LGPL mitigation: extract
  math only, do not import DEAP toolbox classes).
* DEAP bounded-clip via Python ``min/max`` lists → kind-aware
  re-clip in *decoded* space via :func:`unpack` (which also
  re-rounds ``INTEGER`` kinds half-to-even). Every offspring is
  feasible by construction.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE tier — no IO, no clock, no global state. The operators'
  PRNG is seeded by the caller and never reads the wall clock. AST
  tests pin the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``intelligence_engine.`` / ``registry.`` /
  ``ui.`` references.
* INV-13 / INV-14 — operators are pure functions and never mutate
  any external registry or governance ledger; the orchestrator
  (A-04.2 ``evolution_engine.pipeline``) is responsible for routing
  accepted offspring through governance via
  :class:`~core.contracts.learning.PatchProposal`.
* INV-71 / B27 / B28 — this module never constructs a
  :class:`~core.contracts.learning.PatchProposal`; only
  ``evolution_engine.patch_pipeline`` /
  ``evolution_engine.pipeline`` / ``evolution_engine.sandbox`` /
  ``evolution_engine.rllib_trainer`` /
  ``evolution_engine.genetic.cmaes_optimizer`` do.
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
"""No new pip dependencies — the operators are pure-stdlib. The DEAP
formulas are re-implemented from scratch (LGPL-3.0 mitigation)."""

OPERATOR_SBX: str = "sbx"
"""Operator tag stamped into offspring's ``meta['operator']`` for
:func:`simulated_binary_crossover`."""

OPERATOR_BLEND: str = "blend"
"""Operator tag for :func:`blend_crossover`."""

OPERATOR_TWO_POINT: str = "two_point"
"""Operator tag for :func:`two_point_crossover`."""

MAX_META_KEYS: int = 16
"""Hard cap on the number of caller-supplied ``extra_meta`` keys."""

MAX_META_KEY_LEN: int = 64
"""Hard cap on caller-supplied ``extra_meta`` key length."""

MAX_META_VALUE_LEN: int = 256
"""Hard cap on caller-supplied ``extra_meta`` value length."""

_RESERVED_META_KEYS: frozenset[str] = frozenset(
    {
        "operator",
        "parent_a_digest",
        "parent_b_digest",
        "seed",
        "generation",
        "individual",
        "child",
        "cut_lo",
        "cut_hi",
    }
)
"""Meta keys that operators populate themselves; callers cannot
override them via ``extra_meta``."""

_LOG_BASE: float = 10.0


# --- Errors ----------------------------------------------------------------


class CrossoverOperatorError(ValueError):
    """Raised when a crossover-operator invariant is violated."""


# ---------------------------------------------------------------------------
# Stateless deterministic PRNG (mirrors gym_env / cmaes_optimizer /
# mutation_operators).
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    """Stateless 64-bit hash. Used to derive every per-sample seed and
    every uniform draw. Pure stdlib; identical implementation to
    :func:`evolution_engine.gym_env._splitmix64` and
    :func:`evolution_engine.genetic.cmaes_optimizer._splitmix64` and
    :func:`evolution_engine.genetic.mutation_operators._splitmix64`."""

    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _uniform01(*key: int) -> float:
    """Deterministic uniform in ``(0, 1]`` keyed off an integer tuple.

    Folds the key through splitmix64 and rescales the bottom 53 bits
    to a double. Strictly positive (offset by ``1 / 2**53``) so
    callers can pass it through :func:`math.log` without zero guards.
    """

    h = 0
    for k in key:
        h = _splitmix64(h ^ (k & 0xFFFFFFFFFFFFFFFF))
    return ((h >> 11) + 1) / (1 << 53)


def _uniform_int(low: int, high: int, *key: int) -> int:
    """Deterministic integer draw in the closed range ``[low, high]``
    keyed off ``key``. Uses splitmix64 + modulo, biased only at
    levels well below ``1 / 2**64`` for spans < ``2**32`` — fine for
    chromosome dim counts we expect (at most ``2**16``)."""

    if high < low:
        raise CrossoverOperatorError(f"_uniform_int: empty range [{low}, {high}]")
    span = high - low + 1
    h = 0
    for k in key:
        h = _splitmix64(h ^ (k & 0xFFFFFFFFFFFFFFFF))
    return low + (h % span)


# ---------------------------------------------------------------------------
# Encoded-space helpers (mirror cmaes_optimizer._encoded_bounds /
# mutation_operators._encoded_bounds).
# ---------------------------------------------------------------------------


def _encoded_bounds(
    specs: tuple[ParameterSpec, ...],
) -> tuple[tuple[float, float], ...]:
    """Per-spec ``(low_enc, high_enc)`` bounds in encoded (recombination)
    space. ``CONTINUOUS`` and ``INTEGER`` pass through;
    ``LOG_CONTINUOUS`` becomes ``log10(low) .. log10(high)``."""

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
        raise CrossoverOperatorError(f"{name}: must be StrategyChromosome")


def _validate_pair_compatible(
    a: StrategyChromosome,
    b: StrategyChromosome,
    *,
    name_a: str,
    name_b: str,
) -> None:
    """Two chromosomes share the same ``strategy_id`` + ``specs`` (by
    identity-of-content). Without this check, mixing parents from
    different search spaces is silently accepted."""

    if a.strategy_id != b.strategy_id:
        raise CrossoverOperatorError(
            f"{name_a} / {name_b}: strategy_id mismatch ({a.strategy_id!r} vs {b.strategy_id!r})"
        )
    if a.specs != b.specs:
        raise CrossoverOperatorError(
            f"{name_a} / {name_b}: specs mismatch (different search spaces)"
        )


def _validate_int(value: object, name: str, *, allow_negative: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CrossoverOperatorError(f"{name} must be int")
    if not allow_negative and value < 0:
        raise CrossoverOperatorError(f"{name} must be >= 0")
    return value


def _validate_unit_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CrossoverOperatorError(f"{name} must be int|float")
    v = float(value)
    if not math.isfinite(v):
        raise CrossoverOperatorError(f"{name} must be finite")
    if v < 0.0 or v > 1.0:
        raise CrossoverOperatorError(f"{name} must be in [0, 1]")
    return v


def _validate_positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CrossoverOperatorError(f"{name} must be int|float")
    v = float(value)
    if not math.isfinite(v):
        raise CrossoverOperatorError(f"{name} must be finite")
    if v <= 0.0:
        raise CrossoverOperatorError(f"{name} must be > 0")
    return v


def _validate_non_negative_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CrossoverOperatorError(f"{name} must be int|float")
    v = float(value)
    if not math.isfinite(v):
        raise CrossoverOperatorError(f"{name} must be finite")
    if v < 0.0:
        raise CrossoverOperatorError(f"{name} must be >= 0")
    return v


def _validate_extra_meta(
    extra_meta: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate caller-supplied ``extra_meta`` and return a deterministic
    tuple of ``(key, value)`` pairs sorted by key (INV-15)."""

    if extra_meta is None:
        return ()
    if not isinstance(extra_meta, Mapping):
        raise CrossoverOperatorError("extra_meta must be Mapping or None")
    if len(extra_meta) > MAX_META_KEYS:
        raise CrossoverOperatorError(f"extra_meta has {len(extra_meta)} keys > {MAX_META_KEYS}")
    pairs: list[tuple[str, str]] = []
    for k, v in extra_meta.items():
        if not isinstance(k, str):
            raise CrossoverOperatorError("extra_meta keys must be str")
        if not k:
            raise CrossoverOperatorError("extra_meta keys must be non-empty")
        if len(k) > MAX_META_KEY_LEN:
            raise CrossoverOperatorError(f"extra_meta key length {len(k)} > {MAX_META_KEY_LEN}")
        if k in _RESERVED_META_KEYS:
            raise CrossoverOperatorError(f"extra_meta key {k!r} is reserved (operators set it)")
        if not isinstance(v, str):
            raise CrossoverOperatorError(f"extra_meta value for {k!r} must be str")
        if len(v) > MAX_META_VALUE_LEN:
            raise CrossoverOperatorError(
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
    parent_a_digest: str,
    parent_b_digest: str,
    child: int,
    extra: Mapping[str, str] | None,
    extra_meta_pairs: tuple[tuple[str, str], ...],
) -> Mapping[str, str]:
    """Compose the offspring chromosome's ``meta`` mapping, enforcing
    deterministic key ordering (INV-15)."""

    out: dict[str, str] = {
        "operator": operator,
        "seed": str(seed),
        "generation": str(generation),
        "individual": str(individual),
        "parent_a_digest": parent_a_digest,
        "parent_b_digest": parent_b_digest,
        "child": str(child),
    }
    if extra:
        out.update(extra)
    for k, v in extra_meta_pairs:
        out[k] = v
    return MappingProxyType({k: out[k] for k in sorted(out)})


def _decode_to_values(
    specs: tuple[ParameterSpec, ...], encoded: tuple[float, ...]
) -> tuple[float, ...]:
    """Decode an encoded-space tuple back to the spec-order
    ``values`` tuple expected by :class:`StrategyChromosome`. Routes
    through :func:`unpack` so the kind-aware re-clip and ``INTEGER``
    rounding (banker's rounding) apply."""

    decoded = unpack(specs, encoded)
    return tuple(decoded[spec.name] for spec in specs)


def _build_child(
    *,
    parent: StrategyChromosome,
    new_values: tuple[float, ...],
    meta: Mapping[str, str],
    operator: str,
) -> StrategyChromosome:
    """Wrap :class:`StrategyChromosome` construction so any
    contract-violation upstream surfaces as a typed
    :class:`CrossoverOperatorError`."""

    try:
        return StrategyChromosome(
            strategy_id=parent.strategy_id,
            specs=parent.specs,
            values=new_values,
            version=parent.version + 1,
            meta=meta,
        )
    except ChromosomeError as exc:  # pragma: no cover - safety net
        raise CrossoverOperatorError(f"{operator}: {exc}") from exc


# ---------------------------------------------------------------------------
# Operator: Simulated Binary Crossover (SBX, Deb & Agrawal 1995)
# ---------------------------------------------------------------------------


def simulated_binary_crossover(
    *,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    eta: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> tuple[StrategyChromosome, StrategyChromosome]:
    """Simulated Binary Crossover (SBX) in encoded space.

    Per dimension ``i`` with encoded parents ``(y1, y2)`` and a uniform
    draw ``u ~ U(0, 1]`` keyed by ``(seed, generation, individual,
    i)``, compute the contraction-expansion factor::

        if u <= 0.5: beta_q = (2*u)^(1/(eta+1))
        else:        beta_q = (1/(2*(1-u)))^(1/(eta+1))

    and emit two offspring::

        c1 = 0.5 * ((1 + beta_q) * y1 + (1 - beta_q) * y2)
        c2 = 0.5 * ((1 - beta_q) * y1 + (1 + beta_q) * y2)

    Higher ``eta`` → children clustered near parents; ``eta=0`` is
    pure-arithmetic recombination. The DEAP default is ``eta=20``.

    Both children are then per-dimension clipped to the spec's
    encoded bounds and decoded via :func:`unpack` (which re-rounds
    ``INTEGER`` kinds half-to-even). Result is always feasible.
    """

    _validate_chromosome(parent_a, "simulated_binary_crossover.parent_a")
    _validate_chromosome(parent_b, "simulated_binary_crossover.parent_b")
    _validate_pair_compatible(parent_a, parent_b, name_a="parent_a", name_b="parent_b")
    eta_f = _validate_non_negative_float(eta, "simulated_binary_crossover.eta")
    seed_i = _validate_int(seed, "simulated_binary_crossover.seed")
    gen_i = _validate_int(generation, "simulated_binary_crossover.generation")
    ind_i = _validate_int(individual, "simulated_binary_crossover.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    bounds = _encoded_bounds(parent_a.specs)
    enc_a = _encode_vector(parent_a.specs, parent_a.values)
    enc_b = _encode_vector(parent_b.specs, parent_b.values)
    exponent = 1.0 / (eta_f + 1.0)

    c1_enc: list[float] = []
    c2_enc: list[float] = []
    for dim, (y1, y2, (lo, hi)) in enumerate(zip(enc_a, enc_b, bounds, strict=True)):
        u = _uniform01(seed_i, gen_i, ind_i, dim)
        if u <= 0.5:
            beta_q = (2.0 * u) ** exponent
        else:
            beta_q = (1.0 / (2.0 * (1.0 - u))) ** exponent
        c1 = 0.5 * ((1.0 + beta_q) * y1 + (1.0 - beta_q) * y2)
        c2 = 0.5 * ((1.0 - beta_q) * y1 + (1.0 + beta_q) * y2)
        c1_enc.append(min(hi, max(lo, c1)))
        c2_enc.append(min(hi, max(lo, c2)))

    parent_a_digest = chromosome_digest(parent_a)
    parent_b_digest = chromosome_digest(parent_b)
    meta_a = _build_meta(
        operator=OPERATOR_SBX,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_a_digest=parent_a_digest,
        parent_b_digest=parent_b_digest,
        child=0,
        extra={"eta": _format_float(eta_f)},
        extra_meta_pairs=extra_pairs,
    )
    meta_b = _build_meta(
        operator=OPERATOR_SBX,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_a_digest=parent_a_digest,
        parent_b_digest=parent_b_digest,
        child=1,
        extra={"eta": _format_float(eta_f)},
        extra_meta_pairs=extra_pairs,
    )
    child_a = _build_child(
        parent=parent_a,
        new_values=_decode_to_values(parent_a.specs, tuple(c1_enc)),
        meta=meta_a,
        operator="simulated_binary_crossover",
    )
    child_b = _build_child(
        parent=parent_b,
        new_values=_decode_to_values(parent_b.specs, tuple(c2_enc)),
        meta=meta_b,
        operator="simulated_binary_crossover",
    )
    return child_a, child_b


# ---------------------------------------------------------------------------
# Operator: Blend (BLX-alpha) crossover (Eshelman & Schaffer 1993)
# ---------------------------------------------------------------------------


def blend_crossover(
    *,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    alpha: float,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> tuple[StrategyChromosome, StrategyChromosome]:
    """BLX-alpha blend crossover in encoded space.

    Per dimension with encoded parents ``(y1, y2)`` and a uniform
    draw ``u ~ U(0, 1]`` keyed by ``(seed, generation, individual,
    i)``, set::

        gamma = (1 + 2*alpha) * u - alpha
        c1 = (1 - gamma) * y1 + gamma * y2
        c2 = gamma * y1 + (1 - gamma) * y2

    ``alpha=0`` is uniform arithmetic crossover; ``alpha=0.5`` samples
    from the extended interval ``[min - 0.5*d, max + 0.5*d]`` where
    ``d = |y1 - y2|`` (canonical Eshelman-Schaffer setting).

    Both children are then per-dimension clipped to the spec's
    encoded bounds and decoded via :func:`unpack`.
    """

    _validate_chromosome(parent_a, "blend_crossover.parent_a")
    _validate_chromosome(parent_b, "blend_crossover.parent_b")
    _validate_pair_compatible(parent_a, parent_b, name_a="parent_a", name_b="parent_b")
    alpha_f = _validate_non_negative_float(alpha, "blend_crossover.alpha")
    seed_i = _validate_int(seed, "blend_crossover.seed")
    gen_i = _validate_int(generation, "blend_crossover.generation")
    ind_i = _validate_int(individual, "blend_crossover.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    bounds = _encoded_bounds(parent_a.specs)
    enc_a = _encode_vector(parent_a.specs, parent_a.values)
    enc_b = _encode_vector(parent_b.specs, parent_b.values)

    c1_enc: list[float] = []
    c2_enc: list[float] = []
    for dim, (y1, y2, (lo, hi)) in enumerate(zip(enc_a, enc_b, bounds, strict=True)):
        u = _uniform01(seed_i, gen_i, ind_i, dim)
        gamma = (1.0 + 2.0 * alpha_f) * u - alpha_f
        c1 = (1.0 - gamma) * y1 + gamma * y2
        c2 = gamma * y1 + (1.0 - gamma) * y2
        c1_enc.append(min(hi, max(lo, c1)))
        c2_enc.append(min(hi, max(lo, c2)))

    parent_a_digest = chromosome_digest(parent_a)
    parent_b_digest = chromosome_digest(parent_b)
    meta_a = _build_meta(
        operator=OPERATOR_BLEND,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_a_digest=parent_a_digest,
        parent_b_digest=parent_b_digest,
        child=0,
        extra={"alpha": _format_float(alpha_f)},
        extra_meta_pairs=extra_pairs,
    )
    meta_b = _build_meta(
        operator=OPERATOR_BLEND,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_a_digest=parent_a_digest,
        parent_b_digest=parent_b_digest,
        child=1,
        extra={"alpha": _format_float(alpha_f)},
        extra_meta_pairs=extra_pairs,
    )
    child_a = _build_child(
        parent=parent_a,
        new_values=_decode_to_values(parent_a.specs, tuple(c1_enc)),
        meta=meta_a,
        operator="blend_crossover",
    )
    child_b = _build_child(
        parent=parent_b,
        new_values=_decode_to_values(parent_b.specs, tuple(c2_enc)),
        meta=meta_b,
        operator="blend_crossover",
    )
    return child_a, child_b


# ---------------------------------------------------------------------------
# Operator: Two-point crossover (Holland 1975, DEAP cxTwoPoint)
# ---------------------------------------------------------------------------


def two_point_crossover(
    *,
    parent_a: StrategyChromosome,
    parent_b: StrategyChromosome,
    seed: int,
    generation: int,
    individual: int,
    extra_meta: Mapping[str, str] | None = None,
) -> tuple[StrategyChromosome, StrategyChromosome]:
    """Two-point crossover over the decoded value tuple.

    Pick two distinct cut-points ``i <= j`` in ``[0, n]`` deterministic
    in ``(seed, generation, individual)``. Swap the segment
    ``values[i:j]`` between the two parents::

        c1 = a[:i] + b[i:j] + a[j:]
        c2 = b[:i] + a[i:j] + b[j:]

    Operates on **decoded** values (not encoded) — the segments are
    transplanted verbatim, so ``INTEGER`` / ``LOG_CONTINUOUS`` values
    are preserved bit-for-bit on the receiving side without going
    through any encode/decode round-trip.

    For chromosomes of length ``n=1`` both cut-points collapse to
    ``i=j=0`` or ``i=j=1`` and the operator is a no-op (children
    equal to parents up to ``meta``). Pinned by tests.
    """

    _validate_chromosome(parent_a, "two_point_crossover.parent_a")
    _validate_chromosome(parent_b, "two_point_crossover.parent_b")
    _validate_pair_compatible(parent_a, parent_b, name_a="parent_a", name_b="parent_b")
    seed_i = _validate_int(seed, "two_point_crossover.seed")
    gen_i = _validate_int(generation, "two_point_crossover.generation")
    ind_i = _validate_int(individual, "two_point_crossover.individual")
    extra_pairs = _validate_extra_meta(extra_meta)

    n = len(parent_a.values)
    # Two independent draws in [0, n]; sort to produce (i, j) with i <= j.
    raw_i = _uniform_int(0, n, seed_i, gen_i, ind_i, 0)
    raw_j = _uniform_int(0, n, seed_i, gen_i, ind_i, 1)
    cut_lo, cut_hi = (raw_i, raw_j) if raw_i <= raw_j else (raw_j, raw_i)

    a = parent_a.values
    b = parent_b.values
    c1_values = a[:cut_lo] + b[cut_lo:cut_hi] + a[cut_hi:]
    c2_values = b[:cut_lo] + a[cut_lo:cut_hi] + b[cut_hi:]

    # Re-clip / re-round through unpack for safety: a segment from
    # the other parent IS already feasible, but going through unpack
    # canonicalises INTEGER kinds (no-op for valid input).
    c1_clean = _decode_to_values(parent_a.specs, _encode_vector(parent_a.specs, c1_values))
    c2_clean = _decode_to_values(parent_b.specs, _encode_vector(parent_b.specs, c2_values))

    parent_a_digest = chromosome_digest(parent_a)
    parent_b_digest = chromosome_digest(parent_b)
    cut_extra = {"cut_lo": str(cut_lo), "cut_hi": str(cut_hi)}
    meta_a = _build_meta(
        operator=OPERATOR_TWO_POINT,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_a_digest=parent_a_digest,
        parent_b_digest=parent_b_digest,
        child=0,
        extra=cut_extra,
        extra_meta_pairs=extra_pairs,
    )
    meta_b = _build_meta(
        operator=OPERATOR_TWO_POINT,
        seed=seed_i,
        generation=gen_i,
        individual=ind_i,
        parent_a_digest=parent_a_digest,
        parent_b_digest=parent_b_digest,
        child=1,
        extra=cut_extra,
        extra_meta_pairs=extra_pairs,
    )
    child_a = _build_child(
        parent=parent_a,
        new_values=c1_clean,
        meta=meta_a,
        operator="two_point_crossover",
    )
    child_b = _build_child(
        parent=parent_b,
        new_values=c2_clean,
        meta=meta_b,
        operator="two_point_crossover",
    )
    return child_a, child_b


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _format_float(value: float) -> str:
    """Stable repr for floats stamped into ``meta`` (INV-15: the same
    input on any host produces the same digest)."""

    return repr(value)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "MAX_META_KEYS",
    "MAX_META_KEY_LEN",
    "MAX_META_VALUE_LEN",
    "NEW_PIP_DEPENDENCIES",
    "OPERATOR_BLEND",
    "OPERATOR_SBX",
    "OPERATOR_TWO_POINT",
    "CrossoverOperatorError",
    "blend_crossover",
    "simulated_binary_crossover",
    "two_point_crossover",
]
