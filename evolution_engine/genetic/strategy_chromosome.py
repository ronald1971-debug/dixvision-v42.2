# ADAPTED FROM: nnaisense/evotorch
# (evotorch/core.py — Problem / Solution / SolutionBatch parameter-vector
#  contract; evotorch/algorithms/cmaes.py — CMA-ES requires a flat real
#  parameter vector with deterministic bounds and inverse decoding.)
"""A-02.1 — StrategyChromosome: parameter-vector encoding for CMA-ES.

evotorch's :class:`Problem` operates on a flat real-valued parameter
vector (one row per :class:`Solution`). Trading strategy configs are
heterogeneous mappings (continuous risk knobs, log-scaled learning
rates, integer window sizes), so the canonical adapter is a typed
chromosome that round-trips between the two: ``pack`` projects a
strategy config onto the canonical vector that the CMA-ES sampler
mutates, and ``unpack`` decodes a CMA-ES sample back into a typed
strategy config that the simulation harness can evaluate.

What this module is
-------------------

* Pure-stdlib value objects. No numpy / no evotorch import — the
  CMA-ES sampler in :mod:`evolution_engine.genetic.cmaes_optimizer`
  (A-02.2) consumes :class:`StrategyChromosome` directly through plain
  Python tuples of floats. ``NEW_PIP_DEPENDENCIES = ()``.
* OFFLINE_ONLY tier. The chromosome reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``.
* INV-15 byte-identical. Float arithmetic is confined to
  :func:`pack` / :func:`unpack` / :func:`clip_to_bounds` — every
  numeric branch is a deterministic stdlib op (``math.log``,
  ``math.exp``, ``round``-half-to-even, plain ``min`` / ``max``).
* No clock reads. The caller supplies the version stamp.

Three parameter kinds are supported, mirroring evotorch's three
canonical search-space primitives:

* ``CONTINUOUS`` — linear bounds; CMA-ES sees the value verbatim.
* ``LOG_CONTINUOUS`` — log-uniform bounds; CMA-ES sees ``log10(value)``
  so the sampler walks a log scale (matches the canonical "learning
  rate" / "regularisation strength" knob in
  ``evotorch/algorithms/cmaes.py``).
* ``INTEGER`` — integer bounds; CMA-ES sees the value as a float and
  the inverse decode rounds half-to-even (banker's rounding) before
  re-clipping into ``[low, high]``.

What this module is **not**
---------------------------

* Not a CMA-ES implementation. That lives in
  :mod:`evolution_engine.genetic.cmaes_optimizer` (A-02.2) and
  consumes :class:`StrategyChromosome` instances.
* Not a strategy *runner*. The runner is the caller — typically the
  simulation harness or the sandbox PPO loop.
* Not a governance gate. Selection, promotion and rollback all live
  in :mod:`evolution_engine.patch_pipeline` (per Phase 5 spec); this
  module only encodes / decodes one chromosome at a time.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

# --- Constants -------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No new pip dependencies; the CMA-ES adapter is pure-stdlib."""

MAX_PARAMETER_NAME_LEN: int = 64
"""Hard cap on parameter name length — keeps ledger rows bounded."""

MAX_PARAMETERS_PER_CHROMOSOME: int = 256
"""Hard cap on chromosome dimensionality — CMA-ES is O(n^2)."""

MAX_STRATEGY_ID_LEN: int = 256
"""Hard cap on the parent strategy identifier."""

DIGEST_HEX_LEN: int = 16
"""Length of the BLAKE2b chromosome digest (lower-hex)."""

_DIGEST_BYTES: int = 8  # 16 hex chars == 8 bytes
_LOG_BASE: float = 10.0


# --- Errors ----------------------------------------------------------------


class ChromosomeError(ValueError):
    """Raised when a chromosome / parameter-spec invariant is violated."""


# --- Parameter kinds -------------------------------------------------------


class ParameterKind(StrEnum):
    """Search-space primitive for a single parameter."""

    CONTINUOUS = "CONTINUOUS"
    LOG_CONTINUOUS = "LOG_CONTINUOUS"
    INTEGER = "INTEGER"


# --- ParameterSpec ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Bounds + kind for a single search-space dimension.

    ``low`` / ``high`` are stored as ``float`` even for ``INTEGER`` kinds;
    the integer-kind invariant only requires that the bounds are
    integer-valued (``int(x) == x``) and that ``low <= high``.
    """

    name: str
    kind: ParameterKind
    low: float
    high: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):  # pragma: no cover - typed fence
            raise ChromosomeError("ParameterSpec.name must be str")
        if not self.name:
            raise ChromosomeError("ParameterSpec.name must be non-empty")
        if len(self.name) > MAX_PARAMETER_NAME_LEN:
            raise ChromosomeError(
                f"ParameterSpec.name length {len(self.name)} > {MAX_PARAMETER_NAME_LEN}"
            )
        if not isinstance(self.kind, ParameterKind):
            raise ChromosomeError("ParameterSpec.kind must be ParameterKind")
        if not isinstance(self.low, (int, float)) or isinstance(self.low, bool):
            raise ChromosomeError("ParameterSpec.low must be int|float")
        if not isinstance(self.high, (int, float)) or isinstance(self.high, bool):
            raise ChromosomeError("ParameterSpec.high must be int|float")
        if not math.isfinite(self.low):
            raise ChromosomeError("ParameterSpec.low must be finite")
        if not math.isfinite(self.high):
            raise ChromosomeError("ParameterSpec.high must be finite")
        if self.low >= self.high:
            raise ChromosomeError(
                f"ParameterSpec[{self.name}]: low ({self.low}) must be < high ({self.high})"
            )
        if self.kind is ParameterKind.LOG_CONTINUOUS and self.low <= 0.0:
            raise ChromosomeError(f"ParameterSpec[{self.name}]: LOG_CONTINUOUS requires low > 0")
        if self.kind is ParameterKind.INTEGER:
            if int(self.low) != self.low or int(self.high) != self.high:
                raise ChromosomeError(
                    f"ParameterSpec[{self.name}]: INTEGER bounds must be integer-valued"
                )

    def clip(self, value: float) -> float:
        """Clip ``value`` into ``[low, high]``. Linear for all kinds —
        log-scale clipping happens in the sampler (CMA-ES walks log
        space), not here.
        """

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ChromosomeError(f"ParameterSpec[{self.name}].clip: value must be int|float")
        v = float(value)
        if not math.isfinite(v):
            raise ChromosomeError(f"ParameterSpec[{self.name}].clip: value must be finite")
        return min(self.high, max(self.low, v))


# --- StrategyChromosome ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyChromosome:
    """Frozen, slotted chromosome carrying one strategy parameter sample.

    ``specs`` and ``values`` are aligned by index. ``len(specs) ==
    len(values)`` is enforced. Every value is bound-feasible per its
    spec (integer kinds also satisfy ``int(value) == value``).
    """

    strategy_id: str
    specs: tuple[ParameterSpec, ...]
    values: tuple[float, ...]
    version: int
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str):
            raise ChromosomeError("StrategyChromosome.strategy_id must be str")
        if not self.strategy_id:
            raise ChromosomeError("StrategyChromosome.strategy_id must be non-empty")
        if len(self.strategy_id) > MAX_STRATEGY_ID_LEN:
            raise ChromosomeError(
                f"StrategyChromosome.strategy_id length "
                f"{len(self.strategy_id)} > {MAX_STRATEGY_ID_LEN}"
            )
        if not isinstance(self.specs, tuple):
            raise ChromosomeError("StrategyChromosome.specs must be tuple")
        if not isinstance(self.values, tuple):
            raise ChromosomeError("StrategyChromosome.values must be tuple")
        if not self.specs:
            raise ChromosomeError("StrategyChromosome.specs must be non-empty")
        if len(self.specs) > MAX_PARAMETERS_PER_CHROMOSOME:
            raise ChromosomeError(
                f"StrategyChromosome dimensionality {len(self.specs)} > "
                f"{MAX_PARAMETERS_PER_CHROMOSOME}"
            )
        if len(self.specs) != len(self.values):
            raise ChromosomeError(
                f"StrategyChromosome: specs length {len(self.specs)} != "
                f"values length {len(self.values)}"
            )
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ChromosomeError("StrategyChromosome.version must be int")
        if self.version < 0:
            raise ChromosomeError("StrategyChromosome.version must be >= 0")
        seen_names: set[str] = set()
        for idx, (spec, value) in enumerate(zip(self.specs, self.values, strict=True)):
            if not isinstance(spec, ParameterSpec):
                raise ChromosomeError(f"StrategyChromosome.specs[{idx}] is not a ParameterSpec")
            if spec.name in seen_names:
                raise ChromosomeError(f"StrategyChromosome.specs: duplicate name {spec.name!r}")
            seen_names.add(spec.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ChromosomeError(
                    f"StrategyChromosome.values[{idx}] ({spec.name!r}) must be int|float"
                )
            v = float(value)
            if not math.isfinite(v):
                raise ChromosomeError(
                    f"StrategyChromosome.values[{idx}] ({spec.name!r}) must be finite"
                )
            if v < spec.low or v > spec.high:
                raise ChromosomeError(
                    f"StrategyChromosome.values[{idx}] ({spec.name!r}) "
                    f"= {v} out of bounds [{spec.low}, {spec.high}]"
                )
            if spec.kind is ParameterKind.INTEGER and int(v) != v:
                raise ChromosomeError(
                    f"StrategyChromosome.values[{idx}] ({spec.name!r}) "
                    f"INTEGER kind requires integer-valued sample, got {v}"
                )
        if not isinstance(self.meta, Mapping):
            raise ChromosomeError("StrategyChromosome.meta must be Mapping")
        for mk, mv in self.meta.items():
            if not isinstance(mk, str) or not isinstance(mv, str):
                raise ChromosomeError("StrategyChromosome.meta must map str -> str")

    @property
    def dimensionality(self) -> int:
        """Number of parameter dimensions (equal to ``len(values)``)."""

        return len(self.specs)

    def to_mapping(self) -> Mapping[str, float]:
        """Return an immutable mapping ``name -> value`` (decoded for
        the integer kind so callers receive the integer sample as a
        Python ``float`` whose ``int(x) == x``).
        """

        return MappingProxyType(
            {spec.name: value for spec, value in zip(self.specs, self.values, strict=True)}
        )


# --- Pack / unpack ---------------------------------------------------------


def pack(
    specs: tuple[ParameterSpec, ...],
    mapping: Mapping[str, float],
) -> tuple[float, ...]:
    """Project a typed strategy config onto a canonical CMA-ES vector.

    For ``CONTINUOUS`` and ``INTEGER`` parameters the value passes
    through verbatim (after a finite + bound check). For
    ``LOG_CONTINUOUS`` parameters the projection is ``log10(value)``,
    mirroring evotorch's log-scale knob in ``cmaes.py``.

    Raises :class:`ChromosomeError` on bound violation, kind violation,
    or missing parameter.
    """

    if not isinstance(specs, tuple):
        raise ChromosomeError("pack: specs must be tuple")
    if not isinstance(mapping, Mapping):
        raise ChromosomeError("pack: mapping must be Mapping")
    if not specs:
        raise ChromosomeError("pack: specs must be non-empty")

    out: list[float] = []
    for spec in specs:
        if not isinstance(spec, ParameterSpec):  # pragma: no cover - typed
            raise ChromosomeError("pack: specs entry is not a ParameterSpec")
        if spec.name not in mapping:
            raise ChromosomeError(f"pack: mapping missing parameter {spec.name!r}")
        raw = mapping[spec.name]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ChromosomeError(f"pack: value for {spec.name!r} must be int|float")
        v = float(raw)
        if not math.isfinite(v):
            raise ChromosomeError(f"pack: value for {spec.name!r} must be finite")
        if v < spec.low or v > spec.high:
            raise ChromosomeError(
                f"pack: value for {spec.name!r} = {v} out of bounds [{spec.low}, {spec.high}]"
            )
        if spec.kind is ParameterKind.INTEGER:
            if int(v) != v:
                raise ChromosomeError(
                    f"pack: INTEGER value for {spec.name!r} must be integer-valued, got {v}"
                )
            out.append(v)
        elif spec.kind is ParameterKind.LOG_CONTINUOUS:
            # spec.low > 0 is enforced in __post_init__, and v >= low > 0
            # so log10(v) is finite.
            out.append(math.log10(v) / math.log10(_LOG_BASE))
        else:
            out.append(v)
    return tuple(out)


def unpack(
    specs: tuple[ParameterSpec, ...],
    vector: tuple[float, ...],
) -> Mapping[str, float]:
    """Inverse of :func:`pack`. Decodes a CMA-ES sample into a typed
    strategy config.

    For ``LOG_CONTINUOUS`` the inverse is ``10**v``; for ``INTEGER`` the
    inverse is ``round(v)`` (half-to-even / banker's rounding via
    Python's built-in :func:`round`) followed by re-clipping into
    ``[low, high]`` so out-of-range CMA-ES proposals snap onto the grid
    rather than raising.

    The output is bound-feasible for every parameter.
    """

    if not isinstance(specs, tuple):
        raise ChromosomeError("unpack: specs must be tuple")
    if not isinstance(vector, tuple):
        raise ChromosomeError("unpack: vector must be tuple")
    if len(specs) != len(vector):
        raise ChromosomeError(f"unpack: specs length {len(specs)} != vector length {len(vector)}")
    if not specs:
        raise ChromosomeError("unpack: specs must be non-empty")

    decoded: dict[str, float] = {}
    for spec, raw in zip(specs, vector, strict=True):
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ChromosomeError(f"unpack: vector entry for {spec.name!r} must be int|float")
        v = float(raw)
        if not math.isfinite(v):
            raise ChromosomeError(f"unpack: vector entry for {spec.name!r} must be finite")
        if spec.kind is ParameterKind.LOG_CONTINUOUS:
            decoded_value = math.pow(_LOG_BASE, v)
            decoded_value = min(spec.high, max(spec.low, decoded_value))
        elif spec.kind is ParameterKind.INTEGER:
            rounded = float(round(v))
            decoded_value = min(spec.high, max(spec.low, rounded))
        else:
            decoded_value = min(spec.high, max(spec.low, v))
        decoded[spec.name] = decoded_value
    return MappingProxyType(decoded)


def clip_to_bounds(
    specs: tuple[ParameterSpec, ...],
    vector: tuple[float, ...],
) -> tuple[float, ...]:
    """Clip a *decoded* vector (same kind-space as :func:`unpack`'s
    output but as a tuple) into the per-spec bounds. Re-applies the
    integer rounding for ``INTEGER`` parameters. Used by the CMA-ES
    sampler to project a freshly-mutated proposal back onto the
    feasible region before storing it as the next centroid.
    """

    if not isinstance(specs, tuple):
        raise ChromosomeError("clip_to_bounds: specs must be tuple")
    if not isinstance(vector, tuple):
        raise ChromosomeError("clip_to_bounds: vector must be tuple")
    if len(specs) != len(vector):
        raise ChromosomeError(
            f"clip_to_bounds: specs length {len(specs)} != vector length {len(vector)}"
        )
    if not specs:
        raise ChromosomeError("clip_to_bounds: specs must be non-empty")

    out: list[float] = []
    for spec, raw in zip(specs, vector, strict=True):
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ChromosomeError(
                f"clip_to_bounds: vector entry for {spec.name!r} must be int|float"
            )
        v = float(raw)
        if not math.isfinite(v):
            raise ChromosomeError(f"clip_to_bounds: vector entry for {spec.name!r} must be finite")
        if spec.kind is ParameterKind.INTEGER:
            v = float(round(v))
        out.append(min(spec.high, max(spec.low, v)))
    return tuple(out)


# --- Digest ----------------------------------------------------------------


def chromosome_digest(chromosome: StrategyChromosome) -> str:
    """Return a 16-lower-hex BLAKE2b-8 digest content-addressing the
    chromosome. Stable across runs / hosts (no hash randomisation,
    canonical text projection only).
    """

    if not isinstance(chromosome, StrategyChromosome):
        raise ChromosomeError("chromosome_digest: argument must be StrategyChromosome")
    parts: list[str] = [
        f"strategy_id={chromosome.strategy_id}",
        f"version={chromosome.version}",
        f"dim={chromosome.dimensionality}",
    ]
    for spec, value in zip(chromosome.specs, chromosome.values, strict=True):
        parts.append(
            f"spec|name={spec.name}|kind={spec.kind.value}|"
            f"low={spec.low!r}|high={spec.high!r}|value={value!r}"
        )
    for mk in sorted(chromosome.meta.keys()):
        parts.append(f"meta|{mk}={chromosome.meta[mk]}")
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).hexdigest()


# --- Public surface --------------------------------------------------------

__all__ = [
    "DIGEST_HEX_LEN",
    "MAX_PARAMETERS_PER_CHROMOSOME",
    "MAX_PARAMETER_NAME_LEN",
    "MAX_STRATEGY_ID_LEN",
    "NEW_PIP_DEPENDENCIES",
    "ChromosomeError",
    "ParameterKind",
    "ParameterSpec",
    "StrategyChromosome",
    "chromosome_digest",
    "clip_to_bounds",
    "pack",
    "unpack",
]
