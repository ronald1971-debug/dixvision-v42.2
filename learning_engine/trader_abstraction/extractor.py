# ADAPTED FROM: blue-yonder/tsfresh
# (tsfresh/feature_extraction/feature_calculators.py — individual
#  statistical calculators; tsfresh/feature_extraction/extraction.py —
#  extract_features() pipeline; tsfresh/feature_extraction/settings.py —
#  MinimalFCParameters / EfficientFCParameters / ComprehensiveFCParameters
#  presets.)
"""B-06 — tsfresh canonical adaptation: time-series feature extractor.

Pure-Python, stdlib-only port of tsfresh's per-time-series statistical
feature calculators behind a frozen DIX contract. Designed for
``learning_engine``'s OFFLINE trader-abstraction lane: take a single
1-D price / volume / spread time series and emit a deterministic
:class:`FeatureVector` (a sorted-key mapping of feature name → float)
that downstream consumers (feature store, learning lanes, evolution
engine) can ledger, replay, and diff byte-for-byte.

Algorithmic surface ported from tsfresh:

* ``tsfresh.feature_extraction.feature_calculators.mean`` →
  :func:`_mean`
* ``tsfresh.feature_extraction.feature_calculators.median`` →
  :func:`_median`
* ``tsfresh.feature_extraction.feature_calculators.standard_deviation``
  → :func:`_standard_deviation` (population, not sample, to match
  tsfresh)
* ``tsfresh.feature_extraction.feature_calculators.variance`` →
  :func:`_variance` (population)
* ``tsfresh.feature_extraction.feature_calculators.maximum`` /
  ``minimum`` / ``length`` / ``sum_values`` / ``root_mean_square`` /
  ``absolute_maximum`` → corresponding ``_*`` helpers below
* ``tsfresh.feature_extraction.feature_calculators.abs_energy`` →
  :func:`_abs_energy`
* ``tsfresh.feature_extraction.feature_calculators.mean_change`` →
  :func:`_mean_change`
* ``tsfresh.feature_extraction.feature_calculators.mean_abs_change`` →
  :func:`_mean_abs_change`
* ``tsfresh.feature_extraction.feature_calculators.count_above_mean``
  / ``count_below_mean`` → :func:`_count_above_mean` /
  :func:`_count_below_mean`
* ``tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean``
  / ``longest_strike_below_mean`` → :func:`_longest_strike_above_mean`
  / :func:`_longest_strike_below_mean`
* ``tsfresh.feature_extraction.feature_calculators.first_location_of_maximum``
  / ``last_location_of_maximum`` / ``first_location_of_minimum`` /
  ``last_location_of_minimum`` → corresponding ``_*`` helpers
* ``tsfresh.feature_extraction.feature_calculators.skewness`` →
  :func:`_skewness` (Fisher-Pearson, population)
* ``tsfresh.feature_extraction.feature_calculators.kurtosis`` →
  :func:`_kurtosis` (Fisher, population)
* ``tsfresh.feature_extraction.feature_calculators.quantile`` →
  :func:`_quantile` (linear interpolation, matches numpy's default)
* ``tsfresh.feature_extraction.settings.MinimalFCParameters`` →
  :data:`MINIMAL_FEATURE_NAMES`
* ``tsfresh.feature_extraction.settings.EfficientFCParameters`` →
  :data:`EFFICIENT_FEATURE_NAMES`
* ``tsfresh.feature_extraction.extract_features(...)`` →
  :func:`extract_features`

DIX integration rules (verbatim from PART 1 + the B-06 spec at
:file:`docs/DIX_MASTER_CANONICAL.md` lines 1756–1790):

* **OFFLINE_ONLY tier** — every public entrypoint is a pure function;
  no clock, no PRNG, no IO. The hot path never calls it. Authority-lint
  pins this via AST tests (no top-level ``tsfresh`` / ``pandas`` /
  ``numpy`` / ``scipy`` / ``random`` / ``time`` / ``datetime`` /
  ``asyncio`` / ``os`` / ``websockets`` / ``langsmith`` imports).
* **B27 / B28 / INV-71 authority symmetry** — the extractor emits
  :class:`FeatureVector` value objects only. It never constructs typed
  bus events (``PatchProposal`` / ``SignalEvent`` /
  ``GovernanceDecision``). The producing engines lift the feature
  vector onto their own typed envelopes.
* **B1 engine isolation** — no ``governance_engine`` /
  ``system_engine`` / ``execution_engine`` / ``evolution_engine`` /
  ``intelligence_engine`` cross-imports. Pinned.
* **INV-15 byte-identical replay** — every calculation runs on
  caller-supplied :class:`TimeSeries` data with a caller-supplied
  ``ts_ns``. Output is a frozen, sorted-key mapping with a stable
  BLAKE2b-16 digest. 3-run identical-digest equality is pinned by the
  test suite.
* **Pure stdlib** — only ``dataclasses``, ``enum``, ``hashlib``,
  ``math``, ``types``, ``typing``, ``collections.abc``. The full
  ``NEW_PIP_DEPENDENCIES = ()``.

Feature value semantics:

* All features return ``float`` (never ``None``); empty / pathological
  inputs raise :class:`FeatureExtractionError` rather than returning
  ``NaN`` so caller pipelines fail fast at ingestion time.
* For series of length 1, ``mean_change`` / ``mean_abs_change`` /
  ``skewness`` / ``kurtosis`` raise :class:`FeatureExtractionError` —
  callers should pad or drop those windows before extraction.
* Sample statistics (variance, std, skew, kurt) use the **population**
  estimator (divide by ``n``, not ``n - 1``) to match tsfresh's
  upstream defaults.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()

__all__ = [
    "EFFICIENT_FEATURE_NAMES",
    "FEATURE_CALCULATORS",
    "FeatureExtractionError",
    "FeatureSet",
    "FeatureSpec",
    "FeatureVector",
    "MINIMAL_FEATURE_NAMES",
    "NEW_PIP_DEPENDENCIES",
    "TimeSeries",
    "calculate_feature",
    "extract_features",
]


# ---------------------------------------------------------------------------
# Bounds / constants
# ---------------------------------------------------------------------------

MIN_SERIES_LEN: Final[int] = 1
MAX_SERIES_LEN: Final[int] = 1_000_000
MAX_NAME_LEN: Final[int] = 64
DIGEST_HEX_LEN: Final[int] = 32  # BLAKE2b-16 → 32 hex chars


# ---------------------------------------------------------------------------
# Errors / enums / value objects
# ---------------------------------------------------------------------------


class FeatureExtractionError(ValueError):
    """Raised when input is malformed or a feature is undefined for input."""


class FeatureSet(StrEnum):
    """Canonical feature-set preset (mirrors tsfresh's ``*FCParameters``)."""

    MINIMAL = "minimal"
    EFFICIENT = "efficient"


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """A single 1-D numeric time series.

    Attributes:
        name: Series identifier (e.g. ``"close"``, ``"volume"``).
        values: Tuple of finite floats in time order.
        ts_ns: Caller-supplied timestamp for ledger correlation
            (TimeAuthority / T0-04 / INV-15).
    """

    name: str
    values: tuple[float, ...]
    ts_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise FeatureExtractionError("TimeSeries.name must be non-empty str")
        if len(self.name) > MAX_NAME_LEN:
            raise FeatureExtractionError("TimeSeries.name too long")
        if not isinstance(self.values, tuple):
            raise FeatureExtractionError("TimeSeries.values must be tuple")
        if len(self.values) < MIN_SERIES_LEN:
            raise FeatureExtractionError("TimeSeries.values must be non-empty")
        if len(self.values) > MAX_SERIES_LEN:
            raise FeatureExtractionError("TimeSeries.values exceeds MAX_SERIES_LEN")
        for i, value in enumerate(self.values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FeatureExtractionError(f"TimeSeries.values[{i}] must be a real number")
            f = float(value)
            if f != f or f in (math.inf, -math.inf):
                raise FeatureExtractionError(f"TimeSeries.values[{i}] must be finite")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise FeatureExtractionError("TimeSeries.ts_ns must be int")
        if self.ts_ns < 0:
            raise FeatureExtractionError("TimeSeries.ts_ns must be >= 0")
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """An advisory request for a specific feature set on a series.

    Attributes:
        series_name: The :attr:`TimeSeries.name` this spec applies to.
        feature_set: Preset bundle to compute.
        quantile_levels: Quantile probabilities (in ``(0, 1)``) to add
            on top of the preset bundle. Stored sorted-ascending.
    """

    series_name: str
    feature_set: FeatureSet = FeatureSet.MINIMAL
    quantile_levels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.series_name, str) or not self.series_name:
            raise FeatureExtractionError("FeatureSpec.series_name must be non-empty str")
        if len(self.series_name) > MAX_NAME_LEN:
            raise FeatureExtractionError("FeatureSpec.series_name too long")
        if not isinstance(self.feature_set, FeatureSet):
            raise FeatureExtractionError("FeatureSpec.feature_set must be FeatureSet")
        if not isinstance(self.quantile_levels, tuple):
            raise FeatureExtractionError("FeatureSpec.quantile_levels must be tuple")
        for q in self.quantile_levels:
            if isinstance(q, bool) or not isinstance(q, (int, float)):
                raise FeatureExtractionError(
                    "FeatureSpec.quantile_levels entries must be real numbers"
                )
            fq = float(q)
            if not (0.0 < fq < 1.0):
                raise FeatureExtractionError(
                    "FeatureSpec.quantile_levels entries must be in (0, 1)"
                )
        sorted_qs = tuple(sorted(float(q) for q in self.quantile_levels))
        object.__setattr__(self, "quantile_levels", sorted_qs)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A frozen, sorted-key, digest-stable feature vector.

    Attributes:
        series_name: The :attr:`TimeSeries.name` these features were
            extracted from.
        feature_set: Which preset bundle drove the extraction.
        values: Sorted-key mapping of feature name → float value.
        ts_ns: Source :attr:`TimeSeries.ts_ns` (forwarded verbatim).
        digest: 32-char BLAKE2b-16 hex digest over the canonical
            text projection of ``(series_name, feature_set, values,
            ts_ns)``. Pinned by INV-15 byte-identical replay tests.
    """

    series_name: str
    feature_set: FeatureSet
    values: Mapping[str, float]
    ts_ns: int
    digest: str = field(default="")

    def __post_init__(self) -> None:
        if not isinstance(self.series_name, str) or not self.series_name:
            raise FeatureExtractionError("FeatureVector.series_name must be non-empty str")
        if not isinstance(self.feature_set, FeatureSet):
            raise FeatureExtractionError("FeatureVector.feature_set must be FeatureSet")
        if not isinstance(self.values, Mapping) or not self.values:
            raise FeatureExtractionError("FeatureVector.values must be a non-empty Mapping")
        for key, value in self.values.items():
            if not isinstance(key, str) or not key:
                raise FeatureExtractionError("FeatureVector.values keys must be non-empty str")
            if len(key) > MAX_NAME_LEN:
                raise FeatureExtractionError("FeatureVector.values key too long")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FeatureExtractionError(f"FeatureVector.values[{key!r}] must be a real number")
            f = float(value)
            if f != f or f in (math.inf, -math.inf):
                raise FeatureExtractionError(f"FeatureVector.values[{key!r}] must be finite")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise FeatureExtractionError("FeatureVector.ts_ns must be int")
        if self.ts_ns < 0:
            raise FeatureExtractionError("FeatureVector.ts_ns must be >= 0")
        sorted_values = {k: float(self.values[k]) for k in sorted(self.values)}
        object.__setattr__(self, "values", MappingProxyType(sorted_values))
        if not self.digest:
            object.__setattr__(
                self,
                "digest",
                _compute_digest(self.series_name, self.feature_set, sorted_values, self.ts_ns),
            )
        else:
            if not isinstance(self.digest, str) or len(self.digest) != DIGEST_HEX_LEN:
                raise FeatureExtractionError(
                    f"FeatureVector.digest must be {DIGEST_HEX_LEN}-char hex str"
                )
            try:
                int(self.digest, 16)
            except ValueError as exc:  # pragma: no cover - defensive
                raise FeatureExtractionError("FeatureVector.digest must be hex") from exc


# ---------------------------------------------------------------------------
# Float formatting / digesting
# ---------------------------------------------------------------------------


def _format_float(value: float) -> str:
    """Stable, lossless float text (matches `Indicator.meta` formatting)."""

    return repr(float(value))


def _compute_digest(
    series_name: str,
    feature_set: FeatureSet,
    values: Mapping[str, float],
    ts_ns: int,
) -> str:
    """BLAKE2b-16 digest over the canonical text projection."""

    parts: list[str] = [series_name, feature_set.value, str(ts_ns)]
    for key in sorted(values):
        parts.append(f"{key}={_format_float(values[key])}")
    payload = "\u001f".join(parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Calculators (adapted from tsfresh/feature_extraction/feature_calculators.py)
# ---------------------------------------------------------------------------


def _length(x: Sequence[float]) -> float:
    return float(len(x))


def _sum_values(x: Sequence[float]) -> float:
    return float(math.fsum(x))


def _mean(x: Sequence[float]) -> float:
    return math.fsum(x) / len(x)


def _median(x: Sequence[float]) -> float:
    n = len(x)
    sorted_x = sorted(x)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_x[mid])
    return (sorted_x[mid - 1] + sorted_x[mid]) / 2.0


def _maximum(x: Sequence[float]) -> float:
    return float(max(x))


def _minimum(x: Sequence[float]) -> float:
    return float(min(x))


def _absolute_maximum(x: Sequence[float]) -> float:
    return float(max(abs(v) for v in x))


def _variance(x: Sequence[float]) -> float:
    mean = _mean(x)
    return math.fsum((v - mean) ** 2 for v in x) / len(x)


def _standard_deviation(x: Sequence[float]) -> float:
    return math.sqrt(_variance(x))


def _root_mean_square(x: Sequence[float]) -> float:
    return math.sqrt(math.fsum(v * v for v in x) / len(x))


def _abs_energy(x: Sequence[float]) -> float:
    return math.fsum(v * v for v in x)


def _mean_change(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        raise FeatureExtractionError("mean_change requires len >= 2")
    return (x[-1] - x[0]) / (n - 1)


def _mean_abs_change(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        raise FeatureExtractionError("mean_abs_change requires len >= 2")
    return math.fsum(abs(x[i + 1] - x[i]) for i in range(n - 1)) / (n - 1)


def _count_above_mean(x: Sequence[float]) -> float:
    mean = _mean(x)
    return float(sum(1 for v in x if v > mean))


def _count_below_mean(x: Sequence[float]) -> float:
    mean = _mean(x)
    return float(sum(1 for v in x if v < mean))


def _longest_strike_above_mean(x: Sequence[float]) -> float:
    mean = _mean(x)
    best = 0
    run = 0
    for v in x:
        if v > mean:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return float(best)


def _longest_strike_below_mean(x: Sequence[float]) -> float:
    mean = _mean(x)
    best = 0
    run = 0
    for v in x:
        if v < mean:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return float(best)


def _first_location_of_maximum(x: Sequence[float]) -> float:
    n = len(x)
    target = max(x)
    for i, v in enumerate(x):
        if v == target:
            return i / n
    return 0.0  # pragma: no cover - max always present


def _last_location_of_maximum(x: Sequence[float]) -> float:
    n = len(x)
    target = max(x)
    for i in range(n - 1, -1, -1):
        if x[i] == target:
            return (i + 1) / n
    return 0.0  # pragma: no cover - max always present


def _first_location_of_minimum(x: Sequence[float]) -> float:
    n = len(x)
    target = min(x)
    for i, v in enumerate(x):
        if v == target:
            return i / n
    return 0.0  # pragma: no cover - min always present


def _last_location_of_minimum(x: Sequence[float]) -> float:
    n = len(x)
    target = min(x)
    for i in range(n - 1, -1, -1):
        if x[i] == target:
            return (i + 1) / n
    return 0.0  # pragma: no cover - min always present


def _skewness(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        raise FeatureExtractionError("skewness requires len >= 2")
    mean = _mean(x)
    m2 = math.fsum((v - mean) ** 2 for v in x) / n
    if m2 == 0.0:
        return 0.0
    m3 = math.fsum((v - mean) ** 3 for v in x) / n
    return m3 / (m2**1.5)


def _kurtosis(x: Sequence[float]) -> float:
    """Fisher (excess) kurtosis, population estimator."""

    n = len(x)
    if n < 2:
        raise FeatureExtractionError("kurtosis requires len >= 2")
    mean = _mean(x)
    m2 = math.fsum((v - mean) ** 2 for v in x) / n
    if m2 == 0.0:
        return -3.0
    m4 = math.fsum((v - mean) ** 4 for v in x) / n
    return m4 / (m2 * m2) - 3.0


def _quantile(x: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (matches numpy ``interpolation='linear'``)."""

    if not 0.0 < q < 1.0:
        raise FeatureExtractionError("quantile q must be in (0, 1)")
    sorted_x = sorted(x)
    n = len(sorted_x)
    if n == 1:
        return float(sorted_x[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_x[lo])
    frac = pos - lo
    return sorted_x[lo] + (sorted_x[hi] - sorted_x[lo]) * frac


# ---------------------------------------------------------------------------
# Calculator registry
# ---------------------------------------------------------------------------


def _make_quantile_calc(q: float):
    def _calc(x: Sequence[float]) -> float:
        return _quantile(x, q)

    return _calc


# Canonical name → calculator (sorted alphabetically for INV-15 stability).
FEATURE_CALCULATORS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "abs_energy": _abs_energy,
        "absolute_maximum": _absolute_maximum,
        "count_above_mean": _count_above_mean,
        "count_below_mean": _count_below_mean,
        "first_location_of_maximum": _first_location_of_maximum,
        "first_location_of_minimum": _first_location_of_minimum,
        "kurtosis": _kurtosis,
        "last_location_of_maximum": _last_location_of_maximum,
        "last_location_of_minimum": _last_location_of_minimum,
        "length": _length,
        "longest_strike_above_mean": _longest_strike_above_mean,
        "longest_strike_below_mean": _longest_strike_below_mean,
        "maximum": _maximum,
        "mean": _mean,
        "mean_abs_change": _mean_abs_change,
        "mean_change": _mean_change,
        "median": _median,
        "minimum": _minimum,
        "root_mean_square": _root_mean_square,
        "skewness": _skewness,
        "standard_deviation": _standard_deviation,
        "sum_values": _sum_values,
        "variance": _variance,
    }
)


# Mirrors tsfresh's ``MinimalFCParameters`` preset (10 calculators).
MINIMAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "absolute_maximum",
    "length",
    "maximum",
    "mean",
    "median",
    "minimum",
    "root_mean_square",
    "standard_deviation",
    "sum_values",
    "variance",
)


# A curated subset of tsfresh's ``EfficientFCParameters`` — every
# calculator below is O(n) and side-effect-free; matches the upstream
# "efficient" tier where tsfresh excludes O(n log n) and FFT-based
# calculators.
EFFICIENT_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            *MINIMAL_FEATURE_NAMES,
            "abs_energy",
            "count_above_mean",
            "count_below_mean",
            "first_location_of_maximum",
            "first_location_of_minimum",
            "kurtosis",
            "last_location_of_maximum",
            "last_location_of_minimum",
            "longest_strike_above_mean",
            "longest_strike_below_mean",
            "mean_abs_change",
            "mean_change",
            "skewness",
        }
    )
)


def _preset_names(feature_set: FeatureSet) -> tuple[str, ...]:
    if feature_set is FeatureSet.MINIMAL:
        return MINIMAL_FEATURE_NAMES
    if feature_set is FeatureSet.EFFICIENT:
        return EFFICIENT_FEATURE_NAMES
    raise FeatureExtractionError(f"unknown feature_set: {feature_set!r}")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def calculate_feature(name: str, series: TimeSeries) -> float:
    """Compute a single named feature on ``series``.

    Quantile features must be addressed by canonical name
    ``"quantile_{q}"`` where ``q`` is formatted via ``repr(float(q))``.
    """

    if not isinstance(series, TimeSeries):
        raise FeatureExtractionError("series must be TimeSeries")
    if not isinstance(name, str) or not name:
        raise FeatureExtractionError("name must be non-empty str")
    if name in FEATURE_CALCULATORS:
        return float(FEATURE_CALCULATORS[name](series.values))  # type: ignore[operator]
    if name.startswith("quantile_"):
        try:
            q = float(name[len("quantile_") :])
        except ValueError as exc:
            raise FeatureExtractionError(f"unparseable quantile feature name: {name!r}") from exc
        return _quantile(series.values, q)
    raise FeatureExtractionError(f"unknown feature: {name!r}")


def extract_features(series: TimeSeries, spec: FeatureSpec) -> FeatureVector:
    """Extract a deterministic :class:`FeatureVector` from ``series``.

    The output's ``digest`` is the BLAKE2b-16 hex of the canonical text
    projection (sorted feature names + repr-formatted floats + ts_ns).
    Identical input → identical digest across runs / machines / Python
    instances. Pinned by INV-15 byte-identical replay tests.
    """

    if not isinstance(series, TimeSeries):
        raise FeatureExtractionError("series must be TimeSeries")
    if not isinstance(spec, FeatureSpec):
        raise FeatureExtractionError("spec must be FeatureSpec")
    if series.name != spec.series_name:
        raise FeatureExtractionError("FeatureSpec.series_name does not match TimeSeries.name")
    preset_names = _preset_names(spec.feature_set)
    values: dict[str, float] = {}
    for name in preset_names:
        values[name] = float(FEATURE_CALCULATORS[name](series.values))  # type: ignore[operator]
    for q in spec.quantile_levels:
        key = f"quantile_{_format_float(q)}"
        values[key] = _quantile(series.values, q)
    return FeatureVector(
        series_name=series.name,
        feature_set=spec.feature_set,
        values=values,
        ts_ns=series.ts_ns,
    )
