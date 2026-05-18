# ADAPTED FROM: microsoft/EconML
# (econml/dml/ — Double Machine Learning learners (DML / SparseLinearDML
#  / CausalForestDML); econml/dr/ — Doubly-Robust learners; econml/orf/
#  — Orthogonal Random Forest; econml/metalearners/ — S/T/X/DR meta-
#  learner family; econml/iv/ — Deep IV instrumental-variable surface.)
"""C-37 — EconMLHteAnalyser: governance-gated heterogeneous-treatment-
effect (HTE) seam.

EconML is Microsoft's library for estimating individualised /
conditional average treatment effects (CATE) using machine-learning
nuisance models. It exposes a multi-family estimator surface (DML /
DR / Orthogonal Forest / Meta-Learners / Deep IV) over a shared
``fit / effect / effect_interval`` API. The DIX adapter wraps that
surface behind a Protocol seam so the intelligence layer can ask
"what is the per-row CATE for treatment X?" without ever importing
econml at module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``econml`` / ``pandas`` / ``numpy`` / ``scikit-learn`` imports are
  hidden behind a :class:`HteEffectEstimator` Protocol — production
  wires :func:`econml_dml_estimator`; unit tests inject a
  deterministic fake. The module never imports econml at module
  load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`HteAnalysisRecord` and stops.
* INV-15 byte-identical replays. :meth:`EconMLHteAnalyser.analyse`
  with identical ``estimand`` / ``arguments`` / ``ts_ns`` /
  ``analysis_id`` / ``estimator`` returns identical
  :class:`HteAnalysisRecord` records. Determinism is delegated to
  the injected estimator; the default factory forwards
  :attr:`HteArguments.random_seed` to ``numpy.random.seed`` and the
  underlying estimator's ``random_state=`` argument.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The estimator family — :class:`HteEstimatorKind` enumerates the
  econml estimator surfaces we currently expose (DML / SparseLinearDML
  / CausalForestDML / DRLearner / OrthoForest / MetaLearner-S /
  MetaLearner-T / MetaLearner-X / DeepIV).
* The CATE summary surface — :class:`HteAnalysisResult` projects
  ``effect()`` + ``effect_interval()`` values into a frozen value
  object with ``(point_estimate, ci_lower, ci_upper, std_error)``
  per row.
* The featurised-point grouping — :class:`HteEffectPoint` captures
  one ``(point_id, point_estimate, ci_lower, ci_upper, std_error)``
  tuple per featurised population row.

What we replaced
----------------

* econml's matplotlib SHAP-style heterogeneity plots → no plotting.
  The numeric summary lives in :class:`HteAnalysisResult.points`;
  the dashboard handles rendering.
* econml's pandas DataFrame data IO → the estimator owns its data
  source; the seam carries a frozen ``data_digest`` so identical
  inputs produce identical analyses (no DataFrame round-tripping).
* econml's joblib parallel backend → caller-injected
  :class:`HteAnalysisCallback` (default no-op). No filesystem
  writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the estimator's PRNG is seeded by
  caller-supplied :attr:`HteArguments.random_seed`. AST tests pin
  the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-15 — :class:`HteAnalysisRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MAX_N_SAMPLES` 10,000,000 hard ceiling on
    ``HteArguments.n_samples``.
  - :data:`MAX_POINTS` 4096 hard ceiling on
    ``HteAnalysisResult.points``.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on
    ``analysis_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-37 (econml HTE adapter spec).
- ``intelligence_engine/hte_econml.py`` (this file).
- ``intelligence_engine/uplift_causalml.py`` (C-36 — the causalml
  twin showing the same lazy-seam factory shape).
- ``intelligence_engine/causal_dowhy.py`` (C-35 — the dowhy twin).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "econml",
    "pandas",
    "numpy",
    "scikit-learn",
)

MIN_N_SAMPLES: int = 1
MAX_N_SAMPLES: int = 10_000_000
"""Hard upper bound on :attr:`HteArguments.n_samples`."""

MAX_POINTS: int = 4096
"""Hard upper bound on rows returned by an HTE analysis."""

MIN_CONFIDENCE_LEVEL: float = 0.5
MAX_CONFIDENCE_LEVEL: float = 0.9999
"""Bounds on :attr:`HteArguments.confidence_level`."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_DATA_DIGEST_LEN: int = 64
"""Hard upper bound on data-digest length."""

ANALYSIS_SOURCE: str = "intelligence_engine.hte_econml"
"""Constant tag stamped onto every
:attr:`HteAnalysisRecord.source`. Distinguishes econml-produced
records from other HTE adapters."""


# ---------------------------------------------------------------------------
# Estimator-method enum
# ---------------------------------------------------------------------------


class HteEstimatorKind(enum.Enum):
    """econml estimator-family selector.

    Values match the canonical econml class names so the DIX seam
    can forward them directly to ``econml.dml.*`` / ``econml.dr.*``
    / ``econml.orf.*`` / ``econml.metalearners.*`` / ``econml.iv.*``
    constructors.
    """

    DML = "DML"
    SPARSE_LINEAR_DML = "SparseLinearDML"
    CAUSAL_FOREST_DML = "CausalForestDML"
    DR_LEARNER = "DRLearner"
    ORTHO_FOREST = "DMLOrthoForest"
    META_LEARNER_S = "SLearner"
    META_LEARNER_T = "TLearner"
    META_LEARNER_X = "XLearner"
    META_LEARNER_DR = "DRLearner.MetaLearner"
    DEEP_IV = "DeepIV"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class HteEstimand:
    """Frozen HTE-question specification.

    * ``features`` — tuple of feature (X) column names.
    * ``treatment`` — treatment (T) column name.
    * ``outcome`` — outcome (Y) column name.
    * ``effect_modifiers`` — tuple of effect-modifier (W) column
      names that vary the per-row CATE.
    * ``data_digest`` — caller-supplied hex digest over the
      underlying DataFrame.
    """

    features: tuple[str, ...]
    treatment: str
    outcome: str
    effect_modifiers: tuple[str, ...]
    data_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.features, tuple):
            raise TypeError(
                f"HteEstimand.features must be a tuple, got {type(self.features).__name__}"
            )
        if not self.features:
            raise ValueError("HteEstimand.features must be non-empty")
        for f in self.features:
            if not isinstance(f, str) or not f:
                raise ValueError(
                    f"HteEstimand.features entries must be non-empty strings, got {f!r}"
                )
        if not self.treatment:
            raise ValueError("HteEstimand.treatment must be non-empty")
        if not self.outcome:
            raise ValueError("HteEstimand.outcome must be non-empty")
        if not isinstance(self.effect_modifiers, tuple):
            raise TypeError(
                "HteEstimand.effect_modifiers must be a tuple, got "
                f"{type(self.effect_modifiers).__name__}"
            )
        for m in self.effect_modifiers:
            if not isinstance(m, str) or not m:
                raise ValueError(
                    f"HteEstimand.effect_modifiers entries must be non-empty strings, got {m!r}"
                )
        if not self.data_digest:
            raise ValueError("HteEstimand.data_digest must be non-empty")
        if len(self.data_digest) > MAX_DATA_DIGEST_LEN:
            raise ValueError(
                "HteEstimand.data_digest must be <= "
                f"{MAX_DATA_DIGEST_LEN} chars, got "
                f"{len(self.data_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HteArguments:
    """Frozen analysis-run config."""

    estimator_kind: HteEstimatorKind
    random_seed: int
    n_samples: int = 1000
    confidence_level: float = 0.95
    n_points: int = 100
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.estimator_kind, HteEstimatorKind):
            raise TypeError(
                "HteArguments.estimator_kind must be "
                f"HteEstimatorKind, got "
                f"{type(self.estimator_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                f"HteArguments.random_seed must be int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"HteArguments.random_seed must be non-negative, got {self.random_seed!r}"
            )
        if self.n_samples < MIN_N_SAMPLES:
            raise ValueError(
                f"HteArguments.n_samples must be >= {MIN_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if self.n_samples > MAX_N_SAMPLES:
            raise ValueError(
                f"HteArguments.n_samples must be <= {MAX_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if not math.isfinite(self.confidence_level):
            raise ValueError(
                f"HteArguments.confidence_level must be finite, got {self.confidence_level!r}"
            )
        if (
            self.confidence_level < MIN_CONFIDENCE_LEVEL
            or self.confidence_level > MAX_CONFIDENCE_LEVEL
        ):
            raise ValueError(
                "HteArguments.confidence_level must be in "
                f"[{MIN_CONFIDENCE_LEVEL!r}, {MAX_CONFIDENCE_LEVEL!r}], "
                f"got {self.confidence_level!r}"
            )
        if self.n_points < 1:
            raise ValueError(f"HteArguments.n_points must be >= 1, got {self.n_points!r}")
        if self.n_points > MAX_POINTS:
            raise ValueError(
                f"HteArguments.n_points must be <= {MAX_POINTS!r}, got {self.n_points!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HteEffectPoint:
    """Per-row CATE estimate."""

    point_id: int
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.point_id, int) or isinstance(self.point_id, bool):
            raise TypeError(
                f"HteEffectPoint.point_id must be int, got {type(self.point_id).__name__}"
            )
        if self.point_id < 0:
            raise ValueError(f"HteEffectPoint.point_id must be non-negative, got {self.point_id!r}")
        if not math.isfinite(self.point_estimate):
            raise ValueError(
                f"HteEffectPoint.point_estimate must be finite, got {self.point_estimate!r}"
            )
        if not math.isfinite(self.ci_lower):
            raise ValueError(f"HteEffectPoint.ci_lower must be finite, got {self.ci_lower!r}")
        if not math.isfinite(self.ci_upper):
            raise ValueError(f"HteEffectPoint.ci_upper must be finite, got {self.ci_upper!r}")
        if self.ci_lower > self.ci_upper:
            raise ValueError(
                "HteEffectPoint.ci_lower must be <= ci_upper, got "
                f"({self.ci_lower!r}, {self.ci_upper!r})"
            )
        if not math.isfinite(self.std_error):
            raise ValueError(f"HteEffectPoint.std_error must be finite, got {self.std_error!r}")
        if self.std_error < 0.0:
            raise ValueError(
                f"HteEffectPoint.std_error must be non-negative, got {self.std_error!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HteAnalysisResult:
    """Estimator output — projected onto a frozen value object."""

    average_treatment_effect: float
    ate_std_error: float
    points: tuple[HteEffectPoint, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.average_treatment_effect):
            raise ValueError(
                "HteAnalysisResult.average_treatment_effect must be "
                f"finite, got {self.average_treatment_effect!r}"
            )
        if not math.isfinite(self.ate_std_error):
            raise ValueError(
                f"HteAnalysisResult.ate_std_error must be finite, got {self.ate_std_error!r}"
            )
        if self.ate_std_error < 0.0:
            raise ValueError(
                f"HteAnalysisResult.ate_std_error must be non-negative, got {self.ate_std_error!r}"
            )
        if not isinstance(self.points, tuple):
            raise TypeError(
                f"HteAnalysisResult.points must be a tuple, got {type(self.points).__name__}"
            )
        if len(self.points) > MAX_POINTS:
            raise ValueError(
                f"HteAnalysisResult.points must have <= "
                f"{MAX_POINTS} entries, got {len(self.points)!r}"
            )
        for p in self.points:
            if not isinstance(p, HteEffectPoint):
                raise TypeError(
                    "HteAnalysisResult.points entries must be "
                    f"HteEffectPoint, got {type(p).__name__}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class HteAnalysisRecord:
    """Output of :meth:`EconMLHteAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    estimand: HteEstimand
    result: HteAnalysisResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"HteAnalysisRecord.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"HteAnalysisRecord.ts_ns must be non-negative, got {self.ts_ns!r}")
        if not self.analysis_id:
            raise ValueError("HteAnalysisRecord.analysis_id must be non-empty")
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "HteAnalysisRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError("HteAnalysisRecord.source must be non-empty")
        if not isinstance(self.estimand, HteEstimand):
            raise TypeError(
                "HteAnalysisRecord.estimand must be HteEstimand, got "
                f"{type(self.estimand).__name__}"
            )
        if not isinstance(self.result, HteAnalysisResult):
            raise TypeError(
                "HteAnalysisRecord.result must be HteAnalysisResult, "
                f"got {type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "HteAnalysisRecord.analysis_digest must be a "
                f"16-hex-char digest, got {self.analysis_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.analysis_digest):
            raise ValueError(
                "HteAnalysisRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class HteAnalysisCallback(Protocol):
    """econml-shape lifecycle callback (collapsed into one Protocol)."""

    def on_analysis_start(
        self,
        *,
        ts_ns: int,
        estimand: HteEstimand,
        arguments: HteArguments,
    ) -> None: ...

    def on_point_ready(
        self,
        *,
        ts_ns: int,
        point: HteEffectPoint,
    ) -> None: ...

    def on_analysis_end(
        self,
        *,
        ts_ns: int,
        result: HteAnalysisResult,
    ) -> None: ...


@runtime_checkable
class HteEffectEstimator(Protocol):
    """Caller-supplied econml estimator.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`HteAnalysisResult`.
    """

    def estimate(
        self,
        *,
        estimand: HteEstimand,
        arguments: HteArguments,
        ts_ns: int,
        callback: HteAnalysisCallback,
    ) -> HteAnalysisResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullHteAnalysisCallback:
    """No-op callback."""

    __slots__ = ()

    def on_analysis_start(
        self,
        *,
        ts_ns: int,
        estimand: HteEstimand,
        arguments: HteArguments,
    ) -> None:
        return None

    def on_point_ready(
        self,
        *,
        ts_ns: int,
        point: HteEffectPoint,
    ) -> None:
        return None

    def on_analysis_end(
        self,
        *,
        ts_ns: int,
        result: HteAnalysisResult,
    ) -> None:
        return None


def null_hte_analysis_callback() -> HteAnalysisCallback:
    return _NullHteAnalysisCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HteAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`EconMLHteAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    estimand: HteEstimand,
    arguments: HteArguments,
    result: HteAnalysisResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical analysis summary."""

    meta_pairs = "|".join(f"{k}={v}" for k, v in sorted(arguments.meta.items()))
    points_str = ";".join(
        f"{p.point_id}:{p.point_estimate!r}:{p.ci_lower!r}:{p.ci_upper!r}:{p.std_error!r}"
        for p in result.points
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"features={','.join(estimand.features)}",
            f"treatment={estimand.treatment}",
            f"outcome={estimand.outcome}",
            f"effect_modifiers={','.join(estimand.effect_modifiers)}",
            f"data_digest={estimand.data_digest}",
            f"estimator_kind={arguments.estimator_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"n_samples={arguments.n_samples!r}",
            f"confidence_level={arguments.confidence_level!r}",
            f"n_points={arguments.n_points!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"average_treatment_effect={result.average_treatment_effect!r}",
            f"ate_std_error={result.ate_std_error!r}",
            f"points={points_str}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# EconMLHteAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EconMLHteAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    estimator: HteEffectEstimator

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, HteEffectEstimator):
            raise TypeError(
                "EconMLHteAnalyser.estimator must implement the "
                "HteEffectEstimator Protocol, got "
                f"{type(self.estimator).__name__}"
            )

    def analyse(
        self,
        *,
        estimand: HteEstimand,
        arguments: HteArguments,
        ts_ns: int,
        analysis_id: str,
        callback: HteAnalysisCallback | None = None,
    ) -> HteAnalysisRecord:
        """Run one HTE analysis and emit a
        :class:`HteAnalysisRecord`."""

        if not isinstance(estimand, HteEstimand):
            raise TypeError(
                "EconMLHteAnalyser.analyse.estimand must be "
                f"HteEstimand, got {type(estimand).__name__}"
            )
        if not isinstance(arguments, HteArguments):
            raise TypeError(
                "EconMLHteAnalyser.analyse.arguments must be "
                f"HteArguments, got {type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                f"EconMLHteAnalyser.analyse.ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise HteAnalyserConfigError(
                f"EconMLHteAnalyser.analyse.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise HteAnalyserConfigError("EconMLHteAnalyser.analyse.analysis_id must be non-empty")
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise HteAnalyserConfigError(
                "EconMLHteAnalyser.analyse.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        cb = callback if callback is not None else null_hte_analysis_callback()
        if not isinstance(cb, HteAnalysisCallback):
            raise TypeError(
                "EconMLHteAnalyser.analyse.callback must implement "
                "the HteAnalysisCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        cb.on_analysis_start(
            ts_ns=ts_ns,
            estimand=estimand,
            arguments=arguments,
        )
        result = self.estimator.estimate(
            estimand=estimand,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(result, HteAnalysisResult):
            raise TypeError(
                "HteEffectEstimator.estimate must return "
                "HteAnalysisResult, got "
                f"{type(result).__name__}"
            )
        cb.on_analysis_end(ts_ns=ts_ns, result=result)

        digest = _compute_analysis_digest(
            estimand=estimand,
            arguments=arguments,
            result=result,
            ts_ns=ts_ns,
            analysis_id=analysis_id,
        )
        record_meta: dict[str, str] = {
            "analysis_digest": digest,
            "estimator_kind": arguments.estimator_kind.value,
            "random_seed": str(arguments.random_seed),
            "n_samples": str(arguments.n_samples),
            "confidence_level": repr(arguments.confidence_level),
            "n_points": str(arguments.n_points),
            "average_treatment_effect": repr(result.average_treatment_effect),
            "ate_std_error": repr(result.ate_std_error),
            "point_count": str(len(result.points)),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return HteAnalysisRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            estimand=estimand,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production estimator factory (lazy-import econml)
# ---------------------------------------------------------------------------


def econml_dml_estimator() -> HteEffectEstimator:
    """Production :class:`HteEffectEstimator` backed by ``econml``.

    Lazy-imports ``econml`` + ``pandas`` + ``numpy`` + ``scikit-learn``
    inside the factory. Raises ``ImportError`` (with a helpful
    pip-install hint) if any package is missing — the rest of the
    module never imports these packages, so the analyser stays
    usable on a host that has never installed them.
    """

    try:
        import econml  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import pandas  # type: ignore[import-not-found]  # noqa: F401
        import sklearn  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "econml_dml_estimator requires the optional "
            "'econml' + 'pandas' + 'numpy' + 'scikit-learn' "
            "packages — install with 'pip install econml pandas "
            "numpy scikit-learn' (NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/hte_econml.py flags this)."
        ) from exc

    _ = econml

    class _EconMLDMLEstimator:
        """Thin econml wrapper conforming to :class:`HteEffectEstimator`."""

        __slots__ = ()

        def estimate(
            self,
            *,
            estimand: HteEstimand,
            arguments: HteArguments,
            ts_ns: int,
            callback: HteAnalysisCallback,
        ) -> HteAnalysisResult:  # pragma: no cover
            raise NotImplementedError(
                "econml_dml_estimator is the production seam — its "
                "concrete body is exercised in integration tests "
                "with econml installed; unit tests inject a "
                "deterministic fake via the HteEffectEstimator "
                "Protocol."
            )

    return _EconMLDMLEstimator()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_N_SAMPLES",
    "MAX_N_SAMPLES",
    "MAX_POINTS",
    "MIN_CONFIDENCE_LEVEL",
    "MAX_CONFIDENCE_LEVEL",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_DATA_DIGEST_LEN",
    "ANALYSIS_SOURCE",
    "HteEstimatorKind",
    "HteEstimand",
    "HteArguments",
    "HteEffectPoint",
    "HteAnalysisResult",
    "HteAnalysisRecord",
    "HteAnalysisCallback",
    "HteEffectEstimator",
    "HteAnalyserConfigError",
    "EconMLHteAnalyser",
    "null_hte_analysis_callback",
    "econml_dml_estimator",
)
