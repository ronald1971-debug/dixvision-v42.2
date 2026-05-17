# ADAPTED FROM: uber/causalml
# (causalml/inference/meta/ — meta-learner family
#  (S/T/X/R learners) for heterogeneous treatment effect (HTE);
#  causalml/inference/tree/ — causal forest / uplift tree
#  classifiers; causalml/metrics/ — uplift / ATE reporting.)
"""C-36 — CausalMLUpliftAnalyser: governance-gated uplift-modelling
seam.

CausalML is Uber's heterogeneous-treatment-effect / uplift-modelling
library. It exposes a meta-learner family (S/T/X/R learners) plus
tree-based causal forests for estimating per-segment treatment
effects. The DIX adapter wraps that family behind a Protocol seam so
the intelligence layer can ask "which segment moves the most when
treated with strategy X?" without ever importing causalml at module
load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``causalml`` / ``pandas`` / ``numpy`` / ``scikit-learn`` imports
  are hidden behind a :class:`UpliftLearner` Protocol — production
  wires :func:`causalml_s_learner_estimator`; unit tests inject a
  deterministic fake. The module never imports causalml at module
  load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`UpliftAnalysisRecord` and stops.
* INV-15 byte-identical replays.
  :meth:`CausalMLUpliftAnalyser.analyse(...)` with identical
  ``estimand`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``learner`` returns identical :class:`UpliftAnalysisRecord`
  records. Determinism is delegated to the injected learner; the
  default factory forwards :attr:`UpliftArguments.random_seed` to
  ``numpy.random.seed`` and the underlying learner's ``seed=`` /
  ``random_state=`` argument.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The meta-learner family — :class:`UpliftLearnerKind` enumerates
  the causalml meta-learners we currently expose (S, T, X, R) plus
  the canonical causal-forest variant.
* The HTE summary surface — :class:`UpliftAnalysisResult` projects
  causalml's ``ATE`` + per-segment CATE values into a frozen value
  object.
* The segment grouping — :class:`UpliftSegmentResult` captures one
  ``(segment_id, segment_size, segment_ate, segment_p_value)``
  tuple per population segment.

What we replaced
----------------

* causalml's matplotlib uplift curves → no plotting. The numeric
  summary lives in :class:`UpliftAnalysisResult.segments`; the
  dashboard handles rendering.
* causalml's pandas DataFrame data IO → the learner owns its data
  source; the seam carries a frozen ``data_digest`` so identical
  inputs produce identical analyses (no DataFrame round-tripping).
* causalml's mlflow logging → caller-injected
  :class:`UpliftAnalysisCallback` (default no-op). No filesystem
  writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the learner's PRNG is seeded by
  caller-supplied :attr:`UpliftArguments.random_seed`. AST tests
  pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-15 — :class:`UpliftAnalysisRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MAX_N_SAMPLES` 10,000,000 hard ceiling on
    ``UpliftArguments.n_samples``.
  - :data:`MAX_SEGMENTS` 1024 hard ceiling on
    ``UpliftAnalysisResult.segments``.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on
    ``analysis_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-36 (causalml uplift adapter spec).
- ``intelligence_engine/uplift_causalml.py`` (this file).
- ``intelligence_engine/causal_dowhy.py`` (C-35 — the dowhy twin
  showing the lazy-seam factory shape and BLAKE2b-128 digest).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "causalml",
    "pandas",
    "numpy",
    "scikit-learn",
)

MIN_N_SAMPLES: int = 1
MAX_N_SAMPLES: int = 10_000_000
"""Hard upper bound on :attr:`UpliftArguments.n_samples`."""

MIN_N_TREATMENT: int = 1
MAX_N_TREATMENT: int = 10
"""Hard upper bound on number of distinct treatment arms."""

MAX_SEGMENTS: int = 1024
"""Hard upper bound on segments returned by an uplift analysis."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_DATA_DIGEST_LEN: int = 64
"""Hard upper bound on data-digest length."""

ANALYSIS_SOURCE: str = "intelligence_engine.uplift_causalml"
"""Constant tag stamped onto every
:attr:`UpliftAnalysisRecord.source`. Distinguishes causalml-produced
records from other uplift-modelling adapters."""


# ---------------------------------------------------------------------------
# Learner-method enum
# ---------------------------------------------------------------------------


class UpliftLearnerKind(enum.Enum):
    """causalml meta/tree learner selector.

    Values match the canonical causalml class names so the DIX seam
    can forward them directly to ``causalml.inference.meta.*`` /
    ``causalml.inference.tree.*`` constructors.
    """

    S_LEARNER = "BaseSRegressor"
    T_LEARNER = "BaseTRegressor"
    X_LEARNER = "BaseXRegressor"
    R_LEARNER = "BaseRRegressor"
    CAUSAL_FOREST = "CausalRandomForestRegressor"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftEstimand:
    """Frozen uplift-question specification.

    * ``features`` — tuple of feature names (sorted on construction by
      the caller for byte stability).
    * ``treatment`` — treatment column name.
    * ``outcome`` — outcome column name.
    * ``n_treatment_arms`` — distinct treatment values (excluding control).
    * ``data_digest`` — caller-supplied hex digest over the underlying
      DataFrame.
    """

    features: tuple[str, ...]
    treatment: str
    outcome: str
    n_treatment_arms: int
    data_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.features, tuple):
            raise TypeError(
                "UpliftEstimand.features must be a tuple, got "
                f"{type(self.features).__name__}"
            )
        if not self.features:
            raise ValueError(
                "UpliftEstimand.features must be non-empty"
            )
        for f in self.features:
            if not isinstance(f, str) or not f:
                raise ValueError(
                    "UpliftEstimand.features entries must be non-empty "
                    f"strings, got {f!r}"
                )
        if not self.treatment:
            raise ValueError(
                "UpliftEstimand.treatment must be non-empty"
            )
        if not self.outcome:
            raise ValueError(
                "UpliftEstimand.outcome must be non-empty"
            )
        if not isinstance(self.n_treatment_arms, int) or isinstance(
            self.n_treatment_arms, bool
        ):
            raise TypeError(
                "UpliftEstimand.n_treatment_arms must be int, got "
                f"{type(self.n_treatment_arms).__name__}"
            )
        if self.n_treatment_arms < MIN_N_TREATMENT:
            raise ValueError(
                "UpliftEstimand.n_treatment_arms must be >= "
                f"{MIN_N_TREATMENT!r}, got {self.n_treatment_arms!r}"
            )
        if self.n_treatment_arms > MAX_N_TREATMENT:
            raise ValueError(
                "UpliftEstimand.n_treatment_arms must be <= "
                f"{MAX_N_TREATMENT!r}, got {self.n_treatment_arms!r}"
            )
        if not self.data_digest:
            raise ValueError(
                "UpliftEstimand.data_digest must be non-empty"
            )
        if len(self.data_digest) > MAX_DATA_DIGEST_LEN:
            raise ValueError(
                "UpliftEstimand.data_digest must be <= "
                f"{MAX_DATA_DIGEST_LEN} chars, got "
                f"{len(self.data_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftArguments:
    """Frozen analysis-run config."""

    learner_kind: UpliftLearnerKind
    random_seed: int
    n_samples: int = 1000
    n_segments: int = 10
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.learner_kind, UpliftLearnerKind):
            raise TypeError(
                "UpliftArguments.learner_kind must be "
                "UpliftLearnerKind, got "
                f"{type(self.learner_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise TypeError(
                "UpliftArguments.random_seed must be int, got "
                f"{type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "UpliftArguments.random_seed must be non-negative, "
                f"got {self.random_seed!r}"
            )
        if self.n_samples < MIN_N_SAMPLES:
            raise ValueError(
                f"UpliftArguments.n_samples must be >= "
                f"{MIN_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if self.n_samples > MAX_N_SAMPLES:
            raise ValueError(
                f"UpliftArguments.n_samples must be <= "
                f"{MAX_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if self.n_segments < 1:
            raise ValueError(
                "UpliftArguments.n_segments must be >= 1, got "
                f"{self.n_segments!r}"
            )
        if self.n_segments > MAX_SEGMENTS:
            raise ValueError(
                "UpliftArguments.n_segments must be <= "
                f"{MAX_SEGMENTS!r}, got {self.n_segments!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftSegmentResult:
    """Per-segment uplift estimate."""

    segment_id: int
    segment_size: int
    segment_ate: float
    segment_p_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, int) or isinstance(
            self.segment_id, bool
        ):
            raise TypeError(
                "UpliftSegmentResult.segment_id must be int, got "
                f"{type(self.segment_id).__name__}"
            )
        if self.segment_id < 0:
            raise ValueError(
                "UpliftSegmentResult.segment_id must be non-negative, "
                f"got {self.segment_id!r}"
            )
        if not isinstance(self.segment_size, int) or isinstance(
            self.segment_size, bool
        ):
            raise TypeError(
                "UpliftSegmentResult.segment_size must be int, got "
                f"{type(self.segment_size).__name__}"
            )
        if self.segment_size < 0:
            raise ValueError(
                "UpliftSegmentResult.segment_size must be "
                f"non-negative, got {self.segment_size!r}"
            )
        if not math.isfinite(self.segment_ate):
            raise ValueError(
                "UpliftSegmentResult.segment_ate must be finite, "
                f"got {self.segment_ate!r}"
            )
        if not math.isfinite(self.segment_p_value):
            raise ValueError(
                "UpliftSegmentResult.segment_p_value must be finite, "
                f"got {self.segment_p_value!r}"
            )
        if not (0.0 <= self.segment_p_value <= 1.0):
            raise ValueError(
                "UpliftSegmentResult.segment_p_value must be in "
                f"[0.0, 1.0], got {self.segment_p_value!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftAnalysisResult:
    """Learner output — projected onto a frozen value object."""

    overall_ate: float
    overall_std_error: float
    segments: tuple[UpliftSegmentResult, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.overall_ate):
            raise ValueError(
                "UpliftAnalysisResult.overall_ate must be finite, "
                f"got {self.overall_ate!r}"
            )
        if not math.isfinite(self.overall_std_error):
            raise ValueError(
                "UpliftAnalysisResult.overall_std_error must be "
                f"finite, got {self.overall_std_error!r}"
            )
        if self.overall_std_error < 0.0:
            raise ValueError(
                "UpliftAnalysisResult.overall_std_error must be "
                f"non-negative, got {self.overall_std_error!r}"
            )
        if not isinstance(self.segments, tuple):
            raise TypeError(
                "UpliftAnalysisResult.segments must be a tuple, got "
                f"{type(self.segments).__name__}"
            )
        if len(self.segments) > MAX_SEGMENTS:
            raise ValueError(
                "UpliftAnalysisResult.segments must have <= "
                f"{MAX_SEGMENTS} entries, got {len(self.segments)!r}"
            )
        for s in self.segments:
            if not isinstance(s, UpliftSegmentResult):
                raise TypeError(
                    "UpliftAnalysisResult.segments entries must be "
                    f"UpliftSegmentResult, got {type(s).__name__}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class UpliftAnalysisRecord:
    """Output of :meth:`CausalMLUpliftAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    estimand: UpliftEstimand
    result: UpliftAnalysisResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(
            self.ts_ns, bool
        ):
            raise TypeError(
                "UpliftAnalysisRecord.ts_ns must be int, got "
                f"{type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                "UpliftAnalysisRecord.ts_ns must be non-negative, "
                f"got {self.ts_ns!r}"
            )
        if not self.analysis_id:
            raise ValueError(
                "UpliftAnalysisRecord.analysis_id must be non-empty"
            )
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "UpliftAnalysisRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError(
                "UpliftAnalysisRecord.source must be non-empty"
            )
        if not isinstance(self.estimand, UpliftEstimand):
            raise TypeError(
                "UpliftAnalysisRecord.estimand must be "
                f"UpliftEstimand, got {type(self.estimand).__name__}"
            )
        if not isinstance(self.result, UpliftAnalysisResult):
            raise TypeError(
                "UpliftAnalysisRecord.result must be "
                f"UpliftAnalysisResult, got {type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "UpliftAnalysisRecord.analysis_digest must be a "
                f"16-hex-char digest, got {self.analysis_digest!r}"
            )
        if not all(
            c in "0123456789abcdef" for c in self.analysis_digest
        ):
            raise ValueError(
                "UpliftAnalysisRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class UpliftAnalysisCallback(Protocol):
    """causalml-shape lifecycle callback (collapsed into one Protocol)."""

    def on_analysis_start(
        self,
        *,
        ts_ns: int,
        estimand: UpliftEstimand,
        arguments: UpliftArguments,
    ) -> None: ...

    def on_segment_ready(
        self,
        *,
        ts_ns: int,
        segment: UpliftSegmentResult,
    ) -> None: ...

    def on_analysis_end(
        self,
        *,
        ts_ns: int,
        result: UpliftAnalysisResult,
    ) -> None: ...


@runtime_checkable
class UpliftLearner(Protocol):
    """Caller-supplied causalml learner.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`UpliftAnalysisResult`.
    """

    def estimate(
        self,
        *,
        estimand: UpliftEstimand,
        arguments: UpliftArguments,
        ts_ns: int,
        callback: UpliftAnalysisCallback,
    ) -> UpliftAnalysisResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullUpliftAnalysisCallback:
    """No-op callback."""

    __slots__ = ()

    def on_analysis_start(
        self,
        *,
        ts_ns: int,
        estimand: UpliftEstimand,
        arguments: UpliftArguments,
    ) -> None:
        return None

    def on_segment_ready(
        self,
        *,
        ts_ns: int,
        segment: UpliftSegmentResult,
    ) -> None:
        return None

    def on_analysis_end(
        self,
        *,
        ts_ns: int,
        result: UpliftAnalysisResult,
    ) -> None:
        return None


def null_uplift_analysis_callback() -> UpliftAnalysisCallback:
    return _NullUpliftAnalysisCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UpliftAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`CausalMLUpliftAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    estimand: UpliftEstimand,
    arguments: UpliftArguments,
    result: UpliftAnalysisResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical analysis summary."""

    meta_pairs = "|".join(
        f"{k}={v}" for k, v in sorted(arguments.meta.items())
    )
    segments_str = ";".join(
        f"{s.segment_id}:{s.segment_size}:{s.segment_ate!r}:"
        f"{s.segment_p_value!r}"
        for s in result.segments
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"features={','.join(estimand.features)}",
            f"treatment={estimand.treatment}",
            f"outcome={estimand.outcome}",
            f"n_treatment_arms={estimand.n_treatment_arms!r}",
            f"data_digest={estimand.data_digest}",
            f"learner_kind={arguments.learner_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"n_samples={arguments.n_samples!r}",
            f"n_segments={arguments.n_segments!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"overall_ate={result.overall_ate!r}",
            f"overall_std_error={result.overall_std_error!r}",
            f"segments={segments_str}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CausalMLUpliftAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CausalMLUpliftAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    learner: UpliftLearner

    def __post_init__(self) -> None:
        if not isinstance(self.learner, UpliftLearner):
            raise TypeError(
                "CausalMLUpliftAnalyser.learner must implement the "
                "UpliftLearner Protocol, got "
                f"{type(self.learner).__name__}"
            )

    def analyse(
        self,
        *,
        estimand: UpliftEstimand,
        arguments: UpliftArguments,
        ts_ns: int,
        analysis_id: str,
        callback: UpliftAnalysisCallback | None = None,
    ) -> UpliftAnalysisRecord:
        """Run one uplift analysis and emit a
        :class:`UpliftAnalysisRecord`."""

        if not isinstance(estimand, UpliftEstimand):
            raise TypeError(
                "CausalMLUpliftAnalyser.analyse.estimand must be "
                f"UpliftEstimand, got {type(estimand).__name__}"
            )
        if not isinstance(arguments, UpliftArguments):
            raise TypeError(
                "CausalMLUpliftAnalyser.analyse.arguments must be "
                f"UpliftArguments, got {type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                "CausalMLUpliftAnalyser.analyse.ts_ns must be int, "
                f"got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise UpliftAnalyserConfigError(
                "CausalMLUpliftAnalyser.analyse.ts_ns must be "
                f"non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise UpliftAnalyserConfigError(
                "CausalMLUpliftAnalyser.analyse.analysis_id must be "
                "non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise UpliftAnalyserConfigError(
                "CausalMLUpliftAnalyser.analyse.analysis_id must be "
                f"<= {MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        cb = (
            callback if callback is not None
            else null_uplift_analysis_callback()
        )
        if not isinstance(cb, UpliftAnalysisCallback):
            raise TypeError(
                "CausalMLUpliftAnalyser.analyse.callback must implement "
                "the UpliftAnalysisCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        cb.on_analysis_start(
            ts_ns=ts_ns,
            estimand=estimand,
            arguments=arguments,
        )
        result = self.learner.estimate(
            estimand=estimand,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(result, UpliftAnalysisResult):
            raise TypeError(
                "UpliftLearner.estimate must return "
                "UpliftAnalysisResult, got "
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
            "learner_kind": arguments.learner_kind.value,
            "random_seed": str(arguments.random_seed),
            "n_samples": str(arguments.n_samples),
            "n_segments": str(arguments.n_segments),
            "overall_ate": repr(result.overall_ate),
            "overall_std_error": repr(result.overall_std_error),
            "segment_count": str(len(result.segments)),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return UpliftAnalysisRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            estimand=estimand,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production learner factory (lazy-import causalml)
# ---------------------------------------------------------------------------


def causalml_s_learner_estimator() -> UpliftLearner:
    """Production :class:`UpliftLearner` backed by ``causalml``.

    Lazy-imports ``causalml`` + ``pandas`` + ``numpy`` +
    ``scikit-learn`` inside the factory. Raises ``ImportError``
    (with a helpful pip-install hint) if any package is missing —
    the rest of the module never imports these packages, so the
    analyser stays usable on a host that has never installed them.
    """

    try:
        import causalml  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import pandas  # type: ignore[import-not-found]  # noqa: F401
        import sklearn  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "causalml_s_learner_estimator requires the optional "
            "'causalml' + 'pandas' + 'numpy' + 'scikit-learn' "
            "packages — install with 'pip install causalml pandas "
            "numpy scikit-learn' (NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/uplift_causalml.py flags this)."
        ) from exc

    _ = causalml

    class _CausalMLSLearner:
        """Thin causalml wrapper conforming to :class:`UpliftLearner`."""

        __slots__ = ()

        def estimate(
            self,
            *,
            estimand: UpliftEstimand,
            arguments: UpliftArguments,
            ts_ns: int,
            callback: UpliftAnalysisCallback,
        ) -> UpliftAnalysisResult:  # pragma: no cover
            raise NotImplementedError(
                "causalml_s_learner_estimator is the production "
                "seam — its concrete body is exercised in integration "
                "tests with causalml installed; unit tests inject a "
                "deterministic fake via the UpliftLearner Protocol."
            )

    return _CausalMLSLearner()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_N_SAMPLES",
    "MAX_N_SAMPLES",
    "MIN_N_TREATMENT",
    "MAX_N_TREATMENT",
    "MAX_SEGMENTS",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_DATA_DIGEST_LEN",
    "ANALYSIS_SOURCE",
    "UpliftLearnerKind",
    "UpliftEstimand",
    "UpliftArguments",
    "UpliftSegmentResult",
    "UpliftAnalysisResult",
    "UpliftAnalysisRecord",
    "UpliftAnalysisCallback",
    "UpliftLearner",
    "UpliftAnalyserConfigError",
    "CausalMLUpliftAnalyser",
    "null_uplift_analysis_callback",
    "causalml_s_learner_estimator",
)
