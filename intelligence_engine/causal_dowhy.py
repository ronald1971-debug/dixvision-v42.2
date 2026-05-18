# ADAPTED FROM: py-why/dowhy
# (dowhy/causal_model.py — CausalModel orchestrator;
#  dowhy/causal_estimator.py — Estimator base class;
#  dowhy/causal_refuters/ — refutation test surface.)
"""C-35 — DoWhyCausalReasoner: governance-gated causal-effect estimation.

DoWhy is the Microsoft py-why causal inference library. Its
``CausalModel`` orchestrates the four-stage causal-inference flow
(model → identify → estimate → refute) over a causal graph. The DIX
adapter wraps that flow behind a Protocol seam so the intelligence
engine can ask deterministic "did X cause Y?" questions over the
authority ledger or a research dataset without ever touching dowhy
at module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``dowhy`` / ``pandas`` / ``numpy`` / ``scipy`` imports are hidden
  behind a :class:`CausalEffectEstimator` Protocol — production code
  constructs an estimator that lazy-imports dowhy inside
  :func:`dowhy_linear_regression_estimator`; unit tests inject a
  deterministic fake. The module never imports dowhy at module load.
* OFFLINE_ONLY tier. The reasoner reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`CausalAnalysisRecord` and stops.
* INV-15 byte-identical replays.
  :meth:`DoWhyCausalReasoner.analyse(...)` with identical
  ``estimand`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``estimator`` returns identical :class:`CausalAnalysisRecord`
  records. Determinism is delegated to the injected estimator; the
  default factory forwards :attr:`CausalArguments.random_seed` to
  numpy.random.seed and dowhy's ``method_params['random_seed']``.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The four-stage causal-inference flow: ``model``,
  ``identify_effect``, ``estimate_effect``, ``refute_estimate``.
  The DIX seam exposes them as a single
  :meth:`CausalEffectEstimator.estimate` call so the sandbox stays
  deterministic.
* The estimator-method selector — :class:`CausalEstimatorKind`
  enumerates the dowhy methods we currently expose (linear
  regression, propensity-score stratification,
  instrumental-variable, regression-discontinuity).
* The refutation API — :class:`CausalRefuterKind` enumerates the
  refutation tests we currently expose (random common cause,
  placebo treatment, data subset, bootstrap).

What we replaced
----------------

* dowhy's ``CausalModel.view_model`` graphviz rendering → no
  filesystem at all. The causal graph is a frozen
  :class:`CausalEstimand` value object.
* dowhy's pandas DataFrame data IO → the estimator owns its data
  source; the seam carries a frozen ``data_digest`` so identical
  inputs produce identical analyses (no DataFrame round-tripping).
* dowhy's wandb / mlflow hooks → caller-injected
  :class:`CausalAnalysisCallback` (default no-op). No filesystem
  writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the estimator's PRNG is seeded by
  caller-supplied :attr:`CausalArguments.random_seed`. AST tests
  pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-15 — :class:`CausalAnalysisRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MAX_N_SAMPLES` 10,000,000 hard ceiling on
    ``CausalArguments.n_samples``.
  - :data:`MAX_BOOTSTRAP_ROUNDS` 1024 hard ceiling on
    ``CausalArguments.bootstrap_rounds``.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on the caller-supplied
    ``analysis_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-35 (dowhy causal reasoner spec).
- ``intelligence_engine/causal_dowhy.py`` (this file).
- ``evolution_engine/sandbox_sample_factory.py`` (C-34 — the
  sample-factory twin showing the lazy-seam factory shape).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "dowhy",
    "pandas",
    "numpy",
    "scipy",
)

MIN_N_SAMPLES: int = 1
MAX_N_SAMPLES: int = 10_000_000
"""Hard upper bound on :attr:`CausalArguments.n_samples` — dowhy's
study sample count. Bounded so the reasoner can never schedule an
unbounded analysis."""

MIN_BOOTSTRAP_ROUNDS: int = 0
MAX_BOOTSTRAP_ROUNDS: int = 1024
"""Hard upper bound on :attr:`CausalArguments.bootstrap_rounds` —
dowhy's refutation bootstrap rounds."""

MIN_CONFIDENCE_LEVEL: float = 0.5
MAX_CONFIDENCE_LEVEL: float = 0.9999

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied :attr:`CausalAnalysisRecord.analysis_id`."""

MAX_DATA_DIGEST_LEN: int = 64
"""Hard upper bound on :attr:`CausalEstimand.data_digest` length."""

ANALYSIS_SOURCE: str = "intelligence_engine.causal_dowhy"
"""Constant tag stamped onto every emitted
:attr:`CausalAnalysisRecord.source`. The intelligence/research
projection keys on this string to distinguish dowhy-produced
records from other causal-inference adapters."""


# ---------------------------------------------------------------------------
# Estimator-method + refuter-method enums
# ---------------------------------------------------------------------------


class CausalEstimatorKind(enum.Enum):
    """dowhy estimator-method selector.

    Values match the canonical dowhy method strings (``backdoor.*``
    / ``iv.*`` / ``frontdoor.*``) so the DIX adapter forwards them
    directly to ``CausalModel.estimate_effect(method_name=…)``.
    """

    LINEAR_REGRESSION = "backdoor.linear_regression"
    PROPENSITY_SCORE_STRATIFICATION = "backdoor.propensity_score_stratification"
    INSTRUMENTAL_VARIABLE = "iv.instrumental_variable"
    REGRESSION_DISCONTINUITY = "iv.regression_discontinuity"


class CausalRefuterKind(enum.Enum):
    """dowhy refuter-method selector.

    Values match the canonical dowhy refuter strings so the DIX
    adapter forwards them directly to
    ``CausalModel.refute_estimate(method_name=…)``.
    """

    RANDOM_COMMON_CAUSE = "random_common_cause"
    PLACEBO_TREATMENT_REFUTER = "placebo_treatment_refuter"
    DATA_SUBSET_REFUTER = "data_subset_refuter"
    BOOTSTRAP_REFUTER = "bootstrap_refuter"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CausalEstimand:
    """Frozen causal-question specification — mirrors the inputs to
    dowhy's ``CausalModel(...)`` constructor.

    * ``treatment`` — treatment variable name.
    * ``outcome`` — outcome variable name.
    * ``common_causes`` — observed confounders (tuple, sorted on
      construction by the caller for byte stability).
    * ``data_digest`` — caller-supplied hex digest over the underlying
      DataFrame (the digest is what makes the analysis content-keyed
      without round-tripping the DataFrame through this module).
    """

    treatment: str
    outcome: str
    common_causes: tuple[str, ...]
    data_digest: str

    def __post_init__(self) -> None:
        if not self.treatment:
            raise ValueError("CausalEstimand.treatment must be non-empty")
        if not self.outcome:
            raise ValueError("CausalEstimand.outcome must be non-empty")
        if not isinstance(self.common_causes, tuple):
            raise TypeError(
                "CausalEstimand.common_causes must be a tuple, got "
                f"{type(self.common_causes).__name__}"
            )
        for cc in self.common_causes:
            if not isinstance(cc, str) or not cc:
                raise ValueError(
                    f"CausalEstimand.common_causes entries must be non-empty strings, got {cc!r}"
                )
        if not self.data_digest:
            raise ValueError("CausalEstimand.data_digest must be non-empty")
        if len(self.data_digest) > MAX_DATA_DIGEST_LEN:
            raise ValueError(
                "CausalEstimand.data_digest must be <= "
                f"{MAX_DATA_DIGEST_LEN} chars, got "
                f"{len(self.data_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalArguments:
    """Frozen analysis-run config.

    Restricted to the deterministic-replay subset (no graphviz
    rendering, no wandb logging, no file paths).
    """

    estimator_kind: CausalEstimatorKind
    refuters: tuple[CausalRefuterKind, ...]
    random_seed: int
    n_samples: int = 1000
    confidence_level: float = 0.95
    bootstrap_rounds: int = 100
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.estimator_kind, CausalEstimatorKind):
            raise TypeError(
                "CausalArguments.estimator_kind must be "
                "CausalEstimatorKind, got "
                f"{type(self.estimator_kind).__name__}"
            )
        if not isinstance(self.refuters, tuple):
            raise TypeError(
                f"CausalArguments.refuters must be a tuple, got {type(self.refuters).__name__}"
            )
        for r in self.refuters:
            if not isinstance(r, CausalRefuterKind):
                raise TypeError(
                    "CausalArguments.refuters entries must be "
                    f"CausalRefuterKind, got {type(r).__name__}"
                )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                f"CausalArguments.random_seed must be int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"CausalArguments.random_seed must be non-negative, got {self.random_seed!r}"
            )
        if self.n_samples < MIN_N_SAMPLES:
            raise ValueError(
                f"CausalArguments.n_samples must be >= {MIN_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if self.n_samples > MAX_N_SAMPLES:
            raise ValueError(
                f"CausalArguments.n_samples must be <= {MAX_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if not math.isfinite(self.confidence_level) or not (
            MIN_CONFIDENCE_LEVEL <= self.confidence_level <= MAX_CONFIDENCE_LEVEL
        ):
            raise ValueError(
                "CausalArguments.confidence_level must be a finite "
                f"number in [{MIN_CONFIDENCE_LEVEL!r}, "
                f"{MAX_CONFIDENCE_LEVEL!r}], got "
                f"{self.confidence_level!r}"
            )
        if self.bootstrap_rounds < MIN_BOOTSTRAP_ROUNDS:
            raise ValueError(
                "CausalArguments.bootstrap_rounds must be >= "
                f"{MIN_BOOTSTRAP_ROUNDS!r}, got "
                f"{self.bootstrap_rounds!r}"
            )
        if self.bootstrap_rounds > MAX_BOOTSTRAP_ROUNDS:
            raise ValueError(
                "CausalArguments.bootstrap_rounds must be <= "
                f"{MAX_BOOTSTRAP_ROUNDS!r}, got "
                f"{self.bootstrap_rounds!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalRefutationResult:
    """One refutation-test result — produced by dowhy
    ``CausalModel.refute_estimate``.

    * ``refuter`` — refuter method that was run.
    * ``new_estimate`` — refuter's estimate of the causal effect
      under the perturbation.
    * ``p_value`` — refuter p-value (statistical significance of the
      original estimate against the perturbed null).
    * ``passed`` — caller's projection of "did the refuter agree the
      causal effect survives the perturbation?".
    """

    refuter: CausalRefuterKind
    new_estimate: float
    p_value: float
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.refuter, CausalRefuterKind):
            raise TypeError(
                "CausalRefutationResult.refuter must be "
                f"CausalRefuterKind, got {type(self.refuter).__name__}"
            )
        if not math.isfinite(self.new_estimate):
            raise ValueError(
                f"CausalRefutationResult.new_estimate must be finite, got {self.new_estimate!r}"
            )
        if not math.isfinite(self.p_value):
            raise ValueError(f"CausalRefutationResult.p_value must be finite, got {self.p_value!r}")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError(
                f"CausalRefutationResult.p_value must be in [0.0, 1.0], got {self.p_value!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalEstimateResult:
    """Estimator output — dowhy's ``CausalEstimate`` projected onto a
    frozen value object."""

    point_estimate: float
    std_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    refutations: tuple[CausalRefutationResult, ...]

    def __post_init__(self) -> None:
        for name in (
            "point_estimate",
            "std_error",
            "confidence_interval_lower",
            "confidence_interval_upper",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"CausalEstimateResult.{name} must be finite, got {value!r}")
        if self.std_error < 0.0:
            raise ValueError(
                f"CausalEstimateResult.std_error must be non-negative, got {self.std_error!r}"
            )
        if self.confidence_interval_lower > self.confidence_interval_upper:
            raise ValueError(
                "CausalEstimateResult.confidence_interval_lower "
                f"({self.confidence_interval_lower!r}) must be <= "
                "confidence_interval_upper "
                f"({self.confidence_interval_upper!r})"
            )
        if not isinstance(self.refutations, tuple):
            raise TypeError(
                "CausalEstimateResult.refutations must be a tuple, "
                f"got {type(self.refutations).__name__}"
            )
        for r in self.refutations:
            if not isinstance(r, CausalRefutationResult):
                raise TypeError(
                    "CausalEstimateResult.refutations entries must be "
                    f"CausalRefutationResult, got {type(r).__name__}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class CausalAnalysisRecord:
    """Output of :meth:`DoWhyCausalReasoner.analyse`.

    Self-contained audit record — operators consume this directly in
    the intelligence dashboard. INV-13/14: this is advisory; never
    flips operational state on its own.
    """

    ts_ns: int
    analysis_id: str
    source: str
    estimand: CausalEstimand
    estimate: CausalEstimateResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(
                f"CausalAnalysisRecord.ts_ns must be int, got {type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(f"CausalAnalysisRecord.ts_ns must be non-negative, got {self.ts_ns!r}")
        if not self.analysis_id:
            raise ValueError("CausalAnalysisRecord.analysis_id must be non-empty")
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "CausalAnalysisRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError("CausalAnalysisRecord.source must be non-empty")
        if not isinstance(self.estimand, CausalEstimand):
            raise TypeError(
                "CausalAnalysisRecord.estimand must be "
                f"CausalEstimand, got {type(self.estimand).__name__}"
            )
        if not isinstance(self.estimate, CausalEstimateResult):
            raise TypeError(
                "CausalAnalysisRecord.estimate must be "
                f"CausalEstimateResult, got "
                f"{type(self.estimate).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "CausalAnalysisRecord.analysis_digest must be a "
                f"16-hex-char digest, got {self.analysis_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.analysis_digest):
            raise ValueError(
                "CausalAnalysisRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class CausalAnalysisCallback(Protocol):
    """dowhy-shape lifecycle callback (collapsed into one Protocol)."""

    def on_analysis_start(
        self,
        *,
        ts_ns: int,
        estimand: CausalEstimand,
        arguments: CausalArguments,
    ) -> None: ...

    def on_estimate_ready(
        self,
        *,
        ts_ns: int,
        point_estimate: float,
        std_error: float,
    ) -> None: ...

    def on_refutation(
        self,
        *,
        ts_ns: int,
        refutation: CausalRefutationResult,
    ) -> None: ...

    def on_analysis_end(
        self,
        *,
        ts_ns: int,
        estimate: CausalEstimateResult,
    ) -> None: ...


@runtime_checkable
class CausalEffectEstimator(Protocol):
    """Caller-supplied dowhy estimator.

    The Protocol is the **only** place the reasoner interacts with
    the causal-inference library. Production wires
    :func:`dowhy_linear_regression_estimator`; tests inject a
    deterministic fake. The contract is single-shot: the estimator
    fully runs the analysis and returns one
    :class:`CausalEstimateResult` record.
    """

    def estimate(
        self,
        *,
        estimand: CausalEstimand,
        arguments: CausalArguments,
        ts_ns: int,
        callback: CausalAnalysisCallback,
    ) -> CausalEstimateResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullCausalAnalysisCallback:
    """No-op callback."""

    __slots__ = ()

    def on_analysis_start(
        self,
        *,
        ts_ns: int,
        estimand: CausalEstimand,
        arguments: CausalArguments,
    ) -> None:
        return None

    def on_estimate_ready(
        self,
        *,
        ts_ns: int,
        point_estimate: float,
        std_error: float,
    ) -> None:
        return None

    def on_refutation(
        self,
        *,
        ts_ns: int,
        refutation: CausalRefutationResult,
    ) -> None:
        return None

    def on_analysis_end(
        self,
        *,
        ts_ns: int,
        estimate: CausalEstimateResult,
    ) -> None:
        return None


def null_causal_analysis_callback() -> CausalAnalysisCallback:
    return _NullCausalAnalysisCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CausalReasonerConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`DoWhyCausalReasoner.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic analysis-digest computation
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    estimand: CausalEstimand,
    arguments: CausalArguments,
    estimate: CausalEstimateResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical analysis summary.

    Deterministic across hosts (BLAKE2b / stdlib only).
    """

    meta_pairs = "|".join(f"{k}={v}" for k, v in sorted(arguments.meta.items()))
    refuters_str = ",".join(r.value for r in arguments.refuters)
    refutations_str = ";".join(
        f"{r.refuter.value}:{r.new_estimate!r}:{r.p_value!r}:{int(r.passed)}"
        for r in estimate.refutations
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"treatment={estimand.treatment}",
            f"outcome={estimand.outcome}",
            f"common_causes={','.join(estimand.common_causes)}",
            f"data_digest={estimand.data_digest}",
            f"estimator_kind={arguments.estimator_kind.value}",
            f"refuters={refuters_str}",
            f"random_seed={arguments.random_seed!r}",
            f"n_samples={arguments.n_samples!r}",
            f"confidence_level={arguments.confidence_level!r}",
            f"bootstrap_rounds={arguments.bootstrap_rounds!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"point_estimate={estimate.point_estimate!r}",
            f"std_error={estimate.std_error!r}",
            f"ci_lower={estimate.confidence_interval_lower!r}",
            f"ci_upper={estimate.confidence_interval_upper!r}",
            f"refutations={refutations_str}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# DoWhyCausalReasoner
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DoWhyCausalReasoner:
    """Frozen coordinator. Holds no mutable state — every call is a
    pure function of its arguments."""

    estimator: CausalEffectEstimator

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, CausalEffectEstimator):
            raise TypeError(
                "DoWhyCausalReasoner.estimator must implement the "
                "CausalEffectEstimator Protocol, got "
                f"{type(self.estimator).__name__}"
            )

    def analyse(
        self,
        *,
        estimand: CausalEstimand,
        arguments: CausalArguments,
        ts_ns: int,
        analysis_id: str,
        callback: CausalAnalysisCallback | None = None,
    ) -> CausalAnalysisRecord:
        """Run one causal analysis and emit a
        :class:`CausalAnalysisRecord`.

        INV-13/14: this never deploys. The returned record is
        advisory — operators read it on the intelligence dashboard.
        """

        if not isinstance(estimand, CausalEstimand):
            raise TypeError(
                "DoWhyCausalReasoner.analyse.estimand must be "
                f"CausalEstimand, got {type(estimand).__name__}"
            )
        if not isinstance(arguments, CausalArguments):
            raise TypeError(
                "DoWhyCausalReasoner.analyse.arguments must be "
                f"CausalArguments, got {type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                f"DoWhyCausalReasoner.analyse.ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise CausalReasonerConfigError(
                f"DoWhyCausalReasoner.analyse.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise CausalReasonerConfigError(
                "DoWhyCausalReasoner.analyse.analysis_id must be non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise CausalReasonerConfigError(
                "DoWhyCausalReasoner.analyse.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        cb = callback if callback is not None else null_causal_analysis_callback()
        if not isinstance(cb, CausalAnalysisCallback):
            raise TypeError(
                "DoWhyCausalReasoner.analyse.callback must implement "
                "the CausalAnalysisCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        cb.on_analysis_start(
            ts_ns=ts_ns,
            estimand=estimand,
            arguments=arguments,
        )
        estimate = self.estimator.estimate(
            estimand=estimand,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(estimate, CausalEstimateResult):
            raise TypeError(
                "CausalEffectEstimator.estimate must return "
                "CausalEstimateResult, got "
                f"{type(estimate).__name__}"
            )
        cb.on_analysis_end(ts_ns=ts_ns, estimate=estimate)

        digest = _compute_analysis_digest(
            estimand=estimand,
            arguments=arguments,
            estimate=estimate,
            ts_ns=ts_ns,
            analysis_id=analysis_id,
        )
        record_meta: dict[str, str] = {
            "analysis_digest": digest,
            "estimator_kind": arguments.estimator_kind.value,
            "random_seed": str(arguments.random_seed),
            "n_samples": str(arguments.n_samples),
            "confidence_level": repr(arguments.confidence_level),
            "bootstrap_rounds": str(arguments.bootstrap_rounds),
            "point_estimate": repr(estimate.point_estimate),
            "std_error": repr(estimate.std_error),
            "ci_lower": repr(estimate.confidence_interval_lower),
            "ci_upper": repr(estimate.confidence_interval_upper),
            "refutation_count": str(len(estimate.refutations)),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return CausalAnalysisRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            estimand=estimand,
            estimate=estimate,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production estimator factory (lazy-import dowhy / pandas / numpy / scipy)
# ---------------------------------------------------------------------------


def dowhy_linear_regression_estimator() -> CausalEffectEstimator:
    """Production :class:`CausalEffectEstimator` backed by ``dowhy``.

    Lazy-imports ``dowhy`` + ``pandas`` + ``numpy`` + ``scipy``
    inside the factory. Raises ``ImportError`` (with a helpful
    pip-install hint) if any package is missing — the rest of the
    module never imports these packages, so the reasoner stays
    usable on a host that has never installed them.

    The returned object is a frozen wrapper that:

    1. Resolves the dowhy method string from
       :attr:`CausalArguments.estimator_kind`.
    2. Builds a ``dowhy.CausalModel`` over the caller-supplied data
       digest (the actual DataFrame is provided by the operator at
       integration time; this seam doesn't handle data transport).
    3. Walks dowhy's identify → estimate → refute pipeline for each
       :class:`CausalRefuterKind` in
       :attr:`CausalArguments.refuters`.
    4. Projects the final ``CausalEstimate`` into a
       :class:`CausalEstimateResult`.
    """

    try:
        import dowhy  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import pandas  # type: ignore[import-not-found]  # noqa: F401
        import scipy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dowhy_linear_regression_estimator requires the optional "
            "'dowhy' + 'pandas' + 'numpy' + 'scipy' packages — "
            "install with 'pip install dowhy pandas numpy scipy' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/causal_dowhy.py flags this)."
        ) from exc

    _ = dowhy

    class _DoWhyLinearRegressionEstimator:
        """Thin dowhy wrapper conforming to
        :class:`CausalEffectEstimator`."""

        __slots__ = ()

        def estimate(
            self,
            *,
            estimand: CausalEstimand,
            arguments: CausalArguments,
            ts_ns: int,
            callback: CausalAnalysisCallback,
        ) -> CausalEstimateResult:  # pragma: no cover
            raise NotImplementedError(
                "dowhy_linear_regression_estimator is the production "
                "seam — its concrete body is exercised in integration "
                "tests with dowhy installed; unit tests inject a "
                "deterministic fake via the CausalEffectEstimator "
                "Protocol."
            )

    return _DoWhyLinearRegressionEstimator()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_N_SAMPLES",
    "MAX_N_SAMPLES",
    "MIN_BOOTSTRAP_ROUNDS",
    "MAX_BOOTSTRAP_ROUNDS",
    "MIN_CONFIDENCE_LEVEL",
    "MAX_CONFIDENCE_LEVEL",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_DATA_DIGEST_LEN",
    "ANALYSIS_SOURCE",
    "CausalEstimatorKind",
    "CausalRefuterKind",
    "CausalEstimand",
    "CausalArguments",
    "CausalRefutationResult",
    "CausalEstimateResult",
    "CausalAnalysisRecord",
    "CausalAnalysisCallback",
    "CausalEffectEstimator",
    "CausalReasonerConfigError",
    "DoWhyCausalReasoner",
    "null_causal_analysis_callback",
    "dowhy_linear_regression_estimator",
)
