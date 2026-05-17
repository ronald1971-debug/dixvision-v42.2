# ADAPTED FROM: hmmlearn/hmmlearn
# (hmmlearn/base.py — _BaseHMM.decode / .predict_proba / .score; 
#  hmmlearn/hmm.py — GaussianHMM / GMMHMM / MultinomialHMM /
#  CategoricalHMM; hmmlearn/_kl_divergence.py — forward-backward
#  inference.)
"""C-39 — HmmlearnAnalyser: governance-gated Hidden Markov Model
inference seam.

hmmlearn is a Python library for unsupervised learning and inference
of Hidden Markov Models — Gaussian, GMM, multinomial, categorical —
built on numpy + scipy + scikit-learn. The DIX adapter wraps that
inference surface behind a Protocol seam so the intelligence layer
can ask "given observation sequence O, what is the most-likely
hidden-state path, the per-step posterior over states, and the
log-likelihood?" without ever importing hmmlearn at module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``hmmlearn`` / ``numpy`` / ``scipy`` / ``scikit-learn`` imports
  are hidden behind a :class:`HMMInferenceEngine` Protocol —
  production wires :func:`hmmlearn_gaussian_engine`; unit tests
  inject a deterministic fake. The module never imports hmmlearn
  at module load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`HMMInferenceRecord` and stops.
* INV-15 byte-identical replays.
  :meth:`HmmlearnAnalyser.analyse` with identical
  ``spec`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``engine`` returns identical
  :class:`HMMInferenceRecord` records. Determinism is delegated
  to the injected engine; the default factory forwards
  :attr:`HMMInferenceArguments.random_seed` to
  ``numpy.random.seed`` and hmmlearn's ``random_state``.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The model-family selector — :class:`HMMModelKind` enumerates the
  hmmlearn model classes we currently expose (Gaussian,
  multinomial, categorical, GMM).
* The decode/score/posterior surface — :class:`HMMInferenceResult`
  carries the three canonical hmmlearn outputs: ``decode`` →
  ``viterbi_path``, ``predict_proba`` → ``posteriors``,
  ``score`` → ``log_likelihood``.
* The observation-sequence shape — :class:`HMMInferenceArguments`
  carries ``observations`` as a ``T x D`` float tuple-of-tuples
  matching ``_BaseHMM.decode(X)``'s expected input.

What we replaced
----------------

* hmmlearn's matplotlib plots → no plotting. The numeric summary
  lives in :class:`HMMInferenceResult`; the dashboard handles
  rendering.
* hmmlearn's pandas / scikit-learn data IO → the engine owns its
  data source; the seam carries a frozen ``model_digest`` so
  identical model parameters produce identical inferences (no
  parameter round-tripping).
* hmmlearn's tqdm sampler progress bar → caller-injected
  :class:`HMMInferenceCallback` (default no-op). No filesystem
  writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the engine's PRNG is seeded by
  caller-supplied :attr:`HMMInferenceArguments.random_seed`.
  AST tests pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-15 — :class:`HMMInferenceRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MIN_N_COMPONENTS` 1 / :data:`MAX_N_COMPONENTS` 256
    hard floor and ceiling on hidden-state count.
  - :data:`MIN_N_FEATURES` 1 / :data:`MAX_N_FEATURES` 256 hard
    floor and ceiling on observation dimension.
  - :data:`MAX_OBSERVATION_LEN` 100_000 hard ceiling on
    observation sequence length.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on ``analysis_id``.
  - :data:`MAX_MODEL_DIGEST_LEN` 64 chars on model digest.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-39 (hmmlearn HMM spec).
- ``intelligence_engine/hmm_hmmlearn.py`` (this file).
- ``intelligence_engine/pgm_pgmpy.py`` (C-38 — the pgmpy twin
  showing the lazy-seam factory shape).
- ``intelligence_engine/hte_econml.py`` (C-37 — the econml twin).
- ``intelligence_engine/uplift_causalml.py`` (C-36 — the causalml
  twin).
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
    "hmmlearn",
    "numpy",
    "scipy",
    "scikit-learn",
)

MIN_N_COMPONENTS: int = 1
"""Hard lower bound on :attr:`HMMSpec.n_components`."""

MAX_N_COMPONENTS: int = 256
"""Hard upper bound on :attr:`HMMSpec.n_components`."""

MIN_N_FEATURES: int = 1
"""Hard lower bound on :attr:`HMMSpec.n_features`."""

MAX_N_FEATURES: int = 256
"""Hard upper bound on :attr:`HMMSpec.n_features`."""

MIN_OBSERVATION_LEN: int = 1
"""Hard lower bound on :attr:`HMMInferenceArguments.observations` length."""

MAX_OBSERVATION_LEN: int = 100_000
"""Hard upper bound on :attr:`HMMInferenceArguments.observations` length."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_MODEL_DIGEST_LEN: int = 64
"""Hard upper bound on model-digest length."""

ANALYSIS_SOURCE: str = "intelligence_engine.hmm_hmmlearn"
"""Constant tag stamped onto every
:attr:`HMMInferenceRecord.source`. Distinguishes hmmlearn-produced
records from other HMM adapters."""


# ---------------------------------------------------------------------------
# Model-family enum
# ---------------------------------------------------------------------------


class HMMModelKind(enum.Enum):
    """hmmlearn model-family selector.

    Values match the canonical hmmlearn class names so the DIX seam
    can forward them directly to ``hmmlearn.hmm.*`` constructors.
    """

    GAUSSIAN = "GaussianHMM"
    GMM = "GMMHMM"
    MULTINOMIAL = "MultinomialHMM"
    CATEGORICAL = "CategoricalHMM"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class HMMSpec:
    """Frozen HMM-topology specification.

    * ``n_components`` — number of hidden states.
    * ``n_features`` — observation dimension (or number of
      symbols, for the categorical/multinomial families).
    * ``model_digest`` — caller-supplied hex digest over the
      model's transition matrix, start-probability vector, and
      emission parameters.
    """

    n_components: int
    n_features: int
    model_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.n_components, int) or isinstance(
            self.n_components, bool
        ):
            raise TypeError(
                "HMMSpec.n_components must be int, got "
                f"{type(self.n_components).__name__}"
            )
        if self.n_components < MIN_N_COMPONENTS:
            raise ValueError(
                "HMMSpec.n_components must be >= "
                f"{MIN_N_COMPONENTS!r}, got {self.n_components!r}"
            )
        if self.n_components > MAX_N_COMPONENTS:
            raise ValueError(
                "HMMSpec.n_components must be <= "
                f"{MAX_N_COMPONENTS!r}, got {self.n_components!r}"
            )
        if not isinstance(self.n_features, int) or isinstance(
            self.n_features, bool
        ):
            raise TypeError(
                "HMMSpec.n_features must be int, got "
                f"{type(self.n_features).__name__}"
            )
        if self.n_features < MIN_N_FEATURES:
            raise ValueError(
                "HMMSpec.n_features must be >= "
                f"{MIN_N_FEATURES!r}, got {self.n_features!r}"
            )
        if self.n_features > MAX_N_FEATURES:
            raise ValueError(
                "HMMSpec.n_features must be <= "
                f"{MAX_N_FEATURES!r}, got {self.n_features!r}"
            )
        if not self.model_digest:
            raise ValueError(
                "HMMSpec.model_digest must be non-empty"
            )
        if len(self.model_digest) > MAX_MODEL_DIGEST_LEN:
            raise ValueError(
                "HMMSpec.model_digest must be <= "
                f"{MAX_MODEL_DIGEST_LEN} chars, got "
                f"{len(self.model_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HMMInferenceArguments:
    """Frozen inference-run config."""

    model_kind: HMMModelKind
    random_seed: int
    observations: tuple[tuple[float, ...], ...]
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_kind, HMMModelKind):
            raise TypeError(
                "HMMInferenceArguments.model_kind must be "
                f"HMMModelKind, got "
                f"{type(self.model_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise TypeError(
                "HMMInferenceArguments.random_seed must be "
                f"int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "HMMInferenceArguments.random_seed must be "
                f"non-negative, got {self.random_seed!r}"
            )
        if not isinstance(self.observations, tuple):
            raise TypeError(
                "HMMInferenceArguments.observations must be a "
                f"tuple, got {type(self.observations).__name__}"
            )
        if len(self.observations) < MIN_OBSERVATION_LEN:
            raise ValueError(
                "HMMInferenceArguments.observations must have >= "
                f"{MIN_OBSERVATION_LEN!r} steps, got "
                f"{len(self.observations)!r}"
            )
        if len(self.observations) > MAX_OBSERVATION_LEN:
            raise ValueError(
                "HMMInferenceArguments.observations must have <= "
                f"{MAX_OBSERVATION_LEN!r} steps, got "
                f"{len(self.observations)!r}"
            )
        feature_count: int | None = None
        for i, step in enumerate(self.observations):
            if not isinstance(step, tuple):
                raise TypeError(
                    "HMMInferenceArguments.observations entries "
                    f"must be tuples, got {type(step).__name__} "
                    f"at step {i!r}"
                )
            if not step:
                raise ValueError(
                    "HMMInferenceArguments.observations entries "
                    f"must be non-empty, got empty at step {i!r}"
                )
            if feature_count is None:
                feature_count = len(step)
            elif len(step) != feature_count:
                raise ValueError(
                    "HMMInferenceArguments.observations entries "
                    f"must share dimension, got {len(step)!r} at "
                    f"step {i!r} vs {feature_count!r}"
                )
            for j, x in enumerate(step):
                if not isinstance(x, (int, float)) or isinstance(
                    x, bool
                ):
                    raise TypeError(
                        "HMMInferenceArguments.observations "
                        f"values must be float, got "
                        f"{type(x).__name__} at step {i!r} "
                        f"feature {j!r}"
                    )
                if not math.isfinite(x):
                    raise ValueError(
                        "HMMInferenceArguments.observations "
                        f"values must be finite, got {x!r} at "
                        f"step {i!r} feature {j!r}"
                    )
        for k, v in self.meta.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    "HMMInferenceArguments.meta keys must be "
                    f"non-empty strings, got {k!r}"
                )
            if not isinstance(v, str) or not v:
                raise ValueError(
                    "HMMInferenceArguments.meta values must be "
                    f"non-empty strings, got {v!r}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class HMMStatePosterior:
    """Per-step posterior distribution over hidden states."""

    step_index: int
    state_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or isinstance(
            self.step_index, bool
        ):
            raise TypeError(
                "HMMStatePosterior.step_index must be int, got "
                f"{type(self.step_index).__name__}"
            )
        if self.step_index < 0:
            raise ValueError(
                "HMMStatePosterior.step_index must be "
                f"non-negative, got {self.step_index!r}"
            )
        if not isinstance(self.state_probabilities, tuple):
            raise TypeError(
                "HMMStatePosterior.state_probabilities must be a "
                f"tuple, got "
                f"{type(self.state_probabilities).__name__}"
            )
        if not self.state_probabilities:
            raise ValueError(
                "HMMStatePosterior.state_probabilities must be "
                "non-empty"
            )
        if len(self.state_probabilities) > MAX_N_COMPONENTS:
            raise ValueError(
                "HMMStatePosterior.state_probabilities must have "
                f"<= {MAX_N_COMPONENTS} entries, got "
                f"{len(self.state_probabilities)!r}"
            )
        for p in self.state_probabilities:
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                raise TypeError(
                    "HMMStatePosterior.state_probabilities entries "
                    f"must be float, got {type(p).__name__}"
                )
            if not math.isfinite(p):
                raise ValueError(
                    "HMMStatePosterior.state_probabilities entries "
                    f"must be finite, got {p!r}"
                )
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    "HMMStatePosterior.state_probabilities entries "
                    f"must be in [0.0, 1.0], got {p!r}"
                )
        total = math.fsum(self.state_probabilities)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "HMMStatePosterior.state_probabilities must sum "
                f"to 1.0, got {total!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HMMInferenceResult:
    """Inference output — Viterbi path, posteriors, and score."""

    viterbi_path: tuple[int, ...]
    posteriors: tuple[HMMStatePosterior, ...]
    log_likelihood: float

    def __post_init__(self) -> None:
        if not isinstance(self.viterbi_path, tuple):
            raise TypeError(
                "HMMInferenceResult.viterbi_path must be a "
                f"tuple, got {type(self.viterbi_path).__name__}"
            )
        if not self.viterbi_path:
            raise ValueError(
                "HMMInferenceResult.viterbi_path must be non-empty"
            )
        if len(self.viterbi_path) > MAX_OBSERVATION_LEN:
            raise ValueError(
                "HMMInferenceResult.viterbi_path must have <= "
                f"{MAX_OBSERVATION_LEN!r} entries, got "
                f"{len(self.viterbi_path)!r}"
            )
        for i, s in enumerate(self.viterbi_path):
            if not isinstance(s, int) or isinstance(s, bool):
                raise TypeError(
                    "HMMInferenceResult.viterbi_path entries must "
                    f"be int, got {type(s).__name__} at step {i!r}"
                )
            if s < 0:
                raise ValueError(
                    "HMMInferenceResult.viterbi_path entries must "
                    f"be non-negative, got {s!r} at step {i!r}"
                )
            if s >= MAX_N_COMPONENTS:
                raise ValueError(
                    "HMMInferenceResult.viterbi_path entries must "
                    f"be < {MAX_N_COMPONENTS!r}, got {s!r} at "
                    f"step {i!r}"
                )
        if not isinstance(self.posteriors, tuple):
            raise TypeError(
                "HMMInferenceResult.posteriors must be a tuple, "
                f"got {type(self.posteriors).__name__}"
            )
        if len(self.posteriors) != len(self.viterbi_path):
            raise ValueError(
                "HMMInferenceResult.posteriors must have the "
                "same length as viterbi_path, got "
                f"{len(self.posteriors)!r} vs "
                f"{len(self.viterbi_path)!r}"
            )
        n_components: int | None = None
        for i, post in enumerate(self.posteriors):
            if not isinstance(post, HMMStatePosterior):
                raise TypeError(
                    "HMMInferenceResult.posteriors entries must "
                    f"be HMMStatePosterior, got "
                    f"{type(post).__name__} at step {i!r}"
                )
            if post.step_index != i:
                raise ValueError(
                    "HMMInferenceResult.posteriors entries must "
                    f"have step_index == position, got "
                    f"{post.step_index!r} at position {i!r}"
                )
            if n_components is None:
                n_components = len(post.state_probabilities)
            elif len(post.state_probabilities) != n_components:
                raise ValueError(
                    "HMMInferenceResult.posteriors entries must "
                    "share state_probabilities length, got "
                    f"{len(post.state_probabilities)!r} at step "
                    f"{i!r} vs {n_components!r}"
                )
            if self.viterbi_path[i] >= (n_components or 0):
                raise ValueError(
                    "HMMInferenceResult.viterbi_path entry "
                    f"{self.viterbi_path[i]!r} at step {i!r} is "
                    "out of range for posterior state count "
                    f"{n_components!r}"
                )
        if not isinstance(self.log_likelihood, (int, float)) or isinstance(
            self.log_likelihood, bool
        ):
            raise TypeError(
                "HMMInferenceResult.log_likelihood must be "
                f"float, got {type(self.log_likelihood).__name__}"
            )
        if not math.isfinite(self.log_likelihood):
            raise ValueError(
                "HMMInferenceResult.log_likelihood must be "
                f"finite, got {self.log_likelihood!r}"
            )
        if self.log_likelihood > 0.0:
            raise ValueError(
                "HMMInferenceResult.log_likelihood must be "
                f"non-positive, got {self.log_likelihood!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HMMInferenceRecord:
    """Output of :meth:`HmmlearnAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    spec: HMMSpec
    result: HMMInferenceResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(
            self.ts_ns, bool
        ):
            raise TypeError(
                "HMMInferenceRecord.ts_ns must be int, got "
                f"{type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                "HMMInferenceRecord.ts_ns must be non-negative, "
                f"got {self.ts_ns!r}"
            )
        if not self.analysis_id:
            raise ValueError(
                "HMMInferenceRecord.analysis_id must be non-empty"
            )
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "HMMInferenceRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError(
                "HMMInferenceRecord.source must be non-empty"
            )
        if not isinstance(self.spec, HMMSpec):
            raise TypeError(
                "HMMInferenceRecord.spec must be HMMSpec, got "
                f"{type(self.spec).__name__}"
            )
        if not isinstance(self.result, HMMInferenceResult):
            raise TypeError(
                "HMMInferenceRecord.result must be "
                f"HMMInferenceResult, got "
                f"{type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "HMMInferenceRecord.analysis_digest must be a "
                f"16-hex-char digest, got {self.analysis_digest!r}"
            )
        if not all(
            c in "0123456789abcdef" for c in self.analysis_digest
        ):
            raise ValueError(
                "HMMInferenceRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class HMMInferenceCallback(Protocol):
    """hmmlearn-shape lifecycle callback (collapsed into one Protocol)."""

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        spec: HMMSpec,
        arguments: HMMInferenceArguments,
    ) -> None: ...

    def on_step_posterior(
        self,
        *,
        ts_ns: int,
        posterior: HMMStatePosterior,
    ) -> None: ...

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: HMMInferenceResult,
    ) -> None: ...


@runtime_checkable
class HMMInferenceEngine(Protocol):
    """Caller-supplied hmmlearn inference engine.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`HMMInferenceResult` containing the Viterbi path,
    per-step posteriors, and the model log-likelihood.
    """

    def infer(
        self,
        *,
        spec: HMMSpec,
        arguments: HMMInferenceArguments,
        ts_ns: int,
        callback: HMMInferenceCallback,
    ) -> HMMInferenceResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullHMMInferenceCallback:
    """No-op callback."""

    __slots__ = ()

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        spec: HMMSpec,
        arguments: HMMInferenceArguments,
    ) -> None:
        return None

    def on_step_posterior(
        self,
        *,
        ts_ns: int,
        posterior: HMMStatePosterior,
    ) -> None:
        return None

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: HMMInferenceResult,
    ) -> None:
        return None


def null_hmm_inference_callback() -> HMMInferenceCallback:
    return _NullHMMInferenceCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HMMAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`HmmlearnAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    spec: HMMSpec,
    arguments: HMMInferenceArguments,
    result: HMMInferenceResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical inference summary."""

    observations_str = ";".join(
        ",".join(f"{x!r}" for x in step)
        for step in arguments.observations
    )
    meta_pairs = "|".join(
        f"{k}={v}" for k, v in sorted(arguments.meta.items())
    )
    viterbi_str = ",".join(str(s) for s in result.viterbi_path)
    posteriors_str = ";".join(
        f"{post.step_index}:"
        + ",".join(f"{p!r}" for p in post.state_probabilities)
        for post in result.posteriors
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"n_components={spec.n_components!r}",
            f"n_features={spec.n_features!r}",
            f"model_digest={spec.model_digest}",
            f"model_kind={arguments.model_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"observations={observations_str}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"viterbi_path={viterbi_str}",
            f"posteriors={posteriors_str}",
            f"log_likelihood={result.log_likelihood!r}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# HmmlearnAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class HmmlearnAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    engine: HMMInferenceEngine

    def __post_init__(self) -> None:
        if not isinstance(self.engine, HMMInferenceEngine):
            raise TypeError(
                "HmmlearnAnalyser.engine must implement the "
                "HMMInferenceEngine Protocol, got "
                f"{type(self.engine).__name__}"
            )

    def analyse(
        self,
        *,
        spec: HMMSpec,
        arguments: HMMInferenceArguments,
        ts_ns: int,
        analysis_id: str,
        callback: HMMInferenceCallback | None = None,
    ) -> HMMInferenceRecord:
        """Run one HMM inference and emit a
        :class:`HMMInferenceRecord`."""

        if not isinstance(spec, HMMSpec):
            raise TypeError(
                "HmmlearnAnalyser.analyse.spec must be HMMSpec, "
                f"got {type(spec).__name__}"
            )
        if not isinstance(arguments, HMMInferenceArguments):
            raise TypeError(
                "HmmlearnAnalyser.analyse.arguments must be "
                "HMMInferenceArguments, got "
                f"{type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                "HmmlearnAnalyser.analyse.ts_ns must be int, got "
                f"{type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise HMMAnalyserConfigError(
                "HmmlearnAnalyser.analyse.ts_ns must be "
                f"non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise HMMAnalyserConfigError(
                "HmmlearnAnalyser.analyse.analysis_id must be "
                "non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise HMMAnalyserConfigError(
                "HmmlearnAnalyser.analyse.analysis_id must be "
                f"<= {MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        if arguments.observations:
            obs_dim = len(arguments.observations[0])
            if obs_dim != spec.n_features:
                raise HMMAnalyserConfigError(
                    "HmmlearnAnalyser.analyse: observation "
                    f"dimension {obs_dim!r} does not match "
                    f"spec.n_features {spec.n_features!r}"
                )

        cb = (
            callback if callback is not None
            else null_hmm_inference_callback()
        )
        if not isinstance(cb, HMMInferenceCallback):
            raise TypeError(
                "HmmlearnAnalyser.analyse.callback must "
                "implement the HMMInferenceCallback Protocol, "
                f"got {type(cb).__name__}"
            )

        cb.on_inference_start(
            ts_ns=ts_ns,
            spec=spec,
            arguments=arguments,
        )
        result = self.engine.infer(
            spec=spec,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(result, HMMInferenceResult):
            raise TypeError(
                "HMMInferenceEngine.infer must return "
                "HMMInferenceResult, got "
                f"{type(result).__name__}"
            )
        if len(result.viterbi_path) != len(arguments.observations):
            raise HMMAnalyserConfigError(
                "HMMInferenceEngine.infer: viterbi_path length "
                f"{len(result.viterbi_path)!r} does not match "
                "observations length "
                f"{len(arguments.observations)!r}"
            )
        if result.posteriors:
            inferred_n_components = len(
                result.posteriors[0].state_probabilities
            )
            if inferred_n_components != spec.n_components:
                raise HMMAnalyserConfigError(
                    "HMMInferenceEngine.infer: posterior state "
                    f"count {inferred_n_components!r} does not "
                    f"match spec.n_components "
                    f"{spec.n_components!r}"
                )
        for s in result.viterbi_path:
            if s >= spec.n_components:
                raise HMMAnalyserConfigError(
                    "HMMInferenceEngine.infer: viterbi_path "
                    f"entry {s!r} out of range for "
                    f"spec.n_components {spec.n_components!r}"
                )
        cb.on_inference_end(ts_ns=ts_ns, result=result)

        digest = _compute_analysis_digest(
            spec=spec,
            arguments=arguments,
            result=result,
            ts_ns=ts_ns,
            analysis_id=analysis_id,
        )
        record_meta: dict[str, str] = {
            "analysis_digest": digest,
            "model_kind": arguments.model_kind.value,
            "random_seed": str(arguments.random_seed),
            "n_components": str(spec.n_components),
            "n_features": str(spec.n_features),
            "observation_count": str(len(arguments.observations)),
            "viterbi_length": str(len(result.viterbi_path)),
            "posterior_count": str(len(result.posteriors)),
            "log_likelihood": repr(result.log_likelihood),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return HMMInferenceRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            spec=spec,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production engine factory (lazy-import hmmlearn)
# ---------------------------------------------------------------------------


def hmmlearn_gaussian_engine() -> HMMInferenceEngine:
    """Production :class:`HMMInferenceEngine` backed by ``hmmlearn``.

    Lazy-imports ``hmmlearn`` + ``numpy`` + ``scipy`` +
    ``scikit-learn`` inside the factory. Raises ``ImportError``
    (with a helpful pip-install hint) if any package is missing —
    the rest of the module never imports these packages, so the
    analyser stays usable on a host that has never installed them.
    """

    try:
        import hmmlearn  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import scipy  # type: ignore[import-not-found]  # noqa: F401
        import sklearn  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "hmmlearn_gaussian_engine requires the optional "
            "'hmmlearn' + 'numpy' + 'scipy' + 'scikit-learn' "
            "packages — install with 'pip install hmmlearn numpy "
            "scipy scikit-learn' (NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/hmm_hmmlearn.py flags this)."
        ) from exc

    _ = hmmlearn

    class _HmmlearnGaussianEngine:
        """Thin hmmlearn wrapper conforming to
        :class:`HMMInferenceEngine`."""

        __slots__ = ()

        def infer(
            self,
            *,
            spec: HMMSpec,
            arguments: HMMInferenceArguments,
            ts_ns: int,
            callback: HMMInferenceCallback,
        ) -> HMMInferenceResult:  # pragma: no cover
            raise NotImplementedError(
                "hmmlearn_gaussian_engine is the production "
                "seam — its concrete body is exercised in "
                "integration tests with hmmlearn installed; unit "
                "tests inject a deterministic fake via the "
                "HMMInferenceEngine Protocol."
            )

    return _HmmlearnGaussianEngine()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_N_COMPONENTS",
    "MAX_N_COMPONENTS",
    "MIN_N_FEATURES",
    "MAX_N_FEATURES",
    "MIN_OBSERVATION_LEN",
    "MAX_OBSERVATION_LEN",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_MODEL_DIGEST_LEN",
    "ANALYSIS_SOURCE",
    "HMMModelKind",
    "HMMSpec",
    "HMMInferenceArguments",
    "HMMStatePosterior",
    "HMMInferenceResult",
    "HMMInferenceRecord",
    "HMMInferenceCallback",
    "HMMInferenceEngine",
    "HMMAnalyserConfigError",
    "HmmlearnAnalyser",
    "null_hmm_inference_callback",
    "hmmlearn_gaussian_engine",
)
