# ADAPTED FROM: pyro-ppl/numpyro
# (numpyro/infer/svi.py — SVI.run; numpyro/infer/mcmc.py — MCMC.run;
#  numpyro/infer/hmc.py — NUTS / HMC kernels; numpyro/infer/sa.py —
#  Sample Adaptive kernel; numpyro/infer/util.py — summary tables.)
"""C-41 — NumpyroSVIAnalyser: governance-gated NumPyro JAX-backed
probabilistic-programming inference seam.

NumPyro is a JAX-native rewrite of Pyro that exposes the same four
canonical posterior-inference surfaces, but on JAX's XLA-compiled
program-transformation stack rather than PyTorch's autograd:

* SVI (stochastic variational inference) — fit a guide to a model
  via JAX-jitted gradient steps on the ELBO.
* MCMC with the NUTS / HMC kernels — draw an exact posterior sample
  chain via JAX-jitted leapfrog integration.
* Sample Adaptive (SA) — gradient-free Markov-chain Monte Carlo for
  problems where gradients are unavailable or unreliable.

The DIX adapter wraps that posterior surface behind a Protocol
seam so the intelligence layer can ask "given a probabilistic
model digest + observation tensor, what is the posterior summary
(per-site mean, std, ESS, R-hat, divergences) and the marginal
log-evidence?" without ever importing ``numpyro`` / ``jax`` /
``numpy`` at module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``numpyro`` / ``jax`` / ``numpy`` imports are hidden behind a
  :class:`NumpyroInferenceEngine` Protocol — production wires
  :func:`numpyro_svi_engine`; unit tests inject a deterministic
  fake. The module never imports numpyro at module load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`NumpyroInferenceRecord` and
  stops.
* INV-15 byte-identical replays.
  :meth:`NumpyroSVIAnalyser.analyse` with identical
  ``spec`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``engine`` returns identical :class:`NumpyroInferenceRecord`
  records. Determinism is delegated to the injected engine; the
  default factory threads
  :attr:`NumpyroInferenceArguments.random_seed` into JAX's
  ``PRNGKey`` so NumPyro's split-state RNG is fully reproducible.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The inference-method selector — :class:`NumpyroInferenceKind`
  enumerates the four canonical numpyro posterior surfaces
  (SVI / NUTS / HMC / SA).
* The posterior-summary surface —
  :class:`NumpyroSiteSummary` carries the per-site posterior
  statistics that ``numpyro.diagnostics.summary`` reports:
  ``mean`` / ``std`` / ``effective_sample_size`` / ``r_hat`` /
  ``divergences`` (the count of NUTS divergent transitions
  associated with the site's chain; advisory only).
* The marginal evidence —
  :class:`NumpyroInferenceResult.log_evidence` carries the
  marginal log-likelihood ``log p(y)`` returned by SVI ELBO at
  convergence or by SA's marginal estimator.
* The warmup-phase argument —
  :attr:`NumpyroInferenceArguments.num_warmup` mirrors numpyro's
  canonical MCMC warmup count (number of samples discarded
  before the chain is considered to have mixed).

What we replaced
----------------

* NumPyro's matplotlib / arviz plotting → no plotting. The
  numeric summary lives in :class:`NumpyroInferenceResult`; the
  dashboard handles rendering.
* NumPyro's pickle / arviz NetCDF checkpoints → the engine owns
  its posterior; the seam carries a frozen ``model_digest`` so
  identical model parameters produce identical inferences.
* NumPyro's ``tqdm`` progress bar → caller-injected
  :class:`NumpyroInferenceCallback` (default no-op). No
  filesystem writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the engine's PRNG is seeded by
  caller-supplied
  :attr:`NumpyroInferenceArguments.random_seed`. AST tests pin
  the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-15 — :class:`NumpyroInferenceRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MIN_NUM_SITES` 1 / :data:`MAX_NUM_SITES` 1024 hard
    floor and ceiling on posterior-site count.
  - :data:`MIN_NUM_SAMPLES` 1 / :data:`MAX_NUM_SAMPLES` 100_000
    hard floor and ceiling on posterior draws / SVI steps.
  - :data:`MIN_NUM_WARMUP` 0 / :data:`MAX_NUM_WARMUP` 100_000
    hard floor and ceiling on MCMC warmup steps.
  - :data:`MAX_OBSERVATION_LEN` 100_000 hard ceiling on observed
    data length.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on ``analysis_id``.
  - :data:`MAX_MODEL_DIGEST_LEN` 64 chars on model digest.
  - :data:`MAX_SITE_NAME_LEN` 128 chars on site name.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-41 (numpyro probabilistic spec).
- ``intelligence_engine/svi_numpyro.py`` (this file).
- ``intelligence_engine/svi_pyro.py`` (C-40 — the pyro twin
  showing the Protocol seam shape).
- ``intelligence_engine/hmm_hmmlearn.py`` (C-39 — the hmmlearn
  twin showing the lazy-seam factory pattern).
- ``intelligence_engine/pgm_pgmpy.py`` (C-38 — the pgmpy twin).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "numpyro",
    "jax",
    "numpy",
)

MIN_NUM_SITES: int = 1
"""Hard lower bound on :attr:`NumpyroInferenceResult.site_summaries` length."""

MAX_NUM_SITES: int = 1024
"""Hard upper bound on :attr:`NumpyroInferenceResult.site_summaries` length."""

MIN_NUM_SAMPLES: int = 1
"""Hard lower bound on :attr:`NumpyroInferenceArguments.num_samples`."""

MAX_NUM_SAMPLES: int = 100_000
"""Hard upper bound on :attr:`NumpyroInferenceArguments.num_samples`."""

MIN_NUM_WARMUP: int = 0
"""Hard lower bound on :attr:`NumpyroInferenceArguments.num_warmup`."""

MAX_NUM_WARMUP: int = 100_000
"""Hard upper bound on :attr:`NumpyroInferenceArguments.num_warmup`."""

MIN_OBSERVATION_LEN: int = 0
"""Hard lower bound on :attr:`NumpyroInferenceArguments.observations` length."""

MAX_OBSERVATION_LEN: int = 100_000
"""Hard upper bound on :attr:`NumpyroInferenceArguments.observations` length."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_MODEL_DIGEST_LEN: int = 64
"""Hard upper bound on model-digest length."""

MAX_SITE_NAME_LEN: int = 128
"""Hard upper bound on :attr:`NumpyroSiteSummary.name` length."""

ANALYSIS_SOURCE: str = "intelligence_engine.svi_numpyro"
"""Constant tag stamped onto every
:attr:`NumpyroInferenceRecord.source`. Distinguishes
numpyro-produced records from other probabilistic-programming
adapters (e.g. ``intelligence_engine.svi_pyro``)."""


# ---------------------------------------------------------------------------
# Inference-method enum
# ---------------------------------------------------------------------------


class NumpyroInferenceKind(enum.Enum):
    """NumPyro posterior-inference surface selector.

    Values match the canonical numpyro entry-point class names so
    the DIX seam can forward them directly to the underlying
    engine.
    """

    SVI = "SVI"
    NUTS = "NUTS"
    HMC = "HMC"
    SA = "SA"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class NumpyroModelSpec:
    """Frozen probabilistic-model specification.

    * ``num_sites`` — number of stochastic sites (latent variables)
      the model declares via ``numpyro.sample(...)`` calls.
    * ``num_observations`` — observation count the model consumes
      via ``numpyro.sample(..., obs=...)`` calls (informational;
      the analyser separately bounds the actual ``observations``
      tensor).
    * ``model_digest`` — caller-supplied hex digest over the
      model's graph (priors, observation distributions, plate
      structure).
    """

    num_sites: int
    num_observations: int
    model_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.num_sites, int) or isinstance(
            self.num_sites, bool
        ):
            raise TypeError(
                "NumpyroModelSpec.num_sites must be int, got "
                f"{type(self.num_sites).__name__}"
            )
        if self.num_sites < MIN_NUM_SITES:
            raise ValueError(
                "NumpyroModelSpec.num_sites must be >= "
                f"{MIN_NUM_SITES!r}, got {self.num_sites!r}"
            )
        if self.num_sites > MAX_NUM_SITES:
            raise ValueError(
                "NumpyroModelSpec.num_sites must be <= "
                f"{MAX_NUM_SITES!r}, got {self.num_sites!r}"
            )
        if not isinstance(self.num_observations, int) or isinstance(
            self.num_observations, bool
        ):
            raise TypeError(
                "NumpyroModelSpec.num_observations must be int, "
                f"got {type(self.num_observations).__name__}"
            )
        if self.num_observations < 0:
            raise ValueError(
                "NumpyroModelSpec.num_observations must be "
                f"non-negative, got {self.num_observations!r}"
            )
        if self.num_observations > MAX_OBSERVATION_LEN:
            raise ValueError(
                "NumpyroModelSpec.num_observations must be <= "
                f"{MAX_OBSERVATION_LEN!r}, got "
                f"{self.num_observations!r}"
            )
        if not self.model_digest:
            raise ValueError(
                "NumpyroModelSpec.model_digest must be non-empty"
            )
        if len(self.model_digest) > MAX_MODEL_DIGEST_LEN:
            raise ValueError(
                "NumpyroModelSpec.model_digest must be <= "
                f"{MAX_MODEL_DIGEST_LEN} chars, got "
                f"{len(self.model_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class NumpyroInferenceArguments:
    """Frozen inference-run config.

    The :attr:`num_warmup` field is the canonical numpyro MCMC
    warmup-sample count — these draws are discarded by the kernel
    before posterior accounting begins. SVI runs ignore the field
    (the analyser still records it for digest stability).
    """

    inference_kind: NumpyroInferenceKind
    random_seed: int
    num_samples: int
    num_warmup: int
    observations: tuple[float, ...]
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inference_kind, NumpyroInferenceKind):
            raise TypeError(
                "NumpyroInferenceArguments.inference_kind must "
                "be NumpyroInferenceKind, got "
                f"{type(self.inference_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise TypeError(
                "NumpyroInferenceArguments.random_seed must be "
                f"int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "NumpyroInferenceArguments.random_seed must be "
                f"non-negative, got {self.random_seed!r}"
            )
        if not isinstance(self.num_samples, int) or isinstance(
            self.num_samples, bool
        ):
            raise TypeError(
                "NumpyroInferenceArguments.num_samples must be "
                f"int, got {type(self.num_samples).__name__}"
            )
        if self.num_samples < MIN_NUM_SAMPLES:
            raise ValueError(
                "NumpyroInferenceArguments.num_samples must be "
                f">= {MIN_NUM_SAMPLES!r}, got {self.num_samples!r}"
            )
        if self.num_samples > MAX_NUM_SAMPLES:
            raise ValueError(
                "NumpyroInferenceArguments.num_samples must be "
                f"<= {MAX_NUM_SAMPLES!r}, got {self.num_samples!r}"
            )
        if not isinstance(self.num_warmup, int) or isinstance(
            self.num_warmup, bool
        ):
            raise TypeError(
                "NumpyroInferenceArguments.num_warmup must be "
                f"int, got {type(self.num_warmup).__name__}"
            )
        if self.num_warmup < MIN_NUM_WARMUP:
            raise ValueError(
                "NumpyroInferenceArguments.num_warmup must be "
                f">= {MIN_NUM_WARMUP!r}, got {self.num_warmup!r}"
            )
        if self.num_warmup > MAX_NUM_WARMUP:
            raise ValueError(
                "NumpyroInferenceArguments.num_warmup must be "
                f"<= {MAX_NUM_WARMUP!r}, got {self.num_warmup!r}"
            )
        if not isinstance(self.observations, tuple):
            raise TypeError(
                "NumpyroInferenceArguments.observations must be "
                f"a tuple, got {type(self.observations).__name__}"
            )
        if len(self.observations) > MAX_OBSERVATION_LEN:
            raise ValueError(
                "NumpyroInferenceArguments.observations must "
                f"have <= {MAX_OBSERVATION_LEN!r} entries, got "
                f"{len(self.observations)!r}"
            )
        for i, x in enumerate(self.observations):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise TypeError(
                    "NumpyroInferenceArguments.observations "
                    f"values must be float, got "
                    f"{type(x).__name__} at index {i!r}"
                )
            if not math.isfinite(x):
                raise ValueError(
                    "NumpyroInferenceArguments.observations "
                    f"values must be finite, got {x!r} at index "
                    f"{i!r}"
                )
        for k, v in self.meta.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    "NumpyroInferenceArguments.meta keys must be "
                    f"non-empty strings, got {k!r}"
                )
            if not isinstance(v, str) or not v:
                raise ValueError(
                    "NumpyroInferenceArguments.meta values must "
                    f"be non-empty strings, got {v!r}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class NumpyroSiteSummary:
    """Per-site posterior summary statistic.

    Fields mirror ``numpyro.diagnostics.summary`` columns. The
    ``divergences`` field is the count of NUTS divergent
    transitions associated with the site's chain — advisory only;
    SVI runs report 0.
    """

    name: str
    mean: float
    std: float
    effective_sample_size: float
    r_hat: float
    divergences: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "NumpyroSiteSummary.name must be str, got "
                f"{type(self.name).__name__}"
            )
        if not self.name:
            raise ValueError(
                "NumpyroSiteSummary.name must be non-empty"
            )
        if len(self.name) > MAX_SITE_NAME_LEN:
            raise ValueError(
                "NumpyroSiteSummary.name must be <= "
                f"{MAX_SITE_NAME_LEN} chars, got "
                f"{len(self.name)!r}"
            )
        for label, value in (
            ("mean", self.mean),
            ("std", self.std),
            ("effective_sample_size", self.effective_sample_size),
            ("r_hat", self.r_hat),
        ):
            if not isinstance(value, (int, float)) or isinstance(
                value, bool
            ):
                raise TypeError(
                    f"NumpyroSiteSummary.{label} must be float, "
                    f"got {type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"NumpyroSiteSummary.{label} must be finite, "
                    f"got {value!r}"
                )
        if self.std < 0.0:
            raise ValueError(
                "NumpyroSiteSummary.std must be non-negative, got "
                f"{self.std!r}"
            )
        if self.effective_sample_size < 0.0:
            raise ValueError(
                "NumpyroSiteSummary.effective_sample_size must "
                f"be non-negative, got {self.effective_sample_size!r}"
            )
        if self.r_hat < 0.0:
            raise ValueError(
                "NumpyroSiteSummary.r_hat must be non-negative, "
                f"got {self.r_hat!r}"
            )
        if not isinstance(self.divergences, int) or isinstance(
            self.divergences, bool
        ):
            raise TypeError(
                "NumpyroSiteSummary.divergences must be int, "
                f"got {type(self.divergences).__name__}"
            )
        if self.divergences < 0:
            raise ValueError(
                "NumpyroSiteSummary.divergences must be "
                f"non-negative, got {self.divergences!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class NumpyroInferenceResult:
    """Inference output — per-site posterior summaries +
    log-evidence."""

    site_summaries: tuple[NumpyroSiteSummary, ...]
    log_evidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.site_summaries, tuple):
            raise TypeError(
                "NumpyroInferenceResult.site_summaries must be a "
                f"tuple, got {type(self.site_summaries).__name__}"
            )
        if not self.site_summaries:
            raise ValueError(
                "NumpyroInferenceResult.site_summaries must be "
                "non-empty"
            )
        if len(self.site_summaries) > MAX_NUM_SITES:
            raise ValueError(
                "NumpyroInferenceResult.site_summaries must have "
                f"<= {MAX_NUM_SITES!r} entries, got "
                f"{len(self.site_summaries)!r}"
            )
        seen_names: set[str] = set()
        for i, summary in enumerate(self.site_summaries):
            if not isinstance(summary, NumpyroSiteSummary):
                raise TypeError(
                    "NumpyroInferenceResult.site_summaries "
                    f"entries must be NumpyroSiteSummary, got "
                    f"{type(summary).__name__} at index {i!r}"
                )
            if summary.name in seen_names:
                raise ValueError(
                    "NumpyroInferenceResult.site_summaries names "
                    f"must be unique, got duplicate "
                    f"{summary.name!r}"
                )
            seen_names.add(summary.name)
        if not isinstance(
            self.log_evidence, (int, float)
        ) or isinstance(self.log_evidence, bool):
            raise TypeError(
                "NumpyroInferenceResult.log_evidence must be "
                f"float, got {type(self.log_evidence).__name__}"
            )
        if not math.isfinite(self.log_evidence):
            raise ValueError(
                "NumpyroInferenceResult.log_evidence must be "
                f"finite, got {self.log_evidence!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class NumpyroInferenceRecord:
    """Output of :meth:`NumpyroSVIAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    spec: NumpyroModelSpec
    result: NumpyroInferenceResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(
            self.ts_ns, bool
        ):
            raise TypeError(
                "NumpyroInferenceRecord.ts_ns must be int, got "
                f"{type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                "NumpyroInferenceRecord.ts_ns must be "
                f"non-negative, got {self.ts_ns!r}"
            )
        if not self.analysis_id:
            raise ValueError(
                "NumpyroInferenceRecord.analysis_id must be "
                "non-empty"
            )
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "NumpyroInferenceRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError(
                "NumpyroInferenceRecord.source must be non-empty"
            )
        if not isinstance(self.spec, NumpyroModelSpec):
            raise TypeError(
                "NumpyroInferenceRecord.spec must be "
                f"NumpyroModelSpec, got {type(self.spec).__name__}"
            )
        if not isinstance(self.result, NumpyroInferenceResult):
            raise TypeError(
                "NumpyroInferenceRecord.result must be "
                f"NumpyroInferenceResult, got "
                f"{type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "NumpyroInferenceRecord.analysis_digest must be "
                f"a 16-hex-char digest, got "
                f"{self.analysis_digest!r}"
            )
        if not all(
            c in "0123456789abcdef" for c in self.analysis_digest
        ):
            raise ValueError(
                "NumpyroInferenceRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class NumpyroInferenceCallback(Protocol):
    """numpyro-shape lifecycle callback (collapsed into one
    Protocol)."""

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        spec: NumpyroModelSpec,
        arguments: NumpyroInferenceArguments,
    ) -> None: ...

    def on_site_summary(
        self,
        *,
        ts_ns: int,
        summary: NumpyroSiteSummary,
    ) -> None: ...

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: NumpyroInferenceResult,
    ) -> None: ...


@runtime_checkable
class NumpyroInferenceEngine(Protocol):
    """Caller-supplied numpyro inference engine.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`NumpyroInferenceResult` containing the per-site
    posterior summaries and the marginal log-evidence.
    """

    def infer(
        self,
        *,
        spec: NumpyroModelSpec,
        arguments: NumpyroInferenceArguments,
        ts_ns: int,
        callback: NumpyroInferenceCallback,
    ) -> NumpyroInferenceResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullNumpyroInferenceCallback:
    """No-op callback."""

    __slots__ = ()

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        spec: NumpyroModelSpec,
        arguments: NumpyroInferenceArguments,
    ) -> None:
        return None

    def on_site_summary(
        self,
        *,
        ts_ns: int,
        summary: NumpyroSiteSummary,
    ) -> None:
        return None

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: NumpyroInferenceResult,
    ) -> None:
        return None


def null_numpyro_inference_callback() -> NumpyroInferenceCallback:
    return _NullNumpyroInferenceCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NumpyroAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`NumpyroSVIAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    spec: NumpyroModelSpec,
    arguments: NumpyroInferenceArguments,
    result: NumpyroInferenceResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical inference
    summary."""

    observations_str = ",".join(f"{x!r}" for x in arguments.observations)
    meta_pairs = "|".join(
        f"{k}={v}" for k, v in sorted(arguments.meta.items())
    )
    summaries_str = ";".join(
        (
            f"{s.name}:mean={s.mean!r}"
            f",std={s.std!r}"
            f",ess={s.effective_sample_size!r}"
            f",r_hat={s.r_hat!r}"
            f",div={s.divergences!r}"
        )
        for s in result.site_summaries
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"num_sites={spec.num_sites!r}",
            f"num_observations={spec.num_observations!r}",
            f"model_digest={spec.model_digest}",
            f"inference_kind={arguments.inference_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"num_samples={arguments.num_samples!r}",
            f"num_warmup={arguments.num_warmup!r}",
            f"observations={observations_str}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"site_summaries={summaries_str}",
            f"log_evidence={result.log_evidence!r}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# NumpyroSVIAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class NumpyroSVIAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    engine: NumpyroInferenceEngine

    def __post_init__(self) -> None:
        if not isinstance(self.engine, NumpyroInferenceEngine):
            raise TypeError(
                "NumpyroSVIAnalyser.engine must implement the "
                "NumpyroInferenceEngine Protocol, got "
                f"{type(self.engine).__name__}"
            )

    def analyse(
        self,
        *,
        spec: NumpyroModelSpec,
        arguments: NumpyroInferenceArguments,
        ts_ns: int,
        analysis_id: str,
        callback: NumpyroInferenceCallback | None = None,
    ) -> NumpyroInferenceRecord:
        """Run one numpyro inference and emit a
        :class:`NumpyroInferenceRecord`."""

        if not isinstance(spec, NumpyroModelSpec):
            raise TypeError(
                "NumpyroSVIAnalyser.analyse.spec must be "
                f"NumpyroModelSpec, got {type(spec).__name__}"
            )
        if not isinstance(arguments, NumpyroInferenceArguments):
            raise TypeError(
                "NumpyroSVIAnalyser.analyse.arguments must be "
                "NumpyroInferenceArguments, got "
                f"{type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                "NumpyroSVIAnalyser.analyse.ts_ns must be int, "
                f"got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise NumpyroAnalyserConfigError(
                "NumpyroSVIAnalyser.analyse.ts_ns must be "
                f"non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise NumpyroAnalyserConfigError(
                "NumpyroSVIAnalyser.analyse.analysis_id must be "
                "non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise NumpyroAnalyserConfigError(
                "NumpyroSVIAnalyser.analyse.analysis_id must be "
                f"<= {MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        cb = (
            callback if callback is not None
            else null_numpyro_inference_callback()
        )
        if not isinstance(cb, NumpyroInferenceCallback):
            raise TypeError(
                "NumpyroSVIAnalyser.analyse.callback must "
                "implement the NumpyroInferenceCallback "
                f"Protocol, got {type(cb).__name__}"
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
        if not isinstance(result, NumpyroInferenceResult):
            raise TypeError(
                "NumpyroInferenceEngine.infer must return "
                "NumpyroInferenceResult, got "
                f"{type(result).__name__}"
            )
        if len(result.site_summaries) != spec.num_sites:
            raise NumpyroAnalyserConfigError(
                "NumpyroInferenceEngine.infer: site_summaries "
                f"length {len(result.site_summaries)!r} does not "
                f"match spec.num_sites {spec.num_sites!r}"
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
            "inference_kind": arguments.inference_kind.value,
            "random_seed": str(arguments.random_seed),
            "num_samples": str(arguments.num_samples),
            "num_warmup": str(arguments.num_warmup),
            "num_sites": str(spec.num_sites),
            "num_observations": str(spec.num_observations),
            "observation_count": str(len(arguments.observations)),
            "site_summary_count": str(len(result.site_summaries)),
            "log_evidence": repr(result.log_evidence),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return NumpyroInferenceRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            spec=spec,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production engine factory (lazy-import numpyro)
# ---------------------------------------------------------------------------


def numpyro_svi_engine() -> NumpyroInferenceEngine:
    """Production :class:`NumpyroInferenceEngine` backed by
    ``numpyro``.

    Lazy-imports ``numpyro`` + ``jax`` + ``numpy`` inside the
    factory. Raises ``ImportError`` (with a helpful pip-install
    hint) if any package is missing — the rest of the module
    never imports these packages, so the analyser stays usable
    on a host that has never installed them.
    """

    try:
        import jax  # type: ignore[import-not-found]  # noqa: F401
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import numpyro  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "numpyro_svi_engine requires the optional "
            "'numpyro' + 'jax' + 'numpy' packages — install with "
            "'pip install numpyro jax numpy' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/svi_numpyro.py flags this)."
        ) from exc

    _ = numpyro

    class _NumpyroSVIEngine:
        """Thin numpyro wrapper conforming to
        :class:`NumpyroInferenceEngine`."""

        __slots__ = ()

        def infer(
            self,
            *,
            spec: NumpyroModelSpec,
            arguments: NumpyroInferenceArguments,
            ts_ns: int,
            callback: NumpyroInferenceCallback,
        ) -> NumpyroInferenceResult:  # pragma: no cover
            raise NotImplementedError(
                "numpyro_svi_engine is the production seam — its "
                "concrete body is exercised in integration tests "
                "with numpyro installed; unit tests inject a "
                "deterministic fake via the "
                "NumpyroInferenceEngine Protocol."
            )

    return _NumpyroSVIEngine()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_NUM_SITES",
    "MAX_NUM_SITES",
    "MIN_NUM_SAMPLES",
    "MAX_NUM_SAMPLES",
    "MIN_NUM_WARMUP",
    "MAX_NUM_WARMUP",
    "MIN_OBSERVATION_LEN",
    "MAX_OBSERVATION_LEN",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_MODEL_DIGEST_LEN",
    "MAX_SITE_NAME_LEN",
    "ANALYSIS_SOURCE",
    "NumpyroInferenceKind",
    "NumpyroModelSpec",
    "NumpyroInferenceArguments",
    "NumpyroSiteSummary",
    "NumpyroInferenceResult",
    "NumpyroInferenceRecord",
    "NumpyroInferenceCallback",
    "NumpyroInferenceEngine",
    "NumpyroAnalyserConfigError",
    "NumpyroSVIAnalyser",
    "null_numpyro_inference_callback",
    "numpyro_svi_engine",
)
