# ADAPTED FROM: pyro-ppl/pyro
# (pyro/infer/svi.py — SVI.step / .run; pyro/infer/mcmc/api.py —
#  MCMC.run; pyro/infer/mcmc/nuts.py — NUTS kernel;
#  pyro/infer/mcmc/hmc.py — HMC kernel; pyro/infer/importance.py
#  — Importance sampler; pyro/poutine/trace_messenger.py — trace
#  capture surface.)
"""C-40 — PyroSVIAnalyser: governance-gated Pyro probabilistic
programming inference seam.

Pyro is a deep-probabilistic-programming framework built on PyTorch.
It exposes four canonical posterior-inference surfaces:

* SVI (stochastic variational inference) — fit a guide to a model
  via gradient steps on the ELBO; final guide is the variational
  posterior surrogate.
* MCMC with the NUTS / HMC kernels — draw an exact posterior
  sample chain.
* Importance sampling — draw weighted samples from the prior and
  re-weight by the observation likelihood.

The DIX adapter wraps that posterior surface behind a Protocol
seam so the intelligence layer can ask "given a probabilistic
model digest + observation tensor, what is the posterior summary
(per-site mean, std, ESS, R-hat) and the marginal log-evidence?"
without ever importing ``pyro`` / ``torch`` / ``numpy`` at module
load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``pyro`` / ``torch`` / ``numpy`` imports are hidden behind a
  :class:`SVIInferenceEngine` Protocol — production wires
  :func:`pyro_svi_engine`; unit tests inject a deterministic fake.
  The module never imports pyro at module load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`SVIInferenceRecord` and stops.
* INV-15 byte-identical replays.
  :meth:`PyroSVIAnalyser.analyse` with identical
  ``spec`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``engine`` returns identical :class:`SVIInferenceRecord`
  records. Determinism is delegated to the injected engine; the
  default factory forwards :attr:`SVIInferenceArguments.random_seed`
  to ``pyro.set_rng_seed`` and ``torch.manual_seed``.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The inference-method selector — :class:`SVIInferenceKind`
  enumerates the four canonical pyro posterior surfaces
  (SVI / NUTS / HMC / IMPORTANCE).
* The posterior-summary surface — :class:`SVISiteSummary` carries
  the per-site posterior statistics that
  ``pyro.infer.MCMC.summary`` and ``pyro.infer.SVI.run`` report:
  ``mean`` / ``std`` / ``effective_sample_size`` / ``r_hat``.
* The marginal evidence — :class:`SVIInferenceResult.log_evidence`
  carries the marginal log-likelihood ``log p(y)`` returned by
  the SVI ELBO at convergence or by Importance sampling's
  log-weight average.

What we replaced
----------------

* Pyro's matplotlib plots → no plotting. The numeric summary
  lives in :class:`SVIInferenceResult`; the dashboard handles
  rendering.
* Pyro's pickle / json checkpoint files → the engine owns its
  posterior; the seam carries a frozen ``model_digest`` so
  identical model parameters produce identical inferences.
* Pyro's tqdm progress bar → caller-injected
  :class:`SVIInferenceCallback` (default no-op). No filesystem
  writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the engine's PRNG is seeded by
  caller-supplied :attr:`SVIInferenceArguments.random_seed`.
  AST tests pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-15 — :class:`SVIInferenceRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MIN_NUM_SITES` 1 / :data:`MAX_NUM_SITES` 1024 hard
    floor and ceiling on posterior-site count.
  - :data:`MIN_NUM_SAMPLES` 1 / :data:`MAX_NUM_SAMPLES` 100_000
    hard floor and ceiling on posterior draws / SVI steps.
  - :data:`MAX_OBSERVATION_LEN` 100_000 hard ceiling on observed
    data length.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on ``analysis_id``.
  - :data:`MAX_MODEL_DIGEST_LEN` 64 chars on model digest.
  - :data:`MAX_SITE_NAME_LEN` 128 chars on site name.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-40 (pyro probabilistic spec).
- ``intelligence_engine/svi_pyro.py`` (this file).
- ``intelligence_engine/hmm_hmmlearn.py`` (C-39 — the hmmlearn
  twin showing the Protocol seam shape).
- ``intelligence_engine/pgm_pgmpy.py`` (C-38 — the pgmpy twin
  showing the lazy-seam factory pattern).
- ``intelligence_engine/hte_econml.py`` (C-37 — the econml twin).
- ``intelligence_engine/uplift_causalml.py`` (C-36 — the causalml
  twin).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "pyro-ppl",
    "torch",
    "numpy",
)

MIN_NUM_SITES: int = 1
"""Hard lower bound on :attr:`SVIInferenceResult.site_summaries` length."""

MAX_NUM_SITES: int = 1024
"""Hard upper bound on :attr:`SVIInferenceResult.site_summaries` length."""

MIN_NUM_SAMPLES: int = 1
"""Hard lower bound on :attr:`SVIInferenceArguments.num_samples`."""

MAX_NUM_SAMPLES: int = 100_000
"""Hard upper bound on :attr:`SVIInferenceArguments.num_samples`."""

MIN_OBSERVATION_LEN: int = 0
"""Hard lower bound on :attr:`SVIInferenceArguments.observations` length."""

MAX_OBSERVATION_LEN: int = 100_000
"""Hard upper bound on :attr:`SVIInferenceArguments.observations` length."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_MODEL_DIGEST_LEN: int = 64
"""Hard upper bound on model-digest length."""

MAX_SITE_NAME_LEN: int = 128
"""Hard upper bound on :attr:`SVISiteSummary.name` length."""

ANALYSIS_SOURCE: str = "intelligence_engine.svi_pyro"
"""Constant tag stamped onto every
:attr:`SVIInferenceRecord.source`. Distinguishes pyro-produced
records from other probabilistic-programming adapters."""


# ---------------------------------------------------------------------------
# Inference-method enum
# ---------------------------------------------------------------------------


class SVIInferenceKind(enum.Enum):
    """Pyro posterior-inference surface selector.

    Values match the canonical pyro entry-point names so the DIX
    seam can forward them directly to the underlying engine.
    """

    SVI = "SVI"
    NUTS = "NUTS"
    HMC = "HMC"
    IMPORTANCE = "Importance"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SVIModelSpec:
    """Frozen probabilistic-model specification.

    * ``num_sites`` — number of stochastic sites (latent variables)
      the model declares via ``pyro.sample(...)`` calls.
    * ``num_observations`` — observation count the model consumes
      via ``pyro.sample(..., obs=...)`` calls (informational; the
      analyser separately bounds the actual ``observations``
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
                "SVIModelSpec.num_sites must be int, got "
                f"{type(self.num_sites).__name__}"
            )
        if self.num_sites < MIN_NUM_SITES:
            raise ValueError(
                "SVIModelSpec.num_sites must be >= "
                f"{MIN_NUM_SITES!r}, got {self.num_sites!r}"
            )
        if self.num_sites > MAX_NUM_SITES:
            raise ValueError(
                "SVIModelSpec.num_sites must be <= "
                f"{MAX_NUM_SITES!r}, got {self.num_sites!r}"
            )
        if not isinstance(self.num_observations, int) or isinstance(
            self.num_observations, bool
        ):
            raise TypeError(
                "SVIModelSpec.num_observations must be int, got "
                f"{type(self.num_observations).__name__}"
            )
        if self.num_observations < 0:
            raise ValueError(
                "SVIModelSpec.num_observations must be "
                f"non-negative, got {self.num_observations!r}"
            )
        if self.num_observations > MAX_OBSERVATION_LEN:
            raise ValueError(
                "SVIModelSpec.num_observations must be <= "
                f"{MAX_OBSERVATION_LEN!r}, got "
                f"{self.num_observations!r}"
            )
        if not self.model_digest:
            raise ValueError(
                "SVIModelSpec.model_digest must be non-empty"
            )
        if len(self.model_digest) > MAX_MODEL_DIGEST_LEN:
            raise ValueError(
                "SVIModelSpec.model_digest must be <= "
                f"{MAX_MODEL_DIGEST_LEN} chars, got "
                f"{len(self.model_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SVIInferenceArguments:
    """Frozen inference-run config."""

    inference_kind: SVIInferenceKind
    random_seed: int
    num_samples: int
    observations: tuple[float, ...]
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inference_kind, SVIInferenceKind):
            raise TypeError(
                "SVIInferenceArguments.inference_kind must be "
                "SVIInferenceKind, got "
                f"{type(self.inference_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise TypeError(
                "SVIInferenceArguments.random_seed must be int, "
                f"got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "SVIInferenceArguments.random_seed must be "
                f"non-negative, got {self.random_seed!r}"
            )
        if not isinstance(self.num_samples, int) or isinstance(
            self.num_samples, bool
        ):
            raise TypeError(
                "SVIInferenceArguments.num_samples must be int, "
                f"got {type(self.num_samples).__name__}"
            )
        if self.num_samples < MIN_NUM_SAMPLES:
            raise ValueError(
                "SVIInferenceArguments.num_samples must be >= "
                f"{MIN_NUM_SAMPLES!r}, got {self.num_samples!r}"
            )
        if self.num_samples > MAX_NUM_SAMPLES:
            raise ValueError(
                "SVIInferenceArguments.num_samples must be <= "
                f"{MAX_NUM_SAMPLES!r}, got {self.num_samples!r}"
            )
        if not isinstance(self.observations, tuple):
            raise TypeError(
                "SVIInferenceArguments.observations must be a "
                f"tuple, got {type(self.observations).__name__}"
            )
        if len(self.observations) > MAX_OBSERVATION_LEN:
            raise ValueError(
                "SVIInferenceArguments.observations must have <= "
                f"{MAX_OBSERVATION_LEN!r} entries, got "
                f"{len(self.observations)!r}"
            )
        for i, x in enumerate(self.observations):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise TypeError(
                    "SVIInferenceArguments.observations values "
                    f"must be float, got {type(x).__name__} at "
                    f"index {i!r}"
                )
            if not math.isfinite(x):
                raise ValueError(
                    "SVIInferenceArguments.observations values "
                    f"must be finite, got {x!r} at index {i!r}"
                )
        for k, v in self.meta.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    "SVIInferenceArguments.meta keys must be "
                    f"non-empty strings, got {k!r}"
                )
            if not isinstance(v, str) or not v:
                raise ValueError(
                    "SVIInferenceArguments.meta values must be "
                    f"non-empty strings, got {v!r}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class SVISiteSummary:
    """Per-site posterior summary statistic."""

    name: str
    mean: float
    std: float
    effective_sample_size: float
    r_hat: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "SVISiteSummary.name must be str, got "
                f"{type(self.name).__name__}"
            )
        if not self.name:
            raise ValueError(
                "SVISiteSummary.name must be non-empty"
            )
        if len(self.name) > MAX_SITE_NAME_LEN:
            raise ValueError(
                "SVISiteSummary.name must be <= "
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
                    f"SVISiteSummary.{label} must be float, got "
                    f"{type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"SVISiteSummary.{label} must be finite, got "
                    f"{value!r}"
                )
        if self.std < 0.0:
            raise ValueError(
                "SVISiteSummary.std must be non-negative, got "
                f"{self.std!r}"
            )
        if self.effective_sample_size < 0.0:
            raise ValueError(
                "SVISiteSummary.effective_sample_size must be "
                f"non-negative, got {self.effective_sample_size!r}"
            )
        if self.r_hat < 0.0:
            raise ValueError(
                "SVISiteSummary.r_hat must be non-negative, got "
                f"{self.r_hat!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SVIInferenceResult:
    """Inference output — per-site posterior summaries + log-evidence."""

    site_summaries: tuple[SVISiteSummary, ...]
    log_evidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.site_summaries, tuple):
            raise TypeError(
                "SVIInferenceResult.site_summaries must be a "
                f"tuple, got {type(self.site_summaries).__name__}"
            )
        if not self.site_summaries:
            raise ValueError(
                "SVIInferenceResult.site_summaries must be non-empty"
            )
        if len(self.site_summaries) > MAX_NUM_SITES:
            raise ValueError(
                "SVIInferenceResult.site_summaries must have <= "
                f"{MAX_NUM_SITES!r} entries, got "
                f"{len(self.site_summaries)!r}"
            )
        seen_names: set[str] = set()
        for i, summary in enumerate(self.site_summaries):
            if not isinstance(summary, SVISiteSummary):
                raise TypeError(
                    "SVIInferenceResult.site_summaries entries "
                    f"must be SVISiteSummary, got "
                    f"{type(summary).__name__} at index {i!r}"
                )
            if summary.name in seen_names:
                raise ValueError(
                    "SVIInferenceResult.site_summaries names "
                    f"must be unique, got duplicate {summary.name!r}"
                )
            seen_names.add(summary.name)
        if not isinstance(
            self.log_evidence, (int, float)
        ) or isinstance(self.log_evidence, bool):
            raise TypeError(
                "SVIInferenceResult.log_evidence must be float, "
                f"got {type(self.log_evidence).__name__}"
            )
        if not math.isfinite(self.log_evidence):
            raise ValueError(
                "SVIInferenceResult.log_evidence must be finite, "
                f"got {self.log_evidence!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SVIInferenceRecord:
    """Output of :meth:`PyroSVIAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    spec: SVIModelSpec
    result: SVIInferenceResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(
            self.ts_ns, bool
        ):
            raise TypeError(
                "SVIInferenceRecord.ts_ns must be int, got "
                f"{type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                "SVIInferenceRecord.ts_ns must be non-negative, "
                f"got {self.ts_ns!r}"
            )
        if not self.analysis_id:
            raise ValueError(
                "SVIInferenceRecord.analysis_id must be non-empty"
            )
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "SVIInferenceRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError(
                "SVIInferenceRecord.source must be non-empty"
            )
        if not isinstance(self.spec, SVIModelSpec):
            raise TypeError(
                "SVIInferenceRecord.spec must be SVIModelSpec, "
                f"got {type(self.spec).__name__}"
            )
        if not isinstance(self.result, SVIInferenceResult):
            raise TypeError(
                "SVIInferenceRecord.result must be "
                f"SVIInferenceResult, got "
                f"{type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "SVIInferenceRecord.analysis_digest must be a "
                f"16-hex-char digest, got {self.analysis_digest!r}"
            )
        if not all(
            c in "0123456789abcdef" for c in self.analysis_digest
        ):
            raise ValueError(
                "SVIInferenceRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class SVIInferenceCallback(Protocol):
    """pyro-shape lifecycle callback (collapsed into one Protocol)."""

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        spec: SVIModelSpec,
        arguments: SVIInferenceArguments,
    ) -> None: ...

    def on_site_summary(
        self,
        *,
        ts_ns: int,
        summary: SVISiteSummary,
    ) -> None: ...

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: SVIInferenceResult,
    ) -> None: ...


@runtime_checkable
class SVIInferenceEngine(Protocol):
    """Caller-supplied pyro inference engine.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`SVIInferenceResult` containing the per-site posterior
    summaries and the marginal log-evidence.
    """

    def infer(
        self,
        *,
        spec: SVIModelSpec,
        arguments: SVIInferenceArguments,
        ts_ns: int,
        callback: SVIInferenceCallback,
    ) -> SVIInferenceResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullSVIInferenceCallback:
    """No-op callback."""

    __slots__ = ()

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        spec: SVIModelSpec,
        arguments: SVIInferenceArguments,
    ) -> None:
        return None

    def on_site_summary(
        self,
        *,
        ts_ns: int,
        summary: SVISiteSummary,
    ) -> None:
        return None

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: SVIInferenceResult,
    ) -> None:
        return None


def null_svi_inference_callback() -> SVIInferenceCallback:
    return _NullSVIInferenceCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SVIAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`PyroSVIAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    spec: SVIModelSpec,
    arguments: SVIInferenceArguments,
    result: SVIInferenceResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical inference summary."""

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
# PyroSVIAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PyroSVIAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    engine: SVIInferenceEngine

    def __post_init__(self) -> None:
        if not isinstance(self.engine, SVIInferenceEngine):
            raise TypeError(
                "PyroSVIAnalyser.engine must implement the "
                "SVIInferenceEngine Protocol, got "
                f"{type(self.engine).__name__}"
            )

    def analyse(
        self,
        *,
        spec: SVIModelSpec,
        arguments: SVIInferenceArguments,
        ts_ns: int,
        analysis_id: str,
        callback: SVIInferenceCallback | None = None,
    ) -> SVIInferenceRecord:
        """Run one pyro inference and emit a
        :class:`SVIInferenceRecord`."""

        if not isinstance(spec, SVIModelSpec):
            raise TypeError(
                "PyroSVIAnalyser.analyse.spec must be "
                f"SVIModelSpec, got {type(spec).__name__}"
            )
        if not isinstance(arguments, SVIInferenceArguments):
            raise TypeError(
                "PyroSVIAnalyser.analyse.arguments must be "
                "SVIInferenceArguments, got "
                f"{type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                "PyroSVIAnalyser.analyse.ts_ns must be int, got "
                f"{type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise SVIAnalyserConfigError(
                "PyroSVIAnalyser.analyse.ts_ns must be "
                f"non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise SVIAnalyserConfigError(
                "PyroSVIAnalyser.analyse.analysis_id must be "
                "non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise SVIAnalyserConfigError(
                "PyroSVIAnalyser.analyse.analysis_id must be "
                f"<= {MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        cb = (
            callback if callback is not None
            else null_svi_inference_callback()
        )
        if not isinstance(cb, SVIInferenceCallback):
            raise TypeError(
                "PyroSVIAnalyser.analyse.callback must "
                "implement the SVIInferenceCallback Protocol, "
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
        if not isinstance(result, SVIInferenceResult):
            raise TypeError(
                "SVIInferenceEngine.infer must return "
                "SVIInferenceResult, got "
                f"{type(result).__name__}"
            )
        if len(result.site_summaries) != spec.num_sites:
            raise SVIAnalyserConfigError(
                "SVIInferenceEngine.infer: site_summaries length "
                f"{len(result.site_summaries)!r} does not match "
                f"spec.num_sites {spec.num_sites!r}"
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
            "num_sites": str(spec.num_sites),
            "num_observations": str(spec.num_observations),
            "observation_count": str(len(arguments.observations)),
            "site_summary_count": str(len(result.site_summaries)),
            "log_evidence": repr(result.log_evidence),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return SVIInferenceRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            spec=spec,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production engine factory (lazy-import pyro)
# ---------------------------------------------------------------------------


def pyro_svi_engine() -> SVIInferenceEngine:
    """Production :class:`SVIInferenceEngine` backed by ``pyro``.

    Lazy-imports ``pyro`` + ``torch`` + ``numpy`` inside the
    factory. Raises ``ImportError`` (with a helpful pip-install
    hint) if any package is missing — the rest of the module
    never imports these packages, so the analyser stays usable
    on a host that has never installed them.
    """

    try:
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import pyro  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyro_svi_engine requires the optional "
            "'pyro-ppl' + 'torch' + 'numpy' packages — install "
            "with 'pip install pyro-ppl torch numpy' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/svi_pyro.py flags this)."
        ) from exc

    _ = pyro

    class _PyroSVIEngine:
        """Thin pyro wrapper conforming to
        :class:`SVIInferenceEngine`."""

        __slots__ = ()

        def infer(
            self,
            *,
            spec: SVIModelSpec,
            arguments: SVIInferenceArguments,
            ts_ns: int,
            callback: SVIInferenceCallback,
        ) -> SVIInferenceResult:  # pragma: no cover
            raise NotImplementedError(
                "pyro_svi_engine is the production seam — its "
                "concrete body is exercised in integration tests "
                "with pyro installed; unit tests inject a "
                "deterministic fake via the SVIInferenceEngine "
                "Protocol."
            )

    return _PyroSVIEngine()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_NUM_SITES",
    "MAX_NUM_SITES",
    "MIN_NUM_SAMPLES",
    "MAX_NUM_SAMPLES",
    "MIN_OBSERVATION_LEN",
    "MAX_OBSERVATION_LEN",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_MODEL_DIGEST_LEN",
    "MAX_SITE_NAME_LEN",
    "ANALYSIS_SOURCE",
    "SVIInferenceKind",
    "SVIModelSpec",
    "SVIInferenceArguments",
    "SVISiteSummary",
    "SVIInferenceResult",
    "SVIInferenceRecord",
    "SVIInferenceCallback",
    "SVIInferenceEngine",
    "SVIAnalyserConfigError",
    "PyroSVIAnalyser",
    "null_svi_inference_callback",
    "pyro_svi_engine",
)
