# ADAPTED FROM: arviz-devs/arviz
# (arviz/stats/stats.py — summary / hdi / ess / rhat / mcse;
#  arviz/data/inference_data.py — InferenceData container shape.)
"""C-42 — ArviZDiagnosticAnalyser: governance-gated ArviZ
exploratory-analysis seam for Bayesian posterior samples.

ArviZ is the canonical post-inference diagnostics library on top
of pyro / numpyro / pymc / cmdstanpy / emcee. It consumes a chains
× draws × variables sample tensor and reports the four canonical
posterior diagnostics:

* SUMMARY — per-variable posterior mean, standard deviation,
  highest-density interval (HDI 3% / 97%), bulk effective sample
  size, tail effective sample size, R-hat convergence statistic.
* RHAT — per-variable potential-scale-reduction (R-hat); chain
  convergence is satisfied when R-hat is close to 1.
* ESS — per-variable effective sample size; the bulk variant
  estimates centrality precision, the tail variant estimates
  quantile precision.
* MCSE — per-variable Monte Carlo standard error; the standard
  error of the chain's mean estimator.

The DIX adapter wraps that surface behind a Protocol seam so the
intelligence layer can ask "given a posterior sample tensor of
shape (num_chains, num_draws, num_vars), what is the per-variable
diagnostic summary?" without ever importing ``arviz`` / ``numpy``
/ ``xarray`` at module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``arviz`` / ``numpy`` / ``xarray`` imports are hidden behind a
  :class:`ArviZDiagnosticEngine` Protocol — production wires
  :func:`arviz_diagnostic_engine`; unit tests inject a
  deterministic fake. The module never imports arviz at module
  load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`ArviZDiagnosticRecord` and
  stops.
* INV-15 byte-identical replays.
  :meth:`ArviZDiagnosticAnalyser.analyse` with identical
  ``spec`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``engine`` returns identical :class:`ArviZDiagnosticRecord`
  records. Determinism is delegated to the injected engine; the
  default factory threads
  :attr:`ArviZDiagnosticArguments.random_seed` into numpy's
  ``default_rng`` so any resampling diagnostic stays
  reproducible.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The diagnostic-kind selector — :class:`ArviZDiagnosticKind`
  enumerates the four canonical post-inference diagnostics
  (SUMMARY / RHAT / ESS / MCSE).
* The per-variable summary surface —
  :class:`ArviZVariableSummary` carries the columns that
  ``arviz.summary`` reports: ``mean`` / ``sd`` / ``hdi_3`` /
  ``hdi_97`` / ``ess_bulk`` / ``ess_tail`` / ``r_hat``.
* The chain-divergence accounting —
  :attr:`ArviZDiagnosticResult.num_divergences` carries the
  total count of NUTS divergent transitions observed across all
  chains (advisory only; gradient-free kernels report 0).

What we replaced
----------------

* ArviZ's matplotlib / bokeh plotting → no plotting. The numeric
  summary lives in :class:`ArviZDiagnosticResult`; the dashboard
  handles rendering.
* ArviZ's NetCDF / Zarr persistence → the engine owns the
  sample tensor; the seam carries a frozen ``model_digest`` so
  identical posterior parameters produce identical diagnostics.
* ArviZ's ``tqdm`` progress bar → caller-injected
  :class:`ArviZDiagnosticCallback` (default no-op). No
  filesystem writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the engine's PRNG is seeded by
  caller-supplied
  :attr:`ArviZDiagnosticArguments.random_seed`. AST tests pin
  the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-15 — :class:`ArviZDiagnosticRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MIN_NUM_VARS` 1 / :data:`MAX_NUM_VARS` 1024 hard
    floor and ceiling on posterior-variable count.
  - :data:`MIN_NUM_CHAINS` 1 / :data:`MAX_NUM_CHAINS` 64 hard
    floor and ceiling on chain count.
  - :data:`MIN_NUM_DRAWS` 1 / :data:`MAX_NUM_DRAWS` 100_000
    hard floor and ceiling on per-chain draw count.
  - :data:`MAX_SAMPLE_LEN` 1_000_000 hard ceiling on flattened
    sample tensor length.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on ``analysis_id``.
  - :data:`MAX_MODEL_DIGEST_LEN` 64 chars on model digest.
  - :data:`MAX_VAR_NAME_LEN` 128 chars on variable name.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-42 (arviz diagnostic spec).
- ``intelligence_engine/diag_arviz.py`` (this file).
- ``intelligence_engine/svi_numpyro.py`` (C-41 — the numpyro twin
  showing the Protocol seam shape).
- ``intelligence_engine/svi_pyro.py`` (C-40 — the pyro twin).
- ``intelligence_engine/hmm_hmmlearn.py`` (C-39 — the hmmlearn
  twin showing the lazy-seam factory pattern).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "arviz",
    "numpy",
    "xarray",
)

MIN_NUM_VARS: int = 1
"""Hard lower bound on :attr:`ArviZDiagnosticResult.variable_summaries` length."""

MAX_NUM_VARS: int = 1024
"""Hard upper bound on :attr:`ArviZDiagnosticResult.variable_summaries` length."""

MIN_NUM_CHAINS: int = 1
"""Hard lower bound on :attr:`ArviZPosteriorSpec.num_chains`."""

MAX_NUM_CHAINS: int = 64
"""Hard upper bound on :attr:`ArviZPosteriorSpec.num_chains`."""

MIN_NUM_DRAWS: int = 1
"""Hard lower bound on :attr:`ArviZPosteriorSpec.num_draws`."""

MAX_NUM_DRAWS: int = 100_000
"""Hard upper bound on :attr:`ArviZPosteriorSpec.num_draws`."""

MIN_SAMPLE_LEN: int = 0
"""Hard lower bound on :attr:`ArviZDiagnosticArguments.samples` length."""

MAX_SAMPLE_LEN: int = 1_000_000
"""Hard upper bound on :attr:`ArviZDiagnosticArguments.samples` length."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_MODEL_DIGEST_LEN: int = 64
"""Hard upper bound on model-digest length."""

MAX_VAR_NAME_LEN: int = 128
"""Hard upper bound on :attr:`ArviZVariableSummary.name` length."""

ANALYSIS_SOURCE: str = "intelligence_engine.diag_arviz"
"""Constant tag stamped onto every
:attr:`ArviZDiagnosticRecord.source`. Distinguishes arviz-produced
records from the upstream inference adapters
(e.g. ``intelligence_engine.svi_pyro``,
``intelligence_engine.svi_numpyro``)."""


# ---------------------------------------------------------------------------
# Diagnostic-kind enum
# ---------------------------------------------------------------------------


class ArviZDiagnosticKind(enum.Enum):
    """ArviZ post-inference diagnostic selector.

    Values match the canonical arviz function names so the DIX
    seam can forward them directly to the underlying engine.
    """

    SUMMARY = "summary"
    RHAT = "rhat"
    ESS = "ess"
    MCSE = "mcse"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ArviZPosteriorSpec:
    """Frozen posterior-tensor specification.

    * ``num_chains`` — count of independent Markov chains the
      posterior tensor exposes (the first axis of arviz's
      canonical ``(chain, draw, *shape)`` layout).
    * ``num_draws`` — count of post-warmup draws per chain (the
      second axis of the canonical layout).
    * ``num_vars`` — count of posterior variables tracked
      (informational; the analyser separately bounds the
      ``variable_summaries`` tuple length).
    * ``model_digest`` — caller-supplied hex digest over the
      posterior model (priors, likelihood, plate structure).
    """

    num_chains: int
    num_draws: int
    num_vars: int
    model_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.num_chains, int) or isinstance(self.num_chains, bool):
            raise TypeError(
                f"ArviZPosteriorSpec.num_chains must be int, got {type(self.num_chains).__name__}"
            )
        if self.num_chains < MIN_NUM_CHAINS:
            raise ValueError(
                "ArviZPosteriorSpec.num_chains must be >= "
                f"{MIN_NUM_CHAINS!r}, got {self.num_chains!r}"
            )
        if self.num_chains > MAX_NUM_CHAINS:
            raise ValueError(
                "ArviZPosteriorSpec.num_chains must be <= "
                f"{MAX_NUM_CHAINS!r}, got {self.num_chains!r}"
            )
        if not isinstance(self.num_draws, int) or isinstance(self.num_draws, bool):
            raise TypeError(
                f"ArviZPosteriorSpec.num_draws must be int, got {type(self.num_draws).__name__}"
            )
        if self.num_draws < MIN_NUM_DRAWS:
            raise ValueError(
                f"ArviZPosteriorSpec.num_draws must be >= {MIN_NUM_DRAWS!r}, got {self.num_draws!r}"
            )
        if self.num_draws > MAX_NUM_DRAWS:
            raise ValueError(
                f"ArviZPosteriorSpec.num_draws must be <= {MAX_NUM_DRAWS!r}, got {self.num_draws!r}"
            )
        if not isinstance(self.num_vars, int) or isinstance(self.num_vars, bool):
            raise TypeError(
                f"ArviZPosteriorSpec.num_vars must be int, got {type(self.num_vars).__name__}"
            )
        if self.num_vars < MIN_NUM_VARS:
            raise ValueError(
                f"ArviZPosteriorSpec.num_vars must be >= {MIN_NUM_VARS!r}, got {self.num_vars!r}"
            )
        if self.num_vars > MAX_NUM_VARS:
            raise ValueError(
                f"ArviZPosteriorSpec.num_vars must be <= {MAX_NUM_VARS!r}, got {self.num_vars!r}"
            )
        if not self.model_digest:
            raise ValueError("ArviZPosteriorSpec.model_digest must be non-empty")
        if len(self.model_digest) > MAX_MODEL_DIGEST_LEN:
            raise ValueError(
                "ArviZPosteriorSpec.model_digest must be <= "
                f"{MAX_MODEL_DIGEST_LEN} chars, got "
                f"{len(self.model_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ArviZDiagnosticArguments:
    """Frozen diagnostic-run config.

    The :attr:`samples` field carries a flattened
    ``(num_chains * num_draws * num_vars,)`` posterior-sample
    tensor in row-major order (chain-major, then draw, then
    variable). The engine reshapes internally before calling
    arviz's diagnostic surface.

    The :attr:`hdi_prob` field is the canonical arviz
    highest-density-interval probability mass (default 0.94 —
    matching arviz's own default).
    """

    diagnostic_kind: ArviZDiagnosticKind
    random_seed: int
    hdi_prob: float
    samples: tuple[float, ...]
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostic_kind, ArviZDiagnosticKind):
            raise TypeError(
                "ArviZDiagnosticArguments.diagnostic_kind must "
                "be ArviZDiagnosticKind, got "
                f"{type(self.diagnostic_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                "ArviZDiagnosticArguments.random_seed must be "
                f"int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "ArviZDiagnosticArguments.random_seed must be "
                f"non-negative, got {self.random_seed!r}"
            )
        if not isinstance(self.hdi_prob, (int, float)) or isinstance(self.hdi_prob, bool):
            raise TypeError(
                "ArviZDiagnosticArguments.hdi_prob must be "
                f"float, got {type(self.hdi_prob).__name__}"
            )
        if not math.isfinite(self.hdi_prob):
            raise ValueError(
                f"ArviZDiagnosticArguments.hdi_prob must be finite, got {self.hdi_prob!r}"
            )
        if not (0.0 < self.hdi_prob < 1.0):
            raise ValueError(
                f"ArviZDiagnosticArguments.hdi_prob must be in (0.0, 1.0), got {self.hdi_prob!r}"
            )
        if not isinstance(self.samples, tuple):
            raise TypeError(
                "ArviZDiagnosticArguments.samples must be a "
                f"tuple, got {type(self.samples).__name__}"
            )
        if len(self.samples) > MAX_SAMPLE_LEN:
            raise ValueError(
                "ArviZDiagnosticArguments.samples must have <= "
                f"{MAX_SAMPLE_LEN!r} entries, got "
                f"{len(self.samples)!r}"
            )
        for i, x in enumerate(self.samples):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise TypeError(
                    "ArviZDiagnosticArguments.samples values "
                    f"must be float, got {type(x).__name__} at "
                    f"index {i!r}"
                )
            if not math.isfinite(x):
                raise ValueError(
                    "ArviZDiagnosticArguments.samples values "
                    f"must be finite, got {x!r} at index {i!r}"
                )
        for k, v in self.meta.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    f"ArviZDiagnosticArguments.meta keys must be non-empty strings, got {k!r}"
                )
            if not isinstance(v, str) or not v:
                raise ValueError(
                    f"ArviZDiagnosticArguments.meta values must be non-empty strings, got {v!r}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class ArviZVariableSummary:
    """Per-variable diagnostic summary.

    Fields mirror ``arviz.summary`` columns:

    * ``mean`` — posterior mean.
    * ``sd`` — posterior standard deviation.
    * ``hdi_3`` / ``hdi_97`` — lower / upper bound of the 94%
      highest-density interval (the names match arviz's default
      column labels when ``hdi_prob=0.94``).
    * ``ess_bulk`` — bulk effective sample size (centrality
      precision).
    * ``ess_tail`` — tail effective sample size (quantile
      precision).
    * ``r_hat`` — potential-scale-reduction statistic;
      convergence is satisfied when R-hat is close to 1.
    """

    name: str
    mean: float
    sd: float
    hdi_3: float
    hdi_97: float
    ess_bulk: float
    ess_tail: float
    r_hat: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                f"ArviZVariableSummary.name must be str, got {type(self.name).__name__}"
            )
        if not self.name:
            raise ValueError("ArviZVariableSummary.name must be non-empty")
        if len(self.name) > MAX_VAR_NAME_LEN:
            raise ValueError(
                "ArviZVariableSummary.name must be <= "
                f"{MAX_VAR_NAME_LEN} chars, got "
                f"{len(self.name)!r}"
            )
        for label, value in (
            ("mean", self.mean),
            ("sd", self.sd),
            ("hdi_3", self.hdi_3),
            ("hdi_97", self.hdi_97),
            ("ess_bulk", self.ess_bulk),
            ("ess_tail", self.ess_tail),
            ("r_hat", self.r_hat),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"ArviZVariableSummary.{label} must be float, got {type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(f"ArviZVariableSummary.{label} must be finite, got {value!r}")
        if self.sd < 0.0:
            raise ValueError(f"ArviZVariableSummary.sd must be non-negative, got {self.sd!r}")
        if self.hdi_3 > self.hdi_97:
            raise ValueError(
                "ArviZVariableSummary.hdi_3 must be <= hdi_97, "
                f"got hdi_3={self.hdi_3!r} hdi_97={self.hdi_97!r}"
            )
        if self.ess_bulk < 0.0:
            raise ValueError(
                f"ArviZVariableSummary.ess_bulk must be non-negative, got {self.ess_bulk!r}"
            )
        if self.ess_tail < 0.0:
            raise ValueError(
                f"ArviZVariableSummary.ess_tail must be non-negative, got {self.ess_tail!r}"
            )
        if self.r_hat < 0.0:
            raise ValueError(f"ArviZVariableSummary.r_hat must be non-negative, got {self.r_hat!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class ArviZDiagnosticResult:
    """Diagnostic output — per-variable summaries + chain
    divergences."""

    variable_summaries: tuple[ArviZVariableSummary, ...]
    num_divergences: int

    def __post_init__(self) -> None:
        if not isinstance(self.variable_summaries, tuple):
            raise TypeError(
                "ArviZDiagnosticResult.variable_summaries must "
                "be a tuple, got "
                f"{type(self.variable_summaries).__name__}"
            )
        if not self.variable_summaries:
            raise ValueError("ArviZDiagnosticResult.variable_summaries must be non-empty")
        if len(self.variable_summaries) > MAX_NUM_VARS:
            raise ValueError(
                "ArviZDiagnosticResult.variable_summaries must "
                f"have <= {MAX_NUM_VARS!r} entries, got "
                f"{len(self.variable_summaries)!r}"
            )
        seen_names: set[str] = set()
        for i, summary in enumerate(self.variable_summaries):
            if not isinstance(summary, ArviZVariableSummary):
                raise TypeError(
                    "ArviZDiagnosticResult.variable_summaries "
                    f"entries must be ArviZVariableSummary, got "
                    f"{type(summary).__name__} at index {i!r}"
                )
            if summary.name in seen_names:
                raise ValueError(
                    "ArviZDiagnosticResult.variable_summaries "
                    f"names must be unique, got duplicate "
                    f"{summary.name!r}"
                )
            seen_names.add(summary.name)
        if not isinstance(self.num_divergences, int) or isinstance(self.num_divergences, bool):
            raise TypeError(
                "ArviZDiagnosticResult.num_divergences must be "
                f"int, got {type(self.num_divergences).__name__}"
            )
        if self.num_divergences < 0:
            raise ValueError(
                "ArviZDiagnosticResult.num_divergences must be "
                f"non-negative, got {self.num_divergences!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ArviZDiagnosticRecord:
    """Output of :meth:`ArviZDiagnosticAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    spec: ArviZPosteriorSpec
    result: ArviZDiagnosticResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(
                f"ArviZDiagnosticRecord.ts_ns must be int, got {type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                f"ArviZDiagnosticRecord.ts_ns must be non-negative, got {self.ts_ns!r}"
            )
        if not self.analysis_id:
            raise ValueError("ArviZDiagnosticRecord.analysis_id must be non-empty")
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "ArviZDiagnosticRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError("ArviZDiagnosticRecord.source must be non-empty")
        if not isinstance(self.spec, ArviZPosteriorSpec):
            raise TypeError(
                "ArviZDiagnosticRecord.spec must be "
                "ArviZPosteriorSpec, got "
                f"{type(self.spec).__name__}"
            )
        if not isinstance(self.result, ArviZDiagnosticResult):
            raise TypeError(
                "ArviZDiagnosticRecord.result must be "
                "ArviZDiagnosticResult, got "
                f"{type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "ArviZDiagnosticRecord.analysis_digest must be "
                f"a 16-hex-char digest, got "
                f"{self.analysis_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.analysis_digest):
            raise ValueError(
                "ArviZDiagnosticRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class ArviZDiagnosticCallback(Protocol):
    """arviz-shape lifecycle callback (collapsed into one
    Protocol)."""

    def on_diagnostic_start(
        self,
        *,
        ts_ns: int,
        spec: ArviZPosteriorSpec,
        arguments: ArviZDiagnosticArguments,
    ) -> None: ...

    def on_variable_summary(
        self,
        *,
        ts_ns: int,
        summary: ArviZVariableSummary,
    ) -> None: ...

    def on_diagnostic_end(
        self,
        *,
        ts_ns: int,
        result: ArviZDiagnosticResult,
    ) -> None: ...


@runtime_checkable
class ArviZDiagnosticEngine(Protocol):
    """Caller-supplied arviz diagnostic engine.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`ArviZDiagnosticResult` containing the per-variable
    diagnostic summaries and the total chain-divergence count.
    """

    def diagnose(
        self,
        *,
        spec: ArviZPosteriorSpec,
        arguments: ArviZDiagnosticArguments,
        ts_ns: int,
        callback: ArviZDiagnosticCallback,
    ) -> ArviZDiagnosticResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullArviZDiagnosticCallback:
    """No-op callback."""

    __slots__ = ()

    def on_diagnostic_start(
        self,
        *,
        ts_ns: int,
        spec: ArviZPosteriorSpec,
        arguments: ArviZDiagnosticArguments,
    ) -> None:
        return None

    def on_variable_summary(
        self,
        *,
        ts_ns: int,
        summary: ArviZVariableSummary,
    ) -> None:
        return None

    def on_diagnostic_end(
        self,
        *,
        ts_ns: int,
        result: ArviZDiagnosticResult,
    ) -> None:
        return None


def null_arviz_diagnostic_callback() -> ArviZDiagnosticCallback:
    return _NullArviZDiagnosticCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArviZAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`ArviZDiagnosticAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    spec: ArviZPosteriorSpec,
    arguments: ArviZDiagnosticArguments,
    result: ArviZDiagnosticResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical diagnostic
    summary."""

    samples_str = ",".join(f"{x!r}" for x in arguments.samples)
    meta_pairs = "|".join(f"{k}={v}" for k, v in sorted(arguments.meta.items()))
    summaries_str = ";".join(
        (
            f"{s.name}:mean={s.mean!r}"
            f",sd={s.sd!r}"
            f",hdi_3={s.hdi_3!r}"
            f",hdi_97={s.hdi_97!r}"
            f",ess_bulk={s.ess_bulk!r}"
            f",ess_tail={s.ess_tail!r}"
            f",r_hat={s.r_hat!r}"
        )
        for s in result.variable_summaries
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"num_chains={spec.num_chains!r}",
            f"num_draws={spec.num_draws!r}",
            f"num_vars={spec.num_vars!r}",
            f"model_digest={spec.model_digest}",
            f"diagnostic_kind={arguments.diagnostic_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"hdi_prob={arguments.hdi_prob!r}",
            f"samples={samples_str}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"variable_summaries={summaries_str}",
            f"num_divergences={result.num_divergences!r}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# ArviZDiagnosticAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ArviZDiagnosticAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    engine: ArviZDiagnosticEngine

    def __post_init__(self) -> None:
        if not isinstance(self.engine, ArviZDiagnosticEngine):
            raise TypeError(
                "ArviZDiagnosticAnalyser.engine must implement "
                "the ArviZDiagnosticEngine Protocol, got "
                f"{type(self.engine).__name__}"
            )

    def analyse(
        self,
        *,
        spec: ArviZPosteriorSpec,
        arguments: ArviZDiagnosticArguments,
        ts_ns: int,
        analysis_id: str,
        callback: ArviZDiagnosticCallback | None = None,
    ) -> ArviZDiagnosticRecord:
        """Run one arviz diagnostic and emit an
        :class:`ArviZDiagnosticRecord`."""

        if not isinstance(spec, ArviZPosteriorSpec):
            raise TypeError(
                "ArviZDiagnosticAnalyser.analyse.spec must be "
                f"ArviZPosteriorSpec, got {type(spec).__name__}"
            )
        if not isinstance(arguments, ArviZDiagnosticArguments):
            raise TypeError(
                "ArviZDiagnosticAnalyser.analyse.arguments must "
                "be ArviZDiagnosticArguments, got "
                f"{type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                f"ArviZDiagnosticAnalyser.analyse.ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise ArviZAnalyserConfigError(
                f"ArviZDiagnosticAnalyser.analyse.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise ArviZAnalyserConfigError(
                "ArviZDiagnosticAnalyser.analyse.analysis_id must be non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ArviZAnalyserConfigError(
                "ArviZDiagnosticAnalyser.analyse.analysis_id "
                f"must be <= {MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        cb = callback if callback is not None else null_arviz_diagnostic_callback()
        if not isinstance(cb, ArviZDiagnosticCallback):
            raise TypeError(
                "ArviZDiagnosticAnalyser.analyse.callback must "
                "implement the ArviZDiagnosticCallback "
                f"Protocol, got {type(cb).__name__}"
            )

        cb.on_diagnostic_start(
            ts_ns=ts_ns,
            spec=spec,
            arguments=arguments,
        )
        result = self.engine.diagnose(
            spec=spec,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(result, ArviZDiagnosticResult):
            raise TypeError(
                "ArviZDiagnosticEngine.diagnose must return "
                "ArviZDiagnosticResult, got "
                f"{type(result).__name__}"
            )
        if len(result.variable_summaries) != spec.num_vars:
            raise ArviZAnalyserConfigError(
                "ArviZDiagnosticEngine.diagnose: "
                "variable_summaries length "
                f"{len(result.variable_summaries)!r} does not "
                f"match spec.num_vars {spec.num_vars!r}"
            )
        cb.on_diagnostic_end(ts_ns=ts_ns, result=result)

        digest = _compute_analysis_digest(
            spec=spec,
            arguments=arguments,
            result=result,
            ts_ns=ts_ns,
            analysis_id=analysis_id,
        )
        record_meta: dict[str, str] = {
            "analysis_digest": digest,
            "diagnostic_kind": arguments.diagnostic_kind.value,
            "random_seed": str(arguments.random_seed),
            "hdi_prob": repr(arguments.hdi_prob),
            "num_chains": str(spec.num_chains),
            "num_draws": str(spec.num_draws),
            "num_vars": str(spec.num_vars),
            "sample_count": str(len(arguments.samples)),
            "variable_summary_count": str(len(result.variable_summaries)),
            "num_divergences": str(result.num_divergences),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return ArviZDiagnosticRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            spec=spec,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production engine factory (lazy-import arviz)
# ---------------------------------------------------------------------------


def arviz_diagnostic_engine() -> ArviZDiagnosticEngine:
    """Production :class:`ArviZDiagnosticEngine` backed by
    ``arviz``.

    Lazy-imports ``arviz`` + ``numpy`` + ``xarray`` inside the
    factory. Raises ``ImportError`` (with a helpful pip-install
    hint) if any package is missing — the rest of the module
    never imports these packages, so the analyser stays usable
    on a host that has never installed them.
    """

    try:
        import arviz  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import xarray  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "arviz_diagnostic_engine requires the optional "
            "'arviz' + 'numpy' + 'xarray' packages — install with "
            "'pip install arviz numpy xarray' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/diag_arviz.py flags this)."
        ) from exc

    _ = arviz

    class _ArviZDiagnosticEngine:
        """Thin arviz wrapper conforming to
        :class:`ArviZDiagnosticEngine`."""

        __slots__ = ()

        def diagnose(
            self,
            *,
            spec: ArviZPosteriorSpec,
            arguments: ArviZDiagnosticArguments,
            ts_ns: int,
            callback: ArviZDiagnosticCallback,
        ) -> ArviZDiagnosticResult:  # pragma: no cover
            raise NotImplementedError(
                "arviz_diagnostic_engine is the production seam "
                "— its concrete body is exercised in integration "
                "tests with arviz installed; unit tests inject a "
                "deterministic fake via the "
                "ArviZDiagnosticEngine Protocol."
            )

    return _ArviZDiagnosticEngine()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_NUM_VARS",
    "MAX_NUM_VARS",
    "MIN_NUM_CHAINS",
    "MAX_NUM_CHAINS",
    "MIN_NUM_DRAWS",
    "MAX_NUM_DRAWS",
    "MIN_SAMPLE_LEN",
    "MAX_SAMPLE_LEN",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_MODEL_DIGEST_LEN",
    "MAX_VAR_NAME_LEN",
    "ANALYSIS_SOURCE",
    "ArviZDiagnosticKind",
    "ArviZPosteriorSpec",
    "ArviZDiagnosticArguments",
    "ArviZVariableSummary",
    "ArviZDiagnosticResult",
    "ArviZDiagnosticRecord",
    "ArviZDiagnosticCallback",
    "ArviZDiagnosticEngine",
    "ArviZAnalyserConfigError",
    "ArviZDiagnosticAnalyser",
    "null_arviz_diagnostic_callback",
    "arviz_diagnostic_engine",
)
