# ADAPTED FROM: pgmpy/pgmpy
# (pgmpy/inference/ — VariableElimination, BeliefPropagation,
#  ApproxInference; pgmpy/models/ — BayesianNetwork / MarkovNetwork;
#  pgmpy/factors/discrete/ — Tabular CPD / DiscreteFactor; pgmpy/sampling/
#  — Gibbs / LikelihoodWeighting samplers.)
"""C-38 — PgmpyBayesianAnalyser: governance-gated Bayesian-network
inference seam.

pgmpy is a Python probabilistic-graphical-models library exposing
Bayesian + Markov networks, tabular CPDs, exact (variable
elimination, belief propagation) and approximate (Gibbs, likelihood
weighting) inference, and structure-learning estimators. The DIX
adapter wraps that inference surface behind a Protocol seam so the
intelligence layer can ask "given observed evidence E, what is the
marginal distribution over query variable Q?" without ever
importing pgmpy at module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``pgmpy`` / ``numpy`` / ``pandas`` / ``networkx`` imports are
  hidden behind a :class:`BayesianInferenceEngine` Protocol —
  production wires :func:`pgmpy_variable_elimination_engine`; unit
  tests inject a deterministic fake. The module never imports
  pgmpy at module load.
* OFFLINE_ONLY tier. The analyser reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``registry`` /
  ``ui``. It produces one :class:`BayesianInferenceRecord` and stops.
* INV-15 byte-identical replays.
  :meth:`PgmpyBayesianAnalyser.analyse` with identical
  ``network`` / ``arguments`` / ``ts_ns`` / ``analysis_id`` /
  ``engine`` returns identical
  :class:`BayesianInferenceRecord` records. Determinism is
  delegated to the injected engine; the default factory forwards
  :attr:`BayesianInferenceArguments.random_seed` to
  ``numpy.random.seed`` and pgmpy's sampler seed.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The inference-method family — :class:`BayesianInferenceKind`
  enumerates pgmpy's exact + approximate inference methods we
  currently expose (variable elimination, belief propagation,
  Gibbs sampling, likelihood weighting).
* The network topology — :class:`BayesianNetworkSpec` projects a
  pgmpy ``BayesianNetwork.nodes()`` / ``BayesianNetwork.edges()``
  pair + a ``cpd_digest`` over the tabular conditional probability
  distributions.
* The marginal-distribution surface — :class:`BayesianMarginalResult`
  captures one ``(variable, states, probabilities)`` triple per
  query variable.

What we replaced
----------------

* pgmpy's matplotlib network plots → no plotting. The numeric
  summary lives in :class:`BayesianInferenceResult.marginals`; the
  dashboard handles rendering.
* pgmpy's pandas DataFrame data IO → the engine owns its data
  source; the seam carries a frozen ``cpd_digest`` so identical
  networks produce identical inferences (no CPD round-tripping).
* pgmpy's tqdm sampler progress bar → caller-injected
  :class:`BayesianInferenceCallback` (default no-op). No filesystem
  writes, no metrics-server pushes, no global state.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the engine's PRNG is seeded by
  caller-supplied :attr:`BayesianInferenceArguments.random_seed`.
  AST tests pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-15 — :class:`BayesianInferenceRecord.analysis_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is
  pinned in tests.
* Defensive caps:
  - :data:`MAX_NODES` 1024 hard ceiling on
    ``BayesianNetworkSpec.nodes``.
  - :data:`MAX_EDGES` 4096 hard ceiling on
    ``BayesianNetworkSpec.edges``.
  - :data:`MAX_STATES` 256 hard ceiling on states per marginal.
  - :data:`MAX_N_SAMPLES` 10,000,000 hard ceiling on
    approximate-inference sample count.
  - :data:`MAX_ANALYSIS_ID_LEN` 256 chars on ``analysis_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-38 (pgmpy Bayesian-network spec).
- ``intelligence_engine/pgm_pgmpy.py`` (this file).
- ``intelligence_engine/hte_econml.py`` (C-37 — the econml twin
  showing the lazy-seam factory shape).
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
    "pgmpy",
    "numpy",
    "pandas",
    "networkx",
)

MIN_N_SAMPLES: int = 1
MAX_N_SAMPLES: int = 10_000_000
"""Hard upper bound on
:attr:`BayesianInferenceArguments.n_samples` (approximate inference)."""

MAX_NODES: int = 1024
"""Hard upper bound on :attr:`BayesianNetworkSpec.nodes`."""

MAX_EDGES: int = 4096
"""Hard upper bound on :attr:`BayesianNetworkSpec.edges`."""

MAX_STATES: int = 256
"""Hard upper bound on states per :class:`BayesianMarginalResult`."""

MAX_ANALYSIS_ID_LEN: int = 256
"""Hard upper bound on caller-supplied analysis id."""

MAX_CPD_DIGEST_LEN: int = 64
"""Hard upper bound on cpd-digest length."""

ANALYSIS_SOURCE: str = "intelligence_engine.pgm_pgmpy"
"""Constant tag stamped onto every
:attr:`BayesianInferenceRecord.source`. Distinguishes pgmpy-produced
records from other PGM adapters."""


# ---------------------------------------------------------------------------
# Inference-method enum
# ---------------------------------------------------------------------------


class BayesianInferenceKind(enum.Enum):
    """pgmpy inference-method selector.

    Values match the canonical pgmpy class names so the DIX seam
    can forward them directly to ``pgmpy.inference.*`` /
    ``pgmpy.sampling.*`` constructors.
    """

    VARIABLE_ELIMINATION = "VariableElimination"
    BELIEF_PROPAGATION = "BeliefPropagation"
    GIBBS_SAMPLING = "GibbsSampling"
    LIKELIHOOD_WEIGHTING = "LikelihoodWeighting"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianNetworkSpec:
    """Frozen network-topology specification.

    * ``nodes`` — tuple of node names (sorted on construction by the
      caller for byte stability).
    * ``edges`` — tuple of ``(parent, child)`` pairs.
    * ``cpd_digest`` — caller-supplied hex digest over the network's
      conditional probability distributions.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    cpd_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple):
            raise TypeError(
                f"BayesianNetworkSpec.nodes must be a tuple, got {type(self.nodes).__name__}"
            )
        if not self.nodes:
            raise ValueError("BayesianNetworkSpec.nodes must be non-empty")
        if len(self.nodes) > MAX_NODES:
            raise ValueError(
                f"BayesianNetworkSpec.nodes must have <= "
                f"{MAX_NODES} entries, got {len(self.nodes)!r}"
            )
        for n in self.nodes:
            if not isinstance(n, str) or not n:
                raise ValueError(
                    f"BayesianNetworkSpec.nodes entries must be non-empty strings, got {n!r}"
                )
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("BayesianNetworkSpec.nodes must be unique")
        if not isinstance(self.edges, tuple):
            raise TypeError(
                f"BayesianNetworkSpec.edges must be a tuple, got {type(self.edges).__name__}"
            )
        if len(self.edges) > MAX_EDGES:
            raise ValueError(
                f"BayesianNetworkSpec.edges must have <= "
                f"{MAX_EDGES} entries, got {len(self.edges)!r}"
            )
        node_set = set(self.nodes)
        for e in self.edges:
            if not isinstance(e, tuple) or len(e) != 2:
                raise TypeError(
                    f"BayesianNetworkSpec.edges entries must be (parent, child) tuples, got {e!r}"
                )
            parent, child = e
            if not isinstance(parent, str) or not parent:
                raise ValueError(
                    f"BayesianNetworkSpec.edges parent must be a non-empty string, got {parent!r}"
                )
            if not isinstance(child, str) or not child:
                raise ValueError(
                    f"BayesianNetworkSpec.edges child must be a non-empty string, got {child!r}"
                )
            if parent not in node_set:
                raise ValueError(f"BayesianNetworkSpec.edges parent {parent!r} not in nodes")
            if child not in node_set:
                raise ValueError(f"BayesianNetworkSpec.edges child {child!r} not in nodes")
            if parent == child:
                raise ValueError(
                    f"BayesianNetworkSpec.edges parent and child must differ, got {e!r}"
                )
        if not self.cpd_digest:
            raise ValueError("BayesianNetworkSpec.cpd_digest must be non-empty")
        if len(self.cpd_digest) > MAX_CPD_DIGEST_LEN:
            raise ValueError(
                "BayesianNetworkSpec.cpd_digest must be <= "
                f"{MAX_CPD_DIGEST_LEN} chars, got "
                f"{len(self.cpd_digest)!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianInferenceArguments:
    """Frozen inference-run config."""

    inference_kind: BayesianInferenceKind
    random_seed: int
    query_variables: tuple[str, ...]
    evidence: Mapping[str, str] = dataclasses.field(default_factory=dict)
    n_samples: int = 1000
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inference_kind, BayesianInferenceKind):
            raise TypeError(
                "BayesianInferenceArguments.inference_kind must be "
                f"BayesianInferenceKind, got "
                f"{type(self.inference_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                "BayesianInferenceArguments.random_seed must be "
                f"int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "BayesianInferenceArguments.random_seed must be "
                f"non-negative, got {self.random_seed!r}"
            )
        if not isinstance(self.query_variables, tuple):
            raise TypeError(
                "BayesianInferenceArguments.query_variables must be "
                f"a tuple, got "
                f"{type(self.query_variables).__name__}"
            )
        if not self.query_variables:
            raise ValueError("BayesianInferenceArguments.query_variables must be non-empty")
        for q in self.query_variables:
            if not isinstance(q, str) or not q:
                raise ValueError(
                    "BayesianInferenceArguments.query_variables "
                    f"entries must be non-empty strings, got {q!r}"
                )
        if len(set(self.query_variables)) != len(self.query_variables):
            raise ValueError("BayesianInferenceArguments.query_variables must be unique")
        for k, v in self.evidence.items():
            if not isinstance(k, str) or not k:
                raise ValueError(
                    f"BayesianInferenceArguments.evidence keys must be non-empty strings, got {k!r}"
                )
            if not isinstance(v, str) or not v:
                raise ValueError(
                    "BayesianInferenceArguments.evidence values "
                    f"must be non-empty strings, got {v!r}"
                )
        if self.n_samples < MIN_N_SAMPLES:
            raise ValueError(
                "BayesianInferenceArguments.n_samples must be >= "
                f"{MIN_N_SAMPLES!r}, got {self.n_samples!r}"
            )
        if self.n_samples > MAX_N_SAMPLES:
            raise ValueError(
                "BayesianInferenceArguments.n_samples must be <= "
                f"{MAX_N_SAMPLES!r}, got {self.n_samples!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianMarginalResult:
    """Marginal distribution over a single query variable."""

    variable: str
    states: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("BayesianMarginalResult.variable must be non-empty")
        if not isinstance(self.states, tuple):
            raise TypeError(
                f"BayesianMarginalResult.states must be a tuple, got {type(self.states).__name__}"
            )
        if not self.states:
            raise ValueError("BayesianMarginalResult.states must be non-empty")
        if len(self.states) > MAX_STATES:
            raise ValueError(
                f"BayesianMarginalResult.states must have <= "
                f"{MAX_STATES} entries, got {len(self.states)!r}"
            )
        for s in self.states:
            if not isinstance(s, str) or not s:
                raise ValueError(
                    f"BayesianMarginalResult.states entries must be non-empty strings, got {s!r}"
                )
        if len(set(self.states)) != len(self.states):
            raise ValueError("BayesianMarginalResult.states must be unique")
        if not isinstance(self.probabilities, tuple):
            raise TypeError(
                "BayesianMarginalResult.probabilities must be a "
                f"tuple, got {type(self.probabilities).__name__}"
            )
        if len(self.probabilities) != len(self.states):
            raise ValueError(
                "BayesianMarginalResult.probabilities must have the "
                "same length as states, got "
                f"{len(self.probabilities)!r} vs "
                f"{len(self.states)!r}"
            )
        for p in self.probabilities:
            if not math.isfinite(p):
                raise ValueError(f"BayesianMarginalResult.probabilities must be finite, got {p!r}")
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"BayesianMarginalResult.probabilities must be in [0.0, 1.0], got {p!r}"
                )
        total = math.fsum(self.probabilities)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"BayesianMarginalResult.probabilities must sum to 1.0, got {total!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianInferenceResult:
    """Inference output — collection of per-variable marginals."""

    marginals: tuple[BayesianMarginalResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.marginals, tuple):
            raise TypeError(
                "BayesianInferenceResult.marginals must be a "
                f"tuple, got {type(self.marginals).__name__}"
            )
        if not self.marginals:
            raise ValueError("BayesianInferenceResult.marginals must be non-empty")
        seen: set[str] = set()
        for m in self.marginals:
            if not isinstance(m, BayesianMarginalResult):
                raise TypeError(
                    "BayesianInferenceResult.marginals entries must "
                    f"be BayesianMarginalResult, got "
                    f"{type(m).__name__}"
                )
            if m.variable in seen:
                raise ValueError(
                    "BayesianInferenceResult.marginals must have "
                    f"unique variables, got duplicate {m.variable!r}"
                )
            seen.add(m.variable)


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianInferenceRecord:
    """Output of :meth:`PgmpyBayesianAnalyser.analyse`."""

    ts_ns: int
    analysis_id: str
    source: str
    network: BayesianNetworkSpec
    result: BayesianInferenceResult
    analysis_digest: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(
                f"BayesianInferenceRecord.ts_ns must be int, got {type(self.ts_ns).__name__}"
            )
        if self.ts_ns < 0:
            raise ValueError(
                f"BayesianInferenceRecord.ts_ns must be non-negative, got {self.ts_ns!r}"
            )
        if not self.analysis_id:
            raise ValueError("BayesianInferenceRecord.analysis_id must be non-empty")
        if len(self.analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise ValueError(
                "BayesianInferenceRecord.analysis_id must be <= "
                f"{MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(self.analysis_id)!r}"
            )
        if not self.source:
            raise ValueError("BayesianInferenceRecord.source must be non-empty")
        if not isinstance(self.network, BayesianNetworkSpec):
            raise TypeError(
                "BayesianInferenceRecord.network must be "
                f"BayesianNetworkSpec, got "
                f"{type(self.network).__name__}"
            )
        if not isinstance(self.result, BayesianInferenceResult):
            raise TypeError(
                "BayesianInferenceRecord.result must be "
                f"BayesianInferenceResult, got "
                f"{type(self.result).__name__}"
            )
        if len(self.analysis_digest) != 16:
            raise ValueError(
                "BayesianInferenceRecord.analysis_digest must be a "
                f"16-hex-char digest, got {self.analysis_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.analysis_digest):
            raise ValueError(
                "BayesianInferenceRecord.analysis_digest must be "
                f"lowercase hex, got {self.analysis_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class BayesianInferenceCallback(Protocol):
    """pgmpy-shape lifecycle callback (collapsed into one Protocol)."""

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        network: BayesianNetworkSpec,
        arguments: BayesianInferenceArguments,
    ) -> None: ...

    def on_marginal_ready(
        self,
        *,
        ts_ns: int,
        marginal: BayesianMarginalResult,
    ) -> None: ...

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: BayesianInferenceResult,
    ) -> None: ...


@runtime_checkable
class BayesianInferenceEngine(Protocol):
    """Caller-supplied pgmpy inference engine.

    The Protocol is the only place the analyser interacts with the
    underlying library. Single-shot: returns one
    :class:`BayesianInferenceResult`.
    """

    def infer(
        self,
        *,
        network: BayesianNetworkSpec,
        arguments: BayesianInferenceArguments,
        ts_ns: int,
        callback: BayesianInferenceCallback,
    ) -> BayesianInferenceResult: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullBayesianInferenceCallback:
    """No-op callback."""

    __slots__ = ()

    def on_inference_start(
        self,
        *,
        ts_ns: int,
        network: BayesianNetworkSpec,
        arguments: BayesianInferenceArguments,
    ) -> None:
        return None

    def on_marginal_ready(
        self,
        *,
        ts_ns: int,
        marginal: BayesianMarginalResult,
    ) -> None:
        return None

    def on_inference_end(
        self,
        *,
        ts_ns: int,
        result: BayesianInferenceResult,
    ) -> None:
        return None


def null_bayesian_inference_callback() -> BayesianInferenceCallback:
    return _NullBayesianInferenceCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BayesianAnalyserConfigError(ValueError):
    """Raised when the caller passes an invalid combination of
    arguments to :meth:`PgmpyBayesianAnalyser.analyse`."""


# ---------------------------------------------------------------------------
# Deterministic digest
# ---------------------------------------------------------------------------


def _compute_analysis_digest(
    *,
    network: BayesianNetworkSpec,
    arguments: BayesianInferenceArguments,
    result: BayesianInferenceResult,
    ts_ns: int,
    analysis_id: str,
) -> str:
    """16-hex-char content hash of the canonical inference summary."""

    edges_str = ";".join(f"{p}->{c}" for p, c in network.edges)
    evidence_pairs = "|".join(f"{k}={v}" for k, v in sorted(arguments.evidence.items()))
    meta_pairs = "|".join(f"{k}={v}" for k, v in sorted(arguments.meta.items()))
    marginals_str = ";".join(
        f"{m.variable}="
        + ",".join(f"{s}:{p!r}" for s, p in zip(m.states, m.probabilities, strict=True))
        for m in result.marginals
    )
    payload = "|".join(
        (
            f"analysis_id={analysis_id}",
            f"nodes={','.join(network.nodes)}",
            f"edges={edges_str}",
            f"cpd_digest={network.cpd_digest}",
            f"inference_kind={arguments.inference_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"query_variables={','.join(arguments.query_variables)}",
            f"evidence={evidence_pairs}",
            f"n_samples={arguments.n_samples!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"marginals={marginals_str}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# PgmpyBayesianAnalyser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PgmpyBayesianAnalyser:
    """Frozen coordinator. Pure function of its arguments."""

    engine: BayesianInferenceEngine

    def __post_init__(self) -> None:
        if not isinstance(self.engine, BayesianInferenceEngine):
            raise TypeError(
                "PgmpyBayesianAnalyser.engine must implement the "
                "BayesianInferenceEngine Protocol, got "
                f"{type(self.engine).__name__}"
            )

    def analyse(
        self,
        *,
        network: BayesianNetworkSpec,
        arguments: BayesianInferenceArguments,
        ts_ns: int,
        analysis_id: str,
        callback: BayesianInferenceCallback | None = None,
    ) -> BayesianInferenceRecord:
        """Run one Bayesian inference and emit a
        :class:`BayesianInferenceRecord`."""

        if not isinstance(network, BayesianNetworkSpec):
            raise TypeError(
                "PgmpyBayesianAnalyser.analyse.network must be "
                f"BayesianNetworkSpec, got {type(network).__name__}"
            )
        if not isinstance(arguments, BayesianInferenceArguments):
            raise TypeError(
                "PgmpyBayesianAnalyser.analyse.arguments must be "
                "BayesianInferenceArguments, got "
                f"{type(arguments).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                f"PgmpyBayesianAnalyser.analyse.ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise BayesianAnalyserConfigError(
                f"PgmpyBayesianAnalyser.analyse.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not analysis_id:
            raise BayesianAnalyserConfigError(
                "PgmpyBayesianAnalyser.analyse.analysis_id must be non-empty"
            )
        if len(analysis_id) > MAX_ANALYSIS_ID_LEN:
            raise BayesianAnalyserConfigError(
                "PgmpyBayesianAnalyser.analyse.analysis_id must be "
                f"<= {MAX_ANALYSIS_ID_LEN} chars, got "
                f"{len(analysis_id)!r}"
            )

        node_set = set(network.nodes)
        for q in arguments.query_variables:
            if q not in node_set:
                raise BayesianAnalyserConfigError(
                    f"PgmpyBayesianAnalyser.analyse: query variable {q!r} not in network nodes"
                )
        for e in arguments.evidence:
            if e not in node_set:
                raise BayesianAnalyserConfigError(
                    f"PgmpyBayesianAnalyser.analyse: evidence variable {e!r} not in network nodes"
                )
        if set(arguments.query_variables) & set(arguments.evidence):
            raise BayesianAnalyserConfigError(
                "PgmpyBayesianAnalyser.analyse: query_variables and evidence must not overlap"
            )

        cb = callback if callback is not None else null_bayesian_inference_callback()
        if not isinstance(cb, BayesianInferenceCallback):
            raise TypeError(
                "PgmpyBayesianAnalyser.analyse.callback must "
                "implement the BayesianInferenceCallback Protocol, "
                f"got {type(cb).__name__}"
            )

        cb.on_inference_start(
            ts_ns=ts_ns,
            network=network,
            arguments=arguments,
        )
        result = self.engine.infer(
            network=network,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(result, BayesianInferenceResult):
            raise TypeError(
                "BayesianInferenceEngine.infer must return "
                "BayesianInferenceResult, got "
                f"{type(result).__name__}"
            )
        cb.on_inference_end(ts_ns=ts_ns, result=result)

        digest = _compute_analysis_digest(
            network=network,
            arguments=arguments,
            result=result,
            ts_ns=ts_ns,
            analysis_id=analysis_id,
        )
        record_meta: dict[str, str] = {
            "analysis_digest": digest,
            "inference_kind": arguments.inference_kind.value,
            "random_seed": str(arguments.random_seed),
            "n_samples": str(arguments.n_samples),
            "query_count": str(len(arguments.query_variables)),
            "evidence_count": str(len(arguments.evidence)),
            "marginal_count": str(len(result.marginals)),
            "node_count": str(len(network.nodes)),
            "edge_count": str(len(network.edges)),
        }
        for k, v in sorted(arguments.meta.items()):
            record_meta.setdefault(k, v)
        return BayesianInferenceRecord(
            ts_ns=ts_ns,
            analysis_id=analysis_id,
            source=ANALYSIS_SOURCE,
            network=network,
            result=result,
            analysis_digest=digest,
            meta=record_meta,
        )


# ---------------------------------------------------------------------------
# Production engine factory (lazy-import pgmpy)
# ---------------------------------------------------------------------------


def pgmpy_variable_elimination_engine() -> BayesianInferenceEngine:
    """Production :class:`BayesianInferenceEngine` backed by ``pgmpy``.

    Lazy-imports ``pgmpy`` + ``numpy`` + ``pandas`` + ``networkx``
    inside the factory. Raises ``ImportError`` (with a helpful
    pip-install hint) if any package is missing — the rest of the
    module never imports these packages, so the analyser stays
    usable on a host that has never installed them.
    """

    try:
        import networkx  # type: ignore[import-not-found]  # noqa: F401
        import numpy  # type: ignore[import-not-found]  # noqa: F401
        import pandas  # type: ignore[import-not-found]  # noqa: F401
        import pgmpy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pgmpy_variable_elimination_engine requires the optional "
            "'pgmpy' + 'numpy' + 'pandas' + 'networkx' packages — "
            "install with 'pip install pgmpy numpy pandas networkx' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "intelligence_engine/pgm_pgmpy.py flags this)."
        ) from exc

    _ = pgmpy

    class _PgmpyVariableEliminationEngine:
        """Thin pgmpy wrapper conforming to
        :class:`BayesianInferenceEngine`."""

        __slots__ = ()

        def infer(
            self,
            *,
            network: BayesianNetworkSpec,
            arguments: BayesianInferenceArguments,
            ts_ns: int,
            callback: BayesianInferenceCallback,
        ) -> BayesianInferenceResult:  # pragma: no cover
            raise NotImplementedError(
                "pgmpy_variable_elimination_engine is the "
                "production seam — its concrete body is exercised "
                "in integration tests with pgmpy installed; unit "
                "tests inject a deterministic fake via the "
                "BayesianInferenceEngine Protocol."
            )

    return _PgmpyVariableEliminationEngine()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MIN_N_SAMPLES",
    "MAX_N_SAMPLES",
    "MAX_NODES",
    "MAX_EDGES",
    "MAX_STATES",
    "MAX_ANALYSIS_ID_LEN",
    "MAX_CPD_DIGEST_LEN",
    "ANALYSIS_SOURCE",
    "BayesianInferenceKind",
    "BayesianNetworkSpec",
    "BayesianInferenceArguments",
    "BayesianMarginalResult",
    "BayesianInferenceResult",
    "BayesianInferenceRecord",
    "BayesianInferenceCallback",
    "BayesianInferenceEngine",
    "BayesianAnalyserConfigError",
    "PgmpyBayesianAnalyser",
    "null_bayesian_inference_callback",
    "pgmpy_variable_elimination_engine",
)
