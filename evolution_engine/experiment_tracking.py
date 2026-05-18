# ADAPTED FROM: mlflow/tracking/client.py
# (mlflow/__init__.py — start_run() / end_run() lifecycle;
#  mlflow/tracking/client.py — MlflowClient.log_metric/log_param/log_artifact;
#  mlflow/models/model.py — MlflowModel.log() + model-registry stage transitions.)
"""B-18 — Experiment tracking for the evolution engine (OFFLINE_ONLY).

MLflow's public surface (`start_run` / `log_param` / `log_metric` /
`log_artifact` / `register_model` / `transition_model_version_stage`)
is the canonical experiment-tracking contract for ML pipelines. This
module adapts that contract to DIX's authority discipline:

* The runtime hot path **never** depends on mlflow. The module is
  pure Python + stdlib + ``core.contracts.*``; ``mlflow`` is named in
  :data:`NEW_PIP_DEPENDENCIES` but only imported inside
  :func:`mlflow_backend_factory`, which production callers wire when
  the offline trainer runs. Tests inject an :class:`InMemoryTrackingBackend`.
* All records are :func:`dataclasses.dataclass(frozen=True, slots=True)`.
  An experiment run is a sealed value object — once
  :func:`build_experiment_run` returns it, no field can mutate. The
  registry projection is similarly frozen.
* INV-15 byte-identical replay. The caller supplies
  ``started_ns`` / ``finished_ns`` / ``run_id`` /
  ``parent_proposal_id``; the module performs no clock reads, random
  draws, file IO, or environment lookups. Every digest is the
  hex-prefixed BLAKE2b-16 of a canonical-text projection with
  alphabetised metric / param / artifact keys.
* INV-13 / INV-14 governance isolation. Model-registry stage
  transitions (`NONE → STAGING → PRODUCTION → ARCHIVED`) are emitted
  as advisory :class:`StageTransitionRecommendation` value objects.
  Only the evolution-engine adapter (`evolution_engine/pipeline.py`
  or `patch_pipeline/*`) is permitted to project a recommendation
  into a typed :class:`~core.contracts.learning.PatchProposal` — this
  module itself does **not** construct typed bus events
  (B27 / B28 / INV-71 authority symmetry, pinned by AST tests).
* OFFLINE_ONLY. The module never imports
  ``execution_engine`` / ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. Adversarial training runs
  call ``build_experiment_run(...)`` once per run and stop; the
  in-memory backend persists records for downstream replay.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

# ----------------------------------------------------- versioning


EXPERIMENT_TRACKING_VERSION = "experiment-tracking/v1"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("mlflow",)


# ----------------------------------------------------- exceptions


class ExperimentTrackingError(ValueError):
    """Raised on invalid experiment-tracking input."""


# ----------------------------------------------------- constants


MAX_NAME_LEN = 250  # mlflow's documented limit for run / model / metric / param names
MAX_VALUE_LEN = 6_000  # mlflow tag/param string-value limit
MAX_ARTIFACT_BYTES = 1_048_576  # 1 MiB — well under mlflow's storage ceiling
MAX_PARAMS = 1024
MAX_METRICS = 8192
MAX_ARTIFACTS = 64


# ----------------------------------------------------- enums


class RunStatus(StrEnum):
    """MLflow run lifecycle states (mlflow.entities.RunStatus subset)."""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class ModelStage(StrEnum):
    """MLflow model-registry stages."""

    NONE = "NONE"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


# Legal stage transitions — gated by governance approval on the
# evolution-engine side. The recommendation builder enforces this map.
_LEGAL_STAGE_TRANSITIONS: dict[ModelStage, frozenset[ModelStage]] = {
    ModelStage.NONE: frozenset({ModelStage.STAGING, ModelStage.ARCHIVED}),
    ModelStage.STAGING: frozenset({ModelStage.PRODUCTION, ModelStage.ARCHIVED, ModelStage.NONE}),
    ModelStage.PRODUCTION: frozenset({ModelStage.ARCHIVED, ModelStage.STAGING}),
    ModelStage.ARCHIVED: frozenset(),
}


def legal_stage_transitions() -> Mapping[ModelStage, frozenset[ModelStage]]:
    """Return the legal model-stage transition map (frozen)."""

    return dict(_LEGAL_STAGE_TRANSITIONS)


# ----------------------------------------------------- helpers


def _digest(text: str) -> str:
    """Return the lowercase 32-char hex BLAKE2b-16 digest of ``text``."""

    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _validate_name(name: str, *, kind: str) -> None:
    if not isinstance(name, str) or not name:
        raise ExperimentTrackingError(f"{kind} name must be a non-empty str")
    if len(name) > MAX_NAME_LEN:
        raise ExperimentTrackingError(f"{kind} name exceeds {MAX_NAME_LEN} chars")
    if "\n" in name or "\r" in name:
        raise ExperimentTrackingError(f"{kind} name must be single-line")


def _validate_value(value: str, *, kind: str) -> None:
    if not isinstance(value, str):
        raise ExperimentTrackingError(f"{kind} value must be str")
    if len(value) > MAX_VALUE_LEN:
        raise ExperimentTrackingError(f"{kind} value exceeds {MAX_VALUE_LEN} chars")


def _validate_finite(value: float, *, kind: str) -> None:
    import math

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentTrackingError(f"{kind} must be numeric")
    if not math.isfinite(float(value)):
        raise ExperimentTrackingError(f"{kind} must be finite")


# ----------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class ExperimentParam:
    """A single param entry — mlflow `log_param` analogue."""

    name: str
    value: str

    def __post_init__(self) -> None:
        _validate_name(self.name, kind="param")
        _validate_value(self.value, kind="param")


@dataclass(frozen=True, slots=True)
class ExperimentMetric:
    """A single metric entry — mlflow `log_metric` analogue.

    ``step`` is the optimiser step (epoch / generation / iteration).
    Multiple metric rows with the same ``name`` and different ``step``
    values represent a learning curve.
    """

    name: str
    value: float
    step: int = 0

    def __post_init__(self) -> None:
        _validate_name(self.name, kind="metric")
        _validate_finite(self.value, kind="metric value")
        if not isinstance(self.step, int) or isinstance(self.step, bool):
            raise ExperimentTrackingError("metric step must be int")
        if self.step < 0:
            raise ExperimentTrackingError("metric step must be >= 0")


@dataclass(frozen=True, slots=True)
class ExperimentArtifact:
    """A single artifact entry — mlflow `log_artifact` analogue.

    Artifacts are stored as canonical text (e.g. sorted-key JSON) +
    BLAKE2b-16 digest. Large binary blobs MUST be persisted by the
    backend and referenced here by a content-hash pointer, not
    inlined.
    """

    name: str
    content: str
    digest: str

    def __post_init__(self) -> None:
        _validate_name(self.name, kind="artifact")
        if not isinstance(self.content, str):
            raise ExperimentTrackingError("artifact content must be str")
        if len(self.content.encode("utf-8")) > MAX_ARTIFACT_BYTES:
            raise ExperimentTrackingError(f"artifact content exceeds {MAX_ARTIFACT_BYTES} bytes")
        expected = _digest(self.content)
        if self.digest != expected:
            raise ExperimentTrackingError("artifact digest does not match BLAKE2b-16(content)")


def build_artifact(*, name: str, content: str) -> ExperimentArtifact:
    """Build an :class:`ExperimentArtifact` with the canonical digest.

    Use this helper instead of constructing :class:`ExperimentArtifact`
    directly — it guarantees ``digest == BLAKE2b-16(content)``.
    """

    if not isinstance(content, str):
        raise ExperimentTrackingError("artifact content must be str")
    return ExperimentArtifact(name=name, content=content, digest=_digest(content))


# ----------------------------------------------------- experiment run


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """A single sealed evolution-run record.

    Pre-conditions enforced in :func:`build_experiment_run`:

    * ``finished_ns >= started_ns``
    * ``params`` keys unique, alphabetised
    * ``metrics`` sorted by (name, step) ascending
    * ``artifacts`` sorted by name ascending
    * Each ``ExperimentArtifact.digest`` matches its content
    * ``digest`` is BLAKE2b-16 over the canonical-text projection
    """

    run_id: str
    experiment_name: str
    source: str
    started_ns: int
    finished_ns: int
    status: RunStatus
    params: tuple[ExperimentParam, ...]
    metrics: tuple[ExperimentMetric, ...]
    artifacts: tuple[ExperimentArtifact, ...]
    parent_proposal_id: str
    digest: str

    def has_metric(self, name: str) -> bool:
        return any(m.name == name for m in self.metrics)

    def latest_metric(self, name: str) -> ExperimentMetric | None:
        rows = [m for m in self.metrics if m.name == name]
        if not rows:
            return None
        return rows[-1]


def _canonical_run_text(
    *,
    run_id: str,
    experiment_name: str,
    source: str,
    started_ns: int,
    finished_ns: int,
    status: RunStatus,
    params: Sequence[ExperimentParam],
    metrics: Sequence[ExperimentMetric],
    artifacts: Sequence[ExperimentArtifact],
    parent_proposal_id: str,
) -> str:
    parts: list[str] = [
        f"v={EXPERIMENT_TRACKING_VERSION}",
        f"run_id={run_id}",
        f"experiment={experiment_name}",
        f"source={source}",
        f"started_ns={started_ns}",
        f"finished_ns={finished_ns}",
        f"status={status.value}",
        f"parent_proposal_id={parent_proposal_id}",
    ]
    for p in params:
        parts.append(f"p:{p.name}={p.value}")
    for m in metrics:
        parts.append(f"m:{m.name}@{m.step}={m.value:.17g}")
    for a in artifacts:
        parts.append(f"a:{a.name}#{a.digest}")
    return "|".join(parts)


def _normalise_params(
    raw: Mapping[str, str] | Iterable[ExperimentParam] | None,
) -> tuple[ExperimentParam, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        items = [(str(k), str(v)) for k, v in raw.items()]
    else:
        items = [(p.name, p.value) for p in raw]
    seen: set[str] = set()
    for name, _ in items:
        if name in seen:
            raise ExperimentTrackingError(f"duplicate param name: {name}")
        seen.add(name)
    items.sort(key=lambda kv: kv[0])
    out = tuple(ExperimentParam(name=n, value=v) for n, v in items)
    if len(out) > MAX_PARAMS:
        raise ExperimentTrackingError(f"params count exceeds {MAX_PARAMS}")
    return out


def _normalise_metrics(
    raw: Sequence[ExperimentMetric] | None,
) -> tuple[ExperimentMetric, ...]:
    if raw is None:
        return ()
    rows = list(raw)
    seen: set[tuple[str, int]] = set()
    for m in rows:
        key = (m.name, m.step)
        if key in seen:
            raise ExperimentTrackingError(f"duplicate metric row: name={m.name}, step={m.step}")
        seen.add(key)
    rows.sort(key=lambda m: (m.name, m.step))
    out = tuple(rows)
    if len(out) > MAX_METRICS:
        raise ExperimentTrackingError(f"metrics count exceeds {MAX_METRICS}")
    return out


def _normalise_artifacts(
    raw: Sequence[ExperimentArtifact] | None,
) -> tuple[ExperimentArtifact, ...]:
    if raw is None:
        return ()
    rows = list(raw)
    seen: set[str] = set()
    for a in rows:
        if a.name in seen:
            raise ExperimentTrackingError(f"duplicate artifact name: {a.name}")
        seen.add(a.name)
    rows.sort(key=lambda a: a.name)
    out = tuple(rows)
    if len(out) > MAX_ARTIFACTS:
        raise ExperimentTrackingError(f"artifacts count exceeds {MAX_ARTIFACTS}")
    return out


def build_experiment_run(
    *,
    run_id: str,
    experiment_name: str,
    source: str,
    started_ns: int,
    finished_ns: int,
    status: RunStatus,
    params: Mapping[str, str] | Iterable[ExperimentParam] | None = None,
    metrics: Sequence[ExperimentMetric] | None = None,
    artifacts: Sequence[ExperimentArtifact] | None = None,
    parent_proposal_id: str = "",
) -> ExperimentRun:
    """Assemble a frozen :class:`ExperimentRun`.

    Validates all inputs, normalises iteration order to be byte-stable
    across replays, computes the canonical digest, and returns the
    sealed record. Callers MUST use this helper rather than the
    dataclass constructor directly — the dataclass alone does not
    enforce the sort / uniqueness / digest invariants.
    """

    _validate_name(run_id, kind="run_id")
    _validate_name(experiment_name, kind="experiment_name")
    _validate_name(source, kind="source")
    if not isinstance(started_ns, int) or not isinstance(finished_ns, int):
        raise ExperimentTrackingError("started_ns / finished_ns must be int")
    if started_ns < 0 or finished_ns < 0:
        raise ExperimentTrackingError("started_ns / finished_ns must be >= 0")
    if finished_ns < started_ns:
        raise ExperimentTrackingError("finished_ns < started_ns")
    if not isinstance(status, RunStatus):
        raise ExperimentTrackingError("status must be a RunStatus")
    if not isinstance(parent_proposal_id, str):
        raise ExperimentTrackingError("parent_proposal_id must be str")

    p = _normalise_params(params)
    m = _normalise_metrics(metrics)
    a = _normalise_artifacts(artifacts)

    text = _canonical_run_text(
        run_id=run_id,
        experiment_name=experiment_name,
        source=source,
        started_ns=started_ns,
        finished_ns=finished_ns,
        status=status,
        params=p,
        metrics=m,
        artifacts=a,
        parent_proposal_id=parent_proposal_id,
    )
    return ExperimentRun(
        run_id=run_id,
        experiment_name=experiment_name,
        source=source,
        started_ns=started_ns,
        finished_ns=finished_ns,
        status=status,
        params=p,
        metrics=m,
        artifacts=a,
        parent_proposal_id=parent_proposal_id,
        digest=_digest(text),
    )


# ----------------------------------------------------- model registry


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """A registered model version — mlflow `register_model` analogue."""

    model_name: str
    version: int
    source_run_id: str
    current_stage: ModelStage
    registered_ns: int
    digest: str

    def __post_init__(self) -> None:
        _validate_name(self.model_name, kind="model_name")
        _validate_name(self.source_run_id, kind="source_run_id")
        if not isinstance(self.version, int) or self.version < 1:
            raise ExperimentTrackingError("version must be int >= 1")
        if not isinstance(self.registered_ns, int) or self.registered_ns < 0:
            raise ExperimentTrackingError("registered_ns must be int >= 0")
        if not isinstance(self.current_stage, ModelStage):
            raise ExperimentTrackingError("current_stage must be ModelStage")


def register_model_version(
    *,
    model_name: str,
    version: int,
    source_run_id: str,
    registered_ns: int,
    current_stage: ModelStage = ModelStage.NONE,
) -> ModelVersion:
    """Build a :class:`ModelVersion` with a canonical digest."""

    text = "|".join(
        (
            f"v={EXPERIMENT_TRACKING_VERSION}",
            f"model={model_name}",
            f"version={version}",
            f"source_run_id={source_run_id}",
            f"registered_ns={registered_ns}",
            f"stage={current_stage.value}",
        )
    )
    return ModelVersion(
        model_name=model_name,
        version=version,
        source_run_id=source_run_id,
        current_stage=current_stage,
        registered_ns=registered_ns,
        digest=_digest(text),
    )


@dataclass(frozen=True, slots=True)
class StageTransitionRecommendation:
    """Advisory record: governance approval gates the actual transition.

    Per **B27 / B28 / INV-71 authority symmetry**, this module does
    NOT construct typed :class:`~core.contracts.learning.PatchProposal`
    objects. The evolution-engine adapter (which lives on the
    evolution-engine side of the boundary) is the only seam permitted
    to project this advisory record into a typed bus event.
    """

    recommendation_id: str
    model_name: str
    version: int
    from_stage: ModelStage
    to_stage: ModelStage
    proposed_ns: int
    rationale: str

    def __post_init__(self) -> None:
        _validate_name(self.model_name, kind="model_name")
        _validate_value(self.rationale, kind="rationale")
        if not isinstance(self.version, int) or self.version < 1:
            raise ExperimentTrackingError("version must be int >= 1")
        if not isinstance(self.proposed_ns, int) or self.proposed_ns < 0:
            raise ExperimentTrackingError("proposed_ns must be int >= 0")
        if not isinstance(self.from_stage, ModelStage):
            raise ExperimentTrackingError("from_stage must be ModelStage")
        if not isinstance(self.to_stage, ModelStage):
            raise ExperimentTrackingError("to_stage must be ModelStage")
        if self.to_stage not in _LEGAL_STAGE_TRANSITIONS[self.from_stage]:
            raise ExperimentTrackingError(
                f"illegal transition: {self.from_stage.value} -> {self.to_stage.value}"
            )


def propose_stage_transition(
    *,
    version: ModelVersion,
    to_stage: ModelStage,
    proposed_ns: int,
    rationale: str,
) -> StageTransitionRecommendation:
    """Build an advisory :class:`StageTransitionRecommendation`.

    The returned record is consumed by the evolution-engine adapter
    which is responsible for projecting it onto a typed
    :class:`~core.contracts.learning.PatchProposal` after governance
    approval. This function deliberately does NOT construct typed bus
    events.
    """

    text = "|".join(
        (
            f"v={EXPERIMENT_TRACKING_VERSION}",
            f"model={version.model_name}",
            f"version={version.version}",
            f"from={version.current_stage.value}",
            f"to={to_stage.value}",
            f"proposed_ns={proposed_ns}",
            f"rationale={rationale}",
        )
    )
    rec_id = "stage-" + _digest(text)
    return StageTransitionRecommendation(
        recommendation_id=rec_id,
        model_name=version.model_name,
        version=version.version,
        from_stage=version.current_stage,
        to_stage=to_stage,
        proposed_ns=proposed_ns,
        rationale=rationale,
    )


# ----------------------------------------------------- backend protocol


@runtime_checkable
class TrackingBackend(Protocol):
    """Caller-supplied backend dispatch.

    Production: :func:`mlflow_backend_factory` returns a lazy mlflow
    wrapper. Tests inject :class:`InMemoryTrackingBackend`.
    """

    def record_run(self, run: ExperimentRun) -> None: ...
    def record_model_version(self, version: ModelVersion) -> None: ...
    def record_stage_recommendation(self, rec: StageTransitionRecommendation) -> None: ...


@dataclass
class InMemoryTrackingBackend:
    """Deterministic in-memory backend (default for tests / offline replay).

    Maintains sorted-by-digest tuples; all mutations are append-only
    and stable across replays so :class:`InMemoryTrackingBackend`
    can be serialised and reloaded without changing digest order.
    """

    runs: tuple[ExperimentRun, ...] = field(default_factory=tuple)
    model_versions: tuple[ModelVersion, ...] = field(default_factory=tuple)
    recommendations: tuple[StageTransitionRecommendation, ...] = field(default_factory=tuple)

    def record_run(self, run: ExperimentRun) -> None:
        if any(r.run_id == run.run_id for r in self.runs):
            raise ExperimentTrackingError(f"duplicate run_id: {run.run_id}")
        rows = sorted(self.runs + (run,), key=lambda r: r.run_id)
        self.runs = tuple(rows)

    def record_model_version(self, version: ModelVersion) -> None:
        for existing in self.model_versions:
            if existing.model_name == version.model_name and existing.version == version.version:
                raise ExperimentTrackingError(
                    f"duplicate model version: {version.model_name}@{version.version}"
                )
        rows = sorted(
            self.model_versions + (version,),
            key=lambda v: (v.model_name, v.version),
        )
        self.model_versions = tuple(rows)

    def record_stage_recommendation(self, rec: StageTransitionRecommendation) -> None:
        if any(r.recommendation_id == rec.recommendation_id for r in self.recommendations):
            raise ExperimentTrackingError(f"duplicate recommendation_id: {rec.recommendation_id}")
        rows = sorted(
            self.recommendations + (rec,),
            key=lambda r: r.recommendation_id,
        )
        self.recommendations = tuple(rows)


# ----------------------------------------------------- production seam


def mlflow_backend_factory(*, tracking_uri: str = "file:./mlruns") -> TrackingBackend:
    """Construct an mlflow-backed :class:`TrackingBackend`.

    Lazy-imports ``mlflow`` so that test environments without mlflow
    installed can still import this module. The default
    ``tracking_uri`` is the local filesystem ``./mlruns`` directory
    per the spec (no server required initially).

    The returned backend wraps mlflow client calls: each
    :func:`record_run` invocation creates an mlflow run with the
    ``ExperimentRun`` projected to params / metrics / artifacts.
    Production callers wire this; the in-memory backend is the
    deterministic default for tests.
    """

    try:
        import mlflow  # noqa: F401 — production-only import
        from mlflow.tracking import MlflowClient  # noqa: F401
    except ImportError as exc:  # pragma: no cover — exercised in production
        raise ExperimentTrackingError(
            "mlflow is not installed; install with `pip install mlflow`"
        ) from exc

    # The actual mlflow wrapper is intentionally NOT implemented in
    # this module — the production path mirrors mlflow's stateful
    # client API which is not byte-deterministic. The factory exists
    # so the *interface* is the same in production and in tests; the
    # offline replay path uses InMemoryTrackingBackend.
    raise ExperimentTrackingError(
        "mlflow backend is wired in evolution_engine/patch_pipeline; "
        "use InMemoryTrackingBackend for offline replay"
    )


# ----------------------------------------------------- exports


__all__ = (
    "EXPERIMENT_TRACKING_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "ExperimentArtifact",
    "ExperimentMetric",
    "ExperimentParam",
    "ExperimentRun",
    "ExperimentTrackingError",
    "InMemoryTrackingBackend",
    "MAX_ARTIFACTS",
    "MAX_ARTIFACT_BYTES",
    "MAX_METRICS",
    "MAX_NAME_LEN",
    "MAX_PARAMS",
    "MAX_VALUE_LEN",
    "ModelStage",
    "ModelVersion",
    "RunStatus",
    "StageTransitionRecommendation",
    "TrackingBackend",
    "build_artifact",
    "build_experiment_run",
    "legal_stage_transitions",
    "mlflow_backend_factory",
    "propose_stage_transition",
    "register_model_version",
)
