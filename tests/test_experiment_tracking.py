"""Tests for B-18 — `evolution_engine/experiment_tracking.py`.

Authority + contract validation. Covers:

* `# ADAPTED FROM:` header pinned
* `NEW_PIP_DEPENDENCIES = ("mlflow",)` and no top-level mlflow import
* No engine cross-imports (B1)
* No typed-event construction (B27 / B28 / INV-71 authority symmetry)
* No `random` / `time` / `datetime` / `asyncio` / `os` imports (INV-15)
* Freezing on every record dataclass
* Validation of names / values / step / artifact digest / finiteness
* Param / metric / artifact normalisation (sort order, uniqueness)
* `build_experiment_run` digest stability across 3 runs
* Param-dict-insertion-order independence
* Model-stage transition map (legal vs illegal)
* `InMemoryTrackingBackend` deterministic accumulation
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from evolution_engine import experiment_tracking as et
from evolution_engine.experiment_tracking import (
    EXPERIMENT_TRACKING_VERSION,
    NEW_PIP_DEPENDENCIES,
    ExperimentArtifact,
    ExperimentMetric,
    ExperimentParam,
    ExperimentRun,
    ExperimentTrackingError,
    InMemoryTrackingBackend,
    ModelStage,
    ModelVersion,
    RunStatus,
    TrackingBackend,
    build_artifact,
    build_experiment_run,
    legal_stage_transitions,
    propose_stage_transition,
    register_model_version,
)

_MODULE_PATH = Path(et.__file__)
_MODULE_TEXT = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_TEXT)


# ============================================================
# Authority pins
# ============================================================


def test_authority_adapted_from_header() -> None:
    assert _MODULE_TEXT.startswith("# ADAPTED FROM: mlflow/tracking/client.py")


def test_authority_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("mlflow",)


def test_authority_no_top_level_mlflow_import() -> None:
    """mlflow may only be imported inside the production factory."""

    for node in ast.iter_child_nodes(_MODULE_AST):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            assert "mlflow" not in text, f"top-level mlflow import: {text}"


def test_authority_no_forbidden_runtime_imports() -> None:
    """No clock / random / IO imports anywhere in the module (INV-15)."""

    forbidden = {
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "websockets",
        "numpy",
        "torch",
        "polars",
        "pandas",
        "langsmith",
    }
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in forbidden, f"forbidden from-import: {node.module}"


def test_authority_no_engine_cross_imports() -> None:
    """B1: no engine cross-imports (this lives in evolution_engine)."""

    forbidden_modules = (
        "governance_engine",
        "execution_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
        "dashboard_backend",
        "dashboard",
    )
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for forbidden in forbidden_modules:
                assert not mod.startswith(forbidden), f"forbidden cross-import: {node.module}"


def test_authority_no_typed_event_construction() -> None:
    """B27 / B28 / INV-71: this module must NOT construct PatchProposal.

    Stage transitions emit advisory :class:`StageTransitionRecommendation`
    records. Only the evolution-engine adapter (which lives on the
    evolution-engine side of the authority boundary) may project them
    onto typed bus events.
    """

    forbidden_constructors = (
        "PatchProposal",
        "LearningUpdate",
        "SignalEvent",
        "GovernanceDecision",
        "HazardEvent",
    )
    for node in ast.walk(_MODULE_AST):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else "")
            )
            assert name not in forbidden_constructors, f"forbidden constructor call: {name}"


# ============================================================
# Versioning
# ============================================================


def test_version_string_is_pinned() -> None:
    assert EXPERIMENT_TRACKING_VERSION == "experiment-tracking/v1"


# ============================================================
# Freezing
# ============================================================


@pytest.mark.parametrize(
    "instance, field_name",
    [
        (ExperimentParam(name="lr", value="0.01"), "value"),
        (ExperimentMetric(name="loss", value=0.1, step=0), "value"),
        (
            ExperimentArtifact(
                name="cfg",
                content="{}",
                digest=et._digest("{}"),
            ),
            "content",
        ),
    ],
)
def test_freezing_records(instance: object, field_name: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, "x")


def test_freezing_experiment_run() -> None:
    run = build_experiment_run(
        run_id="r1",
        experiment_name="exp1",
        source="sb3",
        started_ns=0,
        finished_ns=1,
        status=RunStatus.FINISHED,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.status = RunStatus.FAILED  # type: ignore[misc]


def test_freezing_model_version() -> None:
    v = register_model_version(
        model_name="m1",
        version=1,
        source_run_id="r1",
        registered_ns=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.current_stage = ModelStage.PRODUCTION  # type: ignore[misc]


def test_freezing_stage_recommendation() -> None:
    v = register_model_version(model_name="m1", version=1, source_run_id="r1", registered_ns=0)
    rec = propose_stage_transition(
        version=v,
        to_stage=ModelStage.STAGING,
        proposed_ns=1,
        rationale="ok",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.to_stage = ModelStage.PRODUCTION  # type: ignore[misc]


# ============================================================
# Validation — params / metrics / artifacts
# ============================================================


def test_param_rejects_empty_name() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentParam(name="", value="v")


def test_param_rejects_overlong_name() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentParam(name="x" * 251, value="v")


def test_param_rejects_overlong_value() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentParam(name="k", value="v" * 6001)


def test_metric_rejects_nan() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentMetric(name="loss", value=float("nan"), step=0)


def test_metric_rejects_inf() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentMetric(name="loss", value=math.inf, step=0)


def test_metric_rejects_negative_step() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentMetric(name="loss", value=0.5, step=-1)


def test_metric_rejects_bool_step() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentMetric(name="loss", value=0.5, step=True)


def test_artifact_rejects_wrong_digest() -> None:
    with pytest.raises(ExperimentTrackingError):
        ExperimentArtifact(name="cfg", content="abc", digest="0" * 32)


def test_artifact_rejects_overlarge_content() -> None:
    with pytest.raises(ExperimentTrackingError):
        big = "x" * (et.MAX_ARTIFACT_BYTES + 1)
        build_artifact(name="big", content=big)


def test_build_artifact_canonical_digest() -> None:
    a = build_artifact(name="cfg", content="hello")
    assert a.digest == et._digest("hello")
    assert len(a.digest) == 32


# ============================================================
# Run validation
# ============================================================


def test_build_run_rejects_finished_before_started() -> None:
    with pytest.raises(ExperimentTrackingError):
        build_experiment_run(
            run_id="r1",
            experiment_name="exp",
            source="sb3",
            started_ns=5,
            finished_ns=4,
            status=RunStatus.FINISHED,
        )


def test_build_run_rejects_negative_timestamps() -> None:
    with pytest.raises(ExperimentTrackingError):
        build_experiment_run(
            run_id="r1",
            experiment_name="exp",
            source="sb3",
            started_ns=-1,
            finished_ns=0,
            status=RunStatus.FINISHED,
        )


def test_build_run_rejects_duplicate_param() -> None:
    with pytest.raises(ExperimentTrackingError):
        build_experiment_run(
            run_id="r1",
            experiment_name="exp",
            source="sb3",
            started_ns=0,
            finished_ns=1,
            status=RunStatus.FINISHED,
            params=(
                ExperimentParam(name="lr", value="0.01"),
                ExperimentParam(name="lr", value="0.02"),
            ),
        )


def test_build_run_rejects_duplicate_metric_row() -> None:
    with pytest.raises(ExperimentTrackingError):
        build_experiment_run(
            run_id="r1",
            experiment_name="exp",
            source="sb3",
            started_ns=0,
            finished_ns=1,
            status=RunStatus.FINISHED,
            metrics=(
                ExperimentMetric(name="loss", value=0.5, step=0),
                ExperimentMetric(name="loss", value=0.6, step=0),
            ),
        )


def test_build_run_rejects_duplicate_artifact() -> None:
    a1 = build_artifact(name="cfg", content="a")
    a2 = build_artifact(name="cfg", content="b")
    with pytest.raises(ExperimentTrackingError):
        build_experiment_run(
            run_id="r1",
            experiment_name="exp",
            source="sb3",
            started_ns=0,
            finished_ns=1,
            status=RunStatus.FINISHED,
            artifacts=(a1, a2),
        )


def test_build_run_rejects_bad_status_type() -> None:
    with pytest.raises(ExperimentTrackingError):
        build_experiment_run(
            run_id="r1",
            experiment_name="exp",
            source="sb3",
            started_ns=0,
            finished_ns=1,
            status="FINISHED",  # type: ignore[arg-type]
        )


# ============================================================
# Normalisation
# ============================================================


def test_params_sorted_by_name() -> None:
    run = build_experiment_run(
        run_id="r1",
        experiment_name="exp",
        source="sb3",
        started_ns=0,
        finished_ns=1,
        status=RunStatus.FINISHED,
        params={"zeta": "1", "alpha": "2", "mu": "3"},
    )
    assert tuple(p.name for p in run.params) == ("alpha", "mu", "zeta")


def test_metrics_sorted_by_name_then_step() -> None:
    run = build_experiment_run(
        run_id="r1",
        experiment_name="exp",
        source="sb3",
        started_ns=0,
        finished_ns=1,
        status=RunStatus.FINISHED,
        metrics=(
            ExperimentMetric(name="loss", value=0.3, step=2),
            ExperimentMetric(name="acc", value=0.9, step=1),
            ExperimentMetric(name="loss", value=0.5, step=1),
            ExperimentMetric(name="acc", value=0.8, step=0),
        ),
    )
    keys = tuple((m.name, m.step) for m in run.metrics)
    assert keys == (("acc", 0), ("acc", 1), ("loss", 1), ("loss", 2))


def test_artifacts_sorted_by_name() -> None:
    a1 = build_artifact(name="z", content="x")
    a2 = build_artifact(name="a", content="y")
    run = build_experiment_run(
        run_id="r1",
        experiment_name="exp",
        source="sb3",
        started_ns=0,
        finished_ns=1,
        status=RunStatus.FINISHED,
        artifacts=(a1, a2),
    )
    assert tuple(a.name for a in run.artifacts) == ("a", "z")


# ============================================================
# Digest determinism (INV-15)
# ============================================================


def _sample_run(**overrides: object) -> ExperimentRun:
    kwargs = {
        "run_id": "r1",
        "experiment_name": "ppo_v3",
        "source": "sb3",
        "started_ns": 1_000,
        "finished_ns": 2_000,
        "status": RunStatus.FINISHED,
        "params": {"lr": "0.01", "gamma": "0.99"},
        "metrics": (
            ExperimentMetric(name="loss", value=0.42, step=0),
            ExperimentMetric(name="loss", value=0.31, step=1),
        ),
        "artifacts": (build_artifact(name="cfg", content="hello"),),
        "parent_proposal_id": "patch-001",
    }
    kwargs.update(overrides)
    return build_experiment_run(**kwargs)  # type: ignore[arg-type]


def test_run_digest_three_run_equality() -> None:
    a = _sample_run()
    b = _sample_run()
    c = _sample_run()
    assert a == b == c
    assert a.digest == b.digest == c.digest
    assert len(a.digest) == 32


def test_run_digest_changes_when_metric_changes() -> None:
    a = _sample_run()
    b = _sample_run(
        metrics=(
            ExperimentMetric(name="loss", value=0.42, step=0),
            ExperimentMetric(name="loss", value=0.31, step=2),  # step changed
        )
    )
    assert a.digest != b.digest


def test_run_digest_changes_when_param_changes() -> None:
    a = _sample_run()
    b = _sample_run(params={"lr": "0.02", "gamma": "0.99"})
    assert a.digest != b.digest


def test_run_digest_is_param_dict_order_independent() -> None:
    a = _sample_run(params={"lr": "0.01", "gamma": "0.99"})
    b = _sample_run(params={"gamma": "0.99", "lr": "0.01"})
    assert a.digest == b.digest


def test_run_digest_changes_when_status_changes() -> None:
    a = _sample_run(status=RunStatus.FINISHED)
    b = _sample_run(status=RunStatus.FAILED)
    assert a.digest != b.digest


def test_run_digest_changes_when_artifact_content_changes() -> None:
    a = _sample_run(artifacts=(build_artifact(name="cfg", content="hello"),))
    b = _sample_run(artifacts=(build_artifact(name="cfg", content="world"),))
    assert a.digest != b.digest


# ============================================================
# Lookup helpers
# ============================================================


def test_has_metric_and_latest_metric() -> None:
    run = _sample_run()
    assert run.has_metric("loss")
    assert not run.has_metric("accuracy")
    latest = run.latest_metric("loss")
    assert latest is not None
    assert latest.step == 1
    assert latest.value == pytest.approx(0.31)
    assert run.latest_metric("missing") is None


# ============================================================
# Model registry
# ============================================================


def test_register_model_digest_stable() -> None:
    a = register_model_version(model_name="ppo_v3", version=1, source_run_id="r1", registered_ns=42)
    b = register_model_version(model_name="ppo_v3", version=1, source_run_id="r1", registered_ns=42)
    assert a == b
    assert a.digest == b.digest


def test_register_model_rejects_zero_version() -> None:
    with pytest.raises(ExperimentTrackingError):
        register_model_version(
            model_name="ppo_v3",
            version=0,
            source_run_id="r1",
            registered_ns=42,
        )


def test_register_model_rejects_negative_registered_ns() -> None:
    with pytest.raises(ExperimentTrackingError):
        register_model_version(
            model_name="ppo_v3",
            version=1,
            source_run_id="r1",
            registered_ns=-1,
        )


# ============================================================
# Stage transitions
# ============================================================


def test_legal_stage_transitions_table() -> None:
    table = legal_stage_transitions()
    assert ModelStage.STAGING in table[ModelStage.NONE]
    assert ModelStage.PRODUCTION in table[ModelStage.STAGING]
    assert ModelStage.ARCHIVED in table[ModelStage.PRODUCTION]
    assert table[ModelStage.ARCHIVED] == frozenset()


def test_propose_stage_transition_records_advisory() -> None:
    v = register_model_version(
        model_name="ppo_v3",
        version=1,
        source_run_id="r1",
        registered_ns=10,
    )
    rec = propose_stage_transition(
        version=v,
        to_stage=ModelStage.STAGING,
        proposed_ns=20,
        rationale="canary-passed",
    )
    assert rec.from_stage == ModelStage.NONE
    assert rec.to_stage == ModelStage.STAGING
    assert rec.recommendation_id.startswith("stage-")
    assert len(rec.recommendation_id) == len("stage-") + 32


def test_propose_stage_transition_three_run_equality() -> None:
    v = register_model_version(
        model_name="ppo_v3",
        version=1,
        source_run_id="r1",
        registered_ns=10,
    )
    a = propose_stage_transition(
        version=v, to_stage=ModelStage.STAGING, proposed_ns=20, rationale="ok"
    )
    b = propose_stage_transition(
        version=v, to_stage=ModelStage.STAGING, proposed_ns=20, rationale="ok"
    )
    c = propose_stage_transition(
        version=v, to_stage=ModelStage.STAGING, proposed_ns=20, rationale="ok"
    )
    assert a == b == c


def test_propose_stage_transition_rejects_illegal_jump() -> None:
    v = register_model_version(
        model_name="ppo_v3",
        version=1,
        source_run_id="r1",
        registered_ns=10,
    )
    # Illegal: NONE -> PRODUCTION (must pass through STAGING).
    with pytest.raises(ExperimentTrackingError):
        propose_stage_transition(
            version=v,
            to_stage=ModelStage.PRODUCTION,
            proposed_ns=20,
            rationale="skip-staging",
        )


def test_propose_stage_transition_rejects_archived_revival() -> None:
    archived = ModelVersion(
        model_name="ppo_v3",
        version=1,
        source_run_id="r1",
        current_stage=ModelStage.ARCHIVED,
        registered_ns=10,
        digest="0" * 32,
    )
    with pytest.raises(ExperimentTrackingError):
        propose_stage_transition(
            version=archived,
            to_stage=ModelStage.PRODUCTION,
            proposed_ns=20,
            rationale="revive",
        )


# ============================================================
# InMemoryTrackingBackend determinism
# ============================================================


def test_backend_protocol_compatibility() -> None:
    backend = InMemoryTrackingBackend()
    assert isinstance(backend, TrackingBackend)


def test_backend_record_run_sorted() -> None:
    backend = InMemoryTrackingBackend()
    r1 = _sample_run(run_id="b_zeta")
    r2 = _sample_run(run_id="a_alpha")
    backend.record_run(r1)
    backend.record_run(r2)
    assert tuple(r.run_id for r in backend.runs) == ("a_alpha", "b_zeta")


def test_backend_record_run_rejects_duplicate() -> None:
    backend = InMemoryTrackingBackend()
    r = _sample_run(run_id="r1")
    backend.record_run(r)
    with pytest.raises(ExperimentTrackingError):
        backend.record_run(r)


def test_backend_record_model_version_sorted() -> None:
    backend = InMemoryTrackingBackend()
    v1 = register_model_version(model_name="m_b", version=1, source_run_id="r1", registered_ns=0)
    v2 = register_model_version(model_name="m_a", version=1, source_run_id="r1", registered_ns=0)
    v3 = register_model_version(model_name="m_a", version=2, source_run_id="r2", registered_ns=0)
    backend.record_model_version(v1)
    backend.record_model_version(v3)
    backend.record_model_version(v2)
    keys = tuple((v.model_name, v.version) for v in backend.model_versions)
    assert keys == (("m_a", 1), ("m_a", 2), ("m_b", 1))


def test_backend_record_model_version_rejects_duplicate() -> None:
    backend = InMemoryTrackingBackend()
    v = register_model_version(model_name="m1", version=1, source_run_id="r1", registered_ns=0)
    backend.record_model_version(v)
    with pytest.raises(ExperimentTrackingError):
        backend.record_model_version(v)


def test_backend_record_recommendation_sorted() -> None:
    backend = InMemoryTrackingBackend()
    v1 = register_model_version(model_name="m_a", version=1, source_run_id="r1", registered_ns=0)
    v2 = register_model_version(model_name="m_b", version=1, source_run_id="r2", registered_ns=0)
    r1 = propose_stage_transition(
        version=v1, to_stage=ModelStage.STAGING, proposed_ns=1, rationale="a"
    )
    r2 = propose_stage_transition(
        version=v2, to_stage=ModelStage.STAGING, proposed_ns=1, rationale="b"
    )
    backend.record_stage_recommendation(r2)
    backend.record_stage_recommendation(r1)
    ids = tuple(r.recommendation_id for r in backend.recommendations)
    assert ids == tuple(sorted(ids))


# ============================================================
# Production factory contract
# ============================================================


def test_mlflow_backend_factory_raises_meaningful_error() -> None:
    """Lazy import — factory raises if mlflow not wired in production."""

    with pytest.raises(ExperimentTrackingError):
        et.mlflow_backend_factory()


# ============================================================
# Importer scan — no production tier imports this module
# ============================================================


_PRODUCTION_TIERS = (
    "execution_engine",
    "governance_engine",
    "system_engine",
    "intelligence_engine",
    "registry",
    "dashboard_backend",
)


def test_no_production_tier_imports_experiment_tracking() -> None:
    """Production tiers must not import the offline tracking module."""

    repo_root = Path(__file__).resolve().parents[1]
    for tier in _PRODUCTION_TIERS:
        tier_path = repo_root / tier
        if not tier_path.exists():
            continue
        for py in tier_path.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "evolution_engine.experiment_tracking" not in text, (
                f"production tier {tier} imports experiment_tracking via {py}"
            )
            assert "from evolution_engine import experiment_tracking" not in text, (
                f"production tier {tier} imports experiment_tracking via {py}"
            )
