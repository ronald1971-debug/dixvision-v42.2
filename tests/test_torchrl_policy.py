"""Tests for C-28 torchrl policy-distillation lane (OFFLINE_ONLY)."""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

from learning_engine.lanes.policy_distillation_torchrl import (
    MAX_ACTION_DIM,
    MAX_OBS_DIM,
    MAX_TRAJECTORIES,
    NEW_PIP_DEPENDENCIES,
    TORCHRL_DISTILLER_VERSION,
    ActorCriticConfig,
    CollectorRollout,
    DistillationProposalTorchRL,
    DistillationResultTorchRL,
    PolicyDistillationTorchRL,
    TensorDictStep,
    TorchRLDistillationError,
    TorchRLPolicyArtifact,
    TrainingMetrics,
    compute_advantages,
    derive_torchrl_artifact_digest,
    null_torchrl_callback,
    torchrl_distiller_factory,
)


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
def test_pip_deps_pinned() -> None:
    assert NEW_PIP_DEPENDENCIES == ("torchrl", "tensordict", "torch")


def test_version_string_pinned() -> None:
    assert TORCHRL_DISTILLER_VERSION == "c-28-torchrl-actor-critic-1"


def test_constants_within_sane_bounds() -> None:
    assert MAX_TRAJECTORIES == 1024
    assert MAX_OBS_DIM == 4096
    assert MAX_ACTION_DIM == 1024


# ---------------------------------------------------------------------------
# compute_advantages
# ---------------------------------------------------------------------------
def test_compute_advantages_simple_two_step() -> None:
    adv = compute_advantages(
        rewards=[1.0, 1.0],
        values=[0.0, 0.0],
        dones=[False, True],
        bootstrap_value=0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    # final delta = r1 + 0*0*0 - 0 = 1; gae_T = 1
    # delta_0 = r0 + 1*0*1 - 0 = 1; gae_0 = 1 + 1*1*1*1 = 2
    assert adv == (2.0, 1.0)


def test_compute_advantages_rejects_mismatched_lengths() -> None:
    with pytest.raises(TorchRLDistillationError):
        compute_advantages(
            rewards=[1.0, 2.0],
            values=[0.0],
            dones=[False, True],
            bootstrap_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_compute_advantages_rejects_bad_gamma() -> None:
    with pytest.raises(TorchRLDistillationError):
        compute_advantages(
            rewards=[1.0],
            values=[0.0],
            dones=[False],
            bootstrap_value=0.0,
            gamma=-0.1,
            gae_lambda=0.95,
        )


def test_compute_advantages_rejects_bad_lambda() -> None:
    with pytest.raises(TorchRLDistillationError):
        compute_advantages(
            rewards=[1.0],
            values=[0.0],
            dones=[False],
            bootstrap_value=0.0,
            gamma=0.99,
            gae_lambda=2.0,
        )


def test_compute_advantages_deterministic() -> None:
    args = dict(
        rewards=[0.3, 0.1, -0.2, 0.5],
        values=[0.0, 0.1, 0.2, 0.3],
        dones=[False, False, False, True],
        bootstrap_value=0.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    runs = [compute_advantages(**args) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_rejects_gpu_device() -> None:
    cfg = ActorCriticConfig(device="cuda")
    with pytest.raises(TorchRLDistillationError, match="cpu"):
        _, _ = _run(cfg=cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"num_epochs": 0},
        {"minibatch_size": 0},
        {"clip_epsilon": 0.0},
        {"clip_epsilon": 1.0},
        {"critic_coef": -0.1},
        {"entropy_coef": -0.1},
        {"gamma": 1.1},
        {"gae_lambda": -0.1},
        {"max_grad_norm": 0.0},
        {"target_kl": 0.0},
        {"obs_dim": 0},
        {"obs_dim": MAX_OBS_DIM + 1},
        {"action_dim": 0},
        {"action_dim": MAX_ACTION_DIM + 1},
        {"hidden_dim": 0},
    ],
)
def test_config_invariants(kwargs: dict[str, object]) -> None:
    base = dict(obs_dim=2, action_dim=1)
    cfg = ActorCriticConfig(**{**base, **kwargs})  # type: ignore[arg-type]
    with pytest.raises(TorchRLDistillationError):
        _run(cfg=cfg)


# ---------------------------------------------------------------------------
# Coordinator — deterministic fake distiller
# ---------------------------------------------------------------------------
class _FakeDistiller:
    def __init__(self, *, reward_mean: float = 1.5, best: float = 2.0) -> None:
        self._reward_mean = reward_mean
        self._best = best

    def train(self, *, rollouts, config, seed, callback):  # noqa: ANN001
        callback.on_epoch(0, 0.01, 0.5, 0.3)
        total = sum(len(r.steps) for r in rollouts)
        metrics = TrainingMetrics(
            mean_reward=self._reward_mean,
            best_reward=self._best,
            loss_objective=0.5,
            loss_critic=0.3,
            loss_entropy=0.02,
            approx_kl=0.01,
            clip_fraction=0.1,
            total_timesteps=total,
        )
        digest = derive_torchrl_artifact_digest(
            weights_bytes=f"seed={seed};dims={config.obs_dim}x{config.action_dim}".encode()
        )
        artifact = TorchRLPolicyArtifact(
            backend="fake-torchrl",
            content_digest=digest,
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
        )
        return metrics, artifact


def _step(obs_dim: int = 2, action_dim: int = 1) -> TensorDictStep:
    return TensorDictStep(
        obs=tuple(float(i) for i in range(obs_dim)),
        action=tuple(0.1 * i for i in range(action_dim)),
        sample_log_prob=-0.5,
        state_value=0.25,
        reward=0.3,
        done=False,
    )


def _rollouts(n: int = 2, steps: int = 4) -> tuple[CollectorRollout, ...]:
    return tuple(
        CollectorRollout(
            rollout_id=f"r{i}",
            steps=tuple(_step() for _ in range(steps)),
            bootstrap_value=0.0,
        )
        for i in range(n)
    )


def _run(
    *,
    cfg: ActorCriticConfig | None = None,
    seed: int = 7,
    distiller: object | None = None,
) -> tuple[DistillationResultTorchRL, DistillationProposalTorchRL]:
    cfg = cfg or ActorCriticConfig(obs_dim=2, action_dim=1)
    distiller = distiller or _FakeDistiller()
    return PolicyDistillationTorchRL().distill(
        ts_ns=1_000_000,
        proposal_id="prop-001",
        target_strategy="strat-a",
        rollouts=_rollouts(),
        config=cfg,
        seed=seed,
        distiller=distiller,  # type: ignore[arg-type]
        touchpoints=("strategy.a.beta",),
        rationale="unit",
    )


def test_distill_returns_result_and_proposal() -> None:
    result, proposal = _run()
    assert isinstance(result, DistillationResultTorchRL)
    assert isinstance(proposal, DistillationProposalTorchRL)
    assert result.policy_digest == proposal.policy_digest
    assert result.config_digest == proposal.config_digest
    assert result.rollout_digest == proposal.rollout_digest


def test_distill_proposal_carries_canonical_fields() -> None:
    _, proposal = _run()
    assert proposal.ts_ns == 1_000_000
    assert proposal.proposal_id == "prop-001"
    assert proposal.target_strategy == "strat-a"
    assert proposal.source == "learning_engine.lanes.policy_distillation_torchrl"
    assert proposal.version == TORCHRL_DISTILLER_VERSION
    assert proposal.touchpoints == ("strategy.a.beta",)
    assert proposal.seed == 7


def test_distill_rejects_negative_ts_ns() -> None:
    with pytest.raises(TorchRLDistillationError):
        PolicyDistillationTorchRL().distill(
            ts_ns=-1,
            proposal_id="p",
            target_strategy="s",
            rollouts=_rollouts(),
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_empty_proposal_id() -> None:
    with pytest.raises(TorchRLDistillationError):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="",
            target_strategy="s",
            rollouts=_rollouts(),
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_empty_target_strategy() -> None:
    with pytest.raises(TorchRLDistillationError):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="p",
            target_strategy="",
            rollouts=_rollouts(),
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_negative_seed() -> None:
    with pytest.raises(TorchRLDistillationError):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="p",
            target_strategy="s",
            rollouts=_rollouts(),
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=-1,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_artifact_dim_mismatch() -> None:
    class _BadDistiller:
        def train(self, *, rollouts, config, seed, callback):  # noqa: ANN001, ARG002
            metrics = TrainingMetrics(
                mean_reward=0.0,
                best_reward=0.0,
                loss_objective=0.0,
                loss_critic=0.0,
                loss_entropy=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                total_timesteps=1,
            )
            artifact = TorchRLPolicyArtifact(
                backend="x",
                content_digest="00",
                obs_dim=99,
                action_dim=99,
            )
            return metrics, artifact

    with pytest.raises(TorchRLDistillationError, match="artifact dims"):
        _run(distiller=_BadDistiller())


def test_distill_rejects_empty_rollouts() -> None:
    with pytest.raises(TorchRLDistillationError):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="p",
            target_strategy="s",
            rollouts=(),
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_duplicate_rollout_ids() -> None:
    rolls = (
        CollectorRollout(rollout_id="x", steps=(_step(),)),
        CollectorRollout(rollout_id="x", steps=(_step(),)),
    )
    with pytest.raises(TorchRLDistillationError, match="duplicate"):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="p",
            target_strategy="s",
            rollouts=rolls,
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_step_obs_dim_mismatch() -> None:
    bad_step = TensorDictStep(
        obs=(0.0,),
        action=(0.1,),
        sample_log_prob=0.0,
        state_value=0.0,
        reward=0.0,
        done=False,
    )
    rolls = (CollectorRollout(rollout_id="x", steps=(bad_step,)),)
    with pytest.raises(TorchRLDistillationError, match="obs length"):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="p",
            target_strategy="s",
            rollouts=rolls,
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


def test_distill_rejects_nonfinite_log_prob() -> None:
    bad = TensorDictStep(
        obs=(0.0, 1.0),
        action=(0.1,),
        sample_log_prob=float("inf"),
        state_value=0.0,
        reward=0.0,
        done=False,
    )
    rolls = (CollectorRollout(rollout_id="x", steps=(bad,)),)
    with pytest.raises(TorchRLDistillationError, match="finite"):
        PolicyDistillationTorchRL().distill(
            ts_ns=0,
            proposal_id="p",
            target_strategy="s",
            rollouts=rolls,
            config=ActorCriticConfig(obs_dim=2, action_dim=1),
            seed=0,
            distiller=_FakeDistiller(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# INV-15 determinism
# ---------------------------------------------------------------------------
def test_three_run_byte_identical_result() -> None:
    runs = [_run() for _ in range(3)]
    a = runs[0][0]
    for b, _ in runs[1:]:
        assert a == b


def test_three_run_byte_identical_proposal() -> None:
    runs = [_run() for _ in range(3)]
    a = runs[0][1]
    for _, b in runs[1:]:
        assert a == b


# ---------------------------------------------------------------------------
# Value objects — frozen + slotted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "obj",
    [
        TensorDictStep(
            obs=(0.0,),
            action=(0.0,),
            sample_log_prob=0.0,
            state_value=0.0,
            reward=0.0,
            done=False,
        ),
        CollectorRollout(rollout_id="x", steps=()),
        ActorCriticConfig(),
        TrainingMetrics(
            mean_reward=0.0,
            best_reward=0.0,
            loss_objective=0.0,
            loss_critic=0.0,
            loss_entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            total_timesteps=0,
        ),
        TorchRLPolicyArtifact(backend="x", content_digest="0", obs_dim=1, action_dim=1),
    ],
)
def test_value_objects_frozen_and_slotted(obj: object) -> None:
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(obj, "_arbitrary_attr_for_test", "y")
    assert not hasattr(obj, "__dict__")


# ---------------------------------------------------------------------------
# derive_torchrl_artifact_digest
# ---------------------------------------------------------------------------
def test_derive_artifact_digest_deterministic() -> None:
    a = derive_torchrl_artifact_digest(weights_bytes=b"weights")
    b = derive_torchrl_artifact_digest(weights_bytes=b"weights")
    assert a == b
    assert len(a) == 16


def test_derive_artifact_digest_rejects_non_bytes() -> None:
    with pytest.raises(TorchRLDistillationError):
        derive_torchrl_artifact_digest(weights_bytes="not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
def test_null_callback_returns_none_for_each_hook() -> None:
    cb = null_torchrl_callback()
    cb.on_training_start(ActorCriticConfig())
    cb.on_epoch(0, 0.0, 0.0, 0.0)
    cb.on_training_end(
        TrainingMetrics(
            mean_reward=0.0,
            best_reward=0.0,
            loss_objective=0.0,
            loss_critic=0.0,
            loss_entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            total_timesteps=0,
        )
    )


# ---------------------------------------------------------------------------
# torchrl_distiller_factory lazy seam
# ---------------------------------------------------------------------------
def test_factory_raises_when_dep_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    if "torchrl" in sys.modules:
        pytest.skip("torchrl is installed; the negative path cannot be tested")
    with pytest.raises(RuntimeError, match="torchrl"):
        torchrl_distiller_factory()


# ---------------------------------------------------------------------------
# AST guards — no top-level torch/torchrl/tensordict imports
# ---------------------------------------------------------------------------
_THIS = pathlib.Path(__file__).resolve()
_MODULE = _THIS.parents[1] / "learning_engine" / "lanes" / "policy_distillation_torchrl.py"


def _module_tree() -> ast.Module:
    return ast.parse(_MODULE.read_text(encoding="utf-8"))


def test_no_top_level_torch_imports() -> None:
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in {"torch", "torchrl", "tensordict", "numpy", "gymnasium"}
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            assert module not in {"torch", "torchrl", "tensordict", "numpy", "gymnasium"}


def test_no_engine_cross_imports() -> None:
    tree = _module_tree()
    forbidden = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            assert module not in forbidden


def test_torch_imports_confined_to_factory() -> None:
    tree = _module_tree()
    factory_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "torchrl_distiller_factory":
            factory_node = node
            break
    assert factory_node is not None
    # walk function body — torch imports must be inside it
    found_torch_import = False
    for child in ast.walk(factory_node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name.split(".")[0] == "torch":
                    found_torch_import = True
        elif isinstance(child, ast.ImportFrom):
            if (child.module or "").split(".")[0] in {"torch", "tensordict", "torchrl"}:
                found_torch_import = True
    assert found_torch_import


def test_module_exports_canonical_symbols() -> None:
    from learning_engine.lanes import policy_distillation_torchrl as mod

    for sym in (
        "PolicyDistillationTorchRL",
        "ActorCriticConfig",
        "CollectorRollout",
        "TensorDictStep",
        "TrainingMetrics",
        "TorchRLPolicyArtifact",
        "DistillationResultTorchRL",
        "DistillationProposalTorchRL",
        "torchrl_distiller_factory",
        "compute_advantages",
        "NEW_PIP_DEPENDENCIES",
        "TORCHRL_DISTILLER_VERSION",
    ):
        assert hasattr(mod, sym)
