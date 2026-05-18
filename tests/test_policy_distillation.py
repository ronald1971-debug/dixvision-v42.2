"""B-13 — Tests for ``learning_engine.lanes.policy_distillation``.

Pinned invariants:
* AST authority pins (no torch / numpy / gymnasium / cleanrl / engine
  cross-imports / PatchProposal called outside the
  :meth:`PolicyDistillation.distill` coordinator).
* GAE-Lambda math against hand-computed reference values.
* Validation guard rails for every value object.
* Coordinator round-trip + INV-15 byte-identical 3-run replay.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from learning_engine.lanes import policy_distillation
from learning_engine.lanes.policy_distillation import (
    MAX_TRAJECTORIES,
    NEW_PIP_DEPENDENCIES,
    POLICY_DISTILLATION_VERSION,
    DistillationCallback,
    DistillationMetrics,
    DistillationProposal,
    DistillationResult,
    PolicyArtifact,
    PolicyDistillation,
    PolicyDistillationError,
    PolicyDistiller,
    PPOConfig,
    Trajectory,
    TrajectoryStep,
    cleanrl_ppo_distiller_factory,
    compute_gae_lambda,
    derive_policy_artifact_digest,
    null_distillation_callback,
)

MODULE_PATH = Path(policy_distillation.__file__)
MODULE_SOURCE = MODULE_PATH.read_text()
MODULE_AST = ast.parse(MODULE_SOURCE)


# ============================================================================ authority pins


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: vwxyzjn/cleanrl")


def test_authority_pip_dependencies_torch_only() -> None:
    assert NEW_PIP_DEPENDENCIES == ("torch",)


def _iter_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _iter_top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_authority_no_top_level_torch_import() -> None:
    tops = _iter_top_level_imports(MODULE_AST)
    forbidden = {
        "torch",
        "numpy",
        "gymnasium",
        "gym",
        "cleanrl",
        "pandas",
        "polars",
        "scipy",
        "tensorboard",
        "wandb",
        "mlflow",
    }
    for name in tops:
        root = name.split(".")[0]
        assert root not in forbidden, f"forbidden top-level import: {name}"


def test_authority_no_runtime_imports() -> None:
    forbidden_roots = {
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "socket",
        "secrets",
        "uuid",
        "requests",
        "httpx",
        "aiohttp",
        "websockets",
    }
    for name in _iter_imports(MODULE_AST):
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden import for OFFLINE_ONLY tier: {name}"


def test_authority_no_engine_cross_imports() -> None:
    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
    }
    for name in _iter_imports(MODULE_AST):
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden engine cross-import: {name}"


def _collect_call_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def test_authority_no_typed_event_construction() -> None:
    """B27/B28/INV-71: learning_engine.* MUST NOT construct typed bus events."""

    forbidden_types = {
        "PatchProposal",
        "SignalEvent",
        "ExecutionIntent",
        "HazardEvent",
        "GovernanceDecision",
        "TradeOutcome",
        "LearningUpdate",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_types, (
                f"forbidden typed-event construction: {node.func.id}"
            )


def test_authority_no_learning_contract_import() -> None:
    """B28: no import of ``core.contracts.learning.PatchProposal`` etc."""

    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "core.contracts.learning":
                for alias in node.names:
                    assert alias.name not in {
                        "PatchProposal",
                        "LearningUpdate",
                    }, f"learning_engine must not import {alias.name}"


def test_authority_no_top_level_io_calls() -> None:
    """No ``open`` / ``print`` / ``input`` / ``exec`` / ``eval`` at module load."""

    forbidden = {"open", "print", "input", "exec", "eval"}
    for node in MODULE_AST.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                assert sub.func.id not in forbidden, f"forbidden top-level IO call: {sub.func.id}"


def test_authority_module_reimport_clean() -> None:
    mod = importlib.import_module("learning_engine.lanes.policy_distillation")
    assert mod is policy_distillation


def test_version_constant() -> None:
    assert POLICY_DISTILLATION_VERSION.startswith("b-13-")


# ============================================================================ helpers


def _step(
    *,
    obs: tuple[float, ...] = (0.5,),
    action: tuple[float, ...] = (0.1,),
    log_prob: float = -0.5,
    value: float = 0.2,
    reward: float = 1.0,
    done: bool = False,
) -> TrajectoryStep:
    return TrajectoryStep(
        obs=obs,
        action=action,
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=done,
    )


def _trajectory(
    *,
    trajectory_id: str = "tr-0",
    n: int = 4,
    bootstrap_value: float = 0.0,
) -> Trajectory:
    steps = tuple(
        _step(
            obs=(float(i),),
            action=(float(i) / 10.0,),
            log_prob=-0.5 - 0.01 * i,
            value=0.1 * i,
            reward=1.0 if i % 2 == 0 else 0.5,
            done=(i == n - 1),
        )
        for i in range(n)
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        steps=steps,
        bootstrap_value=bootstrap_value,
    )


def _config(obs_dim: int = 1, action_dim: int = 1) -> PPOConfig:
    return PPOConfig(obs_dim=obs_dim, action_dim=action_dim)


class _FakeDistiller:
    """Deterministic distiller for tests — no torch."""

    def __init__(self, *, label: str = "fake") -> None:
        self._label = label

    def distill(
        self,
        *,
        trajectories: tuple[Trajectory, ...],
        advantages: tuple[tuple[float, ...], ...],
        returns: tuple[tuple[float, ...], ...],
        config: PPOConfig,
        seed: int,
        callback: DistillationCallback,
    ) -> tuple[DistillationMetrics, PolicyArtifact]:
        episode_rewards: list[float] = []
        total_steps = 0
        for tr in trajectories:
            total_steps += len(tr.steps)
            episode_rewards.append(sum(s.reward for s in tr.steps))
        reward_mean = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        reward_best = max(episode_rewards) if episode_rewards else 0.0
        adv_sq_sum = sum(a * a for adv_list in advantages for a in adv_list)
        ret_sq_sum = sum(r * r for ret_list in returns for r in ret_list)
        policy_loss = -0.0001 * adv_sq_sum
        value_loss = 0.5 * ret_sq_sum
        entropy_loss = -0.01 * float(total_steps)
        approx_kl = 0.0
        for epoch in range(config.update_epochs):
            callback.on_epoch(
                epoch=epoch,
                approx_kl=approx_kl,
                policy_loss=policy_loss,
                value_loss=value_loss,
            )
        metrics = DistillationMetrics(
            episode_reward_mean=reward_mean,
            episode_reward_best=reward_best,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            approx_kl=approx_kl,
            clip_fraction=0.0,
            total_timesteps=total_steps,
        )
        artifact = derive_policy_artifact_digest(
            backend=self._label,
            payload_parts=(
                f"seed={seed}",
                f"od={config.obs_dim}",
                f"ad={config.action_dim}",
                f"t={total_steps}",
                f"erm={reward_mean:.12g}",
            ),
        )
        artifact = PolicyArtifact(
            backend=artifact.backend,
            content_digest=artifact.content_digest,
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
        )
        return metrics, artifact


# ============================================================================ value objects


def test_step_is_frozen() -> None:
    s = _step()
    with pytest.raises((AttributeError, TypeError)):
        s.reward = 9.0  # type: ignore[misc]


def test_trajectory_is_frozen() -> None:
    tr = _trajectory()
    with pytest.raises((AttributeError, TypeError)):
        tr.trajectory_id = "x"  # type: ignore[misc]


def test_config_is_frozen() -> None:
    c = _config()
    with pytest.raises((AttributeError, TypeError)):
        c.gamma = 0.5  # type: ignore[misc]


def test_artifact_is_frozen() -> None:
    a = PolicyArtifact(backend="fake", content_digest="0" * 16, obs_dim=1, action_dim=1)
    with pytest.raises((AttributeError, TypeError)):
        a.backend = "other"  # type: ignore[misc]


# ============================================================================ GAE-Lambda math


def test_gae_zero_reward_no_advantage() -> None:
    """All zero rewards, all zero values → all zero advantages."""

    steps = tuple(
        TrajectoryStep(
            obs=(0.0,),
            action=(0.0,),
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=(i == 3),
        )
        for i in range(4)
    )
    tr = Trajectory(trajectory_id="z", steps=steps, bootstrap_value=0.0)
    adv, ret = compute_gae_lambda(tr, gamma=0.99, gae_lambda=0.95)
    assert adv == (0.0, 0.0, 0.0, 0.0)
    assert ret == (0.0, 0.0, 0.0, 0.0)


def test_gae_terminal_step_advantage_equals_reward_minus_value() -> None:
    """If last step is terminal, advantage[-1] = reward - value."""

    steps = (
        TrajectoryStep(
            obs=(0.0,),
            action=(0.0,),
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=False,
        ),
        TrajectoryStep(
            obs=(0.0,),
            action=(0.0,),
            log_prob=0.0,
            value=0.5,
            reward=2.0,
            done=True,
        ),
    )
    tr = Trajectory(trajectory_id="t", steps=steps, bootstrap_value=99.0)
    adv, ret = compute_gae_lambda(tr, gamma=0.99, gae_lambda=0.95)
    assert adv[1] == pytest.approx(2.0 - 0.5)
    assert ret[1] == pytest.approx(2.0)


def test_gae_nonterminal_bootstrap_value_used() -> None:
    """If last step not done, bootstrap_value enters the recurrence."""

    steps = (
        TrajectoryStep(
            obs=(0.0,),
            action=(0.0,),
            log_prob=0.0,
            value=1.0,
            reward=0.0,
            done=False,
        ),
    )
    tr = Trajectory(trajectory_id="b", steps=steps, bootstrap_value=5.0)
    adv, _ = compute_gae_lambda(tr, gamma=0.9, gae_lambda=0.95)
    expected = 0.0 + 0.9 * 5.0 * 1.0 - 1.0  # delta
    assert adv[0] == pytest.approx(expected)


def test_gae_empty_trajectory() -> None:
    tr = Trajectory(trajectory_id="e", steps=(), bootstrap_value=0.0)
    adv, ret = compute_gae_lambda(tr, gamma=0.99, gae_lambda=0.95)
    assert adv == ()
    assert ret == ()


def test_gae_rejects_bad_gamma() -> None:
    tr = _trajectory()
    with pytest.raises(PolicyDistillationError):
        compute_gae_lambda(tr, gamma=1.5, gae_lambda=0.95)


def test_gae_rejects_bad_lambda() -> None:
    tr = _trajectory()
    with pytest.raises(PolicyDistillationError):
        compute_gae_lambda(tr, gamma=0.99, gae_lambda=-0.1)


def test_gae_returns_equal_advantages_plus_values() -> None:
    tr = _trajectory(n=5, bootstrap_value=0.3)
    adv, ret = compute_gae_lambda(tr, gamma=0.99, gae_lambda=0.95)
    for a, r, step in zip(adv, ret, tr.steps, strict=True):
        assert r == pytest.approx(a + step.value)


# ============================================================================ validation


def test_config_rejects_negative_lr() -> None:
    with pytest.raises(PolicyDistillationError):
        _validate = PPOConfig(learning_rate=-1.0)
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=[_trajectory()],
            config=_validate,
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_config_rejects_zero_minibatches() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=[_trajectory()],
            config=PPOConfig(num_minibatches=0),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_config_rejects_out_of_range_gamma() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=[_trajectory()],
            config=PPOConfig(gamma=1.5),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_empty_trajectories() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(),
            config=_config(),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_duplicate_trajectory_id() -> None:
    t1 = _trajectory(trajectory_id="dup")
    t2 = _trajectory(trajectory_id="dup")
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(t1, t2),
            config=_config(),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_obs_dim_mismatch() -> None:
    tr = _trajectory()  # obs_dim=1
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(tr,),
            config=_config(obs_dim=2),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_action_dim_mismatch() -> None:
    tr = _trajectory()  # action_dim=1
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(tr,),
            config=_config(action_dim=2),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_negative_seed() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(_trajectory(),),
            config=_config(),
            seed=-1,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_negative_ts() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(_trajectory(),),
            config=_config(),
            seed=0,
            ts_ns=-1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_empty_proposal_id() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(_trajectory(),),
            config=_config(),
            seed=0,
            ts_ns=1,
            proposal_id="",
            target_strategy="s",
        )


def test_distill_rejects_empty_target_strategy() -> None:
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(_trajectory(),),
            config=_config(),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="",
        )


def test_distill_rejects_non_finite_value() -> None:
    bad_step = TrajectoryStep(
        obs=(float("nan"),),
        action=(0.0,),
        log_prob=0.0,
        value=0.0,
        reward=0.0,
        done=True,
    )
    tr = Trajectory(trajectory_id="x", steps=(bad_step,), bootstrap_value=0.0)
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(tr,),
            config=_config(),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


def test_distill_rejects_too_many_trajectories() -> None:
    trajectories = tuple(_trajectory(trajectory_id=f"t-{i}") for i in range(MAX_TRAJECTORIES + 1))
    with pytest.raises(PolicyDistillationError):
        PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=trajectories,
            config=_config(),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            target_strategy="s",
        )


# ====================================================== coordinator round-trip


def test_distill_returns_result_and_proposal() -> None:
    coord = PolicyDistillation(distiller=_FakeDistiller())
    result, proposal = coord.distill(
        trajectories=(_trajectory(),),
        config=_config(),
        seed=42,
        ts_ns=1_000,
        proposal_id="patch-1",
        target_strategy="strat-A",
    )
    assert isinstance(result, DistillationResult)
    assert isinstance(proposal, DistillationProposal)
    assert proposal.proposal_id == "patch-1"
    assert proposal.target_strategy == "strat-A"
    assert proposal.source == "learning_engine.lanes.policy_distillation"
    assert proposal.touchpoints == ("policy_weights",)
    assert proposal.policy_digest == result.policy_digest
    assert proposal.seed == 42
    assert proposal.version == POLICY_DISTILLATION_VERSION


def test_distill_policy_digest_is_16_hex() -> None:
    result, _ = PolicyDistillation(distiller=_FakeDistiller()).distill(
        trajectories=(_trajectory(),),
        config=_config(),
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    assert len(result.policy_digest) == 16
    int(result.policy_digest, 16)


def test_distill_metrics_populated() -> None:
    tr = _trajectory(n=6)
    result, _ = PolicyDistillation(distiller=_FakeDistiller()).distill(
        trajectories=(tr,),
        config=_config(),
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    assert result.metrics.total_timesteps == 6
    assert result.metrics.episode_reward_mean > 0.0
    assert result.metrics.episode_reward_best > 0.0


# ============================================================== INV-15 replay


def test_replay_byte_identical_three_runs() -> None:
    """Same inputs, three runs → identical result + proposal."""

    def run() -> tuple[DistillationResult, DistillationProposal]:
        return PolicyDistillation(distiller=_FakeDistiller()).distill(
            trajectories=(_trajectory(n=4), _trajectory(trajectory_id="tr-1", n=3)),
            config=_config(),
            seed=7,
            ts_ns=42_000,
            proposal_id="patch-X",
            target_strategy="strat-X",
        )

    r1, p1 = run()
    r2, p2 = run()
    r3, p3 = run()
    assert r1 == r2 == r3
    assert p1 == p2 == p3


def test_replay_seed_changes_policy_digest() -> None:
    coord = PolicyDistillation(distiller=_FakeDistiller())
    r1, _ = coord.distill(
        trajectories=(_trajectory(),),
        config=_config(),
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    r2, _ = coord.distill(
        trajectories=(_trajectory(),),
        config=_config(),
        seed=1,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    assert r1.policy_digest != r2.policy_digest


def test_replay_config_changes_config_digest() -> None:
    coord = PolicyDistillation(distiller=_FakeDistiller())
    r1, _ = coord.distill(
        trajectories=(_trajectory(),),
        config=_config(),
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    r2, _ = coord.distill(
        trajectories=(_trajectory(),),
        config=PPOConfig(gamma=0.5),  # different gamma
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    assert r1.config_digest != r2.config_digest


def test_replay_rollout_changes_rollout_digest() -> None:
    coord = PolicyDistillation(distiller=_FakeDistiller())
    r1, _ = coord.distill(
        trajectories=(_trajectory(n=3),),
        config=_config(),
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    r2, _ = coord.distill(
        trajectories=(_trajectory(n=5),),
        config=_config(),
        seed=0,
        ts_ns=1,
        proposal_id="p",
        target_strategy="s",
    )
    assert r1.rollout_digest != r2.rollout_digest


# ============================================================================ helpers / factory


def test_null_callback_callable() -> None:
    cb = null_distillation_callback()
    cb.on_distillation_start(_config())
    cb.on_epoch(epoch=0, approx_kl=0.0, policy_loss=0.0, value_loss=0.0)
    cb.on_distillation_end(
        DistillationMetrics(
            episode_reward_mean=0.0,
            episode_reward_best=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy_loss=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            total_timesteps=0,
        )
    )


def test_derive_policy_artifact_digest_deterministic() -> None:
    a = derive_policy_artifact_digest(backend="fake", payload_parts=("a", "b", "c"))
    b = derive_policy_artifact_digest(backend="fake", payload_parts=("a", "b", "c"))
    assert a.content_digest == b.content_digest
    assert a.backend == "fake"


def test_derive_policy_artifact_digest_rejects_empty_backend() -> None:
    with pytest.raises(PolicyDistillationError):
        derive_policy_artifact_digest(backend="", payload_parts=("x",))


def test_cleanrl_ppo_distiller_factory_raises_not_implemented() -> None:
    """The factory body is a production seam — not callable in unit tests."""

    with pytest.raises((NotImplementedError, ImportError, ModuleNotFoundError)):
        cleanrl_ppo_distiller_factory()


def test_policy_distiller_protocol_runtime_friendly() -> None:
    fake: PolicyDistiller = _FakeDistiller()
    metrics, artifact = fake.distill(
        trajectories=(_trajectory(),),
        advantages=((0.0, 0.0, 0.0, 0.0),),
        returns=((0.0, 0.0, 0.0, 0.0),),
        config=_config(),
        seed=0,
        callback=null_distillation_callback(),
    )
    assert isinstance(metrics, DistillationMetrics)
    assert isinstance(artifact, PolicyArtifact)
