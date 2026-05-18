"""C-34 — tests for the sample-factory sandbox surface.

Mirrors the test-shape of :mod:`tests.test_tianshou_sandbox` (C-33):

* value-object validation (frozen + slotted, finite-only floats,
  bounded ints);
* a deterministic in-memory :class:`SampleFactoryPolicyTrainer` fake
  (the Protocol seam — no sample-factory / torch / numpy / gymnasium
  imports happen at any point in this file);
* end-to-end :meth:`SampleFactorySandbox.train` walk producing a
  governance-shaped :class:`PatchProposal`;
* INV-15 3-run byte-identical replay;
* AST guards pinning the OFFLINE_ONLY tier (no sample_factory / torch
  / numpy / gymnasium at module load, no engine cross-imports, no IO
  imports, lazy seam confined to the factories).
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from core.contracts.learning import PatchProposal
from evolution_engine.gym_env import (
    DIXStrategyEnv,
    EpisodeConfig,
    MarketDynamics,
    Observation,
    TradeAction,
    Transition,
)
from evolution_engine.sandbox_sample_factory import (
    MAX_NUM_ENVS_PER_WORKER,
    MAX_NUM_WORKERS,
    MAX_PROPOSAL_ID_LEN,
    MAX_TRAIN_FOR_ENV_STEPS,
    MIN_NUM_ENVS_PER_WORKER,
    MIN_NUM_WORKERS,
    MIN_TRAIN_FOR_ENV_STEPS,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    PolicyArtifact,
    PolicyArtifactSink,
    SampleFactoryAlgoKind,
    SampleFactoryArguments,
    SampleFactorySandbox,
    SampleFactorySandboxCallback,
    SampleFactorySandboxConfigError,
    SampleFactorySandboxMetrics,
    SampleFactorySandboxResult,
    null_sample_factory_callback,
    sample_factory_appo_trainer,
)

# ---------------------------------------------------------------------------
# Constants / module identity
# ---------------------------------------------------------------------------


def test_module_advertises_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ("sample-factory", "gymnasium", "torch")


def test_proposal_source_is_canonical_module_path() -> None:
    assert PROPOSAL_SOURCE == "evolution_engine.sandbox_sample_factory"


def test_train_for_env_steps_bounds() -> None:
    assert MIN_TRAIN_FOR_ENV_STEPS == 1
    assert MAX_TRAIN_FOR_ENV_STEPS == 10_000_000


def test_num_workers_bounds() -> None:
    assert MIN_NUM_WORKERS == 1
    assert MAX_NUM_WORKERS == 64


def test_num_envs_per_worker_bounds() -> None:
    assert MIN_NUM_ENVS_PER_WORKER == 1
    assert MAX_NUM_ENVS_PER_WORKER == 64


def test_max_proposal_id_len_bound() -> None:
    assert MAX_PROPOSAL_ID_LEN == 256


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def test_algo_kind_enum_values_match_sample_factory_strings() -> None:
    assert SampleFactoryAlgoKind.APPO.value == "APPO"


def test_algo_kind_count_is_one() -> None:
    assert len(list(SampleFactoryAlgoKind)) == 1


# ---------------------------------------------------------------------------
# SampleFactoryArguments validation
# ---------------------------------------------------------------------------


def _valid_args(**overrides: Any) -> SampleFactoryArguments:
    base: dict[str, Any] = {
        "algo_kind": SampleFactoryAlgoKind.APPO,
        "random_seed": 0,
        "train_for_env_steps": 1024,
        "batch_size": 64,
        "rollout": 32,
        "num_workers": 2,
        "num_envs_per_worker": 2,
        "gamma": 0.99,
        "learning_rate": 1e-4,
        "target_strategy_id": "test_strategy",
    }
    base.update(overrides)
    return SampleFactoryArguments(**base)


def test_arguments_constructs_with_defaults() -> None:
    args = _valid_args()
    assert args.algo_kind is SampleFactoryAlgoKind.APPO
    assert args.random_seed == 0
    assert args.train_for_env_steps == 1024


def test_arguments_is_frozen_and_slotted() -> None:
    args = _valid_args()
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.random_seed = 99  # type: ignore[misc]
    assert not hasattr(args, "__dict__")


def test_arguments_rejects_non_enum_algo_kind() -> None:
    with pytest.raises(TypeError):
        SampleFactoryArguments(  # type: ignore[arg-type]
            algo_kind="APPO",
            random_seed=0,
        )


def test_arguments_rejects_bool_random_seed() -> None:
    with pytest.raises(TypeError):
        _valid_args(random_seed=True)


def test_arguments_rejects_negative_random_seed() -> None:
    with pytest.raises(ValueError):
        _valid_args(random_seed=-1)


def test_arguments_rejects_below_min_train_for_env_steps() -> None:
    with pytest.raises(ValueError):
        _valid_args(train_for_env_steps=0)


def test_arguments_rejects_above_max_train_for_env_steps() -> None:
    with pytest.raises(ValueError):
        _valid_args(train_for_env_steps=MAX_TRAIN_FOR_ENV_STEPS + 1)


def test_arguments_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        _valid_args(batch_size=0)


def test_arguments_rejects_non_positive_rollout() -> None:
    with pytest.raises(ValueError):
        _valid_args(rollout=0)


def test_arguments_rejects_below_min_num_workers() -> None:
    with pytest.raises(ValueError):
        _valid_args(num_workers=0)


def test_arguments_rejects_above_max_num_workers() -> None:
    with pytest.raises(ValueError):
        _valid_args(num_workers=MAX_NUM_WORKERS + 1)


def test_arguments_rejects_below_min_num_envs_per_worker() -> None:
    with pytest.raises(ValueError):
        _valid_args(num_envs_per_worker=0)


def test_arguments_rejects_above_max_num_envs_per_worker() -> None:
    with pytest.raises(ValueError):
        _valid_args(num_envs_per_worker=MAX_NUM_ENVS_PER_WORKER + 1)


def test_arguments_rejects_nan_gamma() -> None:
    with pytest.raises(ValueError):
        _valid_args(gamma=float("nan"))


def test_arguments_rejects_zero_gamma() -> None:
    with pytest.raises(ValueError):
        _valid_args(gamma=0.0)


def test_arguments_rejects_above_one_gamma() -> None:
    with pytest.raises(ValueError):
        _valid_args(gamma=1.5)


def test_arguments_rejects_nonpositive_learning_rate() -> None:
    with pytest.raises(ValueError):
        _valid_args(learning_rate=0.0)


def test_arguments_rejects_empty_target_strategy_id() -> None:
    with pytest.raises(ValueError):
        _valid_args(target_strategy_id="")


# ---------------------------------------------------------------------------
# SampleFactorySandboxMetrics validation
# ---------------------------------------------------------------------------


def _valid_metrics(**overrides: Any) -> SampleFactorySandboxMetrics:
    base: dict[str, Any] = {
        "iterations_completed": 3,
        "total_steps_executed": 100,
        "mean_episode_reward": 1.5,
        "mean_episode_length": 33.0,
        "best_episode_reward": 2.5,
        "final_value_loss": 0.42,
        "final_policy_loss": 0.13,
    }
    base.update(overrides)
    return SampleFactorySandboxMetrics(**base)


def test_metrics_constructs_with_defaults() -> None:
    metrics = _valid_metrics()
    assert metrics.iterations_completed == 3


def test_metrics_is_frozen_and_slotted() -> None:
    metrics = _valid_metrics()
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.iterations_completed = 99  # type: ignore[misc]
    assert not hasattr(metrics, "__dict__")


def test_metrics_rejects_negative_iterations_completed() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(iterations_completed=-1)


def test_metrics_rejects_negative_total_steps_executed() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(total_steps_executed=-1)


def test_metrics_rejects_nan_mean_episode_reward() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(mean_episode_reward=float("nan"))


def test_metrics_rejects_nan_best_episode_reward() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(best_episode_reward=float("nan"))


def test_metrics_rejects_inf_final_value_loss() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(final_value_loss=float("inf"))


def test_metrics_rejects_inf_final_policy_loss() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(final_policy_loss=float("-inf"))


def test_metrics_rejects_negative_mean_episode_length() -> None:
    with pytest.raises(ValueError):
        _valid_metrics(mean_episode_length=-1.0)


# ---------------------------------------------------------------------------
# SampleFactorySandboxResult validation
# ---------------------------------------------------------------------------


def _valid_proposal(ts_ns: int = 100) -> PatchProposal:
    return PatchProposal(
        ts_ns=ts_ns,
        patch_id="test_patch",
        source=PROPOSAL_SOURCE,
        target_strategy="test_strategy",
        touchpoints=("t1",),
        rationale="test",
    )


def test_result_constructs_with_valid_fields() -> None:
    result = SampleFactorySandboxResult(
        proposal=_valid_proposal(),
        metrics=_valid_metrics(),
        policy_digest="0123456789abcdef",
    )
    assert result.policy_digest == "0123456789abcdef"


def test_result_is_frozen_and_slotted() -> None:
    result = SampleFactorySandboxResult(
        proposal=_valid_proposal(),
        metrics=_valid_metrics(),
        policy_digest="0123456789abcdef",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.policy_digest = "x"  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_result_rejects_non_proposal() -> None:
    with pytest.raises(TypeError):
        SampleFactorySandboxResult(
            proposal="not a proposal",  # type: ignore[arg-type]
            metrics=_valid_metrics(),
            policy_digest="0123456789abcdef",
        )


def test_result_rejects_non_metrics() -> None:
    with pytest.raises(TypeError):
        SampleFactorySandboxResult(
            proposal=_valid_proposal(),
            metrics="not metrics",  # type: ignore[arg-type]
            policy_digest="0123456789abcdef",
        )


def test_result_rejects_wrong_digest_length() -> None:
    with pytest.raises(ValueError):
        SampleFactorySandboxResult(
            proposal=_valid_proposal(),
            metrics=_valid_metrics(),
            policy_digest="short",
        )


def test_result_rejects_non_hex_digest() -> None:
    with pytest.raises(ValueError):
        SampleFactorySandboxResult(
            proposal=_valid_proposal(),
            metrics=_valid_metrics(),
            policy_digest="ZZZZZZZZZZZZZZZZ",
        )


# ---------------------------------------------------------------------------
# Deterministic fakes for SampleFactoryPolicyTrainer + MarketDynamics
# ---------------------------------------------------------------------------


class _FakeDynamics:
    """Deterministic :class:`MarketDynamics` fake — drift-up by 1.0/step."""

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
        return 100.0

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition:
        next_mid = prev_obs.mid_price + 1.0
        next_inv = {
            TradeAction.HOLD: prev_obs.inventory_signed,
            TradeAction.BUY: 1,
            TradeAction.SELL: -1,
        }[action]
        realised = float(next_inv) * (next_mid - prev_obs.mid_price)
        return Transition(
            next_mid_price=next_mid,
            realised_pnl_usd=realised,
            drawdown_usd=0.0,
            next_inventory_signed=next_inv,
            terminated=prev_obs.step_idx >= config.max_steps - 2,
            truncated=False,
        )


class _FakeSampleFactoryTrainer:
    """Deterministic :class:`SampleFactoryPolicyTrainer` — returns canned metrics."""

    __slots__ = ("_metrics",)

    def __init__(self, *, metrics: SampleFactorySandboxMetrics) -> None:
        self._metrics = metrics

    def train(
        self,
        env: DIXStrategyEnv,
        *,
        episode_config: EpisodeConfig,
        arguments: SampleFactoryArguments,
        ts_ns: int,
        callback: SampleFactorySandboxCallback,
    ) -> SampleFactorySandboxMetrics:
        callback.on_training_start(
            ts_ns=ts_ns,
            train_for_env_steps=arguments.train_for_env_steps,
        )
        env.reset(
            seed=arguments.random_seed,
            config=episode_config,
        )
        for step_idx in range(min(3, episode_config.max_steps)):
            action = TradeAction.HOLD
            next_obs, reward, terminated, _truncated, _info = env.step(action)
            callback.on_step(
                ts_ns=ts_ns,
                step_idx=step_idx,
                observation=next_obs,
                action=action,
                reward=reward,
            )
            if terminated:
                break
        callback.on_episode_end(
            ts_ns=ts_ns,
            episode_idx=0,
            episode_reward=float(self._metrics.mean_episode_reward),
            episode_length=int(self._metrics.mean_episode_length),
        )
        return self._metrics


# ---------------------------------------------------------------------------
# SampleFactorySandbox.train end-to-end
# ---------------------------------------------------------------------------


def _sandbox_inputs() -> tuple[
    _FakeDynamics,
    SampleFactoryArguments,
    EpisodeConfig,
    _FakeSampleFactoryTrainer,
]:
    dynamics = _FakeDynamics()
    arguments = _valid_args()
    episode_config = EpisodeConfig(initial_notional_usd=1000.0, max_steps=8)
    trainer = _FakeSampleFactoryTrainer(metrics=_valid_metrics())
    return dynamics, arguments, episode_config, trainer


def test_sandbox_constructs_with_valid_trainer() -> None:
    _, _, _, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    assert sandbox.trainer is trainer


def test_sandbox_is_frozen_and_slotted() -> None:
    _, _, _, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sandbox.trainer = None  # type: ignore[misc]
    assert not hasattr(sandbox, "__dict__")


def test_sandbox_rejects_non_trainer() -> None:
    with pytest.raises(TypeError):
        SampleFactorySandbox(trainer="not a trainer")  # type: ignore[arg-type]


def test_train_emits_patch_proposal() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    result = sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=12345,
        proposal_id="test_patch_0001",
    )
    assert isinstance(result.proposal, PatchProposal)
    assert result.proposal.source == PROPOSAL_SOURCE
    assert result.proposal.target_strategy == arguments.target_strategy_id
    assert result.proposal.patch_id == "test_patch_0001"
    assert result.proposal.ts_ns == 12345


def test_train_proposal_touchpoints_include_module_and_weights() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    result = sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert "evolution_engine.sandbox_sample_factory" in result.proposal.touchpoints
    assert "policy_weights" in result.proposal.touchpoints


def test_train_rationale_includes_algo_kind_and_digest() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    result = sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert "APPO" in result.proposal.rationale
    assert result.policy_digest in result.proposal.rationale


def test_train_proposal_meta_includes_digest_and_algo_kind() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    result = sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert result.proposal.meta["policy_digest"] == result.policy_digest
    assert result.proposal.meta["algo_kind"] == "APPO"
    assert result.proposal.meta["random_seed"] == "0"


def test_train_proposal_meta_overlays_do_not_override_provenance() -> None:
    dynamics, _, episode_config, trainer = _sandbox_inputs()
    arguments = _valid_args(meta={"policy_digest": "ZZZZ", "extra": "ok"})
    sandbox = SampleFactorySandbox(trainer=trainer)
    result = sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert result.proposal.meta["policy_digest"] == result.policy_digest
    assert result.proposal.meta["extra"] == "ok"


def test_train_rejects_non_dynamics() -> None:
    _, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics="not dynamics",  # type: ignore[arg-type]
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_non_arguments() -> None:
    dynamics, _, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=dynamics,
            arguments="bad",  # type: ignore[arg-type]
            episode_config=episode_config,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_non_episode_config() -> None:
    dynamics, arguments, _, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config="bad",  # type: ignore[arg-type]
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_bool_ts_ns() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=True,  # type: ignore[arg-type]
            proposal_id="p",
        )


def test_train_rejects_negative_ts_ns() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(SampleFactorySandboxConfigError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=-1,
            proposal_id="p",
        )


def test_train_rejects_empty_proposal_id() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(SampleFactorySandboxConfigError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=1,
            proposal_id="",
        )


def test_train_rejects_oversize_proposal_id() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(SampleFactorySandboxConfigError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=1,
            proposal_id="x" * (MAX_PROPOSAL_ID_LEN + 1),
        )


def test_train_uses_null_callback_by_default() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    result = sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert isinstance(result, SampleFactorySandboxResult)


def test_train_rejects_non_protocol_callback() -> None:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=1,
            proposal_id="p",
            callback="not a callback",  # type: ignore[arg-type]
        )


def test_train_rejects_trainer_returning_wrong_type() -> None:
    class _BadTrainer:
        def train(
            self,
            env: DIXStrategyEnv,
            *,
            episode_config: EpisodeConfig,
            arguments: SampleFactoryArguments,
            ts_ns: int,
            callback: SampleFactorySandboxCallback,
        ) -> SampleFactorySandboxMetrics:
            return "not metrics"  # type: ignore[return-value]

    dynamics, arguments, episode_config, _ = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=_BadTrainer())
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=dynamics,
            arguments=arguments,
            episode_config=episode_config,
            ts_ns=1,
            proposal_id="p",
        )


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _run_once() -> SampleFactorySandboxResult:
    dynamics, arguments, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    return sandbox.train(
        dynamics=dynamics,
        arguments=arguments,
        episode_config=episode_config,
        ts_ns=42,
        proposal_id="canonical_patch",
    )


def test_inv15_three_run_byte_identical_replay() -> None:
    r1 = _run_once()
    r2 = _run_once()
    r3 = _run_once()
    assert r1.policy_digest == r2.policy_digest == r3.policy_digest
    assert r1.proposal == r2.proposal == r3.proposal
    assert r1.metrics == r2.metrics == r3.metrics


def test_inv15_digest_changes_when_seed_changes() -> None:
    dynamics, _, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    r0 = sandbox.train(
        dynamics=dynamics,
        arguments=_valid_args(random_seed=0),
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    r1 = sandbox.train(
        dynamics=dynamics,
        arguments=_valid_args(random_seed=1),
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert r0.policy_digest != r1.policy_digest


def test_inv15_digest_changes_when_num_workers_changes() -> None:
    dynamics, _, episode_config, trainer = _sandbox_inputs()
    sandbox = SampleFactorySandbox(trainer=trainer)
    r_a = sandbox.train(
        dynamics=dynamics,
        arguments=_valid_args(num_workers=2),
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    r_b = sandbox.train(
        dynamics=dynamics,
        arguments=_valid_args(num_workers=4),
        episode_config=episode_config,
        ts_ns=1,
        proposal_id="p",
    )
    assert r_a.policy_digest != r_b.policy_digest


def test_inv15_digest_is_blake2b_16_hex() -> None:
    r = _run_once()
    assert len(r.policy_digest) == 16
    assert re.fullmatch(r"[0-9a-f]{16}", r.policy_digest)
    h = hashlib.blake2b(b"smoke", digest_size=8).hexdigest()
    assert len(h) == 16


# ---------------------------------------------------------------------------
# null_sample_factory_callback
# ---------------------------------------------------------------------------


def test_null_callback_satisfies_protocol() -> None:
    cb = null_sample_factory_callback()
    assert isinstance(cb, SampleFactorySandboxCallback)


def test_null_callback_methods_return_none() -> None:
    cb = null_sample_factory_callback()
    obs = Observation(
        step_idx=0,
        mid_price=100.0,
        inventory_signed=0,
        cumulative_pnl_usd=0.0,
        state_hash="0" * 16,
    )
    metrics = _valid_metrics()
    assert cb.on_training_start(ts_ns=0, train_for_env_steps=1) is None
    assert (
        cb.on_step(
            ts_ns=0,
            step_idx=0,
            observation=obs,
            action=TradeAction.HOLD,
            reward=0.0,
        )
        is None
    )
    assert (
        cb.on_episode_end(
            ts_ns=0,
            episode_idx=0,
            episode_reward=0.0,
            episode_length=0,
        )
        is None
    )
    assert cb.on_training_end(ts_ns=0, metrics=metrics) is None


# ---------------------------------------------------------------------------
# Convenience factories raise when sample-factory missing
# ---------------------------------------------------------------------------


def test_sample_factory_appo_trainer_raises_when_dep_missing() -> None:
    try:
        importlib.import_module("sample_factory")
    except ImportError:
        with pytest.raises(ImportError, match="sample"):
            sample_factory_appo_trainer()
    else:
        pytest.skip("sample_factory installed — production seam smoke skipped")


def test_artifact_aliases_are_callable_typed() -> None:
    sink: PolicyArtifactSink = lambda blob: None  # noqa: E731
    blob: PolicyArtifact = b"hi"
    assert sink(blob) is None


# ---------------------------------------------------------------------------
# AST guards — OFFLINE_ONLY tier
# ---------------------------------------------------------------------------


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "evolution_engine" / "sandbox_sample_factory.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_sample_factory_import() -> None:
    assert all(not name.startswith("sample_factory") for name in _top_level_imports(_module_ast()))


def test_no_top_level_torch_import() -> None:
    assert all(not name.startswith("torch") for name in _top_level_imports(_module_ast()))


def test_no_top_level_numpy_import() -> None:
    assert all(not name.startswith("numpy") for name in _top_level_imports(_module_ast()))


def test_no_top_level_gymnasium_import() -> None:
    assert all(
        not name.startswith("gymnasium") and not name == "gym"
        for name in _top_level_imports(_module_ast())
    )


def test_no_top_level_io_imports() -> None:
    banned = {"subprocess", "socket", "urllib", "requests", "httpx", "aiohttp"}
    assert not (banned & set(_top_level_imports(_module_ast())))


def test_no_engine_cross_imports_at_top_level() -> None:
    banned_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
    )
    for name in _top_level_imports(_module_ast()):
        for prefix in banned_prefixes:
            assert not name.startswith(prefix), name


def test_no_engine_cross_imports_in_code() -> None:
    tree = _module_ast()
    code_only_segments: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Name)):
            code_only_segments.append(ast.dump(node))
    blob = "\n".join(code_only_segments)
    for needle in (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "registry",
    ):
        assert needle not in blob, needle


def test_sample_factory_import_only_inside_factory() -> None:
    """sample_factory may only be imported inside sample_factory_appo_trainer."""
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
            for name in names:
                if name.startswith(("sample_factory", "torch", "gymnasium")):
                    parent = _find_enclosing_function(tree, node)
                    assert parent is not None, (
                        f"top-level {name} import — must be inside "
                        "sample_factory_appo_trainer factory"
                    )
                    assert parent.name == "sample_factory_appo_trainer", (
                        f"{name} imported in {parent.name!r} — must be "
                        "inside sample_factory_appo_trainer"
                    )


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> ast.FunctionDef | None:
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef):
            for descendant in ast.walk(func):
                if descendant is target:
                    return func
    return None


# ---------------------------------------------------------------------------
# Module reload idempotency (no global state)
# ---------------------------------------------------------------------------


def test_module_reload_is_idempotent() -> None:
    import evolution_engine.sandbox_sample_factory as mod1

    importlib.reload(mod1)
    import evolution_engine.sandbox_sample_factory as mod2

    assert mod1.PROPOSAL_SOURCE == mod2.PROPOSAL_SOURCE
    assert mod1.MAX_TRAIN_FOR_ENV_STEPS == mod2.MAX_TRAIN_FOR_ENV_STEPS
    assert mod1.SampleFactoryAlgoKind.APPO is mod2.SampleFactoryAlgoKind.APPO


def test_marketdynamics_protocol_runtime_check_holds() -> None:
    assert isinstance(_FakeDynamics(), MarketDynamics)
