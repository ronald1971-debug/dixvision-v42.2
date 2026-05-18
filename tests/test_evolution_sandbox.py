"""Tests for ``evolution_engine.sandbox`` (A-01.2 stable-baselines3)."""

from __future__ import annotations

import ast
import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path

import pytest

from core.contracts.learning import PatchProposal
from evolution_engine.gym_env import (
    MAX_EPISODE_STEPS,
    DIXStrategyEnv,
    EpisodeConfig,
    MarketDynamics,
    Observation,
    TradeAction,
    Transition,
)
from evolution_engine.sandbox import (
    MAX_PROPOSAL_ID_LEN,
    MAX_TOTAL_TIMESTEPS,
    MIN_TOTAL_TIMESTEPS,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    EvolutionSandbox,
    PolicyTrainer,
    SandboxCallback,
    SandboxConfig,
    SandboxConfigError,
    SandboxMetrics,
    SandboxResult,
    null_sandbox_callback,
    sb3_ppo_trainer,
)

# ---------------------------------------------------------------------------
# Module-level metadata + AST authority pins
# ---------------------------------------------------------------------------

_MOD_PATH = Path(__file__).resolve().parents[1] / "evolution_engine" / "sandbox.py"


def test_module_path_exists() -> None:
    assert _MOD_PATH.exists(), _MOD_PATH


def test_new_pip_dependencies_is_frozen_tuple() -> None:
    assert isinstance(NEW_PIP_DEPENDENCIES, tuple)
    assert NEW_PIP_DEPENDENCIES == ("gymnasium", "stable-baselines3")


def test_adapted_from_header_present() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    first_50 = "\n".join(src.splitlines()[:6])
    assert "ADAPTED FROM:" in first_50
    assert "stable-baselines3" in first_50.lower()


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module)
    return names


def test_no_top_level_sb3_or_gymnasium_import() -> None:
    """The whole point of the lazy-import seam: SB3 and gymnasium
    must not be imported at module load."""

    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {
        "stable_baselines3",
        "gymnasium",
        "gym",
        "torch",
        "numpy",
    }
    assert not (_top_level_imports(tree) & forbidden)


def test_no_top_level_clock_or_io_imports() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {
        "time",
        "datetime",
        "asyncio",
        "os",
        "secrets",
        "uuid",
        "random",
        "subprocess",
        "socket",
    }
    assert not (_top_level_imports(tree) & forbidden)


def test_no_engine_cross_imports() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
        "cockpit.",
        "dashboard2026.",
        "dashboard_backend.",
    )
    for name in _top_level_imports(tree):
        assert not name.startswith(forbidden_prefixes), name


def test_module_imports_only_from_core_contracts_and_evolution_gym_env() -> None:
    """Positive pin: top-level cross-package imports are restricted to
    ``core.contracts.learning`` (for ``PatchProposal``) and
    ``evolution_engine.gym_env`` (the A-01.1 leaf)."""

    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cross_pkg_imports = {
        n
        for n in _top_level_imports(tree)
        if "." in n and not n.startswith(("collections", "typing"))
    }
    assert cross_pkg_imports == {
        "core.contracts.learning",
        "evolution_engine.gym_env",
    }


def test_constants_are_in_band() -> None:
    assert MAX_TOTAL_TIMESTEPS == 10_000_000
    assert MIN_TOTAL_TIMESTEPS == 1
    assert MAX_PROPOSAL_ID_LEN == 256
    assert PROPOSAL_SOURCE == "evolution_engine.sandbox"


# ---------------------------------------------------------------------------
# SandboxConfig validation
# ---------------------------------------------------------------------------


def _good_config(**overrides: object) -> SandboxConfig:
    base: dict[str, object] = {
        "total_timesteps": 1024,
        "n_steps": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "target_strategy_id": "rl_trained",
        "meta": {},
    }
    base.update(overrides)
    return SandboxConfig(**base)  # type: ignore[arg-type]


def test_sandbox_config_is_frozen() -> None:
    cfg = _good_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.total_timesteps = 9_999  # type: ignore[misc]


def test_sandbox_config_is_slotted() -> None:
    cfg = _good_config()
    assert hasattr(type(cfg), "__slots__")
    assert not hasattr(cfg, "__dict__")


def test_sandbox_config_total_timesteps_lower_bound() -> None:
    with pytest.raises(ValueError):
        _good_config(total_timesteps=0)


def test_sandbox_config_total_timesteps_upper_bound() -> None:
    with pytest.raises(ValueError):
        _good_config(total_timesteps=MAX_TOTAL_TIMESTEPS + 1)


def test_sandbox_config_n_steps_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _good_config(n_steps=0)


def test_sandbox_config_learning_rate_must_be_positive_finite() -> None:
    with pytest.raises(ValueError):
        _good_config(learning_rate=0.0)
    with pytest.raises(ValueError):
        _good_config(learning_rate=float("nan"))
    with pytest.raises(ValueError):
        _good_config(learning_rate=float("inf"))


def test_sandbox_config_gamma_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        _good_config(gamma=0.0)
    with pytest.raises(ValueError):
        _good_config(gamma=1.5)
    with pytest.raises(ValueError):
        _good_config(gamma=float("nan"))
    # Boundary: gamma=1.0 is permitted.
    _good_config(gamma=1.0)


def test_sandbox_config_target_strategy_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        _good_config(target_strategy_id="")


# ---------------------------------------------------------------------------
# SandboxMetrics validation
# ---------------------------------------------------------------------------


def _good_metrics(**overrides: object) -> SandboxMetrics:
    base: dict[str, object] = {
        "episodes_completed": 4,
        "total_steps_executed": 1024,
        "mean_episode_reward": 1.5,
        "mean_episode_length": 256.0,
        "best_episode_reward": 2.5,
    }
    base.update(overrides)
    return SandboxMetrics(**base)  # type: ignore[arg-type]


def test_sandbox_metrics_is_frozen_and_slotted() -> None:
    m = _good_metrics()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.episodes_completed = 99  # type: ignore[misc]
    assert hasattr(type(m), "__slots__")
    assert not hasattr(m, "__dict__")


def test_sandbox_metrics_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        _good_metrics(episodes_completed=-1)
    with pytest.raises(ValueError):
        _good_metrics(total_steps_executed=-1)


def test_sandbox_metrics_rejects_non_finite_rewards() -> None:
    with pytest.raises(ValueError):
        _good_metrics(mean_episode_reward=float("nan"))
    with pytest.raises(ValueError):
        _good_metrics(mean_episode_reward=float("inf"))
    with pytest.raises(ValueError):
        _good_metrics(best_episode_reward=float("nan"))


def test_sandbox_metrics_rejects_negative_mean_length() -> None:
    with pytest.raises(ValueError):
        _good_metrics(mean_episode_length=-0.1)
    with pytest.raises(ValueError):
        _good_metrics(mean_episode_length=float("nan"))


# ---------------------------------------------------------------------------
# Stub MarketDynamics + PolicyTrainer
# ---------------------------------------------------------------------------


class _LinearWalkDynamics:
    """Tiny deterministic dynamics for testing.

    Mid-price walks +1.0 on BUY, -1.0 on SELL, 0.0 on HOLD; PnL is
    the same delta. Pure function of ``(seed, action sequence)``."""

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
        return 100.0 + (seed % 10)

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition:
        if action == TradeAction.BUY:
            delta = 1.0
            inv = 1
        elif action == TradeAction.SELL:
            delta = -1.0
            inv = -1
        else:
            delta = 0.0
            inv = 0
        next_mid = prev_obs.mid_price + delta
        return Transition(
            next_mid_price=next_mid,
            realised_pnl_usd=delta,
            drawdown_usd=0.0,
            next_inventory_signed=inv,
            terminated=False,
            truncated=False,
        )


class _DeterministicPolicyTrainer:
    """Walks a fixed BUY-action policy through the env."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    def train(
        self,
        env: DIXStrategyEnv,
        *,
        episode_config: EpisodeConfig,
        total_timesteps: int,
        seed: int,
        ts_ns: int,
        callback: SandboxCallback,
    ) -> SandboxMetrics:
        self.calls.append(
            {
                "total_timesteps": total_timesteps,
                "seed": seed,
                "ts_ns": ts_ns,
            }
        )
        env.reset(seed=seed, config=episode_config)
        episodes = 0
        ep_reward = 0.0
        ep_len = 0
        rewards: list[float] = []
        lengths: list[int] = []
        best_reward = -math.inf
        steps_left = total_timesteps
        while steps_left > 0:
            obs, reward, terminated, truncated, info = env.step(TradeAction.BUY)
            assert isinstance(info, Mapping)
            ep_reward += reward
            ep_len += 1
            steps_left -= 1
            callback.on_step(
                ts_ns=ts_ns + ep_len,
                step_idx=ep_len,
                observation=obs,
                action=TradeAction.BUY,
                reward=reward,
            )
            if terminated or truncated:
                episodes += 1
                rewards.append(ep_reward)
                lengths.append(ep_len)
                best_reward = max(best_reward, ep_reward)
                callback.on_episode_end(
                    ts_ns=ts_ns + ep_len,
                    episode_idx=episodes,
                    episode_reward=ep_reward,
                    episode_length=ep_len,
                )
                if steps_left > 0:
                    env.reset(seed=seed + episodes, config=episode_config)
                ep_reward = 0.0
                ep_len = 0
        if not rewards:
            return SandboxMetrics(
                episodes_completed=0,
                total_steps_executed=total_timesteps,
                mean_episode_reward=0.0,
                mean_episode_length=0.0,
                best_episode_reward=0.0,
            )
        return SandboxMetrics(
            episodes_completed=episodes,
            total_steps_executed=total_timesteps,
            mean_episode_reward=sum(rewards) / len(rewards),
            mean_episode_length=sum(lengths) / len(lengths),
            best_episode_reward=best_reward,
        )


class _RecordingCallback:
    __slots__ = (
        "training_starts",
        "steps",
        "episode_ends",
        "training_ends",
    )

    def __init__(self) -> None:
        self.training_starts: list[Mapping[str, object]] = []
        self.steps: list[Mapping[str, object]] = []
        self.episode_ends: list[Mapping[str, object]] = []
        self.training_ends: list[Mapping[str, object]] = []

    def on_training_start(self, *, ts_ns: int, total_timesteps: int) -> None:
        self.training_starts.append({"ts_ns": ts_ns, "total_timesteps": total_timesteps})

    def on_step(
        self,
        *,
        ts_ns: int,
        step_idx: int,
        observation: Observation,
        action: TradeAction,
        reward: float,
    ) -> None:
        self.steps.append(
            {
                "ts_ns": ts_ns,
                "step_idx": step_idx,
                "reward": reward,
            }
        )

    def on_episode_end(
        self,
        *,
        ts_ns: int,
        episode_idx: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        self.episode_ends.append(
            {
                "ts_ns": ts_ns,
                "episode_idx": episode_idx,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
            }
        )

    def on_training_end(self, *, ts_ns: int, metrics: SandboxMetrics) -> None:
        self.training_ends.append(
            {
                "ts_ns": ts_ns,
                "episodes": metrics.episodes_completed,
            }
        )


# ---------------------------------------------------------------------------
# null_sandbox_callback
# ---------------------------------------------------------------------------


def test_null_sandbox_callback_implements_protocol() -> None:
    cb = null_sandbox_callback()
    assert isinstance(cb, SandboxCallback)
    obs = Observation(
        step_idx=0,
        mid_price=100.0,
        inventory_signed=0,
        cumulative_pnl_usd=0.0,
        state_hash="0123456789abcdef",
    )
    metrics = _good_metrics()
    cb.on_training_start(ts_ns=1, total_timesteps=100)
    cb.on_step(ts_ns=2, step_idx=1, observation=obs, action=TradeAction.HOLD, reward=0.0)
    cb.on_episode_end(ts_ns=3, episode_idx=1, episode_reward=0.0, episode_length=1)
    cb.on_training_end(ts_ns=4, metrics=metrics)


# ---------------------------------------------------------------------------
# EvolutionSandbox construction
# ---------------------------------------------------------------------------


def test_sandbox_is_frozen_and_slotted() -> None:
    s = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.trainer = _DeterministicPolicyTrainer()  # type: ignore[misc]
    assert hasattr(type(s), "__slots__")
    assert not hasattr(s, "__dict__")


def test_sandbox_rejects_non_protocol_trainer() -> None:
    class _NotATrainer:
        pass

    with pytest.raises(TypeError):
        EvolutionSandbox(_NotATrainer())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EvolutionSandbox.train happy path
# ---------------------------------------------------------------------------


def _episode_cfg(**overrides: object) -> EpisodeConfig:
    base: dict[str, object] = {
        "initial_notional_usd": 1_000.0,
        "max_steps": 16,
        "reward_scale": 1.0,
        "drawdown_penalty_weight": 0.0,
    }
    base.update(overrides)
    return EpisodeConfig(**base)  # type: ignore[arg-type]


def _run_default_sandbox(
    *,
    seed: int = 42,
    ts_ns: int = 1_000,
    proposal_id: str = "patch-001",
    callback: SandboxCallback | None = None,
) -> tuple[EvolutionSandbox, SandboxResult, _DeterministicPolicyTrainer]:
    trainer = _DeterministicPolicyTrainer()
    sandbox = EvolutionSandbox(trainer)
    cfg = _good_config(total_timesteps=64, n_steps=16)
    ep_cfg = _episode_cfg(max_steps=16)
    result = sandbox.train(
        dynamics=_LinearWalkDynamics(),
        config=cfg,
        episode_config=ep_cfg,
        seed=seed,
        ts_ns=ts_ns,
        proposal_id=proposal_id,
        callback=callback,
    )
    return sandbox, result, trainer


def test_train_returns_sandbox_result() -> None:
    _, result, _ = _run_default_sandbox()
    assert isinstance(result, SandboxResult)
    assert isinstance(result.proposal, PatchProposal)
    assert isinstance(result.metrics, SandboxMetrics)


def test_train_proposal_carries_canonical_source_tag() -> None:
    _, result, _ = _run_default_sandbox()
    assert result.proposal.source == PROPOSAL_SOURCE
    assert result.proposal.patch_id == "patch-001"
    assert result.proposal.target_strategy == "rl_trained"
    assert result.proposal.touchpoints == (
        "evolution_engine.sandbox",
        "policy_weights",
    )


def test_train_proposal_meta_has_provenance_fields() -> None:
    _, result, _ = _run_default_sandbox()
    meta = result.proposal.meta
    assert meta["policy_digest"] == result.policy_digest
    assert meta["seed"] == "42"
    assert meta["total_timesteps"] == "64"
    assert meta["episodes_completed"] == str(result.metrics.episodes_completed)
    assert "mean_episode_reward" in meta
    assert "best_episode_reward" in meta


def test_train_proposal_meta_user_overlay_does_not_overwrite_provenance() -> None:
    trainer = _DeterministicPolicyTrainer()
    sandbox = EvolutionSandbox(trainer)
    cfg = _good_config(
        total_timesteps=64,
        n_steps=16,
        meta={
            "policy_digest": "deadbeefdeadbeef",  # MUST be ignored
            "seed": "999",  # MUST be ignored
            "user_label": "canary-rerun",  # MUST be kept
        },
    )
    result = sandbox.train(
        dynamics=_LinearWalkDynamics(),
        config=cfg,
        episode_config=_episode_cfg(max_steps=16),
        seed=42,
        ts_ns=1_000,
        proposal_id="patch-001",
    )
    assert result.proposal.meta["policy_digest"] == result.policy_digest
    assert result.proposal.meta["seed"] == "42"
    assert result.proposal.meta["user_label"] == "canary-rerun"


def test_train_emits_callback_lifecycle() -> None:
    cb = _RecordingCallback()
    _, result, _ = _run_default_sandbox(callback=cb)
    assert len(cb.training_starts) == 1
    assert cb.training_starts[0] == {
        "ts_ns": 1_000,
        "total_timesteps": 64,
    }
    assert len(cb.training_ends) == 1
    assert cb.training_ends[0]["episodes"] == result.metrics.episodes_completed
    assert len(cb.steps) > 0
    assert len(cb.episode_ends) >= 1


def test_train_propagates_seed_and_ts_ns_to_trainer() -> None:
    _, _, trainer = _run_default_sandbox(seed=7, ts_ns=12345)
    assert trainer.calls[0]["seed"] == 7
    assert trainer.calls[0]["ts_ns"] == 12345
    assert trainer.calls[0]["total_timesteps"] == 64


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_train_rejects_non_market_dynamics() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())

    class _NotADynamics:
        pass

    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_NotADynamics(),  # type: ignore[arg-type]
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=1,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_non_sandbox_config() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config={"total_timesteps": 64},  # type: ignore[arg-type]
            episode_config=_episode_cfg(max_steps=16),
            seed=1,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_non_episode_config() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config={"max_steps": 16},  # type: ignore[arg-type]
            seed=1,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_negative_seed() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(SandboxConfigError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=-1,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_negative_ts_ns() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(SandboxConfigError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=0,
            ts_ns=-1,
            proposal_id="p",
        )


def test_train_rejects_bool_seed() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=True,  # type: ignore[arg-type]
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_bool_ts_ns() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=1,
            ts_ns=True,  # type: ignore[arg-type]
            proposal_id="p",
        )


def test_train_rejects_empty_proposal_id() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(SandboxConfigError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=0,
            ts_ns=1,
            proposal_id="",
        )


def test_train_rejects_oversized_proposal_id() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(SandboxConfigError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=0,
            ts_ns=1,
            proposal_id="x" * (MAX_PROPOSAL_ID_LEN + 1),
        )


def test_train_rejects_n_steps_exceeding_episode_max_steps() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())
    with pytest.raises(SandboxConfigError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=64),
            episode_config=_episode_cfg(max_steps=16),
            seed=0,
            ts_ns=1,
            proposal_id="p",
        )


def test_train_rejects_non_protocol_callback() -> None:
    sandbox = EvolutionSandbox(_DeterministicPolicyTrainer())

    class _BadCallback:
        # Missing all four lifecycle methods.
        pass

    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=0,
            ts_ns=1,
            proposal_id="p",
            callback=_BadCallback(),  # type: ignore[arg-type]
        )


def test_train_rejects_trainer_returning_wrong_type() -> None:
    class _BadTrainer:
        def train(
            self,
            env: DIXStrategyEnv,
            *,
            episode_config: EpisodeConfig,
            total_timesteps: int,
            seed: int,
            ts_ns: int,
            callback: SandboxCallback,
        ) -> SandboxMetrics:
            return "not a metrics record"  # type: ignore[return-value]

    sandbox = EvolutionSandbox(_BadTrainer())
    with pytest.raises(TypeError):
        sandbox.train(
            dynamics=_LinearWalkDynamics(),
            config=_good_config(total_timesteps=64, n_steps=16),
            episode_config=_episode_cfg(max_steps=16),
            seed=0,
            ts_ns=1,
            proposal_id="p",
        )


# ---------------------------------------------------------------------------
# Policy digest
# ---------------------------------------------------------------------------


def test_policy_digest_is_16_lower_hex() -> None:
    _, result, _ = _run_default_sandbox()
    assert len(result.policy_digest) == 16
    assert all(c in "0123456789abcdef" for c in result.policy_digest)


def test_policy_digest_changes_with_seed() -> None:
    _, result_a, _ = _run_default_sandbox(seed=1)
    _, result_b, _ = _run_default_sandbox(seed=2)
    assert result_a.policy_digest != result_b.policy_digest


def test_policy_digest_changes_with_ts_ns() -> None:
    _, result_a, _ = _run_default_sandbox(ts_ns=1_000)
    _, result_b, _ = _run_default_sandbox(ts_ns=2_000)
    assert result_a.policy_digest != result_b.policy_digest


def test_policy_digest_changes_with_proposal_id() -> None:
    _, result_a, _ = _run_default_sandbox(proposal_id="alpha")
    _, result_b, _ = _run_default_sandbox(proposal_id="bravo")
    assert result_a.policy_digest != result_b.policy_digest


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _run_canonical() -> SandboxResult:
    trainer = _DeterministicPolicyTrainer()
    sandbox = EvolutionSandbox(trainer)
    return sandbox.train(
        dynamics=_LinearWalkDynamics(),
        config=_good_config(
            total_timesteps=64,
            n_steps=16,
            meta={"label": "test-rep"},
        ),
        episode_config=_episode_cfg(max_steps=16),
        seed=12345,
        ts_ns=10_000,
        proposal_id="patch-rep",
    )


def test_inv15_three_run_byte_identical_replay() -> None:
    a = _run_canonical()
    b = _run_canonical()
    c = _run_canonical()
    assert a == b == c
    assert a.proposal == b.proposal == c.proposal
    assert a.metrics == b.metrics == c.metrics
    assert a.policy_digest == b.policy_digest == c.policy_digest


def test_inv15_different_seeds_diverge() -> None:
    trainer = _DeterministicPolicyTrainer()
    sandbox = EvolutionSandbox(trainer)
    cfg = _good_config(total_timesteps=64, n_steps=16)
    ep_cfg = _episode_cfg(max_steps=16)
    a = sandbox.train(
        dynamics=_LinearWalkDynamics(),
        config=cfg,
        episode_config=ep_cfg,
        seed=1,
        ts_ns=1_000,
        proposal_id="p",
    )
    b = sandbox.train(
        dynamics=_LinearWalkDynamics(),
        config=cfg,
        episode_config=ep_cfg,
        seed=2,
        ts_ns=1_000,
        proposal_id="p",
    )
    assert a != b
    assert a.policy_digest != b.policy_digest


# ---------------------------------------------------------------------------
# SandboxResult validation
# ---------------------------------------------------------------------------


def test_sandbox_result_rejects_non_proposal() -> None:
    with pytest.raises(TypeError):
        SandboxResult(
            proposal="not a PatchProposal",  # type: ignore[arg-type]
            metrics=_good_metrics(),
            policy_digest="0123456789abcdef",
        )


def test_sandbox_result_rejects_non_metrics() -> None:
    proposal = PatchProposal(
        ts_ns=1,
        patch_id="p",
        source=PROPOSAL_SOURCE,
        target_strategy="t",
        touchpoints=(),
        rationale="r",
    )
    with pytest.raises(TypeError):
        SandboxResult(
            proposal=proposal,
            metrics={"x": 1},  # type: ignore[arg-type]
            policy_digest="0123456789abcdef",
        )


def test_sandbox_result_rejects_wrong_digest_length() -> None:
    proposal = PatchProposal(
        ts_ns=1,
        patch_id="p",
        source=PROPOSAL_SOURCE,
        target_strategy="t",
        touchpoints=(),
        rationale="r",
    )
    with pytest.raises(ValueError):
        SandboxResult(
            proposal=proposal,
            metrics=_good_metrics(),
            policy_digest="abc",
        )


def test_sandbox_result_rejects_non_hex_digest() -> None:
    proposal = PatchProposal(
        ts_ns=1,
        patch_id="p",
        source=PROPOSAL_SOURCE,
        target_strategy="t",
        touchpoints=(),
        rationale="r",
    )
    with pytest.raises(ValueError):
        SandboxResult(
            proposal=proposal,
            metrics=_good_metrics(),
            policy_digest="ZZZZZZZZZZZZZZZZ",
        )


def test_sandbox_result_rejects_uppercase_hex() -> None:
    proposal = PatchProposal(
        ts_ns=1,
        patch_id="p",
        source=PROPOSAL_SOURCE,
        target_strategy="t",
        touchpoints=(),
        rationale="r",
    )
    with pytest.raises(ValueError):
        SandboxResult(
            proposal=proposal,
            metrics=_good_metrics(),
            policy_digest="0123456789ABCDEF",
        )


# ---------------------------------------------------------------------------
# Lazy SB3 factory
# ---------------------------------------------------------------------------


def test_sb3_ppo_trainer_factory_either_returns_or_raises_import_error() -> None:
    """The factory must either succeed (SB3 installed) or raise a
    helpful ImportError mentioning the package name."""

    try:
        trainer = sb3_ppo_trainer()
    except ImportError as exc:
        assert "stable-baselines3" in str(exc)
        return
    assert isinstance(trainer, PolicyTrainer)


def test_sb3_ppo_trainer_lazy_import_lives_inside_the_factory() -> None:
    """SB3 / gymnasium imports must be confined to the
    ``sb3_ppo_trainer`` factory body — never at module top level
    and never inside any other function."""

    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    factory_func: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sb3_ppo_trainer":
            factory_func = node
            break
    assert factory_func is not None

    factory_nodes = set(ast.walk(factory_func))

    forbidden = {"stable_baselines3", "gymnasium"}
    for node in ast.walk(tree):
        if node in factory_nodes:
            continue
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden, (
                f"{node.module!r} imported outside sb3_ppo_trainer at line {node.lineno}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, (
                    f"{alias.name!r} imported outside sb3_ppo_trainer at line {node.lineno}"
                )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_deterministic_trainer_implements_policy_trainer_protocol() -> None:
    assert isinstance(_DeterministicPolicyTrainer(), PolicyTrainer)


def test_recording_callback_implements_sandbox_callback_protocol() -> None:
    assert isinstance(_RecordingCallback(), SandboxCallback)


def test_linear_walk_implements_market_dynamics_protocol() -> None:
    assert isinstance(_LinearWalkDynamics(), MarketDynamics)


# ---------------------------------------------------------------------------
# Episode budget integration with gym_env
# ---------------------------------------------------------------------------


def test_train_respects_episode_max_steps_via_env_truncation() -> None:
    """The trainer's loop terminates each episode at
    ``EpisodeConfig.max_steps`` because the env raises
    ``truncated=True`` once the per-episode budget is hit."""

    trainer = _DeterministicPolicyTrainer()
    sandbox = EvolutionSandbox(trainer)
    cfg = _good_config(total_timesteps=12, n_steps=4)
    ep_cfg = _episode_cfg(max_steps=4)
    result = sandbox.train(
        dynamics=_LinearWalkDynamics(),
        config=cfg,
        episode_config=ep_cfg,
        seed=0,
        ts_ns=1,
        proposal_id="p",
    )
    assert result.metrics.episodes_completed == 3
    assert result.metrics.total_steps_executed == 12
    assert result.metrics.mean_episode_length == 4.0


def test_max_episode_steps_global_ceiling_is_imported() -> None:
    # Sanity pin: the gym env's global ceiling is reachable from
    # the sandbox test surface; the sandbox itself respects it
    # transitively via the env.
    assert MAX_EPISODE_STEPS == 100_000
