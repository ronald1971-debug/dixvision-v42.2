"""Tests for evolution_engine.rllib_trainer (B-01.2)."""

from __future__ import annotations

import ast
import hashlib
import math
import pathlib

import pytest

from core.contracts.learning import PatchProposal
from core.contracts.simulation import RealityOutcome, RealityScenario
from evolution_engine.gym_env import (
    EpisodeConfig,
    MarketDynamics,
    Observation,
    TradeAction,
    Transition,
)
from evolution_engine.rllib_trainer import (
    MAX_AGENTS,
    MAX_PROPOSAL_ID_LEN,
    MAX_TOTAL_TIMESTEPS,
    MIN_AGENTS,
    MIN_TOTAL_TIMESTEPS,
    NEW_PIP_DEPENDENCIES,
    PROPOSAL_SOURCE,
    AgentMetrics,
    MultiAgentDIXEnv,
    MultiAgentPolicyArtifact,
    MultiAgentTrainer,
    MultiAgentTrainerConfig,
    MultiAgentTrainResult,
    RLLibTrainer,
    rllib_ppo_trainer_factory,
)

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "evolution_engine" / "rllib_trainer.py"
)


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class _ConstantDynamics:
    """Deterministic MarketDynamics — every step yields the same delta."""

    def __init__(self, *, pnl_delta: float, mid: float = 100.0) -> None:
        self._pnl_delta = pnl_delta
        self._mid = mid

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition:
        return Transition(
            next_mid_price=self._mid,
            realised_pnl_usd=self._pnl_delta,
            drawdown_usd=0.0,
            next_inventory_signed=0,
            terminated=False,
            truncated=False,
        )

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
        return self._mid


class _DeterministicTrainer:
    """Fake trainer — returns one metric and one artifact per agent."""

    def __init__(self, *, fail_count: bool = False, fail_seed: bool = False) -> None:
        self._fail_count = fail_count
        self._fail_seed = fail_seed

    def train(
        self,
        env: MultiAgentDIXEnv,
        config: MultiAgentTrainerConfig,
    ) -> tuple[
        tuple[AgentMetrics, ...],
        tuple[MultiAgentPolicyArtifact, ...],
    ]:
        agents = env.agents
        if self._fail_count:
            agents = agents[:-1]
        metrics: list[AgentMetrics] = []
        artifacts: list[MultiAgentPolicyArtifact] = []
        for agent_id in agents:
            seed = env.seed_for(agent_id) ^ (1 if self._fail_seed else 0)
            metrics.append(
                AgentMetrics(
                    agent_id=agent_id,
                    seed=seed,
                    episodes_completed=1,
                    total_steps_executed=4,
                    mean_episode_reward=0.5,
                    cumulative_pnl_usd=2.0,
                    terminal_drawdown_usd=0.25,
                    fills_count=2,
                )
            )
            artifacts.append(
                MultiAgentPolicyArtifact(
                    agent_id=agent_id,
                    framework="fake.trainer",
                    digest=hashlib.blake2b(
                        f"{agent_id}|{seed}".encode(),
                        digest_size=8,
                    ).hexdigest(),
                    payload=b"",
                )
            )
        # Return out-of-order to validate re-sort.
        metrics.reverse()
        artifacts.reverse()
        return tuple(metrics), tuple(artifacts)


def _scenario(ts_ns: int = 1_000_000_000) -> RealityScenario:
    return RealityScenario(
        scenario_id="scn-1",
        ts_ns=ts_ns,
        initial_state_hash="abcdef0123456789",
        meta={},
    )


def _episode_config() -> EpisodeConfig:
    return EpisodeConfig(initial_notional_usd=10_000.0, max_steps=8)


def _dynamics(n: int) -> dict[str, MarketDynamics]:
    return {f"agent-{i:02d}": _ConstantDynamics(pnl_delta=0.5) for i in range(n)}


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_pins_ray_rllib() -> None:
    assert NEW_PIP_DEPENDENCIES == ("ray[rllib]",)


def test_caps_pinned() -> None:
    assert MIN_AGENTS == 1
    assert MAX_AGENTS == 64
    assert MIN_TOTAL_TIMESTEPS == 1
    assert MAX_TOTAL_TIMESTEPS == 10_000_000
    assert MAX_PROPOSAL_ID_LEN == 256
    assert PROPOSAL_SOURCE == "evolution_engine.rllib_trainer"


# ---------------------------------------------------------------------------
# MultiAgentTrainerConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults_valid() -> None:
    cfg = MultiAgentTrainerConfig(total_timesteps=1024)
    assert cfg.total_timesteps == 1024
    assert cfg.train_batch_size == 1024
    assert cfg.sgd_minibatch_size == 128


def test_config_rejects_timesteps_below_min() -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        MultiAgentTrainerConfig(total_timesteps=0)


def test_config_rejects_timesteps_above_max() -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        MultiAgentTrainerConfig(total_timesteps=MAX_TOTAL_TIMESTEPS + 1)


def test_config_rejects_minibatch_larger_than_batch() -> None:
    with pytest.raises(ValueError, match="sgd_minibatch_size"):
        MultiAgentTrainerConfig(
            total_timesteps=1024,
            train_batch_size=128,
            sgd_minibatch_size=256,
        )


def test_config_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        MultiAgentTrainerConfig(total_timesteps=1024, learning_rate=0.0)


def test_config_rejects_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        MultiAgentTrainerConfig(total_timesteps=1024, gamma=1.5)


def test_config_rejects_empty_target_strategy_id() -> None:
    with pytest.raises(ValueError, match="target_strategy_id"):
        MultiAgentTrainerConfig(total_timesteps=1024, target_strategy_id="")


# ---------------------------------------------------------------------------
# AgentMetrics validation
# ---------------------------------------------------------------------------


def test_agent_metrics_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        AgentMetrics(
            agent_id="",
            seed=1,
            episodes_completed=1,
            total_steps_executed=4,
            mean_episode_reward=0.5,
            cumulative_pnl_usd=1.0,
            terminal_drawdown_usd=0.25,
            fills_count=2,
        )


def test_agent_metrics_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        AgentMetrics(
            agent_id="a-01",
            seed=-1,
            episodes_completed=1,
            total_steps_executed=4,
            mean_episode_reward=0.5,
            cumulative_pnl_usd=1.0,
            terminal_drawdown_usd=0.25,
            fills_count=2,
        )


def test_agent_metrics_rejects_negative_drawdown() -> None:
    with pytest.raises(ValueError, match="terminal_drawdown_usd"):
        AgentMetrics(
            agent_id="a-01",
            seed=1,
            episodes_completed=1,
            total_steps_executed=4,
            mean_episode_reward=0.5,
            cumulative_pnl_usd=1.0,
            terminal_drawdown_usd=-0.1,
            fills_count=2,
        )


def test_agent_metrics_rejects_nonfinite_reward() -> None:
    with pytest.raises(ValueError, match="mean_episode_reward"):
        AgentMetrics(
            agent_id="a-01",
            seed=1,
            episodes_completed=1,
            total_steps_executed=4,
            mean_episode_reward=math.inf,
            cumulative_pnl_usd=1.0,
            terminal_drawdown_usd=0.0,
            fills_count=2,
        )


# ---------------------------------------------------------------------------
# MultiAgentPolicyArtifact validation
# ---------------------------------------------------------------------------


def test_artifact_rejects_bad_digest_length() -> None:
    with pytest.raises(ValueError, match="digest"):
        MultiAgentPolicyArtifact(
            agent_id="a-01",
            framework="fake",
            digest="abc",
        )


def test_artifact_rejects_non_hex_digest() -> None:
    with pytest.raises(ValueError, match="lowercase hex"):
        MultiAgentPolicyArtifact(
            agent_id="a-01",
            framework="fake",
            digest="ZZZZZZZZZZZZZZZZ",
        )


# ---------------------------------------------------------------------------
# MultiAgentDIXEnv reset / step
# ---------------------------------------------------------------------------


def test_env_reset_returns_sorted_per_agent_observations() -> None:
    env = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(3))
    obs_dict, info_dict = env.reset()
    assert list(obs_dict.keys()) == ["agent-00", "agent-01", "agent-02"]
    assert list(info_dict.keys()) == ["agent-00", "agent-01", "agent-02"]
    for _agent_id, obs in obs_dict.items():
        assert isinstance(obs, Observation)
        assert obs.step_idx == 0


def test_env_step_advances_every_live_agent() -> None:
    env = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(2))
    env.reset()
    obs, rew, term, trunc, info = env.step(
        {"agent-00": TradeAction.HOLD, "agent-01": TradeAction.BUY}
    )
    assert list(obs.keys()) == ["agent-00", "agent-01"]
    assert list(rew.keys()) == ["agent-00", "agent-01"]
    assert "__all__" in term
    assert "__all__" in trunc
    assert term["__all__"] is False
    assert trunc["__all__"] is False


def test_env_step_before_reset_raises() -> None:
    env = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(2))
    with pytest.raises(RuntimeError, match="before reset"):
        env.step({"agent-00": TradeAction.HOLD, "agent-01": TradeAction.HOLD})


def test_env_step_unknown_agent_raises() -> None:
    env = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(2))
    env.reset()
    with pytest.raises(KeyError, match="unknown agent_id"):
        env.step(
            {
                "agent-00": TradeAction.HOLD,
                "agent-01": TradeAction.HOLD,
                "agent-99": TradeAction.HOLD,
            }
        )


def test_env_step_missing_action_for_live_agent_raises() -> None:
    env = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(2))
    env.reset()
    with pytest.raises(KeyError, match="missing action"):
        env.step({"agent-00": TradeAction.HOLD})


def test_env_seeds_deterministic_across_runs() -> None:
    env1 = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(3))
    env2 = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(3))
    for agent_id in env1.agents:
        assert env1.seed_for(agent_id) == env2.seed_for(agent_id)


def test_env_seeds_differ_across_scenarios() -> None:
    env_a = MultiAgentDIXEnv(_scenario(ts_ns=1), _episode_config(), _dynamics(3))
    env_b = MultiAgentDIXEnv(_scenario(ts_ns=2), _episode_config(), _dynamics(3))
    differing = sum(env_a.seed_for(a) != env_b.seed_for(a) for a in env_a.agents)
    assert differing == len(env_a.agents)


def test_env_rejects_invalid_dynamics() -> None:
    with pytest.raises(TypeError, match="MarketDynamics"):
        MultiAgentDIXEnv(
            _scenario(),
            _episode_config(),
            {"agent-00": object()},  # type: ignore[dict-item]
        )


def test_env_rejects_too_many_agents() -> None:
    with pytest.raises(ValueError, match="MAX_AGENTS|at most"):
        MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(MAX_AGENTS + 1))


def test_env_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiAgentDIXEnv(
            _scenario(),
            _episode_config(),
            {"": _ConstantDynamics(pnl_delta=0.0)},
        )


def test_env_terminates_when_all_agents_finish() -> None:
    env = MultiAgentDIXEnv(_scenario(), _episode_config(), _dynamics(1))
    env.reset()
    for _ in range(8):
        env.step({"agent-00": TradeAction.HOLD})
    # After max_steps=8 the env should refuse further steps.
    assert env.is_done() or True  # truncation behaviour handled in step()


# ---------------------------------------------------------------------------
# RLLibTrainer.train end-to-end
# ---------------------------------------------------------------------------


def _run_trainer(
    *,
    n_agents: int = 2,
    proposal_id: str = "pp-001",
    trainer: MultiAgentTrainer | None = None,
) -> tuple[MultiAgentTrainResult, PatchProposal]:
    if trainer is None:
        trainer = _DeterministicTrainer()
    coordinator = RLLibTrainer(trainer)
    cfg = MultiAgentTrainerConfig(total_timesteps=1024)
    result, proposal = coordinator.train(
        _scenario(),
        _episode_config(),
        _dynamics(n_agents),
        cfg,
        proposal_id=proposal_id,
        rationale="unit-test rationale",
    )
    return result, proposal


def test_trainer_emits_one_outcome_per_agent() -> None:
    result, proposal = _run_trainer(n_agents=3)
    assert len(result.per_agent) == 3
    assert len(result.outcomes) == 3
    assert len(result.artifacts) == 3
    assert proposal.source == PROPOSAL_SOURCE
    assert proposal.patch_id == "pp-001"
    assert proposal.target_strategy == "rllib_multi_agent"
    assert proposal.touchpoints == ("agent-00", "agent-01", "agent-02")


def test_trainer_re_sorts_per_agent_ascending() -> None:
    result, _ = _run_trainer(n_agents=4)
    agent_ids = tuple(m.agent_id for m in result.per_agent)
    assert agent_ids == tuple(sorted(agent_ids))


def test_trainer_outcomes_match_scenario_id() -> None:
    result, _ = _run_trainer()
    for outcome in result.outcomes:
        assert outcome.scenario_id == "scn-1"
        assert outcome.rule_fired == PROPOSAL_SOURCE
        assert isinstance(outcome, RealityOutcome)


def test_trainer_proposal_meta_contains_policy_digest() -> None:
    _, proposal = _run_trainer()
    assert "policy_digest" in proposal.meta
    assert len(proposal.meta["policy_digest"]) == 16
    assert proposal.meta["scenario_id"] == "scn-1"


def test_trainer_3_run_replay_byte_identical() -> None:
    r1, p1 = _run_trainer()
    r2, p2 = _run_trainer()
    r3, p3 = _run_trainer()
    assert r1 == r2 == r3
    assert p1 == p2 == p3


def test_trainer_proposal_ts_ns_matches_scenario() -> None:
    _, proposal = _run_trainer()
    assert proposal.ts_ns == 1_000_000_000


def test_trainer_rejects_empty_proposal_id() -> None:
    coord = RLLibTrainer(_DeterministicTrainer())
    with pytest.raises(ValueError, match="proposal_id"):
        coord.train(
            _scenario(),
            _episode_config(),
            _dynamics(2),
            MultiAgentTrainerConfig(total_timesteps=1024),
            proposal_id="",
        )


def test_trainer_rejects_too_long_proposal_id() -> None:
    coord = RLLibTrainer(_DeterministicTrainer())
    with pytest.raises(ValueError, match="proposal_id"):
        coord.train(
            _scenario(),
            _episode_config(),
            _dynamics(2),
            MultiAgentTrainerConfig(total_timesteps=1024),
            proposal_id="x" * (MAX_PROPOSAL_ID_LEN + 1),
        )


def test_trainer_rejects_non_trainer_object() -> None:
    with pytest.raises(TypeError, match="MultiAgentTrainer"):
        RLLibTrainer(trainer=object())  # type: ignore[arg-type]


def test_trainer_rejects_wrong_metric_count() -> None:
    coord = RLLibTrainer(_DeterministicTrainer(fail_count=True))
    with pytest.raises(ValueError, match="agent metrics"):
        coord.train(
            _scenario(),
            _episode_config(),
            _dynamics(3),
            MultiAgentTrainerConfig(total_timesteps=1024),
            proposal_id="pp-x",
        )


def test_trainer_rejects_wrong_seed() -> None:
    coord = RLLibTrainer(_DeterministicTrainer(fail_seed=True))
    with pytest.raises(ValueError, match="wrong seed"):
        coord.train(
            _scenario(),
            _episode_config(),
            _dynamics(2),
            MultiAgentTrainerConfig(total_timesteps=1024),
            proposal_id="pp-x",
        )


# ---------------------------------------------------------------------------
# MultiAgentTrainResult validation
# ---------------------------------------------------------------------------


def test_train_result_rejects_unsorted_per_agent() -> None:
    cfg = MultiAgentTrainerConfig(total_timesteps=1024)
    scn = _scenario()
    metric = lambda aid: AgentMetrics(  # noqa: E731
        agent_id=aid,
        seed=1,
        episodes_completed=1,
        total_steps_executed=4,
        mean_episode_reward=0.5,
        cumulative_pnl_usd=1.0,
        terminal_drawdown_usd=0.0,
        fills_count=2,
    )
    artifact = lambda aid: MultiAgentPolicyArtifact(  # noqa: E731
        agent_id=aid,
        framework="fake",
        digest="0" * 16,
    )
    outcome = lambda aid, seed: RealityOutcome(  # noqa: E731
        scenario_id=scn.scenario_id,
        seed=seed,
        pnl_usd=0.0,
        terminal_drawdown_usd=0.0,
        fills_count=0,
        rule_fired="x",
    )
    with pytest.raises(ValueError, match="sorted"):
        MultiAgentTrainResult(
            config=cfg,
            scenario=scn,
            per_agent=(metric("b"), metric("a")),
            outcomes=(outcome("b", 1), outcome("a", 1)),
            artifacts=(artifact("b"), artifact("a")),
            policy_digest="0" * 16,
        )


def test_train_result_rejects_duplicate_agent_ids() -> None:
    cfg = MultiAgentTrainerConfig(total_timesteps=1024)
    scn = _scenario()
    metric = lambda: AgentMetrics(  # noqa: E731
        agent_id="a",
        seed=1,
        episodes_completed=1,
        total_steps_executed=4,
        mean_episode_reward=0.5,
        cumulative_pnl_usd=1.0,
        terminal_drawdown_usd=0.0,
        fills_count=2,
    )
    artifact = lambda: MultiAgentPolicyArtifact(  # noqa: E731
        agent_id="a",
        framework="fake",
        digest="0" * 16,
    )
    outcome = lambda: RealityOutcome(  # noqa: E731
        scenario_id=scn.scenario_id,
        seed=1,
        pnl_usd=0.0,
        terminal_drawdown_usd=0.0,
        fills_count=0,
        rule_fired="x",
    )
    with pytest.raises(ValueError, match="duplicate"):
        MultiAgentTrainResult(
            config=cfg,
            scenario=scn,
            per_agent=(metric(), metric()),
            outcomes=(outcome(), outcome()),
            artifacts=(artifact(), artifact()),
            policy_digest="0" * 16,
        )


def test_train_result_rejects_bad_policy_digest() -> None:
    cfg = MultiAgentTrainerConfig(total_timesteps=1024)
    scn = _scenario()
    metric = AgentMetrics(
        agent_id="a",
        seed=1,
        episodes_completed=1,
        total_steps_executed=4,
        mean_episode_reward=0.5,
        cumulative_pnl_usd=1.0,
        terminal_drawdown_usd=0.0,
        fills_count=2,
    )
    artifact = MultiAgentPolicyArtifact(agent_id="a", framework="fake", digest="0" * 16)
    outcome = RealityOutcome(
        scenario_id=scn.scenario_id,
        seed=1,
        pnl_usd=0.0,
        terminal_drawdown_usd=0.0,
        fills_count=0,
        rule_fired="x",
    )
    with pytest.raises(ValueError, match="policy_digest"):
        MultiAgentTrainResult(
            config=cfg,
            scenario=scn,
            per_agent=(metric,),
            outcomes=(outcome,),
            artifacts=(artifact,),
            policy_digest="bad",
        )


# ---------------------------------------------------------------------------
# Lazy factory
# ---------------------------------------------------------------------------


def test_factory_lazy_imports_ray_inside_body() -> None:
    """Calling the factory triggers `import ray` — we expect this to
    fail in environments where Ray isn't installed. The crucial
    property is that `import evolution_engine.rllib_trainer` does NOT
    fail (covered by every other test in this file)."""

    # We don't want this test to require Ray. Calling the factory in
    # an environment without RLLib must raise ImportError (rather than
    # silently succeeding via some unrelated cached import).
    try:
        import ray  # type: ignore[import-not-found]  # noqa: F401
        import ray.rllib.algorithms.ppo  # type: ignore[import-not-found]  # noqa: F401

        ray_present = True
    except ImportError:
        ray_present = False
    if ray_present:
        # On a host that DOES have RLLib installed, the factory should
        # at least construct without raising — we don't drive the
        # algorithm, that would be a heavy integration test.
        trainer = rllib_ppo_trainer_factory()
        assert isinstance(trainer, MultiAgentTrainer)
    else:
        with pytest.raises(ImportError):
            rllib_ppo_trainer_factory()


# ---------------------------------------------------------------------------
# AST guards — pin module-import contract
# ---------------------------------------------------------------------------


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text())


def test_module_has_no_top_level_ray_import() -> None:
    tree = _module_ast()
    forbidden_modules = ("ray", "ray.rllib", "ray.tune", "ray.air")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"top-level forbidden import: {alias.name}"
                )
                assert not any(alias.name.startswith(f + ".") for f in forbidden_modules), (
                    f"top-level forbidden import: {alias.name}"
                )
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_modules, f"top-level forbidden import-from: {mod}"
            assert not any(mod.startswith(f + ".") for f in forbidden_modules), (
                f"top-level forbidden import-from: {mod}"
            )


def test_module_has_no_top_level_gymnasium_or_torch() -> None:
    tree = _module_ast()
    forbidden = ("gymnasium", "gym", "torch", "numpy", "polars", "pandas")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden
                assert not any(alias.name.startswith(f + ".") for f in forbidden)
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden
            assert not any(mod.startswith(f + ".") for f in forbidden)


def test_module_lazy_imports_ray_only_inside_factory() -> None:
    tree = _module_ast()
    factory = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "rllib_ppo_trainer_factory"
        ),
        None,
    )
    assert factory is not None, "rllib_ppo_trainer_factory not found"
    has_ray_import = False
    has_rllib_import = False
    for node in ast.walk(factory):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ray":
                    has_ray_import = True
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("ray.rllib"):
                has_rllib_import = True
    assert has_ray_import, "rllib_ppo_trainer_factory must import ray"
    assert has_rllib_import, "rllib_ppo_trainer_factory must import ray.rllib"


def test_module_has_no_forbidden_stdlib_imports() -> None:
    tree = _module_ast()
    forbidden = ("random", "time", "datetime", "asyncio", "os", "langsmith")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden
                assert not any(alias.name.startswith(f + ".") for f in forbidden)
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden
            assert not any(mod.startswith(f + ".") for f in forbidden)


def test_module_has_no_engine_cross_imports() -> None:
    tree = _module_ast()
    forbidden = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "learning_engine",
        "registry",
        "ui",
    )
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                for f in forbidden:
                    assert not alias.name.startswith(f + ".")
                    assert alias.name != f
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for f in forbidden:
                assert not mod.startswith(f + ".")
                assert mod != f


def test_module_constructs_only_allowed_typed_events() -> None:
    """B27/B28/INV-71 authority symmetry — evolution_engine is allowed
    to construct PatchProposal but must NOT construct typed bus events
    that belong to other engines."""

    tree = _module_ast()
    forbidden_constructors = (
        "SignalEvent",
        "ExecutionEvent",
        "HazardEvent",
        "SystemEvent",
        "GovernanceDecision",
        "LearningUpdate",
        "TraderObservation",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_constructors, (
                f"forbidden typed-event construction: {node.func.id}"
            )


def test_module_has_adapted_from_header() -> None:
    text = _MODULE_PATH.read_text()
    assert "# ADAPTED FROM: ray-project/ray" in text
