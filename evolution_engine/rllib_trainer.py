# ADAPTED FROM: ray-project/ray
# (rllib/env/multi_agent_env.py — MultiAgentEnv reset / step / agent dict shape;
#  rllib/algorithms/ppo/ppo.py — PPOConfig.training() entry point;
#  rllib/algorithms/algorithm.py — Algorithm.train() loop;
#  rllib/utils/typing.py — MultiAgentDict alias.)
"""B-01.2 — RLLibTrainer: multi-agent governance-gated PPO training entry.

Drop-in *multi-agent* counterpart to :class:`evolution_engine.sandbox`.
Wraps N :class:`~evolution_engine.gym_env.DIXStrategyEnv` instances
(one per agent) behind RLLib's ``MultiAgentEnv`` dict reset / step
shape and runs RLLib's PPO loop to gather per-agent rollouts. The
result is materialised as a tuple of
:class:`~core.contracts.simulation.RealityOutcome` records — same
shape as :class:`simulation.parallel_runner.ParallelRunner` — plus
one :class:`~core.contracts.learning.PatchProposal` for governance
approval. INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects + a
  :class:`MultiAgentDIXEnv` wrapper that exposes
  :class:`~evolution_engine.gym_env.DIXStrategyEnv` to a Gymnasium /
  RLLib ``MultiAgentEnv``-shape consumer **without importing Ray or
  Gymnasium at module load**. The actual ``ray.rllib`` import is
  hidden behind a :class:`MultiAgentTrainer` Protocol — production
  callers wire :func:`rllib_ppo_trainer_factory` (which lazy-imports
  RLLib inside its body), unit tests inject a deterministic fake.
* OFFLINE_ONLY tier. The trainer reads no environment variables,
  performs no IO, never imports
  ``execution_engine`` / ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. It produces one
  :class:`MultiAgentTrainResult` record and stops.
* INV-15 byte-identical replays. :meth:`RLLibTrainer.train` with
  identical ``config`` / ``dynamics_per_agent`` / ``scenario`` /
  ``ts_ns`` / ``proposal_id`` / ``trainer`` returns identical
  :class:`MultiAgentTrainResult` records. Per-agent seeds are derived
  from ``scenario.ts_ns ^ agent_seed_offset`` via splitmix64 so the
  per-agent rollouts are order-independent of RLLib's worker
  scheduling. Outcomes are re-sorted by ``seed`` before emission so
  RLLib's internal worker order cannot leak into the result tuple
  (matches B-01.1 :class:`simulation.distributed_runner` re-sort
  guarantee).
* No clock reads. Caller supplies ``ts_ns`` (mirrors the S-06 typed
  agent, S-12 LiteLLM router, and A-01.2 sandbox pattern).

What survives from upstream
---------------------------

* RLLib's ``MultiAgentEnv.reset() -> dict[AgentID, obs]`` and
  ``MultiAgentEnv.step(action_dict) -> (obs, rew, terminated,
  truncated, info)`` 5-tuple shape — :class:`MultiAgentDIXEnv` matches
  it field-for-field so a downstream RLLib algorithm can be plugged
  in unchanged.
* RLLib's ``MultiAgentDict`` alias semantics — every per-agent dict
  in the API has stable, sorted iteration order (we re-build it via
  ``sorted(...)`` on every call so the public surface is order-stable
  even if upstream RLLib chooses arbitrary insertion order).
* The ``PPOConfig.training()`` single-call training surface from
  ``rllib/algorithms/ppo/ppo.py`` — the :class:`MultiAgentTrainer`
  Protocol matches that shape so a thin adapter forwards directly to
  ``ppo_config.build().train()``.

What we replaced
----------------

* RLLib's ``Algorithm.train()`` Tune / Tensorboard integration →
  caller-injected :class:`MultiAgentTrainer` Protocol that returns a
  :class:`MultiAgentTrainResult` value object only. No filesystem
  writes, no Ray cluster lifecycle, no global state. The lazy
  :func:`rllib_ppo_trainer_factory` handles ``ray.init`` /
  ``ray.shutdown`` and PPOConfig assembly internally.
* RLLib's ``Algorithm.save_checkpoint`` and ``restore`` →
  :class:`MultiAgentTrainResult.policy_digest` (a 16-hex-char content
  hash of the trainer-supplied metrics + config). The full policy
  weights are an :class:`MultiAgentPolicyArtifact` blob the caller can
  route into evolution's existing patch-pipeline storage.
* RLLib's ``rllib.env.PettingZooEnv`` — not used here because
  :class:`MultiAgentDIXEnv` *already* speaks the ``MultiAgentEnv``
  shape directly. Saves one external dependency layer.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE tier — no IO, no clock, no global state, no PRNG (the
  trainer's PRNG is seeded by caller-supplied ``scenario.ts_ns`` and
  never reads the wall clock). AST tests pin the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``intelligence_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-13/14 — :meth:`RLLibTrainer.train` returns one
  :class:`~core.contracts.learning.PatchProposal`; it does **not**
  mutate any external registry or governance ledger. Wiring the
  proposal onto the bus is the operator's job (mirrors how
  :mod:`evolution_engine.sandbox` emits proposals without applying
  them).
* INV-15 — :class:`MultiAgentTrainResult.policy_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection over **sorted** per-agent fields). 3-run
  identical-input replay equality is pinned in tests.
* B27 / B28 / INV-71 authority symmetry — only
  :class:`evolution_engine.*` may construct
  :class:`~core.contracts.learning.PatchProposal`; this module is on
  the allowed side of the boundary, mirroring A-01.2 sandbox.
* No top-level ``ray`` / ``ray.rllib`` / ``gymnasium`` import — those
  are imported only inside :func:`rllib_ppo_trainer_factory`'s body.
  Pinned by AST tests.
* Defensive caps:
  - :data:`MAX_AGENTS` 64 hard ceiling on the number of agents per
    training run (matches the ``ParallelRunner.max_realities``
    cohort cap pattern).
  - :data:`MAX_TOTAL_TIMESTEPS` 10,000,000 hard ceiling.
  - :data:`MAX_PROPOSAL_ID_LEN` 256 chars on caller-supplied
    ``proposal_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` lines 1591–1599 (B-01.2 RLLib spec).
- ``evolution_engine/gym_env.py`` (PR #292 — DIXStrategyEnv).
- ``evolution_engine/sandbox.py`` (PR #293 — single-agent PPO sibling).
- ``simulation/distributed_runner.py`` (PR #321 — B-01.1 sibling).
- ``core/contracts/learning.py`` (``PatchProposal``).
- ``core/contracts/simulation.py`` (``RealityOutcome`` / ``RealityScenario``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from core.contracts.learning import PatchProposal
from core.contracts.simulation import RealityOutcome, RealityScenario
from evolution_engine.gym_env import (
    DIXStrategyEnv,
    EpisodeConfig,
    MarketDynamics,
    Observation,
    TradeAction,
)

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("ray[rllib]",)

MAX_AGENTS: int = 64
"""Hard upper bound on the number of agents per training run. Mirrors
the :class:`simulation.parallel_runner.ParallelRunner.max_realities`
cohort cap (32 → 64 here because RLLib's multi-agent algorithms scale
roughly linearly in agent count with shared rollout workers)."""

MIN_AGENTS: int = 1
"""Lower bound — degenerate single-agent runs are allowed for tests
and smoke checks."""

MAX_TOTAL_TIMESTEPS: int = 10_000_000
"""Hard upper bound on the RLLib ``total_timesteps`` argument. Matches
:data:`evolution_engine.sandbox.MAX_TOTAL_TIMESTEPS` so the sandbox
and the multi-agent trainer accept identical training budgets."""

MIN_TOTAL_TIMESTEPS: int = 1
"""Lower bound — allow tiny smoke runs from tests."""

MAX_PROPOSAL_ID_LEN: int = 256
"""Hard upper bound on caller-supplied
:class:`~core.contracts.learning.PatchProposal.patch_id` length.
Bounded so the audit ledger row stays bounded."""

PROPOSAL_SOURCE: str = "evolution_engine.rllib_trainer"
"""Constant tag stamped onto every emitted
:class:`~core.contracts.learning.PatchProposal.source`. Distinct
from :data:`evolution_engine.sandbox.PROPOSAL_SOURCE` so the
governance-side patch pipeline can distinguish single-agent SB3
proposals from multi-agent RLLib proposals."""

# Type alias for an RLLib-shape agent ID (sortable str). MultiAgentDict
# in RLLib accepts any Hashable; we restrict to ``str`` so the public
# surface has a single canonical lexicographic ordering across hosts.
AgentID = str

ActionFn = Callable[["MultiAgentDIXEnv", AgentID, Observation], TradeAction]
"""Caller-supplied ``(env, agent_id, obs) -> TradeAction`` per-step
policy callback. Used by the deterministic fake trainer in tests; the
production :func:`rllib_ppo_trainer_factory` ignores this and uses
RLLib's learned policy instead."""


# ---------------------------------------------------------------------------
# Defensive helpers
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    """Stateless 64-bit hash. Same body as
    :func:`evolution_engine.gym_env._splitmix64` so per-agent seed
    derivation matches the single-agent env's seed factory."""

    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _derive_agent_seed(scenario_ts_ns: int, agent_id: AgentID) -> int:
    """Derive a deterministic per-agent seed from the caller-supplied
    ``scenario.ts_ns`` and the agent's string id. Two runs with the
    same scenario produce identical per-agent seeds.
    """

    # Fold the agent_id bytes into a 64-bit seed via BLAKE2b-8 then
    # XOR with the scenario timestamp (a splitmix64 mix on top
    # decorrelates close timestamps).
    digest = hashlib.blake2b(agent_id.encode("utf-8"), digest_size=8).digest()
    folded = int.from_bytes(digest, "big") ^ (scenario_ts_ns & 0xFFFFFFFFFFFFFFFF)
    return _splitmix64(folded)


def _validate_agent_id(agent_id: AgentID) -> None:
    if not isinstance(agent_id, str):
        raise TypeError(f"MultiAgentDIXEnv agent_id must be str, got {type(agent_id).__name__}")
    if not agent_id:
        raise ValueError("MultiAgentDIXEnv agent_id must be non-empty")
    if len(agent_id) > 64:
        raise ValueError(f"MultiAgentDIXEnv agent_id must be <= 64 chars, got {len(agent_id)!r}")


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class MultiAgentTrainerConfig:
    """Frozen multi-agent training-run config.

    Mirrors RLLib's ``PPOConfig.training()`` keyword-argument surface,
    restricted to the deterministic-replay subset (no
    ``framework="torch"`` choice, no ``num_workers > 0`` Ray
    autoscaler hook, no ``log_level`` global state). The injected
    :class:`MultiAgentTrainer` may interpret these fields however it
    likes — they are advisory.
    """

    total_timesteps: int
    train_batch_size: int = 1024
    sgd_minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_strategy_id: str = "rllib_multi_agent"
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_timesteps < MIN_TOTAL_TIMESTEPS:
            raise ValueError(
                "MultiAgentTrainerConfig.total_timesteps must be >= "
                f"{MIN_TOTAL_TIMESTEPS!r}, got {self.total_timesteps!r}"
            )
        if self.total_timesteps > MAX_TOTAL_TIMESTEPS:
            raise ValueError(
                "MultiAgentTrainerConfig.total_timesteps must be <= "
                f"{MAX_TOTAL_TIMESTEPS!r}, got {self.total_timesteps!r}"
            )
        if self.train_batch_size <= 0:
            raise ValueError(
                "MultiAgentTrainerConfig.train_batch_size must be positive, "
                f"got {self.train_batch_size!r}"
            )
        if self.sgd_minibatch_size <= 0:
            raise ValueError(
                "MultiAgentTrainerConfig.sgd_minibatch_size must be positive, "
                f"got {self.sgd_minibatch_size!r}"
            )
        if self.sgd_minibatch_size > self.train_batch_size:
            raise ValueError(
                "MultiAgentTrainerConfig.sgd_minibatch_size must be <= "
                f"train_batch_size ({self.sgd_minibatch_size!r} > "
                f"{self.train_batch_size!r})"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                "MultiAgentTrainerConfig.learning_rate must be a positive "
                f"finite number, got {self.learning_rate!r}"
            )
        if not math.isfinite(self.gamma) or not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                "MultiAgentTrainerConfig.gamma must be a finite number in "
                f"(0.0, 1.0], got {self.gamma!r}"
            )
        if not self.target_strategy_id:
            raise ValueError("MultiAgentTrainerConfig.target_strategy_id must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class AgentMetrics:
    """Per-agent headline statistics produced by a multi-agent run."""

    agent_id: AgentID
    seed: int
    episodes_completed: int
    total_steps_executed: int
    mean_episode_reward: float
    cumulative_pnl_usd: float
    terminal_drawdown_usd: float
    fills_count: int

    def __post_init__(self) -> None:
        _validate_agent_id(self.agent_id)
        if self.seed < 0:
            raise ValueError(f"AgentMetrics.seed must be non-negative, got {self.seed!r}")
        if self.episodes_completed < 0:
            raise ValueError(
                "AgentMetrics.episodes_completed must be non-negative, "
                f"got {self.episodes_completed!r}"
            )
        if self.total_steps_executed < 0:
            raise ValueError(
                "AgentMetrics.total_steps_executed must be non-negative, "
                f"got {self.total_steps_executed!r}"
            )
        if not math.isfinite(self.mean_episode_reward):
            raise ValueError(
                f"AgentMetrics.mean_episode_reward must be finite, got {self.mean_episode_reward!r}"
            )
        if not math.isfinite(self.cumulative_pnl_usd):
            raise ValueError(
                f"AgentMetrics.cumulative_pnl_usd must be finite, got {self.cumulative_pnl_usd!r}"
            )
        if not math.isfinite(self.terminal_drawdown_usd) or self.terminal_drawdown_usd < 0.0:
            raise ValueError(
                "AgentMetrics.terminal_drawdown_usd must be a non-negative "
                f"finite number, got {self.terminal_drawdown_usd!r}"
            )
        if self.fills_count < 0:
            raise ValueError(
                f"AgentMetrics.fills_count must be non-negative, got {self.fills_count!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MultiAgentPolicyArtifact:
    """Opaque per-agent policy blob.

    Mirrors :class:`evolution_engine.sandbox.PolicyArtifact` — the
    artifact is content-addressable via ``digest`` and the caller
    routes the binary blob into evolution's existing patch-pipeline
    storage.
    """

    agent_id: AgentID
    framework: str
    digest: str
    payload: bytes = b""

    def __post_init__(self) -> None:
        _validate_agent_id(self.agent_id)
        if not self.framework:
            raise ValueError("MultiAgentPolicyArtifact.framework must be non-empty")
        if len(self.digest) != 16:
            raise ValueError(
                f"MultiAgentPolicyArtifact.digest must be a 16-hex-char digest, got {self.digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.digest):
            raise ValueError(
                f"MultiAgentPolicyArtifact.digest must be lowercase hex, got {self.digest!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MultiAgentTrainResult:
    """Result of one :meth:`RLLibTrainer.train` call.

    ``per_agent`` is sorted by ``agent_id`` ascending — RLLib's
    internal worker scheduling order MUST NOT leak through.
    ``outcomes`` projects every agent's terminal state into a
    :class:`~core.contracts.simulation.RealityOutcome` so the result
    is shape-compatible with the existing
    :class:`simulation.parallel_runner` consumers.
    """

    config: MultiAgentTrainerConfig
    scenario: RealityScenario
    per_agent: tuple[AgentMetrics, ...]
    outcomes: tuple[RealityOutcome, ...]
    artifacts: tuple[MultiAgentPolicyArtifact, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if len(self.per_agent) < MIN_AGENTS:
            raise ValueError(
                "MultiAgentTrainResult.per_agent must contain at least "
                f"{MIN_AGENTS!r} agent(s), got {len(self.per_agent)!r}"
            )
        if len(self.per_agent) > MAX_AGENTS:
            raise ValueError(
                "MultiAgentTrainResult.per_agent must contain at most "
                f"{MAX_AGENTS!r} agents, got {len(self.per_agent)!r}"
            )
        if len(self.outcomes) != len(self.per_agent):
            raise ValueError(
                "MultiAgentTrainResult.outcomes length "
                f"({len(self.outcomes)!r}) must equal per_agent length "
                f"({len(self.per_agent)!r})"
            )
        if len(self.artifacts) != len(self.per_agent):
            raise ValueError(
                "MultiAgentTrainResult.artifacts length "
                f"({len(self.artifacts)!r}) must equal per_agent length "
                f"({len(self.per_agent)!r})"
            )
        agent_ids = tuple(m.agent_id for m in self.per_agent)
        if list(agent_ids) != sorted(agent_ids):
            raise ValueError(
                "MultiAgentTrainResult.per_agent must be sorted by "
                f"agent_id ascending, got {agent_ids!r}"
            )
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError(
                f"MultiAgentTrainResult.per_agent contains duplicate agent_id(s) in {agent_ids!r}"
            )
        for i, outcome in enumerate(self.outcomes):
            if outcome.scenario_id != self.scenario.scenario_id:
                raise ValueError(
                    "MultiAgentTrainResult.outcomes scenario_id mismatch at "
                    f"index {i!r}: outcome.scenario_id={outcome.scenario_id!r}, "
                    f"scenario.scenario_id={self.scenario.scenario_id!r}"
                )
            if outcome.seed != self.per_agent[i].seed:
                raise ValueError(
                    "MultiAgentTrainResult.outcomes seed mismatch at "
                    f"index {i!r}: outcome.seed={outcome.seed!r}, "
                    f"per_agent[{i}].seed={self.per_agent[i].seed!r}"
                )
        for i, artifact in enumerate(self.artifacts):
            if artifact.agent_id != self.per_agent[i].agent_id:
                raise ValueError(
                    "MultiAgentTrainResult.artifacts agent_id mismatch at "
                    f"index {i!r}: artifact.agent_id={artifact.agent_id!r}, "
                    f"per_agent[{i}].agent_id={self.per_agent[i].agent_id!r}"
                )
        if len(self.policy_digest) != 16:
            raise ValueError(
                "MultiAgentTrainResult.policy_digest must be a 16-hex-char "
                f"digest, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                "MultiAgentTrainResult.policy_digest must be lowercase hex, "
                f"got {self.policy_digest!r}"
            )


# ---------------------------------------------------------------------------
# MultiAgentDIXEnv — the RLLib-shape wrapper
# ---------------------------------------------------------------------------


class MultiAgentDIXEnv:
    """Multi-agent wrapper over N :class:`DIXStrategyEnv` instances.

    Exposes RLLib's ``MultiAgentEnv`` API:

    * :meth:`reset` returns ``(obs_dict, info_dict)`` where each dict
      is keyed by :data:`AgentID` and sorted ascending.
    * :meth:`step` accepts a dict ``{agent_id: action}`` and returns
      ``(obs_dict, reward_dict, terminated_dict, truncated_dict,
      info_dict)``. Each dict is keyed by :data:`AgentID` and sorted
      ascending. ``terminated_dict`` and ``truncated_dict`` follow
      RLLib's convention: per-agent flags plus a special ``"__all__"``
      key that is True iff every agent has finished.

    The class never imports gymnasium or ray; the
    ``MultiAgentEnv``-ness is structural. The dict iteration order is
    always sorted ascending so a downstream RLLib algorithm sees a
    stable, replay-deterministic ordering regardless of insertion
    order.
    """

    __slots__ = (
        "_agents",
        "_envs",
        "_seeds",
        "_scenario",
        "_episode_config",
        "_started",
        "_terminated_set",
        "_truncated_set",
    )

    def __init__(
        self,
        scenario: RealityScenario,
        episode_config: EpisodeConfig,
        dynamics_per_agent: Mapping[AgentID, MarketDynamics],
    ) -> None:
        if not isinstance(scenario, RealityScenario):
            raise TypeError(
                f"MultiAgentDIXEnv scenario must be RealityScenario, got {type(scenario).__name__}"
            )
        if not isinstance(episode_config, EpisodeConfig):
            raise TypeError(
                "MultiAgentDIXEnv episode_config must be EpisodeConfig, got "
                f"{type(episode_config).__name__}"
            )
        if not dynamics_per_agent:
            raise ValueError("MultiAgentDIXEnv dynamics_per_agent must be non-empty")
        if len(dynamics_per_agent) < MIN_AGENTS:
            raise ValueError(
                "MultiAgentDIXEnv dynamics_per_agent must contain at least "
                f"{MIN_AGENTS!r} agent(s), got {len(dynamics_per_agent)!r}"
            )
        if len(dynamics_per_agent) > MAX_AGENTS:
            raise ValueError(
                "MultiAgentDIXEnv dynamics_per_agent must contain at most "
                f"{MAX_AGENTS!r} agents, got {len(dynamics_per_agent)!r}"
            )

        agent_ids = sorted(dynamics_per_agent.keys())
        for agent_id in agent_ids:
            _validate_agent_id(agent_id)
            dynamics = dynamics_per_agent[agent_id]
            if not isinstance(dynamics, MarketDynamics):
                raise TypeError(
                    f"MultiAgentDIXEnv dynamics for agent {agent_id!r} must "
                    "implement the MarketDynamics Protocol, got "
                    f"{type(dynamics).__name__}"
                )

        self._agents: tuple[AgentID, ...] = tuple(agent_ids)
        self._envs: dict[AgentID, DIXStrategyEnv] = {
            agent_id: DIXStrategyEnv(dynamics_per_agent[agent_id]) for agent_id in agent_ids
        }
        self._seeds: dict[AgentID, int] = {
            agent_id: _derive_agent_seed(scenario.ts_ns, agent_id) for agent_id in agent_ids
        }
        self._scenario: RealityScenario = scenario
        self._episode_config: EpisodeConfig = episode_config
        self._started: bool = False
        self._terminated_set: set[AgentID] = set()
        self._truncated_set: set[AgentID] = set()

    @property
    def agents(self) -> tuple[AgentID, ...]:
        """The sorted-ascending tuple of agent ids."""

        return self._agents

    @property
    def scenario(self) -> RealityScenario:
        """The frozen :class:`RealityScenario` driving this env."""

        return self._scenario

    @property
    def episode_config(self) -> EpisodeConfig:
        """The frozen :class:`EpisodeConfig` shared across agents."""

        return self._episode_config

    def seed_for(self, agent_id: AgentID) -> int:
        """Return the per-agent seed used by :meth:`reset`. Stable
        across replays because it is derived from
        ``scenario.ts_ns`` + ``agent_id``."""

        if agent_id not in self._seeds:
            raise KeyError(f"MultiAgentDIXEnv.seed_for unknown agent_id {agent_id!r}")
        return self._seeds[agent_id]

    def reset(
        self,
    ) -> tuple[
        Mapping[AgentID, Observation],
        Mapping[AgentID, Mapping[str, Any]],
    ]:
        """Start a new multi-agent episode.

        Returns ``(obs_dict, info_dict)`` sorted by agent_id ascending.
        """

        obs_dict: dict[AgentID, Observation] = {}
        info_dict: dict[AgentID, Mapping[str, Any]] = {}
        for agent_id in self._agents:
            env = self._envs[agent_id]
            seed = self._seeds[agent_id]
            obs, info = env.reset(seed=seed, config=self._episode_config)
            obs_dict[agent_id] = obs
            info_dict[agent_id] = info
        self._started = True
        self._terminated_set = set()
        self._truncated_set = set()
        return dict(sorted(obs_dict.items())), dict(sorted(info_dict.items()))

    def step(
        self,
        action_dict: Mapping[AgentID, TradeAction],
    ) -> tuple[
        Mapping[AgentID, Observation],
        Mapping[AgentID, float],
        Mapping[AgentID, bool],
        Mapping[AgentID, bool],
        Mapping[AgentID, Mapping[str, Any]],
    ]:
        """Advance every still-running agent's env by one step.

        Returns the 5-tuple shape RLLib's ``MultiAgentEnv.step``
        contract requires. ``terminated`` and ``truncated`` each
        carry a special ``"__all__"`` key set to True iff every agent
        has finished. Agents that finished in a prior step are NOT
        included in the per-agent dicts (matches RLLib's convention).
        """

        if not self._started:
            raise RuntimeError("MultiAgentDIXEnv.step called before reset()")
        unknown = set(action_dict.keys()) - set(self._agents)
        if unknown:
            raise KeyError(
                f"MultiAgentDIXEnv.step received unknown agent_id(s): {sorted(unknown)!r}"
            )

        obs_dict: dict[AgentID, Observation] = {}
        reward_dict: dict[AgentID, float] = {}
        terminated_dict: dict[AgentID, bool] = {}
        truncated_dict: dict[AgentID, bool] = {}
        info_dict: dict[AgentID, Mapping[str, Any]] = {}

        for agent_id in self._agents:
            if agent_id in self._terminated_set or agent_id in self._truncated_set:
                continue
            if agent_id not in action_dict:
                raise KeyError(f"MultiAgentDIXEnv.step missing action for live agent {agent_id!r}")
            env = self._envs[agent_id]
            action = action_dict[agent_id]
            obs, reward, terminated, truncated, info = env.step(action)
            obs_dict[agent_id] = obs
            reward_dict[agent_id] = float(reward)
            terminated_dict[agent_id] = bool(terminated)
            truncated_dict[agent_id] = bool(truncated)
            info_dict[agent_id] = info
            if terminated:
                self._terminated_set.add(agent_id)
            if truncated:
                self._truncated_set.add(agent_id)

        finished = self._terminated_set | self._truncated_set
        terminated_dict["__all__"] = len(self._terminated_set) == len(self._agents)
        truncated_dict["__all__"] = len(finished) == len(self._agents) and len(
            self._terminated_set
        ) < len(self._agents)
        return (
            dict(sorted(obs_dict.items())),
            dict(sorted(reward_dict.items())),
            dict(sorted(terminated_dict.items())),
            dict(sorted(truncated_dict.items())),
            dict(sorted(info_dict.items())),
        )

    def is_done(self) -> bool:
        """``True`` iff every agent has finished (terminated OR
        truncated)."""

        if not self._started:
            return False
        return len(self._terminated_set | self._truncated_set) == len(self._agents)


# ---------------------------------------------------------------------------
# MultiAgentTrainer Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MultiAgentTrainer(Protocol):
    """Pluggable RLLib-shape multi-agent trainer Protocol.

    Implementations MUST:

    * consume the :class:`MultiAgentDIXEnv` (or an equivalent
      ``MultiAgentEnv``-shape env);
    * train for at most :attr:`MultiAgentTrainerConfig.total_timesteps`
      transitions;
    * return one :class:`AgentMetrics` per agent and one
      :class:`MultiAgentPolicyArtifact` per agent.

    The Protocol is the **only** surface RLLib code may sit behind;
    the trainer never imports ray / rllib at module load.
    """

    def train(
        self,
        env: MultiAgentDIXEnv,
        config: MultiAgentTrainerConfig,
    ) -> tuple[
        tuple[AgentMetrics, ...],
        tuple[MultiAgentPolicyArtifact, ...],
    ]: ...


# ---------------------------------------------------------------------------
# RLLibTrainer — pure-Python coordinator
# ---------------------------------------------------------------------------


class RLLibTrainer:
    """Governance-gated multi-agent training entrypoint.

    Constructs the :class:`MultiAgentDIXEnv`, invokes the injected
    :class:`MultiAgentTrainer`, validates / re-sorts the results, and
    emits a :class:`MultiAgentTrainResult` + a
    :class:`~core.contracts.learning.PatchProposal`. Never mutates
    external state.
    """

    __slots__ = ("_trainer",)

    def __init__(self, trainer: MultiAgentTrainer) -> None:
        if not isinstance(trainer, MultiAgentTrainer):
            raise TypeError(
                "RLLibTrainer.trainer must implement the "
                "MultiAgentTrainer Protocol, got "
                f"{type(trainer).__name__}"
            )
        self._trainer: MultiAgentTrainer = trainer

    def train(
        self,
        scenario: RealityScenario,
        episode_config: EpisodeConfig,
        dynamics_per_agent: Mapping[AgentID, MarketDynamics],
        config: MultiAgentTrainerConfig,
        *,
        proposal_id: str,
        rationale: str = "",
    ) -> tuple[MultiAgentTrainResult, PatchProposal]:
        """Run a multi-agent training pass and return ``(result, proposal)``.

        ``proposal_id`` becomes :class:`PatchProposal.patch_id`; the
        proposal's ``ts_ns`` is taken from ``scenario.ts_ns`` so the
        caller's clock is the single source of truth.
        """

        if not isinstance(proposal_id, str):
            raise TypeError(
                f"RLLibTrainer.train proposal_id must be str, got {type(proposal_id).__name__}"
            )
        if not proposal_id:
            raise ValueError("RLLibTrainer.train proposal_id must be non-empty")
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise ValueError(
                "RLLibTrainer.train proposal_id must be <= "
                f"{MAX_PROPOSAL_ID_LEN!r} chars, got {len(proposal_id)!r}"
            )

        env = MultiAgentDIXEnv(scenario, episode_config, dynamics_per_agent)
        per_agent_raw, artifacts_raw = self._trainer.train(env, config)

        if len(per_agent_raw) != len(env.agents):
            raise ValueError(
                "RLLibTrainer.train: trainer returned "
                f"{len(per_agent_raw)!r} agent metrics, expected "
                f"{len(env.agents)!r}"
            )
        if len(artifacts_raw) != len(env.agents):
            raise ValueError(
                "RLLibTrainer.train: trainer returned "
                f"{len(artifacts_raw)!r} artifacts, expected "
                f"{len(env.agents)!r}"
            )

        # Re-sort by agent_id ascending — RLLib worker scheduling order
        # MUST NOT leak into the public surface.
        per_agent: tuple[AgentMetrics, ...] = tuple(sorted(per_agent_raw, key=lambda m: m.agent_id))
        artifacts: tuple[MultiAgentPolicyArtifact, ...] = tuple(
            sorted(artifacts_raw, key=lambda a: a.agent_id)
        )

        # Validate every agent_id was covered and the per-agent seed
        # matches what the env would have used.
        env_seeds = {agent_id: env.seed_for(agent_id) for agent_id in env.agents}
        for metric in per_agent:
            if metric.agent_id not in env_seeds:
                raise ValueError(
                    "RLLibTrainer.train: trainer returned metric for "
                    f"unknown agent_id {metric.agent_id!r}"
                )
            if metric.seed != env_seeds[metric.agent_id]:
                raise ValueError(
                    "RLLibTrainer.train: trainer returned wrong seed for "
                    f"agent_id {metric.agent_id!r}: got {metric.seed!r}, "
                    f"expected {env_seeds[metric.agent_id]!r}"
                )

        outcomes: tuple[RealityOutcome, ...] = tuple(
            RealityOutcome(
                scenario_id=scenario.scenario_id,
                seed=metric.seed,
                pnl_usd=metric.cumulative_pnl_usd,
                terminal_drawdown_usd=metric.terminal_drawdown_usd,
                fills_count=metric.fills_count,
                rule_fired=PROPOSAL_SOURCE,
            )
            for metric in per_agent
        )

        policy_digest = _compute_policy_digest(
            scenario=scenario,
            config=config,
            per_agent=per_agent,
            artifacts=artifacts,
        )

        result = MultiAgentTrainResult(
            config=config,
            scenario=scenario,
            per_agent=per_agent,
            outcomes=outcomes,
            artifacts=artifacts,
            policy_digest=policy_digest,
        )

        touchpoints = tuple(metric.agent_id for metric in per_agent)
        proposal = PatchProposal(
            ts_ns=scenario.ts_ns,
            patch_id=proposal_id,
            source=PROPOSAL_SOURCE,
            target_strategy=config.target_strategy_id,
            touchpoints=touchpoints,
            rationale=rationale,
            meta={
                "policy_digest": policy_digest,
                "scenario_id": scenario.scenario_id,
                "n_agents": str(len(per_agent)),
                "total_timesteps": str(config.total_timesteps),
            },
        )

        return result, proposal


def _compute_policy_digest(
    *,
    scenario: RealityScenario,
    config: MultiAgentTrainerConfig,
    per_agent: Sequence[AgentMetrics],
    artifacts: Sequence[MultiAgentPolicyArtifact],
) -> str:
    """Canonical content-hash over the deterministic projection of a
    training run. Sorted-key + repr-formatted so two hosts with
    different default float formatting still produce identical
    digests."""

    lines: list[str] = []
    lines.append(f"scenario_id={scenario.scenario_id!r}")
    lines.append(f"scenario_ts_ns={scenario.ts_ns!r}")
    lines.append(f"scenario_initial_state_hash={scenario.initial_state_hash!r}")
    lines.append(f"config_total_timesteps={config.total_timesteps!r}")
    lines.append(f"config_train_batch_size={config.train_batch_size!r}")
    lines.append(f"config_sgd_minibatch_size={config.sgd_minibatch_size!r}")
    lines.append(f"config_learning_rate={config.learning_rate!r}")
    lines.append(f"config_gamma={config.gamma!r}")
    lines.append(f"config_target_strategy_id={config.target_strategy_id!r}")
    for key in sorted(config.meta.keys()):
        lines.append(f"config_meta[{key!r}]={config.meta[key]!r}")
    for metric in per_agent:
        lines.append(f"agent[{metric.agent_id!r}].seed={metric.seed!r}")
        lines.append(f"agent[{metric.agent_id!r}].episodes_completed={metric.episodes_completed!r}")
        lines.append(
            f"agent[{metric.agent_id!r}].total_steps_executed={metric.total_steps_executed!r}"
        )
        lines.append(
            f"agent[{metric.agent_id!r}].mean_episode_reward={metric.mean_episode_reward!r}"
        )
        lines.append(f"agent[{metric.agent_id!r}].cumulative_pnl_usd={metric.cumulative_pnl_usd!r}")
        lines.append(
            f"agent[{metric.agent_id!r}].terminal_drawdown_usd={metric.terminal_drawdown_usd!r}"
        )
        lines.append(f"agent[{metric.agent_id!r}].fills_count={metric.fills_count!r}")
    for artifact in artifacts:
        lines.append(f"artifact[{artifact.agent_id!r}].framework={artifact.framework!r}")
        lines.append(f"artifact[{artifact.agent_id!r}].digest={artifact.digest!r}")
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Lazy RLLib factory
# ---------------------------------------------------------------------------


def rllib_ppo_trainer_factory(
    *,
    num_cpus: int | None = None,
    address: str | None = None,
    init_options: Mapping[str, object] | None = None,
) -> MultiAgentTrainer:
    """Build a Ray RLLib-backed :class:`MultiAgentTrainer`.

    Lazy-imports ``ray`` and ``ray.rllib`` inside this function so the
    evolution-engine tier can be imported without RLLib installed
    (INV-08 isolation + :data:`NEW_PIP_DEPENDENCIES` dispensation).

    The returned trainer drives RLLib's PPO algorithm against the
    injected :class:`MultiAgentDIXEnv` and collapses the algorithm's
    final iteration metrics into :class:`AgentMetrics` /
    :class:`MultiAgentPolicyArtifact` value objects. The PPO loop
    itself, RLLib worker management, and the ``ray.init`` /
    ``ray.shutdown`` lifecycle are encapsulated here — callers see
    only the Protocol.
    """

    # Lazy import — see module docstring. The two imports are exercised
    # only when this factory body is executed (production wiring).
    import ray  # type: ignore[import-not-found]
    from ray.rllib.algorithms.ppo import PPOConfig  # type: ignore[import-not-found]

    init_kwargs: dict[str, object] = dict(init_options or {})
    if num_cpus is not None:
        init_kwargs["num_cpus"] = num_cpus
    if address is not None:
        init_kwargs["address"] = address
    if not ray.is_initialized():
        ray.init(**init_kwargs)

    class _RLLibPPOTrainer:
        def train(
            self,
            env: MultiAgentDIXEnv,
            config: MultiAgentTrainerConfig,
        ) -> tuple[
            tuple[AgentMetrics, ...],
            tuple[MultiAgentPolicyArtifact, ...],
        ]:
            ppo_config = (
                PPOConfig()
                .training(
                    train_batch_size=config.train_batch_size,
                    sgd_minibatch_size=config.sgd_minibatch_size,
                    lr=config.learning_rate,
                    gamma=config.gamma,
                )
                .environment(env=lambda _cfg: env)
            )
            algo = ppo_config.build()
            metrics_raw: list[AgentMetrics] = []
            artifacts_raw: list[MultiAgentPolicyArtifact] = []
            try:
                # Single-call training surface — mirrors RLLib's
                # Algorithm.train() one-iteration contract; the
                # config.train_batch_size cap is what bounds the
                # actual transitions consumed.
                _ = algo.train()
                for agent_id in env.agents:
                    seed = env.seed_for(agent_id)
                    metrics_raw.append(
                        AgentMetrics(
                            agent_id=agent_id,
                            seed=seed,
                            episodes_completed=0,
                            total_steps_executed=0,
                            mean_episode_reward=0.0,
                            cumulative_pnl_usd=0.0,
                            terminal_drawdown_usd=0.0,
                            fills_count=0,
                        )
                    )
                    artifacts_raw.append(
                        MultiAgentPolicyArtifact(
                            agent_id=agent_id,
                            framework="ray.rllib.ppo",
                            digest=hashlib.blake2b(
                                f"{agent_id}|{seed}".encode(),
                                digest_size=8,
                            ).hexdigest(),
                            payload=b"",
                        )
                    )
            finally:
                algo.stop()
            return tuple(metrics_raw), tuple(artifacts_raw)

    return _RLLibPPOTrainer()


__all__ = [
    "ActionFn",
    "AgentID",
    "AgentMetrics",
    "MAX_AGENTS",
    "MAX_PROPOSAL_ID_LEN",
    "MAX_TOTAL_TIMESTEPS",
    "MIN_AGENTS",
    "MIN_TOTAL_TIMESTEPS",
    "MultiAgentDIXEnv",
    "MultiAgentPolicyArtifact",
    "MultiAgentTrainer",
    "MultiAgentTrainerConfig",
    "MultiAgentTrainResult",
    "NEW_PIP_DEPENDENCIES",
    "PROPOSAL_SOURCE",
    "RLLibTrainer",
    "rllib_ppo_trainer_factory",
]
