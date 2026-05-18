# ADAPTED FROM: PettingZoo-Team/PettingZoo
# (pettingzoo/utils/env.py — AECEnv turn-based shape;
#  pettingzoo/utils/parallel_env.py — ParallelEnv simultaneous-step shape;
#  pettingzoo/utils/agent_selector.py — round-robin turn selector;
#  Farama-Foundation/Gymnasium gymnasium/core.py — Env reset/step API shape.)
"""C-30 — DIXMultiAgentEnv: multi-strategy tournament environment.

`PettingZoo` is the canonical multi-agent companion to Gymnasium.
Two complementary APIs:

* ``AECEnv`` (Agent Environment Cycle) — strictly turn-based,
  one ``agent_selection`` per ``step`` call.
* ``ParallelEnv`` — all agents step simultaneously per tick,
  ``step(actions_dict) -> (observations, rewards, terminations,
  truncations, infos)``.

DIX needs **both shapes** because the tournament evolves through
different phases: scouting rounds are turn-based (so the audit
ledger has a deterministic order of agent actions) but the
production match step is simultaneous (so a tick represents a
single shared market snapshot and every agent reacts to the
same observation).

What this module is
-------------------

* Pure-Python value objects + a stateful env class. No
  ``pettingzoo`` / ``gymnasium`` / ``numpy`` import at module
  top-level. The optional :func:`pettingzoo_multiagent_env_factory`
  lazy-imports ``pettingzoo`` and returns a ``ParallelEnv``-shape
  wrapper. The core :class:`DIXMultiAgentEnv` class is pure stdlib.
* OFFLINE_ONLY tier: `evolution_engine` is an OFFLINE engine. The
  env is wall-clock-free, reads no environment variables, performs
  no IO — caller supplies a frozen multi-agent scenario.
* INV-15 byte-identical replays. ``DIXMultiAgentEnv.reset(seed=K)``
  followed by N calls to ``DIXMultiAgentEnv.step(actions)`` always
  returns identical per-agent observations, rewards, and
  terminations on any host.

What survives from upstream PettingZoo
--------------------------------------

* The ``ParallelEnv`` step return shape:
  ``(observations, rewards, terminations, truncations, infos)`` —
  per-agent dicts keyed by ``agent_id``.
* The ``AECEnv`` cyclic shape: ``agent_selection`` cycles deterministically
  through ``agents`` via :class:`AgentSelector`, mirroring upstream's
  ``agent_selector.AgentSelector``.
* The ``possible_agents`` / ``agents`` split — upstream defines
  ``possible_agents`` as the full roster and ``agents`` as the
  alive subset that filters as agents terminate.
* The ``last() -> (observation, reward, terminated, truncated, info)``
  introspection method for the agent currently in turn (AEC shape).

What we replaced
----------------

* ``numpy`` arrays for observations → ``tuple[float, ...]``. The
  DIX RL stack accepts tuples and we avoid the numpy top-level
  import so the module loads without it.
* Upstream's ``np.random.RandomState`` PRNG plumbing → an explicit
  caller-supplied seed folded into stdlib :func:`hashlib`. INV-15
  forbids any hidden PRNG state.
* Upstream's optional ``rendering`` modes → render is a strict
  no-op. Multi-agent tournaments are audited via the ledger, not
  by visual rendering.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE_ONLY tier. No IO, no clock, no PRNG, no global state.
  The env class is the only stateful object and resets cleanly on
  every :meth:`DIXMultiAgentEnv.reset` call.
* No engine cross-imports. The env never reads ``registry/`` /
  ``system_engine`` / ``governance_engine`` / ``execution_engine``.
  Agent identities are caller-supplied opaque strings.
* INV-15 byte-identical replay. Caller supplies the seed and
  scenario; the seed is folded into a deterministic per-agent
  :class:`MultiAgentObservation` hash.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-30 (pettingzoo multi-agent env).
- ``evolution_engine/environments/anytrading_env.py`` (C-29 — the
  single-agent companion benchmark env).
- ``evolution_engine/gym_env.py`` (A-01.1 — single-agent DIX env).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from typing import Any, Final

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("pettingzoo", "gymnasium")
"""``pettingzoo`` is needed only by the optional
:func:`pettingzoo_multiagent_env_factory`; ``gymnasium`` is needed
only by the lazy seam. The core :class:`DIXMultiAgentEnv` class is
pure stdlib."""

MULTIAGENT_ENV_VERSION: Final[str] = "c-30-pettingzoo-1"
"""Version tag woven into the canonical content-hash of every
observation so a replay run can verify the env-shape is unchanged."""

MIN_AGENTS: Final[int] = 2
"""Minimum number of agents in a tournament. Single-agent envs use
:class:`evolution_engine.gym_env.DIXStrategyEnv` (A-01.1) instead."""

MAX_AGENTS: Final[int] = 64
"""Hard upper bound on roster size. Prevents accidental O(N²)
matchups in tournament scheduling."""

MAX_EPISODE_STEPS: Final[int] = 1_000_000
"""Hard upper bound on episode length. Defensive cap; real
tournament runs are bounded by ``scenario.max_steps``."""


class MultiAgentMode(enum.IntEnum):
    """Step-shape selector — mirrors PettingZoo's two API surfaces.

    * ``PARALLEL`` — all agents step simultaneously per tick
      (upstream ``ParallelEnv``).
    * ``AEC`` — exactly one agent steps per ``step`` call, in a
      deterministic cycle (upstream ``AECEnv``).
    """

    PARALLEL = 0
    AEC = 1


class MultiAgentAction(enum.IntEnum):
    """Discrete action set for the tournament.

    Mirrors C-29 :class:`AnytradingAction` for benchmark parity.
    """

    HOLD = 0
    BUY = 1
    SELL = 2


class MultiAgentEpisodeBudgetExceededError(RuntimeError):
    """Raised when a caller pumps more than :data:`MAX_EPISODE_STEPS`
    ``step`` calls without resetting the env."""


class UnknownAgentError(KeyError):
    """Raised when a caller provides an action / queries an agent
    that is not in ``possible_agents``."""


class WrongStepShapeError(RuntimeError):
    """Raised when caller mixes PARALLEL and AEC step shapes."""


@dataclasses.dataclass(frozen=True, slots=True)
class MultiAgentScenario:
    """Frozen multi-agent tournament scenario.

    Mirrors C-29 :class:`AnytradingConfig` shape — one frozen
    scenario drives the whole episode. Agent IDs are caller-supplied
    opaque strings; the env never interprets them.
    """

    agent_ids: tuple[str, ...]
    prices: tuple[float, ...]
    max_steps: int
    mode: MultiAgentMode = MultiAgentMode.PARALLEL

    def __post_init__(self) -> None:
        if len(self.agent_ids) < MIN_AGENTS:
            raise ValueError(
                "MultiAgentScenario.agent_ids must have >= "
                f"{MIN_AGENTS!r} elements, got {len(self.agent_ids)!r}"
            )
        if len(self.agent_ids) > MAX_AGENTS:
            raise ValueError(
                "MultiAgentScenario.agent_ids must have <= "
                f"{MAX_AGENTS!r} elements, got {len(self.agent_ids)!r}"
            )
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError(f"MultiAgentScenario.agent_ids must be unique, got {self.agent_ids!r}")
        for idx, agent_id in enumerate(self.agent_ids):
            if not isinstance(agent_id, str):
                raise TypeError(
                    f"MultiAgentScenario.agent_ids[{idx}] must be str, got "
                    f"{type(agent_id).__name__}"
                )
            if not agent_id:
                raise ValueError(f"MultiAgentScenario.agent_ids[{idx}] must be non-empty")
        if not self.prices:
            raise ValueError("MultiAgentScenario.prices must be non-empty")
        for idx, price in enumerate(self.prices):
            if not isinstance(price, (int, float)):
                raise TypeError(
                    f"MultiAgentScenario.prices[{idx}] must be float, got {type(price).__name__}"
                )
            if not math.isfinite(float(price)):
                raise ValueError(f"MultiAgentScenario.prices[{idx}] must be finite, got {price!r}")
            if float(price) <= 0.0:
                raise ValueError(
                    f"MultiAgentScenario.prices[{idx}] must be positive, got {price!r}"
                )
        if self.max_steps <= 0:
            raise ValueError(
                f"MultiAgentScenario.max_steps must be positive, got {self.max_steps!r}"
            )
        if self.max_steps > MAX_EPISODE_STEPS:
            raise ValueError(
                "MultiAgentScenario.max_steps must be <= "
                f"{MAX_EPISODE_STEPS!r}, got {self.max_steps!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MultiAgentObservation:
    """Frozen per-agent observation.

    Mirrors C-29 :class:`AnytradingObservation` shape:

    * ``agent_id`` — opaque caller-supplied string identifying the
      agent this observation belongs to.
    * ``step_idx`` — monotonically increasing within an episode,
      starts at 0 on reset.
    * ``mid_price`` — current shared market mid (all agents see the
      same price; tournament reads one market).
    * ``inventory_signed`` — sign of the held position (-1 / 0 / +1).
    * ``cumulative_pnl_usd`` — running pnl of this agent.
    * ``state_hash`` — 16-hex-char BLAKE2b-16 of
      ``(seed, agent_id, step_idx, mid_price, inventory_signed,
      cumulative_pnl_usd)``.
    """

    agent_id: str
    step_idx: int
    mid_price: float
    inventory_signed: int
    cumulative_pnl_usd: float
    state_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise ValueError("MultiAgentObservation.agent_id must be non-empty str")
        if self.step_idx < 0:
            raise ValueError(f"MultiAgentObservation.step_idx must be >= 0, got {self.step_idx!r}")
        if not math.isfinite(self.mid_price):
            raise ValueError(
                f"MultiAgentObservation.mid_price must be finite, got {self.mid_price!r}"
            )
        if self.mid_price <= 0.0:
            raise ValueError(
                f"MultiAgentObservation.mid_price must be positive, got {self.mid_price!r}"
            )
        if self.inventory_signed not in (-1, 0, 1):
            raise ValueError(
                "MultiAgentObservation.inventory_signed must be -1/0/+1, got "
                f"{self.inventory_signed!r}"
            )
        if not math.isfinite(self.cumulative_pnl_usd):
            raise ValueError(
                "MultiAgentObservation.cumulative_pnl_usd must be finite, got "
                f"{self.cumulative_pnl_usd!r}"
            )
        if len(self.state_hash) != 16:
            raise ValueError(
                f"MultiAgentObservation.state_hash must be 16 hex chars, got {self.state_hash!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MultiAgentStepResult:
    """Frozen result of one :meth:`DIXMultiAgentEnv.step` call.

    Mirrors PettingZoo's ``ParallelEnv.step`` return 5-tuple but as
    a structured record so the dict layout is pinned. For AEC mode
    the dicts contain exactly one entry (the agent currently in turn).
    """

    observations: tuple[MultiAgentObservation, ...]
    rewards: tuple[tuple[str, float], ...]
    terminations: tuple[tuple[str, bool], ...]
    truncations: tuple[tuple[str, bool], ...]
    infos: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]

    def __post_init__(self) -> None:
        agent_ids = tuple(o.agent_id for o in self.observations)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("MultiAgentStepResult.observations must have unique agent_ids")
        reward_agents = tuple(a for a, _ in self.rewards)
        if set(reward_agents) != set(agent_ids):
            raise ValueError(
                "MultiAgentStepResult.rewards must cover exactly the same agents as observations"
            )

    def rewards_dict(self) -> dict[str, float]:
        return dict(self.rewards)

    def terminations_dict(self) -> dict[str, bool]:
        return dict(self.terminations)

    def truncations_dict(self) -> dict[str, bool]:
        return dict(self.truncations)


class AgentSelector:
    """Deterministic round-robin agent selector.

    Mirrors upstream ``pettingzoo.utils.agent_selector.AgentSelector``
    exactly — selects the next agent in cyclic order. No state hash
    is needed because the order is deterministic per scenario.
    """

    __slots__ = ("_agents", "_idx")

    def __init__(self, agents: tuple[str, ...]) -> None:
        if len(agents) < MIN_AGENTS:
            raise ValueError(
                f"AgentSelector requires >= {MIN_AGENTS!r} agents, got {len(agents)!r}"
            )
        self._agents: tuple[str, ...] = agents
        self._idx: int = 0

    @property
    def current(self) -> str:
        return self._agents[self._idx]

    def next(self) -> str:
        self._idx = (self._idx + 1) % len(self._agents)
        return self._agents[self._idx]

    def reset(self) -> None:
        self._idx = 0

    def is_last(self) -> bool:
        return self._idx == len(self._agents) - 1


def _agent_obs_hash(
    *,
    seed: int,
    agent_id: str,
    step_idx: int,
    mid_price: float,
    inventory_signed: int,
    cumulative_pnl_usd: float,
) -> str:
    """Return a 16-hex-char BLAKE2b-16 digest of a per-agent
    observation — deterministic across hosts."""

    payload = "|".join(
        (
            f"v={MULTIAGENT_ENV_VERSION}",
            f"seed={seed}",
            f"agent={agent_id}",
            f"step={step_idx}",
            f"mid={mid_price:.17g}",
            f"inv={inventory_signed}",
            f"pnl={cumulative_pnl_usd:.17g}",
        )
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


class DIXMultiAgentEnv:
    """Multi-strategy tournament environment.

    Holds a shared market mid-price series and per-agent inventory /
    pnl state. Supports both PARALLEL (simultaneous step) and AEC
    (turn-based step) modes via :class:`MultiAgentMode`.
    """

    __slots__ = (
        "_scenario",
        "_seed",
        "_step_idx",
        "_inventory",
        "_pnl",
        "_step_call_count",
        "_terminated",
        "_selector",
    )

    def __init__(self, *, scenario: MultiAgentScenario) -> None:
        self._scenario: MultiAgentScenario = scenario
        self._seed: int = 0
        self._step_idx: int = 0
        self._inventory: dict[str, int] = {agent_id: 0 for agent_id in scenario.agent_ids}
        self._pnl: dict[str, float] = {agent_id: 0.0 for agent_id in scenario.agent_ids}
        self._step_call_count: int = 0
        self._terminated: bool = False
        self._selector: AgentSelector = AgentSelector(scenario.agent_ids)

    @property
    def scenario(self) -> MultiAgentScenario:
        return self._scenario

    @property
    def possible_agents(self) -> tuple[str, ...]:
        return self._scenario.agent_ids

    @property
    def agents(self) -> tuple[str, ...]:
        if self._terminated:
            return ()
        return self._scenario.agent_ids

    @property
    def agent_selection(self) -> str:
        return self._selector.current

    @property
    def step_idx(self) -> int:
        return self._step_idx

    def inventory(self, agent_id: str) -> int:
        if agent_id not in self._inventory:
            raise UnknownAgentError(agent_id)
        return self._inventory[agent_id]

    def pnl(self, agent_id: str) -> float:
        if agent_id not in self._pnl:
            raise UnknownAgentError(agent_id)
        return self._pnl[agent_id]

    def _build_agent_observation(self, agent_id: str) -> MultiAgentObservation:
        if agent_id not in self._inventory:
            raise UnknownAgentError(agent_id)
        mid_price = self._scenario.prices[min(self._step_idx, len(self._scenario.prices) - 1)]
        inventory = self._inventory[agent_id]
        pnl = self._pnl[agent_id]
        return MultiAgentObservation(
            agent_id=agent_id,
            step_idx=self._step_idx,
            mid_price=mid_price,
            inventory_signed=(0 if inventory == 0 else (1 if inventory > 0 else -1)),
            cumulative_pnl_usd=pnl,
            state_hash=_agent_obs_hash(
                seed=self._seed,
                agent_id=agent_id,
                step_idx=self._step_idx,
                mid_price=mid_price,
                inventory_signed=(0 if inventory == 0 else (1 if inventory > 0 else -1)),
                cumulative_pnl_usd=pnl,
            ),
        )

    def reset(
        self, *, seed: int = 0
    ) -> tuple[tuple[MultiAgentObservation, ...], dict[str, dict[str, Any]]]:
        if not isinstance(seed, int):
            raise TypeError(
                f"DIXMultiAgentEnv.reset(seed=...) must be int, got {type(seed).__name__}"
            )
        self._seed = seed
        self._step_idx = 0
        self._inventory = {agent_id: 0 for agent_id in self._scenario.agent_ids}
        self._pnl = {agent_id: 0.0 for agent_id in self._scenario.agent_ids}
        self._step_call_count = 0
        self._terminated = False
        self._selector.reset()
        observations = tuple(
            self._build_agent_observation(agent_id) for agent_id in self._scenario.agent_ids
        )
        infos: dict[str, dict[str, Any]] = {
            agent_id: {
                "inventory": 0,
                "pnl": 0.0,
            }
            for agent_id in self._scenario.agent_ids
        }
        return observations, infos

    def _validate_actions_dict(self, actions: dict[str, int]) -> None:
        unknown = set(actions) - set(self._scenario.agent_ids)
        if unknown:
            raise UnknownAgentError(sorted(unknown)[0])
        missing = set(self._scenario.agent_ids) - set(actions)
        if missing:
            raise ValueError(f"PARALLEL step actions missing agents: {sorted(missing)!r}")
        for agent_id, action in actions.items():
            if not isinstance(action, int):
                raise TypeError(f"actions[{agent_id!r}] must be int, got {type(action).__name__}")
            if action not in (
                MultiAgentAction.HOLD.value,
                MultiAgentAction.BUY.value,
                MultiAgentAction.SELL.value,
            ):
                raise ValueError(f"actions[{agent_id!r}] must be in {{0,1,2}}, got {action!r}")

    def _apply_action(self, agent_id: str, action: int, current_price: float) -> float:
        prev_inventory = self._inventory[agent_id]
        if action == MultiAgentAction.BUY.value:
            new_inventory = prev_inventory + 1
        elif action == MultiAgentAction.SELL.value:
            new_inventory = prev_inventory - 1
        else:
            new_inventory = prev_inventory
        if self._step_idx + 1 < len(self._scenario.prices):
            next_price = self._scenario.prices[self._step_idx + 1]
            price_delta = next_price - current_price
        else:
            price_delta = 0.0
        reward = float(prev_inventory) * price_delta
        self._inventory[agent_id] = new_inventory
        self._pnl[agent_id] += reward
        return reward

    def step(self, actions: dict[str, int]) -> MultiAgentStepResult:
        if self._terminated:
            raise RuntimeError(
                "DIXMultiAgentEnv.step called after termination — caller must "
                "honour terminations and call reset() first"
            )
        if self._scenario.mode is not MultiAgentMode.PARALLEL:
            raise WrongStepShapeError(
                "step(actions_dict) requires mode=PARALLEL — for AEC mode "
                "use step_aec(agent_id, action)"
            )
        if not isinstance(actions, dict):
            raise TypeError(
                f"DIXMultiAgentEnv.step(actions) must be dict, got {type(actions).__name__}"
            )
        self._validate_actions_dict(actions)
        self._step_call_count += 1
        if self._step_call_count > MAX_EPISODE_STEPS:
            raise MultiAgentEpisodeBudgetExceededError(
                f"exceeded MAX_EPISODE_STEPS={MAX_EPISODE_STEPS!r} without reset"
            )

        current_price = self._scenario.prices[min(self._step_idx, len(self._scenario.prices) - 1)]
        rewards_map: dict[str, float] = {}
        for agent_id in self._scenario.agent_ids:
            rewards_map[agent_id] = self._apply_action(agent_id, actions[agent_id], current_price)

        self._step_idx += 1
        terminated = (
            self._step_idx >= len(self._scenario.prices) - 1
            or self._step_idx >= self._scenario.max_steps
        )
        self._terminated = terminated

        observations = tuple(
            self._build_agent_observation(agent_id) for agent_id in self._scenario.agent_ids
        )
        return MultiAgentStepResult(
            observations=observations,
            rewards=tuple(
                (agent_id, rewards_map[agent_id]) for agent_id in self._scenario.agent_ids
            ),
            terminations=tuple((agent_id, terminated) for agent_id in self._scenario.agent_ids),
            truncations=tuple((agent_id, False) for agent_id in self._scenario.agent_ids),
            infos=tuple(
                (
                    agent_id,
                    tuple(
                        sorted(
                            {
                                "inventory": self._inventory[agent_id],
                                "pnl": self._pnl[agent_id],
                            }.items()
                        )
                    ),
                )
                for agent_id in self._scenario.agent_ids
            ),
        )

    def step_aec(self, agent_id: str, action: int) -> MultiAgentStepResult:
        if self._terminated:
            raise RuntimeError(
                "DIXMultiAgentEnv.step_aec called after termination — caller "
                "must honour terminations and call reset() first"
            )
        if self._scenario.mode is not MultiAgentMode.AEC:
            raise WrongStepShapeError(
                "step_aec(agent_id, action) requires mode=AEC — for "
                "PARALLEL mode use step(actions_dict)"
            )
        if agent_id not in self._inventory:
            raise UnknownAgentError(agent_id)
        if agent_id != self._selector.current:
            raise ValueError(
                f"AEC turn order violated — selector wants "
                f"{self._selector.current!r}, got {agent_id!r}"
            )
        if not isinstance(action, int):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if action not in (
            MultiAgentAction.HOLD.value,
            MultiAgentAction.BUY.value,
            MultiAgentAction.SELL.value,
        ):
            raise ValueError(f"action must be in {{0,1,2}}, got {action!r}")

        self._step_call_count += 1
        if self._step_call_count > MAX_EPISODE_STEPS:
            raise MultiAgentEpisodeBudgetExceededError(
                f"exceeded MAX_EPISODE_STEPS={MAX_EPISODE_STEPS!r} without reset"
            )

        was_last = self._selector.is_last()
        current_price = self._scenario.prices[min(self._step_idx, len(self._scenario.prices) - 1)]
        reward = self._apply_action(agent_id, action, current_price)

        if was_last:
            self._step_idx += 1
            terminated = (
                self._step_idx >= len(self._scenario.prices) - 1
                or self._step_idx >= self._scenario.max_steps
            )
            self._terminated = terminated
        else:
            terminated = False
        self._selector.next()

        obs = self._build_agent_observation(agent_id)
        return MultiAgentStepResult(
            observations=(obs,),
            rewards=((agent_id, reward),),
            terminations=((agent_id, terminated),),
            truncations=((agent_id, False),),
            infos=(
                (
                    agent_id,
                    tuple(
                        sorted(
                            {
                                "inventory": self._inventory[agent_id],
                                "pnl": self._pnl[agent_id],
                            }.items()
                        )
                    ),
                ),
            ),
        )

    def last(self) -> MultiAgentObservation:
        return self._build_agent_observation(self._selector.current)

    def render(self, mode: str | None = None) -> None:
        return None


def pettingzoo_multiagent_env_factory(*, scenario: MultiAgentScenario) -> Any:
    """Lazy-import factory returning a PettingZoo-shape wrapper.

    Only this function imports ``pettingzoo`` and ``gymnasium``.
    Callers that don't need PettingZoo spaces construct
    :class:`DIXMultiAgentEnv` directly.
    """

    try:
        import gymnasium as gym  # noqa: PLC0415
        from gymnasium import spaces  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "pettingzoo_multiagent_env_factory requires `pip install pettingzoo gymnasium`"
        ) from exc

    base_env = DIXMultiAgentEnv(scenario=scenario)

    class _GymWrappedMultiAgentEnv(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": [None]}
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(low=-1e18, high=1e18, shape=(4,), dtype=float)
        possible_agents = scenario.agent_ids

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[dict[str, tuple[float, ...]], dict[str, dict[str, Any]]]:
            observations, infos = base_env.reset(seed=seed or 0)
            obs_dict: dict[str, tuple[float, ...]] = {
                o.agent_id: (
                    float(o.step_idx),
                    o.mid_price,
                    float(o.inventory_signed),
                    o.cumulative_pnl_usd,
                )
                for o in observations
            }
            return obs_dict, infos

        def step(
            self, actions: dict[str, int]
        ) -> tuple[
            dict[str, tuple[float, ...]],
            dict[str, float],
            dict[str, bool],
            dict[str, bool],
            dict[str, dict[str, Any]],
        ]:
            result = base_env.step(actions)
            obs_dict: dict[str, tuple[float, ...]] = {
                o.agent_id: (
                    float(o.step_idx),
                    o.mid_price,
                    float(o.inventory_signed),
                    o.cumulative_pnl_usd,
                )
                for o in result.observations
            }
            return (
                obs_dict,
                result.rewards_dict(),
                result.terminations_dict(),
                result.truncations_dict(),
                {agent_id: dict(info) for agent_id, info in result.infos},
            )

        def render(self) -> None:
            return None

    return _GymWrappedMultiAgentEnv()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MULTIAGENT_ENV_VERSION",
    "MIN_AGENTS",
    "MAX_AGENTS",
    "MAX_EPISODE_STEPS",
    "MultiAgentMode",
    "MultiAgentAction",
    "MultiAgentEpisodeBudgetExceededError",
    "UnknownAgentError",
    "WrongStepShapeError",
    "MultiAgentScenario",
    "MultiAgentObservation",
    "MultiAgentStepResult",
    "AgentSelector",
    "DIXMultiAgentEnv",
    "pettingzoo_multiagent_env_factory",
)
