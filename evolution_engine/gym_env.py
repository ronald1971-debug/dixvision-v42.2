# ADAPTED FROM: DLR-RM/stable-baselines3
# (stable_baselines3/common/vec_env/base_vec_env.py — VecEnv step/reset Protocol;
#  stable_baselines3/common/base_class.py — BaseAlgorithm.learn() env contract;
#  Farama-Foundation/Gymnasium gymnasium/core.py — Env reset/step API shape.)
"""A-01.1 — DIXStrategyEnv: deterministic Gymnasium-shape environment.

Stable-Baselines3 trains policies by repeatedly calling
``env.reset()`` and ``env.step(action)`` on a Gymnasium-compatible
environment. Every observation, reward, and termination flag flows
through that interface. To plug DIX simulation runs into SB3 (and any
RL framework that consumes the same shape — CleanRL, TorchRL,
ElegantRL, Tianshou, Sample Factory, MushroomRL — see the rest of the
A-tier canonical doc) we expose **DIX SimulationEngine state as a
Gymnasium-shape environment**: not by importing Gymnasium, but by
matching its method signatures so any caller that already speaks the
contract can use the env unchanged.

What this module is
-------------------

* Pure-Python value objects + a stateful env class. No
  ``gymnasium`` / ``gym`` / ``stable_baselines3`` import at module
  top-level. The optional :func:`gymnasium_dix_strategy_env`
  factory lazy-imports ``gymnasium`` and registers DIX action /
  observation spaces, so the module is importable on a host that has
  never installed those packages.
* OFFLINE_ONLY tier: `evolution_engine` is an OFFLINE engine. The
  env is wall-clock-free, reads no environment variables, and
  performs no IO — the caller supplies a deterministic
  :class:`MarketDynamics` callable and a seed.
* INV-15 byte-identical replays. ``DIXStrategyEnv.reset(seed=K,
  scenario=S)`` followed by N calls to ``DIXStrategyEnv.step(action)``
  always returns identical ``Observation`` / ``reward`` / termination
  tuples on any host as long as the seed, scenario, action sequence,
  and dynamics callable are identical.

What survives from upstream
---------------------------

* The Gymnasium / SB3 step return shape:
  ``(observation, reward, terminated, truncated, info)``. SB3's
  ``BaseAlgorithm.collect_rollouts`` reads exactly this 5-tuple
  (Gymnasium ≥ 0.26 split the legacy 4-tuple ``done`` flag into
  ``terminated`` + ``truncated`` — DIXStrategyEnv adopts the new
  convention).
* The ``reset(*, seed, options) -> (observation, info)`` shape from
  Gymnasium's ``Env.reset`` API.
* The action-space discreteness assumption from SB3's discrete-action
  policies. We expose a frozen :class:`Action` value object instead
  of relying on ``gymnasium.spaces.Discrete`` — this keeps the env
  importable without Gymnasium and makes the action set explicit.

What we replaced
----------------

* SB3's ``VecEnv`` multiprocessing → DIX's
  :mod:`simulation.parallel_runner` (SIM-07). The single-env
  :class:`DIXStrategyEnv` is the leaf the parallel runner steps in
  parallel; we never import SB3's ``SubprocVecEnv``.
* Gymnasium's ``np.random.Generator`` PRNG plumbing → an explicit
  caller-supplied seed + a deterministic
  :class:`MarketDynamics`-supplied next-state. INV-15 forbids any
  hidden PRNG state.
* SB3's Tensorboard logger → callbacks injected by the caller (see
  A-01.2 ``evolution_engine/sandbox.py``). The env itself emits no
  side effects.

Authority constraints (manifest §H1)
-----------------------------------

* RUNTIME_SAFE-shape but tier=OFFLINE — no IO, no clock, no global
  state, no PRNG. The env class is the only stateful object and it
  resets cleanly on every :meth:`DIXStrategyEnv.reset` call.
* No engine cross-imports. The env never reads ``registry/`` /
  ``system_engine`` / ``governance_engine``. Strategy parameters
  flow in through :class:`EpisodeConfig` and the caller-supplied
  :class:`MarketDynamics`.
* INV-15 byte-identical replay. Caller supplies the seed; the seed
  is folded into a deterministic :class:`Observation` hash.

Refs:
- ``DIX_MASTER_CANONICAL.md`` lines 768–808 (A-01 stable-baselines3 spec).
- ``simulation/parallel_runner.py`` (SIM-07 N-reality runner — same
  ``StepFn`` shape).
- ``intelligence_engine/cognitive/litellm_router.py`` (S-12 — same
  Protocol-injected transport seam pattern).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("gymnasium", "stable-baselines3")
"""``gymnasium`` is needed only by the optional
:func:`gymnasium_dix_strategy_env` factory; ``stable-baselines3`` is
needed only by the A-01.2 sandbox module that imports this env. The
core :class:`DIXStrategyEnv` class is pure stdlib."""

MAX_EPISODE_STEPS: int = 100_000
"""Hard upper bound on episode length — the env raises
:class:`EpisodeBudgetExceededError` once the caller pumps more than
this many ``step`` calls without a reset. Protects training loops
that forget to honour ``terminated`` / ``truncated``. Mirrors the
30s timeout ceiling we use in S-12 ``LiteLLMRouter`` — same
defensive-cap pattern, different units."""

MIN_INITIAL_NOTIONAL_USD: float = 1.0
"""Lower bound on :class:`EpisodeConfig.initial_notional_usd`. Below
this the reward signal degenerates (every action rounds to a $0
position) and PPO's gradient becomes uninformative."""


class TradeAction(enum.IntEnum):
    """Discrete action set for the long/flat/short policy.

    Integer values match Gymnasium's ``spaces.Discrete(3)`` indexing
    convention (0=HOLD, 1=BUY, 2=SELL) so a downstream policy that
    emits ``int`` actions can be plugged in unchanged.
    """

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Frozen config shared across every step of a single episode.

    Mirrors the role :class:`~core.contracts.simulation.RealityScenario`
    plays in SIM-07: one frozen scenario, many seeded realities. Here
    one frozen episode-config drives many seeded action sequences.
    """

    initial_notional_usd: float
    max_steps: int
    reward_scale: float = 1.0
    drawdown_penalty_weight: float = 0.5
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_notional_usd):
            raise ValueError(
                "EpisodeConfig.initial_notional_usd must be finite, got "
                f"{self.initial_notional_usd!r}"
            )
        if self.initial_notional_usd < MIN_INITIAL_NOTIONAL_USD:
            raise ValueError(
                "EpisodeConfig.initial_notional_usd must be >= "
                f"{MIN_INITIAL_NOTIONAL_USD!r}, got "
                f"{self.initial_notional_usd!r}"
            )
        if self.max_steps <= 0:
            raise ValueError(f"EpisodeConfig.max_steps must be positive, got {self.max_steps!r}")
        if self.max_steps > MAX_EPISODE_STEPS:
            raise ValueError(
                f"EpisodeConfig.max_steps must be <= {MAX_EPISODE_STEPS!r}, got {self.max_steps!r}"
            )
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0.0:
            raise ValueError(
                "EpisodeConfig.reward_scale must be a positive finite "
                f"number, got {self.reward_scale!r}"
            )
        if not math.isfinite(self.drawdown_penalty_weight) or self.drawdown_penalty_weight < 0.0:
            raise ValueError(
                "EpisodeConfig.drawdown_penalty_weight must be a "
                "non-negative finite number, got "
                f"{self.drawdown_penalty_weight!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """Frozen environment observation.

    The fields are deliberately the minimum a discrete long/flat/short
    PPO policy needs:

    * ``step_idx`` — monotonically increasing within an episode,
      starts at 0 on reset.
    * ``mid_price`` — caller-supplied "current" market mid (e.g. from
      a replayed L2 book). Finite, positive.
    * ``inventory_signed`` — sign of the held position
      (-1 short / 0 flat / +1 long). The notional size is held by the
      env; the policy only sees the sign so the action set is
      stationary.
    * ``cumulative_pnl_usd`` — running pnl of the episode in USD
      (positive = profit). Finite.
    * ``state_hash`` — 16-hex-char content-address of the
      ``(seed, step_idx, mid_price, inventory_signed,
      cumulative_pnl_usd)`` tuple. Stable across hosts (BLAKE2b /
      stdlib only) so two replays of the same episode produce
      byte-identical observations.
    """

    step_idx: int
    mid_price: float
    inventory_signed: int
    cumulative_pnl_usd: float
    state_hash: str

    def __post_init__(self) -> None:
        if self.step_idx < 0:
            raise ValueError(f"Observation.step_idx must be non-negative, got {self.step_idx!r}")
        if not math.isfinite(self.mid_price) or self.mid_price <= 0.0:
            raise ValueError(
                f"Observation.mid_price must be a positive finite number, got {self.mid_price!r}"
            )
        if self.inventory_signed not in (-1, 0, 1):
            raise ValueError(
                "Observation.inventory_signed must be in {-1, 0, 1}, "
                f"got {self.inventory_signed!r}"
            )
        if not math.isfinite(self.cumulative_pnl_usd):
            raise ValueError(
                f"Observation.cumulative_pnl_usd must be finite, got {self.cumulative_pnl_usd!r}"
            )
        if len(self.state_hash) != 16:
            raise ValueError(
                f"Observation.state_hash must be a 16-hex-char digest, got {self.state_hash!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.state_hash):
            raise ValueError(
                f"Observation.state_hash must be lowercase hex, got {self.state_hash!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Transition:
    """Frozen ``step()`` return record produced by :class:`MarketDynamics`.

    The dynamics function receives ``(prev_obs, action, seed)`` and
    returns one of these. The env then validates it, advances its
    internal state, and re-emits the canonical 5-tuple to the caller.
    """

    next_mid_price: float
    realised_pnl_usd: float
    drawdown_usd: float
    next_inventory_signed: int
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.next_mid_price) or self.next_mid_price <= 0.0:
            raise ValueError(
                "Transition.next_mid_price must be a positive finite "
                f"number, got {self.next_mid_price!r}"
            )
        if not math.isfinite(self.realised_pnl_usd):
            raise ValueError(
                f"Transition.realised_pnl_usd must be finite, got {self.realised_pnl_usd!r}"
            )
        if not math.isfinite(self.drawdown_usd) or self.drawdown_usd < 0.0:
            raise ValueError(
                "Transition.drawdown_usd must be a non-negative finite "
                f"number, got {self.drawdown_usd!r}"
            )
        if self.next_inventory_signed not in (-1, 0, 1):
            raise ValueError(
                "Transition.next_inventory_signed must be in "
                f"{{-1, 0, 1}}, got {self.next_inventory_signed!r}"
            )


@runtime_checkable
class MarketDynamics(Protocol):
    """Caller-supplied market dynamics — the leaf the env consults.

    The Protocol is the **only** place the env interacts with market
    state. By forcing the caller to supply the dynamics (instead of
    embedding a market simulator inside the env) we keep INV-15:
    identical seed + identical action sequence + identical dynamics →
    identical episode.
    """

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition: ...

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float: ...


SeedFactory = Callable[[int, int], int]
"""``(episode_seed, step_idx) -> per_step_seed`` deterministic seed
splitter. The caller can pass any pure function — the default
splitter mirrors splitmix64."""


class EpisodeBudgetExceededError(RuntimeError):
    """Raised by :meth:`DIXStrategyEnv.step` if the caller exceeds
    :attr:`EpisodeConfig.max_steps` (or the global
    :data:`MAX_EPISODE_STEPS` ceiling) without reset."""


class EpisodeNotStartedError(RuntimeError):
    """Raised by :meth:`DIXStrategyEnv.step` if the caller never
    called :meth:`DIXStrategyEnv.reset`."""


def _splitmix64(x: int) -> int:
    """Stateless 64-bit hash. Used as the default
    :data:`SeedFactory`. Pure stdlib (mirrors S-02.2 ``JitteredLatency``
    splitmix64)."""

    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _default_seed_factory(episode_seed: int, step_idx: int) -> int:
    return _splitmix64(_splitmix64(episode_seed) ^ step_idx)


def _hash_state(
    *,
    seed: int,
    step_idx: int,
    mid_price: float,
    inventory_signed: int,
    cumulative_pnl_usd: float,
) -> str:
    """Deterministic 16-hex-char content hash of an observation tuple.

    Formats every float with ``repr`` (round-trippable) so two hosts
    with different default float formatting still agree on the hash.
    """

    payload = "|".join(
        (
            repr(int(seed)),
            repr(int(step_idx)),
            repr(float(mid_price)),
            repr(int(inventory_signed)),
            repr(float(cumulative_pnl_usd)),
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


class DIXStrategyEnv:
    """Stateful Gymnasium-shape env wrapping a :class:`MarketDynamics`.

    Not a subclass of ``gymnasium.Env`` (we never import gymnasium
    here). The class is deliberately duck-typed: any caller that
    speaks the Gymnasium 0.26+ ``reset`` / ``step`` shape can use it
    unchanged. The optional :func:`gymnasium_dix_strategy_env` factory
    wraps this in an actual ``gymnasium.Env`` subclass for callers
    that need ``isinstance`` to pass.
    """

    __slots__ = (
        "_dynamics",
        "_seed_factory",
        "_config",
        "_episode_seed",
        "_current_obs",
        "_step_count",
        "_terminated",
        "_truncated",
    )

    def __init__(
        self,
        dynamics: MarketDynamics,
        *,
        seed_factory: SeedFactory = _default_seed_factory,
    ) -> None:
        if not isinstance(dynamics, MarketDynamics):
            raise TypeError(
                "DIXStrategyEnv.dynamics must implement the "
                "MarketDynamics Protocol, got "
                f"{type(dynamics).__name__}"
            )
        self._dynamics: MarketDynamics = dynamics
        self._seed_factory: SeedFactory = seed_factory
        self._config: EpisodeConfig | None = None
        self._episode_seed: int | None = None
        self._current_obs: Observation | None = None
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    @property
    def action_space_n(self) -> int:
        """Number of discrete actions (matches ``spaces.Discrete(n)``)."""

        return len(TradeAction)

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Stable observation field order — useful when callers project
        :class:`Observation` into a numpy array for SB3."""

        return (
            "step_idx",
            "mid_price",
            "inventory_signed",
            "cumulative_pnl_usd",
        )

    def reset(
        self,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> tuple[Observation, Mapping[str, Any]]:
        """Start a new episode. Mirrors ``gymnasium.Env.reset`` shape."""

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"DIXStrategyEnv.reset requires int seed, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"DIXStrategyEnv.reset seed must be non-negative, got {seed!r}")
        if not isinstance(config, EpisodeConfig):
            raise TypeError(
                f"DIXStrategyEnv.reset requires EpisodeConfig, got {type(config).__name__}"
            )

        initial_mid = self._dynamics.initial_mid_price(seed=seed, config=config)
        if not math.isfinite(initial_mid) or initial_mid <= 0.0:
            raise ValueError(
                "MarketDynamics.initial_mid_price must return a "
                f"positive finite number, got {initial_mid!r}"
            )

        self._config = config
        self._episode_seed = seed
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._current_obs = Observation(
            step_idx=0,
            mid_price=float(initial_mid),
            inventory_signed=0,
            cumulative_pnl_usd=0.0,
            state_hash=_hash_state(
                seed=seed,
                step_idx=0,
                mid_price=float(initial_mid),
                inventory_signed=0,
                cumulative_pnl_usd=0.0,
            ),
        )
        info: Mapping[str, Any] = {
            "episode_seed": seed,
            "max_steps": config.max_steps,
        }
        return self._current_obs, info

    def step(
        self,
        action: TradeAction,
    ) -> tuple[Observation, float, bool, bool, Mapping[str, Any]]:
        """Advance the env. Mirrors ``gymnasium.Env.step`` shape."""

        if self._current_obs is None or self._config is None:
            raise EpisodeNotStartedError("DIXStrategyEnv.step called before reset")
        if self._terminated or self._truncated:
            raise EpisodeBudgetExceededError(
                "DIXStrategyEnv.step called on a terminated episode; call reset() first"
            )
        if not isinstance(action, TradeAction):
            try:
                action = TradeAction(int(action))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"DIXStrategyEnv.step action must coerce to TradeAction, got {action!r}"
                ) from exc
        if self._step_count >= self._config.max_steps:
            raise EpisodeBudgetExceededError(
                f"DIXStrategyEnv.step exceeded EpisodeConfig.max_steps={self._config.max_steps!r}"
            )
        if self._step_count >= MAX_EPISODE_STEPS:
            raise EpisodeBudgetExceededError(
                f"DIXStrategyEnv.step exceeded MAX_EPISODE_STEPS={MAX_EPISODE_STEPS!r}"
            )
        assert self._episode_seed is not None  # narrowed by reset() guard

        per_step_seed = self._seed_factory(self._episode_seed, self._step_count)
        transition = self._dynamics.step(
            self._current_obs,
            action,
            seed=per_step_seed,
            config=self._config,
        )
        if not isinstance(transition, Transition):
            raise TypeError(
                f"MarketDynamics.step must return Transition, got {type(transition).__name__}"
            )

        next_step_idx = self._step_count + 1
        next_pnl = self._current_obs.cumulative_pnl_usd + transition.realised_pnl_usd
        if not math.isfinite(next_pnl):
            raise ValueError(
                f"DIXStrategyEnv.step produced non-finite cumulative pnl: {next_pnl!r}"
            )
        next_obs = Observation(
            step_idx=next_step_idx,
            mid_price=transition.next_mid_price,
            inventory_signed=transition.next_inventory_signed,
            cumulative_pnl_usd=next_pnl,
            state_hash=_hash_state(
                seed=self._episode_seed,
                step_idx=next_step_idx,
                mid_price=transition.next_mid_price,
                inventory_signed=transition.next_inventory_signed,
                cumulative_pnl_usd=next_pnl,
            ),
        )

        reward = (
            self._config.reward_scale * transition.realised_pnl_usd
            - self._config.drawdown_penalty_weight * transition.drawdown_usd
        )
        if not math.isfinite(reward):
            raise ValueError(f"DIXStrategyEnv.step produced non-finite reward: {reward!r}")

        truncated = bool(transition.truncated) or (next_step_idx >= self._config.max_steps)
        terminated = bool(transition.terminated)

        info_payload: dict[str, Any] = dict(transition.info)
        info_payload.setdefault("step_seed", per_step_seed)
        info_payload.setdefault("step_idx", next_step_idx)
        info_payload.setdefault("drawdown_usd", float(transition.drawdown_usd))

        self._current_obs = next_obs
        self._step_count = next_step_idx
        self._terminated = terminated
        self._truncated = truncated
        return next_obs, float(reward), terminated, truncated, info_payload

    @property
    def is_episode_done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def current_observation(self) -> Observation | None:
        return self._current_obs


def gymnasium_dix_strategy_env(
    dynamics: MarketDynamics,
    *,
    seed_factory: SeedFactory = _default_seed_factory,
) -> Any:
    """Optional factory: wrap :class:`DIXStrategyEnv` as a real
    ``gymnasium.Env`` so callers that need ``isinstance(env,
    gymnasium.Env)`` to pass can plug it into SB3.

    Lazy-imports gymnasium. The factory raises ``ImportError`` (with a
    helpful pip-install hint) if gymnasium isn't available — this is
    the deliberate AST contract: the rest of the module never imports
    gymnasium, so the env class itself stays usable on a host that
    has never installed the package.
    """

    try:
        import gymnasium  # type: ignore[import-not-found]  # noqa: I001
        from gymnasium import spaces  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised when gym missing
        raise ImportError(
            "gymnasium_dix_strategy_env requires the optional "
            "'gymnasium' package — install with "
            "'pip install gymnasium' (NEW_PIP_DEPENDENCIES tuple in "
            "evolution_engine/gym_env.py flags this)."
        ) from exc

    inner = DIXStrategyEnv(dynamics, seed_factory=seed_factory)

    class _GymnasiumWrapper(gymnasium.Env):  # type: ignore[misc]
        action_space = spaces.Discrete(len(TradeAction))
        # Box space matches the Observation fields; mid_price unbounded
        # above, inventory_signed in {-1,0,1}, pnl unbounded.
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(4,))

        def reset(  # type: ignore[override]
            self, *, seed: int | None = None, options: Any = None
        ) -> tuple[Any, Mapping[str, Any]]:
            if not isinstance(options, Mapping) or "config" not in options:
                raise ValueError(
                    "_GymnasiumWrapper.reset requires options={"
                    "'config': EpisodeConfig(...)}; the env wraps a "
                    "DIX strategy episode, not a generic Gymnasium "
                    "scene."
                )
            if seed is None:
                raise ValueError(
                    "_GymnasiumWrapper.reset requires explicit seed= "
                    "(INV-15: caller-supplied seed)."
                )
            obs, info = inner.reset(seed=seed, config=options["config"])
            return _observation_to_tuple(obs), info

        def step(  # type: ignore[override]
            self, action: int
        ) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
            obs, reward, terminated, truncated, info = inner.step(TradeAction(int(action)))
            return (
                _observation_to_tuple(obs),
                reward,
                terminated,
                truncated,
                info,
            )

    return _GymnasiumWrapper()


def _observation_to_tuple(
    obs: Observation,
) -> tuple[float, float, float, float]:
    """Project :class:`Observation` into a 4-tuple in the order given
    by :attr:`DIXStrategyEnv.observation_keys`."""

    return (
        float(obs.step_idx),
        float(obs.mid_price),
        float(obs.inventory_signed),
        float(obs.cumulative_pnl_usd),
    )
