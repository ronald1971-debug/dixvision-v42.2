# ADAPTED FROM: AminHP/gym-anytrading
# (gym_anytrading/envs/trading_env.py — base TradingEnv shape;
#  gym_anytrading/envs/stocks_env.py — StocksEnv reward function;
#  gym_anytrading/envs/forex_env.py — ForexEnv profit accumulation;
#  Farama-Foundation/Gymnasium gymnasium/core.py — Env reset/step API shape.)
"""C-29 — DIXAnytradingEnv: reference benchmark environment.

`gym-anytrading` is a tiny, didactic Gymnasium environment used as a
warm-start for RL trading-bot tutorials. It exposes a discrete
``Sell / Buy`` action set over a fixed pre-loaded price window. The
project value is its **simplicity** — it is the easiest possible env
that still has the right shape for SB3 / CleanRL / TorchRL.

DIX uses this as a **benchmark baseline**, not as a primary training
env. The main training env is :class:`evolution_engine.gym_env.DIXStrategyEnv`
(A-01.1) which is connected to the SIM-07 N-reality runner. This
``DIXAnytradingEnv`` lets us answer a single benchmarking question:
"how does our PPO / actor-critic / CMA-ES wrapping perform on the
**same trivial benchmark env** the rest of the field publishes
numbers on?" — i.e. it is the comparison rig, not the production env.

What this module is
-------------------

* Pure-Python value objects + a stateful env class. No
  ``gymnasium`` / ``gym_anytrading`` / ``numpy`` import at module
  top-level. The optional :func:`gymnasium_anytrading_env_factory`
  lazy-imports ``gymnasium`` and registers a ``spaces.Discrete(2)``
  action space + a ``spaces.Box`` observation space, so the module
  is importable on a host that has never installed either package.
* OFFLINE_ONLY tier: `evolution_engine` is an OFFLINE engine. The
  env is wall-clock-free, reads no environment variables, performs
  no IO — caller supplies a frozen price series and a seed.
* INV-15 byte-identical replays. ``DIXAnytradingEnv.reset(seed=K)``
  followed by N calls to ``DIXAnytradingEnv.step(action)`` always
  returns identical ``Observation`` / ``reward`` / termination
  tuples on any host as long as the seed, price series, window size,
  and action sequence are identical.

What survives from upstream gym-anytrading
------------------------------------------

* The discrete action set ``Sell=0 / Buy=1``. This is the
  long/flat-only shape — there is no separate ``HOLD`` action, the
  env tracks ``Position.SHORT`` / ``Position.LONG`` and an action
  that matches the current position is a no-op.
* The "step return" 5-tuple
  ``(observation, reward, terminated, truncated, info)`` from
  Gymnasium ≥ 0.26 (gym-anytrading >= 2.0 adopted this signature).
* The reward formulation: when the position flips (Long → Short or
  Short → Long), the realized reward is the price-delta over the
  held window (StocksEnv shape: ``current_price - entry_price``).
  HOLD-shape transitions accrue zero reward — this matches the
  upstream ``_calculate_reward`` shape.
* The observation: a flattened window of recent price deltas
  appended with the current position flag. Upstream returns
  ``signal_features[start:end]`` — we return the same shape as a
  tuple of floats.

What we replaced
----------------

* ``numpy`` arrays for ``signal_features`` and observations →
  ``tuple[float, ...]``. Upstream uses ``ndarray`` because Gymnasium
  ``Box`` spaces sample/serialise numpy. DIX RL stack (CleanRL /
  TorchRL / SB3 adapters) all accept tuples — and we avoid the
  numpy top-level import so the module loads without it.
* Upstream's ``frame_bound`` slicing on a global numpy frame →
  caller-supplied ``prices: tuple[float, ...]`` with an explicit
  ``window_size`` parameter. Upstream's slicing is the most
  common source of off-by-one bugs in gym-anytrading tutorials —
  we replace it with a single canonical computation pinned by tests.
* Upstream's ``np.random.RandomState`` PRNG plumbing → an explicit
  caller-supplied seed folded into a stdlib-only :func:`hashlib`
  state hash. INV-15 forbids any hidden PRNG state.
* Upstream's ``matplotlib`` rendering → ``render_mode=None`` only.
  Rendering is a strict no-op; this is a backend benchmark env, the
  caller plots externally if needed.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE_ONLY tier. No IO, no clock, no PRNG, no global state. The
  env class is the only stateful object and it resets cleanly on
  every :meth:`DIXAnytradingEnv.reset` call.
* No engine cross-imports. The env never reads ``registry/`` /
  ``system_engine`` / ``governance_engine`` / ``execution_engine``.
* INV-15 byte-identical replay. Caller supplies the seed and price
  series; the seed is folded into a deterministic
  :class:`Observation` hash.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-29 (gym-anytrading reference env).
- ``evolution_engine/gym_env.py`` (A-01.1 DIXStrategyEnv — primary
  env this benchmarks against).
- ``learning_engine/lanes/policy_distillation_torchrl.py`` (C-28 —
  consumes any Gymnasium-shape env for actor-critic training).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from typing import Any, Final

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("gym-anytrading", "gymnasium")
"""``gym-anytrading`` is the upstream project we mirror; ``gymnasium``
is needed only by the optional :func:`gymnasium_anytrading_env_factory`
factory. The core :class:`DIXAnytradingEnv` class is pure stdlib."""

ANYTRADING_ENV_VERSION: Final[str] = "c-29-anytrading-1"
"""Version tag woven into the canonical content-hash of every
observation so a replay run can verify the env-shape is unchanged."""

MIN_WINDOW_SIZE: Final[int] = 2
"""Minimum number of prices the observation window covers. A window
of size 1 leaves zero deltas; size 2 is the smallest informative
shape. Mirrors upstream's ``window_size=10`` default lower bound."""

MAX_EPISODE_STEPS: Final[int] = 1_000_000
"""Hard upper bound on episode length — the env raises
:class:`AnytradingEpisodeBudgetExceededError` once the caller pumps
more than this many ``step`` calls without a reset. Defensive cap;
real benchmark runs are bounded by ``len(prices) - window_size``."""


class AnytradingAction(enum.IntEnum):
    """Discrete action set — long / flat positions only.

    Integer values match upstream gym-anytrading's
    ``Actions.Sell.value=0`` / ``Actions.Buy.value=1`` indexing so a
    downstream policy that emits ``int`` actions can be plugged in
    unchanged.
    """

    SELL = 0
    BUY = 1


class AnytradingPosition(enum.IntEnum):
    """Held-position state — long / flat shapes.

    Mirrors upstream gym-anytrading's ``Positions`` enum exactly:
    ``Positions.Short.value=0`` (out / flat / short) and
    ``Positions.Long.value=1`` (long).
    """

    SHORT = 0
    LONG = 1


class AnytradingEpisodeBudgetExceededError(RuntimeError):
    """Raised when a caller pumps more than :data:`MAX_EPISODE_STEPS`
    ``step`` calls without resetting the env. Indicates a training
    loop that ignored ``terminated`` / ``truncated``."""


@dataclasses.dataclass(frozen=True, slots=True)
class AnytradingConfig:
    """Frozen episode configuration.

    Mirrors the role :class:`EpisodeConfig` plays in A-01.1: one frozen
    config drives the whole episode. Caller supplies the price series
    and window size; the env never mutates either.
    """

    prices: tuple[float, ...]
    window_size: int
    trade_fee_bid_percent: float = 0.0
    trade_fee_ask_percent: float = 0.0
    initial_position: AnytradingPosition = AnytradingPosition.SHORT

    def __post_init__(self) -> None:
        if not self.prices:
            raise ValueError("AnytradingConfig.prices must be non-empty")
        for idx, price in enumerate(self.prices):
            if not isinstance(price, (int, float)):
                raise TypeError(
                    f"AnytradingConfig.prices[{idx}] must be float, got {type(price).__name__}"
                )
            if not math.isfinite(float(price)):
                raise ValueError(f"AnytradingConfig.prices[{idx}] must be finite, got {price!r}")
            if float(price) <= 0.0:
                raise ValueError(f"AnytradingConfig.prices[{idx}] must be positive, got {price!r}")
        if self.window_size < MIN_WINDOW_SIZE:
            raise ValueError(
                "AnytradingConfig.window_size must be >= "
                f"{MIN_WINDOW_SIZE!r}, got {self.window_size!r}"
            )
        if self.window_size > len(self.prices):
            raise ValueError(
                "AnytradingConfig.window_size must be <= len(prices)="
                f"{len(self.prices)!r}, got {self.window_size!r}"
            )
        for label, fee in (
            ("trade_fee_bid_percent", self.trade_fee_bid_percent),
            ("trade_fee_ask_percent", self.trade_fee_ask_percent),
        ):
            if not math.isfinite(fee):
                raise ValueError(f"AnytradingConfig.{label} must be finite, got {fee!r}")
            if fee < 0.0:
                raise ValueError(f"AnytradingConfig.{label} must be non-negative, got {fee!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class AnytradingObservation:
    """Frozen environment observation.

    Fields chosen to match upstream gym-anytrading's
    ``signal_features`` shape while staying pure-stdlib:

    * ``window`` — tuple of (window_size) most-recent prices.
    * ``deltas`` — tuple of (window_size - 1) consecutive price
      differences (``window[i+1] - window[i]``). This is the
      upstream "signal" channel.
    * ``position`` — current held position flag (SHORT=0 / LONG=1).
    * ``step_idx`` — monotonically increasing within an episode,
      starts at ``window_size - 1`` on reset (matches upstream's
      ``_current_tick`` semantics).
    * ``state_hash`` — 16-hex-char content-address of the
      ``(seed, step_idx, position, window, deltas)`` tuple. Stable
      across hosts (BLAKE2b / stdlib only) so two replays of the
      same episode produce byte-identical observations.
    """

    step_idx: int
    window: tuple[float, ...]
    deltas: tuple[float, ...]
    position: AnytradingPosition
    state_hash: str

    def __post_init__(self) -> None:
        if self.step_idx < 0:
            raise ValueError(f"AnytradingObservation.step_idx must be >= 0, got {self.step_idx!r}")
        if len(self.window) < MIN_WINDOW_SIZE:
            raise ValueError(
                "AnytradingObservation.window must have >= "
                f"{MIN_WINDOW_SIZE!r} elements, got {len(self.window)!r}"
            )
        if len(self.deltas) != len(self.window) - 1:
            raise ValueError(
                "AnytradingObservation.deltas must have len(window)-1="
                f"{len(self.window) - 1!r} elements, got {len(self.deltas)!r}"
            )
        if len(self.state_hash) != 16:
            raise ValueError(
                f"AnytradingObservation.state_hash must be 16 hex chars, got {self.state_hash!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class AnytradingStepResult:
    """Frozen result of one :meth:`DIXAnytradingEnv.step` call.

    Mirrors the upstream Gymnasium ≥ 0.26 5-tuple shape
    ``(observation, reward, terminated, truncated, info)`` but as a
    structured record so the info dict layout is pinned.
    """

    observation: AnytradingObservation
    reward: float
    terminated: bool
    truncated: bool
    total_reward: float
    total_profit: float
    position_changed: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError(f"AnytradingStepResult.reward must be finite, got {self.reward!r}")
        if not math.isfinite(self.total_reward):
            raise ValueError(
                f"AnytradingStepResult.total_reward must be finite, got {self.total_reward!r}"
            )
        if not math.isfinite(self.total_profit):
            raise ValueError(
                f"AnytradingStepResult.total_profit must be finite, got {self.total_profit!r}"
            )


def _state_hash(
    *,
    seed: int,
    step_idx: int,
    position: AnytradingPosition,
    window: tuple[float, ...],
    deltas: tuple[float, ...],
) -> str:
    """Return a 16-hex-char BLAKE2b-16 digest of the observation
    state — deterministic across hosts, used to pin INV-15 replays."""

    payload_parts: list[str] = [
        f"v={ANYTRADING_ENV_VERSION}",
        f"seed={seed}",
        f"step_idx={step_idx}",
        f"position={position.value}",
        "window=" + ",".join(f"{p:.17g}" for p in window),
        "deltas=" + ",".join(f"{d:.17g}" for d in deltas),
    ]
    payload = "|".join(payload_parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return digest


class DIXAnytradingEnv:
    """Reference benchmark environment in the gym-anytrading shape.

    Use this env as a baseline for comparing DIX RL training runs
    against published gym-anytrading numbers. The primary training
    env is :class:`evolution_engine.gym_env.DIXStrategyEnv` (A-01.1).
    """

    __slots__ = (
        "_config",
        "_seed",
        "_current_tick",
        "_last_trade_tick",
        "_position",
        "_total_reward",
        "_total_profit",
        "_step_call_count",
        "_terminated",
    )

    def __init__(self, *, config: AnytradingConfig) -> None:
        self._config: AnytradingConfig = config
        self._seed: int = 0
        self._current_tick: int = config.window_size - 1
        self._last_trade_tick: int = config.window_size - 1
        self._position: AnytradingPosition = config.initial_position
        self._total_reward: float = 0.0
        self._total_profit: float = 1.0
        self._step_call_count: int = 0
        self._terminated: bool = False

    @property
    def config(self) -> AnytradingConfig:
        return self._config

    @property
    def position(self) -> AnytradingPosition:
        return self._position

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def total_profit(self) -> float:
        return self._total_profit

    def _build_observation(self) -> AnytradingObservation:
        end = self._current_tick + 1
        start = end - self._config.window_size
        window = self._config.prices[start:end]
        deltas = tuple(window[i + 1] - window[i] for i in range(len(window) - 1))
        return AnytradingObservation(
            step_idx=self._current_tick,
            window=window,
            deltas=deltas,
            position=self._position,
            state_hash=_state_hash(
                seed=self._seed,
                step_idx=self._current_tick,
                position=self._position,
                window=window,
                deltas=deltas,
            ),
        )

    def reset(self, *, seed: int = 0) -> tuple[AnytradingObservation, dict[str, Any]]:
        if not isinstance(seed, int):
            raise TypeError(
                f"DIXAnytradingEnv.reset(seed=...) must be int, got {type(seed).__name__}"
            )
        self._seed = seed
        self._current_tick = self._config.window_size - 1
        self._last_trade_tick = self._config.window_size - 1
        self._position = self._config.initial_position
        self._total_reward = 0.0
        self._total_profit = 1.0
        self._step_call_count = 0
        self._terminated = False
        info: dict[str, Any] = {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self._position.value,
        }
        return self._build_observation(), info

    def step(self, action: int) -> AnytradingStepResult:
        if self._terminated:
            raise RuntimeError(
                "DIXAnytradingEnv.step called after termination — caller must "
                "honour 'terminated' / 'truncated' and call reset() first"
            )
        if not isinstance(action, int):
            raise TypeError(
                f"DIXAnytradingEnv.step(action) must be int, got {type(action).__name__}"
            )
        if action not in (AnytradingAction.SELL.value, AnytradingAction.BUY.value):
            raise ValueError(f"DIXAnytradingEnv.step(action) must be in {{0, 1}}, got {action!r}")

        self._step_call_count += 1
        if self._step_call_count > MAX_EPISODE_STEPS:
            raise AnytradingEpisodeBudgetExceededError(
                f"DIXAnytradingEnv exceeded MAX_EPISODE_STEPS={MAX_EPISODE_STEPS!r} without reset"
            )

        prev_position = self._position
        self._current_tick += 1

        step_reward = 0.0
        position_changed = False
        new_position = prev_position
        if action == AnytradingAction.BUY.value and prev_position is AnytradingPosition.SHORT:
            new_position = AnytradingPosition.LONG
            position_changed = True
        elif action == AnytradingAction.SELL.value and prev_position is AnytradingPosition.LONG:
            new_position = AnytradingPosition.SHORT
            position_changed = True

        if position_changed:
            current_price = self._config.prices[self._current_tick]
            last_trade_price = self._config.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price
            if prev_position is AnytradingPosition.LONG:
                step_reward = price_diff
                self._total_profit *= current_price / last_trade_price
            else:
                step_reward = 0.0
            self._last_trade_tick = self._current_tick

        self._position = new_position
        self._total_reward += step_reward

        terminated = self._current_tick >= len(self._config.prices) - 1
        truncated = False
        self._terminated = terminated

        observation = self._build_observation()
        return AnytradingStepResult(
            observation=observation,
            reward=step_reward,
            terminated=terminated,
            truncated=truncated,
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position_changed=position_changed,
        )

    def render(self, mode: str | None = None) -> None:
        return None


def gymnasium_anytrading_env_factory(*, config: AnytradingConfig) -> Any:
    """Lazy-import factory returning a Gymnasium-compatible env wrapper.

    Only this function imports ``gymnasium``. Callers that don't need
    Gymnasium spaces (e.g. the DIX RL training stack) construct
    :class:`DIXAnytradingEnv` directly.
    """

    try:
        import gymnasium as gym  # noqa: PLC0415
        from gymnasium import spaces  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "gymnasium_anytrading_env_factory requires `pip install gymnasium`"
        ) from exc

    base_env = DIXAnytradingEnv(config=config)

    class _GymWrappedAnytradingEnv(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": [None]}
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(
            low=-1e18,
            high=1e18,
            shape=(2 * config.window_size - 1 + 1,),
            dtype=float,
        )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[tuple[float, ...], dict[str, Any]]:
            obs, info = base_env.reset(seed=seed or 0)
            flat: tuple[float, ...] = (
                *obs.window,
                *obs.deltas,
                float(obs.position.value),
            )
            return flat, info

        def step(self, action: int) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
            result = base_env.step(action)
            obs = result.observation
            flat: tuple[float, ...] = (
                *obs.window,
                *obs.deltas,
                float(obs.position.value),
            )
            info: dict[str, Any] = {
                "total_reward": result.total_reward,
                "total_profit": result.total_profit,
                "position_changed": result.position_changed,
            }
            return (
                flat,
                result.reward,
                result.terminated,
                result.truncated,
                info,
            )

        def render(self) -> None:
            return None

    return _GymWrappedAnytradingEnv()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "ANYTRADING_ENV_VERSION",
    "MIN_WINDOW_SIZE",
    "MAX_EPISODE_STEPS",
    "AnytradingAction",
    "AnytradingPosition",
    "AnytradingConfig",
    "AnytradingObservation",
    "AnytradingStepResult",
    "AnytradingEpisodeBudgetExceededError",
    "DIXAnytradingEnv",
    "gymnasium_anytrading_env_factory",
)
