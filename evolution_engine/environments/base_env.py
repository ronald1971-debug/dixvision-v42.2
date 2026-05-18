# ADAPTED FROM: Farama-Foundation/Gymnasium
# (gymnasium/core.py — Env base class & step/reset/render interface;
#  gymnasium/spaces/space.py — Space base class & contains/sample contract.)
"""C-31 — DIXBaseEnv: canonical Gymnasium-shape base for every DIX env.

`Gymnasium` defines the de-facto RL environment contract that
SB3 / CleanRL / TorchRL / ElegantRL / Tianshou / Sample Factory
all consume:

* ``Env.reset(*, seed=None, options=None) -> (obs, info)``
* ``Env.step(action) -> (obs, reward, terminated, truncated, info)``
* ``Env.render() -> Any | None``
* ``Env.close() -> None``
* ``observation_space`` and ``action_space`` attributes.

Every DIX environment (C-29 anytrading, C-30 multiagent, C-32 finrl,
C-33+ future envs) must inherit from :class:`DIXBaseEnv` so the
training stack can plug them in interchangeably.

What this module is
-------------------

* Pure-Python abstract base + value objects + lazy factory. No
  ``gymnasium`` / ``gym`` / ``numpy`` import at module top-level.
  The optional :func:`gymnasium_dix_base_env_factory` lazy-imports
  ``gymnasium`` and returns a real ``gym.Env`` subclass wrapping
  any :class:`DIXBaseEnv`, so the module is importable on a host
  that has never installed gymnasium.
* OFFLINE_ONLY tier: `evolution_engine` is an OFFLINE engine. The
  base env is wall-clock-free, reads no environment variables,
  performs no IO. Subclasses inherit the same constraints.
* INV-15 byte-identical replays. Subclasses fold the caller-supplied
  ``seed`` into a stdlib-only :func:`hashlib` content-hash of every
  observation so two replays of the same episode produce
  byte-identical observations.

What survives from upstream Gymnasium
-------------------------------------

* The exact method signatures:
  ``reset(*, seed, options)`` returns ``(observation, info)``;
  ``step(action)`` returns ``(observation, reward, terminated,
  truncated, info)``.
* The ``observation_space`` / ``action_space`` attribute contract —
  subclasses must expose both, even if as the lightweight
  :class:`DIXBoxSpace` / :class:`DIXDiscreteSpace` stdlib shadows
  defined here.
* The Gymnasium ≥ 0.26 ``terminated`` / ``truncated`` split (no
  legacy 4-tuple ``done`` flag).

What we replaced
----------------

* ``numpy`` arrays for observations → ``tuple[float, ...]``. The
  DIX RL stack accepts tuples; staying stdlib means subclasses can
  load on hosts without numpy.
* Gymnasium's ``np.random.Generator`` PRNG plumbing → an explicit
  caller-supplied seed folded into a stdlib :func:`hashlib`
  state hash. INV-15 forbids hidden PRNG state.
* Gymnasium's ``spaces.Box`` / ``spaces.Discrete`` →
  :class:`DIXBoxSpace` and :class:`DIXDiscreteSpace` stdlib shadows
  with the same ``contains(x) -> bool`` and ``sample() -> Any``
  contract. The lazy factory swaps these for real Gymnasium spaces
  on environments that need real Gymnasium compatibility.
* Gymnasium's ``render_mode`` plumbing → strict ``render_mode=None``
  only. Rendering is a no-op; benchmarks plot externally.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE_ONLY tier. No IO, no clock, no PRNG, no global state.
* No engine cross-imports. The base env never reads ``registry/``
  / ``system_engine`` / ``governance_engine`` / ``execution_engine``.
* INV-15 byte-identical replay. Subclasses must fold the seed into
  a stdlib :func:`hashlib` content-hash.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-31 (gymnasium base env spec).
- ``evolution_engine/environments/anytrading_env.py`` (C-29 — the
  first concrete subclass-compatible env).
- ``evolution_engine/environments/multiagent_env.py`` (C-30 — the
  multi-agent companion env).
- ``evolution_engine/gym_env.py`` (A-01.1 — the historical primary
  DIX env that is also Gymnasium-shape).
"""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import math
from typing import Any, Final

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("gymnasium",)
"""``gymnasium`` is needed only by the optional
:func:`gymnasium_dix_base_env_factory`. The base class and stdlib
space shadows are pure-stdlib."""

DIX_BASE_ENV_VERSION: Final[str] = "c-31-base-env-1"
"""Version tag woven into the canonical content-hash of every base
observation so a replay run can verify the env-shape is unchanged."""

MAX_EPISODE_STEPS: Final[int] = 1_000_000
"""Hard upper bound on episode length — subclasses should raise
:class:`DIXBaseEpisodeBudgetExceededError` once the caller pumps
more than this many ``step`` calls without a reset."""


class DIXBaseEpisodeBudgetExceededError(RuntimeError):
    """Raised by subclasses when the caller pumps more than
    :data:`MAX_EPISODE_STEPS` ``step`` calls without resetting."""


class DIXBaseEnvNotResetError(RuntimeError):
    """Raised when :meth:`DIXBaseEnv.step` is called before any
    :meth:`DIXBaseEnv.reset` call. Mirrors upstream Gymnasium's
    ``ResetNeeded`` exception."""


@dataclasses.dataclass(frozen=True, slots=True)
class DIXBoxSpace:
    """Pure-stdlib shadow of ``gymnasium.spaces.Box``.

    Subclasses use this to declare a bounded continuous-shape
    observation or action space without importing Gymnasium.

    * ``low`` / ``high`` — tuple bounds, must be finite.
    * ``shape`` — tuple of positive integers, length matches the
      observation tuple length.
    """

    low: float
    high: float
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.low):
            raise ValueError(f"DIXBoxSpace.low must be finite, got {self.low!r}")
        if not math.isfinite(self.high):
            raise ValueError(f"DIXBoxSpace.high must be finite, got {self.high!r}")
        if self.low >= self.high:
            raise ValueError(f"DIXBoxSpace.low must be < high, got {self.low!r} >= {self.high!r}")
        if not self.shape:
            raise ValueError("DIXBoxSpace.shape must be non-empty")
        for idx, dim in enumerate(self.shape):
            if not isinstance(dim, int):
                raise TypeError(f"DIXBoxSpace.shape[{idx}] must be int, got {type(dim).__name__}")
            if dim <= 0:
                raise ValueError(f"DIXBoxSpace.shape[{idx}] must be positive, got {dim!r}")

    def contains(self, value: tuple[float, ...]) -> bool:
        if not isinstance(value, tuple):
            return False
        if len(value) != self.shape[0]:
            return False
        for v in value:
            if not isinstance(v, (int, float)):
                return False
            if not math.isfinite(float(v)):
                return False
            if not (self.low <= float(v) <= self.high):
                return False
        return True

    def sample(self, *, seed: int = 0) -> tuple[float, ...]:
        """Deterministic seeded "sample" — returns ``low`` repeated
        ``shape[0]`` times offset by a seed-derived fraction of
        ``high - low``. Pure stdlib; no PRNG.
        """

        digest = hashlib.blake2b(
            f"sample|{DIX_BASE_ENV_VERSION}|{seed}|{self.low}|{self.high}|{self.shape}".encode(),
            digest_size=8,
        ).digest()
        fraction = (int.from_bytes(digest, "big") % 10_001) / 10_000.0
        value = self.low + fraction * (self.high - self.low)
        return tuple(value for _ in range(self.shape[0]))


@dataclasses.dataclass(frozen=True, slots=True)
class DIXDiscreteSpace:
    """Pure-stdlib shadow of ``gymnasium.spaces.Discrete``.

    Subclasses use this to declare a discrete action space without
    importing Gymnasium. ``n`` is the number of possible actions
    (action indices are 0..n-1).
    """

    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.n, int):
            raise TypeError(f"DIXDiscreteSpace.n must be int, got {type(self.n).__name__}")
        if self.n <= 0:
            raise ValueError(f"DIXDiscreteSpace.n must be positive, got {self.n!r}")

    def contains(self, value: int) -> bool:
        if not isinstance(value, int):
            return False
        if isinstance(value, bool):
            return False
        return 0 <= value < self.n

    def sample(self, *, seed: int = 0) -> int:
        digest = hashlib.blake2b(
            f"sample|{DIX_BASE_ENV_VERSION}|{seed}|{self.n}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") % self.n


@dataclasses.dataclass(frozen=True, slots=True)
class DIXBaseObservation:
    """Canonical envelope every DIX subclass observation must use.

    Subclasses populate ``payload`` with their typed observation
    value object (e.g. ``AnytradingObservation``) and the
    ``state_hash`` is computed by :meth:`DIXBaseEnv._build_state_hash`
    over ``(seed, step_idx, payload_repr)`` so every subclass shares
    the same INV-15 byte-identical replay guarantee.
    """

    step_idx: int
    payload: Any
    state_hash: str

    def __post_init__(self) -> None:
        if self.step_idx < 0:
            raise ValueError(f"DIXBaseObservation.step_idx must be >= 0, got {self.step_idx!r}")
        if len(self.state_hash) != 16:
            raise ValueError(
                f"DIXBaseObservation.state_hash must be 16 hex chars, got {self.state_hash!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class DIXBaseStepResult:
    """Canonical 5-tuple every DIX subclass step must return.

    Mirrors Gymnasium ≥ 0.26 ``Env.step`` return shape but as a
    structured record so the info dict layout is pinned.
    """

    observation: DIXBaseObservation
    reward: float
    terminated: bool
    truncated: bool
    info: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError(f"DIXBaseStepResult.reward must be finite, got {self.reward!r}")

    def info_dict(self) -> dict[str, Any]:
        return dict(self.info)


class DIXBaseEnv(abc.ABC):
    """Abstract base class every DIX environment must inherit from.

    Subclasses must:

    1. Set ``observation_space`` and ``action_space`` class or
       instance attributes (use :class:`DIXBoxSpace` /
       :class:`DIXDiscreteSpace` stdlib shadows by default).
    2. Implement :meth:`_reset_payload` returning the initial
       payload observation.
    3. Implement :meth:`_step_payload` returning
       ``(payload, reward, terminated, truncated, info_dict)``.
    4. Track ``_step_call_count`` and raise
       :class:`DIXBaseEpisodeBudgetExceededError` when it exceeds
       :data:`MAX_EPISODE_STEPS`.

    The base class then:

    * Folds the seed into the BLAKE2b state hash of every
      observation (INV-15 byte-identical replay).
    * Enforces the Gymnasium ≥ 0.26 reset/step shapes.
    * Detects step-before-reset and raises
      :class:`DIXBaseEnvNotResetError`.
    * Detects step-after-termination and raises ``RuntimeError``.
    """

    observation_space: DIXBoxSpace | DIXDiscreteSpace
    action_space: DIXBoxSpace | DIXDiscreteSpace

    def __init__(self) -> None:
        self._seed: int = 0
        self._step_idx: int = 0
        self._step_call_count: int = 0
        self._terminated: bool = False
        self._has_reset: bool = False

    @property
    def step_idx(self) -> int:
        return self._step_idx

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    def _build_state_hash(self, payload_repr: str) -> str:
        digest = hashlib.blake2b(
            "|".join(
                (
                    f"v={DIX_BASE_ENV_VERSION}",
                    f"seed={self._seed}",
                    f"step={self._step_idx}",
                    f"payload={payload_repr}",
                )
            ).encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        return digest

    @abc.abstractmethod
    def _reset_payload(self, *, options: dict[str, Any] | None) -> Any:
        """Subclass hook — return initial observation payload."""

    @abc.abstractmethod
    def _step_payload(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Subclass hook — return
        ``(payload, reward, terminated, truncated, info_dict)``."""

    def _canonical_payload_repr(self, payload: Any) -> str:
        if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
            items = [
                (field.name, getattr(payload, field.name)) for field in dataclasses.fields(payload)
            ]
            return "{" + ",".join(f"{k}={v!r}" for k, v in items) + "}"
        return repr(payload)

    def reset(
        self,
        *,
        seed: int = 0,
        options: dict[str, Any] | None = None,
    ) -> tuple[DIXBaseObservation, dict[str, Any]]:
        if not isinstance(seed, int):
            raise TypeError(f"DIXBaseEnv.reset(seed=...) must be int, got {type(seed).__name__}")
        self._seed = seed
        self._step_idx = 0
        self._step_call_count = 0
        self._terminated = False
        self._has_reset = True
        payload = self._reset_payload(options=options)
        state_hash = self._build_state_hash(self._canonical_payload_repr(payload))
        observation = DIXBaseObservation(
            step_idx=self._step_idx,
            payload=payload,
            state_hash=state_hash,
        )
        return observation, {}

    def step(self, action: Any) -> DIXBaseStepResult:
        if not self._has_reset:
            raise DIXBaseEnvNotResetError(
                "DIXBaseEnv.step called before reset — caller must call "
                "reset() before the first step()"
            )
        if self._terminated:
            raise RuntimeError(
                "DIXBaseEnv.step called after termination — caller must "
                "honour terminated/truncated and call reset() first"
            )

        self._step_call_count += 1
        if self._step_call_count > MAX_EPISODE_STEPS:
            raise DIXBaseEpisodeBudgetExceededError(
                f"DIXBaseEnv exceeded MAX_EPISODE_STEPS={MAX_EPISODE_STEPS!r} without reset"
            )

        self._step_idx += 1
        payload, reward, terminated, truncated, info = self._step_payload(action)
        self._terminated = bool(terminated or truncated)
        state_hash = self._build_state_hash(self._canonical_payload_repr(payload))
        observation = DIXBaseObservation(
            step_idx=self._step_idx,
            payload=payload,
            state_hash=state_hash,
        )
        return DIXBaseStepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=tuple(sorted(info.items())),
        )

    def render(self, mode: str | None = None) -> None:
        return None

    def close(self) -> None:
        return None


def gymnasium_dix_base_env_factory(*, env: DIXBaseEnv) -> Any:
    """Lazy-import factory wrapping any :class:`DIXBaseEnv` as a
    real ``gymnasium.Env`` subclass for callers that need Gymnasium
    compatibility (e.g. SB3's ``VecEnv``).

    Only this function imports ``gymnasium``. Callers that don't
    need Gymnasium spaces use :class:`DIXBaseEnv` directly.
    """

    try:
        import gymnasium as gym  # noqa: PLC0415
        from gymnasium import spaces  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "gymnasium_dix_base_env_factory requires `pip install gymnasium`"
        ) from exc

    base = env

    def _to_gym_space(space: DIXBoxSpace | DIXDiscreteSpace) -> Any:
        if isinstance(space, DIXBoxSpace):
            return spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=float,
            )
        return spaces.Discrete(space.n)

    class _GymWrappedDIXBaseEnv(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": [None]}
        observation_space = _to_gym_space(env.observation_space)
        action_space = _to_gym_space(env.action_space)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            obs, info = base.reset(seed=seed or 0, options=options)
            return obs.payload, info

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            result = base.step(action)
            return (
                result.observation.payload,
                result.reward,
                result.terminated,
                result.truncated,
                result.info_dict(),
            )

        def render(self) -> None:
            return None

        def close(self) -> None:
            return None

    return _GymWrappedDIXBaseEnv()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "DIX_BASE_ENV_VERSION",
    "MAX_EPISODE_STEPS",
    "DIXBaseEpisodeBudgetExceededError",
    "DIXBaseEnvNotResetError",
    "DIXBoxSpace",
    "DIXDiscreteSpace",
    "DIXBaseObservation",
    "DIXBaseStepResult",
    "DIXBaseEnv",
    "gymnasium_dix_base_env_factory",
)
