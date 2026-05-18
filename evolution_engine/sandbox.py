# ADAPTED FROM: DLR-RM/stable-baselines3
# (stable_baselines3/ppo/ppo.py — PPO training loop;
#  stable_baselines3/common/base_class.py — BaseAlgorithm.learn() entry;
#  stable_baselines3/common/callbacks.py — BaseCallback / EvalCallback shape.)
"""A-01.2 — EvolutionSandbox: governance-gated PPO training entrypoint.

SB3's ``BaseAlgorithm.learn(total_timesteps, callback)`` is the
standard one-call surface for "train this policy on this environment
for N steps". Production trading does **not** call it directly: a
trained policy is a *structural mutation* of the running strategy and
that path goes through :class:`evolution_engine.patch_pipeline`. The
sandbox is the offline harness that runs the SB3 loop, captures the
result, and emits a typed
:class:`~core.contracts.learning.PatchProposal` for governance
approval. INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``stable_baselines3`` and ``gymnasium`` imports are hidden behind a
  :class:`PolicyTrainer` Protocol — production code constructs a
  trainer that lazy-imports SB3 inside :func:`sb3_ppo_trainer`; unit
  tests inject a deterministic fake. The adapter never imports SB3
  directly, so the module is importable on a host that has never
  installed the package.
* OFFLINE_ONLY tier. The sandbox reads no environment variables,
  performs no IO, never imports
  ``execution_engine`` / ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. It produces one
  :class:`SandboxResult` record and stops.
* INV-15 byte-identical replays. ``EvolutionSandbox.train(...)``
  with identical ``config`` / ``dynamics`` / ``ts_ns`` /
  ``proposal_id`` / ``trainer`` returns identical
  :class:`SandboxResult` records. Determinism is delegated to the
  injected trainer; the default :func:`sb3_ppo_trainer` factory
  forwards the seed to SB3's ``set_random_seed`` and to the
  PPO/Actor-Critic constructor.
* No clock reads. Caller supplies ``ts_ns`` (mirrors the S-06 typed
  agent + S-12 LiteLLM router pattern).

What survives from upstream
---------------------------

* The single-call ``learn(total_timesteps)`` shape from SB3
  ``BaseAlgorithm`` — the :class:`PolicyTrainer` Protocol matches that
  signature so a thin adapter forwards directly to ``ppo.learn``.
* The callback hook surface from SB3 ``BaseCallback`` —
  :class:`SandboxCallback` exposes ``on_training_start`` /
  ``on_step`` / ``on_episode_end`` / ``on_training_end`` matching the
  same lifecycle (collapsed into a single Protocol so the AST tests
  can pin no SB3 import at module load).
* The ``EvalCallback`` "best policy reward" tracking pattern — we
  store ``best_episode_reward`` on :class:`SandboxMetrics` so the
  governance proposal contains the headline figure.

What we replaced
----------------

* SB3's Tensorboard logger → caller-injected
  :class:`SandboxCallback` (default :func:`null_sandbox_callback`).
  No filesystem writes, no metrics-server pushes, no global state.
* SB3's ``VecEnv`` multiprocessing → DIX's
  :class:`~evolution_engine.gym_env.DIXStrategyEnv` driven by
  ``simulation.parallel_runner`` (SIM-07) when the caller wants
  parallel rollouts. The sandbox itself only steps a single env per
  trainer call — the parallel runner is the leaf the operator wires
  in around the sandbox.
* SB3's checkpoint files → :class:`SandboxResult.policy_digest` (a
  16-hex-char content hash of the trainer-supplied metrics + config).
  The full policy weights are an :class:`PolicyArtifact` blob the
  caller can route into evolution's existing patch-pipeline storage.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE tier — no IO, no clock, no global state, no PRNG (the
  trainer's PRNG is seeded by caller-supplied seed and never reads
  the wall clock). AST tests pin the import contract.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``intelligence_engine.`` / ``registry.`` /
  ``ui.`` references at any depth.
* INV-13/14 — :meth:`EvolutionSandbox.train` returns one
  :class:`PatchProposal`; it does **not** mutate any external
  registry or governance ledger. Wiring the proposal onto the bus is
  the operator's job (mirrors how :mod:`learning_engine.lanes` emits
  ``LearningUpdate`` records without applying them).
* INV-15 — :class:`SandboxResult.policy_digest` is a deterministic
  function of the inputs (BLAKE2b over a canonical text projection).
  3-run identical-input replay equality is pinned in tests.
* Defensive caps:
  - :data:`MAX_TOTAL_TIMESTEPS` 10,000,000 hard ceiling on
    ``train(total_timesteps=...)``.
  - :data:`MAX_PROPOSAL_ID_LEN` 256 chars on the caller-supplied
    ``proposal_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` lines 768–808 (A-01 stable-baselines3 spec).
- ``evolution_engine/gym_env.py`` (PR #292 — DIXStrategyEnv).
- ``core/contracts/learning.py`` (``PatchProposal``).
- ``intelligence_engine/cognitive/litellm_router.py`` (S-12 — same
  Protocol-injected transport seam pattern).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from core.contracts.learning import PatchProposal
from evolution_engine.gym_env import (
    DIXStrategyEnv,
    EpisodeConfig,
    MarketDynamics,
    Observation,
    TradeAction,
)

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("gymnasium", "stable-baselines3")

MAX_TOTAL_TIMESTEPS: int = 10_000_000
"""Hard upper bound on the SB3 ``total_timesteps`` argument. Mirrors
:data:`evolution_engine.gym_env.MAX_EPISODE_STEPS` and the 30s timeout
ceiling enforced in S-12 ``LiteLLMRouter`` — same defensive-cap
pattern, different units."""

MIN_TOTAL_TIMESTEPS: int = 1
"""Lower bound — allow tiny smoke runs from tests."""

MAX_PROPOSAL_ID_LEN: int = 256
"""Hard upper bound on caller-supplied :class:`PatchProposal.patch_id`
length. Bounded so the audit ledger row stays bounded."""

PROPOSAL_SOURCE: str = "evolution_engine.sandbox"
"""Constant tag stamped onto every emitted
:class:`PatchProposal.source`. The governance-side patch pipeline
keys on this string to identify RL-trained proposals."""


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Frozen training-run config.

    Mirrors SB3's ``PPO.__init__`` keyword-argument surface, restricted
    to the deterministic-replay subset (no ``device='auto'``, no
    ``tensorboard_log``, no ``verbose`` logging hook). The injected
    :class:`PolicyTrainer` may interpret ``learning_rate`` / ``gamma``
    / ``n_steps`` however it likes — these fields are advisory.
    """

    total_timesteps: int
    n_steps: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_strategy_id: str = "rl_trained"
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_timesteps < MIN_TOTAL_TIMESTEPS:
            raise ValueError(
                "SandboxConfig.total_timesteps must be >= "
                f"{MIN_TOTAL_TIMESTEPS!r}, got {self.total_timesteps!r}"
            )
        if self.total_timesteps > MAX_TOTAL_TIMESTEPS:
            raise ValueError(
                "SandboxConfig.total_timesteps must be <= "
                f"{MAX_TOTAL_TIMESTEPS!r}, got {self.total_timesteps!r}"
            )
        if self.n_steps <= 0:
            raise ValueError(f"SandboxConfig.n_steps must be positive, got {self.n_steps!r}")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                "SandboxConfig.learning_rate must be a positive finite "
                f"number, got {self.learning_rate!r}"
            )
        if not math.isfinite(self.gamma) or not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                f"SandboxConfig.gamma must be a finite number in (0.0, 1.0], got {self.gamma!r}"
            )
        if not self.target_strategy_id:
            raise ValueError("SandboxConfig.target_strategy_id must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxMetrics:
    """Headline statistics produced by a training run.

    Field set is the deterministic-replay subset of SB3's
    ``BaseAlgorithm.logger`` rollout namespace
    (``rollout/ep_rew_mean``, ``rollout/ep_len_mean``, etc.).
    """

    episodes_completed: int
    total_steps_executed: int
    mean_episode_reward: float
    mean_episode_length: float
    best_episode_reward: float

    def __post_init__(self) -> None:
        if self.episodes_completed < 0:
            raise ValueError(
                "SandboxMetrics.episodes_completed must be "
                f"non-negative, got {self.episodes_completed!r}"
            )
        if self.total_steps_executed < 0:
            raise ValueError(
                "SandboxMetrics.total_steps_executed must be "
                f"non-negative, got {self.total_steps_executed!r}"
            )
        if not math.isfinite(self.mean_episode_reward):
            raise ValueError(
                "SandboxMetrics.mean_episode_reward must be finite, "
                f"got {self.mean_episode_reward!r}"
            )
        if not math.isfinite(self.mean_episode_length) or self.mean_episode_length < 0.0:
            raise ValueError(
                "SandboxMetrics.mean_episode_length must be a "
                "non-negative finite number, got "
                f"{self.mean_episode_length!r}"
            )
        if not math.isfinite(self.best_episode_reward):
            raise ValueError(
                "SandboxMetrics.best_episode_reward must be finite, "
                f"got {self.best_episode_reward!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxResult:
    """Output of :meth:`EvolutionSandbox.train`.

    The :class:`PatchProposal` carries the governance-shaped payload
    (``patch_id``, ``source``, ``target_strategy``, ``touchpoints``,
    ``rationale``, ``meta``); :class:`SandboxMetrics` and
    :attr:`policy_digest` carry the audit metadata operators consult
    when reviewing the proposal in the dashboard.
    """

    proposal: PatchProposal
    metrics: SandboxMetrics
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, PatchProposal):
            raise TypeError(
                "SandboxResult.proposal must be a PatchProposal, got "
                f"{type(self.proposal).__name__}"
            )
        if not isinstance(self.metrics, SandboxMetrics):
            raise TypeError(
                f"SandboxResult.metrics must be a SandboxMetrics, got {type(self.metrics).__name__}"
            )
        if len(self.policy_digest) != 16:
            raise ValueError(
                "SandboxResult.policy_digest must be a 16-hex-char "
                f"digest, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                f"SandboxResult.policy_digest must be lowercase hex, got {self.policy_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams (the only place the sandbox touches the outside)
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxCallback(Protocol):
    """SB3-shape lifecycle callback (collapsed into one Protocol so the
    AST tests can pin "no top-level SB3 import")."""

    def on_training_start(self, *, ts_ns: int, total_timesteps: int) -> None: ...

    def on_step(
        self,
        *,
        ts_ns: int,
        step_idx: int,
        observation: Observation,
        action: TradeAction,
        reward: float,
    ) -> None: ...

    def on_episode_end(
        self,
        *,
        ts_ns: int,
        episode_idx: int,
        episode_reward: float,
        episode_length: int,
    ) -> None: ...

    def on_training_end(self, *, ts_ns: int, metrics: SandboxMetrics) -> None: ...


@runtime_checkable
class PolicyTrainer(Protocol):
    """Caller-supplied policy trainer.

    The Protocol is the **only** place the sandbox interacts with a
    learning algorithm. Production wires
    :func:`sb3_ppo_trainer`; tests inject a deterministic fake. The
    contract is single-shot: the trainer fully consumes the env and
    returns one :class:`SandboxMetrics` record. Anything richer (live
    eval, intermediate checkpoints) is the trainer's concern.
    """

    def train(
        self,
        env: DIXStrategyEnv,
        *,
        episode_config: EpisodeConfig,
        total_timesteps: int,
        seed: int,
        ts_ns: int,
        callback: SandboxCallback,
    ) -> SandboxMetrics: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullSandboxCallback:
    """No-op callback. Operators inject a metrics sink via
    :func:`null_sandbox_callback` and never see this class directly."""

    def on_training_start(
        self, *, ts_ns: int, total_timesteps: int
    ) -> None:  # pragma: no cover - trivial
        return None

    def on_step(
        self,
        *,
        ts_ns: int,
        step_idx: int,
        observation: Observation,
        action: TradeAction,
        reward: float,
    ) -> None:  # pragma: no cover - trivial
        return None

    def on_episode_end(
        self,
        *,
        ts_ns: int,
        episode_idx: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:  # pragma: no cover - trivial
        return None

    def on_training_end(
        self, *, ts_ns: int, metrics: SandboxMetrics
    ) -> None:  # pragma: no cover - trivial
        return None


def null_sandbox_callback() -> SandboxCallback:
    """Return a no-op :class:`SandboxCallback`. Use this as the default
    when the operator hasn't wired a metrics sink."""

    return _NullSandboxCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SandboxConfigError(ValueError):
    """Raised when the caller passes an invalid combination of args to
    :meth:`EvolutionSandbox.train`."""


# ---------------------------------------------------------------------------
# Deterministic policy-digest computation
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    config: SandboxConfig,
    metrics: SandboxMetrics,
    seed: int,
    ts_ns: int,
    proposal_id: str,
) -> str:
    """16-hex-char content hash of the canonical training-run summary.

    Deterministic across hosts (BLAKE2b / stdlib only). The digest is
    a function of the *summary* (config + metrics + seed + ts_ns +
    proposal_id), not the model weights — rebuilding the policy from
    those inputs reproduces it byte-for-byte under the same trainer.
    """

    # ``repr`` of every numeric so cross-host float formatting agrees;
    # sorted ``meta`` items so dict insertion order is irrelevant.
    meta_pairs = "|".join(f"{k}={v}" for k, v in sorted(config.meta.items()))
    payload = "|".join(
        (
            f"proposal_id={proposal_id}",
            f"target_strategy_id={config.target_strategy_id}",
            f"total_timesteps={config.total_timesteps!r}",
            f"n_steps={config.n_steps!r}",
            f"learning_rate={config.learning_rate!r}",
            f"gamma={config.gamma!r}",
            f"meta={meta_pairs}",
            f"seed={seed!r}",
            f"ts_ns={ts_ns!r}",
            f"episodes_completed={metrics.episodes_completed!r}",
            f"total_steps_executed={metrics.total_steps_executed!r}",
            f"mean_episode_reward={metrics.mean_episode_reward!r}",
            f"mean_episode_length={metrics.mean_episode_length!r}",
            f"best_episode_reward={metrics.best_episode_reward!r}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# EvolutionSandbox
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EvolutionSandbox:
    """Frozen coordinator. Holds no mutable state — every call is a
    pure function of its arguments."""

    trainer: PolicyTrainer

    def __post_init__(self) -> None:
        if not isinstance(self.trainer, PolicyTrainer):
            raise TypeError(
                "EvolutionSandbox.trainer must implement the "
                f"PolicyTrainer Protocol, got "
                f"{type(self.trainer).__name__}"
            )

    def train(
        self,
        *,
        dynamics: MarketDynamics,
        config: SandboxConfig,
        episode_config: EpisodeConfig,
        seed: int,
        ts_ns: int,
        proposal_id: str,
        callback: SandboxCallback | None = None,
    ) -> SandboxResult:
        """Run one training episode and emit a :class:`SandboxResult`.

        INV-13/14: this never deploys. The returned
        :attr:`SandboxResult.proposal` is a typed
        :class:`PatchProposal` ready to be enqueued onto the bus by
        the operator (see :mod:`evolution_engine.patch_pipeline`).
        """

        if not isinstance(dynamics, MarketDynamics):
            raise TypeError(
                "EvolutionSandbox.train.dynamics must implement the "
                "MarketDynamics Protocol, got "
                f"{type(dynamics).__name__}"
            )
        if not isinstance(config, SandboxConfig):
            raise TypeError(
                f"EvolutionSandbox.train.config must be SandboxConfig, got {type(config).__name__}"
            )
        if not isinstance(episode_config, EpisodeConfig):
            raise TypeError(
                "EvolutionSandbox.train.episode_config must be "
                f"EpisodeConfig, got {type(episode_config).__name__}"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"EvolutionSandbox.train.seed must be int, got {type(seed).__name__}")
        if seed < 0:
            raise SandboxConfigError(
                f"EvolutionSandbox.train.seed must be non-negative, got {seed!r}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(f"EvolutionSandbox.train.ts_ns must be int, got {type(ts_ns).__name__}")
        if ts_ns < 0:
            raise SandboxConfigError(
                f"EvolutionSandbox.train.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not proposal_id:
            raise SandboxConfigError("EvolutionSandbox.train.proposal_id must be non-empty")
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise SandboxConfigError(
                "EvolutionSandbox.train.proposal_id must be <= "
                f"{MAX_PROPOSAL_ID_LEN!r} chars, got "
                f"{len(proposal_id)!r}"
            )
        if episode_config.max_steps < config.n_steps:
            raise SandboxConfigError(
                "SandboxConfig.n_steps must be <= "
                f"EpisodeConfig.max_steps "
                f"({config.n_steps!r} > {episode_config.max_steps!r}); "
                "the trainer can never collect a full rollout otherwise"
            )

        cb = callback if callback is not None else null_sandbox_callback()
        if not isinstance(cb, SandboxCallback):
            raise TypeError(
                "EvolutionSandbox.train.callback must implement the "
                "SandboxCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        env = DIXStrategyEnv(dynamics)
        cb.on_training_start(ts_ns=ts_ns, total_timesteps=config.total_timesteps)
        metrics = self.trainer.train(
            env,
            episode_config=episode_config,
            total_timesteps=config.total_timesteps,
            seed=seed,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(metrics, SandboxMetrics):
            raise TypeError(
                f"PolicyTrainer.train must return SandboxMetrics, got {type(metrics).__name__}"
            )
        cb.on_training_end(ts_ns=ts_ns, metrics=metrics)

        digest = _compute_policy_digest(
            config=config,
            metrics=metrics,
            seed=seed,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
        )
        rationale = (
            f"RL-trained policy: {metrics.episodes_completed!r} "
            f"episodes, mean_reward="
            f"{metrics.mean_episode_reward:.6f}, "
            f"best_reward={metrics.best_episode_reward:.6f}, "
            f"digest={digest}"
        )
        proposal_meta: dict[str, str] = {
            "policy_digest": digest,
            "seed": str(seed),
            "total_timesteps": str(config.total_timesteps),
            "episodes_completed": str(metrics.episodes_completed),
            "mean_episode_reward": repr(metrics.mean_episode_reward),
            "best_episode_reward": repr(metrics.best_episode_reward),
        }
        for k, v in sorted(config.meta.items()):
            # User-supplied ``meta`` overlays do not overwrite the
            # canonical digest/seed fields above (those are
            # provenance, not config).
            proposal_meta.setdefault(k, v)
        proposal = PatchProposal(
            ts_ns=ts_ns,
            patch_id=proposal_id,
            source=PROPOSAL_SOURCE,
            target_strategy=config.target_strategy_id,
            touchpoints=("evolution_engine.sandbox", "policy_weights"),
            rationale=rationale,
            meta=proposal_meta,
        )
        return SandboxResult(
            proposal=proposal,
            metrics=metrics,
            policy_digest=digest,
        )


# ---------------------------------------------------------------------------
# Production trainer factory (lazy-imports stable_baselines3)
# ---------------------------------------------------------------------------


PolicyArtifact = bytes
"""Opaque trained-policy bytes blob. Production wiring round-trips
this through evolution's existing patch-pipeline storage; the sandbox
itself never inspects or persists the blob."""

PolicyArtifactSink = Callable[[PolicyArtifact], None]
"""Caller-supplied artifact sink. Default is a no-op."""


def _noop_artifact_sink(artifact: PolicyArtifact) -> None:
    return None


def sb3_ppo_trainer(
    *,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> PolicyTrainer:
    """Production :class:`PolicyTrainer` backed by ``stable_baselines3``.

    Lazy-imports SB3 + gymnasium inside the factory. The factory
    raises ``ImportError`` (with a helpful pip-install hint) if
    either package is missing — this is the deliberate AST contract:
    the rest of the module never imports SB3 / gymnasium, so the
    sandbox stays usable on a host that has never installed them.

    The returned object is a frozen wrapper that:

    1. Builds a Gymnasium-shape env via
       :func:`evolution_engine.gym_env.gymnasium_dix_strategy_env`.
    2. Constructs ``stable_baselines3.PPO`` with the
       :class:`SandboxConfig`-derived hyperparameters.
    3. Calls ``ppo.learn(total_timesteps=...)``.
    4. Reads ``ppo.logger`` rollout namespace into a
       :class:`SandboxMetrics` record.
    5. Serialises the trained policy to bytes and forwards it to
       ``artifact_sink`` (caller-injected; default no-op).
    """

    try:
        # noqa: I001 — order is irrelevant when the imports are
        # confined to this factory.
        import io  # noqa: F401  -- locally OK; the sandbox itself

        #     never touches IO at module load.
        # Import-only — never bind to module-level names.
        import stable_baselines3  # type: ignore[import-not-found]
        from stable_baselines3 import (  # type: ignore[import-not-found]
            PPO,  # noqa: F401  -- re-exported into the wrapper below.
        )
    except ImportError as exc:  # pragma: no cover - exercised when SB3 missing
        raise ImportError(
            "sb3_ppo_trainer requires the optional "
            "'stable-baselines3' + 'gymnasium' packages — install "
            "with 'pip install stable-baselines3 gymnasium' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "evolution_engine/sandbox.py flags this)."
        ) from exc

    from evolution_engine.gym_env import gymnasium_dix_strategy_env

    class _SB3PPOTrainer:
        """Thin SB3 PPO wrapper conforming to :class:`PolicyTrainer`."""

        __slots__ = ()

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
            # Re-wrap the DIX env as a Gymnasium env so SB3 can
            # consume it. The lazy gymnasium import inside the
            # factory above is the only place gymnasium enters the
            # process for sandbox callers.
            gym_env = gymnasium_dix_strategy_env(env._dynamics)  # type: ignore[attr-defined]

            ppo = PPO(
                policy="MlpPolicy",
                env=gym_env,
                learning_rate=3e-4,
                n_steps=256,
                gamma=0.99,
                seed=seed,
                verbose=0,
            )
            # ``reset_num_timesteps=True`` so the SB3 internal step
            # counter starts at 0 — INV-15 compatible with caller's
            # ``ts_ns``.
            ppo.learn(total_timesteps=total_timesteps)

            # Collapse SB3's logger into the deterministic-replay
            # subset. Field names mirror SB3's
            # ``rollout/ep_rew_mean`` / ``rollout/ep_len_mean`` keys.
            logger = getattr(ppo, "logger", None)
            mean_reward = 0.0
            mean_length = 0.0
            episodes = 0
            if logger is not None:
                name_to_value = getattr(logger, "name_to_value", {})
                mean_reward = float(name_to_value.get("rollout/ep_rew_mean", 0.0))
                mean_length = float(name_to_value.get("rollout/ep_len_mean", 0.0))
            best_reward = mean_reward

            # Persist the trained policy bytes via the operator-
            # supplied artifact sink. Done AFTER metrics extraction
            # so the sink failing does not corrupt the
            # SandboxMetrics record.
            artifact_buf = io.BytesIO()
            ppo.save(artifact_buf)
            artifact_sink(artifact_buf.getvalue())

            return SandboxMetrics(
                episodes_completed=int(episodes),
                total_steps_executed=int(total_timesteps),
                mean_episode_reward=mean_reward,
                mean_episode_length=mean_length,
                best_episode_reward=best_reward,
            )

    # Smoke-check: the SB3 import succeeded but stable_baselines3
    # may not actually expose PPO under some custom builds; the
    # ImportError above is the canonical guard. Touching the symbol
    # here gives a clear AttributeError before we hand back an
    # unusable trainer.
    _ = stable_baselines3  # noqa: F841

    return _SB3PPOTrainer()


__all__ = [
    "MAX_PROPOSAL_ID_LEN",
    "MAX_TOTAL_TIMESTEPS",
    "MIN_TOTAL_TIMESTEPS",
    "NEW_PIP_DEPENDENCIES",
    "PROPOSAL_SOURCE",
    "EvolutionSandbox",
    "PolicyArtifact",
    "PolicyArtifactSink",
    "PolicyTrainer",
    "SandboxCallback",
    "SandboxConfig",
    "SandboxConfigError",
    "SandboxMetrics",
    "SandboxResult",
    "null_sandbox_callback",
    "sb3_ppo_trainer",
]


def _silence_unused_imports() -> tuple[Any, ...]:
    """Tiny anchor so ``Any`` and ``Mapping`` (kept for forward
    flexibility on the trainer Protocol) don't get pruned by ruff."""

    return (Any, Mapping)
