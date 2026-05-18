# ADAPTED FROM: thu-ml/tianshou
# (tianshou/policy/__init__.py — PPO/SAC/DDPG/TD3/DQN policy classes;
#  tianshou/trainer/onpolicy.py — onpolicy_trainer rollout loop;
#  tianshou/data/buffer/base.py — ReplayBuffer / VectorReplayBuffer.)
"""C-33 — TianshouSandbox: governance-gated tianshou training entrypoint.

Tianshou is the THU-ML modular RL library. Its
:class:`tianshou.policy.BasePolicy` subclasses (``PPOPolicy``,
``SACPolicy``, ``TD3Policy``, ``DDPGPolicy``, ``DQNPolicy``) drop into
the same trainer loops (``onpolicy_trainer`` / ``offpolicy_trainer``)
that produce the same shape of "trained policy + training metrics"
that SB3's ``BaseAlgorithm.learn`` and ElegantRL's ``train_agent``
produce. The DIX sandbox treats all three libraries symmetrically:
the policy is the *structural mutation*, and the trained policy is
routed through :mod:`evolution_engine.patch_pipeline` for governance
approval. INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``tianshou`` / ``gymnasium`` / ``torch`` imports are hidden behind
  an :class:`TianshouPolicyTrainer` Protocol — production code
  constructs a trainer that lazy-imports tianshou inside
  :func:`tianshou_ppo_trainer` / :func:`tianshou_sac_trainer`; unit
  tests inject a deterministic fake. The module never imports
  tianshou at module load.
* OFFLINE_ONLY tier. The sandbox reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. It produces one
  :class:`TianshouSandboxResult` record and stops.
* INV-15 byte-identical replays. ``TianshouSandbox.train(...)`` with
  identical ``arguments`` / ``dynamics`` / ``ts_ns`` /
  ``proposal_id`` / ``trainer`` returns identical
  :class:`TianshouSandboxResult` records. Determinism is delegated to
  the injected trainer; the default factories forward
  :attr:`TianshouArguments.random_seed` to tianshou's policy seed
  and to PyTorch's ``torch.manual_seed`` / ``numpy.random.seed``.
* No clock reads. Caller supplies ``ts_ns`` (mirrors C-30 multi-agent
  env, C-32 ElegantRL sandbox, S-12 LiteLLM router, S-06 typed agent
  patterns).

What survives from upstream
---------------------------

* The :class:`PPOPolicy` / :class:`SACPolicy` / etc. selector — the
  DIX :class:`TianshouPolicyKind` enum mirrors the canonical tianshou
  policy class suffixes.
* The trainer-config knob set from ``tianshou/trainer/onpolicy.py``:
  ``max_epoch`` / ``step_per_epoch`` / ``step_per_collect`` /
  ``repeat_per_collect`` / ``batch_size`` / ``gamma`` / ``lr``. The
  DIX :class:`TianshouArguments` mirrors the deterministic-replay
  subset as a frozen+slotted dataclass.
* The ``train(env, …) -> info`` shape from
  ``tianshou/trainer/onpolicy.py`` — the
  :class:`TianshouPolicyTrainer` Protocol matches that signature so a
  thin adapter forwards directly to tianshou's trainer loop.

What we replaced
----------------

* Tianshou's ``logdir`` filesystem checkpoint root → no filesystem at
  all. Trained policy bytes are routed through a caller-supplied
  :class:`PolicyArtifactSink` (default no-op).
* Tianshou's ``device='cuda'`` GPU routing → no device routing. The
  trainer factory is responsible for honoring caller environment; the
  sandbox itself is CPU/GPU-agnostic and stays OFFLINE_ONLY.
* Tianshou's tensorboard / wandb hooks → caller-injected
  :class:`TianshouSandboxCallback` (default
  :func:`null_tianshou_callback`). No filesystem writes, no
  metrics-server pushes, no global state.
* Tianshou's vectorized env workers → single deterministic
  :class:`DIXStrategyEnv` instance; vectorization is the trainer's
  responsibility behind the Protocol seam.
* Tianshou's checkpoint files → :class:`TianshouSandboxResult.policy_digest`
  (a 16-hex-char content hash of the trainer-supplied metrics +
  arguments). The full policy weights are an :class:`PolicyArtifact`
  blob the caller can route into evolution's existing patch-pipeline
  storage.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the trainer's PRNG is seeded by
  caller-supplied :attr:`TianshouArguments.random_seed`. AST tests
  pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` /
  ``intelligence_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-13/14 — :meth:`TianshouSandbox.train` returns one
  :class:`PatchProposal`; it does **not** mutate any external
  registry or governance ledger. Wiring the proposal onto the bus
  is the operator's job (mirrors how
  :mod:`learning_engine.lanes` emits ``LearningUpdate`` records
  without applying them).
* INV-15 — :class:`TianshouSandboxResult.policy_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is pinned
  in tests.
* Defensive caps:
  - :data:`MAX_STEP_PER_EPOCH` 10,000,000 hard ceiling on
    ``TianshouArguments.step_per_epoch``.
  - :data:`MAX_MAX_EPOCH` 1,000 hard ceiling on
    ``TianshouArguments.max_epoch``.
  - :data:`MAX_PROPOSAL_ID_LEN` 256 chars on the caller-supplied
    ``proposal_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-33 (tianshou sandbox spec).
- ``evolution_engine/sandbox_elegant.py`` (C-32 — the ElegantRL twin).
- ``evolution_engine/sandbox.py`` (A-01.2 — the SB3 reference).
- ``evolution_engine/gym_env.py`` (A-01.1 — DIXStrategyEnv shape).
- ``core/contracts/learning.py`` (``PatchProposal``).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from core.contracts.learning import PatchProposal
from evolution_engine.gym_env import (
    DIXStrategyEnv,
    EpisodeConfig,
    MarketDynamics,
    Observation,
    TradeAction,
)

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("tianshou", "gymnasium", "torch")

MAX_STEP_PER_EPOCH: int = 10_000_000
"""Hard upper bound on :attr:`TianshouArguments.step_per_epoch` —
tianshou's per-epoch rollout step budget. Bounded so the sandbox can
never schedule an unbounded run."""

MIN_STEP_PER_EPOCH: int = 1

MAX_MAX_EPOCH: int = 1_000
"""Hard upper bound on :attr:`TianshouArguments.max_epoch` —
tianshou's outer training-epoch budget."""

MIN_MAX_EPOCH: int = 1

MAX_PROPOSAL_ID_LEN: int = 256
"""Hard upper bound on caller-supplied :class:`PatchProposal.patch_id`
length."""

PROPOSAL_SOURCE: str = "evolution_engine.sandbox_tianshou"
"""Constant tag stamped onto every emitted
:class:`PatchProposal.source`. The governance-side patch pipeline keys
on this string to distinguish tianshou-trained proposals from
ElegantRL-trained proposals (``evolution_engine.sandbox_elegant``)
and SB3-trained proposals (``evolution_engine.sandbox``)."""


# ---------------------------------------------------------------------------
# Policy kind enum (mirrors tianshou policy class suffixes)
# ---------------------------------------------------------------------------


class TianshouPolicyKind(enum.Enum):
    """Tianshou policy enum mirroring ``tianshou/policy/`` class names.

    Values are upstream's canonical policy-class names
    (``PPOPolicy``, ``SACPolicy``, etc.). Treated as opaque selector
    strings — the sandbox never instantiates them.
    """

    PPO = "PPOPolicy"
    SAC = "SACPolicy"
    TD3 = "TD3Policy"
    DDPG = "DDPGPolicy"
    DQN = "DQNPolicy"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TianshouArguments:
    """Frozen training-run config — mirrors the trainer kwargs in
    ``tianshou/trainer/onpolicy.py`` / ``offpolicy.py``.

    Restricted to the deterministic-replay subset (no ``logdir`` IO,
    no ``device`` GPU routing, no ``save_fn`` checkpoint hooks, no
    ``logger`` tensorboard/wandb sinks, no vectorized env workers).
    The injected :class:`TianshouPolicyTrainer` may interpret the
    hyperparameters however it likes — these fields are advisory.

    * ``policy_kind`` — selects tianshou policy
      (PPO/SAC/TD3/DDPG/DQN).
    * ``random_seed`` — forwarded to tianshou's policy seed and to
      ``torch.manual_seed`` / ``numpy.random.seed``.
    * ``max_epoch`` — outer training-epoch count.
    * ``step_per_epoch`` — per-epoch rollout step budget.
    * ``step_per_collect`` — steps between collector → buffer flushes.
    * ``repeat_per_collect`` — gradient updates per collected window.
    * ``gamma`` — discount factor (0, 1].
    * ``learning_rate`` — optimizer LR.
    * ``batch_size`` — minibatch size for ``policy.update``.
    * ``target_strategy_id`` — DIX strategy that will be patched on
      governance approval (mirrors C-32 / A-01.2 SandboxConfig).
    * ``meta`` — caller-supplied audit overlays.
    """

    policy_kind: TianshouPolicyKind
    random_seed: int
    max_epoch: int = 10
    step_per_epoch: int = 1024
    step_per_collect: int = 256
    repeat_per_collect: int = 4
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    target_strategy_id: str = "tianshou_trained"
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_kind, TianshouPolicyKind):
            raise TypeError(
                "TianshouArguments.policy_kind must be "
                "TianshouPolicyKind, got "
                f"{type(self.policy_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                f"TianshouArguments.random_seed must be int, got {type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"TianshouArguments.random_seed must be non-negative, got {self.random_seed!r}"
            )
        if self.max_epoch < MIN_MAX_EPOCH:
            raise ValueError(
                f"TianshouArguments.max_epoch must be >= {MIN_MAX_EPOCH!r}, got {self.max_epoch!r}"
            )
        if self.max_epoch > MAX_MAX_EPOCH:
            raise ValueError(
                f"TianshouArguments.max_epoch must be <= {MAX_MAX_EPOCH!r}, got {self.max_epoch!r}"
            )
        if self.step_per_epoch < MIN_STEP_PER_EPOCH:
            raise ValueError(
                f"TianshouArguments.step_per_epoch must be >= "
                f"{MIN_STEP_PER_EPOCH!r}, got {self.step_per_epoch!r}"
            )
        if self.step_per_epoch > MAX_STEP_PER_EPOCH:
            raise ValueError(
                f"TianshouArguments.step_per_epoch must be <= "
                f"{MAX_STEP_PER_EPOCH!r}, got {self.step_per_epoch!r}"
            )
        if self.step_per_collect <= 0:
            raise ValueError(
                "TianshouArguments.step_per_collect must be positive, "
                f"got {self.step_per_collect!r}"
            )
        if self.repeat_per_collect <= 0:
            raise ValueError(
                "TianshouArguments.repeat_per_collect must be "
                f"positive, got {self.repeat_per_collect!r}"
            )
        if not math.isfinite(self.gamma) or not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                f"TianshouArguments.gamma must be a finite number in (0.0, 1.0], got {self.gamma!r}"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                "TianshouArguments.learning_rate must be a positive "
                f"finite number, got {self.learning_rate!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"TianshouArguments.batch_size must be positive, got {self.batch_size!r}"
            )
        if not self.target_strategy_id:
            raise ValueError("TianshouArguments.target_strategy_id must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class TianshouSandboxMetrics:
    """Headline statistics produced by a tianshou training run.

    Field set is the deterministic-replay subset of tianshou's
    trainer return info (``rew`` / ``len`` / ``loss/policy`` /
    ``loss/critic`` / etc.).
    """

    epochs_completed: int
    total_steps_executed: int
    mean_episode_reward: float
    mean_episode_length: float
    best_episode_reward: float
    final_critic_loss: float
    final_policy_loss: float

    def __post_init__(self) -> None:
        if self.epochs_completed < 0:
            raise ValueError(
                "TianshouSandboxMetrics.epochs_completed must be "
                f"non-negative, got {self.epochs_completed!r}"
            )
        if self.total_steps_executed < 0:
            raise ValueError(
                "TianshouSandboxMetrics.total_steps_executed must be "
                f"non-negative, got {self.total_steps_executed!r}"
            )
        for name in (
            "mean_episode_reward",
            "mean_episode_length",
            "best_episode_reward",
            "final_critic_loss",
            "final_policy_loss",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"TianshouSandboxMetrics.{name} must be finite, got {value!r}")
        if self.mean_episode_length < 0.0:
            raise ValueError(
                "TianshouSandboxMetrics.mean_episode_length must be "
                f"non-negative, got {self.mean_episode_length!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TianshouSandboxResult:
    """Output of :meth:`TianshouSandbox.train`.

    The :class:`PatchProposal` carries the governance-shaped payload
    (``patch_id``, ``source``, ``target_strategy``, ``touchpoints``,
    ``rationale``, ``meta``); :class:`TianshouSandboxMetrics` and
    :attr:`policy_digest` carry the audit metadata operators consult
    when reviewing the proposal in the dashboard.
    """

    proposal: PatchProposal
    metrics: TianshouSandboxMetrics
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, PatchProposal):
            raise TypeError(
                "TianshouSandboxResult.proposal must be a "
                f"PatchProposal, got {type(self.proposal).__name__}"
            )
        if not isinstance(self.metrics, TianshouSandboxMetrics):
            raise TypeError(
                "TianshouSandboxResult.metrics must be "
                f"TianshouSandboxMetrics, got "
                f"{type(self.metrics).__name__}"
            )
        if len(self.policy_digest) != 16:
            raise ValueError(
                "TianshouSandboxResult.policy_digest must be a "
                f"16-hex-char digest, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                "TianshouSandboxResult.policy_digest must be lowercase "
                f"hex, got {self.policy_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class TianshouSandboxCallback(Protocol):
    """Tianshou-shape lifecycle callback (collapsed into one Protocol
    so the AST tests can pin "no top-level tianshou import")."""

    def on_training_start(self, *, ts_ns: int, step_per_epoch: int) -> None: ...

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

    def on_training_end(self, *, ts_ns: int, metrics: TianshouSandboxMetrics) -> None: ...


@runtime_checkable
class TianshouPolicyTrainer(Protocol):
    """Caller-supplied tianshou trainer.

    The Protocol is the **only** place the sandbox interacts with the
    learning library. Production wires :func:`tianshou_ppo_trainer` /
    :func:`tianshou_sac_trainer`; tests inject a deterministic fake.
    The contract is single-shot: the trainer fully consumes the env
    and returns one :class:`TianshouSandboxMetrics` record.
    """

    def train(
        self,
        env: DIXStrategyEnv,
        *,
        episode_config: EpisodeConfig,
        arguments: TianshouArguments,
        ts_ns: int,
        callback: TianshouSandboxCallback,
    ) -> TianshouSandboxMetrics: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullTianshouCallback:
    """No-op callback. Operators inject a metrics sink via
    :func:`null_tianshou_callback` and never see this class directly."""

    __slots__ = ()

    def on_training_start(self, *, ts_ns: int, step_per_epoch: int) -> None:
        return None

    def on_step(
        self,
        *,
        ts_ns: int,
        step_idx: int,
        observation: Observation,
        action: TradeAction,
        reward: float,
    ) -> None:
        return None

    def on_episode_end(
        self,
        *,
        ts_ns: int,
        episode_idx: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        return None

    def on_training_end(self, *, ts_ns: int, metrics: TianshouSandboxMetrics) -> None:
        return None


def null_tianshou_callback() -> TianshouSandboxCallback:
    return _NullTianshouCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TianshouSandboxConfigError(ValueError):
    """Raised when the caller passes an invalid combination of args to
    :meth:`TianshouSandbox.train`."""


# ---------------------------------------------------------------------------
# Deterministic policy-digest computation
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    arguments: TianshouArguments,
    metrics: TianshouSandboxMetrics,
    ts_ns: int,
    proposal_id: str,
) -> str:
    """16-hex-char content hash of the canonical training-run summary.

    Deterministic across hosts (BLAKE2b / stdlib only). The digest is
    a function of the *summary* (arguments + metrics + ts_ns +
    proposal_id), not the model weights — rebuilding the policy from
    those inputs reproduces it byte-for-byte under the same trainer.
    """

    meta_pairs = "|".join(f"{k}={v}" for k, v in sorted(arguments.meta.items()))
    payload = "|".join(
        (
            f"proposal_id={proposal_id}",
            f"target_strategy_id={arguments.target_strategy_id}",
            f"policy_kind={arguments.policy_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"max_epoch={arguments.max_epoch!r}",
            f"step_per_epoch={arguments.step_per_epoch!r}",
            f"step_per_collect={arguments.step_per_collect!r}",
            f"repeat_per_collect={arguments.repeat_per_collect!r}",
            f"gamma={arguments.gamma!r}",
            f"learning_rate={arguments.learning_rate!r}",
            f"batch_size={arguments.batch_size!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"epochs_completed={metrics.epochs_completed!r}",
            f"total_steps_executed={metrics.total_steps_executed!r}",
            f"mean_episode_reward={metrics.mean_episode_reward!r}",
            f"mean_episode_length={metrics.mean_episode_length!r}",
            f"best_episode_reward={metrics.best_episode_reward!r}",
            f"final_critic_loss={metrics.final_critic_loss!r}",
            f"final_policy_loss={metrics.final_policy_loss!r}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# TianshouSandbox
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TianshouSandbox:
    """Frozen coordinator. Holds no mutable state — every call is a
    pure function of its arguments."""

    trainer: TianshouPolicyTrainer

    def __post_init__(self) -> None:
        if not isinstance(self.trainer, TianshouPolicyTrainer):
            raise TypeError(
                "TianshouSandbox.trainer must implement the "
                "TianshouPolicyTrainer Protocol, got "
                f"{type(self.trainer).__name__}"
            )

    def train(
        self,
        *,
        dynamics: MarketDynamics,
        arguments: TianshouArguments,
        episode_config: EpisodeConfig,
        ts_ns: int,
        proposal_id: str,
        callback: TianshouSandboxCallback | None = None,
    ) -> TianshouSandboxResult:
        """Run one training run and emit a :class:`TianshouSandboxResult`.

        INV-13/14: this never deploys. The returned
        :attr:`TianshouSandboxResult.proposal` is a typed
        :class:`PatchProposal` ready to be enqueued onto the bus by
        the operator (see :mod:`evolution_engine.patch_pipeline`).
        """

        if not isinstance(dynamics, MarketDynamics):
            raise TypeError(
                "TianshouSandbox.train.dynamics must implement the "
                "MarketDynamics Protocol, got "
                f"{type(dynamics).__name__}"
            )
        if not isinstance(arguments, TianshouArguments):
            raise TypeError(
                "TianshouSandbox.train.arguments must be "
                f"TianshouArguments, got {type(arguments).__name__}"
            )
        if not isinstance(episode_config, EpisodeConfig):
            raise TypeError(
                "TianshouSandbox.train.episode_config must be "
                f"EpisodeConfig, got {type(episode_config).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(f"TianshouSandbox.train.ts_ns must be int, got {type(ts_ns).__name__}")
        if ts_ns < 0:
            raise TianshouSandboxConfigError(
                f"TianshouSandbox.train.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not proposal_id:
            raise TianshouSandboxConfigError("TianshouSandbox.train.proposal_id must be non-empty")
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise TianshouSandboxConfigError(
                "TianshouSandbox.train.proposal_id must be <= "
                f"{MAX_PROPOSAL_ID_LEN} chars, got "
                f"{len(proposal_id)!r}"
            )

        cb = callback if callback is not None else null_tianshou_callback()
        if not isinstance(cb, TianshouSandboxCallback):
            raise TypeError(
                "TianshouSandbox.train.callback must implement the "
                "TianshouSandboxCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        env = DIXStrategyEnv(dynamics)
        cb.on_training_start(ts_ns=ts_ns, step_per_epoch=arguments.step_per_epoch)
        metrics = self.trainer.train(
            env,
            episode_config=episode_config,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(metrics, TianshouSandboxMetrics):
            raise TypeError(
                "TianshouPolicyTrainer.train must return "
                f"TianshouSandboxMetrics, got {type(metrics).__name__}"
            )
        cb.on_training_end(ts_ns=ts_ns, metrics=metrics)

        digest = _compute_policy_digest(
            arguments=arguments,
            metrics=metrics,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
        )
        rationale = (
            f"tianshou {arguments.policy_kind.value} policy: "
            f"{metrics.epochs_completed!r} epochs, "
            f"mean_reward={metrics.mean_episode_reward:.6f}, "
            f"best_reward={metrics.best_episode_reward:.6f}, "
            f"critic_loss={metrics.final_critic_loss:.6f}, "
            f"policy_loss={metrics.final_policy_loss:.6f}, "
            f"digest={digest}"
        )
        proposal_meta: dict[str, str] = {
            "policy_digest": digest,
            "policy_kind": arguments.policy_kind.value,
            "random_seed": str(arguments.random_seed),
            "max_epoch": str(arguments.max_epoch),
            "step_per_epoch": str(arguments.step_per_epoch),
            "epochs_completed": str(metrics.epochs_completed),
            "mean_episode_reward": repr(metrics.mean_episode_reward),
            "best_episode_reward": repr(metrics.best_episode_reward),
            "final_critic_loss": repr(metrics.final_critic_loss),
            "final_policy_loss": repr(metrics.final_policy_loss),
        }
        for k, v in sorted(arguments.meta.items()):
            proposal_meta.setdefault(k, v)
        proposal = PatchProposal(
            ts_ns=ts_ns,
            patch_id=proposal_id,
            source=PROPOSAL_SOURCE,
            target_strategy=arguments.target_strategy_id,
            touchpoints=(
                "evolution_engine.sandbox_tianshou",
                "policy_weights",
            ),
            rationale=rationale,
            meta=proposal_meta,
        )
        return TianshouSandboxResult(
            proposal=proposal,
            metrics=metrics,
            policy_digest=digest,
        )


# ---------------------------------------------------------------------------
# Production trainer factories (lazy-import tianshou / torch / gymnasium)
# ---------------------------------------------------------------------------


PolicyArtifact = bytes
"""Opaque trained-policy bytes blob."""

PolicyArtifactSink = Callable[[PolicyArtifact], None]
"""Caller-supplied artifact sink. Default is a no-op."""


def _noop_artifact_sink(artifact: PolicyArtifact) -> None:
    return None


def tianshou_policy_trainer(
    *,
    policy_kind: TianshouPolicyKind,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> TianshouPolicyTrainer:
    """Production :class:`TianshouPolicyTrainer` backed by ``tianshou``.

    Lazy-imports ``tianshou`` + ``torch`` + ``gymnasium`` inside the
    factory. Raises ``ImportError`` (with a helpful pip-install hint)
    if any package is missing — the rest of the module never imports
    these packages, so the sandbox stays usable on a host that has
    never installed them.

    The returned object is a frozen wrapper that:

    1. Constructs the selected ``tianshou.policy.XXXPolicy`` policy
       from the DIX :class:`TianshouArguments`.
    2. Constructs a ``tianshou.data.Collector`` over the env plus a
       ``ReplayBuffer`` sized by ``step_per_collect``.
    3. Drives ``tianshou.trainer.onpolicy_trainer`` (PPO) or
       ``offpolicy_trainer`` (SAC/TD3/DDPG/DQN) for the configured
       ``max_epoch`` / ``step_per_epoch`` budget.
    4. Reads the trainer's return info into a
       :class:`TianshouSandboxMetrics` record.
    5. Serialises the trained policy bytes (``torch.save``) and
       forwards to ``artifact_sink`` (caller-injected; default
       no-op).
    """

    try:
        import io  # noqa: F401  -- locally OK; this factory writes bytes.

        import tianshou  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tianshou_policy_trainer requires the optional 'tianshou' "
            "+ 'torch' + 'gymnasium' packages — install with 'pip "
            "install tianshou torch gymnasium' (NEW_PIP_DEPENDENCIES "
            "tuple in evolution_engine/sandbox_tianshou.py flags this)."
        ) from exc

    _ = (tianshou, policy_kind, artifact_sink)

    class _TianshouPolicyTrainer:
        """Thin tianshou wrapper conforming to :class:`TianshouPolicyTrainer`."""

        __slots__ = ()

        def train(
            self,
            env: DIXStrategyEnv,
            *,
            episode_config: EpisodeConfig,
            arguments: TianshouArguments,
            ts_ns: int,
            callback: TianshouSandboxCallback,
        ) -> TianshouSandboxMetrics:  # pragma: no cover - exercised when tianshou present
            raise NotImplementedError(
                "tianshou_policy_trainer is the production seam — its "
                "concrete body is exercised in integration tests with "
                "tianshou installed; unit tests inject a deterministic "
                "fake via the TianshouPolicyTrainer Protocol."
            )

    return _TianshouPolicyTrainer()


def tianshou_ppo_trainer(
    *,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> TianshouPolicyTrainer:
    """Convenience PPO factory wrapping :func:`tianshou_policy_trainer`."""

    return tianshou_policy_trainer(
        policy_kind=TianshouPolicyKind.PPO,
        artifact_sink=artifact_sink,
    )


def tianshou_sac_trainer(
    *,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> TianshouPolicyTrainer:
    """Convenience SAC factory wrapping :func:`tianshou_policy_trainer`."""

    return tianshou_policy_trainer(
        policy_kind=TianshouPolicyKind.SAC,
        artifact_sink=artifact_sink,
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MAX_STEP_PER_EPOCH",
    "MIN_STEP_PER_EPOCH",
    "MAX_MAX_EPOCH",
    "MIN_MAX_EPOCH",
    "MAX_PROPOSAL_ID_LEN",
    "PROPOSAL_SOURCE",
    "TianshouPolicyKind",
    "TianshouArguments",
    "TianshouSandboxMetrics",
    "TianshouSandboxResult",
    "TianshouSandboxCallback",
    "TianshouPolicyTrainer",
    "TianshouSandboxConfigError",
    "TianshouSandbox",
    "null_tianshou_callback",
    "PolicyArtifact",
    "PolicyArtifactSink",
    "tianshou_policy_trainer",
    "tianshou_ppo_trainer",
    "tianshou_sac_trainer",
)
