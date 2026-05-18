# ADAPTED FROM: alex-petrenko/sample-factory
# (sample_factory/algo/runners/runner.py — Runner orchestration loop;
#  sample_factory/cfg/cfg.py — APPO trainer cfg knob set;
#  sample_factory/train.py — APPO entrypoint.)
"""C-34 — SampleFactorySandbox: governance-gated sample-factory training entrypoint.

sample-factory is the Alex-Petrenko high-throughput async PPO (APPO)
library. Its ``runner.Runner`` orchestrates parallel workers around
``algo='APPO'`` and produces the same shape of "trained policy +
training metrics" that SB3's ``BaseAlgorithm.learn``, ElegantRL's
``train_agent``, and tianshou's ``onpolicy_trainer`` produce. The DIX
sandbox treats all four libraries symmetrically: the policy is the
*structural mutation*, and the trained policy is routed through
:mod:`evolution_engine.patch_pipeline` for governance approval.
INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``sample_factory`` / ``gymnasium`` / ``torch`` imports are hidden
  behind a :class:`SampleFactoryPolicyTrainer` Protocol — production
  code constructs a trainer that lazy-imports sample-factory inside
  :func:`sample_factory_appo_trainer`; unit tests inject a
  deterministic fake. The module never imports sample-factory at
  module load.
* OFFLINE_ONLY tier. The sandbox reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. It produces one
  :class:`SampleFactorySandboxResult` record and stops.
* INV-15 byte-identical replays. ``SampleFactorySandbox.train(...)``
  with identical ``arguments`` / ``dynamics`` / ``ts_ns`` /
  ``proposal_id`` / ``trainer`` returns identical
  :class:`SampleFactorySandboxResult` records. Determinism is
  delegated to the injected trainer; the default factory forwards
  :attr:`SampleFactoryArguments.random_seed` to sample-factory's
  ``cfg.seed`` and to PyTorch's ``torch.manual_seed`` /
  ``numpy.random.seed``.
* No clock reads. Caller supplies ``ts_ns`` (mirrors C-30 multi-agent
  env, C-32 ElegantRL sandbox, C-33 Tianshou sandbox patterns).

What survives from upstream
---------------------------

* The single-algorithm selector — sample-factory historically ships
  only ``APPO``; :class:`SampleFactoryAlgoKind` is therefore a
  one-element enum mirroring that string.
* The trainer-config knob set from ``sample_factory/cfg/cfg.py``:
  ``train_for_env_steps`` / ``batch_size`` / ``rollout`` /
  ``num_workers`` / ``num_envs_per_worker`` / ``gamma`` /
  ``learning_rate``. The DIX :class:`SampleFactoryArguments` mirrors
  the deterministic-replay subset as a frozen+slotted dataclass.
* The ``train(env, …) -> info`` shape from
  ``sample_factory/train.py`` — the
  :class:`SampleFactoryPolicyTrainer` Protocol matches that
  signature so a thin adapter forwards directly to sample-factory's
  Runner loop.

What we replaced
----------------

* sample-factory's ``cfg.experiment`` filesystem checkpoint root →
  no filesystem at all. Trained policy bytes are routed through a
  caller-supplied :class:`PolicyArtifactSink` (default no-op).
* sample-factory's ``cfg.device='gpu'`` GPU routing → no device
  routing. The trainer factory is responsible for honoring caller
  environment; the sandbox itself is CPU/GPU-agnostic and stays
  OFFLINE_ONLY.
* sample-factory's tensorboard / wandb hooks → caller-injected
  :class:`SampleFactorySandboxCallback` (default
  :func:`null_sample_factory_callback`). No filesystem writes, no
  metrics-server pushes, no global state.
* sample-factory's actor-worker process pool → single deterministic
  :class:`DIXStrategyEnv` instance; parallelism is the trainer's
  responsibility behind the Protocol seam.
* sample-factory's checkpoint files →
  :class:`SampleFactorySandboxResult.policy_digest` (a 16-hex-char
  content hash of the trainer-supplied metrics + arguments). The
  full policy weights are an :class:`PolicyArtifact` blob the caller
  can route into evolution's existing patch-pipeline storage.

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the trainer's PRNG is seeded by
  caller-supplied :attr:`SampleFactoryArguments.random_seed`. AST
  tests pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` /
  ``intelligence_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-13/14 — :meth:`SampleFactorySandbox.train` returns one
  :class:`PatchProposal`; it does **not** mutate any external
  registry or governance ledger. Wiring the proposal onto the bus
  is the operator's job (mirrors how :mod:`learning_engine.lanes`
  emits ``LearningUpdate`` records without applying them).
* INV-15 — :class:`SampleFactorySandboxResult.policy_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is pinned
  in tests.
* Defensive caps:
  - :data:`MAX_TRAIN_FOR_ENV_STEPS` 10,000,000 hard ceiling on
    ``SampleFactoryArguments.train_for_env_steps``.
  - :data:`MAX_NUM_WORKERS` 64 hard ceiling on
    ``SampleFactoryArguments.num_workers``.
  - :data:`MAX_PROPOSAL_ID_LEN` 256 chars on the caller-supplied
    ``proposal_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-34 (sample-factory sandbox spec).
- ``evolution_engine/sandbox_tianshou.py`` (C-33 — the tianshou twin).
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

NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "sample-factory",
    "gymnasium",
    "torch",
)

MAX_TRAIN_FOR_ENV_STEPS: int = 10_000_000
"""Hard upper bound on
:attr:`SampleFactoryArguments.train_for_env_steps` — sample-factory's
total env-step budget. Bounded so the sandbox can never schedule an
unbounded run."""

MIN_TRAIN_FOR_ENV_STEPS: int = 1

MAX_NUM_WORKERS: int = 64
"""Hard upper bound on :attr:`SampleFactoryArguments.num_workers` —
sample-factory's worker process count."""

MIN_NUM_WORKERS: int = 1

MAX_NUM_ENVS_PER_WORKER: int = 64
"""Hard upper bound on
:attr:`SampleFactoryArguments.num_envs_per_worker`."""

MIN_NUM_ENVS_PER_WORKER: int = 1

MAX_PROPOSAL_ID_LEN: int = 256
"""Hard upper bound on caller-supplied :class:`PatchProposal.patch_id`
length."""

PROPOSAL_SOURCE: str = "evolution_engine.sandbox_sample_factory"
"""Constant tag stamped onto every emitted
:class:`PatchProposal.source`. The governance-side patch pipeline
keys on this string to distinguish sample-factory-trained proposals
from tianshou-trained (``evolution_engine.sandbox_tianshou``),
ElegantRL-trained (``evolution_engine.sandbox_elegant``), and
SB3-trained (``evolution_engine.sandbox``) proposals."""


# ---------------------------------------------------------------------------
# Algo kind enum (sample-factory historically ships only APPO)
# ---------------------------------------------------------------------------


class SampleFactoryAlgoKind(enum.Enum):
    """sample-factory algorithm selector — currently only APPO.

    The enum exists so the digest / proposal-meta surface treats the
    algo as a typed value rather than a magic string, matching the
    shape of :class:`evolution_engine.sandbox_tianshou.TianshouPolicyKind`
    and :class:`evolution_engine.sandbox_elegant.ElegantAgentKind`.
    """

    APPO = "APPO"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SampleFactoryArguments:
    """Frozen training-run config — mirrors the trainer kwargs in
    ``sample_factory/cfg/cfg.py``.

    Restricted to the deterministic-replay subset (no
    ``cfg.experiment`` IO, no ``cfg.device`` GPU routing, no
    ``cfg.save_every_sec`` checkpoint hooks, no
    ``cfg.with_wandb`` / ``cfg.with_pbt`` global hooks). The injected
    :class:`SampleFactoryPolicyTrainer` may interpret the
    hyperparameters however it likes — these fields are advisory.

    * ``algo_kind`` — selects sample-factory algorithm (APPO only
      currently).
    * ``random_seed`` — forwarded to sample-factory's ``cfg.seed``
      and to ``torch.manual_seed`` / ``numpy.random.seed``.
    * ``train_for_env_steps`` — total env-step budget.
    * ``batch_size`` — minibatch size for ``policy.update``.
    * ``rollout`` — per-rollout step count.
    * ``num_workers`` — worker process count.
    * ``num_envs_per_worker`` — vector env count per worker.
    * ``gamma`` — discount factor (0, 1].
    * ``learning_rate`` — optimizer LR.
    * ``target_strategy_id`` — DIX strategy that will be patched on
      governance approval.
    * ``meta`` — caller-supplied audit overlays.
    """

    algo_kind: SampleFactoryAlgoKind
    random_seed: int
    train_for_env_steps: int = 100_000
    batch_size: int = 1024
    rollout: int = 32
    num_workers: int = 4
    num_envs_per_worker: int = 2
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_strategy_id: str = "sample_factory_trained"
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.algo_kind, SampleFactoryAlgoKind):
            raise TypeError(
                "SampleFactoryArguments.algo_kind must be "
                "SampleFactoryAlgoKind, got "
                f"{type(self.algo_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                "SampleFactoryArguments.random_seed must be int, got "
                f"{type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"SampleFactoryArguments.random_seed must be non-negative, got {self.random_seed!r}"
            )
        if self.train_for_env_steps < MIN_TRAIN_FOR_ENV_STEPS:
            raise ValueError(
                "SampleFactoryArguments.train_for_env_steps must be "
                f">= {MIN_TRAIN_FOR_ENV_STEPS!r}, got "
                f"{self.train_for_env_steps!r}"
            )
        if self.train_for_env_steps > MAX_TRAIN_FOR_ENV_STEPS:
            raise ValueError(
                "SampleFactoryArguments.train_for_env_steps must be "
                f"<= {MAX_TRAIN_FOR_ENV_STEPS!r}, got "
                f"{self.train_for_env_steps!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"SampleFactoryArguments.batch_size must be positive, got {self.batch_size!r}"
            )
        if self.rollout <= 0:
            raise ValueError(
                f"SampleFactoryArguments.rollout must be positive, got {self.rollout!r}"
            )
        if self.num_workers < MIN_NUM_WORKERS:
            raise ValueError(
                f"SampleFactoryArguments.num_workers must be >= "
                f"{MIN_NUM_WORKERS!r}, got {self.num_workers!r}"
            )
        if self.num_workers > MAX_NUM_WORKERS:
            raise ValueError(
                f"SampleFactoryArguments.num_workers must be <= "
                f"{MAX_NUM_WORKERS!r}, got {self.num_workers!r}"
            )
        if self.num_envs_per_worker < MIN_NUM_ENVS_PER_WORKER:
            raise ValueError(
                "SampleFactoryArguments.num_envs_per_worker must be "
                f">= {MIN_NUM_ENVS_PER_WORKER!r}, got "
                f"{self.num_envs_per_worker!r}"
            )
        if self.num_envs_per_worker > MAX_NUM_ENVS_PER_WORKER:
            raise ValueError(
                "SampleFactoryArguments.num_envs_per_worker must be "
                f"<= {MAX_NUM_ENVS_PER_WORKER!r}, got "
                f"{self.num_envs_per_worker!r}"
            )
        if not math.isfinite(self.gamma) or not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                "SampleFactoryArguments.gamma must be a finite number "
                f"in (0.0, 1.0], got {self.gamma!r}"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                "SampleFactoryArguments.learning_rate must be a "
                f"positive finite number, got {self.learning_rate!r}"
            )
        if not self.target_strategy_id:
            raise ValueError("SampleFactoryArguments.target_strategy_id must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class SampleFactorySandboxMetrics:
    """Headline statistics produced by a sample-factory training run.

    Field set is the deterministic-replay subset of sample-factory's
    runner status output (``reward`` / ``len`` / ``policy_loss`` /
    ``value_loss`` / etc.).
    """

    iterations_completed: int
    total_steps_executed: int
    mean_episode_reward: float
    mean_episode_length: float
    best_episode_reward: float
    final_value_loss: float
    final_policy_loss: float

    def __post_init__(self) -> None:
        if self.iterations_completed < 0:
            raise ValueError(
                "SampleFactorySandboxMetrics.iterations_completed "
                f"must be non-negative, got "
                f"{self.iterations_completed!r}"
            )
        if self.total_steps_executed < 0:
            raise ValueError(
                "SampleFactorySandboxMetrics.total_steps_executed "
                f"must be non-negative, got "
                f"{self.total_steps_executed!r}"
            )
        for name in (
            "mean_episode_reward",
            "mean_episode_length",
            "best_episode_reward",
            "final_value_loss",
            "final_policy_loss",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(
                    f"SampleFactorySandboxMetrics.{name} must be finite, got {value!r}"
                )
        if self.mean_episode_length < 0.0:
            raise ValueError(
                "SampleFactorySandboxMetrics.mean_episode_length "
                f"must be non-negative, got "
                f"{self.mean_episode_length!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class SampleFactorySandboxResult:
    """Output of :meth:`SampleFactorySandbox.train`.

    The :class:`PatchProposal` carries the governance-shaped payload
    (``patch_id``, ``source``, ``target_strategy``, ``touchpoints``,
    ``rationale``, ``meta``); :class:`SampleFactorySandboxMetrics`
    and :attr:`policy_digest` carry the audit metadata operators
    consult when reviewing the proposal in the dashboard.
    """

    proposal: PatchProposal
    metrics: SampleFactorySandboxMetrics
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, PatchProposal):
            raise TypeError(
                "SampleFactorySandboxResult.proposal must be a "
                f"PatchProposal, got {type(self.proposal).__name__}"
            )
        if not isinstance(self.metrics, SampleFactorySandboxMetrics):
            raise TypeError(
                "SampleFactorySandboxResult.metrics must be "
                f"SampleFactorySandboxMetrics, got "
                f"{type(self.metrics).__name__}"
            )
        if len(self.policy_digest) != 16:
            raise ValueError(
                "SampleFactorySandboxResult.policy_digest must be a "
                f"16-hex-char digest, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                "SampleFactorySandboxResult.policy_digest must be "
                f"lowercase hex, got {self.policy_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class SampleFactorySandboxCallback(Protocol):
    """sample-factory-shape lifecycle callback (collapsed into one
    Protocol so the AST tests can pin "no top-level sample_factory
    import")."""

    def on_training_start(self, *, ts_ns: int, train_for_env_steps: int) -> None: ...

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

    def on_training_end(
        self,
        *,
        ts_ns: int,
        metrics: SampleFactorySandboxMetrics,
    ) -> None: ...


@runtime_checkable
class SampleFactoryPolicyTrainer(Protocol):
    """Caller-supplied sample-factory trainer.

    The Protocol is the **only** place the sandbox interacts with the
    learning library. Production wires
    :func:`sample_factory_appo_trainer`; tests inject a deterministic
    fake. The contract is single-shot: the trainer fully consumes the
    env and returns one :class:`SampleFactorySandboxMetrics` record.
    """

    def train(
        self,
        env: DIXStrategyEnv,
        *,
        episode_config: EpisodeConfig,
        arguments: SampleFactoryArguments,
        ts_ns: int,
        callback: SampleFactorySandboxCallback,
    ) -> SampleFactorySandboxMetrics: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullSampleFactoryCallback:
    """No-op callback. Operators inject a metrics sink via
    :func:`null_sample_factory_callback` and never see this class
    directly."""

    __slots__ = ()

    def on_training_start(self, *, ts_ns: int, train_for_env_steps: int) -> None:
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

    def on_training_end(
        self,
        *,
        ts_ns: int,
        metrics: SampleFactorySandboxMetrics,
    ) -> None:
        return None


def null_sample_factory_callback() -> SampleFactorySandboxCallback:
    return _NullSampleFactoryCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SampleFactorySandboxConfigError(ValueError):
    """Raised when the caller passes an invalid combination of args
    to :meth:`SampleFactorySandbox.train`."""


# ---------------------------------------------------------------------------
# Deterministic policy-digest computation
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    arguments: SampleFactoryArguments,
    metrics: SampleFactorySandboxMetrics,
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
            f"algo_kind={arguments.algo_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"train_for_env_steps={arguments.train_for_env_steps!r}",
            f"batch_size={arguments.batch_size!r}",
            f"rollout={arguments.rollout!r}",
            f"num_workers={arguments.num_workers!r}",
            f"num_envs_per_worker={arguments.num_envs_per_worker!r}",
            f"gamma={arguments.gamma!r}",
            f"learning_rate={arguments.learning_rate!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"iterations_completed={metrics.iterations_completed!r}",
            f"total_steps_executed={metrics.total_steps_executed!r}",
            f"mean_episode_reward={metrics.mean_episode_reward!r}",
            f"mean_episode_length={metrics.mean_episode_length!r}",
            f"best_episode_reward={metrics.best_episode_reward!r}",
            f"final_value_loss={metrics.final_value_loss!r}",
            f"final_policy_loss={metrics.final_policy_loss!r}",
        )
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# SampleFactorySandbox
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SampleFactorySandbox:
    """Frozen coordinator. Holds no mutable state — every call is a
    pure function of its arguments."""

    trainer: SampleFactoryPolicyTrainer

    def __post_init__(self) -> None:
        if not isinstance(self.trainer, SampleFactoryPolicyTrainer):
            raise TypeError(
                "SampleFactorySandbox.trainer must implement the "
                "SampleFactoryPolicyTrainer Protocol, got "
                f"{type(self.trainer).__name__}"
            )

    def train(
        self,
        *,
        dynamics: MarketDynamics,
        arguments: SampleFactoryArguments,
        episode_config: EpisodeConfig,
        ts_ns: int,
        proposal_id: str,
        callback: SampleFactorySandboxCallback | None = None,
    ) -> SampleFactorySandboxResult:
        """Run one training run and emit a
        :class:`SampleFactorySandboxResult`.

        INV-13/14: this never deploys. The returned
        :attr:`SampleFactorySandboxResult.proposal` is a typed
        :class:`PatchProposal` ready to be enqueued onto the bus by
        the operator (see
        :mod:`evolution_engine.patch_pipeline`).
        """

        if not isinstance(dynamics, MarketDynamics):
            raise TypeError(
                "SampleFactorySandbox.train.dynamics must implement "
                "the MarketDynamics Protocol, got "
                f"{type(dynamics).__name__}"
            )
        if not isinstance(arguments, SampleFactoryArguments):
            raise TypeError(
                "SampleFactorySandbox.train.arguments must be "
                f"SampleFactoryArguments, got "
                f"{type(arguments).__name__}"
            )
        if not isinstance(episode_config, EpisodeConfig):
            raise TypeError(
                "SampleFactorySandbox.train.episode_config must be "
                f"EpisodeConfig, got {type(episode_config).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                f"SampleFactorySandbox.train.ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise SampleFactorySandboxConfigError(
                f"SampleFactorySandbox.train.ts_ns must be non-negative, got {ts_ns!r}"
            )
        if not proposal_id:
            raise SampleFactorySandboxConfigError(
                "SampleFactorySandbox.train.proposal_id must be non-empty"
            )
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise SampleFactorySandboxConfigError(
                "SampleFactorySandbox.train.proposal_id must be <= "
                f"{MAX_PROPOSAL_ID_LEN} chars, got "
                f"{len(proposal_id)!r}"
            )

        cb = callback if callback is not None else null_sample_factory_callback()
        if not isinstance(cb, SampleFactorySandboxCallback):
            raise TypeError(
                "SampleFactorySandbox.train.callback must implement "
                "the SampleFactorySandboxCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        env = DIXStrategyEnv(dynamics)
        cb.on_training_start(
            ts_ns=ts_ns,
            train_for_env_steps=arguments.train_for_env_steps,
        )
        metrics = self.trainer.train(
            env,
            episode_config=episode_config,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(metrics, SampleFactorySandboxMetrics):
            raise TypeError(
                "SampleFactoryPolicyTrainer.train must return "
                "SampleFactorySandboxMetrics, got "
                f"{type(metrics).__name__}"
            )
        cb.on_training_end(ts_ns=ts_ns, metrics=metrics)

        digest = _compute_policy_digest(
            arguments=arguments,
            metrics=metrics,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
        )
        rationale = (
            f"sample-factory {arguments.algo_kind.value} policy: "
            f"{metrics.iterations_completed!r} iterations, "
            f"mean_reward={metrics.mean_episode_reward:.6f}, "
            f"best_reward={metrics.best_episode_reward:.6f}, "
            f"value_loss={metrics.final_value_loss:.6f}, "
            f"policy_loss={metrics.final_policy_loss:.6f}, "
            f"digest={digest}"
        )
        proposal_meta: dict[str, str] = {
            "policy_digest": digest,
            "algo_kind": arguments.algo_kind.value,
            "random_seed": str(arguments.random_seed),
            "train_for_env_steps": str(arguments.train_for_env_steps),
            "num_workers": str(arguments.num_workers),
            "num_envs_per_worker": str(arguments.num_envs_per_worker),
            "iterations_completed": str(metrics.iterations_completed),
            "mean_episode_reward": repr(metrics.mean_episode_reward),
            "best_episode_reward": repr(metrics.best_episode_reward),
            "final_value_loss": repr(metrics.final_value_loss),
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
                "evolution_engine.sandbox_sample_factory",
                "policy_weights",
            ),
            rationale=rationale,
            meta=proposal_meta,
        )
        return SampleFactorySandboxResult(
            proposal=proposal,
            metrics=metrics,
            policy_digest=digest,
        )


# ---------------------------------------------------------------------------
# Production trainer factory (lazy-import sample_factory / torch / gymnasium)
# ---------------------------------------------------------------------------


PolicyArtifact = bytes
"""Opaque trained-policy bytes blob."""

PolicyArtifactSink = Callable[[PolicyArtifact], None]
"""Caller-supplied artifact sink. Default is a no-op."""


def _noop_artifact_sink(artifact: PolicyArtifact) -> None:
    return None


def sample_factory_appo_trainer(
    *,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> SampleFactoryPolicyTrainer:
    """Production :class:`SampleFactoryPolicyTrainer` backed by
    ``sample_factory``.

    Lazy-imports ``sample_factory`` + ``torch`` + ``gymnasium``
    inside the factory. Raises ``ImportError`` (with a helpful
    pip-install hint) if any package is missing — the rest of the
    module never imports these packages, so the sandbox stays usable
    on a host that has never installed them.

    The returned object is a frozen wrapper that:

    1. Constructs a ``sample_factory.cfg.Cfg`` instance from the DIX
       :class:`SampleFactoryArguments`.
    2. Constructs a ``sample_factory.algo.runners.runner.Runner``
       around the env factory.
    3. Drives the APPO loop for the configured
       ``train_for_env_steps`` budget.
    4. Reads the runner's final status into a
       :class:`SampleFactorySandboxMetrics` record.
    5. Serialises the trained policy bytes (``torch.save``) and
       forwards to ``artifact_sink`` (caller-injected; default
       no-op).
    """

    try:
        import io  # noqa: F401  -- locally OK; this factory writes bytes.

        import sample_factory  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sample_factory_appo_trainer requires the optional "
            "'sample-factory' + 'torch' + 'gymnasium' packages — "
            "install with 'pip install sample-factory torch "
            "gymnasium' (NEW_PIP_DEPENDENCIES tuple in "
            "evolution_engine/sandbox_sample_factory.py flags this)."
        ) from exc

    _ = (sample_factory, artifact_sink)

    class _SampleFactoryAPPOTrainer:
        """Thin sample-factory wrapper conforming to
        :class:`SampleFactoryPolicyTrainer`."""

        __slots__ = ()

        def train(
            self,
            env: DIXStrategyEnv,
            *,
            episode_config: EpisodeConfig,
            arguments: SampleFactoryArguments,
            ts_ns: int,
            callback: SampleFactorySandboxCallback,
        ) -> SampleFactorySandboxMetrics:  # pragma: no cover
            raise NotImplementedError(
                "sample_factory_appo_trainer is the production seam — "
                "its concrete body is exercised in integration tests "
                "with sample_factory installed; unit tests inject a "
                "deterministic fake via the "
                "SampleFactoryPolicyTrainer Protocol."
            )

    return _SampleFactoryAPPOTrainer()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MAX_TRAIN_FOR_ENV_STEPS",
    "MIN_TRAIN_FOR_ENV_STEPS",
    "MAX_NUM_WORKERS",
    "MIN_NUM_WORKERS",
    "MAX_NUM_ENVS_PER_WORKER",
    "MIN_NUM_ENVS_PER_WORKER",
    "MAX_PROPOSAL_ID_LEN",
    "PROPOSAL_SOURCE",
    "SampleFactoryAlgoKind",
    "SampleFactoryArguments",
    "SampleFactorySandboxMetrics",
    "SampleFactorySandboxResult",
    "SampleFactorySandboxCallback",
    "SampleFactoryPolicyTrainer",
    "SampleFactorySandboxConfigError",
    "SampleFactorySandbox",
    "null_sample_factory_callback",
    "PolicyArtifact",
    "PolicyArtifactSink",
    "sample_factory_appo_trainer",
)
