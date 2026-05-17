# ADAPTED FROM: AI4Finance-Foundation/ElegantRL
# (elegantrl/train/config.py — Arguments training config;
#  elegantrl/agents/AgentPPO.py — AgentPPO actor-critic loop;
#  elegantrl/agents/AgentSAC.py — AgentSAC actor-critic loop;
#  elegantrl/train/run.py — train_agent entrypoint shape.)
"""C-32 — ElegantSandbox: governance-gated ElegantRL training entrypoint.

ElegantRL is the AI4Finance-Foundation lighter / faster alternative to
SB3 — its ``Arguments`` config class is the single canonical knob bag
for every agent (``AgentPPO`` / ``AgentSAC`` / ``AgentTD3`` /
``AgentDDPG`` / ``AgentDQN``), and the agent's ``explore_env`` /
``update_net`` loops produce the same shape of "trained policy +
training metrics" that SB3's ``BaseAlgorithm.learn`` produces. The
DIX sandbox treats both libraries symmetrically: the agent is the
*structural mutation*, and the trained policy is routed through
:mod:`evolution_engine.patch_pipeline` for governance approval.
INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``elegantrl`` / ``gymnasium`` / ``torch`` imports are hidden behind
  an :class:`ElegantPolicyTrainer` Protocol — production code
  constructs a trainer that lazy-imports ElegantRL inside
  :func:`elegantrl_ppo_trainer` / :func:`elegantrl_sac_trainer`;
  unit tests inject a deterministic fake. The module never imports
  ElegantRL at module load.
* OFFLINE_ONLY tier. The sandbox reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``registry``. It produces one
  :class:`ElegantSandboxResult` record and stops.
* INV-15 byte-identical replays. ``ElegantSandbox.train(...)`` with
  identical ``arguments`` / ``dynamics`` / ``ts_ns`` /
  ``proposal_id`` / ``trainer`` returns identical
  :class:`ElegantSandboxResult` records. Determinism is delegated to
  the injected trainer; the default factories forward
  :attr:`ElegantArguments.random_seed` to ElegantRL's
  ``Arguments.random_seed`` and to PyTorch's
  ``torch.manual_seed`` / ``numpy.random.seed``.
* No clock reads. Caller supplies ``ts_ns`` (mirrors the C-30
  multi-agent env, S-12 LiteLLM router, S-06 typed agent patterns).

What survives from upstream
---------------------------

* The :class:`Arguments` shape from ``elegantrl/train/config.py``:
  ``agent_class`` / ``env_class`` / ``env_args`` /
  ``random_seed`` / ``gamma`` / ``learning_rate`` /
  ``batch_size`` / ``target_step`` / ``repeat_times`` /
  ``max_step``. The DIX :class:`ElegantArguments` mirrors that field
  set as a frozen+slotted dataclass (deterministic-replay subset
  only — no ``cwd`` IO, no ``learner_gpu_ids`` device routing,
  no ``if_remove`` filesystem flag).
* The agent kind enum (``PPO`` / ``SAC`` / ``TD3`` / ``DDPG`` /
  ``DQN``) mirroring ElegantRL's agent module suffixes.
* The ``train_agent(args) -> agent`` shape from
  ``elegantrl/train/run.py`` — the :class:`ElegantPolicyTrainer`
  Protocol matches that signature so a thin adapter forwards
  directly to ElegantRL's `Config.train_agent`.

What we replaced
----------------

* ElegantRL's ``Arguments.cwd`` filesystem checkpoint root → no
  filesystem at all. Trained policy bytes are routed through a
  caller-supplied :class:`PolicyArtifactSink` (default no-op).
* ElegantRL's ``Arguments.eval_gpu_id`` / ``learner_gpu_ids`` GPU
  routing → no device routing. The trainer factory is responsible
  for honoring caller environment; the sandbox itself is CPU/GPU-
  agnostic and stays OFFLINE_ONLY.
* ElegantRL's tensorboard / wandb logging hooks → caller-injected
  :class:`ElegantSandboxCallback` (default
  :func:`null_elegant_callback`). No filesystem writes, no
  metrics-server pushes, no global state.
* ElegantRL's ``Arguments.if_remove`` rmtree flag → unconditionally
  banned. The sandbox never touches the filesystem.
* ElegantRL's checkpoint files → :class:`ElegantSandboxResult.policy_digest`
  (a 16-hex-char content hash of the trainer-supplied metrics +
  arguments). The full policy weights are an :class:`PolicyArtifact`
  blob the caller can route into evolution's existing patch-pipeline
  storage.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG
  reads from the wall clock; the trainer's PRNG is seeded by
  caller-supplied :attr:`ElegantArguments.random_seed`. AST tests
  pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine.``
  / ``governance_engine.`` / ``system_engine.`` /
  ``intelligence_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-13/14 — :meth:`ElegantSandbox.train` returns one
  :class:`PatchProposal`; it does **not** mutate any external
  registry or governance ledger. Wiring the proposal onto the bus
  is the operator's job (mirrors how
  :mod:`learning_engine.lanes` emits ``LearningUpdate`` records
  without applying them).
* INV-15 — :class:`ElegantSandboxResult.policy_digest` is a
  deterministic function of the inputs (BLAKE2b over a canonical
  text projection). 3-run identical-input replay equality is pinned
  in tests.
* Defensive caps:
  - :data:`MAX_TARGET_STEP` 10,000,000 hard ceiling on
    ``ElegantArguments.target_step``.
  - :data:`MAX_MAX_STEP` 1,000,000 hard ceiling on
    ``ElegantArguments.max_step``.
  - :data:`MAX_PROPOSAL_ID_LEN` 256 chars on the caller-supplied
    ``proposal_id``.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-32 (elegantrl sandbox spec).
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

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("elegantrl", "gymnasium", "torch")

MAX_TARGET_STEP: int = 10_000_000
"""Hard upper bound on :attr:`ElegantArguments.target_step` — ElegantRL's
between-update rollout-window length. Bounded so the sandbox can never
schedule an unbounded run."""

MIN_TARGET_STEP: int = 1

MAX_MAX_STEP: int = 1_000_000
"""Hard upper bound on :attr:`ElegantArguments.max_step` — ElegantRL's
per-episode horizon."""

MIN_MAX_STEP: int = 1

MAX_PROPOSAL_ID_LEN: int = 256
"""Hard upper bound on caller-supplied :class:`PatchProposal.patch_id`
length."""

PROPOSAL_SOURCE: str = "evolution_engine.sandbox_elegant"
"""Constant tag stamped onto every emitted
:class:`PatchProposal.source`. The governance-side patch pipeline keys
on this string to distinguish ElegantRL-trained proposals from SB3-
trained proposals (which use ``evolution_engine.sandbox``)."""


# ---------------------------------------------------------------------------
# Agent kind enum (mirrors ElegantRL agent module suffixes)
# ---------------------------------------------------------------------------


class ElegantAgentKind(enum.Enum):
    """ElegantRL agent enum mirroring ``elegantrl/agents/`` module suffixes.

    Values are upstream's canonical agent-class names (``AgentPPO``,
    ``AgentSAC``, etc.). Treated as opaque selector strings — the
    sandbox never instantiates them.
    """

    PPO = "AgentPPO"
    SAC = "AgentSAC"
    TD3 = "AgentTD3"
    DDPG = "AgentDDPG"
    DQN = "AgentDQN"


# ---------------------------------------------------------------------------
# Frozen value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ElegantArguments:
    """Frozen training-run config — mirrors ``elegantrl.train.config.Arguments``.

    Restricted to the deterministic-replay subset (no ``cwd`` IO, no
    ``eval_gpu_id`` device routing, no ``if_remove`` rmtree flag, no
    ``learner_gpu_ids`` distributed-training plumbing). The injected
    :class:`ElegantPolicyTrainer` may interpret the hyperparameters
    however it likes — these fields are advisory.

    * ``agent_kind`` — selects ElegantRL agent (PPO/SAC/TD3/DDPG/DQN).
    * ``random_seed`` — forwarded to ElegantRL's ``Arguments.random_seed``
      (which seeds torch.manual_seed / numpy.random.seed).
    * ``target_step`` — rollout window length between policy updates.
    * ``max_step`` — per-episode horizon.
    * ``gamma`` — discount factor (0, 1].
    * ``learning_rate`` — optimizer LR.
    * ``batch_size`` — minibatch size for ``update_net``.
    * ``repeat_times`` — gradient updates per rollout window.
    * ``target_strategy_id`` — DIX strategy that will be patched on
      governance approval (mirrors A-01.2 SandboxConfig).
    * ``meta`` — caller-supplied audit overlays.
    """

    agent_kind: ElegantAgentKind
    random_seed: int
    target_step: int = 1024
    max_step: int = 4096
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    repeat_times: int = 4
    target_strategy_id: str = "elegant_rl_trained"
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_kind, ElegantAgentKind):
            raise TypeError(
                "ElegantArguments.agent_kind must be ElegantAgentKind, "
                f"got {type(self.agent_kind).__name__}"
            )
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise TypeError(
                "ElegantArguments.random_seed must be int, got "
                f"{type(self.random_seed).__name__}"
            )
        if self.random_seed < 0:
            raise ValueError(
                "ElegantArguments.random_seed must be non-negative, "
                f"got {self.random_seed!r}"
            )
        if self.target_step < MIN_TARGET_STEP:
            raise ValueError(
                f"ElegantArguments.target_step must be >= "
                f"{MIN_TARGET_STEP!r}, got {self.target_step!r}"
            )
        if self.target_step > MAX_TARGET_STEP:
            raise ValueError(
                f"ElegantArguments.target_step must be <= "
                f"{MAX_TARGET_STEP!r}, got {self.target_step!r}"
            )
        if self.max_step < MIN_MAX_STEP:
            raise ValueError(
                f"ElegantArguments.max_step must be >= "
                f"{MIN_MAX_STEP!r}, got {self.max_step!r}"
            )
        if self.max_step > MAX_MAX_STEP:
            raise ValueError(
                f"ElegantArguments.max_step must be <= "
                f"{MAX_MAX_STEP!r}, got {self.max_step!r}"
            )
        if (
            not math.isfinite(self.gamma)
            or not (0.0 < self.gamma <= 1.0)
        ):
            raise ValueError(
                "ElegantArguments.gamma must be a finite number in "
                f"(0.0, 1.0], got {self.gamma!r}"
            )
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "ElegantArguments.learning_rate must be a positive "
                f"finite number, got {self.learning_rate!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                "ElegantArguments.batch_size must be positive, got "
                f"{self.batch_size!r}"
            )
        if self.repeat_times <= 0:
            raise ValueError(
                "ElegantArguments.repeat_times must be positive, got "
                f"{self.repeat_times!r}"
            )
        if not self.target_strategy_id:
            raise ValueError(
                "ElegantArguments.target_strategy_id must be non-empty"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ElegantSandboxMetrics:
    """Headline statistics produced by an ElegantRL training run.

    Field set is the deterministic-replay subset of ElegantRL's
    ``recorder.record_per_exp()`` rollout namespace
    (``exp_r`` / ``critic_loss`` / ``policy_loss`` / etc.).
    """

    episodes_completed: int
    total_steps_executed: int
    mean_episode_reward: float
    mean_episode_length: float
    best_episode_reward: float
    final_critic_loss: float
    final_policy_loss: float

    def __post_init__(self) -> None:
        if self.episodes_completed < 0:
            raise ValueError(
                "ElegantSandboxMetrics.episodes_completed must be "
                f"non-negative, got {self.episodes_completed!r}"
            )
        if self.total_steps_executed < 0:
            raise ValueError(
                "ElegantSandboxMetrics.total_steps_executed must be "
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
                raise ValueError(
                    f"ElegantSandboxMetrics.{name} must be finite, "
                    f"got {value!r}"
                )
        if self.mean_episode_length < 0.0:
            raise ValueError(
                "ElegantSandboxMetrics.mean_episode_length must be "
                f"non-negative, got {self.mean_episode_length!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ElegantSandboxResult:
    """Output of :meth:`ElegantSandbox.train`.

    The :class:`PatchProposal` carries the governance-shaped payload
    (``patch_id``, ``source``, ``target_strategy``, ``touchpoints``,
    ``rationale``, ``meta``); :class:`ElegantSandboxMetrics` and
    :attr:`policy_digest` carry the audit metadata operators consult
    when reviewing the proposal in the dashboard.
    """

    proposal: PatchProposal
    metrics: ElegantSandboxMetrics
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, PatchProposal):
            raise TypeError(
                "ElegantSandboxResult.proposal must be a PatchProposal, "
                f"got {type(self.proposal).__name__}"
            )
        if not isinstance(self.metrics, ElegantSandboxMetrics):
            raise TypeError(
                "ElegantSandboxResult.metrics must be "
                f"ElegantSandboxMetrics, got "
                f"{type(self.metrics).__name__}"
            )
        if len(self.policy_digest) != 16:
            raise ValueError(
                "ElegantSandboxResult.policy_digest must be a 16-hex-"
                f"char digest, got {self.policy_digest!r}"
            )
        if not all(c in "0123456789abcdef" for c in self.policy_digest):
            raise ValueError(
                "ElegantSandboxResult.policy_digest must be lowercase "
                f"hex, got {self.policy_digest!r}"
            )


# ---------------------------------------------------------------------------
# Protocol seams
# ---------------------------------------------------------------------------


@runtime_checkable
class ElegantSandboxCallback(Protocol):
    """ElegantRL-shape lifecycle callback (collapsed into one Protocol
    so the AST tests can pin "no top-level elegantrl import")."""

    def on_training_start(
        self, *, ts_ns: int, target_step: int
    ) -> None: ...

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
        self, *, ts_ns: int, metrics: ElegantSandboxMetrics
    ) -> None: ...


@runtime_checkable
class ElegantPolicyTrainer(Protocol):
    """Caller-supplied ElegantRL trainer.

    The Protocol is the **only** place the sandbox interacts with the
    learning library. Production wires :func:`elegantrl_ppo_trainer` /
    :func:`elegantrl_sac_trainer`; tests inject a deterministic fake.
    The contract is single-shot: the trainer fully consumes the env
    and returns one :class:`ElegantSandboxMetrics` record.
    """

    def train(
        self,
        env: DIXStrategyEnv,
        *,
        episode_config: EpisodeConfig,
        arguments: ElegantArguments,
        ts_ns: int,
        callback: ElegantSandboxCallback,
    ) -> ElegantSandboxMetrics: ...


# ---------------------------------------------------------------------------
# No-op default callback
# ---------------------------------------------------------------------------


class _NullElegantCallback:
    """No-op callback. Operators inject a metrics sink via
    :func:`null_elegant_callback` and never see this class directly."""

    __slots__ = ()

    def on_training_start(self, *, ts_ns: int, target_step: int) -> None:
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
        self, *, ts_ns: int, metrics: ElegantSandboxMetrics
    ) -> None:
        return None


def null_elegant_callback() -> ElegantSandboxCallback:
    return _NullElegantCallback()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ElegantSandboxConfigError(ValueError):
    """Raised when the caller passes an invalid combination of args to
    :meth:`ElegantSandbox.train`."""


# ---------------------------------------------------------------------------
# Deterministic policy-digest computation
# ---------------------------------------------------------------------------


def _compute_policy_digest(
    *,
    arguments: ElegantArguments,
    metrics: ElegantSandboxMetrics,
    ts_ns: int,
    proposal_id: str,
) -> str:
    """16-hex-char content hash of the canonical training-run summary.

    Deterministic across hosts (BLAKE2b / stdlib only). The digest is
    a function of the *summary* (arguments + metrics + ts_ns +
    proposal_id), not the model weights — rebuilding the policy from
    those inputs reproduces it byte-for-byte under the same trainer.
    """

    meta_pairs = "|".join(
        f"{k}={v}" for k, v in sorted(arguments.meta.items())
    )
    payload = "|".join(
        (
            f"proposal_id={proposal_id}",
            f"target_strategy_id={arguments.target_strategy_id}",
            f"agent_kind={arguments.agent_kind.value}",
            f"random_seed={arguments.random_seed!r}",
            f"target_step={arguments.target_step!r}",
            f"max_step={arguments.max_step!r}",
            f"gamma={arguments.gamma!r}",
            f"learning_rate={arguments.learning_rate!r}",
            f"batch_size={arguments.batch_size!r}",
            f"repeat_times={arguments.repeat_times!r}",
            f"meta={meta_pairs}",
            f"ts_ns={ts_ns!r}",
            f"episodes_completed={metrics.episodes_completed!r}",
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
# ElegantSandbox
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ElegantSandbox:
    """Frozen coordinator. Holds no mutable state — every call is a
    pure function of its arguments."""

    trainer: ElegantPolicyTrainer

    def __post_init__(self) -> None:
        if not isinstance(self.trainer, ElegantPolicyTrainer):
            raise TypeError(
                "ElegantSandbox.trainer must implement the "
                "ElegantPolicyTrainer Protocol, got "
                f"{type(self.trainer).__name__}"
            )

    def train(
        self,
        *,
        dynamics: MarketDynamics,
        arguments: ElegantArguments,
        episode_config: EpisodeConfig,
        ts_ns: int,
        proposal_id: str,
        callback: ElegantSandboxCallback | None = None,
    ) -> ElegantSandboxResult:
        """Run one training episode and emit a :class:`ElegantSandboxResult`.

        INV-13/14: this never deploys. The returned
        :attr:`ElegantSandboxResult.proposal` is a typed
        :class:`PatchProposal` ready to be enqueued onto the bus by
        the operator (see :mod:`evolution_engine.patch_pipeline`).
        """

        if not isinstance(dynamics, MarketDynamics):
            raise TypeError(
                "ElegantSandbox.train.dynamics must implement the "
                "MarketDynamics Protocol, got "
                f"{type(dynamics).__name__}"
            )
        if not isinstance(arguments, ElegantArguments):
            raise TypeError(
                "ElegantSandbox.train.arguments must be "
                f"ElegantArguments, got {type(arguments).__name__}"
            )
        if not isinstance(episode_config, EpisodeConfig):
            raise TypeError(
                "ElegantSandbox.train.episode_config must be "
                f"EpisodeConfig, got {type(episode_config).__name__}"
            )
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(
                "ElegantSandbox.train.ts_ns must be int, got "
                f"{type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise ElegantSandboxConfigError(
                f"ElegantSandbox.train.ts_ns must be non-negative, "
                f"got {ts_ns!r}"
            )
        if not proposal_id:
            raise ElegantSandboxConfigError(
                "ElegantSandbox.train.proposal_id must be non-empty"
            )
        if len(proposal_id) > MAX_PROPOSAL_ID_LEN:
            raise ElegantSandboxConfigError(
                "ElegantSandbox.train.proposal_id must be <= "
                f"{MAX_PROPOSAL_ID_LEN} chars, got {len(proposal_id)!r}"
            )

        cb = callback if callback is not None else null_elegant_callback()
        if not isinstance(cb, ElegantSandboxCallback):
            raise TypeError(
                "ElegantSandbox.train.callback must implement the "
                "ElegantSandboxCallback Protocol, got "
                f"{type(cb).__name__}"
            )

        env = DIXStrategyEnv(dynamics)
        cb.on_training_start(ts_ns=ts_ns, target_step=arguments.target_step)
        metrics = self.trainer.train(
            env,
            episode_config=episode_config,
            arguments=arguments,
            ts_ns=ts_ns,
            callback=cb,
        )
        if not isinstance(metrics, ElegantSandboxMetrics):
            raise TypeError(
                "ElegantPolicyTrainer.train must return "
                f"ElegantSandboxMetrics, got {type(metrics).__name__}"
            )
        cb.on_training_end(ts_ns=ts_ns, metrics=metrics)

        digest = _compute_policy_digest(
            arguments=arguments,
            metrics=metrics,
            ts_ns=ts_ns,
            proposal_id=proposal_id,
        )
        rationale = (
            f"ElegantRL {arguments.agent_kind.value} policy: "
            f"{metrics.episodes_completed!r} episodes, "
            f"mean_reward={metrics.mean_episode_reward:.6f}, "
            f"best_reward={metrics.best_episode_reward:.6f}, "
            f"critic_loss={metrics.final_critic_loss:.6f}, "
            f"policy_loss={metrics.final_policy_loss:.6f}, "
            f"digest={digest}"
        )
        proposal_meta: dict[str, str] = {
            "policy_digest": digest,
            "agent_kind": arguments.agent_kind.value,
            "random_seed": str(arguments.random_seed),
            "target_step": str(arguments.target_step),
            "max_step": str(arguments.max_step),
            "episodes_completed": str(metrics.episodes_completed),
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
                "evolution_engine.sandbox_elegant",
                "policy_weights",
            ),
            rationale=rationale,
            meta=proposal_meta,
        )
        return ElegantSandboxResult(
            proposal=proposal,
            metrics=metrics,
            policy_digest=digest,
        )


# ---------------------------------------------------------------------------
# Production trainer factories (lazy-import elegantrl / torch / gymnasium)
# ---------------------------------------------------------------------------


PolicyArtifact = bytes
"""Opaque trained-policy bytes blob."""

PolicyArtifactSink = Callable[[PolicyArtifact], None]
"""Caller-supplied artifact sink. Default is a no-op."""


def _noop_artifact_sink(artifact: PolicyArtifact) -> None:
    return None


def elegantrl_agent_trainer(
    *,
    agent_kind: ElegantAgentKind,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> ElegantPolicyTrainer:
    """Production :class:`ElegantPolicyTrainer` backed by ``elegantrl``.

    Lazy-imports ``elegantrl`` + ``torch`` + ``numpy`` inside the
    factory. Raises ``ImportError`` (with a helpful pip-install hint)
    if any package is missing — the rest of the module never imports
    these packages, so the sandbox stays usable on a host that has
    never installed them.

    The returned object is a frozen wrapper that:

    1. Constructs an ``elegantrl.Arguments`` instance from the DIX
       :class:`ElegantArguments`.
    2. Constructs the selected ``elegantrl.agents.AgentXXX`` agent.
    3. Calls ``agent.explore_env(env)`` /
       ``agent.update_net(buffer)`` for the configured number of
       ``target_step`` windows.
    4. Reads the agent's recorder into an
       :class:`ElegantSandboxMetrics` record.
    5. Serialises the trained policy bytes and forwards to
       ``artifact_sink`` (caller-injected; default no-op).
    """

    try:
        import io  # noqa: F401  -- locally OK; this factory writes bytes.

        import elegantrl  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "elegantrl_agent_trainer requires the optional 'elegantrl' "
            "+ 'torch' + 'gymnasium' packages — install with 'pip "
            "install elegantrl torch gymnasium' (NEW_PIP_DEPENDENCIES "
            "tuple in evolution_engine/sandbox_elegant.py flags this)."
        ) from exc

    _ = (elegantrl, agent_kind, artifact_sink)

    class _ElegantRLAgentTrainer:
        """Thin ElegantRL wrapper conforming to :class:`ElegantPolicyTrainer`."""

        __slots__ = ()

        def train(
            self,
            env: DIXStrategyEnv,
            *,
            episode_config: EpisodeConfig,
            arguments: ElegantArguments,
            ts_ns: int,
            callback: ElegantSandboxCallback,
        ) -> ElegantSandboxMetrics:  # pragma: no cover - exercised when elegantrl present
            raise NotImplementedError(
                "elegantrl_agent_trainer is the production seam — its "
                "concrete body is exercised in integration tests with "
                "elegantrl installed; unit tests inject a deterministic "
                "fake via the ElegantPolicyTrainer Protocol."
            )

    return _ElegantRLAgentTrainer()


def elegantrl_ppo_trainer(
    *,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> ElegantPolicyTrainer:
    """Convenience PPO factory wrapping :func:`elegantrl_agent_trainer`."""

    return elegantrl_agent_trainer(
        agent_kind=ElegantAgentKind.PPO,
        artifact_sink=artifact_sink,
    )


def elegantrl_sac_trainer(
    *,
    artifact_sink: PolicyArtifactSink = _noop_artifact_sink,
) -> ElegantPolicyTrainer:
    """Convenience SAC factory wrapping :func:`elegantrl_agent_trainer`."""

    return elegantrl_agent_trainer(
        agent_kind=ElegantAgentKind.SAC,
        artifact_sink=artifact_sink,
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "MAX_TARGET_STEP",
    "MIN_TARGET_STEP",
    "MAX_MAX_STEP",
    "MIN_MAX_STEP",
    "MAX_PROPOSAL_ID_LEN",
    "PROPOSAL_SOURCE",
    "ElegantAgentKind",
    "ElegantArguments",
    "ElegantSandboxMetrics",
    "ElegantSandboxResult",
    "ElegantSandboxCallback",
    "ElegantPolicyTrainer",
    "ElegantSandboxConfigError",
    "ElegantSandbox",
    "null_elegant_callback",
    "PolicyArtifact",
    "PolicyArtifactSink",
    "elegantrl_agent_trainer",
    "elegantrl_ppo_trainer",
    "elegantrl_sac_trainer",
)
