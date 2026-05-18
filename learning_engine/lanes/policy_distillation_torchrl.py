# ADAPTED FROM: pytorch/rl (torchrl) + pytorch/tensordict
#   torchrl/envs/      — env wrappers (GymEnv, TransformedEnv)
#   torchrl/modules/   — actor-critic architectures (ProbabilisticActor,
#                        ValueOperator)
#   torchrl/collectors — SyncDataCollector (rollout collection)
#   tensordict         — TensorDict batched-tensor container
#
# License: MIT.
"""C-28 — TorchRL-flavoured policy distillation lane (OFFLINE_ONLY).

Mirrors B-13 (cleanrl/ppo) but uses TorchRL's actor-critic + TensorDict
collector shape. The actual ``torchrl`` / ``tensordict`` / ``torch``
imports live **only** inside :func:`torchrl_distiller_factory`. The
module is importable without the deps installed; AST tests pin no
top-level torch / tensordict / torchrl / numpy / gymnasium import.

Authority constraints (manifest §H1):

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG (the
  distiller's PRNG is seeded by caller-supplied seed and never reads
  the wall clock).
* No engine cross-imports — AST test pins no ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``intelligence_engine``
  import.
* INV-13/14: this lane never deploys directly. It emits a frozen
  :class:`DistillationProposalTorchRL` advisory; only
  ``evolution_engine.*`` may translate it into a typed
  :class:`~core.contracts.learning.PatchProposal` for governance
  approval.
* CPU-only by contract. ``device="cpu"`` is fixed on the config and
  the factory pins ``torch.backends.cudnn.deterministic = True`` /
  ``benchmark = False``.

INV-15 (replay determinism):

* :meth:`PolicyDistillationTorchRL.distill` with identical
  ``trajectories`` / ``config`` / ``ts_ns`` / ``proposal_id`` /
  ``distiller`` returns byte-identical :class:`DistillationResultTorchRL`
  records.
* :func:`compute_advantages` uses ``math.fsum`` for stable GAE-Lambda
  accumulation.
* All canonical projections sort their keys and the rollout digest
  is BLAKE2b-16 over the canonical text.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("torchrl", "tensordict", "torch")
TORCHRL_DISTILLER_VERSION: Final[str] = "c-28-torchrl-actor-critic-1"

MAX_TRAJECTORIES: Final[int] = 1024
MAX_STEPS_PER_TRAJECTORY: Final[int] = 1_000_000
MAX_OBS_DIM: Final[int] = 4096
MAX_ACTION_DIM: Final[int] = 1024
_FLOAT_EPSILON: Final[float] = 1e-9


class TorchRLDistillationError(ValueError):
    """Base error for the TorchRL distillation lane."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TensorDictStep:
    """One step recorded from a SyncDataCollector rollout.

    Mirrors the per-step keys TorchRL emits into a TensorDict batch
    (``observation``, ``action``, ``sample_log_prob``, ``state_value``,
    ``reward``, ``done``) but as a frozen value object so 3-run
    replays compare byte-identical.
    """

    obs: tuple[float, ...]
    action: tuple[float, ...]
    sample_log_prob: float
    state_value: float
    reward: float
    done: bool


@dataclass(frozen=True, slots=True)
class CollectorRollout:
    """One contiguous on-policy rollout from a SyncDataCollector worker."""

    rollout_id: str
    steps: tuple[TensorDictStep, ...]
    bootstrap_value: float = 0.0


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    """TorchRL ProbabilisticActor + ValueOperator hyperparameters.

    Field names mirror upstream where reasonable. ``device="cpu"`` is
    fixed by contract (determinism); GPU training is intentionally
    disabled at this seam.
    """

    learning_rate: float = 3e-4
    num_epochs: int = 4
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    critic_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: float | None = None
    obs_dim: int = 1
    action_dim: int = 1
    hidden_dim: int = 64
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Headline numbers from one TorchRL training pass.

    Mirrors the keys ``torchrl.objectives.ClipPPOLoss`` emits into the
    TensorDict ``loss`` namespace.
    """

    mean_reward: float
    best_reward: float
    loss_objective: float
    loss_critic: float
    loss_entropy: float
    approx_kl: float
    clip_fraction: float
    total_timesteps: int


@dataclass(frozen=True, slots=True)
class TorchRLPolicyArtifact:
    """Caller-provided policy blob projected to a stable digest."""

    backend: str
    content_digest: str
    obs_dim: int
    action_dim: int


@dataclass(frozen=True, slots=True)
class DistillationResultTorchRL:
    """Output of :meth:`PolicyDistillationTorchRL.distill`."""

    metrics: TrainingMetrics
    artifact: TorchRLPolicyArtifact
    policy_digest: str
    config_digest: str
    rollout_digest: str
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DistillationProposalTorchRL:
    """Advisory record consumed by the evolution-engine adapter."""

    ts_ns: int
    proposal_id: str
    target_strategy: str
    source: str
    touchpoints: tuple[str, ...]
    rationale: str
    policy_digest: str
    config_digest: str
    rollout_digest: str
    artifact_backend: str
    artifact_digest: str
    seed: int
    version: str


# ---------------------------------------------------------------------------
# Distiller Protocol + callback Protocol
# ---------------------------------------------------------------------------
class TorchRLDistillationCallback(Protocol):
    """Optional callback — mirrors TorchRL's TensorBoard logger surface."""

    def on_training_start(self, config: ActorCriticConfig) -> None: ...

    def on_epoch(
        self,
        epoch: int,
        approx_kl: float,
        loss_objective: float,
        loss_critic: float,
    ) -> None: ...

    def on_training_end(self, metrics: TrainingMetrics) -> None: ...


def null_torchrl_callback() -> TorchRLDistillationCallback:
    class _Null:
        def on_training_start(self, config: ActorCriticConfig) -> None:  # noqa: ARG002
            return None

        def on_epoch(
            self,
            epoch: int,  # noqa: ARG002
            approx_kl: float,  # noqa: ARG002
            loss_objective: float,  # noqa: ARG002
            loss_critic: float,  # noqa: ARG002
        ) -> None:
            return None

        def on_training_end(self, metrics: TrainingMetrics) -> None:  # noqa: ARG002
            return None

    return _Null()


class TorchRLDistiller(Protocol):
    """Pluggable distiller — production wires :func:`torchrl_distiller_factory`."""

    def train(
        self,
        *,
        rollouts: Sequence[CollectorRollout],
        config: ActorCriticConfig,
        seed: int,
        callback: TorchRLDistillationCallback,
    ) -> tuple[TrainingMetrics, TorchRLPolicyArtifact]: ...


# ---------------------------------------------------------------------------
# Pure determinism helpers
# ---------------------------------------------------------------------------
def compute_advantages(
    *,
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[float, ...]:
    """GAE-Lambda advantage estimation — stable ``math.fsum`` accumulation."""
    if not (len(rewards) == len(values) == len(dones)):
        raise TorchRLDistillationError("rewards / values / dones must have equal length")
    if not 0.0 <= gamma <= 1.0:
        raise TorchRLDistillationError("gamma must be in [0,1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise TorchRLDistillationError("gae_lambda must be in [0,1]")
    n = len(rewards)
    advantages: list[float] = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = bootstrap_value
            non_terminal = 0.0 if dones[t] else 1.0
        else:
            next_value = values[t + 1]
            non_terminal = 0.0 if dones[t] else 1.0
        delta = math.fsum((rewards[t], gamma * next_value * non_terminal, -values[t]))
        last_gae = math.fsum((delta, gamma * gae_lambda * non_terminal * last_gae))
        advantages[t] = last_gae
    return tuple(advantages)


def _blake2b_16(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _canonical_step(step: TensorDictStep) -> str:
    obs = "|".join(f"{v:.10g}" for v in step.obs)
    act = "|".join(f"{v:.10g}" for v in step.action)
    return (
        f"obs={obs};act={act};lp={step.sample_log_prob:.10g};"
        f"v={step.state_value:.10g};r={step.reward:.10g};"
        f"d={int(step.done)}"
    )


def _canonical_rollout(rollouts: tuple[CollectorRollout, ...]) -> str:
    pieces: list[str] = []
    for r in rollouts:
        steps = ";".join(_canonical_step(s) for s in r.steps)
        pieces.append(f"id={r.rollout_id};boot={r.bootstrap_value:.10g};steps=[{steps}]")
    return "||".join(pieces)


def _canonical_config(config: ActorCriticConfig) -> str:
    return (
        f"lr={config.learning_rate:.10g};epochs={config.num_epochs};"
        f"mb={config.minibatch_size};clip={config.clip_epsilon:.10g};"
        f"vc={config.critic_coef:.10g};ec={config.entropy_coef:.10g};"
        f"g={config.gamma:.10g};lam={config.gae_lambda:.10g};"
        f"mgn={config.max_grad_norm:.10g};"
        f"na={int(config.normalize_advantage)};"
        f"tkl={'' if config.target_kl is None else f'{config.target_kl:.10g}'};"
        f"obs={config.obs_dim};act={config.action_dim};"
        f"hid={config.hidden_dim};dev={config.device}"
    )


def _canonical_metrics(metrics: TrainingMetrics) -> str:
    return (
        f"r_mean={metrics.mean_reward:.10g};r_best={metrics.best_reward:.10g};"
        f"lobj={metrics.loss_objective:.10g};"
        f"lcri={metrics.loss_critic:.10g};lent={metrics.loss_entropy:.10g};"
        f"akl={metrics.approx_kl:.10g};clipf={metrics.clip_fraction:.10g};"
        f"ts={metrics.total_timesteps}"
    )


def _canonical_artifact(art: TorchRLPolicyArtifact) -> str:
    return (
        f"backend={art.backend};digest={art.content_digest};obs={art.obs_dim};act={art.action_dim}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_config(config: ActorCriticConfig) -> None:
    if config.device != "cpu":
        raise TorchRLDistillationError(
            "device must be 'cpu' (GPU training disabled for determinism)"
        )
    if config.learning_rate <= 0.0:
        raise TorchRLDistillationError("learning_rate must be > 0")
    if config.num_epochs <= 0:
        raise TorchRLDistillationError("num_epochs must be > 0")
    if config.minibatch_size <= 0:
        raise TorchRLDistillationError("minibatch_size must be > 0")
    if not 0.0 < config.clip_epsilon < 1.0:
        raise TorchRLDistillationError("clip_epsilon must be in (0,1)")
    if config.critic_coef < 0.0 or config.entropy_coef < 0.0:
        raise TorchRLDistillationError("critic_coef / entropy_coef must be >= 0")
    if not 0.0 <= config.gamma <= 1.0:
        raise TorchRLDistillationError("gamma must be in [0,1]")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise TorchRLDistillationError("gae_lambda must be in [0,1]")
    if config.max_grad_norm <= 0.0:
        raise TorchRLDistillationError("max_grad_norm must be > 0")
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise TorchRLDistillationError("target_kl must be > 0 or None")
    if not 1 <= config.obs_dim <= MAX_OBS_DIM:
        raise TorchRLDistillationError("obs_dim out of range")
    if not 1 <= config.action_dim <= MAX_ACTION_DIM:
        raise TorchRLDistillationError("action_dim out of range")
    if config.hidden_dim <= 0:
        raise TorchRLDistillationError("hidden_dim must be > 0")


def _validate_step(step: TensorDictStep, obs_dim: int, action_dim: int) -> None:
    if len(step.obs) != obs_dim:
        raise TorchRLDistillationError(
            f"obs length {len(step.obs)} != configured obs_dim {obs_dim}"
        )
    if len(step.action) != action_dim:
        raise TorchRLDistillationError(
            f"action length {len(step.action)} != configured action_dim {action_dim}"
        )
    if not (math.isfinite(step.sample_log_prob) and math.isfinite(step.state_value)):
        raise TorchRLDistillationError("sample_log_prob and state_value must be finite floats")
    if not math.isfinite(step.reward):
        raise TorchRLDistillationError("reward must be finite")


def _validate_rollouts(
    rollouts: Sequence[CollectorRollout],
    config: ActorCriticConfig,
) -> None:
    if len(rollouts) == 0:
        raise TorchRLDistillationError("at least one rollout required")
    if len(rollouts) > MAX_TRAJECTORIES:
        raise TorchRLDistillationError(f"too many rollouts (>{MAX_TRAJECTORIES})")
    seen_ids: set[str] = set()
    for r in rollouts:
        if not isinstance(r.rollout_id, str) or not r.rollout_id:
            raise TorchRLDistillationError("rollout_id must be a non-empty str")
        if r.rollout_id in seen_ids:
            raise TorchRLDistillationError(f"duplicate rollout_id {r.rollout_id!r}")
        seen_ids.add(r.rollout_id)
        if len(r.steps) == 0:
            raise TorchRLDistillationError(f"rollout {r.rollout_id!r} has zero steps")
        if len(r.steps) > MAX_STEPS_PER_TRAJECTORY:
            raise TorchRLDistillationError(f"rollout {r.rollout_id!r} too long")
        for step in r.steps:
            _validate_step(step, config.obs_dim, config.action_dim)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PolicyDistillationTorchRL:
    """Pure-Python coordinator over a caller-supplied :class:`TorchRLDistiller`."""

    callback: TorchRLDistillationCallback = field(default_factory=null_torchrl_callback)

    def distill(
        self,
        *,
        ts_ns: int,
        proposal_id: str,
        target_strategy: str,
        rollouts: Sequence[CollectorRollout],
        config: ActorCriticConfig,
        seed: int,
        distiller: TorchRLDistiller,
        touchpoints: Sequence[str] = (),
        rationale: str = "",
    ) -> tuple[DistillationResultTorchRL, DistillationProposalTorchRL]:
        if not isinstance(ts_ns, int) or ts_ns < 0:
            raise TorchRLDistillationError("ts_ns must be a non-negative int")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise TorchRLDistillationError("proposal_id must be non-empty str")
        if not isinstance(target_strategy, str) or not target_strategy:
            raise TorchRLDistillationError("target_strategy must be non-empty str")
        if not isinstance(seed, int) or seed < 0:
            raise TorchRLDistillationError("seed must be a non-negative int")
        _validate_config(config)
        rollouts_tuple = tuple(rollouts)
        _validate_rollouts(rollouts_tuple, config)
        self.callback.on_training_start(config)
        metrics, artifact = distiller.train(
            rollouts=rollouts_tuple,
            config=config,
            seed=seed,
            callback=self.callback,
        )
        if artifact.obs_dim != config.obs_dim or artifact.action_dim != config.action_dim:
            raise TorchRLDistillationError("artifact dims do not match config")
        self.callback.on_training_end(metrics)
        config_text = _canonical_config(config)
        rollout_text = _canonical_rollout(rollouts_tuple)
        metrics_text = _canonical_metrics(metrics)
        artifact_text = _canonical_artifact(artifact)
        config_digest = _blake2b_16(config_text)
        rollout_digest = _blake2b_16(rollout_text)
        policy_digest = _blake2b_16(
            f"v={TORCHRL_DISTILLER_VERSION};cfg={config_digest};"
            f"roll={rollout_digest};m={metrics_text};a={artifact_text}"
        )
        result = DistillationResultTorchRL(
            metrics=metrics,
            artifact=artifact,
            policy_digest=policy_digest,
            config_digest=config_digest,
            rollout_digest=rollout_digest,
            meta={"seed": str(seed), "version": TORCHRL_DISTILLER_VERSION},
        )
        proposal = DistillationProposalTorchRL(
            ts_ns=ts_ns,
            proposal_id=proposal_id,
            target_strategy=target_strategy,
            source="learning_engine.lanes.policy_distillation_torchrl",
            touchpoints=tuple(touchpoints),
            rationale=rationale,
            policy_digest=policy_digest,
            config_digest=config_digest,
            rollout_digest=rollout_digest,
            artifact_backend=artifact.backend,
            artifact_digest=artifact.content_digest,
            seed=seed,
            version=TORCHRL_DISTILLER_VERSION,
        )
        return result, proposal


def derive_torchrl_artifact_digest(*, weights_bytes: bytes) -> str:
    if not isinstance(weights_bytes, (bytes, bytearray)):
        raise TorchRLDistillationError("weights_bytes must be bytes")
    return hashlib.blake2b(bytes(weights_bytes), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Lazy factory — production seam
# ---------------------------------------------------------------------------
def torchrl_distiller_factory() -> TorchRLDistiller:  # pragma: no cover
    """Lazy-bind ``torchrl`` + ``tensordict`` + ``torch``.

    Production callers wire this. The function imports the heavy deps
    only at call time so the module can be imported on a host that has
    never installed them. CPU-only deterministic configuration is
    pinned (``torch.backends.cudnn.deterministic = True``,
    ``benchmark = False``).
    """
    try:
        import torch  # noqa: PLC0415
        import torchrl  # noqa: F401, PLC0415
        from tensordict import TensorDict  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "torchrl / tensordict / torch is not installed; see NEW_PIP_DEPENDENCIES"
        ) from exc
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class _LiveDistiller:
        def train(
            self,
            *,
            rollouts: Sequence[CollectorRollout],
            config: ActorCriticConfig,
            seed: int,
            callback: TorchRLDistillationCallback,
        ) -> tuple[TrainingMetrics, TorchRLPolicyArtifact]:
            torch.manual_seed(seed)
            raise NotImplementedError(
                "live TorchRL training loop is a follow-up — production "
                "callers should inject a typed distiller for now"
            )

    return _LiveDistiller()


__all__ = [
    "ActorCriticConfig",
    "CollectorRollout",
    "DistillationProposalTorchRL",
    "DistillationResultTorchRL",
    "MAX_ACTION_DIM",
    "MAX_OBS_DIM",
    "MAX_STEPS_PER_TRAJECTORY",
    "MAX_TRAJECTORIES",
    "NEW_PIP_DEPENDENCIES",
    "PolicyDistillationTorchRL",
    "TORCHRL_DISTILLER_VERSION",
    "TensorDictStep",
    "TorchRLDistillationCallback",
    "TorchRLDistillationError",
    "TorchRLDistiller",
    "TorchRLPolicyArtifact",
    "TrainingMetrics",
    "compute_advantages",
    "derive_torchrl_artifact_digest",
    "null_torchrl_callback",
    "torchrl_distiller_factory",
]
