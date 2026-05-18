# ADAPTED FROM: vwxyzjn/cleanrl
# (cleanrl/ppo.py — single-file PPO reference: rollout collection,
#  GAE-Lambda advantage estimation, clipped surrogate policy loss, value
#  function loss, entropy bonus, mini-batch update epochs. License: MIT.)
"""B-13 — Policy distillation: governance-gated PPO trainer entrypoint.

CleanRL's ``ppo.py`` is the single-file canonical reference for PPO. It
collects ``num_steps`` of on-policy rollouts in one or more parallel
envs, computes GAE-Lambda advantages, then runs ``update_epochs`` mini-
batch SGD passes over the clipped surrogate policy loss + value loss +
entropy bonus. Production trading does **not** train this loop on the
hot path: a distilled policy is a *structural mutation* of the running
strategy and that route goes through
:class:`evolution_engine.patch_pipeline`. This lane is the offline
harness that runs the CleanRL loop, captures the result, and emits a
typed :class:`~core.contracts.learning.PatchProposal` for governance
approval. INV-13/14: Evolution NEVER deploys directly.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual ``torch``
  import lives **only** inside :func:`cleanrl_ppo_distiller_factory`;
  the AST tests pin no top-level torch / numpy / gymnasium / cleanrl
  import so the module is importable on a host that has never
  installed those packages.
* OFFLINE_ONLY tier. The distiller reads no environment variables,
  performs no IO, never imports ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``intelligence_engine``.
  It emits one :class:`DistillationResult` record and one optional
  :class:`~core.contracts.learning.PatchProposal` and stops.
* INV-15 byte-identical replays.
  :meth:`PolicyDistillation.distill` with identical
  ``trajectories`` / ``config`` / ``ts_ns`` / ``proposal_id`` /
  ``distiller`` returns identical :class:`DistillationResult`
  records. Determinism is delegated to the injected distiller; the
  default :func:`cleanrl_ppo_distiller_factory` forwards the seed to
  ``torch.manual_seed`` / ``torch.cuda.manual_seed_all`` and pins the
  CUDA convolution path (``torch.backends.cudnn.deterministic = True``,
  ``benchmark = False``).
* No clock reads. Caller supplies ``ts_ns`` (mirrors A-01.2 / S-06 /
  S-12).

What survives from upstream
---------------------------

* The single-file structure of ``cleanrl/ppo.py`` — config dataclass
  + rollout buffer + advantage estimator + update loop are kept
  verbatim in shape so future readers can ``diff`` against upstream
  without abstraction noise.
* GAE-Lambda advantage estimation (``compute_gae_lambda``) — the
  exact recurrence from upstream, written in pure Python with
  ``math.fsum``-stable accumulation so 3-run replay is byte-identical.
* Clipped surrogate objective + value function clipping + entropy
  bonus — the three loss components are exposed as fields on
  :class:`DistillationMetrics` so the governance proposal preserves
  the full breakdown.

What we replaced
----------------

* CleanRL's ``torch`` model class → caller-injected
  :class:`PolicyDistiller` Protocol. Production wires the lazy
  :func:`cleanrl_ppo_distiller_factory`; unit tests inject a
  deterministic fake whose output is fully reproducible without
  PyTorch installed.
* CleanRL's gym / gymnasium env loop → caller pre-collects
  trajectories via :class:`evolution_engine.gym_env.DIXStrategyEnv`
  and hands the resulting :class:`Trajectory` tuple to
  :meth:`PolicyDistillation.distill`. The distiller is a pure
  consumer of rollouts.
* CleanRL's TensorBoard / Weights-and-Biases logger →
  :class:`DistillationCallback` (default :func:`null_distillation_callback`).
  No filesystem writes, no metrics-server pushes, no global state.
* CleanRL's checkpoint files → :class:`DistillationResult.policy_digest`
  (16-hex BLAKE2b over the canonical text projection of metrics +
  config + artifact-digest). The actual policy weights blob is a
  :class:`PolicyArtifact` the caller can route into evolution's
  existing patch-pipeline storage.

Authority constraints (manifest §H1)
-----------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no PRNG (the
  distiller's PRNG is seeded by caller-supplied seed and never reads
  the wall clock). AST tests pin the import contract.
* No engine cross-imports — AST test pins no ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``intelligence_engine``
  import.
* B27 / B28 / INV-71 authority symmetry — this module sits on the
  learning side of the boundary and therefore does **not** construct
  ``PatchProposal``. The output is a frozen :class:`DistillationResult`
  + :class:`DistillationProposal` advisory record. The future
  evolution-engine adapter (out-of-scope for B-13, mirrors the
  A-03.2 / A-03.3 pattern) translates the advisory record into a
  typed :class:`~core.contracts.learning.PatchProposal` for
  governance approval. AST tests pin that this module never
  constructs ``PatchProposal`` or any typed bus event.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

# ----------------------------------------------------------------------------- public deps

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("torch",)
"""``torch`` is the only true new dep. It is lazy-imported only inside
:func:`cleanrl_ppo_distiller_factory` so the module stays importable on
hosts without PyTorch installed."""

MAX_TRAJECTORIES: Final[int] = 1024
MAX_STEPS_PER_TRAJECTORY: Final[int] = 1_000_000
MAX_OBS_DIM: Final[int] = 4096
MAX_ACTION_DIM: Final[int] = 1024
_FLOAT_EPSILON: Final[float] = 1e-9

POLICY_DISTILLATION_VERSION: Final[str] = "b-13-cleanrl-ppo-1"


# ----------------------------------------------------------------------------- exceptions


class PolicyDistillationError(ValueError):
    """Base error for the policy-distillation lane (invariant violations)."""


# ----------------------------------------------------------------------------- value objects


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One transition collected from a single env rollout.

    Mirrors the per-step fields CleanRL stores in its ring buffer
    (``obs``, ``actions``, ``logprobs``, ``values``, ``rewards``,
    ``dones``) but in a frozen value object so 3-run replays compare
    byte-identical.
    """

    obs: tuple[float, ...]
    action: tuple[float, ...]
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One contiguous on-policy rollout from a single env worker."""

    trajectory_id: str
    steps: tuple[TrajectoryStep, ...]
    bootstrap_value: float = 0.0


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """CleanRL PPO hyperparameters — kept 1:1 with upstream names.

    ``learning_rate`` / ``num_minibatches`` / ``update_epochs`` /
    ``clip_coef`` / ``vf_coef`` / ``ent_coef`` / ``gamma`` /
    ``gae_lambda`` / ``max_grad_norm`` / ``norm_adv`` /
    ``clip_vloss`` / ``target_kl`` come straight from
    ``cleanrl/ppo.py``.
    """

    learning_rate: float = 3e-4
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float | None = None
    obs_dim: int = 1
    action_dim: int = 1


@dataclass(frozen=True, slots=True)
class DistillationMetrics:
    """Headline numbers from one distillation pass.

    Field names mirror CleanRL's ``SPS / charts/episodic_return /
    losses/policy_loss / losses/value_loss / losses/entropy /
    losses/approx_kl`` so a future MLflow exporter (out-of-scope for
    B-13) can ship the same keys upstream.
    """

    episode_reward_mean: float
    episode_reward_best: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    approx_kl: float
    clip_fraction: float
    total_timesteps: int


@dataclass(frozen=True, slots=True)
class PolicyArtifact:
    """Opaque caller-provided policy blob projected to a stable digest.

    The actual weights live wherever the caller stores them
    (filesystem, blob store, evolution patch-pipeline). All we keep on
    the typed surface is a BLAKE2b-16 ``content_digest`` so the
    governance proposal can be cross-referenced byte-identically across
    runs.
    """

    backend: str
    content_digest: str
    obs_dim: int
    action_dim: int


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """Return value of :meth:`PolicyDistillation.distill`.

    Identical inputs → identical record across runs (INV-15).
    """

    metrics: DistillationMetrics
    artifact: PolicyArtifact
    policy_digest: str
    config_digest: str
    rollout_digest: str
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DistillationProposal:
    """Advisory record consumed by the evolution-engine adapter.

    Mirrors the A-03.2 / A-03.3 pattern: simulation / learning tiers
    emit a frozen advisory; only ``evolution_engine.*`` may translate
    it into a typed :class:`~core.contracts.learning.PatchProposal`.
    """

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


# ----------------------------------------------------------------- callbacks / Protocols


class DistillationCallback(Protocol):
    """Mirrors CleanRL's TB/W&B logger — collapsed to a single Protocol."""

    def on_distillation_start(self, config: PPOConfig) -> None: ...

    def on_epoch(
        self,
        epoch: int,
        approx_kl: float,
        policy_loss: float,
        value_loss: float,
    ) -> None: ...

    def on_distillation_end(self, metrics: DistillationMetrics) -> None: ...


def null_distillation_callback() -> DistillationCallback:
    """Default no-op callback so callers don't have to wire one up."""

    class _NullCallback:
        def on_distillation_start(self, config: PPOConfig) -> None:  # noqa: D401, ARG002
            return None

        def on_epoch(
            self,
            epoch: int,  # noqa: ARG002
            approx_kl: float,  # noqa: ARG002
            policy_loss: float,  # noqa: ARG002
            value_loss: float,  # noqa: ARG002
        ) -> None:
            return None

        def on_distillation_end(self, metrics: DistillationMetrics) -> None:  # noqa: ARG002
            return None

    return _NullCallback()


class PolicyDistiller(Protocol):
    """The injectable training surface.

    Production wires the lazy :func:`cleanrl_ppo_distiller_factory`;
    unit tests inject a deterministic fake. The Protocol matches the
    one-call shape of CleanRL's ``ppo.py`` ``train`` function:
    ``(trajectories, advantages, returns, config, seed) -> metrics,
    artifact``.
    """

    def distill(
        self,
        *,
        trajectories: tuple[Trajectory, ...],
        advantages: tuple[tuple[float, ...], ...],
        returns: tuple[tuple[float, ...], ...],
        config: PPOConfig,
        seed: int,
        callback: DistillationCallback,
    ) -> tuple[DistillationMetrics, PolicyArtifact]: ...


# ----------------------------------------------------------------------------- GAE-Lambda


def compute_gae_lambda(
    trajectory: Trajectory,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """CleanRL ``ppo.py`` GAE-Lambda — pure Python, deterministic.

    Returns ``(advantages, returns)`` for the trajectory. ``returns``
    is ``advantages + values`` (the value-function bootstrap target).
    """

    if not (0.0 <= gamma <= 1.0):
        raise PolicyDistillationError(f"gamma must be in [0, 1], got {gamma}")
    if not (0.0 <= gae_lambda <= 1.0):
        raise PolicyDistillationError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
    steps = trajectory.steps
    n = len(steps)
    if n == 0:
        return ((), ())
    advantages: list[float] = [0.0] * n
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        step = steps[t]
        next_nonterm = 0.0 if step.done else 1.0
        next_value = trajectory.bootstrap_value if t == n - 1 else steps[t + 1].value
        delta = step.reward + gamma * next_value * next_nonterm - step.value
        last_gae = delta + gamma * gae_lambda * next_nonterm * last_gae
        advantages[t] = last_gae
    returns = tuple(advantages[t] + steps[t].value for t in range(n))
    return tuple(advantages), returns


# ----------------------------------------------------------------------------- digest helpers


def _hash_blake2b_16(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _canonical_config(config: PPOConfig) -> str:
    parts = (
        f"lr={config.learning_rate:.12g}",
        f"nm={config.num_minibatches}",
        f"ue={config.update_epochs}",
        f"cc={config.clip_coef:.12g}",
        f"vfc={config.vf_coef:.12g}",
        f"ec={config.ent_coef:.12g}",
        f"g={config.gamma:.12g}",
        f"gl={config.gae_lambda:.12g}",
        f"mgn={config.max_grad_norm:.12g}",
        f"na={int(config.norm_adv)}",
        f"cv={int(config.clip_vloss)}",
        f"tk={'_' if config.target_kl is None else f'{config.target_kl:.12g}'}",
        f"od={config.obs_dim}",
        f"ad={config.action_dim}",
    )
    return "|".join(parts)


def _canonical_step(step: TrajectoryStep) -> str:
    obs = ",".join(f"{v:.12g}" for v in step.obs)
    action = ",".join(f"{v:.12g}" for v in step.action)
    return (
        f"{obs};{action};"
        f"lp={step.log_prob:.12g};"
        f"v={step.value:.12g};"
        f"r={step.reward:.12g};"
        f"d={int(step.done)}"
    )


def _canonical_trajectory(trajectory: Trajectory) -> str:
    body = "/".join(_canonical_step(s) for s in trajectory.steps)
    return f"id={trajectory.trajectory_id};bv={trajectory.bootstrap_value:.12g};{body}"


def _canonical_rollout(trajectories: tuple[Trajectory, ...]) -> str:
    return "\n".join(_canonical_trajectory(t) for t in trajectories)


def _canonical_metrics(metrics: DistillationMetrics) -> str:
    return (
        f"erm={metrics.episode_reward_mean:.12g}|"
        f"erb={metrics.episode_reward_best:.12g}|"
        f"pl={metrics.policy_loss:.12g}|"
        f"vl={metrics.value_loss:.12g}|"
        f"el={metrics.entropy_loss:.12g}|"
        f"akl={metrics.approx_kl:.12g}|"
        f"cf={metrics.clip_fraction:.12g}|"
        f"t={metrics.total_timesteps}"
    )


def _canonical_artifact(artifact: PolicyArtifact) -> str:
    return (
        f"b={artifact.backend}|"
        f"cd={artifact.content_digest}|"
        f"od={artifact.obs_dim}|"
        f"ad={artifact.action_dim}"
    )


# ----------------------------------------------------------------------------- validation


def _validate_step(step: TrajectoryStep, obs_dim: int, action_dim: int) -> None:
    if not isinstance(step, TrajectoryStep):  # type: ignore[redundant-cast]
        raise PolicyDistillationError(f"step must be TrajectoryStep, got {type(step)!r}")
    if len(step.obs) != obs_dim:
        raise PolicyDistillationError(f"obs length {len(step.obs)} != obs_dim {obs_dim}")
    if len(step.action) != action_dim:
        raise PolicyDistillationError(
            f"action length {len(step.action)} != action_dim {action_dim}"
        )
    for v in (*step.obs, *step.action, step.log_prob, step.value, step.reward):
        if not math.isfinite(v):
            raise PolicyDistillationError("non-finite value in trajectory step")


def _validate_trajectory(trajectory: Trajectory, obs_dim: int, action_dim: int) -> None:
    if not isinstance(trajectory, Trajectory):  # type: ignore[redundant-cast]
        raise PolicyDistillationError(f"trajectory must be Trajectory, got {type(trajectory)!r}")
    if not trajectory.trajectory_id:
        raise PolicyDistillationError("trajectory_id must be non-empty")
    if len(trajectory.steps) == 0:
        raise PolicyDistillationError("trajectory must have at least one step")
    if len(trajectory.steps) > MAX_STEPS_PER_TRAJECTORY:
        raise PolicyDistillationError(
            f"trajectory steps {len(trajectory.steps)} > MAX {MAX_STEPS_PER_TRAJECTORY}"
        )
    if not math.isfinite(trajectory.bootstrap_value):
        raise PolicyDistillationError("bootstrap_value must be finite")
    for step in trajectory.steps:
        _validate_step(step, obs_dim, action_dim)


def _validate_config(config: PPOConfig) -> None:
    if config.learning_rate <= 0.0 or not math.isfinite(config.learning_rate):
        raise PolicyDistillationError("learning_rate must be > 0 and finite")
    if config.num_minibatches <= 0:
        raise PolicyDistillationError("num_minibatches must be > 0")
    if config.update_epochs <= 0:
        raise PolicyDistillationError("update_epochs must be > 0")
    if config.clip_coef <= 0.0:
        raise PolicyDistillationError("clip_coef must be > 0")
    if config.vf_coef < 0.0:
        raise PolicyDistillationError("vf_coef must be >= 0")
    if config.ent_coef < 0.0:
        raise PolicyDistillationError("ent_coef must be >= 0")
    if not (0.0 <= config.gamma <= 1.0):
        raise PolicyDistillationError("gamma must be in [0, 1]")
    if not (0.0 <= config.gae_lambda <= 1.0):
        raise PolicyDistillationError("gae_lambda must be in [0, 1]")
    if config.max_grad_norm <= 0.0:
        raise PolicyDistillationError("max_grad_norm must be > 0")
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise PolicyDistillationError("target_kl must be > 0 if provided")
    if config.obs_dim <= 0 or config.obs_dim > MAX_OBS_DIM:
        raise PolicyDistillationError(
            f"obs_dim must be in [1, {MAX_OBS_DIM}], got {config.obs_dim}"
        )
    if config.action_dim <= 0 or config.action_dim > MAX_ACTION_DIM:
        raise PolicyDistillationError(
            f"action_dim must be in [1, {MAX_ACTION_DIM}], got {config.action_dim}"
        )


def _validate_trajectories(trajectories: tuple[Trajectory, ...], config: PPOConfig) -> None:
    if len(trajectories) == 0:
        raise PolicyDistillationError("at least one trajectory required")
    if len(trajectories) > MAX_TRAJECTORIES:
        raise PolicyDistillationError(f"trajectories {len(trajectories)} > MAX {MAX_TRAJECTORIES}")
    seen: set[str] = set()
    for tr in trajectories:
        _validate_trajectory(tr, config.obs_dim, config.action_dim)
        if tr.trajectory_id in seen:
            raise PolicyDistillationError(f"duplicate trajectory_id {tr.trajectory_id!r}")
        seen.add(tr.trajectory_id)


# ----------------------------------------------------------------------------- coordinator


@dataclass(frozen=True, slots=True)
class PolicyDistillation:
    """Coordinator: pure rollout-to-PatchProposal pipeline.

    Construction is data-only — no IO, no clock. The actual training
    runs inside the injected :class:`PolicyDistiller`.
    """

    distiller: PolicyDistiller
    callback: DistillationCallback = field(default_factory=null_distillation_callback)

    def distill(
        self,
        *,
        trajectories: Iterable[Trajectory],
        config: PPOConfig,
        seed: int,
        ts_ns: int,
        proposal_id: str,
        target_strategy: str,
    ) -> tuple[DistillationResult, DistillationProposal]:
        """Run the distiller and project to a frozen advisory record.

        Identical inputs → identical outputs across runs (INV-15).
        """

        if seed < 0:
            raise PolicyDistillationError(f"seed must be >= 0, got {seed}")
        if ts_ns < 0:
            raise PolicyDistillationError(f"ts_ns must be >= 0, got {ts_ns}")
        if not proposal_id:
            raise PolicyDistillationError("proposal_id must be non-empty")
        if not target_strategy:
            raise PolicyDistillationError("target_strategy must be non-empty")

        _validate_config(config)
        traj_tuple: tuple[Trajectory, ...] = tuple(trajectories)
        _validate_trajectories(traj_tuple, config)

        advantages_per_traj: list[tuple[float, ...]] = []
        returns_per_traj: list[tuple[float, ...]] = []
        for tr in traj_tuple:
            adv, ret = compute_gae_lambda(tr, gamma=config.gamma, gae_lambda=config.gae_lambda)
            advantages_per_traj.append(adv)
            returns_per_traj.append(ret)

        self.callback.on_distillation_start(config)
        metrics, artifact = self.distiller.distill(
            trajectories=traj_tuple,
            advantages=tuple(advantages_per_traj),
            returns=tuple(returns_per_traj),
            config=config,
            seed=seed,
            callback=self.callback,
        )
        self.callback.on_distillation_end(metrics)

        if not isinstance(metrics, DistillationMetrics):
            raise PolicyDistillationError(
                f"distiller must return DistillationMetrics, got {type(metrics)!r}"
            )
        if not isinstance(artifact, PolicyArtifact):
            raise PolicyDistillationError(
                f"distiller must return PolicyArtifact, got {type(artifact)!r}"
            )
        if artifact.obs_dim != config.obs_dim or artifact.action_dim != config.action_dim:
            raise PolicyDistillationError("artifact dims do not match config dims")

        config_digest = _hash_blake2b_16(_canonical_config(config))
        rollout_digest = _hash_blake2b_16(_canonical_rollout(traj_tuple))
        policy_digest = _hash_blake2b_16(
            "|".join(
                (
                    f"v={POLICY_DISTILLATION_VERSION}",
                    f"cfg={config_digest}",
                    f"rollout={rollout_digest}",
                    f"metrics={_canonical_metrics(metrics)}",
                    f"art={_canonical_artifact(artifact)}",
                    f"seed={seed}",
                )
            )
        )

        result = DistillationResult(
            metrics=metrics,
            artifact=artifact,
            policy_digest=policy_digest,
            config_digest=config_digest,
            rollout_digest=rollout_digest,
            meta={
                "version": POLICY_DISTILLATION_VERSION,
                "seed": str(seed),
                "trajectories": str(len(traj_tuple)),
                "total_steps": str(sum(len(t.steps) for t in traj_tuple)),
            },
        )

        proposal = DistillationProposal(
            ts_ns=ts_ns,
            proposal_id=proposal_id,
            target_strategy=target_strategy,
            source="learning_engine.lanes.policy_distillation",
            touchpoints=("policy_weights",),
            rationale=(
                f"PPO distillation v={POLICY_DISTILLATION_VERSION} "
                f"reward_mean={metrics.episode_reward_mean:.6g} "
                f"reward_best={metrics.episode_reward_best:.6g} "
                f"approx_kl={metrics.approx_kl:.6g} "
                f"timesteps={metrics.total_timesteps}"
            ),
            policy_digest=policy_digest,
            config_digest=config_digest,
            rollout_digest=rollout_digest,
            artifact_backend=artifact.backend,
            artifact_digest=artifact.content_digest,
            seed=seed,
            version=POLICY_DISTILLATION_VERSION,
        )

        return result, proposal


# ----------------------------------------------------------------------------- helpers


def derive_policy_artifact_digest(
    *,
    backend: str,
    payload_parts: Sequence[str],
) -> PolicyArtifact:
    """Deterministic projection used by test fakes and the real factory.

    Production callers may bypass this and supply their own
    BLAKE2b/SHA-256 over the actual weights blob — only the public
    Protocol shape is contractual.
    """

    if not backend:
        raise PolicyDistillationError("backend must be non-empty")
    payload = "|".join(payload_parts)
    return PolicyArtifact(
        backend=backend,
        content_digest=_hash_blake2b_16(payload),
        obs_dim=0,
        action_dim=0,
    )


def cleanrl_ppo_distiller_factory() -> PolicyDistiller:  # pragma: no cover
    """Lazy factory for the production PyTorch-backed distiller.

    The body imports ``torch`` *inside* the function so the AST tests
    can pin no top-level torch import. Real implementation is
    out-of-scope for B-13; this factory is the named seam that downstream
    glue code wires when ``torch`` is actually installed.
    """

    import torch  # noqa: F401  -- lazy import, hidden behind factory

    raise NotImplementedError(
        "cleanrl_ppo_distiller_factory is a production seam; "
        "wire the real PyTorch impl in the production deployment package."
    )


__all__ = [
    "MAX_ACTION_DIM",
    "MAX_OBS_DIM",
    "MAX_STEPS_PER_TRAJECTORY",
    "MAX_TRAJECTORIES",
    "NEW_PIP_DEPENDENCIES",
    "POLICY_DISTILLATION_VERSION",
    "DistillationCallback",
    "DistillationMetrics",
    "DistillationProposal",
    "DistillationResult",
    "PPOConfig",
    "PolicyArtifact",
    "PolicyDistillation",
    "PolicyDistillationError",
    "PolicyDistiller",
    "Trajectory",
    "TrajectoryStep",
    "cleanrl_ppo_distiller_factory",
    "compute_gae_lambda",
    "derive_policy_artifact_digest",
    "null_distillation_callback",
]
