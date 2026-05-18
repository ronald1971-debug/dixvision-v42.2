# ADAPTED FROM: BindsNET/bindsnet —
#   bindsnet/learning/learning.py (PostPre STDP rule)
#   bindsnet/network/network.py (Network topology + Connection)
#   bindsnet/models/models.py (DiehlAndCook2015 unsupervised SNN)
"""B-15 — STDP-trained spiking neural network for governance-pattern risk.

A pure-Python OFFLINE_ONLY trainer + RUNTIME_SAFE inference detector that
adapts the BindsNET ``PostPre`` spike-timing-dependent-plasticity (STDP)
learning rule and the DiehlAndCook2015 unsupervised topology to the DIX
governance audit perimeter.

Architecture
~~~~~~~~~~~~

Two-layer feed-forward SNN:

* **Input layer** — one channel per governance feature (one-hot row in
  the spike train; e.g. ``approved`` / ``hazard_active`` /
  ``unauthorized_directive`` / ``rejected`` flags projected as 0/1 per
  decision-tick).
* **Hidden / readout layer** — Leaky Integrate-and-Fire neurons whose
  membrane dynamics are identical to :mod:`sensory.neuromorphic.snn_lif`
  (B-14, Norse reference).

Synaptic weights are trained offline against historical governance
event sequences via the PostPre STDP rule:

.. math::

    \\Delta w_{ij}^{LTP} = +\\eta_{post} \\cdot x_{pre,i} \\cdot s_{post,j}
    \\Delta w_{ij}^{LTD} = -\\eta_{pre}  \\cdot x_{post,j} \\cdot s_{pre,i}

where ``x_pre`` / ``x_post`` are exponentially-decaying spike traces:

.. math::

    x \\leftarrow x \\cdot \\exp(-\\Delta t / \\tau) + s

After training, :meth:`GovernanceRiskSNN.detect` runs LIF inference over
a fresh governance window and emits a :class:`RiskPulse` advisory —
**never** a :class:`HazardEvent` (INV-19 / B27 / B28 / INV-71 authority
symmetry). Dyon (the system engine) is the sole authority allowed to
construct typed bus events.

Authority discipline (sensory tier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* No imports from any engine.
* Output is :class:`RiskPulse` (frozen advisory value object) — pinned
  by AST tests.
* Module never constructs :class:`HazardEvent` / :class:`SignalEvent` /
  any typed bus event — pinned by AST tests.
* Weights are **immutable at runtime** (INV-20): both
  :class:`FrozenSNNWeights` and :class:`GovernanceRiskSNN` are
  ``frozen=True``; STDP runs in :func:`stdp_train_offline` and returns
  a *new* :class:`FrozenSNNWeights`. No online weight mutation path.

Determinism (INV-15)
~~~~~~~~~~~~~~~~~~~~

* Pure stdlib only — no ``torch`` / ``bindsnet`` / ``numpy`` imports.
* Caller supplies all timestamps; no ``time`` / ``datetime`` / ``random``
  imports anywhere in this module.
* :meth:`FrozenSNNWeights.digest` uses BLAKE2b-16 over a canonical
  fixed-precision text projection, identical scheme to the B-14 LIF
  detector.
* :func:`stdp_train_offline` is deterministic: same inputs → same
  output weights (no PRNG in the training loop; spike trains are
  deterministic 0/1).
* :meth:`GovernanceRiskSNN.detect` is byte-identical across runs given
  the same window + ts_ns + frozen weights.

Production seam
~~~~~~~~~~~~~~~

:func:`bindsnet_diehl_cook_factory` is a lazy factory that, once wired
by a deployment, would load BindsNET-trained weights via
``torch.save()``. It raises :class:`NotImplementedError` until then so
no top-level ``bindsnet`` / ``torch`` import is required.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sensory.neuromorphic.contracts import RiskPulse

# ===================================================================== version
SNN_GOVERNANCE_VERSION: str = "snn-governance-risk/v1"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("bindsnet",)

# ===================================================================== limits
MAX_WINDOW: int = 4_096
MAX_INPUT_DIM: int = 256
MAX_HIDDEN_DIM: int = 256
MAX_TRAIN_STEPS: int = 100_000
_DIGEST_BYTES: int = 16


# ===================================================================== errors
class SNNGovernanceError(ValueError):
    """Raised on validation failure inside this module."""


# ===================================================================== helpers
def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise SNNGovernanceError(f"{name} must be finite (got {value!r})")


def _check_positive(value: float, name: str) -> None:
    _check_finite(value, name)
    if value <= 0.0:
        raise SNNGovernanceError(f"{name} must be > 0 (got {value!r})")


def _check_nonneg(value: float, name: str) -> None:
    _check_finite(value, name)
    if value < 0.0:
        raise SNNGovernanceError(f"{name} must be >= 0 (got {value!r})")


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


# ===================================================================== configs
@dataclass(frozen=True, slots=True)
class STDPConfig:
    """PostPre STDP hyperparameters (BindsNET names preserved).

    Attributes:
        eta_post: LTP scaling. Δw ← Δw + eta_post * pre_trace when the
            post-synaptic neuron spikes.
        eta_pre: LTD scaling. Δw ← Δw - eta_pre * post_trace when the
            pre-synaptic neuron spikes.
        tau_pre: Pre-synaptic spike-trace decay time-constant (s).
        tau_post: Post-synaptic spike-trace decay time-constant (s).
        w_min: Lower weight clip. Default 0.0 (BindsNET default).
        w_max: Upper weight clip. Default 1.0.
        dt: Discrete time-step (s). Defaults to 1 ms.
    """

    eta_post: float = 1.0e-3
    eta_pre: float = 1.0e-3
    tau_pre: float = 20.0e-3
    tau_post: float = 20.0e-3
    w_min: float = 0.0
    w_max: float = 1.0
    dt: float = 1.0e-3

    def __post_init__(self) -> None:
        _check_nonneg(self.eta_post, "STDPConfig.eta_post")
        _check_nonneg(self.eta_pre, "STDPConfig.eta_pre")
        _check_positive(self.tau_pre, "STDPConfig.tau_pre")
        _check_positive(self.tau_post, "STDPConfig.tau_post")
        _check_finite(self.w_min, "STDPConfig.w_min")
        _check_finite(self.w_max, "STDPConfig.w_max")
        _check_positive(self.dt, "STDPConfig.dt")
        if self.w_min >= self.w_max:
            raise SNNGovernanceError(
                f"STDPConfig.w_min must be < w_max (got {self.w_min} >= {self.w_max})"
            )


@dataclass(frozen=True, slots=True)
class LIFParams:
    """Discrete-time LIF hyperparameters (Norse names preserved).

    Mirrors :class:`sensory.neuromorphic.snn_lif.LIFConfig`.
    """

    tau_mem: float = 20.0e-3
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_leak: float = 0.0
    dt: float = 1.0e-3

    def __post_init__(self) -> None:
        _check_positive(self.tau_mem, "LIFParams.tau_mem")
        _check_finite(self.v_threshold, "LIFParams.v_threshold")
        _check_finite(self.v_reset, "LIFParams.v_reset")
        _check_finite(self.v_leak, "LIFParams.v_leak")
        _check_positive(self.dt, "LIFParams.dt")
        if self.dt > self.tau_mem:
            raise SNNGovernanceError(
                f"LIFParams.dt must be <= tau_mem (got dt={self.dt}, tau_mem={self.tau_mem})"
            )


# ===================================================================== weights
def _canonical_weights(weight: Sequence[Sequence[float]], input_dim: int, hidden_dim: int) -> bytes:
    """Canonical fixed-precision text projection for BLAKE2b digest."""
    parts: list[str] = [f"in={input_dim};out={hidden_dim}"]
    for row in weight:
        parts.append("|".join(f"{v:.17g}" for v in row))
    return "\n".join(parts).encode("utf-8")


@dataclass(frozen=True, slots=True)
class FrozenSNNWeights:
    """Frozen synaptic projection ``W[input_dim, hidden_dim]``.

    INV-20: immutable at runtime. STDP training in
    :func:`stdp_train_offline` returns a *new* instance — the input
    matrix is never mutated in place.
    """

    weight: tuple[tuple[float, ...], ...]
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.input_dim > MAX_INPUT_DIM:
            raise SNNGovernanceError(
                f"FrozenSNNWeights.input_dim must be in [1, {MAX_INPUT_DIM}] (got {self.input_dim})"
            )
        if self.hidden_dim < 1 or self.hidden_dim > MAX_HIDDEN_DIM:
            raise SNNGovernanceError(
                "FrozenSNNWeights.hidden_dim must be in "
                f"[1, {MAX_HIDDEN_DIM}] (got {self.hidden_dim})"
            )
        if len(self.weight) != self.input_dim:
            raise SNNGovernanceError(
                "FrozenSNNWeights.weight must have input_dim rows "
                f"(got {len(self.weight)}, expected {self.input_dim})"
            )
        for i, row in enumerate(self.weight):
            if len(row) != self.hidden_dim:
                raise SNNGovernanceError(
                    f"FrozenSNNWeights.weight row {i} length mismatch "
                    f"(got {len(row)}, expected {self.hidden_dim})"
                )
            for j, v in enumerate(row):
                if not math.isfinite(v):
                    raise SNNGovernanceError(f"FrozenSNNWeights.weight[{i}][{j}] must be finite")

    def digest(self) -> str:
        """BLAKE2b-16 hex of canonical fixed-precision projection."""
        h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
        h.update(_canonical_weights(self.weight, self.input_dim, self.hidden_dim))
        return h.hexdigest()


def identity_governance_weights(input_dim: int, hidden_dim: int) -> FrozenSNNWeights:
    """Canonical "no-op" initial weights — identity-padded zeros.

    Diagonal entries are 1.0 (or 0.5 if ``input_dim != hidden_dim`` so
    the identity collapses to nearest match) and off-diagonal entries
    are 0.0. Useful as initial weights for STDP training when no
    pre-trained checkpoint is available.
    """

    if input_dim < 1 or input_dim > MAX_INPUT_DIM:
        raise SNNGovernanceError(
            f"identity_governance_weights.input_dim must be in "
            f"[1, {MAX_INPUT_DIM}] (got {input_dim})"
        )
    if hidden_dim < 1 or hidden_dim > MAX_HIDDEN_DIM:
        raise SNNGovernanceError(
            f"identity_governance_weights.hidden_dim must be in "
            f"[1, {MAX_HIDDEN_DIM}] (got {hidden_dim})"
        )
    rows: list[tuple[float, ...]] = []
    for i in range(input_dim):
        row = [0.0] * hidden_dim
        if i < hidden_dim:
            row[i] = 1.0
        rows.append(tuple(row))
    return FrozenSNNWeights(weight=tuple(rows), input_dim=input_dim, hidden_dim=hidden_dim)


# ===================================================================== STDP
def _validate_spike_row(
    row: Sequence[bool], expected_dim: int, kind: str, step: int
) -> tuple[bool, ...]:
    if len(row) != expected_dim:
        raise SNNGovernanceError(
            f"{kind} step {step} length mismatch (got {len(row)}, expected {expected_dim})"
        )
    return tuple(bool(s) for s in row)


def stdp_train_offline(
    *,
    initial_weights: FrozenSNNWeights,
    pre_spikes: Sequence[Sequence[bool]],
    post_spikes: Sequence[Sequence[bool]],
    stdp: STDPConfig,
) -> FrozenSNNWeights:
    """Run PostPre STDP over a paired spike train.

    Args:
        initial_weights: Starting weights (frozen). Will not be mutated.
        pre_spikes: ``T x input_dim`` boolean spike train.
        post_spikes: ``T x hidden_dim`` boolean target spike train. Must
            be the same length as ``pre_spikes``.
        stdp: Hyperparameters.

    Returns:
        A *new* :class:`FrozenSNNWeights` with weights updated by the
        PostPre rule and clipped to ``[w_min, w_max]``.

    Raises:
        SNNGovernanceError: On any dimensional / range / finiteness
            violation.
    """

    n_steps = len(pre_spikes)
    if n_steps < 1 or n_steps > MAX_TRAIN_STEPS:
        raise SNNGovernanceError(
            f"stdp_train_offline.pre_spikes must have 1..{MAX_TRAIN_STEPS} steps (got {n_steps})"
        )
    if len(post_spikes) != n_steps:
        raise SNNGovernanceError(
            "stdp_train_offline.post_spikes length must match pre_spikes "
            f"(got {len(post_spikes)} vs {n_steps})"
        )

    input_dim = initial_weights.input_dim
    hidden_dim = initial_weights.hidden_dim

    weight: list[list[float]] = [list(row) for row in initial_weights.weight]
    pre_trace: list[float] = [0.0] * input_dim
    post_trace: list[float] = [0.0] * hidden_dim

    pre_decay = math.exp(-stdp.dt / stdp.tau_pre)
    post_decay = math.exp(-stdp.dt / stdp.tau_post)

    for t in range(n_steps):
        pre_row = _validate_spike_row(pre_spikes[t], input_dim, "pre_spikes", t)
        post_row = _validate_spike_row(post_spikes[t], hidden_dim, "post_spikes", t)

        # Decay traces.
        for i in range(input_dim):
            pre_trace[i] *= pre_decay
        for j in range(hidden_dim):
            post_trace[j] *= post_decay

        # LTP: when post-synaptic neuron j spikes, strengthen connection
        # from every recently-active pre-synaptic input i.
        if stdp.eta_post > 0.0:
            for j in range(hidden_dim):
                if not post_row[j]:
                    continue
                for i in range(input_dim):
                    weight[i][j] = _clip(
                        weight[i][j] + stdp.eta_post * pre_trace[i],
                        stdp.w_min,
                        stdp.w_max,
                    )

        # LTD: when pre-synaptic neuron i spikes, weaken connection to
        # every recently-active post-synaptic output j.
        if stdp.eta_pre > 0.0:
            for i in range(input_dim):
                if not pre_row[i]:
                    continue
                for j in range(hidden_dim):
                    weight[i][j] = _clip(
                        weight[i][j] - stdp.eta_pre * post_trace[j],
                        stdp.w_min,
                        stdp.w_max,
                    )

        # Update traces with current spikes (after delta-w applied).
        for i in range(input_dim):
            if pre_row[i]:
                pre_trace[i] += 1.0
        for j in range(hidden_dim):
            if post_row[j]:
                post_trace[j] += 1.0

    return FrozenSNNWeights(
        weight=tuple(tuple(row) for row in weight),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )


# ===================================================================== LIF step
def _lif_step(
    v: list[float], input_current: Sequence[float], lif: LIFParams
) -> tuple[list[float], tuple[bool, ...]]:
    """One discrete-time LIF step.

    Mirrors :func:`sensory.neuromorphic.snn_lif.lif_feed_forward_step`.
    """

    if len(input_current) != len(v):
        raise SNNGovernanceError(
            f"_lif_step.input_current length mismatch (got {len(input_current)}, expected {len(v)})"
        )
    next_v: list[float] = []
    spikes: list[bool] = []
    leak_factor = lif.dt / lif.tau_mem
    for k in range(len(v)):
        i_k = float(input_current[k])
        if not math.isfinite(i_k):
            raise SNNGovernanceError(f"_lif_step.input_current[{k}] must be finite")
        v_next = v[k] + leak_factor * (lif.v_leak - v[k] + i_k)
        if v_next >= lif.v_threshold:
            spikes.append(True)
            next_v.append(lif.v_reset)
        else:
            spikes.append(False)
            next_v.append(v_next)
    return next_v, tuple(spikes)


# ===================================================================== detector
@dataclass(frozen=True, slots=True)
class GovernanceRiskSNN:
    """Runtime-safe SNN detector — frozen weights, advisory output only.

    The detector runs LIF dynamics over a fresh governance window using
    pre-trained (offline) weights and reports the spike density as a
    risk score in ``[0, 1]``. Construction is the only place where the
    weight matrix can be supplied; both :attr:`weights` and the
    detector itself are frozen (INV-20).
    """

    weights: FrozenSNNWeights
    lif: LIFParams = field(default_factory=LIFParams)
    risk_kind: str = "GOVERNANCE_PATTERN_RISK"

    def __post_init__(self) -> None:
        if not self.risk_kind:
            raise SNNGovernanceError("GovernanceRiskSNN.risk_kind must be non-empty")

    def detect(
        self,
        *,
        ts_ns: int,
        source: str,
        window: Sequence[Sequence[float]] | Iterable[Sequence[float]],
        evidence: Mapping[str, str] | None = None,
        risk_kind: str | None = None,
    ) -> RiskPulse:
        """Run LIF inference over ``window`` and emit a :class:`RiskPulse`.

        Args:
            ts_ns: Caller-supplied window-close timestamp.
            source: Stable source identifier (e.g.
                ``"governance.decision_audit"``).
            window: Sequence of feature vectors of length
                :attr:`weights.input_dim`. The window is consumed in
                order; the LIF state is reset at the start of every
                call so detection is **stateless** across detection
                rounds.
            evidence: Optional structural metadata. ``weights_digest``
                and ``spike_count`` are auto-injected (caller-supplied
                keys with the same names are overwritten).
            risk_kind: Optional override for the emitted
                :attr:`RiskPulse.risk_kind`. Defaults to the detector's
                configured :attr:`risk_kind`.

        Returns:
            :class:`RiskPulse` whose ``risk_score`` is the spike density
            ``spike_count / (n_steps * hidden_dim)``.

        Raises:
            SNNGovernanceError: On any dimensional / range / finiteness
                violation.
        """

        if ts_ns < 0:
            raise SNNGovernanceError(f"GovernanceRiskSNN.detect.ts_ns must be >= 0 (got {ts_ns})")
        if not source:
            raise SNNGovernanceError("GovernanceRiskSNN.detect.source must be non-empty")

        rows: list[Sequence[float]] = list(window)
        n_steps = len(rows)
        if n_steps < 1:
            raise SNNGovernanceError("GovernanceRiskSNN.detect.window must contain at least 1 step")
        if n_steps > MAX_WINDOW:
            raise SNNGovernanceError(
                f"GovernanceRiskSNN.detect.window length must be <= {MAX_WINDOW} (got {n_steps})"
            )

        input_dim = self.weights.input_dim
        hidden_dim = self.weights.hidden_dim
        weight = self.weights.weight

        v: list[float] = [0.0] * hidden_dim
        spike_count = 0

        for t, row in enumerate(rows):
            if len(row) != input_dim:
                raise SNNGovernanceError(
                    f"GovernanceRiskSNN.detect.window[{t}] length mismatch "
                    f"(got {len(row)}, expected {input_dim})"
                )
            # Project input through frozen weights.
            current: list[float] = [0.0] * hidden_dim
            for i in range(input_dim):
                x_i = float(row[i])
                if not math.isfinite(x_i):
                    raise SNNGovernanceError(
                        f"GovernanceRiskSNN.detect.window[{t}][{i}] must be finite"
                    )
                if x_i == 0.0:
                    continue
                w_row = weight[i]
                for j in range(hidden_dim):
                    current[j] += w_row[j] * x_i
            v, spikes = _lif_step(v, current, self.lif)
            spike_count += sum(1 for s in spikes if s)

        max_spikes = n_steps * hidden_dim
        risk_score = spike_count / max_spikes if max_spikes > 0 else 0.0
        if risk_score > 1.0:  # safety against floating-point slop
            risk_score = 1.0

        emitted_kind = risk_kind if risk_kind is not None else self.risk_kind
        if not emitted_kind:
            raise SNNGovernanceError("GovernanceRiskSNN.detect.risk_kind must be non-empty")

        merged_evidence: dict[str, str] = {}
        if evidence is not None:
            for k in sorted(evidence.keys()):
                merged_evidence[k] = str(evidence[k])
        merged_evidence["weights_digest"] = self.weights.digest()
        merged_evidence["spike_count"] = str(spike_count)
        merged_evidence["hidden_dim"] = str(hidden_dim)

        return RiskPulse(
            ts_ns=ts_ns,
            source=source,
            risk_kind=emitted_kind,
            risk_score=risk_score,
            sample_count=n_steps,
            evidence=merged_evidence,
        )


# ===================================================================== production seam
class BindsNetSNNFactory(Protocol):
    """Production seam — load a BindsNET-trained network at deploy time.

    A deployment may supply an implementation that:

    1. Calls ``torch.load(weights_path)`` to materialise the trained
       network.
    2. Sets ``requires_grad_(False)`` on every parameter and freezes the
       network with ``model.eval()``.
    3. Projects the network's weight tensor into a
       :class:`FrozenSNNWeights` value object and returns a
       :class:`GovernanceRiskSNN` bound to it.

    Until a deployment wires this seam, calling
    :func:`bindsnet_diehl_cook_factory` raises
    :class:`NotImplementedError`. This keeps the sensory-tier import
    surface zero-dependency at runtime.
    """

    def __call__(self, weights_path: str) -> GovernanceRiskSNN: ...


def bindsnet_diehl_cook_factory(weights_path: str) -> GovernanceRiskSNN:
    """Lazy production factory for a BindsNET-trained governance SNN.

    Raises:
        NotImplementedError: Always. Deployments must replace this
            symbol with a concrete factory implementation.
    """

    raise NotImplementedError(
        "bindsnet_diehl_cook_factory is a production seam. A deployment "
        "must wire BindsNET checkpoint loading + requires_grad_(False) + "
        f"model.eval() before consuming weights_path={weights_path!r}."
    )


# ===================================================================== exports
__all__ = [
    "MAX_HIDDEN_DIM",
    "MAX_INPUT_DIM",
    "MAX_TRAIN_STEPS",
    "MAX_WINDOW",
    "NEW_PIP_DEPENDENCIES",
    "SNN_GOVERNANCE_VERSION",
    "BindsNetSNNFactory",
    "FrozenSNNWeights",
    "GovernanceRiskSNN",
    "LIFParams",
    "STDPConfig",
    "SNNGovernanceError",
    "bindsnet_diehl_cook_factory",
    "identity_governance_weights",
    "stdp_train_offline",
]
