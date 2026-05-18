# ADAPTED FROM: norse/norse — norse/torch/module/lif.py + norse/torch/functional/lif.py
"""B-14 — Spiking Neural Network: Leaky Integrate-and-Fire detector.

Pure-Python LIF neuron + Poisson rate encoder, adapted from the Norse
reference (``norse.torch.module.lif.LIFCell`` and
``norse.torch.functional.lif.lif_feed_forward_step``). The math is the
classical discrete-time LIF recurrence

    v_{t+1} = v_t + (dt / tau_mem) * (v_leak - v_t + R * I_t)
    if v_{t+1} >= v_threshold: spike, v_{t+1} = v_reset

with one extension over the textbook form: a linear synaptic projection
``I_t = W @ x_t + b`` where ``W`` and ``b`` are **frozen** at
construction (INV-20). The default identity / pass-through matrix lets
the detector run as a pure-Python threshold layer with zero external
deps, which is the canonical fallback when no offline-trained weight
file is supplied.

Authority (sensory tier discipline):

* No imports from any engine. No clock reads. No I/O at runtime.
* Outputs are typed value objects only (:class:`SpikePulse`). The
  detector NEVER emits :class:`HazardEvent` — Dyon decides whether a
  spike pulse warrants escalation. This mirrors the existing
  ``dyon_anomaly.AnomalyPulse`` pattern (NEUR-02).
* Weights are **immutable** at runtime (INV-20). The
  :class:`LIFWeights` dataclass is frozen and ``LIFCell`` rejects any
  attempt to swap or mutate them after construction.
* No online learning. There is no ``backward()`` and no autograd. If a
  caller wants to train weights they must do so in an OFFLINE tool and
  pass the frozen tensor in.
* The optional ``torch`` backend (``torch_lif_cell_factory()``) is
  lazy-imported only inside the factory body. Top-level imports are
  pure stdlib so :mod:`sensory.neuromorphic.snn_lif` can be imported in
  any tier — including environments without PyTorch.

Determinism (INV-15):

* No ``random`` / ``time`` / ``datetime`` / ``asyncio`` / ``os``
  imports at module level. The Poisson encoder takes a caller-supplied
  ``seed`` and uses a stateless splitmix64 PRNG.
* All hashing is BLAKE2b-16 over a canonical text projection with
  sorted keys / fixed precision so three runs over the same input
  produce a byte-identical ``weights_digest`` and ``pulse_digest``.

Adapted concepts:

* ``LIFCell`` — Norse ``norse/torch/module/lif.py``: stateful
  forward-pass module with frozen weights and bias.
* ``lif_feed_forward_step()`` — Norse
  ``norse/torch/functional/lif.py``: single discrete-time step of the
  LIF recurrence (potential update + threshold compare + reset).
* ``PoissonEncoder`` — Norse ``norse/torch/module/encode.py``:
  rate-encode a continuous signal to a Poisson spike train. The
  deterministic, pure-Python form here uses a uniform draw from
  splitmix64 + the spike-probability identity ``p = rate * dt``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------- constants

SNN_LIF_VERSION: str = "snn-lif/v1"

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("torch",)
"""Optional production backend (lazy-imported via
``torch_lif_cell_factory``). Pure-Python detector has zero deps."""

MAX_WINDOW: int = 4_096
"""Hard cap on the per-call window length. Above this the detector
refuses input. Keeps single-step inference under the 1ms budget
(INV-26) on the pure-Python path."""

MAX_INPUT_DIM: int = 256
"""Hard cap on the number of feature channels (rows of W)."""

MAX_HIDDEN_DIM: int = 256
"""Hard cap on the number of LIF neurons (columns of W)."""

_DIGEST_BYTES: int = 16

_POLARITY_LONG: str = "LONG"
_POLARITY_SHORT: str = "SHORT"
_POLARITY_NEUTRAL: str = "NEUTRAL"

_POLARITIES: frozenset[str] = frozenset({_POLARITY_LONG, _POLARITY_SHORT, _POLARITY_NEUTRAL})


# ---------------------------------------------------------------- errors


class SNNLIFError(ValueError):
    """Raised for malformed inputs / configuration in the LIF detector."""


# ---------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class LIFConfig:
    """Discrete-time LIF neuron hyperparameters.

    The field names mirror ``norse.torch.functional.lif.LIFParameters``
    so a future port that swaps the inference path for the upstream
    Norse module can pass these straight through.

    Attributes:
        tau_mem: Membrane time constant in **the same time unit as**
            ``dt``. Must be finite and ``> 0``. Norse default ``1e-2``
            (10 ms); we default ``1e-2`` to match.
        v_threshold: Spike threshold. Default ``1.0``.
        v_reset: Resting / reset potential after a spike. Default
            ``0.0``.
        v_leak: Equilibrium potential the neuron relaxes toward in
            absence of input. Default ``0.0``.
        dt: Integration step in the same time unit as ``tau_mem``.
            Default ``1e-3`` (1 ms).
    """

    tau_mem: float = 1.0e-2
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_leak: float = 0.0
    dt: float = 1.0e-3

    def __post_init__(self) -> None:
        if not (math.isfinite(self.tau_mem) and self.tau_mem > 0.0):
            raise SNNLIFError("LIFConfig.tau_mem must be finite and > 0")
        if not math.isfinite(self.v_threshold):
            raise SNNLIFError("LIFConfig.v_threshold must be finite")
        if not math.isfinite(self.v_reset):
            raise SNNLIFError("LIFConfig.v_reset must be finite")
        if not math.isfinite(self.v_leak):
            raise SNNLIFError("LIFConfig.v_leak must be finite")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise SNNLIFError("LIFConfig.dt must be finite and > 0")
        if self.dt > self.tau_mem:
            raise SNNLIFError("LIFConfig.dt must be <= tau_mem for stable integration")


# ---------------------------------------------------------------- weights


@dataclass(frozen=True, slots=True)
class LIFWeights:
    """Frozen synaptic projection ``y = W @ x + b``.

    ``W`` is laid out as ``input_dim`` rows × ``hidden_dim`` columns,
    stored as a tuple-of-tuples for hashability. Both ``W`` and ``b``
    are validated for finiteness at construction; once built, the
    dataclass is immutable so no runtime path can mutate the weights.

    Attributes:
        weight: Row-major synaptic matrix ``[input_dim][hidden_dim]``.
        bias: Per-neuron bias of length ``hidden_dim``.
        input_dim: Number of input features (must equal ``len(weight)``).
        hidden_dim: Number of LIF neurons (must equal
            ``len(weight[i])`` for every row, and ``len(bias)``).
    """

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.input_dim > MAX_INPUT_DIM:
            raise SNNLIFError(f"LIFWeights.input_dim must be in [1, {MAX_INPUT_DIM}]")
        if self.hidden_dim < 1 or self.hidden_dim > MAX_HIDDEN_DIM:
            raise SNNLIFError(f"LIFWeights.hidden_dim must be in [1, {MAX_HIDDEN_DIM}]")
        if len(self.weight) != self.input_dim:
            raise SNNLIFError("LIFWeights.weight row count must equal input_dim")
        for row in self.weight:
            if len(row) != self.hidden_dim:
                raise SNNLIFError("LIFWeights.weight row width must equal hidden_dim")
            for value in row:
                if not math.isfinite(value):
                    raise SNNLIFError("LIFWeights.weight entries must be finite")
        if len(self.bias) != self.hidden_dim:
            raise SNNLIFError("LIFWeights.bias length must equal hidden_dim")
        for value in self.bias:
            if not math.isfinite(value):
                raise SNNLIFError("LIFWeights.bias entries must be finite")

    def digest(self) -> str:
        """Stable 16-hex BLAKE2b digest of (weight, bias, dims).

        Determinism (INV-15): the projection is a fixed-precision
        canonical text form so identical weights produce identical
        digests across machines / Python builds.
        """

        return _digest(_canonical_weights(self))


def identity_weights(dim: int) -> LIFWeights:
    """Build a frozen identity-projection ``LIFWeights`` of ``dim``×``dim``.

    Useful as the canonical "no-op" projection: the detector behaves
    like a pure threshold layer over the raw input channels (one LIF
    neuron per channel, weight ``1.0`` on the diagonal). This is the
    fallback when no offline-trained weight file is supplied.
    """

    if dim < 1 or dim > MAX_HIDDEN_DIM:
        raise SNNLIFError(f"identity_weights: dim must be in [1, {MAX_HIDDEN_DIM}]")
    weight = tuple(tuple(1.0 if i == j else 0.0 for j in range(dim)) for i in range(dim))
    bias = tuple(0.0 for _ in range(dim))
    return LIFWeights(weight=weight, bias=bias, input_dim=dim, hidden_dim=dim)


# ---------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class LIFState:
    """Membrane potential of every LIF neuron.

    Replaces Norse's ``LIFFeedForwardState`` named-tuple. Frozen +
    slotted so the caller cannot accidentally mutate state between
    steps; each call returns a new instance.
    """

    v: tuple[float, ...]

    def __post_init__(self) -> None:
        for value in self.v:
            if not math.isfinite(value):
                raise SNNLIFError("LIFState.v entries must be finite")


def initial_state(hidden_dim: int, *, v_leak: float = 0.0) -> LIFState:
    """Build the rest-potential initial :class:`LIFState`."""

    if hidden_dim < 1 or hidden_dim > MAX_HIDDEN_DIM:
        raise SNNLIFError(f"initial_state: hidden_dim must be in [1, {MAX_HIDDEN_DIM}]")
    if not math.isfinite(v_leak):
        raise SNNLIFError("initial_state: v_leak must be finite")
    return LIFState(v=tuple(v_leak for _ in range(hidden_dim)))


# ---------------------------------------------------------------- output


@dataclass(frozen=True, slots=True)
class SpikePulse:
    """Frozen advisory output of a :class:`LIFCell` over a window.

    Mirrors the existing :class:`~sensory.neuromorphic.contracts.PulseSignal`
    /  :class:`~sensory.neuromorphic.contracts.AnomalyPulse` shape:
    polarity + intensity + sample count + caller-supplied ``ts_ns``.

    The pulse is **advisory only**. Dyon (system engine) decides
    whether a high spike count + intensity warrants escalating to a
    :class:`~core.contracts.events.HazardEvent` — the sensor never
    emits one itself (INV-19, sensor-side replay purity).

    Attributes:
        ts_ns: Window-close timestamp in nanoseconds (caller-supplied,
            INV-15).
        source: Stable source id (e.g. ``"BINANCE"``). Empty rejected.
        symbol: Per-instrument id (e.g. ``"BTCUSDT"``). Empty rejected.
        polarity: ``LONG`` / ``SHORT`` / ``NEUTRAL``. Derived by the
            caller from spike count vs. configured neurons.
        intensity: Fraction of (steps × neurons) that fired, in
            ``[0.0, 1.0]``.
        spike_count: Total spike count across the window.
        sample_count: Window length (``>= 1``).
        weights_digest: BLAKE2b-16 hex of the projection used to
            generate the pulse, for replay-time provenance.
        evidence: Free-form structural metadata.
    """

    ts_ns: int
    source: str
    symbol: str
    polarity: str
    intensity: float
    spike_count: int
    sample_count: int
    weights_digest: str
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise SNNLIFError("SpikePulse.source must be non-empty")
        if not self.symbol:
            raise SNNLIFError("SpikePulse.symbol must be non-empty")
        if self.polarity not in _POLARITIES:
            raise SNNLIFError(f"SpikePulse.polarity must be one of {sorted(_POLARITIES)}")
        if not (math.isfinite(self.intensity) and 0.0 <= self.intensity <= 1.0):
            raise SNNLIFError("SpikePulse.intensity must be finite in [0.0, 1.0]")
        if self.spike_count < 0:
            raise SNNLIFError("SpikePulse.spike_count must be >= 0")
        if self.sample_count < 1:
            raise SNNLIFError("SpikePulse.sample_count must be >= 1")
        if len(self.weights_digest) != _DIGEST_BYTES * 2:
            raise SNNLIFError("SpikePulse.weights_digest must be 16-hex BLAKE2b-16")


# ---------------------------------------------------------------- functional


def lif_feed_forward_step(
    state: LIFState,
    input_current: Sequence[float],
    config: LIFConfig,
) -> tuple[LIFState, tuple[bool, ...]]:
    """One discrete-time LIF step (functional, pure).

    Implements::

        v_next = v + (dt / tau_mem) * (v_leak - v + I)
        spike  = v_next >= v_threshold
        v_next = v_reset if spike else v_next

    Mirrors ``norse/torch/functional/lif.lif_feed_forward_step``: the
    state is a plain frozen container, the parameters are frozen,
    nothing about the call is hidden. Returns a fresh
    :class:`LIFState` plus the boolean spike vector.

    Args:
        state: Current membrane potentials.
        input_current: Synaptic input ``I`` of the same length as
            ``state.v``. Must be finite.
        config: LIF hyperparameters.

    Returns:
        ``(next_state, spikes)`` where ``spikes`` is a tuple of bools
        the same length as ``state.v``.
    """

    if len(input_current) != len(state.v):
        raise SNNLIFError("lif_feed_forward_step: input_current length must equal state.v length")
    decay = config.dt / config.tau_mem
    next_v: list[float] = []
    spikes: list[bool] = []
    for v, i in zip(state.v, input_current, strict=True):
        if not math.isfinite(i):
            raise SNNLIFError("lif_feed_forward_step: input_current entries must be finite")
        v_after = v + decay * (config.v_leak - v + i)
        spiked = v_after >= config.v_threshold
        spikes.append(spiked)
        next_v.append(config.v_reset if spiked else v_after)
    return LIFState(v=tuple(next_v)), tuple(spikes)


def _project(weights: LIFWeights, x: Sequence[float]) -> tuple[float, ...]:
    """Linear projection ``W @ x + b`` for the frozen weight matrix."""

    if len(x) != weights.input_dim:
        raise SNNLIFError("_project: input length must equal weights.input_dim")
    out: list[float] = list(weights.bias)
    for i, xi in enumerate(x):
        if not math.isfinite(xi):
            raise SNNLIFError("_project: input entries must be finite")
        row = weights.weight[i]
        for j, wij in enumerate(row):
            out[j] += wij * xi
    return tuple(out)


# ---------------------------------------------------------------- LIFCell


@dataclass(frozen=True, slots=True)
class LIFCell:
    """Frozen LIF cell: projection ``W @ x + b`` → LIF step.

    Mirrors ``norse.torch.module.lif.LIFCell``. The cell is **frozen**:
    both ``weights`` and ``config`` are immutable dataclasses, and
    ``__setattr__`` is blocked by ``frozen=True``. The pure-Python
    forward pass is intended as a fallback when no PyTorch backend is
    available. The production seam is
    :func:`torch_lif_cell_factory` (lazy import).
    """

    weights: LIFWeights
    config: LIFConfig = field(default_factory=LIFConfig)

    def forward(self, state: LIFState, x: Sequence[float]) -> tuple[LIFState, tuple[bool, ...]]:
        """Single forward step: returns ``(next_state, spikes)``."""

        if len(state.v) != self.weights.hidden_dim:
            raise SNNLIFError("LIFCell.forward: state.v length must equal weights.hidden_dim")
        i = _project(self.weights, x)
        return lif_feed_forward_step(state, i, self.config)


# ---------------------------------------------------------------- encoder


def _splitmix64(state: int) -> tuple[int, int]:
    """Stateless splitmix64 step. Returns ``(uint64_value, next_state)``."""

    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z, state


def _uniform01(state: int) -> tuple[float, int]:
    """Draw a deterministic uniform sample from splitmix64."""

    z, next_state = _splitmix64(state)
    # 53-bit mantissa
    return (z >> 11) * (1.0 / (1 << 53)), next_state


def poisson_encode(
    rates: Sequence[float],
    *,
    n_steps: int,
    dt: float,
    seed: int,
) -> tuple[tuple[bool, ...], ...]:
    """Pure-Python deterministic Poisson rate encoder.

    Adapted from ``norse.torch.module.encode.PoissonEncoder``: every
    step draws ``u ~ U(0, 1)`` per channel and fires a spike if
    ``u < rate * dt``. We use a stateless splitmix64 PRNG seeded with
    ``seed`` so two calls with identical inputs produce byte-identical
    spike trains across machines / Python builds (INV-15).

    Args:
        rates: Per-channel firing rate in Hz (or any unit consistent
            with ``dt``). Must be finite and ``>= 0``.
        n_steps: Number of timesteps. ``>= 1``.
        dt: Integration step in seconds. ``> 0``, finite.
        seed: Deterministic seed for the splitmix64 PRNG.

    Returns:
        Tuple of length ``n_steps``; each element is a tuple of bools
        length ``len(rates)``.
    """

    if n_steps < 1 or n_steps > MAX_WINDOW:
        raise SNNLIFError(f"poisson_encode: n_steps must be in [1, {MAX_WINDOW}]")
    if not (math.isfinite(dt) and dt > 0.0):
        raise SNNLIFError("poisson_encode: dt must be finite and > 0")
    if seed < 0:
        raise SNNLIFError("poisson_encode: seed must be >= 0")
    if len(rates) == 0 or len(rates) > MAX_INPUT_DIM:
        raise SNNLIFError(f"poisson_encode: rates length must be in [1, {MAX_INPUT_DIM}]")
    for r in rates:
        if not (math.isfinite(r) and r >= 0.0):
            raise SNNLIFError("poisson_encode: rate entries must be finite and >= 0")
    state = seed
    out: list[tuple[bool, ...]] = []
    for _ in range(n_steps):
        row: list[bool] = []
        for r in rates:
            u, state = _uniform01(state)
            row.append(u < r * dt)
        out.append(tuple(row))
    return tuple(out)


# ---------------------------------------------------------------- detector


@runtime_checkable
class LIFForwardCallable(Protocol):
    """Anything that exposes a single-step ``forward(state, x)``."""

    def forward(self, state: LIFState, x: Sequence[float]) -> tuple[LIFState, tuple[bool, ...]]: ...


@dataclass(frozen=True, slots=True)
class SNNDetector:
    """Window-level coordinator over a frozen :class:`LIFCell`.

    Consumes a sequence of feature vectors (one per timestep), drives
    the LIF dynamics, aggregates the spike count, and projects the
    result onto an advisory :class:`SpikePulse`. The detector is
    **frozen + advisory only**: it never emits a typed bus event, never
    mutates the registry, and never reads the clock.

    Attributes:
        cell: The frozen :class:`LIFCell` (weights + config).
        spike_polarity_threshold: Fraction in ``[0.0, 1.0]`` controlling
            when the pulse polarity becomes ``LONG`` / ``SHORT``. The
            default ``0.5`` is the geometric centre — half or more
            neurons firing flips polarity from neutral. Subclasses can
            sharpen this gate via a different threshold.
    """

    cell: LIFCell
    spike_polarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.spike_polarity_threshold)
            and 0.0 < self.spike_polarity_threshold <= 1.0
        ):
            raise SNNLIFError("SNNDetector.spike_polarity_threshold must be in (0, 1]")

    def detect(
        self,
        *,
        ts_ns: int,
        source: str,
        symbol: str,
        window: Sequence[Sequence[float]] | Iterable[Sequence[float]],
        evidence: Mapping[str, str] | None = None,
        polarity_sign: int = 1,
    ) -> SpikePulse:
        """Run the LIF cell over ``window`` and emit a :class:`SpikePulse`.

        Args:
            ts_ns: Caller-supplied window-close timestamp.
            source: Stable source id.
            symbol: Per-instrument id.
            window: Sequence of feature vectors, each of length
                ``cell.weights.input_dim``. The detector runs one
                LIF step per row.
            evidence: Optional structural metadata.
            polarity_sign: ``+1`` (default) maps spike intensity to
                ``LONG``; ``-1`` maps to ``SHORT``. ``0`` forces
                ``NEUTRAL`` regardless of activity. The caller (Indira
                / Dyon adapter) decides which channel "means" LONG.

        Returns:
            A frozen advisory :class:`SpikePulse`.
        """

        if ts_ns < 0:
            raise SNNLIFError("SNNDetector.detect: ts_ns must be >= 0")
        if not source:
            raise SNNLIFError("SNNDetector.detect: source must be non-empty")
        if not symbol:
            raise SNNLIFError("SNNDetector.detect: symbol must be non-empty")
        if polarity_sign not in (-1, 0, 1):
            raise SNNLIFError("SNNDetector.detect: polarity_sign must be in {-1, 0, 1}")
        rows = list(window)
        if len(rows) > MAX_WINDOW:
            raise SNNLIFError(f"SNNDetector.detect: window length must be <= {MAX_WINDOW}")
        state = initial_state(self.cell.weights.hidden_dim, v_leak=self.cell.config.v_leak)
        spike_count = 0
        total_neurons = max(1, len(rows)) * self.cell.weights.hidden_dim
        for x in rows:
            state, spikes = self.cell.forward(state, x)
            for s in spikes:
                if s:
                    spike_count += 1
        intensity_raw = spike_count / total_neurons if total_neurons > 0 else 0.0
        intensity = max(0.0, min(1.0, intensity_raw))
        if polarity_sign == 0 or intensity < self.spike_polarity_threshold:
            polarity = _POLARITY_NEUTRAL
        elif polarity_sign > 0:
            polarity = _POLARITY_LONG
        else:
            polarity = _POLARITY_SHORT
        sample_count = max(1, len(rows))
        return SpikePulse(
            ts_ns=ts_ns,
            source=source,
            symbol=symbol,
            polarity=polarity,
            intensity=intensity,
            spike_count=spike_count,
            sample_count=sample_count,
            weights_digest=self.cell.weights.digest(),
            evidence=dict(evidence) if evidence else {},
        )


# ---------------------------------------------------------------- helpers


def _digest(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()


def _canonical_weights(weights: LIFWeights) -> str:
    rows = ";".join(",".join(f"{v:.17g}" for v in row) for row in weights.weight)
    bias = ",".join(f"{v:.17g}" for v in weights.bias)
    return f"v={SNN_LIF_VERSION}|i={weights.input_dim}|h={weights.hidden_dim}|W={rows}|b={bias}"


# ---------------------------------------------------------------- factories


def torch_lif_cell_factory() -> LIFForwardCallable:
    """Lazy production seam over ``norse.torch.module.lif.LIFCell``.

    Imports ``torch`` and ``norse`` inside the function body so the
    pure-Python detector remains usable without either dependency.

    The returned object must expose ``forward(state, x) -> (state,
    spikes)`` consistent with :class:`LIFForwardCallable`. This factory
    intentionally raises :class:`NotImplementedError` — production
    deployments must wire the upstream module after running an
    offline-trained checkpoint through ``requires_grad_(False)`` +
    ``.eval()`` and serialising the frozen weight tensor to disk.
    """

    raise NotImplementedError(
        "torch_lif_cell_factory: production PyTorch backend must be wired "
        "by the deployment package (requires_grad_(False) + .eval() + "
        "torch.save() of frozen weights)"
    )


__all__ = [
    "MAX_HIDDEN_DIM",
    "MAX_INPUT_DIM",
    "MAX_WINDOW",
    "NEW_PIP_DEPENDENCIES",
    "SNN_LIF_VERSION",
    "LIFCell",
    "LIFConfig",
    "LIFForwardCallable",
    "LIFState",
    "LIFWeights",
    "SNNDetector",
    "SNNLIFError",
    "SpikePulse",
    "identity_weights",
    "initial_state",
    "lif_feed_forward_step",
    "poisson_encode",
    "torch_lif_cell_factory",
]
