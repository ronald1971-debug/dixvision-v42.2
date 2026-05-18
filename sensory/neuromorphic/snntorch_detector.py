# ADAPTED FROM: jeshraghian/snntorch —
#   snntorch/_neurons/leaky.py  (Leaky.forward, multiplicative-decay LIF)
#   snntorch/functional/loss.py (SpikeCountLoss — kept as offline reference)
#   snntorch/surrogate.py       (fast_sigmoid — kept as offline reference)
"""B-17 — snnTorch alternative spiking neural network backend.

Pure-Python adaptation of snnTorch's :class:`snntorch.Leaky` neuron, the
canonical alternative to the Norse :class:`LIFCell` shipped in B-14.
Unlike the Norse forward-Euler recurrence

    v_{t+1} = v_t + (dt / tau_mem) * (v_leak - v_t + I_t)

snnTorch's ``Leaky`` integrates the LIF ODE with a **multiplicative
decay** factor and a **subtractive** spike reset

    v_{t+1} = beta * v_t + I_t
    spike   = v_{t+1} >= v_threshold
    v_{t+1} = v_{t+1} - spike * v_threshold       # SUBTRACT reset
       OR
    v_{t+1} = v_{t+1} * (1 - spike)               # ZERO reset

where ``beta = exp(-dt / tau_mem)`` is the exact discrete-time solution
to the homogeneous LIF equation between events. This gives different
numerical properties from Norse (smaller integration error at large
``dt``, slightly different spike timing) which is exactly why the
spec wants both backends present: production deploys whichever has
better precision per the per-symbol benchmark (see
:mod:`tests.bench.test_snn_backend_comparison`).

Authority (sensory tier discipline):

* No imports from any engine. No clock reads. No I/O at runtime.
* Output is a frozen value object (:class:`SpikePulse` re-exported
  from :mod:`sensory.neuromorphic.snn_lif`). The detector NEVER emits
  :class:`HazardEvent` — Dyon decides whether activity warrants
  escalation. Mirrors B-14 and the existing
  :class:`~sensory.neuromorphic.dyon_anomaly.AnomalyPulse` pattern
  (NEUR-02 / INV-19).
* Weights are **immutable** at runtime (INV-20). Frozen dataclasses
  + identity-pass-through fallback when no offline-trained weight
  file is supplied. There is no ``backward()`` and no autograd. Any
  training MUST happen in an offline tool that then hands a frozen
  tensor to the detector.
* The optional ``snntorch`` / ``torch`` backend
  (:func:`snntorch_cell_factory`) is lazy-imported only inside the
  factory body. Top-level imports are pure stdlib so this module can
  be imported in any tier — including environments without snnTorch.

Determinism (INV-15):

* No ``random`` / ``time`` / ``datetime`` / ``asyncio`` / ``os``
  imports at module level. Identical inputs produce byte-identical
  outputs.
* All hashing is BLAKE2b-16 over a canonical text projection with
  sorted keys / fixed precision so three runs over the same input
  produce a byte-identical ``weights_digest`` and ``pulse_digest``.

Adapted concepts:

* :class:`SNNTorchLeakyCell` — snnTorch ``snntorch/_neurons/leaky.py``:
  stateful forward-pass module with multiplicative decay ``beta`` and
  subtractive (or zero) spike reset.
* :class:`SNNTorchDetector` — window-level coordinator. Consumes a
  pre-projected current trace and emits a frozen :class:`SpikePulse`,
  re-using the same contract that B-14's Norse :class:`SNNDetector`
  emits so the runtime can swap backends transparently.
* ``benchmark_against_norse()`` — deterministic spike-count / timing
  comparator. Mirrors the canonical benchmark gate described in spec
  line 2206.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------- constants

SNNTORCH_DETECTOR_VERSION: str = "snntorch-detector/v1"

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("snntorch", "torch")
"""Production lazy backends. The pure-Python detector has zero deps."""

MAX_WINDOW: int = 4_096
"""Hard cap on the per-call window length. Same as B-14 to keep the
benchmark fair."""

MAX_INPUT_DIM: int = 256
MAX_HIDDEN_DIM: int = 256

_DIGEST_BYTES: int = 16

_POLARITY_LONG: str = "LONG"
_POLARITY_SHORT: str = "SHORT"
_POLARITY_NEUTRAL: str = "NEUTRAL"

_POLARITIES: frozenset[str] = frozenset({_POLARITY_LONG, _POLARITY_SHORT, _POLARITY_NEUTRAL})

RESET_SUBTRACT: str = "SUBTRACT"
"""snnTorch ``reset_mechanism="subtract"``: ``v -= spike * v_threshold``."""

RESET_ZERO: str = "ZERO"
"""snnTorch ``reset_mechanism="zero"``:  ``v *= (1 - spike)``."""

_RESET_MECHANISMS: frozenset[str] = frozenset({RESET_SUBTRACT, RESET_ZERO})


# ---------------------------------------------------------------- errors


class SNNTorchDetectorError(ValueError):
    """Raised for malformed inputs / configuration in the snnTorch detector."""


# ---------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class LeakyConfig:
    """snnTorch ``Leaky`` hyperparameters.

    The field names mirror :class:`snntorch.Leaky` so a future port
    that swaps the inference path for the upstream module can pass
    these straight through.

    Attributes:
        beta: Multiplicative decay factor in ``(0.0, 1.0)``. Equivalent
            to ``exp(-dt / tau_mem)``. snnTorch's reference default is
            ``0.9``.
        v_threshold: Spike threshold. Default ``1.0``.
        v_reset: Resting / reset offset used by ``RESET_ZERO``. The
            ``RESET_SUBTRACT`` path ignores this and subtracts
            ``v_threshold`` instead (matches snnTorch's default).
            Default ``0.0``.
        reset_mechanism: One of :data:`RESET_SUBTRACT` or
            :data:`RESET_ZERO`. snnTorch's default is ``"subtract"``;
            we mirror that.
    """

    beta: float = 0.9
    v_threshold: float = 1.0
    v_reset: float = 0.0
    reset_mechanism: str = RESET_SUBTRACT

    def __post_init__(self) -> None:
        if not (math.isfinite(self.beta) and 0.0 < self.beta < 1.0):
            raise SNNTorchDetectorError(
                f"LeakyConfig.beta must be in (0.0, 1.0), got {self.beta!r}"
            )
        if not math.isfinite(self.v_threshold):
            raise SNNTorchDetectorError("LeakyConfig.v_threshold must be finite")
        if not math.isfinite(self.v_reset):
            raise SNNTorchDetectorError("LeakyConfig.v_reset must be finite")
        if self.reset_mechanism not in _RESET_MECHANISMS:
            raise SNNTorchDetectorError(
                "LeakyConfig.reset_mechanism must be one of "
                f"{sorted(_RESET_MECHANISMS)}, got {self.reset_mechanism!r}"
            )


def beta_from_tau(dt: float, tau_mem: float) -> float:
    """Convert ``dt`` / ``tau_mem`` to the exact ``beta = exp(-dt/tau)``.

    snnTorch users typically specify ``beta`` directly; this helper
    lets a caller wired through B-14 reuse the Norse ``(dt, tau_mem)``
    parameters without re-deriving the decay constant.
    """

    if not (math.isfinite(dt) and dt > 0.0):
        raise SNNTorchDetectorError("beta_from_tau: dt must be finite and > 0")
    if not (math.isfinite(tau_mem) and tau_mem > 0.0):
        raise SNNTorchDetectorError("beta_from_tau: tau_mem must be finite and > 0")
    return math.exp(-dt / tau_mem)


# ---------------------------------------------------------------- weights


@dataclass(frozen=True, slots=True)
class LeakyWeights:
    """Frozen synaptic projection ``y = W @ x + b`` for snnTorch ``Leaky``.

    Identical contract shape to B-14's :class:`LIFWeights` so the two
    backends can be benchmarked head-to-head on the same offline
    checkpoint. Stored as tuple-of-tuples for hashability.

    Attributes:
        weight: Row-major synaptic matrix ``[input_dim][hidden_dim]``.
        bias: Per-neuron bias of length ``hidden_dim``.
        input_dim: Number of input features.
        hidden_dim: Number of Leaky neurons.
    """

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.input_dim > MAX_INPUT_DIM:
            raise SNNTorchDetectorError(f"LeakyWeights.input_dim must be in [1, {MAX_INPUT_DIM}]")
        if self.hidden_dim < 1 or self.hidden_dim > MAX_HIDDEN_DIM:
            raise SNNTorchDetectorError(f"LeakyWeights.hidden_dim must be in [1, {MAX_HIDDEN_DIM}]")
        if len(self.weight) != self.input_dim:
            raise SNNTorchDetectorError("LeakyWeights.weight row count must equal input_dim")
        for row in self.weight:
            if len(row) != self.hidden_dim:
                raise SNNTorchDetectorError("LeakyWeights.weight row width must equal hidden_dim")
            for value in row:
                if not math.isfinite(value):
                    raise SNNTorchDetectorError("LeakyWeights.weight entries must be finite")
        if len(self.bias) != self.hidden_dim:
            raise SNNTorchDetectorError("LeakyWeights.bias length must equal hidden_dim")
        for value in self.bias:
            if not math.isfinite(value):
                raise SNNTorchDetectorError("LeakyWeights.bias entries must be finite")

    def digest(self) -> str:
        """Stable 16-hex BLAKE2b-16 digest of (weight, bias, dims)."""

        return _digest(_canonical_weights(self))


def identity_leaky_weights(dim: int) -> LeakyWeights:
    """Identity-projection :class:`LeakyWeights` (``dim``×``dim``).

    Canonical no-op projection: behaves as a pure threshold layer over
    the raw input channels with one Leaky neuron per channel.
    """

    if dim < 1 or dim > MAX_HIDDEN_DIM:
        raise SNNTorchDetectorError(f"identity_leaky_weights: dim must be in [1, {MAX_HIDDEN_DIM}]")
    weight = tuple(tuple(1.0 if i == j else 0.0 for j in range(dim)) for i in range(dim))
    bias = tuple(0.0 for _ in range(dim))
    return LeakyWeights(weight=weight, bias=bias, input_dim=dim, hidden_dim=dim)


# ---------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class LeakyState:
    """Membrane potential of every Leaky neuron.

    Same shape as B-14's :class:`LIFState` (a tuple of floats), kept
    as a distinct dataclass so type-checkers catch accidental cross-
    backend mixing.
    """

    v: tuple[float, ...]

    def __post_init__(self) -> None:
        for value in self.v:
            if not math.isfinite(value):
                raise SNNTorchDetectorError("LeakyState.v entries must be finite")


def initial_leaky_state(hidden_dim: int, *, v_init: float = 0.0) -> LeakyState:
    """Build the initial :class:`LeakyState` with all neurons at ``v_init``."""

    if hidden_dim < 1 or hidden_dim > MAX_HIDDEN_DIM:
        raise SNNTorchDetectorError(
            f"initial_leaky_state: hidden_dim must be in [1, {MAX_HIDDEN_DIM}]"
        )
    if not math.isfinite(v_init):
        raise SNNTorchDetectorError("initial_leaky_state: v_init must be finite")
    return LeakyState(v=tuple(v_init for _ in range(hidden_dim)))


# ---------------------------------------------------------------- output


@dataclass(frozen=True, slots=True)
class SpikePulse:
    """Frozen advisory output of an :class:`SNNTorchDetector`.

    Mirrors B-14's ``sensory.neuromorphic.snn_lif.SpikePulse`` exactly
    so the runtime can swap LIF backends without changing downstream
    consumers. Carries the snnTorch ``weights_digest`` so the
    benchmark layer can confirm both backends ran on the same
    offline-trained checkpoint.

    Attributes:
        ts_ns: Caller-supplied window-close timestamp.
        source: Stable source id.
        symbol: Per-instrument id.
        polarity: One of ``LONG`` / ``SHORT`` / ``NEUTRAL``.
        intensity: Spike-density in ``[0.0, 1.0]``.
        spike_count: Total spikes across all neurons + steps.
        sample_count: Number of windowed steps consumed.
        weights_digest: 16-hex BLAKE2b-16 over the canonical weight
            projection.
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
            raise SNNTorchDetectorError("SpikePulse.source must be non-empty")
        if not self.symbol:
            raise SNNTorchDetectorError("SpikePulse.symbol must be non-empty")
        if self.polarity not in _POLARITIES:
            raise SNNTorchDetectorError(f"SpikePulse.polarity must be one of {sorted(_POLARITIES)}")
        if not (math.isfinite(self.intensity) and 0.0 <= self.intensity <= 1.0):
            raise SNNTorchDetectorError("SpikePulse.intensity must be finite in [0.0, 1.0]")
        if self.spike_count < 0:
            raise SNNTorchDetectorError("SpikePulse.spike_count must be >= 0")
        if self.sample_count < 1:
            raise SNNTorchDetectorError("SpikePulse.sample_count must be >= 1")
        if len(self.weights_digest) != _DIGEST_BYTES * 2:
            raise SNNTorchDetectorError("SpikePulse.weights_digest must be 16-hex BLAKE2b-16")


# ---------------------------------------------------------------- functional


def leaky_feed_forward_step(
    state: LeakyState,
    input_current: Sequence[float],
    config: LeakyConfig,
) -> tuple[LeakyState, tuple[bool, ...]]:
    """One discrete-time snnTorch ``Leaky`` step (functional, pure).

    Implements snnTorch's ``Leaky.forward``::

        v_next = beta * v + I
        spike  = v_next >= v_threshold
        if reset_mechanism == SUBTRACT:
            v_next -= spike * v_threshold
        else:  # ZERO
            v_next *= (1 - spike)

    Args:
        state: Current membrane potentials.
        input_current: Synaptic input ``I`` of the same length as
            ``state.v``. Must be finite.
        config: Leaky hyperparameters.

    Returns:
        ``(next_state, spikes)``.
    """

    if len(input_current) != len(state.v):
        raise SNNTorchDetectorError(
            "leaky_feed_forward_step: input_current length must equal state.v length"
        )
    next_v: list[float] = []
    spikes: list[bool] = []
    use_subtract = config.reset_mechanism == RESET_SUBTRACT
    for v, i in zip(state.v, input_current, strict=True):
        if not math.isfinite(i):
            raise SNNTorchDetectorError(
                "leaky_feed_forward_step: input_current entries must be finite"
            )
        v_after = config.beta * v + i
        spiked = v_after >= config.v_threshold
        spikes.append(spiked)
        if spiked:
            if use_subtract:
                v_after = v_after - config.v_threshold
            else:
                v_after = 0.0
        next_v.append(v_after)
    return LeakyState(v=tuple(next_v)), tuple(spikes)


def _project(weights: LeakyWeights, x: Sequence[float]) -> tuple[float, ...]:
    """Linear projection ``W @ x + b`` for the frozen weight matrix."""

    if len(x) != weights.input_dim:
        raise SNNTorchDetectorError("_project: input length must equal weights.input_dim")
    out: list[float] = list(weights.bias)
    for i, xi in enumerate(x):
        if not math.isfinite(xi):
            raise SNNTorchDetectorError("_project: input entries must be finite")
        row = weights.weight[i]
        for j, wij in enumerate(row):
            out[j] += wij * xi
    return tuple(out)


# ---------------------------------------------------------------- cell


@dataclass(frozen=True, slots=True)
class SNNTorchLeakyCell:
    """Frozen Leaky cell: projection ``W @ x + b`` → snnTorch ``Leaky`` step.

    Mirrors :class:`snntorch.Leaky`. The cell is **frozen**: both
    ``weights`` and ``config`` are immutable dataclasses and
    ``__setattr__`` is blocked by ``frozen=True``. The pure-Python
    forward pass is intended as a fallback when no snnTorch / torch
    backend is available. The production seam is
    :func:`snntorch_cell_factory` (lazy import).
    """

    weights: LeakyWeights
    config: LeakyConfig = field(default_factory=LeakyConfig)

    def forward(self, state: LeakyState, x: Sequence[float]) -> tuple[LeakyState, tuple[bool, ...]]:
        """Single forward step: ``(next_state, spikes)``."""

        if len(state.v) != self.weights.hidden_dim:
            raise SNNTorchDetectorError(
                "SNNTorchLeakyCell.forward: state.v length must equal weights.hidden_dim"
            )
        i = _project(self.weights, x)
        return leaky_feed_forward_step(state, i, self.config)


# ---------------------------------------------------------------- detector


@runtime_checkable
class LeakyForwardCallable(Protocol):
    """Anything that exposes a single-step ``forward(state, x)``."""

    def forward(
        self, state: LeakyState, x: Sequence[float]
    ) -> tuple[LeakyState, tuple[bool, ...]]: ...


@dataclass(frozen=True, slots=True)
class SNNTorchDetector:
    """Window-level coordinator over a frozen :class:`SNNTorchLeakyCell`.

    Consumes a sequence of feature vectors, drives the Leaky dynamics,
    aggregates spike count, projects onto an advisory
    :class:`SpikePulse`. Frozen + advisory only — never emits a typed
    bus event, never mutates the registry, never reads the clock.

    Attributes:
        cell: The frozen :class:`SNNTorchLeakyCell`.
        spike_polarity_threshold: Fraction in ``(0.0, 1.0]`` controlling
            when polarity becomes ``LONG`` / ``SHORT``. Default ``0.5``.
    """

    cell: SNNTorchLeakyCell
    spike_polarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.spike_polarity_threshold)
            and 0.0 < self.spike_polarity_threshold <= 1.0
        ):
            raise SNNTorchDetectorError(
                "SNNTorchDetector.spike_polarity_threshold must be in (0, 1]"
            )

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
        """Run the Leaky cell over ``window`` and emit a :class:`SpikePulse`."""

        if ts_ns < 0:
            raise SNNTorchDetectorError("SNNTorchDetector.detect: ts_ns must be >= 0")
        if not source:
            raise SNNTorchDetectorError("SNNTorchDetector.detect: source must be non-empty")
        if not symbol:
            raise SNNTorchDetectorError("SNNTorchDetector.detect: symbol must be non-empty")
        if polarity_sign not in (-1, 0, 1):
            raise SNNTorchDetectorError(
                "SNNTorchDetector.detect: polarity_sign must be in {-1, 0, 1}"
            )
        rows = list(window)
        if len(rows) > MAX_WINDOW:
            raise SNNTorchDetectorError(
                f"SNNTorchDetector.detect: window length must be <= {MAX_WINDOW}"
            )
        state = initial_leaky_state(self.cell.weights.hidden_dim)
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


# ---------------------------------------------------------------- benchmark


@dataclass(frozen=True, slots=True)
class BackendBenchmark:
    """Frozen result of head-to-head backend comparison.

    The benchmark is the canonical promotion gate described in spec
    line 2206: "Deploy whichever gives better precision per benchmark
    results." Both backends consume the same projected current trace
    so the numerical difference is purely the integration scheme
    (forward-Euler vs multiplicative decay).

    Attributes:
        norse_spike_count: Total spikes from Norse-style integration.
        snntorch_spike_count: Total spikes from snnTorch-style
            integration.
        first_spike_step_norse: Step index of the first spike on the
            Norse path (``-1`` if silent).
        first_spike_step_snntorch: Step index of the first spike on
            the snnTorch path (``-1`` if silent).
        spike_count_delta: ``snntorch_spike_count - norse_spike_count``.
        first_spike_step_delta: Absolute difference between the two
            first-spike-step values when both are non-negative. ``-1``
            when one side never spiked.
        digest: Stable 16-hex BLAKE2b-16 over a canonical projection.
    """

    norse_spike_count: int
    snntorch_spike_count: int
    first_spike_step_norse: int
    first_spike_step_snntorch: int
    spike_count_delta: int
    first_spike_step_delta: int
    digest: str

    def __post_init__(self) -> None:
        if self.norse_spike_count < 0:
            raise SNNTorchDetectorError("BackendBenchmark.norse_spike_count must be >= 0")
        if self.snntorch_spike_count < 0:
            raise SNNTorchDetectorError("BackendBenchmark.snntorch_spike_count must be >= 0")
        if len(self.digest) != _DIGEST_BYTES * 2:
            raise SNNTorchDetectorError("BackendBenchmark.digest must be 16-hex BLAKE2b-16")

    def is_precision_match(
        self,
        *,
        count_tolerance: int = 0,
        first_spike_step_tolerance: int = 0,
    ) -> bool:
        """True iff both backends agree within the supplied tolerances.

        Used by the per-symbol promotion gate to decide whether the two
        integrators are interchangeable on this regime. When they
        disagree, the promotion script picks whichever has the better
        scoring against an offline ground-truth (spike-time-vs-event
        loss, see :class:`snntorch.functional.loss.SpikeCountLoss`).
        """

        if count_tolerance < 0:
            raise SNNTorchDetectorError("is_precision_match: count_tolerance must be >= 0")
        if first_spike_step_tolerance < 0:
            raise SNNTorchDetectorError(
                "is_precision_match: first_spike_step_tolerance must be >= 0"
            )
        if abs(self.spike_count_delta) > count_tolerance:
            return False
        if self.first_spike_step_norse < 0 or self.first_spike_step_snntorch < 0:
            return self.first_spike_step_norse == self.first_spike_step_snntorch
        return self.first_spike_step_delta <= first_spike_step_tolerance


def _norse_reference_step(
    v: float, i: float, *, dt: float, tau_mem: float, v_threshold: float
) -> tuple[float, bool]:
    """Norse-style forward-Euler LIF step for the benchmark integrator.

    Mirrors :func:`sensory.neuromorphic.snn_lif.lif_feed_forward_step`
    on a single neuron, with reset-to-zero. Kept local so the benchmark
    runs without importing B-14 (the spec wants snnTorch testable in
    isolation before Norse lands on main).
    """

    v_after = v + (dt / tau_mem) * (0.0 - v + i)
    spiked = v_after >= v_threshold
    return (0.0 if spiked else v_after), spiked


def benchmark_against_norse(
    *,
    input_current: Sequence[float],
    dt: float = 1.0e-3,
    tau_mem: float = 1.0e-2,
    v_threshold: float = 1.0,
    reset_mechanism: str = RESET_SUBTRACT,
) -> BackendBenchmark:
    """Head-to-head spike-count + first-spike-step comparison.

    Runs a single-neuron Norse forward-Euler LIF and a single-neuron
    snnTorch ``Leaky`` over identical input current. Both use the
    exact same ``dt`` / ``tau_mem`` / ``v_threshold`` so the
    numerical difference is the integration scheme only.

    Args:
        input_current: One-channel current trace ``I_t``. Must be
            finite and non-empty.
        dt: Integration step. Default ``1e-3``.
        tau_mem: Membrane time constant. Default ``1e-2``.
        v_threshold: Spike threshold. Default ``1.0``.
        reset_mechanism: snnTorch reset semantics:
            :data:`RESET_SUBTRACT` (default) or :data:`RESET_ZERO`.

    Returns:
        A frozen :class:`BackendBenchmark`.
    """

    if not (math.isfinite(dt) and dt > 0.0):
        raise SNNTorchDetectorError("benchmark_against_norse: dt must be finite and > 0")
    if not (math.isfinite(tau_mem) and tau_mem > 0.0):
        raise SNNTorchDetectorError("benchmark_against_norse: tau_mem must be finite and > 0")
    if not math.isfinite(v_threshold):
        raise SNNTorchDetectorError("benchmark_against_norse: v_threshold must be finite")
    if reset_mechanism not in _RESET_MECHANISMS:
        raise SNNTorchDetectorError(
            f"benchmark_against_norse: reset_mechanism must be one of {sorted(_RESET_MECHANISMS)}"
        )
    rows = list(input_current)
    if not rows:
        raise SNNTorchDetectorError("benchmark_against_norse: input_current must be non-empty")
    if len(rows) > MAX_WINDOW:
        raise SNNTorchDetectorError(
            f"benchmark_against_norse: trace length must be <= {MAX_WINDOW}"
        )
    for value in rows:
        if not math.isfinite(value):
            raise SNNTorchDetectorError("benchmark_against_norse: input entries must be finite")

    # Norse forward-Euler reference
    norse_v = 0.0
    norse_first = -1
    norse_count = 0
    for step, i in enumerate(rows):
        norse_v, spiked = _norse_reference_step(
            norse_v,
            i,
            dt=dt,
            tau_mem=tau_mem,
            v_threshold=v_threshold,
        )
        if spiked:
            norse_count += 1
            if norse_first < 0:
                norse_first = step

    # snnTorch Leaky
    beta = beta_from_tau(dt, tau_mem)
    config = LeakyConfig(
        beta=beta,
        v_threshold=v_threshold,
        v_reset=0.0,
        reset_mechanism=reset_mechanism,
    )
    state = initial_leaky_state(1)
    snn_first = -1
    snn_count = 0
    for step, i in enumerate(rows):
        state, spikes = leaky_feed_forward_step(state, (i,), config)
        if spikes[0]:
            snn_count += 1
            if snn_first < 0:
                snn_first = step

    spike_count_delta = snn_count - norse_count
    if norse_first < 0 or snn_first < 0:
        first_spike_step_delta = -1
    else:
        first_spike_step_delta = abs(norse_first - snn_first)

    input_digest = _digest(",".join(f"{v:.17g}" for v in rows))
    digest = _digest(
        f"v={SNNTORCH_DETECTOR_VERSION}|n_norse={norse_count}|n_snn={snn_count}"
        f"|fs_norse={norse_first}|fs_snn={snn_first}"
        f"|dt={dt:.17g}|tau={tau_mem:.17g}|thr={v_threshold:.17g}"
        f"|reset={reset_mechanism}|len={len(rows)}|input={input_digest}"
    )
    return BackendBenchmark(
        norse_spike_count=norse_count,
        snntorch_spike_count=snn_count,
        first_spike_step_norse=norse_first,
        first_spike_step_snntorch=snn_first,
        spike_count_delta=spike_count_delta,
        first_spike_step_delta=first_spike_step_delta,
        digest=digest,
    )


# ---------------------------------------------------------------- helpers


def _digest(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()


def _canonical_weights(weights: LeakyWeights) -> str:
    rows = ";".join(",".join(f"{v:.17g}" for v in row) for row in weights.weight)
    bias = ",".join(f"{v:.17g}" for v in weights.bias)
    return (
        f"v={SNNTORCH_DETECTOR_VERSION}|i={weights.input_dim}"
        f"|h={weights.hidden_dim}|W={rows}|b={bias}"
    )


# ---------------------------------------------------------------- factories


def snntorch_cell_factory() -> LeakyForwardCallable:
    """Lazy production seam over :class:`snntorch.Leaky`.

    Imports ``snntorch`` and ``torch`` inside the function body so the
    pure-Python detector remains usable without either dependency.

    Raises:
        NotImplementedError: Always. Production deployments must wire
            the upstream module after running an offline-trained
            checkpoint through ``requires_grad_(False)`` + ``.eval()``
            and serialising the frozen state. The pure-Python path is
            the canonical fallback until that wiring lands.
    """

    raise NotImplementedError(
        "snntorch_cell_factory: production snntorch wiring is not yet "
        "implemented. Use the pure-Python SNNTorchLeakyCell with a "
        "frozen LeakyWeights as the canonical fallback, or wire a "
        "snntorch.Leaky checkpoint behind a LeakyForwardCallable shim."
    )
