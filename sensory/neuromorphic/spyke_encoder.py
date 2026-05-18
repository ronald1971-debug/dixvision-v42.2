# ADAPTED FROM: SpykeTorch/snn.py
# (SpykeTorch/snn.py — convolutional SNN + intensity-to-latency encoder;
#  SpykeTorch/utils.py — rate / temporal coding utilities;
#  SpykeTorch/functional.py — rank-order time-to-first-spike codes.)
"""B-19 — Spike train encoder for the Norse / snnTorch consumers.

Feeds spike trains into :class:`sensory.neuromorphic.snn_lif.SNNDetector`
(B-14) and :class:`sensory.neuromorphic.snntorch_detector.SNNTorchDetector`
(B-17). Two encoding methods are supported:

* :attr:`EncodingMethod.RATE` — Poisson-style rate coding. Each feature
  value in ``[0, 1]`` is treated as a per-step spike probability.
  Determinism is preserved by a caller-supplied ``seed`` that drives a
  stateless splitmix64 PRNG (no ``random`` import). Adapted from
  ``SpykeTorch/utils.py``'s ``Intensity2Latency`` rate-encoding helper.
* :attr:`EncodingMethod.TEMPORAL` — rank-order / time-to-first-spike
  coding. The highest-valued feature spikes at ``t=0``; the next at
  ``t=1``; ties broken by feature index ascending. Fully deterministic
  (no PRNG draws). Adapted from ``SpykeTorch/functional.py``'s
  ``intensity_to_latency`` reference. **This is the canonical
  hot-path encoder** because it has zero entropy budget.

Both methods produce the same :class:`SpikeTrain` shape: a frozen
tuple of :class:`SpikeEvent` ``(neuron_idx, time_step)`` pairs sorted
by ``time_step ascending, neuron_idx ascending`` so byte-identical
replay is guaranteed.

Authority (sensory tier discipline, mirrors B-14 / B-15 / B-16 / B-17):

* OFFLINE_ONLY (any future learnable encoder weights are trained
  offline) | RUNTIME_SAFE (``encode()`` is a pure function with no
  clock / IO / random / engine cross-imports).
* Encoder output is a frozen value object only. The encoder NEVER
  constructs :class:`HazardEvent`, :class:`SignalEvent`,
  :class:`PatchProposal`, :class:`GovernanceDecision`, or
  :class:`LearningUpdate` — only the evolution-engine / governance-
  engine adapters may project advisory records into typed bus events
  (B27 / B28 / INV-71 authority symmetry, pinned by AST test).
* No imports from any engine. No imports from ``random`` /
  ``time`` / ``datetime`` / ``asyncio`` / ``os`` / ``numpy`` /
  ``torch`` / ``polars`` / ``pandas`` / ``langsmith`` (pinned by AST
  test). Caller supplies all timestamps and seeds.
* If the production ``SpykeTorch`` package is ever wired in, the
  factory ``spyketorch_intensity_to_latency_factory()`` is the seam
  — it lazy-imports ``SpykeTorch`` only inside its body and is
  guarded behind ``NotImplementedError`` until the consumer wires the
  real package. The default pure-Python encoder is always available.

Determinism (INV-15):

* No top-level non-stdlib imports.
* All :class:`SpikeTrain` digests are BLAKE2b-16 over a canonical
  text projection with fixed precision and sorted event order. Three
  encodings of the same ``(features, num_steps, method, seed)`` tuple
  produce a byte-identical ``digest`` (pinned by test).
* The rate-encoded path uses splitmix64 with a deterministic per-
  neuron sub-seed derived from ``seed ^ neuron_idx`` (no global
  state, no thread-local state).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

SPYKE_ENCODER_VERSION = "spyke-encoder/v1"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("spyketorch",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SpykeEncoderError(ValueError):
    """Raised when a feature vector or encoding parameter is invalid."""


# ---------------------------------------------------------------------------
# Method enum
# ---------------------------------------------------------------------------


class EncodingMethod(StrEnum):
    """Encoding strategy.

    ``TEMPORAL`` is the canonical hot-path encoder. ``RATE`` is provided
    for parity with the SpykeTorch reference but spends a deterministic
    entropy budget (splitmix64) per neuron-step.
    """

    RATE = "rate"
    TEMPORAL = "temporal"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpikeEvent:
    """A single spike: neuron ``neuron_idx`` fires at ``time_step``.

    ``neuron_idx`` is 0-based over the feature vector. ``time_step`` is
    0-based; ``time_step < num_steps`` is enforced at construction.
    """

    neuron_idx: int
    time_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.neuron_idx, int) or isinstance(self.neuron_idx, bool):
            raise SpykeEncoderError("SpikeEvent.neuron_idx must be int")
        if not isinstance(self.time_step, int) or isinstance(self.time_step, bool):
            raise SpykeEncoderError("SpikeEvent.time_step must be int")
        if self.neuron_idx < 0:
            raise SpykeEncoderError("SpikeEvent.neuron_idx must be >= 0")
        if self.time_step < 0:
            raise SpykeEncoderError("SpikeEvent.time_step must be >= 0")


@dataclass(frozen=True, slots=True)
class SpikeTrain:
    """Frozen output of :func:`encode`.

    ``events`` is sorted ``(time_step ascending, neuron_idx ascending)``
    so two encodings of the same ``(features, num_steps, method, seed)``
    produce byte-identical tuples.

    ``digest`` is the BLAKE2b-16 hex of a canonical text projection
    that includes ``num_neurons / num_steps / method / seed / events``
    — three runs over the same input yield identical digests
    (INV-15, pinned by test).
    """

    num_neurons: int
    num_steps: int
    method: EncodingMethod
    seed: int
    events: tuple[SpikeEvent, ...]
    digest: str = field(default="")

    def __post_init__(self) -> None:
        if not isinstance(self.num_neurons, int) or isinstance(self.num_neurons, bool):
            raise SpykeEncoderError("SpikeTrain.num_neurons must be int")
        if not isinstance(self.num_steps, int) or isinstance(self.num_steps, bool):
            raise SpykeEncoderError("SpikeTrain.num_steps must be int")
        if self.num_neurons <= 0:
            raise SpykeEncoderError("SpikeTrain.num_neurons must be > 0")
        if self.num_steps <= 0:
            raise SpykeEncoderError("SpikeTrain.num_steps must be > 0")
        if not isinstance(self.events, tuple):
            raise SpykeEncoderError("SpikeTrain.events must be a tuple")
        if not isinstance(self.method, EncodingMethod):
            raise SpykeEncoderError("SpikeTrain.method must be EncodingMethod")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SpykeEncoderError("SpikeTrain.seed must be int")
        for ev in self.events:
            if not isinstance(ev, SpikeEvent):
                raise SpykeEncoderError("SpikeTrain.events must be SpikeEvent")
            if ev.neuron_idx >= self.num_neurons:
                raise SpykeEncoderError(
                    f"SpikeTrain event neuron_idx {ev.neuron_idx} >= num_neurons {self.num_neurons}"
                )
            if ev.time_step >= self.num_steps:
                raise SpykeEncoderError(
                    f"SpikeTrain event time_step {ev.time_step} >= num_steps {self.num_steps}"
                )
        keys = [(ev.time_step, ev.neuron_idx) for ev in self.events]
        if keys != sorted(keys):
            raise SpykeEncoderError("SpikeTrain.events must be sorted by (time_step, neuron_idx)")
        if len(keys) != len(set(keys)):
            raise SpykeEncoderError("SpikeTrain.events must be unique on (time_step, neuron_idx)")
        if not self.digest:
            object.__setattr__(self, "digest", _digest(self))

    def spike_count(self) -> int:
        return len(self.events)

    def to_dense(self) -> tuple[tuple[int, ...], ...]:
        """Project to a dense ``num_steps × num_neurons`` 0/1 matrix.

        Useful for Norse / snnTorch consumers that want a per-step
        binary vector. Outer axis is time. Inner axis is neuron.
        """
        rows = tuple(tuple(0 for _ in range(self.num_neurons)) for _ in range(self.num_steps))
        # We rebuild rows because tuples are immutable.
        out: list[list[int]] = [list(r) for r in rows]
        for ev in self.events:
            out[ev.time_step][ev.neuron_idx] = 1
        return tuple(tuple(row) for row in out)


# ---------------------------------------------------------------------------
# Backend protocol (production seam for the real SpykeTorch package)
# ---------------------------------------------------------------------------


@runtime_checkable
class SpykeBackend(Protocol):
    """Production seam.

    Implementations may forward to the real ``SpykeTorch`` package.
    The default pure-Python encoder does NOT use this Protocol; it is
    only invoked when the caller explicitly opts into the lazy
    ``spyketorch_intensity_to_latency_factory()`` seam.
    """

    def intensity_to_latency(
        self,
        features: Sequence[float],
        num_steps: int,
    ) -> SpikeTrain: ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_MASK64 = (1 << 64) - 1


def _splitmix64(state: int) -> tuple[int, int]:
    """One step of the splitmix64 PRNG (deterministic, stateless)."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z, state


def _uniform_unit(state: int) -> tuple[float, int]:
    """Return a uniform float in [0, 1) and the next splitmix64 state."""
    z, state = _splitmix64(state)
    # 53-bit mantissa for double-precision uniformity.
    return (z >> 11) / float(1 << 53), state


def _validate_features(features: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(features, (list, tuple)):
        raise SpykeEncoderError("features must be a list or tuple")
    if not features:
        raise SpykeEncoderError("features must be non-empty")
    out: list[float] = []
    for i, f in enumerate(features):
        if isinstance(f, bool):
            raise SpykeEncoderError(f"features[{i}] must not be bool")
        if not isinstance(f, (int, float)):
            raise SpykeEncoderError(f"features[{i}] must be a real number, got {type(f).__name__}")
        if f != f:  # NaN check without math.isnan import dance.
            raise SpykeEncoderError(f"features[{i}] is NaN")
        if f in (float("inf"), float("-inf")):
            raise SpykeEncoderError(f"features[{i}] is infinite")
        if f < 0.0 or f > 1.0:
            raise SpykeEncoderError(f"features[{i}]={f!r} must be in [0.0, 1.0]")
        out.append(float(f))
    return tuple(out)


def _validate_num_steps(num_steps: int) -> None:
    if not isinstance(num_steps, int) or isinstance(num_steps, bool):
        raise SpykeEncoderError("num_steps must be int")
    if num_steps <= 0:
        raise SpykeEncoderError("num_steps must be > 0")
    if num_steps > 1_000_000:
        raise SpykeEncoderError("num_steps must be <= 1_000_000")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SpykeEncoderError("seed must be int")
    if seed < 0:
        raise SpykeEncoderError("seed must be >= 0")
    if seed > _MASK64:
        raise SpykeEncoderError("seed must fit in 64 bits")


def _canonical_text(
    *,
    num_neurons: int,
    num_steps: int,
    method: EncodingMethod,
    seed: int,
    events: tuple[SpikeEvent, ...],
) -> str:
    """Stable text form for the digest. Field order is fixed."""
    parts = [
        f"version={SPYKE_ENCODER_VERSION}",
        f"num_neurons={num_neurons}",
        f"num_steps={num_steps}",
        f"method={method.value}",
        f"seed={seed}",
    ]
    for ev in events:
        parts.append(f"e={ev.time_step}:{ev.neuron_idx}")
    return ";".join(parts)


def _digest(train: SpikeTrain) -> str:
    text = _canonical_text(
        num_neurons=train.num_neurons,
        num_steps=train.num_steps,
        method=train.method,
        seed=train.seed,
        events=train.events,
    )
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Rate encoding
# ---------------------------------------------------------------------------


def rate_encode(
    features: Sequence[float],
    *,
    num_steps: int,
    seed: int,
) -> SpikeTrain:
    """Poisson-style rate encoding.

    Each feature ``f_i in [0, 1]`` produces a spike at step ``t`` with
    probability ``f_i`` (Bernoulli per step). Per-neuron PRNG state is
    initialised from ``seed ^ neuron_idx`` so swapping neuron order
    in the input changes the output deterministically (and reversibly
    via inverse permutation).

    Note that ``RATE`` spends an entropy budget. Prefer
    :func:`temporal_encode` on the hot path.
    """
    feats = _validate_features(features)
    _validate_num_steps(num_steps)
    _validate_seed(seed)

    events: list[SpikeEvent] = []
    for n_idx, p in enumerate(feats):
        state = seed ^ n_idx
        for t in range(num_steps):
            u, state = _uniform_unit(state)
            if u < p:
                events.append(SpikeEvent(neuron_idx=n_idx, time_step=t))
    events.sort(key=lambda ev: (ev.time_step, ev.neuron_idx))
    return SpikeTrain(
        num_neurons=len(feats),
        num_steps=num_steps,
        method=EncodingMethod.RATE,
        seed=seed,
        events=tuple(events),
    )


# ---------------------------------------------------------------------------
# Temporal encoding (rank-order / time-to-first-spike)
# ---------------------------------------------------------------------------


def temporal_encode(
    features: Sequence[float],
    *,
    num_steps: int,
) -> SpikeTrain:
    """Rank-order time-to-first-spike encoding (no PRNG, fully deterministic).

    The highest-valued feature spikes at ``t=0``; the next at ``t=1``;
    and so on. Ties are broken by feature index ascending. Features
    equal to ``0.0`` are NOT emitted (they would map past the last
    time step). When the number of non-zero features exceeds
    ``num_steps``, the lowest-ranked features past ``t=num_steps - 1``
    are dropped (the canonical SpykeTorch behaviour).

    This is the canonical hot-path encoder per the spec: zero entropy
    budget, byte-identical across machines.
    """
    feats = _validate_features(features)
    _validate_num_steps(num_steps)

    indexed = sorted(
        ((feats[i], i) for i in range(len(feats))),
        key=lambda pair: (-pair[0], pair[1]),
    )
    events: list[SpikeEvent] = []
    for rank, (val, n_idx) in enumerate(indexed):
        if val <= 0.0:
            break
        if rank >= num_steps:
            break
        events.append(SpikeEvent(neuron_idx=n_idx, time_step=rank))
    events.sort(key=lambda ev: (ev.time_step, ev.neuron_idx))
    return SpikeTrain(
        num_neurons=len(feats),
        num_steps=num_steps,
        method=EncodingMethod.TEMPORAL,
        seed=0,
        events=tuple(events),
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def encode(
    features: Sequence[float],
    *,
    method: EncodingMethod,
    num_steps: int,
    seed: int = 0,
) -> SpikeTrain:
    """Dispatch to :func:`rate_encode` or :func:`temporal_encode`.

    ``seed`` is ignored for :attr:`EncodingMethod.TEMPORAL` and required
    for :attr:`EncodingMethod.RATE`. Centralising dispatch through this
    function lets callers swap methods at config time without changing
    the call site.
    """
    if not isinstance(method, EncodingMethod):
        raise SpykeEncoderError("method must be EncodingMethod")
    if method is EncodingMethod.RATE:
        return rate_encode(features, num_steps=num_steps, seed=seed)
    if method is EncodingMethod.TEMPORAL:
        return temporal_encode(features, num_steps=num_steps)
    raise SpykeEncoderError(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Dense-projection helper
# ---------------------------------------------------------------------------


def spike_train_to_step_inputs(
    train: SpikeTrain,
) -> tuple[tuple[float, ...], ...]:
    """Project a :class:`SpikeTrain` to ``num_steps × num_neurons`` floats.

    Each row is a per-step current vector fed into Norse's
    :class:`~sensory.neuromorphic.snn_lif.LIFCell` or snnTorch's
    :class:`~sensory.neuromorphic.snntorch_detector.SNNTorchDetector`.
    Spikes project to ``1.0``, silence to ``0.0`` — exact arithmetic,
    no floating drift.
    """
    if not isinstance(train, SpikeTrain):
        raise SpykeEncoderError("train must be SpikeTrain")
    dense = train.to_dense()
    return tuple(tuple(float(c) for c in row) for row in dense)


# ---------------------------------------------------------------------------
# Production seam
# ---------------------------------------------------------------------------


def spyketorch_intensity_to_latency_factory(*_args: object, **_kwargs: object) -> SpykeBackend:
    """Lazy production seam — wires the real SpykeTorch package.

    Raises :class:`NotImplementedError` until the consumer wires the
    actual SpykeTorch ``intensity_to_latency`` helper. The default
    pure-Python encoders are always available and have no SpykeTorch
    dependency.
    """
    raise NotImplementedError(
        "SpykeTorch production seam is not wired; "
        "use sensory.neuromorphic.spyke_encoder.encode() instead."
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "SPYKE_ENCODER_VERSION",
    "EncodingMethod",
    "SpikeEvent",
    "SpikeTrain",
    "SpykeBackend",
    "SpykeEncoderError",
    "encode",
    "rate_encode",
    "spike_train_to_step_inputs",
    "spyketorch_intensity_to_latency_factory",
    "temporal_encode",
]
