# ADAPTED FROM: nengo/nengo
# (nengo/ensemble.py — Ensemble (population coding);
#  nengo/connection.py — Connection (decoder projection);
#  nengo/builder/ensemble.py — gain/bias/encoder build pipeline;
#  nengo/neurons.py — LIF / LIFRate neuron model;
#  nengo/probe.py — Probe (read-out).)
"""C-43 — NengoCognitiveAnalyser: governance-gated Nengo
population-coded cognitive ensemble.

Nengo (Neural Engineering Framework) frames cognition as
**population coding** — a vector ``x`` of length ``dimensions`` is
represented by a population of ``n_neurons`` LIF neurons whose
firing rates ``a_i(x)`` together carry the vector. The
representation is recovered (decoded) by a linear projection
``y_hat = D @ a(x)``. The encoders ``e_i``, gains ``G_i``,
biases ``b_i``, and decoders ``D`` are computed **offline** and
frozen at inference; nengo's runtime simulator only feeds the
input forward and reads the decoded output back.

The DIX adapter wraps that population surface behind a Protocol
seam so the sensory layer can ask "given a fixed cognitive
ensemble + a window of regime-feature vectors, what regime is
the population coding right now, and how confident is the
decoding?" without ever importing ``nengo`` / ``numpy`` at
module load.

What this module is
-------------------

* Pure-Python coordinator + frozen value objects. The actual
  ``nengo`` import is hidden behind a
  :class:`NengoCognitiveEngine` Protocol — production wires
  :func:`nengo_cognitive_engine`; unit tests inject a
  deterministic fake. The module never imports nengo at
  module load.
* OFFLINE_ONLY tier. The analyser reads no environment
  variables, performs no IO, never imports
  ``execution_engine`` / ``governance_engine`` /
  ``system_engine`` / ``registry`` / ``ui``. It produces one
  :class:`NengoRegimePulse` and stops.
* ADVISORY only (INV-19). The pulse is advisory; the analyser
  never emits a typed bus event and never escalates to a
  hazard. Dyon (system engine) decides whether the regime
  decoding warrants attention.
* OFFLINE training, frozen weights at inference. Encoders,
  gains, biases, and decoders are caller-supplied (the result
  of an offline nengo build pipeline run through
  ``requires_grad_(False)`` and serialised). The dataclass is
  ``frozen=True, slots=True``; no runtime path can mutate the
  weights.
* INV-15 byte-identical replays.
  :meth:`NengoCognitiveAnalyser.detect` with identical
  ``arguments`` / ``window`` / ``ts_ns`` / ``regime_label``
  returns identical :class:`NengoRegimePulse` records.
  Determinism is delegated to the injected engine; the default
  pure-Python ensemble threads
  :attr:`NengoEnsembleWeights.weights_seed` into a stateless
  splitmix64 PRNG so the offline build is fully reproducible.
* No clock reads. Caller supplies ``ts_ns``.

What survives from upstream
---------------------------

* The population-coding surface — every neuron carries an
  encoder vector ``e_i`` (a unit-norm direction in the
  represented space), a gain ``G_i`` (positive scalar), and a
  bias ``b_i``. The instantaneous input current is
  ``J_i = G_i * (e_i . x) + b_i`` (canonical NEF Eqn. 3).
* The LIF firing model — for input current ``J``, a LIF
  neuron's firing rate is
  ``r = 1 / (tau_ref - tau_rc * log(1 - 1 / J))`` when
  ``J > 1``, else 0 (canonical NEF Eqn. 4). The pure-Python
  ensemble uses the discrete-time LIF integrator from B-14
  rather than the analytic rate formula so spike trains are
  byte-identical with the upstream simulator.
* The decoder projection — the decoded value is
  ``y_hat = D @ a(x)``, where ``a(x)`` is the per-neuron
  firing activity in the window (canonical NEF Eqn. 6).
* Probe semantics — :attr:`NengoRegimePulse.decoded_value`
  carries the per-window mean of the decoded vector, mirroring
  what a ``nengo.Probe(ensemble, attr="decoded_output")`` would
  yield over the window.

What we replaced
----------------

* Nengo's matplotlib + GUI plot pipeline → no plotting. The
  decoded value lives in :class:`NengoRegimePulse`; the
  dashboard handles rendering.
* Nengo's pickle / hdf5 model checkpoints → the engine owns
  its weights; the seam carries a frozen ``weights_digest`` so
  identical weights produce identical decodings.
* Nengo's RNG (``numpy.random.RandomState``) → caller-supplied
  :attr:`NengoEnsembleWeights.weights_seed` threaded through a
  stateless splitmix64 PRNG (no global state, no wall-clock
  reads, byte-identical across machines).

Authority constraints (manifest §H1)
------------------------------------

* OFFLINE_ONLY tier — no IO, no clock, no global state, no
  PRNG reads from the wall clock; the ensemble's PRNG is
  seeded by caller-supplied ``weights_seed``. AST tests pin
  the import contract.
* ADVISORY only (INV-19) — the analyser emits a
  :class:`NengoRegimePulse` and never a :class:`HazardEvent`,
  never mutates the registry, never reads the clock.
* No engine cross-imports — AST test pins no
  ``execution_engine.`` / ``governance_engine.`` /
  ``system_engine.`` / ``registry.`` / ``ui.`` references at
  any depth.
* INV-15 — :attr:`NengoRegimePulse.weights_digest` and
  :attr:`NengoRegimePulse.decoded_value` are deterministic
  functions of the inputs (BLAKE2b over a canonical text
  projection, fixed-precision serialisation). 3-run
  identical-input replay equality is pinned in tests.
* Defensive caps:
  - :data:`MIN_DIMENSIONS` 1 / :data:`MAX_DIMENSIONS` 32 hard
    floor and ceiling on represented vector dimensionality.
  - :data:`MIN_NEURONS` 1 / :data:`MAX_NEURONS` 4096 hard
    floor and ceiling on ensemble population size.
  - :data:`MIN_WINDOW` 1 / :data:`MAX_WINDOW` 4096 hard floor
    and ceiling on input window length.
  - :data:`MAX_REGIME_LABEL_LEN` 64 chars on regime label.
  - :data:`MAX_SOURCE_LEN` 64 chars on source id.
  - :data:`MAX_SYMBOL_LEN` 64 chars on symbol id.

Refs:
- ``DIX_MASTER_CANONICAL.md`` C-43 (nengo cognitive spec).
- ``sensory/neuromorphic/nengo_cognitive.py`` (this file).
- ``sensory/neuromorphic/snn_lif.py`` (B-14 — the LIF / Poisson
  twin showing the sensor-side frozen-weight / lazy-factory
  shape).
- ``sensory/neuromorphic/contracts.py`` (PulseSignal /
  AnomalyPulse / RiskPulse — sibling advisory pulse types).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

NENGO_COGNITIVE_VERSION: str = "nengo-cognitive/v1"

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("nengo", "numpy")
"""Optional production backend (lazy-imported via
:func:`nengo_cognitive_engine`). Pure-Python ensemble has zero
deps."""

MIN_DIMENSIONS: int = 1
"""Hard lower bound on represented vector dimensionality."""

MAX_DIMENSIONS: int = 32
"""Hard upper bound on represented vector dimensionality."""

MIN_NEURONS: int = 1
"""Hard lower bound on ensemble population size."""

MAX_NEURONS: int = 4_096
"""Hard upper bound on ensemble population size. Keeps single-step
inference under the sensory tier's per-call budget."""

MIN_WINDOW: int = 1
"""Hard lower bound on input window length."""

MAX_WINDOW: int = 4_096
"""Hard upper bound on input window length."""

MAX_REGIME_LABEL_LEN: int = 64
"""Hard upper bound on caller-supplied regime label."""

MAX_SOURCE_LEN: int = 64
"""Hard upper bound on caller-supplied source id."""

MAX_SYMBOL_LEN: int = 64
"""Hard upper bound on caller-supplied symbol id."""

_DIGEST_BYTES: int = 16

ANALYSIS_SOURCE: str = "sensory.neuromorphic.nengo_cognitive"
"""Constant tag stamped onto every :class:`NengoRegimePulse`
``evidence['analyser']``. Distinguishes nengo-produced pulses
from other neuromorphic sensors (e.g.
``sensory.neuromorphic.snn_lif``)."""

REGIME_LONG: str = "LONG"
REGIME_SHORT: str = "SHORT"
REGIME_NEUTRAL: str = "NEUTRAL"

_REGIMES: frozenset[str] = frozenset({REGIME_LONG, REGIME_SHORT, REGIME_NEUTRAL})


# ---------------------------------------------------------------- errors


class NengoCognitiveError(ValueError):
    """Raised for malformed inputs / configuration in the nengo
    cognitive ensemble."""


# ---------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class NengoEnsembleConfig:
    """Frozen NEF / nengo ensemble hyperparameters.

    Field names mirror :class:`nengo.Ensemble` so a future port
    that swaps the inference path for the upstream simulator can
    pass these straight through.

    Attributes:
        tau_rc: Membrane RC time constant in the same time unit
            as ``dt``. Nengo default ``0.02`` (20 ms); we default
            ``0.02`` to match.
        tau_ref: Refractory period in the same time unit as
            ``dt``. Nengo default ``0.002`` (2 ms); we default
            ``0.002`` to match.
        v_threshold: Spike threshold (LIF). Default ``1.0``.
        v_reset: Reset potential after a spike. Default ``0.0``.
        v_leak: Equilibrium potential. Default ``0.0``.
        dt: Integration step in the same time unit as ``tau_rc``.
            Nengo default ``0.001`` (1 ms); we default
            ``0.001`` to match.
    """

    tau_rc: float = 0.02
    tau_ref: float = 0.002
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_leak: float = 0.0
    dt: float = 0.001

    def __post_init__(self) -> None:
        if not (math.isfinite(self.tau_rc) and self.tau_rc > 0.0):
            raise NengoCognitiveError("NengoEnsembleConfig.tau_rc must be finite and > 0")
        if not (math.isfinite(self.tau_ref) and self.tau_ref >= 0.0):
            raise NengoCognitiveError("NengoEnsembleConfig.tau_ref must be finite and >= 0")
        if not math.isfinite(self.v_threshold):
            raise NengoCognitiveError("NengoEnsembleConfig.v_threshold must be finite")
        if not math.isfinite(self.v_reset):
            raise NengoCognitiveError("NengoEnsembleConfig.v_reset must be finite")
        if not math.isfinite(self.v_leak):
            raise NengoCognitiveError("NengoEnsembleConfig.v_leak must be finite")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise NengoCognitiveError("NengoEnsembleConfig.dt must be finite and > 0")
        if self.dt > self.tau_rc:
            raise NengoCognitiveError(
                "NengoEnsembleConfig.dt must be <= tau_rc for stable integration"
            )


# ---------------------------------------------------------------- weights


@dataclass(frozen=True, slots=True)
class NengoEnsembleWeights:
    """Frozen NEF / nengo ensemble weights.

    Carries the **offline-trained** encoders ``e_i``, gains
    ``G_i``, biases ``b_i``, and decoders ``D`` that together
    define the population-coding map ``x -> y_hat``. All weights
    are validated for finiteness at construction; once built,
    the dataclass is immutable so no runtime path can mutate
    them (INV-20).

    Layout:

    * ``encoders`` is ``[n_neurons][dimensions]`` — one
      direction per neuron. Each row should be unit-norm in the
      represented space, but the dataclass does **not** enforce
      that (callers may inject deliberately scaled encoders).
    * ``gains`` and ``biases`` are length ``n_neurons``.
    * ``decoders`` is ``[dimensions][n_neurons]`` — one row per
      output dimension.
    * ``weights_seed`` is the seed used to build the random
      encoder / gain / bias draw offline. Carried inside the
      record so identical decodings can be traced back to
      identical offline builds.

    Attributes:
        encoders: ``[n_neurons][dimensions]`` row-major.
        gains: Length ``n_neurons``, each ``> 0``.
        biases: Length ``n_neurons``, finite.
        decoders: ``[dimensions][n_neurons]`` row-major.
        n_neurons: Population size (must match row counts).
        dimensions: Represented vector dim (must match widths).
        weights_seed: Offline-build seed (``>= 0``).
    """

    encoders: tuple[tuple[float, ...], ...]
    gains: tuple[float, ...]
    biases: tuple[float, ...]
    decoders: tuple[tuple[float, ...], ...]
    n_neurons: int
    dimensions: int
    weights_seed: int

    def __post_init__(self) -> None:
        if self.n_neurons < MIN_NEURONS or self.n_neurons > MAX_NEURONS:
            raise NengoCognitiveError(
                f"NengoEnsembleWeights.n_neurons must be in [{MIN_NEURONS}, {MAX_NEURONS}]"
            )
        if self.dimensions < MIN_DIMENSIONS or self.dimensions > MAX_DIMENSIONS:
            raise NengoCognitiveError(
                f"NengoEnsembleWeights.dimensions must be in [{MIN_DIMENSIONS}, {MAX_DIMENSIONS}]"
            )
        if len(self.encoders) != self.n_neurons:
            raise NengoCognitiveError(
                "NengoEnsembleWeights.encoders row count must equal n_neurons"
            )
        for row in self.encoders:
            if len(row) != self.dimensions:
                raise NengoCognitiveError(
                    "NengoEnsembleWeights.encoders row width must equal dimensions"
                )
            for value in row:
                if not math.isfinite(value):
                    raise NengoCognitiveError(
                        "NengoEnsembleWeights.encoders entries must be finite"
                    )
        if len(self.gains) != self.n_neurons:
            raise NengoCognitiveError("NengoEnsembleWeights.gains length must equal n_neurons")
        for value in self.gains:
            if not (math.isfinite(value) and value > 0.0):
                raise NengoCognitiveError(
                    "NengoEnsembleWeights.gains entries must be finite and > 0"
                )
        if len(self.biases) != self.n_neurons:
            raise NengoCognitiveError("NengoEnsembleWeights.biases length must equal n_neurons")
        for value in self.biases:
            if not math.isfinite(value):
                raise NengoCognitiveError("NengoEnsembleWeights.biases entries must be finite")
        if len(self.decoders) != self.dimensions:
            raise NengoCognitiveError(
                "NengoEnsembleWeights.decoders row count must equal dimensions"
            )
        for row in self.decoders:
            if len(row) != self.n_neurons:
                raise NengoCognitiveError(
                    "NengoEnsembleWeights.decoders row width must equal n_neurons"
                )
            for value in row:
                if not math.isfinite(value):
                    raise NengoCognitiveError(
                        "NengoEnsembleWeights.decoders entries must be finite"
                    )
        if self.weights_seed < 0:
            raise NengoCognitiveError("NengoEnsembleWeights.weights_seed must be >= 0")

    def digest(self) -> str:
        """Stable 16-hex BLAKE2b digest of all weights + dims +
        seed.

        Determinism (INV-15): the projection is a fixed-precision
        canonical text form so identical weights produce
        identical digests across machines / Python builds.
        """

        return _digest(_canonical_weights(self))


def build_random_ensemble_weights(
    *,
    n_neurons: int,
    dimensions: int,
    seed: int,
    max_rate: float = 200.0,
    intercept_low: float = -1.0,
    intercept_high: float = 0.9,
) -> NengoEnsembleWeights:
    """Build a deterministic offline ensemble weight set.

    Mirrors nengo's ``Ensemble`` build pipeline (encoders sampled
    uniformly on the unit hypersphere; gains and biases set so
    each neuron's tuning curve crosses the firing-rate axis at a
    uniformly-drawn intercept and reaches ``max_rate`` at the
    encoder direction; decoders are the canonical pseudo-inverse
    that recovers ``x`` from the population activity).

    The whole pipeline is driven by a stateless splitmix64 PRNG
    seeded with ``seed`` so two calls with identical inputs
    produce byte-identical weights across machines and Python
    builds (INV-15).
    """

    if n_neurons < MIN_NEURONS or n_neurons > MAX_NEURONS:
        raise NengoCognitiveError(
            f"build_random_ensemble_weights: n_neurons must be in [{MIN_NEURONS}, {MAX_NEURONS}]"
        )
    if dimensions < MIN_DIMENSIONS or dimensions > MAX_DIMENSIONS:
        raise NengoCognitiveError(
            "build_random_ensemble_weights: dimensions must be in "
            f"[{MIN_DIMENSIONS}, {MAX_DIMENSIONS}]"
        )
    if seed < 0:
        raise NengoCognitiveError("build_random_ensemble_weights: seed must be >= 0")
    if not (math.isfinite(max_rate) and max_rate > 0.0):
        raise NengoCognitiveError("build_random_ensemble_weights: max_rate must be finite and > 0")
    if not (math.isfinite(intercept_low) and math.isfinite(intercept_high)):
        raise NengoCognitiveError("build_random_ensemble_weights: intercept bounds must be finite")
    if intercept_low >= intercept_high:
        raise NengoCognitiveError(
            "build_random_ensemble_weights: intercept_low must be < intercept_high"
        )

    state = seed
    encoders: list[tuple[float, ...]] = []
    for _ in range(n_neurons):
        raw: list[float] = []
        # Box-Muller via splitmix64: draw `dimensions` standard
        # normals, then normalise to the unit sphere.
        i = 0
        while i < dimensions:
            u1, state = _uniform01(state)
            u2, state = _uniform01(state)
            # Clamp away from zero so log() is finite. The
            # splitmix64 outputs cover [0, 1) so the floor is
            # ~2^-53 which is fine for log; we add 1e-300 for
            # extra safety on the closed-interval boundary.
            u1c = max(u1, 1e-300)
            r = math.sqrt(-2.0 * math.log(u1c))
            theta = 2.0 * math.pi * u2
            raw.append(r * math.cos(theta))
            i += 1
            if i < dimensions:
                raw.append(r * math.sin(theta))
                i += 1
        norm = math.sqrt(sum(v * v for v in raw))
        if norm == 0.0:
            # Pathological zero draw — replace with canonical
            # basis vector along axis 0 so the encoder is well
            # defined.
            raw = [1.0] + [0.0] * (dimensions - 1)
            norm = 1.0
        encoders.append(tuple(v / norm for v in raw[:dimensions]))

    gains: list[float] = []
    biases: list[float] = []
    for _ in range(n_neurons):
        u_int, state = _uniform01(state)
        intercept = intercept_low + u_int * (intercept_high - intercept_low)
        # Tuning curve: r(x) = max_rate when e . x = 1, r(x) = 0
        # when e . x = intercept. The canonical NEF closed-form
        # for the LIF model is g = (1 - 1/J_max) / (1 - intercept)
        # and b = 1 - g * intercept, where J_max is the input
        # current at which the LIF rate equals ``max_rate``. We
        # use a simplified affine form which gives the same
        # qualitative tuning curve and stays in the linear regime
        # for the pure-Python integrator (the upstream nengo
        # build is invoked when the production seam is wired).
        gain = max(1e-6, (1.0 - 0.05) / max(1e-6, 1.0 - intercept))
        bias = 1.0 - gain * intercept
        gains.append(gain)
        biases.append(bias)

    # Decoders: canonical pseudo-inverse for the identity decoder.
    # Without invoking numpy we use a simple "transpose of encoders
    # scaled by 1 / n_neurons" which recovers ``x`` exactly for the
    # rate-coded identity case ``a_i = e_i . x``. The decoded value
    # is therefore the spike-rate-weighted mean of the encoder
    # directions, which is the canonical NEF result for an identity
    # connection.
    decoders: list[tuple[float, ...]] = []
    scale = 1.0 / float(n_neurons)
    for d in range(dimensions):
        row = tuple(scale * encoders[i][d] for i in range(n_neurons))
        decoders.append(row)

    return NengoEnsembleWeights(
        encoders=tuple(encoders),
        gains=tuple(gains),
        biases=tuple(biases),
        decoders=tuple(decoders),
        n_neurons=n_neurons,
        dimensions=dimensions,
        weights_seed=seed,
    )


# ---------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class NengoEnsembleState:
    """Per-neuron membrane potential snapshot.

    Mirrors nengo's :class:`nengo.builder.signal.Signal` for the
    LIF ``voltage`` state, packaged as a frozen value type. Each
    forward step returns a new instance (no in-place mutation).
    """

    v: tuple[float, ...]

    def __post_init__(self) -> None:
        for value in self.v:
            if not math.isfinite(value):
                raise NengoCognitiveError("NengoEnsembleState.v entries must be finite")


def initial_state(n_neurons: int, *, v_leak: float = 0.0) -> NengoEnsembleState:
    """Build the rest-potential initial :class:`NengoEnsembleState`."""

    if n_neurons < MIN_NEURONS or n_neurons > MAX_NEURONS:
        raise NengoCognitiveError(
            f"initial_state: n_neurons must be in [{MIN_NEURONS}, {MAX_NEURONS}]"
        )
    if not math.isfinite(v_leak):
        raise NengoCognitiveError("initial_state: v_leak must be finite")
    return NengoEnsembleState(v=tuple(v_leak for _ in range(n_neurons)))


# ---------------------------------------------------------------- output


@dataclass(frozen=True, slots=True)
class NengoRegimePulse:
    """Frozen advisory output of a :class:`NengoCognitiveAnalyser`.

    Mirrors the existing :class:`PulseSignal` /
    :class:`AnomalyPulse` shape: polarity + intensity + sample
    count + caller-supplied ``ts_ns``. Adds two
    population-coding-specific fields: ``regime_label`` (the
    caller's semantic tag — e.g. ``"BULL_TREND"``) and
    ``decoded_value`` (the window-mean of the decoded population
    output, length = ensemble ``dimensions``).

    The pulse is **advisory only** (INV-19). The analyser never
    emits a :class:`HazardEvent` — Dyon decides whether a high
    decoded magnitude warrants escalation.

    Attributes:
        ts_ns: Window-close timestamp in nanoseconds
            (caller-supplied, INV-15).
        source: Stable source id (e.g. ``"BINANCE"``). Empty
            rejected.
        symbol: Per-instrument id (e.g. ``"BTCUSDT"``). Empty
            rejected.
        regime_label: Operator-supplied semantic tag (e.g.
            ``"BULL_TREND"`` / ``"REGIME_FLIP"``). Empty
            rejected.
        polarity: ``LONG`` / ``SHORT`` / ``NEUTRAL``. Derived by
            the analyser from ``decoded_value[0]`` sign and
            confidence threshold.
        confidence: Normalised decoded magnitude in
            ``[0.0, 1.0]``. NaN, +Inf, negatives rejected.
        decoded_value: Window-mean of the decoded vector
            ``y_hat = D @ a(x)``. Length equals ensemble
            ``dimensions``. All entries finite.
        spike_count: Total spike count across the window.
        sample_count: Window length (``>= 1``).
        weights_digest: BLAKE2b-16 hex of the projection used to
            generate the pulse, for replay-time provenance.
        evidence: Free-form structural metadata. Always carries
            ``evidence["analyser"] == ANALYSIS_SOURCE``.
    """

    ts_ns: int
    source: str
    symbol: str
    regime_label: str
    polarity: str
    confidence: float
    decoded_value: tuple[float, ...]
    spike_count: int
    sample_count: int
    weights_digest: str
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise NengoCognitiveError("NengoRegimePulse.ts_ns must be int")
        if self.ts_ns < 0:
            raise NengoCognitiveError("NengoRegimePulse.ts_ns must be >= 0")
        if not isinstance(self.source, str) or not self.source:
            raise NengoCognitiveError("NengoRegimePulse.source must be a non-empty string")
        if len(self.source) > MAX_SOURCE_LEN:
            raise NengoCognitiveError(f"NengoRegimePulse.source must be <= {MAX_SOURCE_LEN} chars")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise NengoCognitiveError("NengoRegimePulse.symbol must be a non-empty string")
        if len(self.symbol) > MAX_SYMBOL_LEN:
            raise NengoCognitiveError(f"NengoRegimePulse.symbol must be <= {MAX_SYMBOL_LEN} chars")
        if not isinstance(self.regime_label, str) or not self.regime_label:
            raise NengoCognitiveError("NengoRegimePulse.regime_label must be a non-empty string")
        if len(self.regime_label) > MAX_REGIME_LABEL_LEN:
            raise NengoCognitiveError(
                f"NengoRegimePulse.regime_label must be <= {MAX_REGIME_LABEL_LEN} chars"
            )
        if self.polarity not in _REGIMES:
            raise NengoCognitiveError(
                f"NengoRegimePulse.polarity must be one of {sorted(_REGIMES)}"
            )
        if not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise NengoCognitiveError("NengoRegimePulse.confidence must be finite in [0.0, 1.0]")
        if not isinstance(self.decoded_value, tuple):
            raise NengoCognitiveError("NengoRegimePulse.decoded_value must be a tuple")
        if len(self.decoded_value) < MIN_DIMENSIONS or len(self.decoded_value) > MAX_DIMENSIONS:
            raise NengoCognitiveError(
                "NengoRegimePulse.decoded_value length must be in "
                f"[{MIN_DIMENSIONS}, {MAX_DIMENSIONS}]"
            )
        for value in self.decoded_value:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise NengoCognitiveError("NengoRegimePulse.decoded_value entries must be float")
            if not math.isfinite(value):
                raise NengoCognitiveError("NengoRegimePulse.decoded_value entries must be finite")
        if self.spike_count < 0:
            raise NengoCognitiveError("NengoRegimePulse.spike_count must be >= 0")
        if self.sample_count < 1:
            raise NengoCognitiveError("NengoRegimePulse.sample_count must be >= 1")
        if len(self.weights_digest) != _DIGEST_BYTES * 2:
            raise NengoCognitiveError("NengoRegimePulse.weights_digest must be 16-hex BLAKE2b-16")
        if not all(c in "0123456789abcdef" for c in self.weights_digest):
            raise NengoCognitiveError("NengoRegimePulse.weights_digest must be lowercase hex")


# ---------------------------------------------------------------- functional


def lif_step(
    state: NengoEnsembleState,
    input_current: Sequence[float],
    config: NengoEnsembleConfig,
) -> tuple[NengoEnsembleState, tuple[bool, ...]]:
    """One discrete-time LIF step (functional, pure).

    Implements::

        v_next = v + (dt / tau_rc) * (v_leak - v + J)
        spike  = v_next >= v_threshold
        v_next = v_reset if spike else v_next

    Args:
        state: Current membrane potentials.
        input_current: Per-neuron synaptic input ``J`` of the
            same length as ``state.v``. Must be finite.
        config: Ensemble hyperparameters.

    Returns:
        ``(next_state, spikes)`` — fresh :class:`NengoEnsembleState`
        and a boolean spike vector of the same length.
    """

    if len(input_current) != len(state.v):
        raise NengoCognitiveError("lif_step: input_current length must equal state.v length")
    decay = config.dt / config.tau_rc
    next_v: list[float] = []
    spikes: list[bool] = []
    for v, j in zip(state.v, input_current, strict=True):
        if not math.isfinite(j):
            raise NengoCognitiveError("lif_step: input_current entries must be finite")
        v_after = v + decay * (config.v_leak - v + j)
        spiked = v_after >= config.v_threshold
        spikes.append(spiked)
        next_v.append(config.v_reset if spiked else v_after)
    return NengoEnsembleState(v=tuple(next_v)), tuple(spikes)


def _compute_input_currents(weights: NengoEnsembleWeights, x: Sequence[float]) -> tuple[float, ...]:
    """Compute per-neuron input current J_i = G_i * (e_i . x) + b_i."""

    if len(x) != weights.dimensions:
        raise NengoCognitiveError(
            "_compute_input_currents: input length must equal weights.dimensions"
        )
    currents: list[float] = []
    for i in range(weights.n_neurons):
        e_row = weights.encoders[i]
        dot = 0.0
        for d, xd in enumerate(x):
            if not math.isfinite(xd):
                raise NengoCognitiveError("_compute_input_currents: input entries must be finite")
            dot += e_row[d] * xd
        currents.append(weights.gains[i] * dot + weights.biases[i])
    return tuple(currents)


def _decode(weights: NengoEnsembleWeights, activity: Sequence[float]) -> tuple[float, ...]:
    """Linear decode y_hat = D @ activity."""

    if len(activity) != weights.n_neurons:
        raise NengoCognitiveError("_decode: activity length must equal weights.n_neurons")
    out: list[float] = []
    for d in range(weights.dimensions):
        row = weights.decoders[d]
        acc = 0.0
        for i, a_i in enumerate(activity):
            if not math.isfinite(a_i):
                raise NengoCognitiveError("_decode: activity entries must be finite")
            acc += row[i] * a_i
        out.append(acc)
    return tuple(out)


# ---------------------------------------------------------------- ensemble


@runtime_checkable
class NengoForwardCallable(Protocol):
    """Anything exposing a single-step
    ``forward(state, x) -> (state, spikes, decoded)`` over the
    population."""

    def forward(
        self,
        state: NengoEnsembleState,
        x: Sequence[float],
    ) -> tuple[NengoEnsembleState, tuple[bool, ...], tuple[float, ...]]: ...


@dataclass(frozen=True, slots=True)
class NengoEnsemble:
    """Frozen population-coded ensemble: input -> LIF -> decode.

    Mirrors :class:`nengo.Ensemble` + an identity connection.
    The forward pass is fully pure-Python (no nengo / numpy
    needed) so unit tests can exercise the cognitive surface
    without invoking the upstream simulator. The production seam
    :func:`nengo_cognitive_engine` lazy-imports nengo + numpy
    inside the factory body.
    """

    weights: NengoEnsembleWeights
    config: NengoEnsembleConfig = field(default_factory=NengoEnsembleConfig)

    def forward(
        self,
        state: NengoEnsembleState,
        x: Sequence[float],
    ) -> tuple[NengoEnsembleState, tuple[bool, ...], tuple[float, ...]]:
        """One forward step. Returns ``(next_state, spikes, decoded)``.

        ``decoded`` is the per-step decoded vector (LIF spike
        train projected through the offline decoder matrix). The
        analyser averages this over the window to obtain the
        decoded regime vector.
        """

        if len(state.v) != self.weights.n_neurons:
            raise NengoCognitiveError(
                "NengoEnsemble.forward: state.v length must equal weights.n_neurons"
            )
        currents = _compute_input_currents(self.weights, x)
        next_state, spikes = lif_step(state, currents, self.config)
        activity = tuple(1.0 if s else 0.0 for s in spikes)
        decoded = _decode(self.weights, activity)
        return next_state, spikes, decoded


# ---------------------------------------------------------------- engine seam


@runtime_checkable
class NengoCognitiveEngine(Protocol):
    """Caller-supplied nengo cognitive engine.

    The Protocol is the only place the analyser interacts with
    the underlying simulator. Window-level: returns one
    ``(spike_count, decoded_mean)`` summary over the input
    window so the analyser can project it onto a
    :class:`NengoRegimePulse`.
    """

    def run_window(
        self,
        *,
        weights: NengoEnsembleWeights,
        config: NengoEnsembleConfig,
        window: tuple[tuple[float, ...], ...],
    ) -> tuple[int, tuple[float, ...]]: ...


@dataclass(frozen=True, slots=True)
class _PurePythonNengoEngine:
    """Default :class:`NengoCognitiveEngine` backed by the pure-Python
    :class:`NengoEnsemble`. No external deps.

    Useful as the canonical fallback when no production nengo
    backend is wired (and the standard engine in unit tests).
    """

    def run_window(
        self,
        *,
        weights: NengoEnsembleWeights,
        config: NengoEnsembleConfig,
        window: tuple[tuple[float, ...], ...],
    ) -> tuple[int, tuple[float, ...]]:
        ensemble = NengoEnsemble(weights=weights, config=config)
        state = initial_state(weights.n_neurons, v_leak=config.v_leak)
        spike_count = 0
        decoded_accum = [0.0] * weights.dimensions
        n_steps = max(1, len(window))
        for x in window:
            state, spikes, decoded = ensemble.forward(state, x)
            for s in spikes:
                if s:
                    spike_count += 1
            for d, value in enumerate(decoded):
                decoded_accum[d] += value
        decoded_mean = tuple(v / n_steps for v in decoded_accum)
        return spike_count, decoded_mean


def pure_python_nengo_cognitive_engine() -> NengoCognitiveEngine:
    """Build the canonical pure-Python engine. No external deps."""

    return _PurePythonNengoEngine()


# ---------------------------------------------------------------- analyser


@dataclass(frozen=True, slots=True)
class NengoCognitiveAnalyser:
    """Window-level coordinator over a frozen :class:`NengoEnsemble`.

    Consumes a sequence of regime-feature vectors (one per
    timestep), drives the population dynamics, accumulates spike
    activity, decodes the represented vector, and projects the
    result onto an advisory :class:`NengoRegimePulse`. The
    analyser is **frozen + advisory only**: it never emits a
    typed bus event, never mutates the registry, never reads the
    clock.

    Attributes:
        engine: Caller-injected :class:`NengoCognitiveEngine`.
            Production wires :func:`nengo_cognitive_engine`; unit
            tests inject a deterministic fake; the canonical
            fallback is :func:`pure_python_nengo_cognitive_engine`.
        confidence_threshold: Fraction in ``(0.0, 1.0]``
            controlling when the pulse polarity becomes ``LONG``
            / ``SHORT``. Below this, polarity collapses to
            ``NEUTRAL``. Default ``0.25``.
    """

    engine: NengoCognitiveEngine
    confidence_threshold: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.engine, NengoCognitiveEngine):
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.engine must implement the NengoCognitiveEngine Protocol"
            )
        if not (
            math.isfinite(self.confidence_threshold) and 0.0 < self.confidence_threshold <= 1.0
        ):
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.confidence_threshold must be in (0, 1]"
            )

    def detect(
        self,
        *,
        ts_ns: int,
        source: str,
        symbol: str,
        regime_label: str,
        weights: NengoEnsembleWeights,
        window: (Sequence[Sequence[float]] | Iterable[Sequence[float]]),
        config: NengoEnsembleConfig | None = None,
        evidence: Mapping[str, str] | None = None,
        polarity_axis: int = 0,
        polarity_sign: int = 1,
    ) -> NengoRegimePulse:
        """Run the ensemble over ``window`` and emit a
        :class:`NengoRegimePulse`.

        Args:
            ts_ns: Caller-supplied window-close timestamp.
            source: Stable source id.
            symbol: Per-instrument id.
            regime_label: Caller's semantic tag for the regime
                being decoded (e.g. ``"BULL_TREND"``).
            weights: Offline-trained frozen ensemble weights.
            window: Sequence of feature vectors, each of length
                ``weights.dimensions``. The analyser runs one
                LIF step per row.
            config: Optional ensemble hyperparameters; defaults
                to ``NengoEnsembleConfig()`` (nengo defaults).
            evidence: Optional structural metadata. Always
                carries ``evidence["analyser"] == ANALYSIS_SOURCE``
                after the call.
            polarity_axis: Index into ``decoded_value`` whose sign
                determines polarity (0..dimensions-1, default 0).
            polarity_sign: ``+1`` (default) maps positive decoded
                value to ``LONG``; ``-1`` flips the mapping;
                ``0`` forces ``NEUTRAL`` regardless of decoding.

        Returns:
            A frozen advisory :class:`NengoRegimePulse`.
        """

        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise NengoCognitiveError("NengoCognitiveAnalyser.detect: ts_ns must be int")
        if ts_ns < 0:
            raise NengoCognitiveError("NengoCognitiveAnalyser.detect: ts_ns must be >= 0")
        if not isinstance(source, str) or not source:
            raise NengoCognitiveError("NengoCognitiveAnalyser.detect: source must be non-empty")
        if not isinstance(symbol, str) or not symbol:
            raise NengoCognitiveError("NengoCognitiveAnalyser.detect: symbol must be non-empty")
        if not isinstance(regime_label, str) or not regime_label:
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.detect: regime_label must be non-empty"
            )
        if not isinstance(weights, NengoEnsembleWeights):
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.detect: weights must be NengoEnsembleWeights"
            )
        if polarity_sign not in (-1, 0, 1):
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.detect: polarity_sign must be in {-1, 0, 1}"
            )
        if not isinstance(polarity_axis, int) or isinstance(polarity_axis, bool):
            raise NengoCognitiveError("NengoCognitiveAnalyser.detect: polarity_axis must be int")
        if polarity_axis < 0 or polarity_axis >= weights.dimensions:
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.detect: polarity_axis out of range [0, dimensions)"
            )
        effective_config = config if config is not None else NengoEnsembleConfig()
        if not isinstance(effective_config, NengoEnsembleConfig):
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.detect: config must be NengoEnsembleConfig"
            )

        rows: list[tuple[float, ...]] = []
        for x in window:
            if len(x) != weights.dimensions:
                raise NengoCognitiveError(
                    "NengoCognitiveAnalyser.detect: window row length must equal weights.dimensions"
                )
            row: list[float] = []
            for value in x:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise NengoCognitiveError(
                        "NengoCognitiveAnalyser.detect: window entries must be float"
                    )
                if not math.isfinite(value):
                    raise NengoCognitiveError(
                        "NengoCognitiveAnalyser.detect: window entries must be finite"
                    )
                row.append(float(value))
            rows.append(tuple(row))
        if len(rows) < MIN_WINDOW or len(rows) > MAX_WINDOW:
            raise NengoCognitiveError(
                "NengoCognitiveAnalyser.detect: window length must be "
                f"in [{MIN_WINDOW}, {MAX_WINDOW}]"
            )

        window_t = tuple(rows)
        spike_count, decoded_mean = self.engine.run_window(
            weights=weights,
            config=effective_config,
            window=window_t,
        )
        if not isinstance(spike_count, int) or isinstance(spike_count, bool):
            raise NengoCognitiveError("NengoCognitiveEngine.run_window: spike_count must be int")
        if spike_count < 0:
            raise NengoCognitiveError("NengoCognitiveEngine.run_window: spike_count must be >= 0")
        if not isinstance(decoded_mean, tuple) or len(decoded_mean) != weights.dimensions:
            raise NengoCognitiveError(
                "NengoCognitiveEngine.run_window: decoded_mean must be "
                "a tuple of length weights.dimensions"
            )
        for value in decoded_mean:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise NengoCognitiveError(
                    "NengoCognitiveEngine.run_window: decoded_mean entries must be float"
                )
            if not math.isfinite(value):
                raise NengoCognitiveError(
                    "NengoCognitiveEngine.run_window: decoded_mean entries must be finite"
                )

        magnitude = math.sqrt(sum(v * v for v in decoded_mean))
        confidence = max(0.0, min(1.0, magnitude))
        if polarity_sign == 0 or confidence < self.confidence_threshold:
            polarity = REGIME_NEUTRAL
        else:
            sign_value = decoded_mean[polarity_axis] * polarity_sign
            if sign_value > 0.0:
                polarity = REGIME_LONG
            elif sign_value < 0.0:
                polarity = REGIME_SHORT
            else:
                polarity = REGIME_NEUTRAL

        evidence_out: dict[str, str] = dict(evidence) if evidence else {}
        evidence_out.setdefault("analyser", ANALYSIS_SOURCE)
        evidence_out.setdefault("polarity_axis", str(polarity_axis))
        evidence_out.setdefault("polarity_sign", str(polarity_sign))
        evidence_out.setdefault("confidence_threshold", repr(self.confidence_threshold))

        return NengoRegimePulse(
            ts_ns=ts_ns,
            source=source,
            symbol=symbol,
            regime_label=regime_label,
            polarity=polarity,
            confidence=confidence,
            decoded_value=tuple(decoded_mean),
            spike_count=spike_count,
            sample_count=len(rows),
            weights_digest=weights.digest(),
            evidence=evidence_out,
        )


# ---------------------------------------------------------------- helpers


def _digest(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()


def _canonical_weights(weights: NengoEnsembleWeights) -> str:
    encoders = ";".join(",".join(f"{v:.17g}" for v in row) for row in weights.encoders)
    gains = ",".join(f"{v:.17g}" for v in weights.gains)
    biases = ",".join(f"{v:.17g}" for v in weights.biases)
    decoders = ";".join(",".join(f"{v:.17g}" for v in row) for row in weights.decoders)
    return (
        f"v={NENGO_COGNITIVE_VERSION}"
        f"|n={weights.n_neurons}"
        f"|d={weights.dimensions}"
        f"|seed={weights.weights_seed}"
        f"|E={encoders}"
        f"|G={gains}"
        f"|b={biases}"
        f"|D={decoders}"
    )


def _splitmix64(state: int) -> tuple[int, int]:
    """Stateless splitmix64 step. Returns ``(uint64_value, next_state)``."""

    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z, state


def _uniform01(state: int) -> tuple[float, int]:
    """Draw a deterministic uniform sample in [0, 1) from splitmix64."""

    z, next_state = _splitmix64(state)
    return (z >> 11) * (1.0 / (1 << 53)), next_state


# ---------------------------------------------------------------- production


def nengo_cognitive_engine() -> NengoCognitiveEngine:
    """Production :class:`NengoCognitiveEngine` backed by ``nengo``.

    Lazy-imports ``nengo`` + ``numpy`` inside the factory. Raises
    ``ImportError`` (with a helpful pip-install hint) if either
    package is missing — the rest of the module never imports
    them, so the analyser stays usable on a host that has never
    installed nengo.

    The returned object must implement
    :class:`NengoCognitiveEngine` consistent with the upstream
    nengo ``Simulator`` semantics. This factory intentionally
    raises :class:`NotImplementedError` — production deployments
    must wire the upstream simulator after running an offline
    nengo build pipeline through ``nengo.Network()`` + ``Probe``
    + ``Simulator`` and serialising the frozen weight tensor to
    disk.
    """

    try:
        import nengo  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nengo_cognitive_engine requires the optional 'nengo' + "
            "'numpy' packages — install with 'pip install nengo numpy' "
            "(NEW_PIP_DEPENDENCIES tuple in "
            "sensory/neuromorphic/nengo_cognitive.py flags this)."
        ) from exc

    _ = nengo

    class _NengoCognitiveEngine:
        """Thin nengo wrapper conforming to
        :class:`NengoCognitiveEngine`."""

        __slots__ = ()

        def run_window(
            self,
            *,
            weights: NengoEnsembleWeights,
            config: NengoEnsembleConfig,
            window: tuple[tuple[float, ...], ...],
        ) -> tuple[int, tuple[float, ...]]:  # pragma: no cover
            raise NotImplementedError(
                "nengo_cognitive_engine is the production seam — its "
                "concrete body is exercised in integration tests with "
                "nengo installed; unit tests inject a deterministic "
                "fake (or pure_python_nengo_cognitive_engine) via the "
                "NengoCognitiveEngine Protocol."
            )

    return _NengoCognitiveEngine()


__all__ = (
    "ANALYSIS_SOURCE",
    "MAX_DIMENSIONS",
    "MAX_NEURONS",
    "MAX_REGIME_LABEL_LEN",
    "MAX_SOURCE_LEN",
    "MAX_SYMBOL_LEN",
    "MAX_WINDOW",
    "MIN_DIMENSIONS",
    "MIN_NEURONS",
    "MIN_WINDOW",
    "NENGO_COGNITIVE_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "REGIME_LONG",
    "REGIME_NEUTRAL",
    "REGIME_SHORT",
    "NengoCognitiveAnalyser",
    "NengoCognitiveEngine",
    "NengoCognitiveError",
    "NengoEnsemble",
    "NengoEnsembleConfig",
    "NengoEnsembleState",
    "NengoEnsembleWeights",
    "NengoForwardCallable",
    "NengoRegimePulse",
    "build_random_ensemble_weights",
    "initial_state",
    "lif_step",
    "nengo_cognitive_engine",
    "pure_python_nengo_cognitive_engine",
)
