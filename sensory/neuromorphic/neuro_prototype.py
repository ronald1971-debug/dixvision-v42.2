# ADAPTED FROM: brian-team/brian2 —
#   brian2/core/network.py (Network.run, time-stepped simulation)
#   brian2/equations/equations.py (NeuronGroup equation DSL)
#   brian2/monitors/spikemonitor.py (SpikeMonitor)
"""B-16 — Brian2 continuous-time LIF research prototype.

RESEARCH_SOURCE — this module is invoked **only from tests/ and offline
research scripts**. It is NEVER imported by any production runtime
module (no Indira/Dyon/governance/execution path consumes it). The
purpose is to prototype SNN topologies in Brian2's continuous-time DSL
*before* porting them to the production Norse-style discrete-step
detector in ``sensory/neuromorphic/snn_lif.py`` (B-14).

What this file provides
~~~~~~~~~~~~~~~~~~~~~~~

* :func:`continuous_time_lif_reference` — a pure-Python forward-Euler
  integrator of the Brian2 LIF equation

      dv/dt = (v_leak - v) / tau_mem        # sub-threshold leak
      I_t   = R * input_current(t)          # injected current term
      if v >= v_thresh: spike, v <- v_reset # threshold + reset

  with caller-supplied ``dt`` and step count. This is the **ground
  truth** that a production Norse-step detector should converge to as
  ``dt -> 0``. It is deterministic by construction (no PRNG, no clock,
  no I/O).

* :func:`norse_style_discrete_lif` — a parallel discrete-step
  recurrence

      v_{t+1} = v_t + (dt / tau_mem) * (v_leak - v_t + R * I_t)

  exactly equivalent to the canonical Norse ``lif_feed_forward_step``
  body. Pure-Python, zero deps. Identical seed-free spike train when
  ``dt`` is small.

* :func:`prototype_lif_market_signal` — the canonical comparison
  utility. Runs both integrators over the same input current trace,
  records spike counts + first-spike timings, and returns a frozen
  :class:`LIFComparisonReport` whose
  :meth:`LIFComparisonReport.is_consistent` predicate is the
  research-acceptance gate before a Norse cell is wired into a
  production detector.

* :func:`brian2_prototype_factory` — production seam. Raises
  :class:`NotImplementedError` until an offline workflow wires a real
  Brian2 ``Network`` + ``NeuronGroup`` + ``SpikeMonitor`` and projects
  results back into a :class:`LIFTrace`. ``NEW_PIP_DEPENDENCIES =
  ("brian2",)`` — brian2 is **never** imported at top level (RESEARCH
  classification → tests must remain importable without brian2 on the
  Python path).

Authority discipline
~~~~~~~~~~~~~~~~~~~~

* Sensory tier — no engine imports.
* Output is :class:`LIFTrace` / :class:`LIFComparisonReport` — frozen
  value objects. The module NEVER constructs typed bus events
  (``HazardEvent`` / ``SignalEvent`` / ``ExecutionIntent`` / ...).
* AST tests pin: no top-level ``brian2`` / ``torch`` / ``norse`` /
  ``numpy`` / ``scipy``; no ``random`` / ``time`` / ``datetime`` /
  ``asyncio`` / ``os`` / ``socket``; no engine cross-imports.
* RESEARCH_SOURCE — pinned by a sensory-tier AST test that confirms no
  module under ``sensory/`` / ``intelligence_engine/`` /
  ``execution_engine/`` / ``governance_engine/`` /
  ``system_engine/`` / ``evolution_engine/`` / ``learning_engine/``
  imports :mod:`sensory.neuromorphic.neuro_prototype`. The only
  permitted importers are ``tests/`` and ``offline/`` scripts.

Determinism (INV-15)
~~~~~~~~~~~~~~~~~~~~

No PRNG anywhere. The two integrators are deterministic forward-Euler
walks over a caller-supplied input trace, so three independent runs
over identical inputs produce a byte-identical
:class:`LIFComparisonReport.digest`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# ---------------------------------------------------------------- constants

NEURO_PROTOTYPE_VERSION: str = "neuro-prototype/v1"

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("brian2",)
"""Brian2 is a RESEARCH dependency. It is never imported at module
load — only lazy-imported inside :func:`brian2_prototype_factory` by
offline research scripts."""

MAX_STEPS: int = 100_000
"""Hard cap on the number of integration steps any single research
run may request. Prevents pathological inputs from running unbounded
inside a pytest worker."""

MAX_TRACE_LEN: int = MAX_STEPS

_DIGEST_BYTES: int = 16


# ----------------------------------------------------------------- errors


class NeuroPrototypeError(ValueError):
    """Raised by configuration / input validators in this module."""


# ----------------------------------------------------------------- params


@dataclass(frozen=True, slots=True)
class LIFParams:
    """Brian2-style LIF parameters (SI units).

    Attributes:
        tau_mem: Membrane time-constant in seconds.
        v_leak: Resting potential (volts, dimensionless here).
        v_thresh: Spike threshold.
        v_reset: Post-spike reset potential.
        r_input: Input resistance (scales the injected current term).
        dt: Integration step (seconds). Default 1 ms.
        refractory_steps: Number of steps the neuron is clamped to
            ``v_reset`` after a spike. ``0`` disables the refractory
            window.
    """

    tau_mem: float = 20.0e-3
    v_leak: float = 0.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    r_input: float = 1.0
    dt: float = 1.0e-3
    refractory_steps: int = 0

    def __post_init__(self) -> None:
        if not (self.tau_mem > 0 and math.isfinite(self.tau_mem)):
            raise NeuroPrototypeError(f"tau_mem must be > 0, got {self.tau_mem}")
        if not math.isfinite(self.v_leak):
            raise NeuroPrototypeError(f"v_leak must be finite, got {self.v_leak}")
        if not math.isfinite(self.v_thresh):
            raise NeuroPrototypeError(f"v_thresh must be finite, got {self.v_thresh}")
        if not math.isfinite(self.v_reset):
            raise NeuroPrototypeError(f"v_reset must be finite, got {self.v_reset}")
        if self.v_thresh <= self.v_reset:
            raise NeuroPrototypeError(
                f"v_thresh ({self.v_thresh}) must exceed v_reset ({self.v_reset})"
            )
        if not (self.r_input > 0 and math.isfinite(self.r_input)):
            raise NeuroPrototypeError(f"r_input must be > 0, got {self.r_input}")
        if not (self.dt > 0 and math.isfinite(self.dt)):
            raise NeuroPrototypeError(f"dt must be > 0, got {self.dt}")
        if self.refractory_steps < 0:
            raise NeuroPrototypeError(
                f"refractory_steps must be >= 0, got {self.refractory_steps}"
            )


# ----------------------------------------------------------------- trace


@dataclass(frozen=True, slots=True)
class LIFTrace:
    """Result of running a single LIF integrator over an input trace.

    Attributes:
        backend: Free-form name of the integrator that produced this
            trace. Conventionally ``"continuous"`` or ``"discrete"``.
        spikes: Boolean spike train of the same length as the input
            current trace.
        v_history: Membrane potential at the end of each step.
        spike_times_steps: Sorted indices of steps at which the neuron
            spiked. Convenience projection of ``spikes``.
        params: The :class:`LIFParams` used to produce the trace.
    """

    backend: str
    spikes: tuple[bool, ...]
    v_history: tuple[float, ...]
    spike_times_steps: tuple[int, ...]
    params: LIFParams

    @property
    def spike_count(self) -> int:
        return len(self.spike_times_steps)

    @property
    def first_spike_step(self) -> int | None:
        return self.spike_times_steps[0] if self.spike_times_steps else None

    def digest(self) -> str:
        """BLAKE2b-16 hex over a canonical text projection."""
        h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
        h.update(b"neuro-prototype/lif-trace/v1\n")
        h.update(self.backend.encode("utf-8"))
        h.update(b"\n")
        for s in self.spikes:
            h.update(b"1" if s else b"0")
        h.update(b"\n")
        for v in self.v_history:
            h.update(f"{v:.17g}\n".encode("ascii"))
        return h.hexdigest()


# ----------------------------------------------------------------- report


@dataclass(frozen=True, slots=True)
class LIFComparisonReport:
    """Side-by-side comparison of two LIF traces over the same input.

    The :meth:`is_consistent` predicate is the research-acceptance
    gate: a Norse-style discrete-step integrator should agree with the
    Brian2-style continuous-time reference on (i) the total spike count
    within ``count_tolerance`` and (ii) the first-spike timing within
    ``first_spike_step_tolerance`` steps.
    """

    continuous: LIFTrace
    discrete: LIFTrace
    count_tolerance: int = 0
    first_spike_step_tolerance: int = 0

    def __post_init__(self) -> None:
        if len(self.continuous.spikes) != len(self.discrete.spikes):
            raise NeuroPrototypeError(
                "continuous and discrete traces must have the same length, "
                f"got {len(self.continuous.spikes)} vs {len(self.discrete.spikes)}"
            )
        if self.count_tolerance < 0:
            raise NeuroPrototypeError(
                f"count_tolerance must be >= 0, got {self.count_tolerance}"
            )
        if self.first_spike_step_tolerance < 0:
            raise NeuroPrototypeError(
                "first_spike_step_tolerance must be >= 0, "
                f"got {self.first_spike_step_tolerance}"
            )

    @property
    def spike_count_delta(self) -> int:
        return abs(self.continuous.spike_count - self.discrete.spike_count)

    @property
    def first_spike_step_delta(self) -> int | None:
        a = self.continuous.first_spike_step
        b = self.discrete.first_spike_step
        if a is None or b is None:
            return None if a is None and b is None else None
        return abs(a - b)

    def is_consistent(self) -> bool:
        """``True`` iff both traces agree within configured tolerances."""
        if self.spike_count_delta > self.count_tolerance:
            return False
        a = self.continuous.first_spike_step
        b = self.discrete.first_spike_step
        if (a is None) != (b is None):
            return False
        if a is None or b is None:
            return True
        return abs(a - b) <= self.first_spike_step_tolerance

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
        h.update(b"neuro-prototype/lif-comparison/v1\n")
        h.update(self.continuous.digest().encode("ascii"))
        h.update(b"\n")
        h.update(self.discrete.digest().encode("ascii"))
        h.update(b"\n")
        h.update(f"{self.count_tolerance}\n".encode("ascii"))
        h.update(f"{self.first_spike_step_tolerance}\n".encode("ascii"))
        return h.hexdigest()


# --------------------------------------------------------------- helpers


def _validate_current(current: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(current, Sequence):
        raise NeuroPrototypeError(
            f"current must be a Sequence[float], got {type(current).__name__}"
        )
    if len(current) == 0:
        raise NeuroPrototypeError("current trace must be non-empty")
    if len(current) > MAX_TRACE_LEN:
        raise NeuroPrototypeError(
            f"current trace length {len(current)} exceeds MAX_TRACE_LEN={MAX_TRACE_LEN}"
        )
    out: list[float] = []
    for i, value in enumerate(current):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise NeuroPrototypeError(
                f"current[{i}] must be float, got {type(value).__name__}"
            )
        fv = float(value)
        if not math.isfinite(fv):
            raise NeuroPrototypeError(f"current[{i}] must be finite, got {fv}")
        out.append(fv)
    return tuple(out)


# ---------------------------------------------------- continuous reference


def continuous_time_lif_reference(
    *,
    current: Sequence[float],
    params: LIFParams,
    sub_steps: int = 10,
) -> LIFTrace:
    """Brian2-style continuous-time LIF integrator (forward Euler).

    Each input-trace step is split into ``sub_steps`` micro-steps so
    the continuous-time limit is approached as ``sub_steps -> inf``.
    Input current is held constant across sub-steps (zero-order hold)
    which matches Brian2's default ``NeuronGroup`` semantics when fed
    a ``TimedArray``.

    Args:
        current: Injected-current trace of length ``T``.
        params: LIF parameters (`dt` is the macro step).
        sub_steps: Number of micro-steps per macro step.

    Returns:
        :class:`LIFTrace` with ``T`` macro-step samples.
    """

    trace = _validate_current(current)
    if sub_steps < 1:
        raise NeuroPrototypeError(f"sub_steps must be >= 1, got {sub_steps}")
    if sub_steps > 1_000:
        raise NeuroPrototypeError(
            f"sub_steps must be <= 1000 (sanity cap), got {sub_steps}"
        )

    micro_dt = params.dt / sub_steps
    decay_micro = micro_dt / params.tau_mem

    v = params.v_reset
    refractory_left = 0
    spikes: list[bool] = []
    v_history: list[float] = []
    spike_steps: list[int] = []

    for t, i_t in enumerate(trace):
        spiked = False
        for _ in range(sub_steps):
            if refractory_left > 0:
                v = params.v_reset
                refractory_left -= 1
                continue
            dv = decay_micro * (params.v_leak - v + params.r_input * i_t)
            v = v + dv
            if v >= params.v_thresh:
                spiked = True
                v = params.v_reset
                refractory_left = params.refractory_steps * sub_steps
        spikes.append(spiked)
        v_history.append(v)
        if spiked:
            spike_steps.append(t)

    return LIFTrace(
        backend="continuous",
        spikes=tuple(spikes),
        v_history=tuple(v_history),
        spike_times_steps=tuple(spike_steps),
        params=params,
    )


# ---------------------------------------------------- discrete (Norse-style)


def norse_style_discrete_lif(
    *,
    current: Sequence[float],
    params: LIFParams,
) -> LIFTrace:
    """Norse-style discrete-time LIF step (one tick per macro step).

    Equivalent to a Brian2 simulation with ``sub_steps=1``. Provided
    explicitly so research scripts can run both side-by-side without
    re-implementing the recurrence.
    """

    trace = _validate_current(current)
    decay = params.dt / params.tau_mem

    v = params.v_reset
    refractory_left = 0
    spikes: list[bool] = []
    v_history: list[float] = []
    spike_steps: list[int] = []

    for t, i_t in enumerate(trace):
        if refractory_left > 0:
            v = params.v_reset
            refractory_left -= 1
            spikes.append(False)
            v_history.append(v)
            continue
        dv = decay * (params.v_leak - v + params.r_input * i_t)
        v = v + dv
        if v >= params.v_thresh:
            spikes.append(True)
            spike_steps.append(t)
            v = params.v_reset
            refractory_left = params.refractory_steps
        else:
            spikes.append(False)
        v_history.append(v)

    return LIFTrace(
        backend="discrete",
        spikes=tuple(spikes),
        v_history=tuple(v_history),
        spike_times_steps=tuple(spike_steps),
        params=params,
    )


# --------------------------------------------------------------- comparison


def prototype_lif_market_signal(
    *,
    current: Sequence[float],
    params: LIFParams,
    sub_steps: int = 10,
    count_tolerance: int = 0,
    first_spike_step_tolerance: int = 0,
) -> LIFComparisonReport:
    """Side-by-side LIF research check.

    Runs the Brian2-style continuous reference and the Norse-style
    discrete recurrence over the same input current, then projects
    them into a :class:`LIFComparisonReport`. Use the report's
    :meth:`LIFComparisonReport.is_consistent` predicate as the gate
    before promoting a research topology into a Norse-based detector.

    Args:
        current: Injected-current trace.
        params: LIF parameters shared by both integrators.
        sub_steps: Continuous-reference sub-step count.
        count_tolerance: Maximum permitted ``|c.count - d.count|``.
        first_spike_step_tolerance: Maximum permitted
            ``|c.first_spike - d.first_spike|`` (in macro steps).
    """

    cont = continuous_time_lif_reference(
        current=current, params=params, sub_steps=sub_steps
    )
    disc = norse_style_discrete_lif(current=current, params=params)
    return LIFComparisonReport(
        continuous=cont,
        discrete=disc,
        count_tolerance=count_tolerance,
        first_spike_step_tolerance=first_spike_step_tolerance,
    )


# --------------------------------------------------------- production seam


class Brian2PrototypeFactory(Protocol):
    """Research seam — a real Brian2 ``Network`` + ``SpikeMonitor``
    wrapper that returns an :class:`LIFTrace` projected from a Brian2
    run. Only invoked from offline research scripts."""

    def __call__(
        self,
        *,
        current: Sequence[float],
        params: LIFParams,
    ) -> LIFTrace: ...


def brian2_prototype_factory(
    *,
    current: Sequence[float],
    params: LIFParams,
) -> LIFTrace:
    """Lazy research-only Brian2 entry point.

    Raises:
        NotImplementedError: Always. A research script must wire
            ``brian2.NeuronGroup`` + ``brian2.SpikeMonitor`` +
            ``brian2.Network.run`` and project the recorded spikes
            into an :class:`LIFTrace` before calling this from
            ``offline/`` or ``tests/`` code.
    """

    raise NotImplementedError(
        "brian2_prototype_factory is a research seam. An offline workflow "
        "must wire brian2.NeuronGroup + brian2.SpikeMonitor + brian2.Network.run "
        f"and project results to LIFTrace before consuming "
        f"current[len={len(current)}] / params={params!r}."
    )


__all__ = [
    "Brian2PrototypeFactory",
    "LIFComparisonReport",
    "LIFParams",
    "LIFTrace",
    "MAX_STEPS",
    "MAX_TRACE_LEN",
    "NEURO_PROTOTYPE_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "NeuroPrototypeError",
    "brian2_prototype_factory",
    "continuous_time_lif_reference",
    "norse_style_discrete_lif",
    "prototype_lif_market_signal",
]
