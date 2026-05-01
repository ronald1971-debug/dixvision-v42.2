"""GOV-CP-09 -- Continuous Drift Oracle (Reviewer #4 finding 3, Reviewer #5 AUTO safeguards).

Reviewer #4 framed the gap precisely: the B30 Belief-State unification
lint enforces *direction* (every intelligence path flows through
BeliefState) but does not *produce* the System Coherence Layer outputs
(drift score, causal-distribution divergence, mode-validation signal).
Reviewer #5 named the consequence: ``AUTO`` mode relaxes operator
oversight to exception-only; without a continuous composite drift score
there is no "exception" to trigger.

The :class:`DriftOracle` provides a deterministic, ledger-anchored
composite drift score over four axes:

1. **Model drift** -- BeliefState delta variance over the recent
   window. High values mean the belief state is being whipsawed.
2. **Execution drift** -- realised vs expected slippage, normalised.
3. **Latency drift** -- hot-path p99 excursion above its training
   baseline.
4. **Causal drift** -- decision-trace ``why`` distribution shift
   (Jensen-Shannon-style symmetric KL on the recent window vs the
   reference window).

Each axis is a unit-scaled float in ``[0.0, 1.0]``. The oracle stores a
bounded ring of recent samples and exposes:

* ``observe(...)`` -- append a new ``DriftSample`` and write a
  ``DRIFT_OBSERVATION`` ledger row. INV-15 determinism: same inputs ->
  same ledger row -> same composite score.
* ``composite()`` -- weighted L1 sum of the four axes over the most
  recent ``window_size`` samples (default ``32``). Pure function of the
  ring buffer.
* ``check(target_mode_name)`` -- returns ``(False, "DRIFT_ORACLE_*")``
  if a forward transition into ``AUTO`` would proceed with a composite
  score above ``threshold`` (default ``0.25``), insufficient samples
  for the window, or no observations at all.

Hookup: :class:`StateTransitionManager` accepts an optional
``drift_oracle`` parameter (same shape as ``promotion_gates``); the
gate check runs before the policy gate so the rejection_code surfaces
the actual blocker. Auto-degradation (forced ``AUTO -> CANARY`` on
sustained drift) is the follow-up PR; this module ships only the
forward-gate refusal.

This module is the **only** writer of ``DRIFT_OBSERVATION`` ledger
rows.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)

LEDGER_KIND_DRIFT_OBSERVATION = "DRIFT_OBSERVATION"

# Forward modes whose entry is gated by the drift oracle. Today only
# AUTO -- LIVE entry is gated by promotion_gates only. CANARY and LIVE
# can be added here once their drift baselines are calibrated; keeping
# the set narrow today reduces the surface for false-positive blocks
# during initial bring-up.
_GATED_FORWARD_TARGETS: frozenset[str] = frozenset({"AUTO"})

# Default composite threshold above which AUTO entry is refused. Set
# conservatively -- 0.25 means "more than a quarter of the way to
# the maximum measurable drift on any axis, on average". Operators
# tune this per-deployment in ``docs/drift_thresholds.yaml`` (follow-up
# PR will load from yaml; today the default is a code constant).
DEFAULT_COMPOSITE_THRESHOLD: float = 0.25

# Default minimum number of samples required before the oracle
# considers itself "ready" to gate. Below this, ``check`` refuses with
# code ``DRIFT_ORACLE_INSUFFICIENT_SAMPLES``. The intent is that a
# fresh oracle cannot rubber-stamp AUTO entry; AUTO must be reached
# through a window of observed stability.
DEFAULT_MIN_SAMPLES: int = 32

# Default ring-buffer capacity. Picked to comfortably exceed
# ``DEFAULT_MIN_SAMPLES`` so a healthy oracle always has a full window
# to reason over.
DEFAULT_WINDOW_SIZE: int = 32


def _clamp(value: float) -> float:
    """Clamp ``value`` into ``[0.0, 1.0]`` with NaN -> 1.0 (worst case)."""

    if value != value:  # NaN
        return 1.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True, slots=True)
class DriftSample:
    """One immutable observation across the four drift axes.

    All fields are unit-scaled floats in ``[0.0, 1.0]``. Producers are
    responsible for normalising raw measurements (variance, latency
    excursion, KL divergence, etc.) into the unit interval before
    handing the sample to the oracle. Out-of-range values are clamped
    on construction via :meth:`from_raw`.
    """

    ts_ns: int
    model_drift: float
    execution_drift: float
    latency_drift: float
    causal_drift: float

    @staticmethod
    def from_raw(
        *,
        ts_ns: int,
        model_drift: float,
        execution_drift: float,
        latency_drift: float,
        causal_drift: float,
    ) -> DriftSample:
        """Build a sample, clamping each axis into ``[0.0, 1.0]``."""

        return DriftSample(
            ts_ns=ts_ns,
            model_drift=_clamp(model_drift),
            execution_drift=_clamp(execution_drift),
            latency_drift=_clamp(latency_drift),
            causal_drift=_clamp(causal_drift),
        )


# Default per-axis weights. Sum to 1.0 so the composite is itself in
# ``[0.0, 1.0]``. Chosen so that model drift and causal drift carry
# slightly more weight than execution and latency -- the SCE outputs
# reviewer #4 named are precisely model + causal; execution and
# latency are pre-existing signals already covered by hazard sensors.
DEFAULT_AXIS_WEIGHTS: Mapping[str, float] = {
    "model_drift": 0.35,
    "execution_drift": 0.15,
    "latency_drift": 0.15,
    "causal_drift": 0.35,
}


def _validate_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """Validate per-axis weights: positive, summing to 1.0 within tol."""

    expected = {"model_drift", "execution_drift", "latency_drift", "causal_drift"}
    if set(weights.keys()) != expected:
        raise ValueError(f"weights must cover exactly {sorted(expected)}, got {sorted(weights)}")
    for name, weight in weights.items():
        if weight < 0.0:
            raise ValueError(f"weight for {name!r} is negative: {weight}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {total}")
    return dict(weights)


class DriftOracle:
    """Deterministic, ledger-anchored composite drift oracle.

    Construction is cheap; the oracle accepts samples via
    :meth:`observe` and exposes the composite score via
    :meth:`composite`. The forward gate :meth:`check` is what
    :class:`StateTransitionManager` calls before approving a transition
    into ``AUTO``.
    """

    name: str = "drift_oracle"
    spec_id: str = "GOV-CP-09"

    def __init__(
        self,
        *,
        ledger: LedgerAuthorityWriter,
        threshold: float = DEFAULT_COMPOSITE_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        window_size: int = DEFAULT_WINDOW_SIZE,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if window_size < min_samples:
            raise ValueError(f"window_size ({window_size}) must be >= min_samples ({min_samples})")

        self._ledger = ledger
        self._threshold = threshold
        self._min_samples = min_samples
        self._weights = _validate_weights(weights or DEFAULT_AXIS_WEIGHTS)
        self._samples: deque[DriftSample] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """The composite threshold above which AUTO entry is refused."""

        return self._threshold

    @property
    def min_samples(self) -> int:
        """Minimum samples required for the oracle to gate."""

        return self._min_samples

    def sample_count(self) -> int:
        """Number of samples currently in the ring buffer."""

        return len(self._samples)

    def samples(self) -> tuple[DriftSample, ...]:
        """Snapshot of the current ring buffer (oldest first)."""

        return tuple(self._samples)

    def observe(self, sample: DriftSample) -> None:
        """Append ``sample`` to the ring buffer and ledger.

        Writes a single ``DRIFT_OBSERVATION`` row whose payload carries
        the four axis values plus the running composite. The composite
        is recomputed *after* the new sample is added so the ledger row
        records the score the gate would see right now.
        """

        self._samples.append(sample)
        composite = self._composite_unchecked()
        payload: Mapping[str, str] = {
            "model_drift": _format(sample.model_drift),
            "execution_drift": _format(sample.execution_drift),
            "latency_drift": _format(sample.latency_drift),
            "causal_drift": _format(sample.causal_drift),
            "composite": _format(composite),
            "sample_count": str(len(self._samples)),
        }
        self._ledger.append(
            ts_ns=sample.ts_ns,
            kind=LEDGER_KIND_DRIFT_OBSERVATION,
            payload=payload,
        )

    def composite(self) -> float:
        """Return the weighted composite drift score over the window.

        Returns ``0.0`` if the buffer is empty. The result is the mean
        per-sample weighted sum across all four axes, then clamped into
        ``[0.0, 1.0]`` for paranoia (with valid weights and clamped
        samples the clamp is a no-op).
        """

        return self._composite_unchecked()

    def check(self, target_mode_name: str) -> tuple[bool, str]:
        """Verify the oracle permits a forward transition.

        Returns ``(True, "")`` for non-gated targets unconditionally.
        For gated targets (``AUTO``) returns:

        * ``(False, "DRIFT_ORACLE_INSUFFICIENT_SAMPLES")`` if the
          buffer holds fewer than ``min_samples`` observations.
        * ``(False, "DRIFT_ORACLE_THRESHOLD_BREACH")`` if composite
          score is at or above ``threshold``.
        * ``(True, "")`` otherwise.
        """

        if target_mode_name not in _GATED_FORWARD_TARGETS:
            return True, ""
        if len(self._samples) < self._min_samples:
            return False, "DRIFT_ORACLE_INSUFFICIENT_SAMPLES"
        if self._composite_unchecked() >= self._threshold:
            return False, "DRIFT_ORACLE_THRESHOLD_BREACH"
        return True, ""

    def replay_from_ledger(self) -> None:
        """Reconstitute the ring buffer by walking the ledger.

        Adopts the most recent ``window_size`` ``DRIFT_OBSERVATION``
        rows in ledger order (INV-15 determinism). Older rows are
        discarded -- the buffer's capacity is the contract.
        """

        rebuilt: deque[DriftSample] = deque(maxlen=self._samples.maxlen)
        for entry in self._ledger.read():
            if entry.kind != LEDGER_KIND_DRIFT_OBSERVATION:
                continue
            payload = entry.payload
            rebuilt.append(
                DriftSample(
                    ts_ns=entry.ts_ns,
                    model_drift=float(payload["model_drift"]),
                    execution_drift=float(payload["execution_drift"]),
                    latency_drift=float(payload["latency_drift"]),
                    causal_drift=float(payload["causal_drift"]),
                )
            )
        self._samples = rebuilt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _composite_unchecked(self) -> float:
        """Composite over the current buffer; ``0.0`` if empty."""

        if not self._samples:
            return 0.0
        return _clamp(_mean_weighted_sum(self._samples, self._weights))


def _mean_weighted_sum(samples: Iterable[DriftSample], weights: Mapping[str, float]) -> float:
    """Return the mean per-sample weighted sum across the four axes."""

    total = 0.0
    count = 0
    for s in samples:
        total += (
            s.model_drift * weights["model_drift"]
            + s.execution_drift * weights["execution_drift"]
            + s.latency_drift * weights["latency_drift"]
            + s.causal_drift * weights["causal_drift"]
        )
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _format(value: float) -> str:
    """Format a float with stable, deterministic precision (12 sig figs)."""

    return f"{value:.12f}"


__all__ = [
    "DEFAULT_AXIS_WEIGHTS",
    "DEFAULT_COMPOSITE_THRESHOLD",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_WINDOW_SIZE",
    "DriftOracle",
    "DriftSample",
    "LEDGER_KIND_DRIFT_OBSERVATION",
]
