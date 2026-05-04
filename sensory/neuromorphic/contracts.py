"""Value types for the sensory.neuromorphic perimeter (NEUR-01..03).

The neuromorphic sensors are pure feature extractors. They consume a
window of structured observations (trades, metrics, decision-trace
rows) and emit one of three typed value classes:

* :class:`PulseSignal` — NEUR-01, ``indira_signal.py``. Direction +
  intensity of the market microstructure pulse. Indira (intelligence
  engine) reads these and projects them onto the BeliefState.
* :class:`AnomalyPulse` — NEUR-02, ``dyon_anomaly.py``. Anomaly
  observation in a numeric metric stream. Dyon (system engine) decides
  whether to escalate a HazardEvent. The sensor never emits a hazard
  by itself.
* :class:`RiskPulse` — NEUR-03, ``governance_risk.py``. Operator-side
  risk perception derived from the decision audit trail. Governance
  consumes it as one input among many.

All three are frozen + slotted dataclasses (INV-15 deterministic
replay) and emit a caller-supplied ``ts_ns`` rather than reading the
wall clock.

Authority discipline (per docs/directory_tree.md sensory/ contract):

* No module under :mod:`sensory.neuromorphic` may import an engine,
  write to the audit ledger, or mutate the SystemMode FSM.
* Outputs are typed values only.
* Validation in ``__post_init__`` rejects malformed inputs (NaN,
  negative magnitudes, empty source ids) so the perimeter is the
  single place where a bad observation is discovered.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

# Polarity sign convention (NEUR-01). Long-only systems are the
# operator's prerogative; the sensor emits the raw direction and
# leaves filtering to downstream policy.
POLARITY_LONG: str = "LONG"
POLARITY_SHORT: str = "SHORT"
POLARITY_NEUTRAL: str = "NEUTRAL"

_POLARITIES: frozenset[str] = frozenset(
    {POLARITY_LONG, POLARITY_SHORT, POLARITY_NEUTRAL}
)


def _is_finite_nonneg(x: float) -> bool:
    """Return True iff ``x`` is finite and ``>= 0``.

    Using ``not (x >= 0)`` rather than ``x < 0`` is intentional —
    NaN/-Inf both fail the predicate (IEEE 754). ``math.isfinite``
    additionally rejects ``+inf`` so neuromorphic intensities cannot
    saturate the BeliefState by smuggling an infinite magnitude.
    """
    return math.isfinite(x) and x >= 0.0


@dataclass(frozen=True, slots=True)
class PulseSignal:
    """NEUR-01 — pulse / microstructure signal.

    Emitted by :func:`sensory.neuromorphic.indira_signal.extract_pulse`
    after observing a window of trades. The pulse is the directional
    component (``polarity``) plus a normalized magnitude
    (``intensity`` in ``[0.0, 1.0]``).

    Attributes:
        ts_ns: Window-close timestamp in nanoseconds (caller-supplied,
            INV-15).
        source: Stable source identifier (e.g. ``"BINANCE"``,
            ``"PUMPFUN"``). Empty string is rejected.
        symbol: Instrument identifier (e.g. ``"BTCUSDT"``). Empty
            string is rejected because every pulse is per-instrument.
        polarity: One of ``LONG`` / ``SHORT`` / ``NEUTRAL``.
        intensity: Normalized pulse magnitude in ``[0.0, 1.0]``. NaN,
            +Inf, and negatives are rejected.
        sample_count: Number of trades / quotes that contributed to
            the window. Must be ``>= 1``; a sensor cannot emit a
            pulse from an empty window.
        evidence: Free-form structural metadata (no PII / secrets).
    """

    ts_ns: int
    source: str
    symbol: str
    polarity: str
    intensity: float
    sample_count: int
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("PulseSignal.source must be non-empty")
        if not self.symbol:
            raise ValueError("PulseSignal.symbol must be non-empty")
        if self.polarity not in _POLARITIES:
            raise ValueError(
                "PulseSignal.polarity must be one of "
                f"{sorted(_POLARITIES)}"
            )
        if not (math.isfinite(self.intensity) and 0.0 <= self.intensity <= 1.0):
            raise ValueError(
                "PulseSignal.intensity must be finite in [0.0, 1.0]"
            )
        if self.sample_count < 1:
            raise ValueError(
                "PulseSignal.sample_count must be >= 1"
            )


@dataclass(frozen=True, slots=True)
class AnomalyPulse:
    """NEUR-02 — anomaly perception in a numeric metric stream.

    Emitted by :func:`sensory.neuromorphic.dyon_anomaly.detect_anomaly`
    after observing a fixed-size window of metric samples. The pulse
    captures how far the latest sample sits from the window's mean,
    expressed as ``z_score`` (signed) plus a clipped severity in
    ``[0.0, 1.0]``.

    The sensor never emits a HazardEvent. Dyon is responsible for
    deciding whether ``severity`` warrants an escalation — that
    keeps the sensor side replay-pure.

    Attributes:
        ts_ns: Sample timestamp (INV-15, caller-supplied).
        source: Stable identifier of the metric stream
            (e.g. ``"system.fast_risk_cache.lag_ns"``).
        anomaly_kind: Categorical label so consumers can route by
            type. Empty string is rejected.
        z_score: Signed deviation in std-deviation units. NaN/Inf
            are rejected.
        severity: Clipped magnitude in ``[0.0, 1.0]``. ``0.0`` means
            no anomaly, ``1.0`` is the saturated upper bound.
        window_size: Window size used for the mean / std estimate.
            Must be ``>= 2`` (sample variance needs at least 2
            samples).
        evidence: Free-form structural metadata.
    """

    ts_ns: int
    source: str
    anomaly_kind: str
    z_score: float
    severity: float
    window_size: int
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("AnomalyPulse.source must be non-empty")
        if not self.anomaly_kind:
            raise ValueError(
                "AnomalyPulse.anomaly_kind must be non-empty"
            )
        if not math.isfinite(self.z_score):
            raise ValueError(
                "AnomalyPulse.z_score must be finite"
            )
        if not (math.isfinite(self.severity) and 0.0 <= self.severity <= 1.0):
            raise ValueError(
                "AnomalyPulse.severity must be finite in [0.0, 1.0]"
            )
        if self.window_size < 2:
            raise ValueError(
                "AnomalyPulse.window_size must be >= 2"
            )


@dataclass(frozen=True, slots=True)
class RiskPulse:
    """NEUR-03 — governance-risk perception.

    Emitted by :func:`sensory.neuromorphic.governance_risk.assess_risk`
    after observing a window of recent decision-trace records. The
    sensor captures *operator-side* risk drift: rising rejection
    rates, hazard density, or unauthorized-directive counts. The
    output is a typed value — Governance is the sole authority that
    decides whether to act on it.

    Attributes:
        ts_ns: Window-close timestamp (INV-15, caller-supplied).
        source: Stable source identifier (e.g.
            ``"governance.decision_audit"``). Empty string is
            rejected.
        risk_kind: Categorical label (``"REJECT_RATE"`` /
            ``"HAZARD_DENSITY"`` / ``"UNAUTH_DIRECTIVE_RATE"`` /
            free-form). Empty string is rejected.
        risk_score: Magnitude in ``[0.0, 1.0]``. NaN/Inf are
            rejected.
        sample_count: Decision rows that contributed. Must be
            ``>= 1``.
        evidence: Free-form structural metadata.
    """

    ts_ns: int
    source: str
    risk_kind: str
    risk_score: float
    sample_count: int
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("RiskPulse.source must be non-empty")
        if not self.risk_kind:
            raise ValueError("RiskPulse.risk_kind must be non-empty")
        if not (
            math.isfinite(self.risk_score)
            and 0.0 <= self.risk_score <= 1.0
        ):
            raise ValueError(
                "RiskPulse.risk_score must be finite in [0.0, 1.0]"
            )
        if self.sample_count < 1:
            raise ValueError(
                "RiskPulse.sample_count must be >= 1"
            )


__all__ = [
    "POLARITY_LONG",
    "POLARITY_NEUTRAL",
    "POLARITY_SHORT",
    "AnomalyPulse",
    "PulseSignal",
    "RiskPulse",
    "_is_finite_nonneg",
]
