"""sensory.neuromorphic — NEUR-01..03.

Three pure feature extractors that emit typed values:

* :func:`extract_pulse` (NEUR-01) → :class:`PulseSignal`
* :func:`detect_anomaly` (NEUR-02) → :class:`AnomalyPulse`
* :func:`assess_risk` (NEUR-03) → :class:`RiskPulse`

The package follows the sensory authority discipline: no imports
from any engine, no I/O, no clock reads. The three functions are
deterministic over their inputs, so the engines that consume them
remain replay-safe (INV-15).
"""

from sensory.neuromorphic.contracts import (
    POLARITY_LONG,
    POLARITY_NEUTRAL,
    POLARITY_SHORT,
    AnomalyPulse,
    PulseSignal,
    RiskPulse,
)
from sensory.neuromorphic.dyon_anomaly import detect_anomaly
from sensory.neuromorphic.governance_risk import (
    RISK_HAZARD_DENSITY,
    RISK_REJECT_RATE,
    RISK_UNAUTH_DIRECTIVE_RATE,
    DecisionObservation,
    assess_risk,
)
from sensory.neuromorphic.indira_signal import (
    TradeSample,
    extract_pulse,
)

__all__ = [
    "POLARITY_LONG",
    "POLARITY_NEUTRAL",
    "POLARITY_SHORT",
    "RISK_HAZARD_DENSITY",
    "RISK_REJECT_RATE",
    "RISK_UNAUTH_DIRECTIVE_RATE",
    "AnomalyPulse",
    "DecisionObservation",
    "PulseSignal",
    "RiskPulse",
    "TradeSample",
    "assess_risk",
    "detect_anomaly",
    "extract_pulse",
]
