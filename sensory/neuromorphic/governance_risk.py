"""NEUR-03 — Governance-risk perception.

Pure function over a window of recent decision-trace observations.
Emits a :class:`RiskPulse` describing operator-side risk drift —
e.g. rising rejection rates, hazard density, or
unauthorized-directive counts.

Authority:
    No engine imports, no I/O, no clock, no FSM mutation. The
    sensor produces a typed value; Governance decides what to do
    with it.

Algorithm (v1):

* Inputs are :class:`DecisionObservation` rows (one per recent
  decision). The caller supplies the window — the sensor does not
  query the audit ledger directly (that would couple the sensory
  perimeter to the engine boundary).
* For each known ``risk_kind``, compute a fraction in ``[0.0, 1.0]``:

  * ``"REJECT_RATE"`` — fraction of rows whose ``approved`` is
    False.
  * ``"HAZARD_DENSITY"`` — fraction of rows whose ``had_hazard``
    is True.
  * ``"UNAUTH_DIRECTIVE_RATE"`` — fraction of rows whose
    ``unauthorized_directive`` is True.

The fraction *is* the risk_score: a 0.0..1.0 ratio with no
saturation magic, so the operator can read the dashboard value as
"X% of recent decisions had this risk property". This keeps the
sensor honest — Governance applies thresholds, not the sensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sensory.neuromorphic.contracts import RiskPulse

RISK_REJECT_RATE: str = "REJECT_RATE"
RISK_HAZARD_DENSITY: str = "HAZARD_DENSITY"
RISK_UNAUTH_DIRECTIVE_RATE: str = "UNAUTH_DIRECTIVE_RATE"


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    """One row in the governance-risk window.

    The caller projects audit-ledger rows into this lightweight
    boolean shape so :func:`assess_risk` does not need to know the
    full DecisionTrace schema. This keeps the sensor's surface
    minimal and replay-pure.

    Attributes:
        approved: Whether the decision was approved by Governance.
        had_hazard: Whether a hazard was active at decision time.
        unauthorized_directive: Whether the originating directive
            failed authority validation.
    """

    approved: bool
    had_hazard: bool
    unauthorized_directive: bool


def assess_risk(
    *,
    ts_ns: int,
    source: str,
    risk_kind: str,
    window: Sequence[DecisionObservation],
    evidence: Mapping[str, str] | None = None,
) -> RiskPulse:
    """Project a window of decisions onto a single risk-fraction.

    Args:
        ts_ns: Caller-supplied window-close timestamp.
        source: Stable source identifier for the perception.
        risk_kind: One of :data:`RISK_REJECT_RATE`,
            :data:`RISK_HAZARD_DENSITY`,
            :data:`RISK_UNAUTH_DIRECTIVE_RATE`. A custom string is
            allowed but the sensor only knows how to compute the
            three named kinds; an unknown kind raises.
        window: Sequence of :class:`DecisionObservation`. Must be
            non-empty.
        evidence: Optional structural metadata.

    Returns:
        :class:`RiskPulse` whose ``risk_score`` is the fraction of
        the window for which the row's matching predicate is True.

    Raises:
        ValueError: If ``window`` is empty or ``risk_kind`` is not
            one of the supported labels.
    """
    n = len(window)
    if n < 1:
        raise ValueError(
            "assess_risk.window must contain at least 1 observation"
        )

    if risk_kind == RISK_REJECT_RATE:
        hits = sum(1 for o in window if not o.approved)
    elif risk_kind == RISK_HAZARD_DENSITY:
        hits = sum(1 for o in window if o.had_hazard)
    elif risk_kind == RISK_UNAUTH_DIRECTIVE_RATE:
        hits = sum(1 for o in window if o.unauthorized_directive)
    else:
        raise ValueError(
            "assess_risk.risk_kind must be one of "
            f"{[RISK_REJECT_RATE, RISK_HAZARD_DENSITY, RISK_UNAUTH_DIRECTIVE_RATE]}"
        )

    risk_score = hits / n
    return RiskPulse(
        ts_ns=ts_ns,
        source=source,
        risk_kind=risk_kind,
        risk_score=risk_score,
        sample_count=n,
        evidence=dict(evidence) if evidence else {},
    )


__all__ = [
    "RISK_HAZARD_DENSITY",
    "RISK_REJECT_RATE",
    "RISK_UNAUTH_DIRECTIVE_RATE",
    "DecisionObservation",
    "assess_risk",
]
