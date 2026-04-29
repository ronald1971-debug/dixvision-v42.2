"""Hazard → Governance coupling layer (BEHAVIOR-P3).

Pure offline modules that translate observed :class:`HazardEvent`s into
deterministic throttle decisions and apply them to a frozen
:class:`RiskSnapshot`. Together they let *Dyon's* hazard taxonomy
*immediately* impact execution decisions without any clock read,
PRNG, or runtime I/O (INV-64).

Public surface:

* :class:`HazardThrottleConfig` / :class:`HazardSeverityRule` /
  :class:`HazardCodeOverride` — the policy table.
* :class:`HazardObservation` — a window-storable subset of
  ``HazardEvent``.
* :class:`ThrottleDecision` — the deterministic result.
* :func:`compute_throttle` — pure function (window of observations
  + ``now_ns`` + config → decision).
* :class:`HazardObserver` — bounded ring buffer; ``observe`` +
  ``current_throttle``.
* :func:`apply_throttle` — pure mutator from :class:`RiskSnapshot`
  + :class:`ThrottleDecision` → tightened :class:`RiskSnapshot`.

The CRITICAL/HIGH emergency-LOCK path remains owned by Governance
(``EventClassifier`` → ``StateTransitionManager``). This module
*adds* a second, fine-grained throttle path for LOW/MEDIUM hazards
that would previously have only been audited.
"""

from system_engine.coupling.hazard_throttle import (
    HAZARD_THROTTLE_VERSION,
    HazardCodeOverride,
    HazardObservation,
    HazardObserver,
    HazardSeverityRule,
    HazardThrottleConfig,
    ThrottleDecision,
    compute_throttle,
)
from system_engine.coupling.risk_snapshot_throttle import apply_throttle

__all__ = [
    "HAZARD_THROTTLE_VERSION",
    "HazardCodeOverride",
    "HazardObservation",
    "HazardObserver",
    "HazardSeverityRule",
    "HazardThrottleConfig",
    "ThrottleDecision",
    "apply_throttle",
    "compute_throttle",
]
