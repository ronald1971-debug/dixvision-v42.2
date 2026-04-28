"""Shadow Policy — INV-52 alternative-decision tracker.

Phase 6.T1b. The shadow policy runs alongside the primary
:func:`intelligence_engine.meta_controller.policy.execution_policy.decide_execution_policy`
on the same inputs, computes an alternative
:class:`ExecutionDecision`, and emits a ``META_DIVERGENCE``
:class:`SystemEvent` whenever the two disagree.

**Critical invariant (INV-52 + manifest §B17):**

* The shadow path is *non-acting*. Its output never reaches the
  PolicyEngine, the ExecutionEngine, or anything that mutates state.
* It is allowed to read Belief State and Pressure Vector projections
  (they are read-only) and to construct ``SystemEvent`` records, but
  it is forbidden by authority lint rule **B17** from importing
  ``governance_engine``.

The point of the shadow path is to give the offline learning loop a
counter-factual ledger so it can compare ``primary_decision`` against
``shadow_decision`` per realised outcome window.

Authority constraints:

* Imports only :mod:`core.contracts` and
  :mod:`intelligence_engine.meta_controller.policy.execution_policy`
  (sibling, in-package).
* No ``governance_engine`` import (B17).
* No clock, no PRNG; replay-deterministic per INV-15.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.coherence.belief_state import Regime
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import Side, SystemEvent, SystemEventKind
from intelligence_engine.meta_controller.policy.execution_policy import (
    ExecutionDecision,
    decide_execution_policy,
)

SHADOW_POLICY_VERSION = "v3.3-T1b"


# ---------------------------------------------------------------------------
# Shadow decision
# ---------------------------------------------------------------------------


def compute_shadow_decision(
    *,
    regime: Regime,
    pressure: PressureVector,
    proposed_side: Side,
    proposed_size: float,
    proposed_confidence: float,
    latency_budget_ns: int,
    elapsed_ns: int,
) -> ExecutionDecision:
    """Compute the shadow alternative decision.

    The T1b shadow runs the *same* primary policy logic but with two
    deliberate perturbations that make it useful as a counterfactual:

    * **No latency fallback.** The shadow ignores the latency budget
      so we can audit whether the primary path's INV-48 fallback
      cost a profitable opportunity.
    * **No safety_modifier folding.** The shadow uses the *raw*
      proposed size and confidence; this lets calibration spot when
      pressure damping was overly conservative.

    The shadow is still pure / deterministic and never side-effects.
    Future revisions (e.g. T1f experimentation lane) may replace this
    body with an entirely different policy under governance gating.
    """
    # Disabled latency fallback — call into the primary with a budget
    # that cannot be exceeded.
    return decide_execution_policy(
        regime=regime,
        pressure=_unitary_safety(pressure),
        proposed_side=proposed_side,
        proposed_size=proposed_size,
        proposed_confidence=proposed_confidence,
        latency_budget_ns=_INFINITY_BUDGET,
        elapsed_ns=0,
    )


_INFINITY_BUDGET = 1 << 62  # large enough to never trip in practice


def _unitary_safety(pressure: PressureVector) -> PressureVector:
    """Return ``pressure`` with ``safety_modifier`` forced to 1.0.

    Used by the shadow path so that pressure damping is *not* folded
    into the alternative decision; only Governance hard-override
    semantics survive.
    """
    if pressure.safety_modifier == 1.0:
        return pressure
    # Construct a copy with safety_modifier=1.0; PressureVector is
    # frozen so we instantiate a new one. Only the modifier moves.
    return PressureVector(
        ts_ns=pressure.ts_ns,
        perf=pressure.perf,
        risk=pressure.risk,
        drift=pressure.drift,
        latency=pressure.latency,
        uncertainty=pressure.uncertainty,
        safety_modifier=1.0,
        cross_signal_entropy=pressure.cross_signal_entropy,
        signal_count=pressure.signal_count,
        version=pressure.version,
    )


# ---------------------------------------------------------------------------
# Divergence record
# ---------------------------------------------------------------------------


def divergence_payload(
    *,
    primary: ExecutionDecision,
    shadow: ExecutionDecision,
) -> Mapping[str, str]:
    """Build a stringly-typed payload describing a primary / shadow gap.

    All values are strings because :attr:`SystemEvent.payload` is a
    ``Mapping[str, str]`` (deterministic ledger encoding). The keys
    are stable so calibration consumers can parse without reflection.
    """
    return {
        "primary_side": primary.side.value,
        "primary_size": f"{primary.size_fraction:.6f}",
        "primary_confidence": f"{primary.confidence:.6f}",
        "primary_rationale": primary.rationale,
        "primary_fallback": "true" if primary.fallback else "false",
        "shadow_side": shadow.side.value,
        "shadow_size": f"{shadow.size_fraction:.6f}",
        "shadow_confidence": f"{shadow.confidence:.6f}",
        "shadow_rationale": shadow.rationale,
        "shadow_fallback": "true" if shadow.fallback else "false",
        "side_diverged": "true" if primary.side is not shadow.side else "false",
        "version": SHADOW_POLICY_VERSION,
    }


def emit_divergence_event(
    *,
    ts_ns: int,
    primary: ExecutionDecision,
    shadow: ExecutionDecision,
) -> SystemEvent | None:
    """Build a ``META_DIVERGENCE`` SystemEvent, or ``None`` if equal.

    Returns ``None`` when ``primary == shadow`` (no information for
    the calibrator). The caller is responsible for handing the event
    to the bus / ledger; this function does not write anything.
    """
    if primary == shadow:
        return None
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.META_DIVERGENCE,
        source="intelligence.meta_controller.shadow",
        payload=divergence_payload(primary=primary, shadow=shadow),
    )


__all__ = [
    "SHADOW_POLICY_VERSION",
    "compute_shadow_decision",
    "divergence_payload",
    "emit_divergence_event",
]
