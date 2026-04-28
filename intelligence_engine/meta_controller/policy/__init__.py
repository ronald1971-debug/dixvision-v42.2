"""Meta-Controller / policy — final gate (Phase 6.T1b).

Two modules:

* :mod:`intelligence_engine.meta_controller.policy.execution_policy`
  — primary decision function. INV-48 latency-budget fallback to
  :data:`FALLBACK_POLICY` (constant-time, no Belief / Pressure deps).
* :mod:`intelligence_engine.meta_controller.policy.shadow_policy`
  — INV-52 alternative decision computed alongside the primary path.
  Never reaches PolicyEngine. Emits ``META_DIVERGENCE`` SystemEvent
  for the offline learning loop only.

Authority lint:

* B1  — no cross-engine direct imports.
* B17 — ``policy/shadow_policy`` may not import
        ``governance_engine``. The shadow path is decision-only;
        side-effects are out of bounds.
"""

from intelligence_engine.meta_controller.policy.execution_policy import (
    EXECUTION_POLICY_VERSION,
    FALLBACK_POLICY,
    ExecutionDecision,
    decide_execution_policy,
)
from intelligence_engine.meta_controller.policy.shadow_policy import (
    SHADOW_POLICY_VERSION,
    compute_shadow_decision,
    divergence_payload,
    emit_divergence_event,
)

__all__ = [
    "EXECUTION_POLICY_VERSION",
    "FALLBACK_POLICY",
    "ExecutionDecision",
    "decide_execution_policy",
    "SHADOW_POLICY_VERSION",
    "compute_shadow_decision",
    "divergence_payload",
    "emit_divergence_event",
]
