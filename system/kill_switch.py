"""SAFE-01 — system-wide kill switch primitive (P0-1b).

Single named chokepoint for system-wide kill engagement. The kill
action is the most safety-critical edge in the system: it forces the
mode FSM into ``SystemMode.LOCKED`` regardless of the prior mode and
halts every subsequent execute path.

Before this module the same edge was reachable through three named
seams: :class:`OperatorInterfaceBridge._handle_kill` (operator), the
hazard sensors that escalate to ``SystemMode.LOCKED`` via direct
``StateTransitionManager.propose`` calls, and ad-hoc subsystem-level
kills (memecoin control panel). The action of *engaging* the kill
switch was therefore not visible as a single primitive in the call
graph, which made:

  * authority-matrix audit (``registry/authority_matrix.yaml`` calls
    out a ``kill_switch`` actor) impossible to verify, and
  * the next P0-2 hazard-throttle chain unable to route at the
    primitive level — it would have had to import three different
    seams.

This module exposes :class:`KillSwitch` with one canonical method,
:meth:`KillSwitch.engage`. Every ``ModeTransitionRequest`` whose
``target_mode`` is ``LOCKED`` must originate from this primitive
(enforced at runtime by callers and by :data:`KILL_SWITCH_ALLOWED`
allowlist on the lint side once B-KILL is added in a follow-up
PR).

The primitive does not own any new state — it is a thin, typed
façade over :class:`StateTransitionManager`. It guarantees:

  1. Every kill is journaled by the existing
     :class:`LedgerAuthorityWriter` chain (because we go through
     ``StateTransitionManager.propose``).
  2. Every kill carries ``operator_authorized=True`` so the policy
     gate cannot reject it for missing operator authorisation. The
     FSM legality check (``LIVE → LOCKED`` etc.) and the policy
     gate's other reject codes (operator authorisation never gets
     checked for the kill edge because it is always set to True
     here, so the only relevant rejections are FSM illegality and
     promotion-gate breaches) still fire.
  3. The :class:`GovernanceDecision` returned uses
     :data:`DecisionKind.KILL` so downstream consumers can match
     the primitive name in the audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.contracts.governance import (
    DecisionKind,
    GovernanceDecision,
    ModeTransitionRequest,
    SystemMode,
)
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
)
from system.time_source import wall_ns

__all__ = (
    "KillReason",
    "KillRequest",
    "KillSwitch",
)


class KillReason(StrEnum):
    """Origin tag for a kill engagement.

    Recorded on the audit ledger so the post-hoc audit can answer
    "who pulled the cord" without correlating across multiple
    structured fields. Three legitimate sources today:

    * ``OPERATOR``  — human operator clicked KILL on the dashboard.
    * ``HAZARD``    — hazard sensor (HAZ-01..N) escalated into
                       a system-wide lock.
    * ``EXTERNAL``  — out-of-band script / cockpit pairing trigger.
    """

    OPERATOR = "OPERATOR"
    HAZARD = "HAZARD"
    EXTERNAL = "EXTERNAL"


@dataclass(frozen=True, slots=True)
class KillRequest:
    """Typed input to :meth:`KillSwitch.engage`."""

    requestor: str
    reason: str
    origin: KillReason
    ts_ns: int


class KillSwitch:
    """SAFE-01 — kill-switch primitive."""

    name: str = "kill_switch"
    spec_id: str = "SAFE-01"

    def __init__(self, *, state_transitions: StateTransitionManager) -> None:
        self._state = state_transitions

    def engage(self, request: KillRequest) -> GovernanceDecision:
        """Force the mode FSM into ``SystemMode.LOCKED``.

        The kill edge is **operator-authorised by construction**:
        every kill, regardless of origin, sets
        ``operator_authorized=True`` on the underlying
        :class:`ModeTransitionRequest`. This is intentional —
        hazard-driven kills must be able to fire without a human in
        the loop, and the operator-authorised flag exists to bypass
        the policy engine's "operator must explicitly opt in to a
        more permissive mode" gate. Going LIVE → LOCKED is a
        *restrictive* edge, so the gate semantics do not apply.

        Returns a :class:`GovernanceDecision` carrying
        :data:`DecisionKind.KILL` plus the FSM rejection code if the
        transition was illegal (e.g. trying to lock from a mode that
        has no legal edge to LOCKED). The caller is expected to
        forward the decision verbatim to the operator surface so a
        rejection is visible.
        """

        current = self._state.current_mode()
        decision = self._state.propose(
            ModeTransitionRequest(
                ts_ns=request.ts_ns,
                requestor=request.requestor,
                current_mode=current,
                target_mode=SystemMode.LOCKED,
                reason=_compose_reason(request),
                operator_authorized=True,
            )
        )
        return GovernanceDecision(
            ts_ns=decision.ts_ns,
            kind=DecisionKind.KILL,
            approved=decision.approved,
            summary=(
                f"system locked (was {decision.prev_mode.name})"
                if decision.approved
                else f"kill denied ({decision.rejection_code})"
            ),
            rejection_code=decision.rejection_code,
            ledger_seq=decision.ledger_seq,
        )

    # Convenience constructors --------------------------------------

    def engage_operator(
        self, *, requestor: str, reason: str, ts_ns: int | None = None
    ) -> GovernanceDecision:
        """Operator-initiated kill (UI button, REST endpoint)."""

        return self.engage(
            KillRequest(
                requestor=requestor,
                reason=reason or "operator kill",
                origin=KillReason.OPERATOR,
                ts_ns=ts_ns if ts_ns is not None else wall_ns(),
            )
        )

    def engage_hazard(
        self,
        *,
        sensor: str,
        reason: str,
        ts_ns: int | None = None,
    ) -> GovernanceDecision:
        """Hazard-sensor-initiated kill (HAZ-01..N escalations)."""

        return self.engage(
            KillRequest(
                requestor=sensor,
                reason=reason or f"hazard kill ({sensor})",
                origin=KillReason.HAZARD,
                ts_ns=ts_ns if ts_ns is not None else wall_ns(),
            )
        )

    def engage_external(
        self, *, requestor: str, reason: str, ts_ns: int | None = None
    ) -> GovernanceDecision:
        """Out-of-band kill (cockpit pairing, ops script)."""

        return self.engage(
            KillRequest(
                requestor=requestor,
                reason=reason or "external kill",
                origin=KillReason.EXTERNAL,
                ts_ns=ts_ns if ts_ns is not None else wall_ns(),
            )
        )


def _compose_reason(request: KillRequest) -> str:
    """Stable string layout so the ledger is parseable post-hoc."""

    return f"[{request.origin.value}] {request.reason}"
