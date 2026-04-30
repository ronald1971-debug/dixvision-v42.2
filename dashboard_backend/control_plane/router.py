"""Dashboard control-plane router (Phase 6).

Per Build Compiler Spec §6 the dashboard is a *Control Plane*. Every
operator-originated action enters the system through this router, which
is a thin Python seam that:

1. Receives a strongly typed :class:`OperatorRequest` (one of the four
   :class:`OperatorAction` categories).
2. Forwards it verbatim to the GOV-CP-07
   :class:`~governance_engine.control_plane.operator_interface_bridge.OperatorInterfaceBridge`.
3. Returns the resulting :class:`GovernanceDecision` to the caller
   (typically the FastAPI surface in :mod:`ui.server`).

The router never inspects, mutates, or expands the request. It is the
audit-only seam between the UI and Governance — the dashboard's only
write path. INV-12 / INV-37 enforced.

Authority lint:

* B7 — only imports allowed: ``core.contracts``, the GOV-CP-07
  bridge surface from ``governance_engine.control_plane``.
* B1 — no plugin-style cross-engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.governance import (
    GovernanceDecision,
    OperatorRequest,
)
from governance_engine.control_plane.operator_interface_bridge import (
    OperatorInterfaceBridge,
)


@dataclass(frozen=True, slots=True)
class RouteOutcome:
    """Decision plus a one-line audit summary suitable for the UI."""

    decision: GovernanceDecision
    summary: str

    @property
    def approved(self) -> bool:
        return self.decision.approved


class ControlPlaneRouter:
    """Thin seam between UI and GOV-CP-07.

    The router holds a reference to the bridge but performs no logic of
    its own. Two things make it worth existing as a separate object:

    * It localises *the* dashboard write seam in a single auditable
      module — every dashboard write traces through here (B7 lint).
    * It produces a UI-friendly :class:`RouteOutcome` so widget code
      doesn't reach into ``GovernanceDecision`` internals.
    """

    name: str = "dashboard_control_plane_router"
    spec_id: str = "DASH-CP-01"

    def __init__(self, *, bridge: OperatorInterfaceBridge) -> None:
        self._bridge = bridge

    def submit(self, request: OperatorRequest) -> RouteOutcome:
        decision = self._bridge.submit(request)
        summary = self._render_summary(request, decision)
        return RouteOutcome(decision=decision, summary=summary)

    @staticmethod
    def _render_summary(
        request: OperatorRequest, decision: GovernanceDecision
    ) -> str:
        if decision.approved:
            return f"{request.action.value} approved by Governance"
        suffix = decision.rejection_code or decision.summary or "no reason"
        return f"{request.action.value} rejected by Governance ({suffix})"
