"""GOV-CP-07 — Operator Interface Bridge.

Single seam between the Dashboard Control Plane and the Governance
authority. The dashboard never writes state; it constructs an
:class:`OperatorRequest` and hands it to ``submit``. The bridge:

1. Asks :class:`PolicyEngine` whether the action is permitted under
   the current mode.
2. Routes to the appropriate authority owner:
   * REQUEST_MODE   → :class:`StateTransitionManager`
   * REQUEST_KILL   → emergency transition to LOCKED
   * REQUEST_UNLOCK → LOCKED → SAFE (operator-authorised only)
   * REQUEST_PLUGIN_LIFECYCLE → ledger-only audit row (Phase 1; full
     dispatch lands in Phase 5 with the Learning ↔ Evolution closed
     loop).
3. Returns a :class:`GovernanceDecision` to the operator.

Per Build Compiler Spec §6: this is the dashboard's only write path
into the system.
"""

from __future__ import annotations

from core.coherence.system_intent import (
    INTENT_KEY_FOCUS,
    INTENT_KEY_HORIZON,
    INTENT_KEY_OBJECTIVE,
    INTENT_KEY_REASON,
    INTENT_KEY_RISK_MODE,
    decode_focus,
)
from core.contracts.governance import (
    DecisionKind,
    GovernanceDecision,
    IntentHorizon,
    IntentObjective,
    IntentRiskMode,
    IntentTransitionRequest,
    ModeTransitionRequest,
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_engine import PolicyEngine
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
)


def _parse_mode(name: str) -> SystemMode | None:
    try:
        return SystemMode[name]
    except KeyError:
        return None


def _parse_intent_objective(raw: str) -> IntentObjective | None:
    try:
        return IntentObjective(raw)
    except ValueError:
        return None


def _parse_intent_risk_mode(raw: str) -> IntentRiskMode | None:
    try:
        return IntentRiskMode(raw)
    except ValueError:
        return None


def _parse_intent_horizon(raw: str) -> IntentHorizon | None:
    try:
        return IntentHorizon(raw)
    except ValueError:
        return None


class OperatorInterfaceBridge:
    name: str = "operator_interface_bridge"
    spec_id: str = "GOV-CP-07"

    def __init__(
        self,
        *,
        policy: PolicyEngine,
        state_transitions: StateTransitionManager,
        ledger: LedgerAuthorityWriter,
    ) -> None:
        self._policy = policy
        self._state = state_transitions
        self._ledger = ledger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: OperatorRequest) -> GovernanceDecision:
        current = self._state.current_mode()
        permit_ok, permit_code = self._policy.permit_operator_action(
            request, current
        )
        if not permit_ok:
            entry = self._ledger.append(
                ts_ns=request.ts_ns,
                kind="OPERATOR_REJECTED",
                payload={
                    "requestor": request.requestor,
                    "action": request.action,
                    "rejection_code": permit_code,
                    "current_mode": current.name,
                },
            )
            return GovernanceDecision(
                ts_ns=request.ts_ns,
                kind=DecisionKind.REJECTED,
                approved=False,
                summary=f"{request.action} rejected by policy",
                rejection_code=permit_code,
                ledger_seq=entry.seq,
            )

        if request.action is OperatorAction.REQUEST_MODE:
            return self._handle_mode(request, current)
        if request.action is OperatorAction.REQUEST_KILL:
            return self._handle_kill(request, current)
        if request.action is OperatorAction.REQUEST_UNLOCK:
            return self._handle_unlock(request, current)
        if request.action is OperatorAction.REQUEST_PLUGIN_LIFECYCLE:
            return self._handle_plugin_lifecycle(request)
        if request.action is OperatorAction.REQUEST_INTENT:
            return self._handle_intent(request)

        # Unknown action — should be unreachable because PolicyEngine
        # would have rejected it; defensive fall-through preserves
        # determinism rather than raising.
        entry = self._ledger.append(
            ts_ns=request.ts_ns,
            kind="OPERATOR_REJECTED",
            payload={
                "requestor": request.requestor,
                "action": str(request.action),
                "rejection_code": "BRIDGE_UNROUTED",
                "current_mode": current.name,
            },
        )
        return GovernanceDecision(
            ts_ns=request.ts_ns,
            kind=DecisionKind.REJECTED,
            approved=False,
            summary=f"{request.action} could not be routed",
            rejection_code="BRIDGE_UNROUTED",
            ledger_seq=entry.seq,
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_mode(
        self, request: OperatorRequest, current: SystemMode
    ) -> GovernanceDecision:
        target_name = request.payload.get("target_mode", "")
        target = _parse_mode(target_name)
        if target is None:
            entry = self._ledger.append(
                ts_ns=request.ts_ns,
                kind="OPERATOR_REJECTED",
                payload={
                    "requestor": request.requestor,
                    "action": request.action,
                    "rejection_code": "BRIDGE_UNKNOWN_MODE",
                    "raw_target": target_name,
                },
            )
            return GovernanceDecision(
                ts_ns=request.ts_ns,
                kind=DecisionKind.REJECTED,
                approved=False,
                summary=f"unknown target mode '{target_name}'",
                rejection_code="BRIDGE_UNKNOWN_MODE",
                ledger_seq=entry.seq,
            )

        operator_authorized = (
            request.payload.get("operator_authorized", "false").lower() == "true"
        )
        decision = self._state.propose(
            ModeTransitionRequest(
                ts_ns=request.ts_ns,
                requestor=request.requestor,
                current_mode=current,
                target_mode=target,
                reason=request.payload.get("reason", ""),
                operator_authorized=operator_authorized,
            )
        )
        return GovernanceDecision(
            ts_ns=decision.ts_ns,
            kind=DecisionKind.MODE_TRANSITION,
            approved=decision.approved,
            summary=(
                f"{decision.prev_mode.name} -> {decision.new_mode.name}"
                if decision.approved
                else f"mode transition denied ({decision.rejection_code})"
            ),
            rejection_code=decision.rejection_code,
            ledger_seq=decision.ledger_seq,
        )

    def _handle_kill(
        self, request: OperatorRequest, current: SystemMode
    ) -> GovernanceDecision:
        decision = self._state.propose(
            ModeTransitionRequest(
                ts_ns=request.ts_ns,
                requestor=request.requestor,
                current_mode=current,
                target_mode=SystemMode.LOCKED,
                reason=request.payload.get("reason", "operator kill"),
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

    def _handle_unlock(
        self, request: OperatorRequest, current: SystemMode
    ) -> GovernanceDecision:
        decision = self._state.propose(
            ModeTransitionRequest(
                ts_ns=request.ts_ns,
                requestor=request.requestor,
                current_mode=current,
                target_mode=SystemMode.SAFE,
                reason=request.payload.get("reason", "operator unlock"),
                operator_authorized=True,
            )
        )
        return GovernanceDecision(
            ts_ns=decision.ts_ns,
            kind=DecisionKind.MODE_TRANSITION,
            approved=decision.approved,
            summary=(
                "unlocked to SAFE"
                if decision.approved
                else f"unlock denied ({decision.rejection_code})"
            ),
            rejection_code=decision.rejection_code,
            ledger_seq=decision.ledger_seq,
        )

    def _handle_plugin_lifecycle(
        self, request: OperatorRequest
    ) -> GovernanceDecision:
        plugin_path = request.payload.get("plugin_path", "")
        target_status = request.payload.get("target_status", "")
        if not plugin_path or not target_status:
            entry = self._ledger.append(
                ts_ns=request.ts_ns,
                kind="OPERATOR_REJECTED",
                payload={
                    "requestor": request.requestor,
                    "action": request.action,
                    "rejection_code": "BRIDGE_INCOMPLETE_PAYLOAD",
                    "plugin_path": plugin_path,
                    "target_status": target_status,
                },
            )
            return GovernanceDecision(
                ts_ns=request.ts_ns,
                kind=DecisionKind.REJECTED,
                approved=False,
                summary="plugin lifecycle request incomplete",
                rejection_code="BRIDGE_INCOMPLETE_PAYLOAD",
                ledger_seq=entry.seq,
            )

        entry = self._ledger.append(
            ts_ns=request.ts_ns,
            kind="PLUGIN_LIFECYCLE",
            payload={
                "requestor": request.requestor,
                "plugin_path": plugin_path,
                "target_status": target_status,
                "reason": request.payload.get("reason", ""),
            },
        )
        return GovernanceDecision(
            ts_ns=request.ts_ns,
            kind=DecisionKind.PLUGIN_LIFECYCLE,
            approved=True,
            summary=f"{plugin_path} -> {target_status}",
            ledger_seq=entry.seq,
        )


    def _handle_intent(self, request: OperatorRequest) -> GovernanceDecision:
        """Route a ``REQUEST_INTENT`` action to ``propose_intent``.

        The dashboard packs the desired ``objective`` / ``risk_mode`` /
        ``horizon`` (StrEnum values) and an optional ``focus`` (encoded
        with ``encode_focus``) into the ``OperatorRequest.payload``
        ``Mapping[str, str]``. Unknown values short-circuit to a
        ``BRIDGE_UNKNOWN_INTENT`` rejection ledger row before
        ``propose_intent`` runs, so malformed dashboard input never
        reaches the GOV-CP-03 writer.
        """

        objective_raw = request.payload.get(INTENT_KEY_OBJECTIVE, "")
        risk_mode_raw = request.payload.get(INTENT_KEY_RISK_MODE, "")
        horizon_raw = request.payload.get(INTENT_KEY_HORIZON, "")

        objective = _parse_intent_objective(objective_raw)
        risk_mode = _parse_intent_risk_mode(risk_mode_raw)
        horizon = _parse_intent_horizon(horizon_raw)

        if objective is None or risk_mode is None or horizon is None:
            entry = self._ledger.append(
                ts_ns=request.ts_ns,
                kind="OPERATOR_REJECTED",
                payload={
                    "requestor": request.requestor,
                    "action": request.action,
                    "rejection_code": "BRIDGE_UNKNOWN_INTENT",
                    INTENT_KEY_OBJECTIVE: objective_raw,
                    INTENT_KEY_RISK_MODE: risk_mode_raw,
                    INTENT_KEY_HORIZON: horizon_raw,
                },
            )
            return GovernanceDecision(
                ts_ns=request.ts_ns,
                kind=DecisionKind.REJECTED,
                approved=False,
                summary="intent transition rejected (unknown enum value)",
                rejection_code="BRIDGE_UNKNOWN_INTENT",
                ledger_seq=entry.seq,
            )

        focus = decode_focus(request.payload.get(INTENT_KEY_FOCUS, ""))

        decision = self._state.propose_intent(
            IntentTransitionRequest(
                ts_ns=request.ts_ns,
                requestor=request.requestor,
                objective=objective,
                risk_mode=risk_mode,
                horizon=horizon,
                focus=focus,
                reason=request.payload.get(INTENT_KEY_REASON, ""),
            )
        )

        return GovernanceDecision(
            ts_ns=decision.ts_ns,
            kind=DecisionKind.INTENT_TRANSITION,
            approved=decision.approved,
            summary=(
                f"intent set: objective={decision.objective.value} "
                f"risk_mode={decision.risk_mode.value} "
                f"horizon={decision.horizon.value}"
                if decision.approved
                else f"intent rejected ({decision.rejection_code})"
            ),
            rejection_code=decision.rejection_code,
            ledger_seq=decision.ledger_seq,
        )


__all__ = ["OperatorInterfaceBridge"]
