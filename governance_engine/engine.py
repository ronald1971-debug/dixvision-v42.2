"""GovernanceEngine — RUNTIME-ENGINE-04.

Phase 1 wires the seven Governance Control Plane modules
(GOV-CP-01..07) behind the engine's :class:`RuntimeEngine` shape.
``process(event)`` is the hot path the runtime bus already calls;
``check_self()`` reports the health of every CP module.

Operator-originated requests do **not** flow through ``process``;
they enter via the ``OperatorInterfaceBridge`` (GOV-CP-07), which is
the dashboard's only authorised write path. ``process`` handles
HAZARD events (HIGH/CRITICAL → emergency LOCK) and SYSTEM events
(UPDATE_PROPOSED / PLUGIN_LIFECYCLE audit rows).

INV-08 / INV-11 / INV-15 still hold — the engine emits zero events
on the runtime bus by default; everything Governance writes lands in
the authority ledger via ``LedgerAuthorityWriter``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    Plugin,
    RuntimeEngine,
)
from core.contracts.events import (
    Event,
    EventKind,
    HazardEvent,
    HazardSeverity,
    SystemEvent,
    SystemEventKind,
)
from core.contracts.governance import (
    Constraint,
    ModeTransitionRequest,
    SystemMode,
)
from governance_engine.control_plane import (
    ComplianceValidator,
    EventClassifier,
    LedgerAuthorityWriter,
    OperatorInterfaceBridge,
    PolicyEngine,
    RiskEvaluator,
    StateTransitionManager,
)
from governance_engine.control_plane.event_classifier import PipelineStage


class GovernanceEngine(RuntimeEngine):
    name: str = "governance"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        *,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        constraints: Sequence[Constraint] = (),
        initial_mode: SystemMode = SystemMode.SAFE,
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )

        self.ledger = LedgerAuthorityWriter()
        self.policy = PolicyEngine(constraints=constraints)
        self.risk = RiskEvaluator(constraints=tuple(constraints))
        self.compliance = ComplianceValidator()
        self.classifier = EventClassifier()
        self.state_transitions = StateTransitionManager(
            policy=self.policy,
            ledger=self.ledger,
            initial_mode=initial_mode,
        )
        self.operator = OperatorInterfaceBridge(
            policy=self.policy,
            state_transitions=self.state_transitions,
            ledger=self.ledger,
        )

    # ------------------------------------------------------------------
    # Engine surface
    # ------------------------------------------------------------------

    def process(self, event: Event) -> Sequence[Event]:
        route = self.classifier.classify(event)
        if PipelineStage.NOOP in route.stages:
            return ()

        if event.kind is EventKind.HAZARD:
            self._handle_hazard(event, route.emergency_lock)  # type: ignore[arg-type]
            return ()

        if event.kind is EventKind.SYSTEM:
            self._handle_system(event)  # type: ignore[arg-type]
            return ()

        # SIGNAL / EXECUTION events are audited but never produce a
        # downstream bus event from Governance — gate decisions are
        # made by Execution against the FastRiskCache; Governance is
        # the slow-path owner of the limits, not the per-tick gate.
        # The classifier routes both event kinds through LEDGER, so
        # an audit row preserves replay determinism (INV-15).
        if event.kind is EventKind.EXECUTION:
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="EXECUTION_AUDIT",
                payload={"event": "EXECUTION"},
            )
        elif event.kind is EventKind.SIGNAL:
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="SIGNAL_AUDIT",
                payload={"event": "SIGNAL"},
            )
        return ()

    def check_self(self) -> HealthStatus:
        cp_states = {
            self.policy.spec_id: HealthState.OK,
            self.risk.spec_id: HealthState.OK,
            self.state_transitions.spec_id: (
                HealthState.OK
                if self.state_transitions.current_mode() is not SystemMode.LOCKED
                else HealthState.DEGRADED
            ),
            self.classifier.spec_id: HealthState.OK,
            self.ledger.spec_id: (
                HealthState.OK if self.ledger.verify() else HealthState.FAIL
            ),
            self.compliance.spec_id: HealthState.OK,
            self.operator.spec_id: HealthState.OK,
        }
        plugin_states: dict[str, dict[str, HealthState]] = {
            "control_plane": cp_states
        }

        worst = HealthState.OK
        for s in cp_states.values():
            if s is HealthState.FAIL:
                worst = HealthState.FAIL
                break
            if s is HealthState.DEGRADED and worst is HealthState.OK:
                worst = HealthState.DEGRADED

        mode = self.state_transitions.current_mode().name
        return HealthStatus(
            state=worst,
            detail=(
                f"Phase 1 — control_plane wired (GOV-CP-01..07); "
                f"mode={mode}; ledger_rows={len(self.ledger)}"
            ),
            plugin_states=plugin_states,
        )

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_hazard(
        self, event: HazardEvent, emergency_lock: bool
    ) -> None:
        if emergency_lock:
            self.state_transitions.propose(
                ModeTransitionRequest(
                    ts_ns=event.ts_ns,
                    requestor=f"hazard:{event.source}",
                    current_mode=self.state_transitions.current_mode(),
                    target_mode=SystemMode.LOCKED,
                    reason=f"{event.code} severity={event.severity.name}",
                    operator_authorized=True,
                )
            )
            return

        self.ledger.append(
            ts_ns=event.ts_ns,
            kind="HAZARD_AUDIT",
            payload={
                "code": event.code,
                "severity": event.severity.value,
                "source": event.source,
                "detail": event.detail,
            },
        )

    def _handle_system(self, event: SystemEvent) -> None:
        if event.sub_kind is SystemEventKind.UPDATE_PROPOSED:
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="UPDATE_PROPOSED_AUDIT",
                payload={
                    "source": event.source,
                    **{f"p_{k}": v for k, v in event.payload.items()},
                },
            )
            return
        if event.sub_kind is SystemEventKind.PLUGIN_LIFECYCLE:
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="PLUGIN_LIFECYCLE_AUDIT",
                payload={
                    "source": event.source,
                    **{f"p_{k}": v for k, v in event.payload.items()},
                },
            )
            return

        # HEARTBEAT, HEALTH_REPORT, LEDGER_COMMIT — handled elsewhere.
        # Log nothing; classifier marks these NOOP for governance.
        del event

    # ------------------------------------------------------------------
    # Convenience accessors used by the UI / tests
    # ------------------------------------------------------------------

    def current_mode(self) -> SystemMode:
        return self.state_transitions.current_mode()


_ = HazardSeverity  # silence unused-import noise; severity referenced in tests
