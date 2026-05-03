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
from governance_engine.control_plane.drift_oracle import DriftCompositeOracle
from governance_engine.control_plane.event_classifier import PipelineStage
from governance_engine.control_plane.policy_engine import install_policy_table
from governance_engine.control_plane.update_applier import UpdateApplier
from governance_engine.control_plane.update_validator import (
    ProposedUpdate,
    UpdateValidator,
    UpdateVerdict,
)
from governance_engine.strategy_registry import StrategyRegistry


class GovernanceEngine(RuntimeEngine):
    name: str = "governance"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        *,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        constraints: Sequence[Constraint] = (),
        initial_mode: SystemMode = SystemMode.SAFE,
        policy_table_installed_at_ns: int = 0,
        strategy_registry: StrategyRegistry | None = None,
        ledger: LedgerAuthorityWriter | None = None,
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )

        # Wave-04.6 PR-E — share one canonical ledger with the
        # strategy registry when provided so the audit chain stays
        # coherent (UPDATE_RATIFIED and STRATEGY_PARAMETER_UPDATE
        # rows on the same chain). Older callers without a registry
        # get an isolated ledger as before.
        #
        # Sprint-1 / Class-B "Trust the Ledger" — a caller (typically
        # ``ui.server.STATE``) may inject a ``LedgerAuthorityWriter``
        # constructed with ``db_path=...`` so every governance row is
        # persisted to SQLite. The injected ledger is shared with the
        # strategy registry when both are provided; passing two
        # different ledgers is a wiring mistake and is rejected
        # eagerly.
        if strategy_registry is not None:
            registry_ledger = strategy_registry._ledger  # type: ignore[attr-defined]
            if ledger is not None and ledger is not registry_ledger:
                raise ValueError(
                    "GovernanceEngine: ``ledger`` and "
                    "``strategy_registry._ledger`` must be the same "
                    "instance so the audit chain stays coherent"
                )
            self.ledger = registry_ledger
        else:
            self.ledger = ledger if ledger is not None else LedgerAuthorityWriter()
        self.policy = PolicyEngine(constraints=constraints)
        # GOV-CP-01-PERF — record the precompiled decision-table hash as
        # the very first ledger row. Replay can re-verify it via
        # ``verify_policy_table_hash`` (SAFE-47).
        install_policy_table(
            self.policy,
            self.ledger,
            ts_ns=policy_table_installed_at_ns,
        )
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
        # GOV-CP-08 / P0-7 -- drift composite oracle. Computes the
        # max-component composite drift on demand and proposes a
        # one-step backward transition through the FSM (AUTO -> LIVE
        # -> CANARY -> SHADOW) when any component breaches its
        # threshold. Routes every proposal through
        # ``state_transitions`` so the FSM legality check, policy
        # gate, promotion-gate hash anchor, and authority-ledger row
        # all remain in the loop (B32 single-mutator invariant).
        self.drift_oracle = DriftCompositeOracle()
        # Wave-04.6 PR-E — closed learning loop. When a registry is
        # injected, ``UPDATE_PROPOSED`` events are validated against
        # the strategy's mutable-parameter whitelist and either
        # ratified (and applied) or rejected. With no registry the
        # engine retains the legacy audit-only behaviour so existing
        # callers that don't yet wire the learning loop are
        # unaffected.
        self.strategy_registry = strategy_registry
        if strategy_registry is not None:
            self.update_validator: UpdateValidator | None = UpdateValidator(
                registry=strategy_registry
            )
            self.update_applier: UpdateApplier | None = UpdateApplier(
                registry=strategy_registry
            )
        else:
            self.update_validator = None
            self.update_applier = None

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

    def _handle_update_proposed(self, event: SystemEvent) -> None:
        """Validate and either ratify or reject a learning update.

        Falls back to the legacy ``UPDATE_PROPOSED_AUDIT`` row when
        no :class:`StrategyRegistry` is wired (callers that haven't
        yet adopted Wave-04.6 PR-E).
        """

        if (
            self.update_validator is None
            or self.update_applier is None
            or self.strategy_registry is None
        ):
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="UPDATE_PROPOSED_AUDIT",
                payload={
                    "source": event.source,
                    **{f"p_{k}": v for k, v in event.payload.items()},
                },
            )
            return

        try:
            update = ProposedUpdate(
                ts_ns=event.ts_ns,
                strategy_id=event.payload["strategy_id"],
                parameter=event.payload["parameter"],
                old_value=event.payload["old_value"],
                new_value=event.payload["new_value"],
                reason=event.payload["reason"],
                meta=dict(event.meta),
            )
        except KeyError as exc:
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="UPDATE_REJECTED",
                payload={
                    "source": event.source,
                    "code": "MALFORMED_PAYLOAD",
                    "detail": f"missing field: {exc.args[0]}",
                },
            )
            return

        decision = self.update_validator.validate(
            update=update,
            mode=self.state_transitions.current_mode(),
        )
        if decision.verdict is UpdateVerdict.REJECT:
            self.ledger.append(
                ts_ns=event.ts_ns,
                kind="UPDATE_REJECTED",
                payload={
                    "source": event.source,
                    "strategy_id": update.strategy_id,
                    "parameter": update.parameter,
                    "code": decision.code.value if decision.code else "",
                    "detail": decision.detail,
                },
            )
            return

        # RATIFY — append a ledger row first, then apply. The
        # apply step itself appends a ``STRATEGY_PARAMETER_UPDATE``
        # row, so the chain reads:
        #   UPDATE_RATIFIED → STRATEGY_PARAMETER_UPDATE
        self.ledger.append(
            ts_ns=event.ts_ns,
            kind="UPDATE_RATIFIED",
            payload={
                "source": event.source,
                "strategy_id": update.strategy_id,
                "parameter": update.parameter,
                "old_value": update.old_value,
                "new_value": update.new_value,
                "detail": decision.detail,
            },
        )
        self.update_applier.apply(decision=decision, update=update)

    def _handle_system(self, event: SystemEvent) -> None:
        if event.sub_kind is SystemEventKind.UPDATE_PROPOSED:
            self._handle_update_proposed(event)
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
