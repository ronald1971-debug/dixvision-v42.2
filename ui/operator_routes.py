"""C-2 / P2-4 / R-1 part 4 — operator management routes.

Extracted from :mod:`ui.server` so the FastAPI host module is no
longer the sole home for the operator cockpit's read/write surface.
The thirteen endpoints mounted here back the dashboard's operator
panel and the dash_meme wallet column:

* ``GET  /api/operator/summary``
* ``POST /api/operator/action/kill``
* ``POST /api/operator/action/unlock``
* ``POST /api/operator/action/mode``
* ``POST /api/operator/audit``
* ``GET  /api/feeds/memecoin/summary``
* ``GET  /api/wallet/info``
* ``GET  /api/operator/source-trust``
* ``POST /api/operator/source-trust/promote``
* ``POST /api/operator/source-trust/demote``
* ``GET  /api/operator/learning-override``
* ``POST /api/operator/learning-override``
* ``GET  /api/operator/development-mode``
* ``POST /api/operator/development-mode``
* ``GET  /api/operator/trading-allowed``
* ``POST /api/operator/trading-allowed``

URL paths, HTTP methods, request bodies, response models and HTTP
status codes are preserved byte-for-byte from the inline handlers
that lived in ``ui/server.py``. The route module never imports
``ui.server`` or any ``*_engine`` package directly — it reads its
dependencies through a Protocol-based state accessor, the same
pattern used by :mod:`ui.dashboard_routes`,
:mod:`ui.execution_routes`, :mod:`ui.governance_routes`,
:mod:`ui.runtime_routes`, :mod:`ui.feeds_routes`, and
:mod:`ui.cognitive_routes`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from threading import Lock
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException

from core.contracts.api.operator import (
    DevelopmentModeRequest,
    DevelopmentModeResponse,
    LearningOverrideRequest,
    LearningOverrideResponse,
    OperatorActionResponse,
    OperatorAuditRequest,
    OperatorAuditResponse,
    OperatorEngineRow,
    OperatorKillRequest,
    OperatorMemecoinSnapshot,
    OperatorModeRequest,
    OperatorModeSnapshot,
    OperatorStrategyCounts,
    OperatorSummaryResponse,
    OperatorUnlockRequest,
    TradingAllowedRequest,
    WalletInfoResponse,
)
from core.contracts.api.source_trust import (
    SourceTrustDemotionRequest,
    SourceTrustListResponse,
    SourceTrustPromotionRequest,
    SourceTrustPromotionResponse,
    SourceTrustRow,
)
from core.contracts.development_mode import (
    POLICY_VERSION as DEVELOPMENT_MODE_POLICY_VERSION,
)
from core.contracts.development_mode import DevelopmentModePolicy
from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)
from core.contracts.signal_trust import SignalTrust, default_cap_for
from core.contracts.source_trust_promotions import (
    DEMOTION_LEDGER_KIND,
    PROMOTION_LEDGER_KIND,
    is_promotable_target,
)
from system.time_source import wall_ns

_STRATEGY_STATE_KEYS = (
    ("PROPOSED", "proposed"),
    ("CANARY", "canary"),
    ("LIVE", "live"),
    ("RETIRED", "retired"),
    ("FAILED", "failed"),
)
# Mirrors ``StrategyState`` one-for-one. Strategy-level SHADOW was
# demolished by SHADOW-DEMOLITION-02 (PR #216); ``PAPER`` at the
# system-mode layer supplies the equivalent observe-only behaviour.


def _decision_to_dict(decision: Any) -> dict[str, Any]:
    """Project a typed Governance decision onto a JSON-safe dict.

    Mirrors the helper that lived in :mod:`ui.server`. The dashboard
    route never reaches into the decision shape directly — it gets
    handed a plain ``dict[str, Any]`` so the response stays stable
    even if Governance reshuffles its internal dataclasses.
    """

    if decision is None:
        return {}
    if is_dataclass(decision):
        return _decision_to_dict(asdict(decision))  # type: ignore[no-any-return]
    if isinstance(decision, dict):
        return {str(k): _decision_to_dict(v) for k, v in decision.items()}
    if isinstance(decision, list | tuple):
        return [_decision_to_dict(v) for v in decision]  # type: ignore[return-value]
    if hasattr(decision, "value"):
        return decision.value
    return decision


class _OperatorStateLike(Protocol):
    """Read-only accessor the host installs into FastAPI app.

    Only the attributes the operator routes actually touch are
    declared here so the route module stays decoupled from the
    full harness ``_State`` type.
    """

    @property
    def lock(self) -> Lock: ...
    @property
    def mode_widget(self) -> Any: ...
    @property
    def engines_widget(self) -> Any: ...
    @property
    def strategies_widget(self) -> Any: ...
    @property
    def memecoin_widget(self) -> Any: ...
    @property
    def decisions_widget(self) -> Any: ...
    @property
    def dashboard_router(self) -> Any: ...
    @property
    def governance(self) -> Any: ...
    @property
    def execution(self) -> Any: ...
    @property
    def signal_trust_registry(self) -> Any: ...
    @property
    def signal_trust_promotions(self) -> Any: ...

    # Mutable scalars the operator routes flip under ``lock``.
    learning_override_enabled: bool
    development_mode_enabled: bool
    trading_allowed: bool
    development_mode_policy: DevelopmentModePolicy


def build_operator_router(
    state_accessor: Callable[[], _OperatorStateLike],
) -> APIRouter:
    """Construct the operator-management router.

    Args:
        state_accessor: Callable returning the live state object.
            The route module never holds a direct reference to the
            object; it re-reads through the accessor on every
            request so the same factory works in tests with a
            stub state and in the production harness.

    Returns:
        An :class:`APIRouter` mounting the operator endpoints at
        their canonical absolute paths.
    """

    router = APIRouter(tags=["operator"])

    # -----------------------------------------------------------------
    # /api/operator/summary
    # -----------------------------------------------------------------
    @router.get("/api/operator/summary", response_model=OperatorSummaryResponse)
    def operator_summary() -> OperatorSummaryResponse:
        """Typed read projection of mode + engines + strategies + memecoin."""

        state = state_accessor()
        with state.lock:
            mode_snap = state.mode_widget.snapshot()
            engine_rows = state.engines_widget.snapshot()
            strategies_by_state = state.strategies_widget.by_state()
            memecoin_snap = state.memecoin_widget.status()
            chain_count = len(state.decisions_widget.chains(limit=200))

        counts: dict[str, int] = {field: 0 for _, field in _STRATEGY_STATE_KEYS}
        for state_key, field in _STRATEGY_STATE_KEYS:
            counts[field] = len(strategies_by_state.get(state_key, ()))

        return OperatorSummaryResponse(
            mode=OperatorModeSnapshot(
                current_mode=mode_snap.current_mode,
                legal_targets=list(mode_snap.legal_targets),
                is_locked=mode_snap.is_locked,
            ),
            engines=[
                OperatorEngineRow(
                    engine_name=row.engine_name,
                    bucket=row.bucket,
                    detail=row.detail,
                    plugin_count=len(row.plugin_states),
                )
                for row in engine_rows
            ],
            strategies=OperatorStrategyCounts(**counts),
            memecoin=OperatorMemecoinSnapshot(
                enabled=memecoin_snap.enabled,
                killed=memecoin_snap.killed,
                summary=memecoin_snap.summary,
            ),
            decision_chain_count=chain_count,
        )

    # -----------------------------------------------------------------
    # /api/operator/action/{kill,unlock,mode}
    # -----------------------------------------------------------------
    @router.post(
        "/api/operator/action/kill",
        response_model=OperatorActionResponse,
    )
    def operator_action_kill(body: OperatorKillRequest) -> OperatorActionResponse:
        """Submit an operator KILL request through the governance bridge."""

        state = state_accessor()
        request = OperatorRequest(
            ts_ns=wall_ns(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_KILL,
            payload={"reason": body.reason},
        )
        with state.lock:
            outcome = state.dashboard_router.submit(request)
        decision_dict = _decision_to_dict(outcome.decision)
        return OperatorActionResponse(
            approved=outcome.approved,
            summary=outcome.summary,
            decision=decision_dict,
        )

    @router.post(
        "/api/operator/action/unlock",
        response_model=OperatorActionResponse,
    )
    def operator_action_unlock(
        body: OperatorUnlockRequest,
    ) -> OperatorActionResponse:
        """Submit an operator UNLOCK (LOCKED → SAFE) through the bridge."""

        state = state_accessor()
        request = OperatorRequest(
            ts_ns=wall_ns(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_UNLOCK,
            payload={"reason": body.reason},
        )
        with state.lock:
            outcome = state.dashboard_router.submit(request)
        decision_dict = _decision_to_dict(outcome.decision)
        return OperatorActionResponse(
            approved=outcome.approved,
            summary=outcome.summary,
            decision=decision_dict,
        )

    @router.post(
        "/api/operator/action/mode",
        response_model=OperatorActionResponse,
    )
    def operator_action_mode(
        body: OperatorModeRequest,
    ) -> OperatorActionResponse:
        """Submit an operator REQUEST_MODE through the governance bridge."""

        state = state_accessor()
        payload: dict[str, str] = {
            "target_mode": body.target_mode,
            "reason": body.reason,
            "operator_authorized": "true" if body.operator_authorized else "false",
        }
        if body.consent_operator_id:
            payload["consent_operator_id"] = body.consent_operator_id
        if body.consent_policy_hash:
            payload["consent_policy_hash"] = body.consent_policy_hash
        if body.consent_nonce:
            payload["consent_nonce"] = body.consent_nonce
        if body.consent_ts_ns:
            payload["consent_ts_ns"] = str(body.consent_ts_ns)

        request = OperatorRequest(
            ts_ns=wall_ns(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_MODE,
            payload=payload,
        )
        with state.lock:
            outcome = state.dashboard_router.submit(request)
        decision_dict = _decision_to_dict(outcome.decision)
        return OperatorActionResponse(
            approved=outcome.approved,
            summary=outcome.summary,
            decision=decision_dict,
        )

    # -----------------------------------------------------------------
    # /api/operator/audit
    # -----------------------------------------------------------------
    @router.post(
        "/api/operator/audit",
        response_model=OperatorAuditResponse,
    )
    def operator_audit(body: OperatorAuditRequest) -> OperatorAuditResponse:
        """AUDIT-P1.5 — write an ``OPERATOR_SETTINGS_CHANGED`` ledger row."""

        state = state_accessor()
        ts_ns = wall_ns()
        payload: dict[str, str] = {
            "setting": body.setting,
            "previous_json": json.dumps(body.previous, sort_keys=True, default=str),
            "next_json": json.dumps(body.next, sort_keys=True, default=str),
            "autonomy_mode": body.autonomy_mode,
            "timestamp_iso": body.timestamp_iso,
        }
        with state.lock:
            entry = state.governance.ledger.append(
                ts_ns=ts_ns,
                kind=body.kind,
                payload=payload,
            )
        return OperatorAuditResponse(
            accepted=True,
            seq=entry.seq,
            kind=entry.kind,
            persisted=state.governance.ledger.db_path is not None,
        )

    # -----------------------------------------------------------------
    # /api/feeds/memecoin/summary  +  /api/wallet/info
    # -----------------------------------------------------------------
    @router.get(
        "/api/feeds/memecoin/summary",
        response_model=OperatorMemecoinSnapshot,
    )
    def feeds_memecoin_summary() -> OperatorMemecoinSnapshot:
        """AUDIT-P1.5 — typed memecoin subsystem summary."""

        state = state_accessor()
        with state.lock:
            snap = state.memecoin_widget.status()
        return OperatorMemecoinSnapshot(
            enabled=snap.enabled,
            killed=snap.killed,
            summary=snap.summary,
        )

    @router.get(
        "/api/wallet/info",
        response_model=WalletInfoResponse,
    )
    def wallet_info() -> WalletInfoResponse:
        """AUDIT-P1.5 — wallet connection summary (DISCONNECTED stub)."""

        return WalletInfoResponse(
            connected=False,
            chain="",
            address="",
            reason="wallet credentials not configured",
        )

    # -----------------------------------------------------------------
    # /api/operator/source-trust + promote / demote
    # -----------------------------------------------------------------
    def _source_trust_row(
        *,
        source_id: str,
        declared_trust: SignalTrust,
        declared_cap: float | None,
    ) -> SourceTrustRow:
        """Project one ``(source_id, declared_trust, declared_cap)`` triple."""

        state = state_accessor()
        promotion = state.signal_trust_promotions.get(source_id)
        effective_trust = state.signal_trust_promotions.effective_trust(
            source_id,
            declared_trust,
        )
        if state.signal_trust_registry is not None:
            effective_cap = state.signal_trust_registry.cap_for(source_id, effective_trust)
        else:
            effective_cap = default_cap_for(effective_trust)
        return SourceTrustRow(
            source_id=source_id,
            declared_trust=declared_trust.value,
            effective_trust=effective_trust.value,
            declared_cap=declared_cap,
            effective_cap=effective_cap,
            promoted=promotion is not None,
            promoted_target_trust=(promotion.target_trust.value if promotion is not None else ""),
            promoted_ts_ns=promotion.ts_ns if promotion is not None else 0,
            promoted_requestor=(promotion.requestor if promotion is not None else ""),
            promoted_reason=promotion.reason if promotion is not None else "",
        )

    @router.get(
        "/api/operator/source-trust",
        response_model=SourceTrustListResponse,
    )
    def operator_source_trust_list() -> SourceTrustListResponse:
        """Paper-S6 -- enumerate every source the gate knows about."""

        state = state_accessor()
        with state.lock:
            registry_rows: list[SourceTrustRow] = []
            registered_ids: set[str] = set()
            if state.signal_trust_registry is not None:
                for source_id, row in state.signal_trust_registry.sources.items():
                    registered_ids.add(source_id)
                    registry_rows.append(
                        _source_trust_row(
                            source_id=source_id,
                            declared_trust=row.trust,
                            declared_cap=row.cap,
                        )
                    )
            for source_id in state.signal_trust_promotions.list_all():
                if source_id in registered_ids:
                    continue
                registry_rows.append(
                    _source_trust_row(
                        source_id=source_id,
                        declared_trust=SignalTrust.EXTERNAL_LOW,
                        declared_cap=None,
                    )
                )
            registry_rows.sort(key=lambda r: r.source_id)
            promotion_count = len(state.signal_trust_promotions)
        return SourceTrustListResponse(
            rows=registry_rows,
            promotion_count=promotion_count,
        )

    @router.post(
        "/api/operator/source-trust/promote",
        response_model=SourceTrustPromotionResponse,
    )
    def operator_source_trust_promote(
        body: SourceTrustPromotionRequest,
    ) -> SourceTrustPromotionResponse:
        """Paper-S6 -- promote a source from EXTERNAL_LOW to EXTERNAL_MED."""

        state = state_accessor()
        try:
            target_trust = SignalTrust(body.target_trust)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"unknown trust class {body.target_trust!r}",
            ) from exc
        if not is_promotable_target(target_trust):
            raise HTTPException(
                status_code=400,
                detail=(f"only EXTERNAL_MED is a valid promotion target; got {target_trust.value}"),
            )
        ts_ns = wall_ns()
        payload: dict[str, str] = {
            "source_id": body.source_id,
            "target_trust": target_trust.value,
            "requestor": body.requestor,
            "reason": body.reason,
            "ts_ns": str(ts_ns),
        }
        with state.lock:
            entry = state.governance.ledger.append(
                ts_ns=ts_ns,
                kind=PROMOTION_LEDGER_KIND,
                payload=payload,
            )
            state.signal_trust_promotions.promote(
                source_id=body.source_id,
                target_trust=target_trust,
                requestor=body.requestor,
                reason=body.reason,
                ts_ns=ts_ns,
            )
            if (
                state.signal_trust_registry is not None
                and body.source_id in state.signal_trust_registry.sources
            ):
                row = state.signal_trust_registry.sources[body.source_id]
                declared_trust = row.trust
                declared_cap = row.cap
            else:
                declared_trust = SignalTrust.EXTERNAL_LOW
                declared_cap = None
            projection = _source_trust_row(
                source_id=body.source_id,
                declared_trust=declared_trust,
                declared_cap=declared_cap,
            )
        return SourceTrustPromotionResponse(
            accepted=True,
            source_id=projection.source_id,
            declared_trust=projection.declared_trust,
            effective_trust=projection.effective_trust,
            declared_cap=projection.declared_cap,
            effective_cap=projection.effective_cap,
            promoted=projection.promoted,
            promoted_target_trust=projection.promoted_target_trust,
            promoted_ts_ns=projection.promoted_ts_ns,
            promoted_requestor=projection.promoted_requestor,
            promoted_reason=projection.promoted_reason,
            ledger_seq=entry.seq,
            ledger_kind=entry.kind,
        )

    @router.post(
        "/api/operator/source-trust/demote",
        response_model=SourceTrustPromotionResponse,
    )
    def operator_source_trust_demote(
        body: SourceTrustDemotionRequest,
    ) -> SourceTrustPromotionResponse:
        """Paper-S6 -- revert a previously-applied operator promotion."""

        state = state_accessor()
        ts_ns = wall_ns()
        payload: dict[str, str] = {
            "source_id": body.source_id,
            "requestor": body.requestor,
            "reason": body.reason,
            "ts_ns": str(ts_ns),
        }
        with state.lock:
            entry = state.governance.ledger.append(
                ts_ns=ts_ns,
                kind=DEMOTION_LEDGER_KIND,
                payload=payload,
            )
            state.signal_trust_promotions.demote(body.source_id)
            if (
                state.signal_trust_registry is not None
                and body.source_id in state.signal_trust_registry.sources
            ):
                row = state.signal_trust_registry.sources[body.source_id]
                declared_trust = row.trust
                declared_cap = row.cap
            else:
                declared_trust = SignalTrust.EXTERNAL_LOW
                declared_cap = None
            projection = _source_trust_row(
                source_id=body.source_id,
                declared_trust=declared_trust,
                declared_cap=declared_cap,
            )
        return SourceTrustPromotionResponse(
            accepted=True,
            source_id=projection.source_id,
            declared_trust=projection.declared_trust,
            effective_trust=projection.effective_trust,
            declared_cap=projection.declared_cap,
            effective_cap=projection.effective_cap,
            promoted=projection.promoted,
            promoted_target_trust=projection.promoted_target_trust,
            promoted_ts_ns=projection.promoted_ts_ns,
            promoted_requestor=projection.promoted_requestor,
            promoted_reason=projection.promoted_reason,
            ledger_seq=entry.seq,
            ledger_kind=entry.kind,
        )

    # -----------------------------------------------------------------
    # /api/operator/learning-override
    # -----------------------------------------------------------------
    def _project_learning_override(*, enabled: bool, mode: SystemMode) -> LearningOverrideResponse:
        """Build a typed response from a snapshotted (enabled, mode) tuple."""

        policy = LearningEvolutionFreezePolicy(mode=mode, operator_override=enabled)
        return LearningOverrideResponse(
            enabled=enabled,
            mode=mode.name,
            is_freeze_active=policy.is_frozen(),
        )

    def _learning_override_response() -> LearningOverrideResponse:
        """Snapshot the live learning-override flag + freeze state."""

        state = state_accessor()
        with state.lock:
            enabled = state.learning_override_enabled
            mode = state.governance.state_transitions.current_mode()
        return _project_learning_override(enabled=enabled, mode=mode)

    @router.get(
        "/api/operator/learning-override",
        response_model=LearningOverrideResponse,
    )
    def operator_learning_override_get() -> LearningOverrideResponse:
        """AUDIT-P1.7 — typed read of the operator learning-override flag."""

        return _learning_override_response()

    @router.post(
        "/api/operator/learning-override",
        response_model=LearningOverrideResponse,
    )
    def operator_learning_override_post(
        body: LearningOverrideRequest,
    ) -> LearningOverrideResponse:
        """AUDIT-P1.7 — flip the learning-override flag with audit."""

        state = state_accessor()
        new_enabled = bool(body.enabled)
        with state.lock:
            previous = state.learning_override_enabled
            state.learning_override_enabled = new_enabled
            mode = state.governance.state_transitions.current_mode()
            flip_ts_ns = wall_ns()
            state.governance.ledger.append(
                ts_ns=flip_ts_ns,
                kind="OPERATOR_LEARNING_OVERRIDE_CHANGED",
                payload={
                    "requestor": body.requestor,
                    "reason": body.reason,
                    "previous": "true" if previous else "false",
                    "next": "true" if new_enabled else "false",
                    "mode": mode.name,
                },
            )
            policy_after = LearningEvolutionFreezePolicy(
                mode=mode,
                operator_override=new_enabled,
            )
            policy_event = policy_after.to_system_event(
                ts_ns=flip_ts_ns,
                source="operator.api",
            )
            state.governance.ledger.append(
                ts_ns=policy_event.ts_ns,
                kind=policy_event.sub_kind.value,
                payload=dict(policy_event.payload),
            )
        return _project_learning_override(enabled=new_enabled, mode=mode)

    # -----------------------------------------------------------------
    # /api/operator/development-mode  +  /api/operator/trading-allowed
    # -----------------------------------------------------------------
    def _project_development_mode_policy(
        policy: DevelopmentModePolicy,
    ) -> DevelopmentModeResponse:
        """Project a :class:`DevelopmentModePolicy` snapshot to the wire."""

        return DevelopmentModeResponse(
            development_enabled=policy.development_enabled,
            trading_allowed=policy.trading_allowed,
            mode=policy.mode.name if policy.mode is not None else "",
            learning_unblocked=policy.is_learning_unblocked(),
            trading_unblocked=policy.is_trading_unblocked(),
            policy_version=DEVELOPMENT_MODE_POLICY_VERSION,
        )

    def _development_mode_snapshot() -> DevelopmentModeResponse:
        """Snapshot the live :class:`DevelopmentModePolicy` from ``_State``."""

        state = state_accessor()
        with state.lock:
            policy = DevelopmentModePolicy(
                development_enabled=state.development_mode_enabled,
                trading_allowed=state.trading_allowed,
                mode=state.governance.state_transitions.current_mode(),
            )
        return _project_development_mode_policy(policy)

    @router.get(
        "/api/operator/development-mode",
        response_model=DevelopmentModeResponse,
    )
    def operator_development_mode_get() -> DevelopmentModeResponse:
        """PR-DEV-A — typed read of the operator development-mode policy."""

        return _development_mode_snapshot()

    @router.post(
        "/api/operator/development-mode",
        response_model=DevelopmentModeResponse,
    )
    def operator_development_mode_post(
        body: DevelopmentModeRequest,
    ) -> DevelopmentModeResponse:
        """PR-DEV-A — flip the development-mode flag with audit."""

        state = state_accessor()
        new_enabled = bool(body.enabled)
        with state.lock:
            previous = state.development_mode_enabled
            state.development_mode_enabled = new_enabled
            mode = state.governance.state_transitions.current_mode()
            flip_ts_ns = wall_ns()
            state.governance.ledger.append(
                ts_ns=flip_ts_ns,
                kind="OPERATOR_DEVELOPMENT_MODE_CHANGED",
                payload={
                    "requestor": body.requestor,
                    "reason": body.reason,
                    "previous": "true" if previous else "false",
                    "next": "true" if new_enabled else "false",
                    "mode": mode.name,
                },
            )
            policy_after = DevelopmentModePolicy(
                development_enabled=new_enabled,
                trading_allowed=state.trading_allowed,
                mode=mode,
            )
            policy_event = policy_after.to_system_event(
                ts_ns=flip_ts_ns,
                source="operator.api",
            )
            state.governance.ledger.append(
                ts_ns=policy_event.ts_ns,
                kind=policy_event.sub_kind.value,
                payload=dict(policy_event.payload),
            )
            state.development_mode_policy = policy_after
            state.execution.set_development_mode_policy(policy_after)
        return _project_development_mode_policy(policy_after)

    @router.get(
        "/api/operator/trading-allowed",
        response_model=DevelopmentModeResponse,
    )
    def operator_trading_allowed_get() -> DevelopmentModeResponse:
        """PR-DEV-A — typed read of the trading-allowed gate."""

        return _development_mode_snapshot()

    @router.post(
        "/api/operator/trading-allowed",
        response_model=DevelopmentModeResponse,
    )
    def operator_trading_allowed_post(
        body: TradingAllowedRequest,
    ) -> DevelopmentModeResponse:
        """PR-DEV-A — flip the trading-allowed flag with audit."""

        state = state_accessor()
        new_enabled = bool(body.enabled)
        with state.lock:
            previous = state.trading_allowed
            state.trading_allowed = new_enabled
            mode = state.governance.state_transitions.current_mode()
            flip_ts_ns = wall_ns()
            state.governance.ledger.append(
                ts_ns=flip_ts_ns,
                kind="OPERATOR_TRADING_ALLOWED_CHANGED",
                payload={
                    "requestor": body.requestor,
                    "reason": body.reason,
                    "previous": "true" if previous else "false",
                    "next": "true" if new_enabled else "false",
                    "mode": mode.name,
                },
            )
            policy_after = DevelopmentModePolicy(
                development_enabled=state.development_mode_enabled,
                trading_allowed=new_enabled,
                mode=mode,
            )
            policy_event = policy_after.to_system_event(
                ts_ns=flip_ts_ns,
                source="operator.api",
            )
            state.governance.ledger.append(
                ts_ns=policy_event.ts_ns,
                kind=policy_event.sub_kind.value,
                payload=dict(policy_event.payload),
            )
            state.development_mode_policy = policy_after
            state.execution.set_development_mode_policy(policy_after)
        return _project_development_mode_policy(policy_after)

    return router


__all__ = ["build_operator_router"]
