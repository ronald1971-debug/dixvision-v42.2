"""DASH-1 — read-only HTTP projections of the five Phase 6 widgets.

Per Build Compiler Spec §6, the operator dashboard is a **Control
Plane**. Phase 6 shipped five immutable widgets (DASH-02 / DASH-EG-01
/ DASH-SLP-01 / DASH-04 / DASH-MCP-01) as pure-Python data shapers
producing frozen dataclass snapshots. This module is the thin HTTP
seam that exposes each widget's read projection as JSON, so the
single-page operator UI can render them.

Each endpoint is GET-only. Operator *actions* (mode change, kill,
intent, plugin lifecycle) are not handled here — they enter the system
through the GOV-CP-07 ``OperatorInterfaceBridge`` and will land in
DASH-2.

Authority constraints (B7 lint):

* This module does not import any ``*_engine`` package — engines and
  widgets are passed in via the :class:`DashboardState` accessor that
  the host (``ui.server``) provides.
* This module never writes the ledger or constructs governance
  decisions. Every endpoint returns a snapshot only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.coherence.system_intent import (
    INTENT_KEY_FOCUS,
    INTENT_KEY_HORIZON,
    INTENT_KEY_OBJECTIVE,
    INTENT_KEY_REASON,
    INTENT_KEY_RISK_MODE,
    encode_focus,
)
from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
)
from dashboard.control_plane.decision_trace import DecisionTracePanel
from dashboard.control_plane.engine_status_grid import EngineStatusGrid
from dashboard.control_plane.memecoin_control_panel import MemecoinControlPanel
from dashboard.control_plane.mode_control_bar import ModeControlBar
from dashboard.control_plane.router import ControlPlaneRouter
from dashboard.control_plane.strategy_lifecycle_panel import (
    StrategyLifecyclePanel,
)


class DashboardWidgets(Protocol):
    """Read-only accessor that the host installs into the FastAPI app.

    The protocol exists so the FastAPI module never imports concrete
    widget instances — it only knows there is a callable returning the
    five widget objects, which keeps the route module decoupled from
    the engine wiring in :mod:`ui.server`.
    """

    lock: threading.Lock

    @property
    def mode(self) -> ModeControlBar: ...
    @property
    def engines(self) -> EngineStatusGrid: ...
    @property
    def strategies(self) -> StrategyLifecyclePanel: ...
    @property
    def decisions(self) -> DecisionTracePanel: ...
    @property
    def memecoin(self) -> MemecoinControlPanel: ...
    @property
    def dashboard_router(self) -> ControlPlaneRouter: ...

    def next_ts(self) -> int: ...


_WidgetsProvider = Callable[[], DashboardWidgets]


# ---------------------------------------------------------------------------
# Pydantic action request models (DASH-2)
# ---------------------------------------------------------------------------


class ModeActionIn(BaseModel):
    target_mode: str = Field(..., min_length=1, max_length=32)
    reason: str = Field("", max_length=512)
    operator_authorized: bool = False
    requestor: str = Field("operator", min_length=1, max_length=64)


class IntentActionIn(BaseModel):
    objective: str = Field(..., min_length=1, max_length=64)
    risk_mode: str = Field(..., min_length=1, max_length=64)
    horizon: str = Field(..., min_length=1, max_length=64)
    focus: Sequence[str] = Field(default_factory=tuple)
    reason: str = Field("", max_length=512)
    requestor: str = Field("operator", min_length=1, max_length=64)


class KillActionIn(BaseModel):
    reason: str = Field("operator kill", max_length=512)
    requestor: str = Field("operator", min_length=1, max_length=64)


class LifecycleActionIn(BaseModel):
    plugin_path: str = Field(..., min_length=1, max_length=256)
    target_status: str = Field(..., min_length=1, max_length=64)
    reason: str = Field("", max_length=512)
    requestor: str = Field("operator", min_length=1, max_length=64)


def _to_dict(obj: Any) -> Any:
    """JSON-friendly conversion for frozen dataclass snapshots.

    Handles nested dataclasses, tuples, dicts, and StrEnum values via
    ``str()`` fallback. Snapshot dataclasses are slotted + frozen, so
    ``asdict`` walks them deterministically (same input → same JSON).
    """

    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_dict(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_dashboard_router(provider: _WidgetsProvider) -> APIRouter:
    """Construct the read-only ``/api/dashboard/...`` router.

    The host passes a zero-arg ``provider`` that returns the latest
    widget bundle — this keeps the router stateless and lets the host
    swap state safely under a lock.
    """

    router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

    @router.get("/mode")
    def get_mode() -> dict[str, Any]:
        widgets = provider()
        return {"mode": _to_dict(widgets.mode.snapshot())}

    @router.get("/engines")
    def get_engines() -> dict[str, Any]:
        widgets = provider()
        return {"engines": _to_dict(widgets.engines.snapshot())}

    @router.get("/strategies")
    def get_strategies() -> dict[str, Any]:
        widgets = provider()
        return {"strategies": _to_dict(widgets.strategies.by_state())}

    @router.get("/decisions")
    def get_decisions(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        widgets = provider()
        return {
            "chains": _to_dict(widgets.decisions.chains(limit=limit)),
        }

    @router.get("/memecoin")
    def get_memecoin() -> dict[str, Any]:
        widgets = provider()
        return {"memecoin": _to_dict(widgets.memecoin.status())}

    # ------------------------------------------------------------------
    # DASH-2 — operator action endpoints
    # ------------------------------------------------------------------
    #
    # Each action posts an ``OperatorRequest`` through the
    # ``ControlPlaneRouter`` (DASH-CP-01) → ``OperatorInterfaceBridge``
    # (GOV-CP-07). The route handler never writes the ledger directly
    # and never bypasses Governance — it constructs the typed request,
    # submits it, and returns the resulting ``GovernanceDecision``
    # verbatim (so a rejection is visible in the UI).

    def _submit(
        widgets: DashboardWidgets,
        request: OperatorRequest,
    ) -> dict[str, Any]:
        with widgets.lock:
            outcome = widgets.dashboard_router.submit(request)
        return {
            "approved": outcome.approved,
            "summary": outcome.summary,
            "decision": _to_dict(outcome.decision),
        }

    @router.post("/action/mode")
    def action_mode(body: ModeActionIn) -> dict[str, Any]:
        widgets = provider()
        request = OperatorRequest(
            ts_ns=widgets.next_ts(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_MODE,
            payload={
                "target_mode": body.target_mode,
                "reason": body.reason,
                "operator_authorized": "true"
                if body.operator_authorized
                else "false",
            },
        )
        return _submit(widgets, request)

    @router.post("/action/intent")
    def action_intent(body: IntentActionIn) -> dict[str, Any]:
        widgets = provider()
        request = OperatorRequest(
            ts_ns=widgets.next_ts(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_INTENT,
            payload={
                INTENT_KEY_OBJECTIVE: body.objective,
                INTENT_KEY_RISK_MODE: body.risk_mode,
                INTENT_KEY_HORIZON: body.horizon,
                INTENT_KEY_FOCUS: encode_focus(tuple(body.focus)),
                INTENT_KEY_REASON: body.reason,
            },
        )
        return _submit(widgets, request)

    @router.post("/action/kill")
    def action_kill(body: KillActionIn) -> dict[str, Any]:
        widgets = provider()
        request = OperatorRequest(
            ts_ns=widgets.next_ts(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_KILL,
            payload={"reason": body.reason},
        )
        return _submit(widgets, request)

    @router.post("/action/lifecycle")
    def action_lifecycle(body: LifecycleActionIn) -> dict[str, Any]:
        widgets = provider()
        request = OperatorRequest(
            ts_ns=widgets.next_ts(),
            requestor=body.requestor,
            action=OperatorAction.REQUEST_PLUGIN_LIFECYCLE,
            payload={
                "plugin_path": body.plugin_path,
                "target_status": body.target_status,
                "reason": body.reason,
            },
        )
        return _submit(widgets, request)

    @router.get("/summary")
    def get_summary() -> dict[str, Any]:
        """Composite snapshot — one round-trip for the index page."""

        widgets = provider()
        try:
            return {
                "mode": _to_dict(widgets.mode.snapshot()),
                "engines": _to_dict(widgets.engines.snapshot()),
                "strategies": _to_dict(widgets.strategies.by_state()),
                "memecoin": _to_dict(widgets.memecoin.status()),
                "chains": _to_dict(widgets.decisions.chains(limit=50)),
            }
        except Exception as exc:  # pragma: no cover — surfaced as 500
            raise HTTPException(500, f"summary failed: {exc!r}") from exc

    return router


__all__ = ["DashboardWidgets", "build_dashboard_router"]
