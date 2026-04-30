"""Mode Control Bar — Phase 6 IMMUTABLE WIDGET 1 (DASH-02).

Exposes the system mode + the legal mode-transition graph for the UI,
and turns operator clicks into :class:`OperatorRequest` objects routed
through GOV-CP-07.

Authority constraints (Build Compiler Spec §6 + §7):

* The UI may *display* the current mode and the set of legal next
  modes. (read)
* The UI may *request* a mode transition. (write via GOV-CP-07 only)
* The UI may *not* change the mode directly. (INV-37)

The Mode FSM is locked by Build Compiler Spec §7. This widget never
encodes the legality rules itself — it asks the
:class:`StateTransitionManager` what the legal next modes are. That
keeps the dashboard's view of the FSM consistent with the actual
governance rules even if the FSM is later refined.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from dashboard_backend.control_plane.router import ControlPlaneRouter, RouteOutcome
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
    _is_legal_edge,
)


@dataclass(frozen=True, slots=True)
class ModeControlBarState:
    """Renderable read-projection for the Mode Control Bar."""

    current_mode: str
    legal_targets: tuple[str, ...]
    is_locked: bool


class ModeControlBar:
    """DASH-02 — Mode Control Bar widget backend."""

    name: str = "mode_control_bar"
    spec_id: str = "DASH-02"

    def __init__(
        self,
        *,
        state_transitions: StateTransitionManager,
        router: ControlPlaneRouter,
    ) -> None:
        self._state = state_transitions
        self._router = router

    def snapshot(self) -> ModeControlBarState:
        current = self._state.current_mode()
        legal = tuple(
            target.name
            for target in SystemMode
            if target is not current and _is_legal_edge(current, target)[0]
        )
        return ModeControlBarState(
            current_mode=current.name,
            legal_targets=legal,
            is_locked=current is SystemMode.LOCKED,
        )

    def request_transition(
        self,
        *,
        ts_ns: int,
        requestor: str,
        target_mode: str,
        reason: str,
        operator_authorized: bool = False,
    ) -> RouteOutcome:
        if target_mode not in SystemMode.__members__:
            raise ValueError(f"unknown target mode: {target_mode!r}")
        payload: Mapping[str, str] = {
            "target_mode": target_mode,
            "reason": reason,
            "operator_authorized": "true" if operator_authorized else "false",
        }
        return self._router.submit(
            OperatorRequest(
                ts_ns=ts_ns,
                requestor=requestor,
                action=OperatorAction.REQUEST_MODE,
                payload=payload,
            )
        )

    def request_kill(
        self, *, ts_ns: int, requestor: str, reason: str
    ) -> RouteOutcome:
        return self._router.submit(
            OperatorRequest(
                ts_ns=ts_ns,
                requestor=requestor,
                action=OperatorAction.REQUEST_KILL,
                payload={"reason": reason},
            )
        )

    def request_unlock(
        self, *, ts_ns: int, requestor: str, reason: str
    ) -> RouteOutcome:
        return self._router.submit(
            OperatorRequest(
                ts_ns=ts_ns,
                requestor=requestor,
                action=OperatorAction.REQUEST_UNLOCK,
                payload={"reason": reason},
            )
        )
