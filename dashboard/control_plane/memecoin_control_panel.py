"""Memecoin Control Panel — Phase 6 IMMUTABLE WIDGET 5 (DASH-MCP-01).

The memecoin subsystem is **isolated** per Build Compiler Spec §6: it
runs as a separate process / lifecycle, with its own enable/disable
gate and its own kill switch. The Memecoin Control Panel is the
dashboard's read-and-request surface for that subsystem.

The actual memecoin runtime is not yet implemented (deferred to a
later phase). This widget therefore exposes:

* A lightweight read projection (:class:`MemecoinSubsystemStatus`) that
  reports whether the subsystem is enabled, whether the kill switch
  is engaged, and a UI-friendly summary string.
* A request seam: every operator action (enable, disable, kill) is
  forwarded through the GOV-CP-07 :class:`OperatorInterfaceBridge` as
  a ``REQUEST_PLUGIN_LIFECYCLE`` request scoped to the memecoin
  process group. No memecoin state mutates here directly.

Authority constraints: identical to every other Phase 6 widget — the
dashboard reads, requests, and never writes. (INV-37, B7 lint)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
)
from dashboard.control_plane.router import ControlPlaneRouter, RouteOutcome

# The memecoin subsystem is treated by Governance as a single plugin
# group; per-action ``target_status`` matches the bridge's lifecycle
# vocabulary (``SHADOW``/``ACTIVE``/``DISABLED``).
_MEMECOIN_PLUGIN_PATH = "memecoin/subsystem"
_LIFECYCLE_ENABLE = "ACTIVE"
_LIFECYCLE_DISABLE = "DISABLED"
_LIFECYCLE_KILL = "DISABLED"


@dataclass(frozen=True, slots=True)
class MemecoinSubsystemStatus:
    """Renderable read-projection for the memecoin subsystem.

    For Phase 6 scaffolding the status is held in-process — it is
    bumped by :meth:`MemecoinControlPanel.note_*`, and only by
    Governance-approved decisions returning to the panel via the
    router. The full runtime implementation will replace this with a
    ledger-derived projection.
    """

    enabled: bool
    killed: bool
    summary: str


class MemecoinControlPanel:
    """DASH-MCP-01 — Memecoin Control Panel widget backend."""

    name: str = "memecoin_control_panel"
    spec_id: str = "DASH-MCP-01"

    def __init__(self, *, router: ControlPlaneRouter) -> None:
        self._router = router
        self._status = MemecoinSubsystemStatus(
            enabled=False,
            killed=False,
            summary="memecoin subsystem disabled (default)",
        )

    def status(self) -> MemecoinSubsystemStatus:
        return self._status

    def request_enable(
        self, *, ts_ns: int, requestor: str, reason: str
    ) -> RouteOutcome:
        return self._submit(
            ts_ns=ts_ns,
            requestor=requestor,
            payload={
                "plugin_path": _MEMECOIN_PLUGIN_PATH,
                "target_status": _LIFECYCLE_ENABLE,
                "reason": reason,
            },
            on_approve=lambda: self._note(
                enabled=True,
                killed=False,
                summary=f"memecoin enabled by {requestor}",
            ),
        )

    def request_disable(
        self, *, ts_ns: int, requestor: str, reason: str
    ) -> RouteOutcome:
        return self._submit(
            ts_ns=ts_ns,
            requestor=requestor,
            payload={
                "plugin_path": _MEMECOIN_PLUGIN_PATH,
                "target_status": _LIFECYCLE_DISABLE,
                "reason": reason,
            },
            on_approve=lambda: self._note(
                enabled=False,
                killed=False,
                summary=f"memecoin disabled by {requestor}",
            ),
        )

    def request_kill(
        self, *, ts_ns: int, requestor: str, reason: str
    ) -> RouteOutcome:
        return self._submit(
            ts_ns=ts_ns,
            requestor=requestor,
            payload={
                "plugin_path": _MEMECOIN_PLUGIN_PATH,
                "target_status": _LIFECYCLE_KILL,
                "reason": f"KILL: {reason}",
            },
            on_approve=lambda: self._note(
                enabled=False,
                killed=True,
                summary=f"memecoin killed by {requestor}",
            ),
        )

    # ------------------------------------------------------------------

    def _submit(
        self,
        *,
        ts_ns: int,
        requestor: str,
        payload: Mapping[str, str],
        on_approve,
    ) -> RouteOutcome:
        outcome = self._router.submit(
            OperatorRequest(
                ts_ns=ts_ns,
                requestor=requestor,
                action=OperatorAction.REQUEST_PLUGIN_LIFECYCLE,
                payload=payload,
            )
        )
        if outcome.approved:
            on_approve()
        return outcome

    def _note(self, *, enabled: bool, killed: bool, summary: str) -> None:
        self._status = MemecoinSubsystemStatus(
            enabled=enabled, killed=killed, summary=summary
        )
