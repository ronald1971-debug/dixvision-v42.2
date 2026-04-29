"""Pydantic response models for the ``/api/operator/*`` HTTP surface.

Wave-02 PR-2 (operator dashboard React port). The legacy
``/api/dashboard/*`` endpoints remain untouched; they return loosely-
typed ``dict[str, Any]`` projections of the Phase 6 widget snapshots
and are consumed by the vanilla ``/operator`` page.

This module ships a typed parallel surface used by the React port at
``/dash2/#/operator``:

* ``GET /api/operator/summary`` — read-only snapshot of mode, engine
  health, strategy counts per FSM state, and memecoin status.
* ``POST /api/operator/action/kill`` — operator KILL request that
  enters the system through ``ControlPlaneRouter`` →
  ``OperatorInterfaceBridge`` (GOV-CP-07). The route handler never
  writes the ledger and never bypasses Governance — it constructs the
  typed request, submits it, and returns the resulting decision
  verbatim (so a rejection is visible in the UI).

The response shapes are deliberately a *slim* projection of the legacy
widget snapshots. Adding richer detail (decision-trace chains, per-
strategy history, plugin transition history) is reserved for follow-
up wave-02 PRs so each port is reviewable on its own.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OperatorModeSnapshot(BaseModel):
    """Mode FSM read projection (DASH-02)."""

    model_config = ConfigDict(extra="forbid")

    current_mode: str
    legal_targets: list[str]
    is_locked: bool


class OperatorEngineRow(BaseModel):
    """One row of the engine status grid (DASH-EG-01)."""

    model_config = ConfigDict(extra="forbid")

    engine_name: str
    bucket: str  # alive | degraded | halted | offline
    detail: str
    plugin_count: int


class OperatorStrategyCounts(BaseModel):
    """Per-state strategy counts (DASH-SLP-01).

    The full per-strategy roll-up stays on the legacy endpoint; the
    React port renders aggregate counts only in this PR.
    """

    model_config = ConfigDict(extra="forbid")

    proposed: int
    shadow: int
    canary: int
    live: int
    retired: int
    failed: int


class OperatorMemecoinSnapshot(BaseModel):
    """Memecoin subsystem read projection (DASH-MCP-01)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    killed: bool
    summary: str


class OperatorSummaryResponse(BaseModel):
    """Top-level body of ``GET /api/operator/summary``."""

    model_config = ConfigDict(extra="forbid")

    mode: OperatorModeSnapshot
    engines: list[OperatorEngineRow]
    strategies: OperatorStrategyCounts
    memecoin: OperatorMemecoinSnapshot
    decision_chain_count: int


class OperatorKillRequest(BaseModel):
    """Operator KILL request body for ``POST /api/operator/action/kill``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field("operator kill", max_length=512)
    requestor: str = Field("operator", min_length=1, max_length=64)


class OperatorActionResponse(BaseModel):
    """Result envelope for every ``POST /api/operator/action/*`` route.

    The ``decision`` payload is intentionally typed as ``dict`` because
    governance decisions carry per-action shapes that are not part of
    the typed wave-02 surface; the React UI only reads ``approved`` and
    ``summary`` to render the action log. A future wave-02 PR can lift
    governance decisions into the typed contract.
    """

    model_config = ConfigDict(extra="forbid")

    approved: bool
    summary: str
    decision: dict


__all__ = [
    "OperatorActionResponse",
    "OperatorEngineRow",
    "OperatorKillRequest",
    "OperatorMemecoinSnapshot",
    "OperatorModeSnapshot",
    "OperatorStrategyCounts",
    "OperatorSummaryResponse",
]
