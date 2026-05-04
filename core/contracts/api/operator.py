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

from typing import Any

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


class OperatorAuditRequest(BaseModel):
    """Settings-changed audit body for ``POST /api/operator/audit``.

    AUDIT-P1.5 -- the dashboard fires a fire-and-forget POST on every
    autonomy-mode flip and SL/TP commit so the change is captured in
    the authority ledger as ``OPERATOR_SETTINGS_CHANGED``. Without
    this route the dashboard's existing call silently 404s and the
    ledger never sees the transition.

    ``previous`` / ``next`` accept arbitrary JSON because the
    dashboard ships richer shapes than a plain string (the SL/TP
    builder commits a whole form object, the autonomy panel commits
    a mode label). The route handler serialises both to JSON strings
    when constructing the ledger payload so the row stays
    ``Mapping[str, str]``-shaped.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., min_length=1, max_length=128)
    setting: str = Field(..., min_length=1, max_length=128)
    previous: Any = None
    next: Any = None
    autonomy_mode: str = Field("", max_length=64)
    timestamp_iso: str = Field("", max_length=64)


class OperatorAuditResponse(BaseModel):
    """Acknowledgement envelope for ``POST /api/operator/audit``."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    seq: int
    kind: str
    persisted: bool


class WalletInfoResponse(BaseModel):
    """Disconnected stub for ``GET /api/wallet/info`` (AUDIT-P1.5).

    The dash_meme ``WalletInfoPage`` historically read from
    ``/api/dashboard/summary``; the audit referenced
    ``/api/wallet/info`` as a typed surface that returns wallet
    connection status without dragging the whole dashboard payload.
    Until real wallet credentials are wired (UniswapX EIP-712 signer
    + Solana keypair), this route reports DISCONNECTED with an
    explicit reason so the UI can render an actionable message
    instead of a generic null.
    """

    model_config = ConfigDict(extra="forbid")

    connected: bool
    chain: str
    address: str
    reason: str


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
    "OperatorAuditRequest",
    "OperatorAuditResponse",
    "OperatorEngineRow",
    "OperatorKillRequest",
    "OperatorMemecoinSnapshot",
    "OperatorModeSnapshot",
    "OperatorStrategyCounts",
    "OperatorSummaryResponse",
    "WalletInfoResponse",
]
