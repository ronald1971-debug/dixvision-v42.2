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


class DevelopmentModeRequest(BaseModel):
    """Body for ``POST /api/operator/development-mode`` (PR-DEV-A).

    The operator-toggled flag that determines whether Indira and Dyon
    run at full potential (trader discovery, profiling, modeling,
    heavy learning, structural evolution, slow-loop critique, patch
    pipeline). Defaults to ``True`` at boot via
    ``DIXVISION_DEVELOPMENT_MODE``; flipping to ``False`` pauses the
    learning + research surface while leaving every safety gate in
    place. Independent of :class:`TradingAllowedRequest` -- the two
    flags compose into one ``DevelopmentModePolicy``.
    """

    enabled: bool = Field(
        ...,
        description=(
            "Target value of the development-mode flag. When True "
            "(the boot default), Indira and Dyon are unblocked."
        ),
    )
    requestor: str = Field(
        default="dashboard",
        min_length=1,
        max_length=64,
        description=(
            "Caller identity recorded on the audit row; defaults to the dashboard origin."
        ),
    )
    reason: str = Field(
        default="",
        max_length=256,
        description=("Free-form rationale; included verbatim in the audit ledger payload."),
    )


class TradingAllowedRequest(BaseModel):
    """Body for ``POST /api/operator/trading-allowed`` (PR-DEV-A).

    The single operator-toggled switch that opens the Execution Gate.
    Defaults to ``False`` at boot via ``DIXVISION_TRADING_ALLOWED``;
    while False the gate emits a synthetic ``REJECTED`` ExecutionEvent
    with ``meta.reason='development_mode_trading_blocked'`` for every
    intent. The AuthorityGuard, hazard throttle, kill-switch,
    mode-effect table, FSM consent envelopes, and HARDEN-04 remain in
    force as defense-in-depth.
    """

    enabled: bool = Field(
        ...,
        description=(
            "Target value of the trading-allowed flag. When True "
            "the Execution Gate dispatches normally; when False "
            "(the boot default) the gate refuses all dispatch."
        ),
    )
    requestor: str = Field(
        default="dashboard",
        min_length=1,
        max_length=64,
        description=(
            "Caller identity recorded on the audit row; defaults to the dashboard origin."
        ),
    )
    reason: str = Field(
        default="",
        max_length=256,
        description=("Free-form rationale; included verbatim in the audit ledger payload."),
    )


class DevelopmentModeResponse(BaseModel):
    """Response shape for GET / POST development-mode + trading-allowed.

    Carries the full ``DevelopmentModePolicy`` projection so the
    cockpit can render both flags + the live ``SystemMode`` from a
    single round-trip.
    """

    development_enabled: bool
    trading_allowed: bool
    mode: str
    learning_unblocked: bool
    trading_unblocked: bool
    policy_version: str


class LearningOverrideRequest(BaseModel):
    """Body for ``POST /api/operator/learning-override`` (AUDIT-P1.7).

    The operator-toggled override that, in conjunction with
    ``mode is SystemMode.LIVE``, unfreezes the
    ``LearningEvolutionFreezePolicy`` so the slow learning loop +
    evolution patch pipeline can emit mutations. Defaults to ``False``
    so flipping the override is always an explicit operator act (B36 /
    HARDEN-04 invariant).
    """

    enabled: bool = Field(
        ...,
        description=(
            "Target value of the operator override. Adaptive "
            "mutations require both this flag to be True *and* "
            "the system mode to be LIVE."
        ),
    )
    requestor: str = Field(
        default="dashboard",
        min_length=1,
        max_length=64,
        description=(
            "Caller identity recorded on the audit row; defaults to the dashboard origin."
        ),
    )
    reason: str = Field(
        default="",
        max_length=256,
        description=("Free-form rationale; included verbatim in the audit ledger payload."),
    )


class LearningOverrideResponse(BaseModel):
    """Response shape for both GET and POST learning-override routes."""

    enabled: bool
    mode: str
    is_freeze_active: bool


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

    Mirrors :class:`intelligence_engine.strategy_runtime.state_machine.StrategyState`
    one-for-one. Strategy-level SHADOW was demolished by
    SHADOW-DEMOLITION-02 (PR #216); the surviving lifecycle is
    ``PROPOSED → CANARY → LIVE → RETIRED`` (plus ``FAILED`` from
    anywhere). The signals-on/execution-off observation tier now
    lives only at the system-mode layer (``PAPER``).
    """

    model_config = ConfigDict(extra="forbid")

    proposed: int
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


class OperatorUnlockRequest(BaseModel):
    """Operator UNLOCK request body for ``POST /api/operator/action/unlock``.

    Drives the LOCKED -> SAFE edge through ``ControlPlaneRouter`` ->
    ``OperatorInterfaceBridge`` (REQUEST_UNLOCK). The state transition
    manager re-resolves the current mode under its lock; submitting
    while not locked returns the mode-FSM rejection verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    reason: str = Field("operator unlock", max_length=512)
    requestor: str = Field("operator", min_length=1, max_length=64)


class OperatorModeRequest(BaseModel):
    """Operator REQUEST_MODE body for ``POST /api/operator/action/mode``.

    The operator-side mode-transition surface for the typed React
    port. Mirrors the legacy ``/api/dashboard/action/mode`` shape so
    the same governance bridge handles both. Consent envelope fields
    are optional and only consulted by ``OperatorInterfaceBridge`` on
    edges that ``edge_requires_consent`` reports as gated.
    """

    model_config = ConfigDict(extra="forbid")

    target_mode: str = Field(..., min_length=1, max_length=16)
    reason: str = Field("operator mode change", max_length=512)
    requestor: str = Field("operator", min_length=1, max_length=64)
    operator_authorized: bool = False
    consent_operator_id: str = Field("", max_length=64)
    consent_policy_hash: str = Field("", max_length=128)
    consent_nonce: str = Field("", max_length=64)
    consent_ts_ns: int = 0


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
    "DevelopmentModeRequest",
    "DevelopmentModeResponse",
    "LearningOverrideRequest",
    "LearningOverrideResponse",
    "OperatorActionResponse",
    "OperatorAuditRequest",
    "OperatorAuditResponse",
    "OperatorEngineRow",
    "OperatorKillRequest",
    "OperatorMemecoinSnapshot",
    "OperatorModeRequest",
    "OperatorModeSnapshot",
    "OperatorStrategyCounts",
    "OperatorSummaryResponse",
    "OperatorUnlockRequest",
    "TradingAllowedRequest",
    "WalletInfoResponse",
]
