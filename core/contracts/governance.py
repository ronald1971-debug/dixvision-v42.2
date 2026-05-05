"""Governance contracts (Phase 1).

Shared types used by the seven Governance Control Plane modules
(GOV-CP-01..07). Lives in ``core.contracts`` so engines and the
dashboard control plane can import these without taking a direct
dependency on ``governance_engine`` (the ``B1`` lint rule forbids
direct cross-engine imports anyway).

Refs:

* ``manifest.md`` §0.5 (GOV-CP), §0.6 (Mode FSM)
* ``docs/directory_tree.md`` §governance_engine/control_plane/
* ``build_plan.md`` Phase 1 (Governance core)

Mode FSM (per the operator's Build Compiler Spec §7):

    SAFE → PAPER → CANARY → LIVE → AUTO

The forward path is a strict ratchet: each step requires policy
approval, risk approval and compliance approval, and the AUTO step
additionally requires an explicit ``operator_authorized`` request.
``LOCKED`` is reachable from ANY state via emergency request, and
``LOCKED → SAFE`` is the only path out of ``LOCKED``. Backward
de-escalation (e.g. ``AUTO → LIVE``, ``LIVE → CANARY``) is always
permitted.

System-mode ``SHADOW`` was demolished by SHADOW-DEMOLITION-02. The
"signals-on, no execution" tier is no longer a distinct mode -- ``PAPER``
emits signals and dispatches via the simulated paper broker, and operators
who want strictly-no-fills set up the paper broker with a refusal hook.
The rank-2 slot is intentionally left vacant so persisted ledger rows
carrying ``mode=2`` remain decodable in archival readers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# System mode (locked by Build Compiler Spec §7)
# ---------------------------------------------------------------------------


class SystemMode(IntEnum):
    """Canonical system mode (Build Compiler Spec §7).

    Ordered for reasoning about ratcheting, but transitions are not
    arbitrary — see :class:`StateTransitionManager` for the legal-edge
    set.
    """

    SAFE = 0
    PAPER = 1
    # rank=2 vacated by SHADOW-DEMOLITION-02 (system-level SHADOW removed).
    # Kept as a gap so persisted ledger rows that historically carried
    # ``mode=2`` decode without renumbering CANARY/LIVE/AUTO.
    CANARY = 3
    LIVE = 4
    AUTO = 5
    LOCKED = 99


# ---------------------------------------------------------------------------
# Operator-facing requests (Dashboard Control Plane → Governance)
# ---------------------------------------------------------------------------


class OperatorAction(StrEnum):
    """Categories of dashboard-originated request.

    Per Build Compiler Spec §6 the dashboard is a Control Plane: it
    *requests* but never *writes*. Every operator action becomes one
    of these strongly typed records routed through
    ``OperatorInterfaceBridge`` (GOV-CP-07).
    """

    REQUEST_MODE = "REQUEST_MODE"
    REQUEST_PLUGIN_LIFECYCLE = "REQUEST_PLUGIN_LIFECYCLE"
    REQUEST_KILL = "REQUEST_KILL"
    REQUEST_UNLOCK = "REQUEST_UNLOCK"
    REQUEST_INTENT = "REQUEST_INTENT"


@dataclass(frozen=True, slots=True)
class OperatorRequest:
    """Inbound dashboard request before Governance processing."""

    ts_ns: int
    requestor: str
    action: OperatorAction
    payload: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mode transitions (proposals + decisions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModeTransitionRequest:
    """Request to move from one ``SystemMode`` to another.

    ``consent`` carries a typed
    :class:`core.contracts.operator_consent.OperatorConsent`
    envelope when the edge requires explicit operator approval per
    Hardening-S1 item 8 (currently ``SAFE → PAPER`` and
    ``LIVE → AUTO``). The legacy ``operator_authorized`` bool is kept
    for backward compatibility on edges that are still gated by
    promotion-gates + operator-authorised flag (PAPER → CANARY,
    CANARY → LIVE).
    """

    ts_ns: int
    requestor: str
    current_mode: SystemMode
    target_mode: SystemMode
    reason: str
    operator_authorized: bool = False
    # ``object | None`` rather than the typed import so this contract
    # module stays a leaf — :mod:`core.contracts.operator_consent`
    # imports :class:`SystemMode` from here. The state transition
    # manager performs the typed downcast at validation time.
    consent: object | None = None


@dataclass(frozen=True, slots=True)
class ModeTransitionDecision:
    """Outcome produced by ``StateTransitionManager.propose``."""

    ts_ns: int
    approved: bool
    prev_mode: SystemMode
    new_mode: SystemMode
    reason: str
    rejection_code: str = ""
    ledger_seq: int = -1


# ---------------------------------------------------------------------------
# System Intent (Phase 6.T1d, INV-38)
# ---------------------------------------------------------------------------
#
# Intent is the operator-set strategic vector ("what should the system want
# to do this week?"). Per v3.1 G1 the *operator* writes intent — never the
# system. Operator proposes an :class:`IntentTransitionRequest` through
# ``OperatorInterfaceBridge`` (GOV-CP-07); the request is gated by
# :class:`PolicyEngine` and committed by
# ``state_transition_manager.propose_intent`` (GOV-CP-03), which is the
# only writer of ``INTENT_TRANSITION`` ledger rows. The current intent is
# projected by the read-only ``core.coherence.system_intent`` module.


class IntentObjective(StrEnum):
    """Top-level mission the operator has selected for the system."""

    RISK_ADJUSTED_GROWTH = "RISK_ADJUSTED_GROWTH"
    ABSOLUTE_RETURN = "ABSOLUTE_RETURN"
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"
    EXPLORATION = "EXPLORATION"


class IntentRiskMode(StrEnum):
    """Operator-set risk posture aligned with the Mode FSM ratchet."""

    DEFENSIVE = "DEFENSIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"


class IntentHorizon(StrEnum):
    """Operator-set planning horizon for capital deployment."""

    INTRADAY = "INTRADAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"


@dataclass(frozen=True, slots=True)
class IntentTransitionRequest:
    """Operator-originated proposal to overwrite the System Intent vector.

    Constructed by ``OperatorInterfaceBridge`` from a
    :class:`OperatorRequest` whose action is ``REQUEST_INTENT`` and
    routed to ``StateTransitionManager.propose_intent``. The ``focus``
    tuple is the ordered list of strategic foci (e.g.
    ``("crypto_microstructure", "fx_carry")``) — order is preserved on
    the ledger.
    """

    ts_ns: int
    requestor: str
    objective: IntentObjective
    risk_mode: IntentRiskMode
    horizon: IntentHorizon
    focus: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class IntentTransitionDecision:
    """Outcome produced by ``StateTransitionManager.propose_intent``."""

    ts_ns: int
    approved: bool
    objective: IntentObjective
    risk_mode: IntentRiskMode
    horizon: IntentHorizon
    focus: tuple[str, ...] = ()
    reason: str = ""
    rejection_code: str = ""
    ledger_seq: int = -1


# ---------------------------------------------------------------------------
# Policy / constraints (GOV-CP-01)
# ---------------------------------------------------------------------------


class ConstraintScope(StrEnum):
    GLOBAL = "GLOBAL"
    MODE = "MODE"
    SYMBOL = "SYMBOL"
    DOMAIN = "DOMAIN"


class ConstraintKind(StrEnum):
    MAX_POSITION_QTY = "MAX_POSITION_QTY"
    MAX_SYMBOL_EXPOSURE = "MAX_SYMBOL_EXPOSURE"
    MAX_DRAWDOWN_PCT = "MAX_DRAWDOWN_PCT"
    REQUIRE_OPERATOR = "REQUIRE_OPERATOR"
    DOMAIN_ISOLATION = "DOMAIN_ISOLATION"


@dataclass(frozen=True, slots=True)
class Constraint:
    """One policy constraint loaded into the PolicyEngine."""

    id: str
    scope: ConstraintScope
    kind: ConstraintKind
    params: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Risk assessment (GOV-CP-02)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Outcome of a risk evaluation against current exposure + limits."""

    ts_ns: int
    symbol: str
    side: str
    qty: float
    approved: bool
    rejection_code: str = ""
    breached_limits: tuple[str, ...] = ()
    exposure_after: float = 0.0


# ---------------------------------------------------------------------------
# Compliance (GOV-CP-06)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """Outcome of compliance validation for an action."""

    passed: bool
    violations: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Final governance decision (GOV-CP-07 → dashboard)
# ---------------------------------------------------------------------------


class DecisionKind(StrEnum):
    MODE_TRANSITION = "MODE_TRANSITION"
    PLUGIN_LIFECYCLE = "PLUGIN_LIFECYCLE"
    KILL = "KILL"
    REJECTED = "REJECTED"
    NOOP = "NOOP"
    INTENT_TRANSITION = "INTENT_TRANSITION"


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """The single result returned to the operator for any request.

    ``ledger_seq`` is the row index produced by
    ``LedgerAuthorityWriter`` — every approved decision lands in the
    authority ledger before this value is set.
    """

    ts_ns: int
    kind: DecisionKind
    approved: bool
    summary: str
    rejection_code: str = ""
    ledger_seq: int = -1


# ---------------------------------------------------------------------------
# Authority ledger row (GOV-CP-05)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One row in the authority ledger.

    The authority ledger is the only ledger that records governance
    decisions; ``LedgerAuthorityWriter`` is the only module allowed to
    append to it (GOV-CP-05). ``hash_chain`` is
    ``sha256(prev_hash || canonical_row_bytes)``.
    """

    seq: int
    ts_ns: int
    kind: str
    payload: Mapping[str, str]
    prev_hash: str
    hash_chain: str


# ---------------------------------------------------------------------------
# AUDIT-P1.2 — StateTransitionProtocol
#
# Structural type that ``system/kill_switch.py`` (SAFE-01) and other
# ``system/`` primitives consume so they do not have to import the
# concrete :class:`governance_engine.control_plane.state_transition_manager
# .StateTransitionManager`. Keeping ``system/`` a true leaf module is a
# lint invariant (B1 + B30): ``system.*`` may import from
# ``core.contracts.*`` and ``core.types.*`` but never from
# ``governance_engine.*``.
#
# The protocol exposes the minimum surface a ``system/`` primitive needs:
# inspect the current mode and propose a transition. The full
# :class:`StateTransitionManager` keeps additional methods
# (``propose_intent``, drift hooks, etc.) that ``system/`` callers do not
# use. The runtime check is keyed on duck-typed attribute presence so
# the existing :class:`StateTransitionManager` satisfies it without any
# inheritance change.
# ---------------------------------------------------------------------------


@runtime_checkable
class StateTransitionProtocol(Protocol):
    """Minimum mode-FSM surface required by ``system/`` leaf primitives."""

    def current_mode(self) -> SystemMode:
        """Return the live :class:`SystemMode` (snapshot, no lock leak)."""

    def propose(
        self, request: ModeTransitionRequest
    ) -> ModeTransitionDecision:
        """Apply the full transition pipeline atomically."""


# ---------------------------------------------------------------------------
# Hazard ingress (Phase 1 backlog item B-01)
# ---------------------------------------------------------------------------
#
# The build_plan.md Phase 1 deliverable list calls out a typed
# ``IGovernanceHazardSink`` Protocol so any sensor or coupling adapter
# (HazardThrottleAdapter, the SCVS source-liveness FSM, the policy-hash
# anchor) can deliver a :class:`HazardEvent` into governance through a
# single typed surface rather than relying solely on the event-bus
# ``process()`` overload. Both shapes are equivalent in semantics --
# governance internally classifies the event and routes through the
# existing ``_handle_hazard`` pipeline -- but the typed Protocol gives
# callers an explicit, lint-checkable contract.


@runtime_checkable
class IGovernanceHazardSink(Protocol):
    """Typed ingress contract for components that deliver hazards to governance.

    Implementations must accept a :class:`HazardEvent` (from
    ``core.contracts.events``) and route it through the same
    classifier+ledger pipeline that ``process()`` uses on the event
    bus. ``HazardSeverity.CRITICAL`` events MUST trigger the
    LOCKED-emergency path; lower severities MUST be appended to the
    audit ledger via GOV-CP-05.

    Backward-compatibility: callers may continue to use the bus
    surface (``GovernanceEngine.process(event)``); this Protocol is
    additive and does not change the existing semantics. It exists so
    type-checked code paths can declare "I deliver hazards to
    governance" without importing the concrete engine class.
    """

    def accept_hazard(self, event: object) -> None:
        """Deliver one ``HazardEvent`` to governance (typed contract).

        ``event`` is typed as :class:`object` here because adding the
        concrete ``HazardEvent`` import would couple this contracts
        module to ``core.contracts.events`` for a single annotation.
        Implementations narrow the type at the call site.
        """


__all__ = [
    "ComplianceReport",
    "Constraint",
    "ConstraintKind",
    "ConstraintScope",
    "DecisionKind",
    "GovernanceDecision",
    "IGovernanceHazardSink",
    "IntentHorizon",
    "IntentObjective",
    "IntentRiskMode",
    "IntentTransitionDecision",
    "IntentTransitionRequest",
    "LedgerEntry",
    "ModeTransitionDecision",
    "ModeTransitionRequest",
    "OperatorAction",
    "OperatorRequest",
    "RiskAssessment",
    "StateTransitionProtocol",
    "SystemMode",
]
