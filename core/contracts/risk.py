"""
core/contracts/risk.py
DIX VISION v42.2 — Risk Protocol Contract (SYSTEM_HAZARD schema)

Phase 0 Build Plan §1.2. The governance/risk contract has two public
surfaces:

  1. ``IRiskCache`` — what the risk constants look like to callers.
     Matches ``system.fast_risk_cache.RiskConstraints`` and the JSON
     emitted by ``cockpit.app:/api/risk``.
  2. ``ISystemHazardEvent`` — the wire format every hazard emission must
     honour. Anything the ``HazardBus`` delivers to ``GovernanceKernel``
     must satisfy this shape; CI can assert ``isinstance(obj,
     ISystemHazardEvent)`` at runtime thanks to ``@runtime_checkable``.
  3. ``IHazardEmitter`` — the Dyon-only producer interface.

Breaking either contract is a sandbox-blocked change.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRiskConstraints(Protocol):
    """The precomputed risk constants snapshot the hot path reads in O(1)."""

    max_order_size_usd: float
    max_position_pct: float
    circuit_breaker_drawdown: float    # <= 0.04 per safety_axiom S1
    circuit_breaker_loss_pct: float    # <= 0.01 per safety_axiom S2
    trading_allowed: bool
    safe_mode: bool
    last_updated_utc: str


@runtime_checkable
class IRiskCache(Protocol):
    """Governance-writable, Indira-readable cache of :class:`IRiskConstraints`.

    Implemented by :class:`system.fast_risk_cache.FastRiskCache`. Governance
    is the sole writer (``halt_trading`` / ``enter_safe_mode`` / ``update``);
    Indira is the sole fast-path reader (``get``).
    """

    def get(self) -> IRiskConstraints: ...
    def halt_trading(self, reason: str = ...) -> IRiskConstraints: ...
    def enter_safe_mode(self) -> IRiskConstraints: ...


@runtime_checkable
class ISystemHazardEvent(Protocol):
    """
    SYSTEM_HAZARD wire schema (build plan §1.2).

    Every object Dyon pushes onto the hazard bus MUST expose these
    attributes; governance reads nothing else. Extra attributes are
    allowed but ignored by the contract.
    """

    hazard_type: Any          # HazardType enum value (str-backed)
    severity: Any             # HazardSeverity enum value (str-backed)
    source: str               # Dyon sub-component that detected it
    details: dict[str, Any]   # Free-form context
    timestamp_utc: str        # ISO-8601 UTC, set at construction
    sequence: int             # Monotonic per-process sequence
    resolved: bool            # False until governance observes + acts


@runtime_checkable
class IHazardEmitter(Protocol):
    """Producer contract — implemented once in execution/hazard/event_emitter.py.

    Governance must NOT implement or subclass this; Indira must NOT
    implement or subclass this. Only Dyon (system_monitor.*) holds this
    authority — hazard_axioms.lean H2.
    """

    def emit(self,
             hazard_type: Any,
             severity: Any,
             details: dict[str, Any] | None = None) -> ISystemHazardEvent: ...


@runtime_checkable
class IGovernanceHazardSink(Protocol):
    """Consumer contract — implemented once in governance.kernel.GovernanceKernel.

    Per hazard_axioms.lean H3, only Governance interprets hazards into
    mode changes. Anything else wiring itself as a consumer is a
    violation.
    """

    def _on_hazard(self, event: ISystemHazardEvent) -> None: ...
