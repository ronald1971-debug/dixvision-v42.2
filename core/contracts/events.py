"""Canonical 4-event types (Phase E0).

The only events that cross engine boundaries are the four typed events defined
here (INV-08). Cross-engine direct imports are forbidden — engines communicate
exclusively through these dataclasses, which mirror the Protobuf definitions
in ``contracts/events.proto``.

Refs:
- ``manifest.md`` §0.4 (CORE TRUTH), §1 (invariants)
- ``docs/total_recall_index.md`` §13 (EVT-01..04)
- INV-08 (only typed events cross domain), INV-11 (no direct cross-engine
  calls), INV-15 (replay determinism)

The Python dataclasses are the source of truth at Phase E0 (Lane A,
Python-first). The ``contracts/events.proto`` file mirrors them exactly so
that future polyglot ports (Phase E9) are mechanical translations.

Frozen + slotted dataclasses are used so that events are immutable hashable
values. Equality is structural, which TEST-01 relies on for replay parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class EventKind(StrEnum):
    """Discriminator for the four canonical event kinds (EVT-01..04)."""

    SIGNAL = "SIGNAL_EVENT"
    EXECUTION = "EXECUTION_EVENT"
    SYSTEM = "SYSTEM_EVENT"
    HAZARD = "HAZARD_EVENT"


# ---------------------------------------------------------------------------
# Shared sub-types
# ---------------------------------------------------------------------------


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class HazardSeverity(StrEnum):
    """Hazard severity classes (HAZ-01..12 envelope)."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SystemEventKind(StrEnum):
    """Sub-type discriminator for :class:`SystemEvent`."""

    HEARTBEAT = "HEARTBEAT"
    HEALTH_REPORT = "HEALTH_REPORT"
    PLUGIN_LIFECYCLE = "PLUGIN_LIFECYCLE"  # ties into PLUGIN-ACT-06
    UPDATE_PROPOSED = "UPDATE_PROPOSED"  # offline engines emit this
    LEDGER_COMMIT = "LEDGER_COMMIT"
    # ------------------------------------------------------------------
    # v3.3 T1a — coherence projection snapshots (INV-53 calibration hook).
    # Read-only by definition; emitted by ``core.coherence`` so the
    # offline ``learning_engine.calibration.coherence_calibrator`` has a
    # ledgered window to read against realised outcomes.
    # ------------------------------------------------------------------
    BELIEF_STATE_SNAPSHOT = "BELIEF_STATE_SNAPSHOT"
    PRESSURE_VECTOR_SNAPSHOT = "PRESSURE_VECTOR_SNAPSHOT"


# ---------------------------------------------------------------------------
# EVT-01 SIGNAL_EVENT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """EVT-01. Intelligence engine output.

    Attributes:
        kind: Always :attr:`EventKind.SIGNAL`.
        ts_ns: Monotonic timestamp in nanoseconds (TimeAuthority, T0-04).
        symbol: Instrument identifier (e.g. ``"EURUSD"``, ``"BTCUSDT"``).
        side: Direction.
        confidence: ``[0.0, 1.0]`` confidence band.
        plugin_chain: Tuple of plugin names that contributed to this signal,
            in order. Used for DecisionTrace (DASH-04).
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    symbol: str
    side: Side
    confidence: float
    plugin_chain: tuple[str, ...] = ()
    meta: Mapping[str, str] = field(default_factory=dict)
    kind: EventKind = EventKind.SIGNAL


# ---------------------------------------------------------------------------
# EVT-02 EXECUTION_EVENT
# ---------------------------------------------------------------------------


class ExecutionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """EVT-02. Execution engine output."""

    ts_ns: int
    symbol: str
    side: Side
    qty: float
    price: float
    status: ExecutionStatus
    venue: str = ""
    order_id: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)
    kind: EventKind = EventKind.EXECUTION


# ---------------------------------------------------------------------------
# EVT-03 SYSTEM_EVENT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemEvent:
    """EVT-03. System / Governance / offline-engine coordination event.

    Sub-typed via :attr:`sub_kind`. Includes ``UPDATE_PROPOSED`` (offline
    engines → Governance) and ``PLUGIN_LIFECYCLE`` (PLUGIN-ACT-06).
    """

    ts_ns: int
    sub_kind: SystemEventKind
    source: str  # engine name, e.g. "system", "learning"
    payload: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, str] = field(default_factory=dict)
    kind: EventKind = EventKind.SYSTEM


# ---------------------------------------------------------------------------
# EVT-04 HAZARD_EVENT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HazardEvent:
    """EVT-04. The only signal that crosses the Indira ↔ Dyon domain
    boundary (CORE TRUTH).

    Emitted by ``system_engine`` (Dyon domain), consumed by
    ``governance_engine`` (sole authority).
    """

    ts_ns: int
    code: str  # HAZ-01..12 etc.
    severity: HazardSeverity
    source: str
    detail: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)
    kind: EventKind = EventKind.HAZARD


Event = SignalEvent | ExecutionEvent | SystemEvent | HazardEvent


__all__ = [
    "Event",
    "EventKind",
    "ExecutionEvent",
    "ExecutionStatus",
    "HazardEvent",
    "HazardSeverity",
    "Side",
    "SignalEvent",
    "SystemEvent",
    "SystemEventKind",
]
