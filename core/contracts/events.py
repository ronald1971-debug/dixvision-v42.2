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
    # ------------------------------------------------------------------
    # v3.3 T1b — INV-52 shadow meta-controller divergence record.
    # The shadow policy never reaches PolicyEngine; it only emits this
    # event so the offline ``learning_engine`` can compare alternative
    # decisions against the primary policy's actual outcome.
    # ------------------------------------------------------------------
    META_DIVERGENCE = "META_DIVERGENCE"
    # ------------------------------------------------------------------
    # v3.1 H5 + v3.3 J3 — reward shaping per-component breakdown
    # (Tier 1.5). Emitted by
    # ``learning_engine.lanes.reward_shaping`` per realised trade so
    # the offline calibrator can attribute reward drift to individual
    # components (consensus / strength / coverage / sizing rationale /
    # latency / slippage / fallback). The raw PnL is preserved
    # alongside the shaped reward to keep shaping invertible /
    # auditable per INV-47.
    # ------------------------------------------------------------------
    REWARD_BREAKDOWN = "REWARD_BREAKDOWN"
    # ------------------------------------------------------------------
    # v3.3 J3 — per-tick meta-controller audit record (Phase 6.T1c).
    # Emitted by
    # ``intelligence_engine.meta_controller.runtime_adapter`` next to
    # BELIEF_STATE_SNAPSHOT / PRESSURE_VECTOR_SNAPSHOT every hot-path
    # tick so the offline calibrator can attribute drift to the
    # confidence / sizing components and the final decision.
    # ------------------------------------------------------------------
    META_AUDIT = "META_AUDIT"
    # ------------------------------------------------------------------
    # v3.4 Wave 2 — INV-53 reader-side calibration report (Phase
    # 6.T1c reader). Emitted by
    # ``learning_engine.calibration.coherence_calibrator`` once per
    # closed window over BELIEF_STATE_SNAPSHOT /
    # PRESSURE_VECTOR_SNAPSHOT / META_AUDIT / REWARD_BREAKDOWN ledger
    # rows. Read-only by Governance — never gates execution; only
    # surfaces drift between the runtime's projected lenses and the
    # realised outcome distribution.
    # ------------------------------------------------------------------
    CALIBRATION_REPORT = "CALIBRATION_REPORT"
    # ------------------------------------------------------------------
    # v3.5 SCVS Phase 2 — runtime source liveness transitions emitted by
    # ``system_engine.scvs.source_manager``. Pure projection of a
    # caller-supplied ``now_ns`` against the per-source heartbeat memo;
    # one row per status transition (UNKNOWN→LIVE, LIVE→STALE,
    # STALE→LIVE). Critical-source STALE transitions additionally emit
    # an ``HAZ-13`` hazard for governance escalation (SCVS-06).
    # ------------------------------------------------------------------
    SOURCE_HEARTBEAT = "SOURCE_HEARTBEAT"
    SOURCE_STALE = "SOURCE_STALE"
    SOURCE_RECOVERED = "SOURCE_RECOVERED"
    # ------------------------------------------------------------------
    # v3.5 SCVS Phase 3 — silent-fallback audit (rule SCVS-10). When a
    # registered source fails and the engine activates a fallback, the
    # engine MUST emit this event so the governance ledger has an
    # explicit record of the substitution. "Silent fallback" — i.e.
    # swapping data sources without recording it — is the precise thing
    # SCVS-10 forbids.
    # ------------------------------------------------------------------
    SOURCE_FALLBACK_ACTIVATED = "SOURCE_FALLBACK_ACTIVATED"
    # ------------------------------------------------------------------
    # v3.6 BEHAVIOR-P4 — per-decision audit record. Emitted by
    # ``core.coherence.decision_trace.as_system_event`` once per
    # decision so the offline calibrator and the operator dashboard's
    # Decision-Trace widget (DASH-04) can reconstruct *why* each
    # decision happened (confidence breakdown, active hazards,
    # throttle applied, execution outcome). The trace_id is a
    # deterministic hash of (symbol, ts_ns, plugin_chain) so replays
    # produce identical ledger rows (INV-15).
    # ------------------------------------------------------------------
    DECISION_TRACE = "DECISION_TRACE"
    # ------------------------------------------------------------------
    # v3.6 BEHAVIOR-P5 — patch pipeline ledger surface. Emitted by
    # ``evolution_engine.patch_pipeline.events`` when a structural
    # mutation enters / advances through / leaves the Phase 4 patch
    # pipeline. The dashboard's Strategy-Lifecycle widget (DASH-SLP-01)
    # and the Indira/Dyon chat widgets read these to render reviewable
    # cards. Distinct from UPDATE_PROPOSED, which is the non-structural
    # parameter-update surface.
    #
    #  * ``PATCH_PROPOSED``       — a new ``PatchProposal`` was
    #    registered with the bridge (Stage = PROPOSED).
    #  * ``PATCH_STAGE_VERDICT``  — one stage finished and recorded a
    #    verdict (Stage ∈ {SANDBOX, STATIC_ANALYSIS, BACKTEST, SHADOW,
    #    CANARY}).
    #  * ``PATCH_DECISION``       — terminal decision driven by the
    #    bridge (APPROVED / REJECTED / ROLLED_BACK).
    #
    # All three events have deterministic, key-sorted JSON payloads so
    # replays produce byte-identical ledger rows (INV-15, INV-66).
    # ------------------------------------------------------------------
    PATCH_PROPOSED = "PATCH_PROPOSED"
    PATCH_STAGE_VERDICT = "PATCH_STAGE_VERDICT"
    PATCH_DECISION = "PATCH_DECISION"


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
