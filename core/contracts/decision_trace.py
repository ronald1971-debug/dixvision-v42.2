"""Decision-trace contracts shared across engine boundaries (BEHAVIOR-P4).

A :class:`DecisionTrace` is the per-decision structured record that
captures **why** a trade happened — the confidence breakdown, the
active hazards at decision time, the throttle decision applied to the
risk snapshot, and (when known) the resulting execution outcome.

Pure data types — no engine logic, no clock reads, no I/O. Frozen,
slotted, hashable. May be imported from any engine package
(``ALLOWED_SHARED_PREFIXES`` in ``authority_lint``).

The trace is assembled by :mod:`core.coherence.decision_trace` from
inputs supplied by the caller and serialised to a
:class:`~core.contracts.events.SystemEvent` (``DECISION_TRACE``
sub-kind) for the audit ledger. Other engines never construct
:class:`DecisionTrace` directly — they project their own outputs into
the contract types here and hand them to the builder.

Refs:
- BEHAVIOR-P4 (priority 4 from the v3.5 critique)
- INV-15 (replay determinism — trace_id is a deterministic hash)
- INV-65 (decision-trace contract is pure / no I/O)
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.events import ExecutionStatus, HazardSeverity, Side
from core.contracts.signal_trust import SignalTrust

DECISION_TRACE_VERSION: int = 3


@dataclass(frozen=True, slots=True)
class ConfidenceContribution:
    """One component of a decision's confidence breakdown.

    The trace ``final_confidence`` is the sum of ``weighted`` over all
    contributions when the breakdown is exhaustive; partial breakdowns
    are allowed (e.g. when only the meta-controller's J3 components are
    captured) and are checked monotonically rather than for equality.

    Attributes:
        name: Component name (e.g. ``"consensus"``, ``"strength"``,
            ``"coverage"``, ``"sentiment"``, ``"microstructure"``).
        value: Raw component value in ``[0.0, 1.0]``.
        weight: Component weight in ``[0.0, 1.0]``.
        weighted: ``value * weight`` (precomputed for ledger
            serialisation; the builder validates the relation).
    """

    name: str
    value: float
    weight: float
    weighted: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConfidenceContribution.name must be non-empty")
        _check_unit("ConfidenceContribution.value", self.value)
        _check_unit("ConfidenceContribution.weight", self.weight)
        if self.weighted < 0.0 or self.weighted > 1.0:
            raise ValueError(
                f"ConfidenceContribution.weighted must be in [0.0, 1.0]; got {self.weighted}"
            )
        expected = self.value * self.weight
        if abs(self.weighted - expected) > 1e-9:
            raise ValueError(
                "ConfidenceContribution.weighted must equal value * weight; "
                f"got weighted={self.weighted} expected={expected}"
            )


@dataclass(frozen=True, slots=True)
class PressureSummary:
    """A frozen 5-axis projection of :class:`PressureVector`.

    Captured at decision time so the offline calibrator can correlate
    decisions with the runtime stress lenses without a separate
    PRESSURE_VECTOR_SNAPSHOT lookup.
    """

    perf: float
    risk: float
    drift: float
    latency: float
    uncertainty: float

    def __post_init__(self) -> None:
        for name, axis in (
            ("perf", self.perf),
            ("risk", self.risk),
            ("drift", self.drift),
            ("latency", self.latency),
            ("uncertainty", self.uncertainty),
        ):
            _check_unit(f"PressureSummary.{name}", axis)


@dataclass(frozen=True, slots=True)
class HazardInfluence:
    """One hazard observation that was active at decision time.

    Captured from :class:`system_engine.coupling.HazardObserver`'s
    active set; the caller projects each active observation into this
    record before handing it to the trace builder (B1: the builder
    must not import from ``system_engine``).
    """

    code: str
    severity: HazardSeverity
    source: str
    ts_ns: int

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("HazardInfluence.code must be non-empty")
        if not self.source:
            raise ValueError("HazardInfluence.source must be non-empty")
        if self.ts_ns < 0:
            raise ValueError(f"HazardInfluence.ts_ns must be non-negative; got {self.ts_ns}")


@dataclass(frozen=True, slots=True)
class ThrottleInfluence:
    """The throttle decision applied to the risk snapshot at decision time.

    Mirrors :class:`system_engine.coupling.ThrottleDecision` without
    importing from ``system_engine`` (B1). Callers project the live
    decision into this contract type before invoking the builder.
    """

    block: bool
    qty_multiplier: float
    confidence_floor: float
    contributing_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_unit("ThrottleInfluence.qty_multiplier", self.qty_multiplier)
        _check_unit("ThrottleInfluence.confidence_floor", self.confidence_floor)
        for code in self.contributing_codes:
            if not code:
                raise ValueError("ThrottleInfluence.contributing_codes entries must be non-empty")


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """The execution result observed for the trace's signal, when known.

    Mirrors the relevant subset of :class:`ExecutionEvent` so the trace
    is self-contained; callers attach this once the signal's
    corresponding execution event has been observed (or the signal has
    been rejected outright).
    """

    status: ExecutionStatus
    qty: float
    price: float
    venue: str = ""
    order_id: str = ""

    def __post_init__(self) -> None:
        if self.qty < 0.0:
            raise ValueError(f"ExecutionOutcome.qty must be non-negative; got {self.qty}")
        if self.price < 0.0:
            raise ValueError(f"ExecutionOutcome.price must be non-negative; got {self.price}")


@dataclass(frozen=True, slots=True)
class BeliefReference:
    """A philosophy-belief reference captured in the Why layer.

    Anchors a decision to a specific belief on the trader's
    :class:`PhilosophyProfile` (e.g. ``("trend_following", 0.8)``) so
    the offline learning calibrator and the Why widget can answer
    *which belief was acting* on this decision, not just *what the
    score was*. Beliefs below the composition engine's
    ``BELIEF_THRESHOLD`` are noise and should not appear here.

    Attributes:
        name: Belief key as it appears on
            :attr:`PhilosophyProfile.belief_system`.
        strength: Belief strength in ``[0.0, 1.0]``.
    """

    name: str
    strength: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BeliefReference.name must be non-empty")
        _check_unit("BeliefReference.strength", self.strength)


@dataclass(frozen=True, slots=True)
class WhyLayer:
    """Structured *why* references for a decision (Wave-04 PR-5).

    A pointer-only projection of the strategy components that produced
    a decision: the philosophy profile, the active beliefs, the
    decomposed components (entry/exit/risk/timeframe/market) and, when
    the decision came from the composition engine, the parent
    :class:`ComposedStrategy`. Every field is optional so traces
    emitted before Wave-04 components were wired (i.e. legacy
    monolithic strategies) still round-trip cleanly.

    The Why layer never embeds the components themselves — only their
    ``component_id`` strings — so a decision trace stays small and
    replay-stable even if the underlying registry rows are revised.

    Attributes:
        philosophy_id: Trader id whose ``PhilosophyProfile`` informed
            this decision.
        beliefs: Beliefs above ``BELIEF_THRESHOLD`` that were active.
        entry_logic_id: ``EntryLogic.component_id`` if known.
        exit_logic_id: ``ExitLogic.component_id`` if known.
        risk_model_id: ``RiskModel.component_id`` if known.
        timeframe_id: ``Timeframe.component_id`` if known.
        market_condition_id: ``MarketCondition.component_id`` if known.
        composition_id: ``ComposedStrategy.composition_id`` when this
            decision was sourced from the composition engine; ``None``
            for legacy monolithic plugins.
        notes: Optional human-readable annotations (key → short text).
            Sorted on serialise so byte-identical replay survives dict
            ordering.
    """

    philosophy_id: str | None = None
    beliefs: tuple[BeliefReference, ...] = ()
    entry_logic_id: str | None = None
    exit_logic_id: str | None = None
    risk_model_id: str | None = None
    timeframe_id: str | None = None
    market_condition_id: str | None = None
    composition_id: str | None = None
    notes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for fld in (
            "philosophy_id",
            "entry_logic_id",
            "exit_logic_id",
            "risk_model_id",
            "timeframe_id",
            "market_condition_id",
            "composition_id",
        ):
            value = getattr(self, fld)
            if value is not None and not value:
                raise ValueError(
                    f"WhyLayer.{fld} must be non-empty when set; use None to mean 'unknown'"
                )
        seen_belief_names: set[str] = set()
        for belief in self.beliefs:
            if belief.name in seen_belief_names:
                raise ValueError(
                    "WhyLayer.beliefs must not contain duplicate names; "
                    f"got duplicate {belief.name!r}"
                )
            seen_belief_names.add(belief.name)
        seen_note_keys: set[str] = set()
        for key, text in self.notes:
            if not key:
                raise ValueError("WhyLayer.notes keys must be non-empty")
            if key in seen_note_keys:
                raise ValueError(
                    f"WhyLayer.notes must not contain duplicate keys; got duplicate {key!r}"
                )
            if not isinstance(text, str):
                raise ValueError(
                    "WhyLayer.notes values must be strings; "
                    f"got {type(text).__name__} for key {key!r}"
                )
            seen_note_keys.add(key)


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Per-decision structured record (BEHAVIOR-P4).

    Anchored by ``trace_id`` (deterministic hash of
    ``(symbol, ts_ns, plugin_chain)`` — see
    :func:`core.coherence.decision_trace.compute_trace_id`). All
    optional fields default to ``None`` / empty so a trace can be
    emitted with whatever subset of the lenses the caller has at
    decision time; downstream readers (Decision-Trace widget,
    learning calibrator) tolerate sparse traces.

    Attributes:
        version: Schema version (``DECISION_TRACE_VERSION``).
        trace_id: Stable 16-hex-char identifier.
        ts_ns: Timestamp of the originating signal.
        symbol: Instrument identifier.
        side: Decided side.
        final_confidence: Net confidence after all contributions and
            (if applied) the throttle's confidence floor.
        plugin_chain: Plugin names that contributed, in order.
        regime: Committed regime label (e.g. ``"TREND_UP"``), if known.
        pressure_summary: 5-axis pressure projection, if known.
        safety_modifier: Composite safety modifier in ``[0.0, 1.0]``,
            if known.
        confidence_breakdown: Per-component contributions (may be empty
            when the caller has no breakdown to project).
        active_hazards: Hazards active at decision time, in observation
            order.
        throttle_applied: Throttle decision applied to the risk
            snapshot, if any.
        execution_outcome: Execution result, if known.
        why: Structured Why-layer references back to the strategy
            components that produced this decision (Wave-04 PR-5);
            ``None`` for traces emitted by pre-Wave-04 monolithic
            plugins.
    """

    version: int
    trace_id: str
    ts_ns: int
    symbol: str
    side: Side
    final_confidence: float
    plugin_chain: tuple[str, ...]
    regime: str | None
    pressure_summary: PressureSummary | None
    safety_modifier: float | None
    confidence_breakdown: tuple[ConfidenceContribution, ...]
    active_hazards: tuple[HazardInfluence, ...]
    throttle_applied: ThrottleInfluence | None
    execution_outcome: ExecutionOutcome | None
    why: WhyLayer | None = None
    # Paper-S1 — provenance triplet projected from the SignalEvent +
    # SCVS validator. Optional because pre-Paper-S1 traces never had
    # them; the audit-ledger reader must tolerate ``None``.
    signal_trust: SignalTrust | None = None
    signal_source: str | None = None
    validation_score: float | None = None
    # Paper-S7 — confidence-cap audit triplet. ``original_confidence``
    # is the producer-emitted confidence BEFORE the Paper-S5/S6 cap was
    # applied at the harness gate; ``confidence_cap_applied`` is
    # ``True`` iff the cap actually clamped the value down (i.e. the
    # original strictly exceeded the cap); ``confidence_cap_value`` is
    # the cap that was used (``None`` for INTERNAL signals where no
    # cap is applied). All three default to ``None`` / ``False`` so a
    # pre-Paper-S7 trace can still round-trip through the audit ledger
    # and the audit reader can detect the absence of the lens by
    # checking ``original_confidence is None``.
    original_confidence: float | None = None
    confidence_cap_applied: bool = False
    confidence_cap_value: float | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"DecisionTrace.version must be >= 1; got {self.version}")
        if not self.trace_id:
            raise ValueError("DecisionTrace.trace_id must be non-empty")
        if self.ts_ns < 0:
            raise ValueError(f"DecisionTrace.ts_ns must be non-negative; got {self.ts_ns}")
        if not self.symbol:
            raise ValueError("DecisionTrace.symbol must be non-empty")
        _check_unit("DecisionTrace.final_confidence", self.final_confidence)
        if self.safety_modifier is not None:
            _check_unit("DecisionTrace.safety_modifier", self.safety_modifier)
        if self.validation_score is not None:
            _check_unit("DecisionTrace.validation_score", self.validation_score)
        if self.signal_source is not None and not self.signal_source:
            raise ValueError(
                "DecisionTrace.signal_source must be either None or a non-empty string"
            )
        if self.original_confidence is not None:
            _check_unit("DecisionTrace.original_confidence", self.original_confidence)
            # The cap is monotone — the post-cap value cannot exceed
            # the pre-cap value. Catching this at construction prevents
            # a bad upstream projection from forging a trace that
            # claims the cap *amplified* the signal.
            if self.final_confidence - self.original_confidence > 1e-9:
                raise ValueError(
                    "DecisionTrace.final_confidence must not exceed "
                    f"original_confidence; got final={self.final_confidence}, "
                    f"original={self.original_confidence}"
                )
        if self.confidence_cap_value is not None:
            _check_unit(
                "DecisionTrace.confidence_cap_value", self.confidence_cap_value
            )
        if self.confidence_cap_applied:
            # ``applied=True`` is only meaningful when both the
            # pre-cap value and the cap itself are recorded; otherwise
            # the audit cannot answer "by how much was it clamped?".
            if self.original_confidence is None:
                raise ValueError(
                    "DecisionTrace.confidence_cap_applied is True but "
                    "original_confidence is None"
                )
            if self.confidence_cap_value is None:
                raise ValueError(
                    "DecisionTrace.confidence_cap_applied is True but "
                    "confidence_cap_value is None"
                )


def _check_unit(name: str, value: float) -> None:
    """Raise ``ValueError`` if *value* is not in ``[0.0, 1.0]``."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0]; got {value}")


__all__ = [
    "DECISION_TRACE_VERSION",
    "BeliefReference",
    "ConfidenceContribution",
    "DecisionTrace",
    "ExecutionOutcome",
    "HazardInfluence",
    "PressureSummary",
    "ThrottleInfluence",
    "WhyLayer",
]
