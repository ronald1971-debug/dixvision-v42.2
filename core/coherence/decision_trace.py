"""Decision-trace builder + serializer (BEHAVIOR-P4).

Pure, offline projection layer that assembles a
:class:`~core.contracts.decision_trace.DecisionTrace` from inputs the
caller has at decision time, and serialises it into a canonical
:class:`~core.contracts.events.SystemEvent` (``DECISION_TRACE``
sub-kind) for the audit ledger.

This module is the **only** place the trace is constructed; runtime
engines must project their own outputs into the contract types under
:mod:`core.contracts.decision_trace` before calling
:func:`build_decision_trace`. That keeps the builder dependency-free
of any specific engine (B1) and lets it live alongside the existing
coherence projections (BeliefState, PressureVector).

Refs:
- BEHAVIOR-P4 (priority 4 from the v3.5 critique)
- INV-15 (replay determinism — :func:`compute_trace_id` is a pure
  hash; :func:`build_decision_trace` performs no clock reads / PRNG
  / I/O)
- INV-65 (decision trace is pure; same inputs → same trace bytes)
"""

from __future__ import annotations

import hashlib
import json

from core.contracts.decision_trace import (
    DECISION_TRACE_VERSION,
    BeliefReference,
    ConfidenceContribution,
    DecisionTrace,
    ExecutionOutcome,
    HazardInfluence,
    PressureSummary,
    ThrottleInfluence,
    WhyLayer,
)
from core.contracts.events import (
    ExecutionStatus,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from core.contracts.signal_trust import SignalTrust

DECISION_TRACE_BUILDER_SOURCE = "core.coherence.decision_trace"
_TRACE_ID_LEN: int = 16


def compute_trace_id(
    *,
    symbol: str,
    ts_ns: int,
    plugin_chain: tuple[str, ...],
) -> str:
    """Return a stable 16-hex-char identifier for a decision trace.

    Deterministic: same inputs → same id, byte-identical across
    replays (INV-15). The hash includes ``plugin_chain`` so two
    decisions for the same ``(symbol, ts_ns)`` from different plugin
    paths get distinct ids.
    """
    if not symbol:
        raise ValueError("compute_trace_id: symbol must be non-empty")
    if ts_ns < 0:
        raise ValueError(
            f"compute_trace_id: ts_ns must be non-negative; got {ts_ns}"
        )
    # Use ASCII control bytes as separators so plugin names containing
    # any printable character (including '|' or ':') cannot collide with
    # the field delimiter:
    #   * \x1f (Unit Separator) between (symbol, ts_ns, chain)
    #   * \x00 (NUL) between plugin_chain entries
    # This means ("a", "b") and ("a|b",) hash to distinct ids, and
    # () and ("",) hash to distinct ids (the latter contains an explicit
    # empty entry).
    chain = "\x00".join(plugin_chain)
    payload = (
        f"{symbol}\x1f{ts_ns}\x1f{len(plugin_chain)}\x1f{chain}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:_TRACE_ID_LEN]


def build_decision_trace(
    *,
    signal: SignalEvent,
    confidence_breakdown: tuple[ConfidenceContribution, ...] = (),
    regime: str | None = None,
    pressure_summary: PressureSummary | None = None,
    safety_modifier: float | None = None,
    active_hazards: tuple[HazardInfluence, ...] = (),
    throttle_applied: ThrottleInfluence | None = None,
    execution_outcome: ExecutionOutcome | None = None,
    why: WhyLayer | None = None,
    validation_score: float | None = None,
) -> DecisionTrace:
    """Assemble a :class:`DecisionTrace` from the supplied inputs.

    Pure: no clock reads, no PRNG, no I/O. Only validation logic.

    The trace's ``final_confidence`` is taken from the
    :class:`SignalEvent` itself — the caller is expected to have
    already applied the throttle's ``confidence_floor`` if relevant
    (the floor is captured separately in ``throttle_applied`` so the
    Decision-Trace widget can show *why* the floor moved). The
    builder additionally enforces a monotonic check: when an explicit
    ``confidence_breakdown`` is supplied, the sum of weighted
    contributions must not exceed ``final_confidence`` by more than
    ``1e-6`` — partial breakdowns (sum < final) are allowed.
    """
    plugin_chain = tuple(signal.plugin_chain)
    trace_id = compute_trace_id(
        symbol=signal.symbol,
        ts_ns=signal.ts_ns,
        plugin_chain=plugin_chain,
    )

    if confidence_breakdown:
        total = sum(c.weighted for c in confidence_breakdown)
        if total - signal.confidence > 1e-6:
            raise ValueError(
                "build_decision_trace: confidence_breakdown weighted-sum "
                f"({total}) exceeds signal.confidence ({signal.confidence}); "
                "breakdowns may be partial but never over-state"
            )

    return DecisionTrace(
        version=DECISION_TRACE_VERSION,
        trace_id=trace_id,
        ts_ns=signal.ts_ns,
        symbol=signal.symbol,
        side=signal.side,
        final_confidence=signal.confidence,
        plugin_chain=plugin_chain,
        regime=regime,
        pressure_summary=pressure_summary,
        safety_modifier=safety_modifier,
        confidence_breakdown=tuple(confidence_breakdown),
        active_hazards=tuple(active_hazards),
        throttle_applied=throttle_applied,
        execution_outcome=execution_outcome,
        why=why,
        signal_trust=signal.signal_trust,
        signal_source=signal.signal_source or None,
        validation_score=validation_score,
    )


def as_system_event(
    trace: DecisionTrace,
    *,
    source: str = DECISION_TRACE_BUILDER_SOURCE,
) -> SystemEvent:
    """Serialise *trace* into a canonical ``DECISION_TRACE`` SystemEvent.

    The payload is a single ``"trace"`` key whose value is a
    JSON-serialised, key-sorted projection of the trace's fields.
    Sorting + JSON serialisation make the event byte-identical across
    replays of the same input (INV-15).
    """
    if not source:
        raise ValueError("as_system_event: source must be non-empty")

    body: dict[str, object] = {
        "version": trace.version,
        "trace_id": trace.trace_id,
        "ts_ns": trace.ts_ns,
        "symbol": trace.symbol,
        "side": trace.side.value,
        "final_confidence": trace.final_confidence,
        "plugin_chain": list(trace.plugin_chain),
        "regime": trace.regime,
        "safety_modifier": trace.safety_modifier,
        "pressure_summary": _pressure_to_json(trace.pressure_summary),
        "confidence_breakdown": [
            _contribution_to_json(c) for c in trace.confidence_breakdown
        ],
        "active_hazards": [_hazard_to_json(h) for h in trace.active_hazards],
        "throttle_applied": _throttle_to_json(trace.throttle_applied),
        "execution_outcome": _execution_to_json(trace.execution_outcome),
        "why": _why_to_json(trace.why),
        "signal_trust": (
            trace.signal_trust.value if trace.signal_trust is not None else None
        ),
        "signal_source": trace.signal_source,
        "validation_score": trace.validation_score,
    }
    payload = {
        "trace": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    return SystemEvent(
        ts_ns=trace.ts_ns,
        sub_kind=SystemEventKind.DECISION_TRACE,
        source=source,
        payload=payload,
    )


def trace_from_system_event(event: SystemEvent) -> DecisionTrace:
    """Reverse of :func:`as_system_event` for tests and audit replay.

    Strict — raises :class:`ValueError` on any structural mismatch so
    a malformed ledger row never silently round-trips into a
    half-populated :class:`DecisionTrace`.
    """
    if event.sub_kind is not SystemEventKind.DECISION_TRACE:
        raise ValueError(
            "trace_from_system_event: event must be a DECISION_TRACE "
            f"SystemEvent; got {event.sub_kind}"
        )
    raw = event.payload.get("trace")
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            "trace_from_system_event: payload must contain a 'trace' string"
        )
    body = json.loads(raw)
    return DecisionTrace(
        version=int(body["version"]),
        trace_id=str(body["trace_id"]),
        ts_ns=int(body["ts_ns"]),
        symbol=str(body["symbol"]),
        side=Side(body["side"]),
        final_confidence=float(body["final_confidence"]),
        plugin_chain=tuple(body["plugin_chain"]),
        regime=body["regime"],
        pressure_summary=_pressure_from_json(body["pressure_summary"]),
        safety_modifier=body["safety_modifier"],
        confidence_breakdown=tuple(
            _contribution_from_json(c) for c in body["confidence_breakdown"]
        ),
        active_hazards=tuple(
            _hazard_from_json(h) for h in body["active_hazards"]
        ),
        throttle_applied=_throttle_from_json(body["throttle_applied"]),
        execution_outcome=_execution_from_json(body["execution_outcome"]),
        why=_why_from_json(body.get("why")),
        signal_trust=(
            SignalTrust(body["signal_trust"])
            if body.get("signal_trust") is not None
            else None
        ),
        signal_source=body.get("signal_source"),
        validation_score=body.get("validation_score"),
    )


# ---------------------------------------------------------------------------
# JSON projection helpers (kept module-private; symmetric pairs).
# ---------------------------------------------------------------------------


def _pressure_to_json(p: PressureSummary | None) -> dict[str, float] | None:
    if p is None:
        return None
    return {
        "perf": p.perf,
        "risk": p.risk,
        "drift": p.drift,
        "latency": p.latency,
        "uncertainty": p.uncertainty,
    }


def _pressure_from_json(body: object) -> PressureSummary | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise ValueError("pressure_summary must be a JSON object or null")
    return PressureSummary(
        perf=float(body["perf"]),
        risk=float(body["risk"]),
        drift=float(body["drift"]),
        latency=float(body["latency"]),
        uncertainty=float(body["uncertainty"]),
    )


def _contribution_to_json(c: ConfidenceContribution) -> dict[str, object]:
    return {
        "name": c.name,
        "value": c.value,
        "weight": c.weight,
        "weighted": c.weighted,
    }


def _contribution_from_json(body: object) -> ConfidenceContribution:
    if not isinstance(body, dict):
        raise ValueError("confidence_breakdown entries must be JSON objects")
    return ConfidenceContribution(
        name=str(body["name"]),
        value=float(body["value"]),
        weight=float(body["weight"]),
        weighted=float(body["weighted"]),
    )


def _hazard_to_json(h: HazardInfluence) -> dict[str, object]:
    return {
        "code": h.code,
        "severity": h.severity.value,
        "source": h.source,
        "ts_ns": h.ts_ns,
    }


def _hazard_from_json(body: object) -> HazardInfluence:
    if not isinstance(body, dict):
        raise ValueError("active_hazards entries must be JSON objects")
    return HazardInfluence(
        code=str(body["code"]),
        severity=HazardSeverity(body["severity"]),
        source=str(body["source"]),
        ts_ns=int(body["ts_ns"]),
    )


def _throttle_to_json(t: ThrottleInfluence | None) -> dict[str, object] | None:
    if t is None:
        return None
    return {
        "block": t.block,
        "qty_multiplier": t.qty_multiplier,
        "confidence_floor": t.confidence_floor,
        "contributing_codes": list(t.contributing_codes),
    }


def _throttle_from_json(body: object) -> ThrottleInfluence | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise ValueError("throttle_applied must be a JSON object or null")
    return ThrottleInfluence(
        block=bool(body["block"]),
        qty_multiplier=float(body["qty_multiplier"]),
        confidence_floor=float(body["confidence_floor"]),
        contributing_codes=tuple(str(c) for c in body["contributing_codes"]),
    )


def _execution_to_json(e: ExecutionOutcome | None) -> dict[str, object] | None:
    if e is None:
        return None
    return {
        "status": e.status.value,
        "qty": e.qty,
        "price": e.price,
        "venue": e.venue,
        "order_id": e.order_id,
    }


def _execution_from_json(body: object) -> ExecutionOutcome | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise ValueError("execution_outcome must be a JSON object or null")
    return ExecutionOutcome(
        status=ExecutionStatus(body["status"]),
        qty=float(body["qty"]),
        price=float(body["price"]),
        venue=str(body["venue"]),
        order_id=str(body["order_id"]),
    )


def _why_to_json(w: WhyLayer | None) -> dict[str, object] | None:
    if w is None:
        return None
    # ``beliefs`` and ``notes`` are sorted by key on serialise so traces
    # are byte-identical across replays regardless of caller insertion
    # order (INV-15).
    return {
        "philosophy_id": w.philosophy_id,
        "beliefs": [
            {"name": b.name, "strength": b.strength}
            for b in sorted(w.beliefs, key=lambda b: b.name)
        ],
        "entry_logic_id": w.entry_logic_id,
        "exit_logic_id": w.exit_logic_id,
        "risk_model_id": w.risk_model_id,
        "timeframe_id": w.timeframe_id,
        "market_condition_id": w.market_condition_id,
        "composition_id": w.composition_id,
        "notes": [list(n) for n in sorted(w.notes, key=lambda n: n[0])],
    }


def _why_from_json(body: object) -> WhyLayer | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise ValueError("why must be a JSON object or null")
    raw_beliefs = body.get("beliefs", [])
    if not isinstance(raw_beliefs, list):
        raise ValueError("why.beliefs must be a JSON array")
    beliefs_list: list[BeliefReference] = []
    for b in raw_beliefs:
        if not isinstance(b, dict):
            raise ValueError(
                "why.beliefs entries must be JSON objects with "
                "'name' and 'strength'"
            )
        beliefs_list.append(
            BeliefReference(name=str(b["name"]), strength=float(b["strength"]))
        )
    beliefs = tuple(beliefs_list)
    raw_notes = body.get("notes", [])
    if not isinstance(raw_notes, list):
        raise ValueError("why.notes must be a JSON array")
    notes_list: list[tuple[str, str]] = []
    for n in raw_notes:
        if not isinstance(n, list) or len(n) != 2:
            raise ValueError(
                "why.notes entries must be 2-element JSON arrays "
                "[key, text]"
            )
        notes_list.append((str(n[0]), str(n[1])))
    notes = tuple(notes_list)
    return WhyLayer(
        philosophy_id=_optional_str(body.get("philosophy_id")),
        beliefs=beliefs,
        entry_logic_id=_optional_str(body.get("entry_logic_id")),
        exit_logic_id=_optional_str(body.get("exit_logic_id")),
        risk_model_id=_optional_str(body.get("risk_model_id")),
        timeframe_id=_optional_str(body.get("timeframe_id")),
        market_condition_id=_optional_str(body.get("market_condition_id")),
        composition_id=_optional_str(body.get("composition_id")),
        notes=notes,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "DECISION_TRACE_BUILDER_SOURCE",
    "as_system_event",
    "build_decision_trace",
    "compute_trace_id",
    "trace_from_system_event",
]
