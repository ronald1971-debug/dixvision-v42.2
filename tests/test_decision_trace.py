"""Tests for the BEHAVIOR-P4 decision-trace layer.

Covers:

* Frozen dataclass invariants (all contract types under
  :mod:`core.contracts.decision_trace`).
* :func:`compute_trace_id` determinism (INV-15).
* :func:`build_decision_trace` minimal + full assembly + monotonic
  weighted-sum check.
* :func:`as_system_event` / :func:`trace_from_system_event` round-trip
  (lossless replay).
* Validation (unit-interval enforcement, non-empty fields,
  non-negative qty/price, ts_ns).
* Replay determinism — two builds from the same inputs produce
  byte-identical SystemEvents.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from core.coherence.decision_trace import (
    DECISION_TRACE_BUILDER_SOURCE,
    as_system_event,
    build_decision_trace,
    compute_trace_id,
    trace_from_system_event,
)
from core.contracts.decision_trace import (
    DECISION_TRACE_VERSION,
    ConfidenceContribution,
    DecisionTrace,
    ExecutionOutcome,
    HazardInfluence,
    PressureSummary,
    ThrottleInfluence,
)
from core.contracts.events import (
    EventKind,
    ExecutionStatus,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEventKind,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    *,
    ts_ns: int = 1_000_000_000,
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    confidence: float = 0.8,
    plugin_chain: tuple[str, ...] = ("microstructure_v1", "sentiment_v2"),
) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        confidence=confidence,
        plugin_chain=plugin_chain,
    )


def _full_breakdown() -> tuple[ConfidenceContribution, ...]:
    return (
        ConfidenceContribution("consensus", 0.8, 0.5, 0.4),
        ConfidenceContribution("strength", 0.6, 0.5, 0.3),
    )


def _pressure() -> PressureSummary:
    return PressureSummary(
        perf=0.6, risk=0.3, drift=0.1, latency=0.05, uncertainty=0.2
    )


def _hazards() -> tuple[HazardInfluence, ...]:
    return (
        HazardInfluence(
            code="HAZ-13",
            severity=HazardSeverity.MEDIUM,
            source="system_engine.scvs",
            ts_ns=999_000_000,
        ),
    )


def _throttle() -> ThrottleInfluence:
    return ThrottleInfluence(
        block=False,
        qty_multiplier=0.5,
        confidence_floor=0.6,
        contributing_codes=("HAZ-13",),
    )


def _execution() -> ExecutionOutcome:
    return ExecutionOutcome(
        status=ExecutionStatus.FILLED,
        qty=0.25,
        price=42_000.0,
        venue="paper",
        order_id="ORD-1",
    )


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance,attr",
    [
        (ConfidenceContribution("c", 0.5, 0.5, 0.25), "value"),
        (
            PressureSummary(
                perf=0.0, risk=0.0, drift=0.0, latency=0.0, uncertainty=0.0
            ),
            "perf",
        ),
        (
            HazardInfluence(
                code="HAZ-01", severity=HazardSeverity.LOW, source="x", ts_ns=1
            ),
            "code",
        ),
        (
            ThrottleInfluence(
                block=False,
                qty_multiplier=1.0,
                confidence_floor=0.0,
                contributing_codes=(),
            ),
            "block",
        ),
        (
            ExecutionOutcome(status=ExecutionStatus.PROPOSED, qty=0.0, price=0.0),
            "qty",
        ),
    ],
)
def test_contracts_are_frozen(instance: object, attr: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, attr, "x")


def test_decision_trace_is_frozen() -> None:
    trace = build_decision_trace(signal=_signal())
    with pytest.raises(dataclasses.FrozenInstanceError):
        trace.trace_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConfidenceContribution validation
# ---------------------------------------------------------------------------


def test_confidence_contribution_requires_name() -> None:
    with pytest.raises(ValueError, match="name"):
        ConfidenceContribution("", 0.5, 0.5, 0.25)


@pytest.mark.parametrize("field_name", ["value", "weight"])
def test_confidence_contribution_unit_interval(field_name: str) -> None:
    kwargs: dict[str, float] = {"value": 0.5, "weight": 0.5, "weighted": 0.25}
    kwargs[field_name] = 1.5
    if field_name == "value":
        kwargs["weighted"] = 1.5 * 0.5
    else:
        kwargs["weighted"] = 0.5 * 1.5
    with pytest.raises(ValueError, match=field_name):
        ConfidenceContribution(name="c", **kwargs)


def test_confidence_contribution_weighted_must_match_value_times_weight() -> None:
    with pytest.raises(ValueError, match="weighted"):
        ConfidenceContribution("c", 0.5, 0.5, 0.99)


def test_confidence_contribution_weighted_unit_interval() -> None:
    with pytest.raises(ValueError, match="weighted"):
        ConfidenceContribution("c", 0.5, 0.5, -0.1)


# ---------------------------------------------------------------------------
# Other contract validation
# ---------------------------------------------------------------------------


def test_pressure_summary_unit_interval() -> None:
    with pytest.raises(ValueError, match="perf"):
        PressureSummary(perf=1.1, risk=0.0, drift=0.0, latency=0.0, uncertainty=0.0)


def test_hazard_influence_requires_code() -> None:
    with pytest.raises(ValueError, match="code"):
        HazardInfluence(
            code="", severity=HazardSeverity.LOW, source="x", ts_ns=1
        )


def test_hazard_influence_requires_source() -> None:
    with pytest.raises(ValueError, match="source"):
        HazardInfluence(
            code="HAZ-01", severity=HazardSeverity.LOW, source="", ts_ns=1
        )


def test_hazard_influence_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        HazardInfluence(
            code="HAZ-01", severity=HazardSeverity.LOW, source="x", ts_ns=-1
        )


def test_throttle_influence_unit_interval() -> None:
    with pytest.raises(ValueError, match="qty_multiplier"):
        ThrottleInfluence(
            block=False,
            qty_multiplier=1.5,
            confidence_floor=0.0,
            contributing_codes=(),
        )
    with pytest.raises(ValueError, match="confidence_floor"):
        ThrottleInfluence(
            block=False,
            qty_multiplier=1.0,
            confidence_floor=1.5,
            contributing_codes=(),
        )


def test_throttle_influence_rejects_empty_codes() -> None:
    with pytest.raises(ValueError, match="contributing_codes"):
        ThrottleInfluence(
            block=False,
            qty_multiplier=1.0,
            confidence_floor=0.0,
            contributing_codes=("",),
        )


def test_execution_outcome_rejects_negative_qty() -> None:
    with pytest.raises(ValueError, match="qty"):
        ExecutionOutcome(status=ExecutionStatus.FILLED, qty=-1.0, price=1.0)


def test_execution_outcome_rejects_negative_price() -> None:
    with pytest.raises(ValueError, match="price"):
        ExecutionOutcome(status=ExecutionStatus.FILLED, qty=1.0, price=-1.0)


# ---------------------------------------------------------------------------
# compute_trace_id
# ---------------------------------------------------------------------------


def test_compute_trace_id_is_deterministic() -> None:
    a = compute_trace_id(symbol="BTCUSDT", ts_ns=1, plugin_chain=("p1", "p2"))
    b = compute_trace_id(symbol="BTCUSDT", ts_ns=1, plugin_chain=("p1", "p2"))
    assert a == b
    assert len(a) == 16


def test_compute_trace_id_distinguishes_inputs() -> None:
    a = compute_trace_id(symbol="BTCUSDT", ts_ns=1, plugin_chain=("p1",))
    b = compute_trace_id(symbol="ETHUSDT", ts_ns=1, plugin_chain=("p1",))
    c = compute_trace_id(symbol="BTCUSDT", ts_ns=2, plugin_chain=("p1",))
    d = compute_trace_id(symbol="BTCUSDT", ts_ns=1, plugin_chain=("p2",))
    e = compute_trace_id(symbol="BTCUSDT", ts_ns=1, plugin_chain=())
    assert len({a, b, c, d, e}) == 5


def test_compute_trace_id_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        compute_trace_id(symbol="", ts_ns=1, plugin_chain=())


def test_compute_trace_id_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        compute_trace_id(symbol="BTCUSDT", ts_ns=-1, plugin_chain=())


# ---------------------------------------------------------------------------
# build_decision_trace
# ---------------------------------------------------------------------------


def test_build_decision_trace_minimal() -> None:
    trace = build_decision_trace(signal=_signal())
    assert isinstance(trace, DecisionTrace)
    assert trace.version == DECISION_TRACE_VERSION
    assert trace.symbol == "BTCUSDT"
    assert trace.side is Side.BUY
    assert trace.final_confidence == pytest.approx(0.8)
    assert trace.plugin_chain == ("microstructure_v1", "sentiment_v2")
    assert trace.regime is None
    assert trace.pressure_summary is None
    assert trace.safety_modifier is None
    assert trace.confidence_breakdown == ()
    assert trace.active_hazards == ()
    assert trace.throttle_applied is None
    assert trace.execution_outcome is None
    # trace_id is the deterministic hash
    assert trace.trace_id == compute_trace_id(
        symbol="BTCUSDT",
        ts_ns=1_000_000_000,
        plugin_chain=("microstructure_v1", "sentiment_v2"),
    )


def test_build_decision_trace_full() -> None:
    trace = build_decision_trace(
        signal=_signal(),
        confidence_breakdown=_full_breakdown(),
        regime="TREND_UP",
        pressure_summary=_pressure(),
        safety_modifier=0.85,
        active_hazards=_hazards(),
        throttle_applied=_throttle(),
        execution_outcome=_execution(),
    )
    assert trace.regime == "TREND_UP"
    assert trace.pressure_summary is not None
    assert trace.safety_modifier == pytest.approx(0.85)
    assert len(trace.confidence_breakdown) == 2
    assert trace.confidence_breakdown[0].name == "consensus"
    assert trace.active_hazards[0].code == "HAZ-13"
    assert trace.throttle_applied is not None
    assert trace.throttle_applied.qty_multiplier == pytest.approx(0.5)
    assert trace.execution_outcome is not None
    assert trace.execution_outcome.status is ExecutionStatus.FILLED


def test_build_decision_trace_partial_breakdown_allowed() -> None:
    # Sum of weighted contributions = 0.4, signal confidence = 0.8 → OK.
    trace = build_decision_trace(
        signal=_signal(confidence=0.8),
        confidence_breakdown=(
            ConfidenceContribution("consensus", 0.8, 0.5, 0.4),
        ),
    )
    assert trace.final_confidence == pytest.approx(0.8)


def test_build_decision_trace_overstating_breakdown_rejected() -> None:
    # Sum = 0.9 > signal.confidence = 0.5.
    with pytest.raises(ValueError, match="weighted-sum"):
        build_decision_trace(
            signal=_signal(confidence=0.5),
            confidence_breakdown=(
                ConfidenceContribution("consensus", 0.9, 1.0, 0.9),
            ),
        )


def test_build_decision_trace_replay_parity() -> None:
    a = build_decision_trace(
        signal=_signal(),
        confidence_breakdown=_full_breakdown(),
        regime="TREND_UP",
        pressure_summary=_pressure(),
        safety_modifier=0.85,
        active_hazards=_hazards(),
        throttle_applied=_throttle(),
        execution_outcome=_execution(),
    )
    b = build_decision_trace(
        signal=_signal(),
        confidence_breakdown=_full_breakdown(),
        regime="TREND_UP",
        pressure_summary=_pressure(),
        safety_modifier=0.85,
        active_hazards=_hazards(),
        throttle_applied=_throttle(),
        execution_outcome=_execution(),
    )
    assert a == b


# ---------------------------------------------------------------------------
# as_system_event / trace_from_system_event round-trip
# ---------------------------------------------------------------------------


def test_as_system_event_is_decision_trace() -> None:
    trace = build_decision_trace(signal=_signal())
    event = as_system_event(trace)
    assert event.kind is EventKind.SYSTEM
    assert event.sub_kind is SystemEventKind.DECISION_TRACE
    assert event.source == DECISION_TRACE_BUILDER_SOURCE
    assert event.ts_ns == trace.ts_ns


def test_as_system_event_payload_is_sorted_json() -> None:
    trace = build_decision_trace(signal=_signal())
    event = as_system_event(trace)
    raw = event.payload["trace"]
    body = json.loads(raw)
    assert body["trace_id"] == trace.trace_id
    # round-trip through json.dumps with sort_keys must be stable
    repacked = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert repacked == raw


def test_as_system_event_replay_parity_byte_identical() -> None:
    trace = build_decision_trace(
        signal=_signal(),
        confidence_breakdown=_full_breakdown(),
        regime="TREND_UP",
        pressure_summary=_pressure(),
        safety_modifier=0.85,
        active_hazards=_hazards(),
        throttle_applied=_throttle(),
        execution_outcome=_execution(),
    )
    a = as_system_event(trace)
    b = as_system_event(trace)
    assert a == b
    assert a.payload == b.payload


def test_round_trip_full_trace() -> None:
    original = build_decision_trace(
        signal=_signal(),
        confidence_breakdown=_full_breakdown(),
        regime="TREND_UP",
        pressure_summary=_pressure(),
        safety_modifier=0.85,
        active_hazards=_hazards(),
        throttle_applied=_throttle(),
        execution_outcome=_execution(),
    )
    event = as_system_event(original)
    recovered = trace_from_system_event(event)
    assert recovered == original


def test_round_trip_minimal_trace() -> None:
    original = build_decision_trace(signal=_signal())
    event = as_system_event(original)
    recovered = trace_from_system_event(event)
    assert recovered == original


def test_trace_from_system_event_rejects_wrong_sub_kind() -> None:
    trace = build_decision_trace(signal=_signal())
    event = as_system_event(trace)
    bad = type(event)(
        ts_ns=event.ts_ns,
        sub_kind=SystemEventKind.HEARTBEAT,
        source=event.source,
        payload=event.payload,
    )
    with pytest.raises(ValueError, match="DECISION_TRACE"):
        trace_from_system_event(bad)


def test_trace_from_system_event_rejects_missing_payload() -> None:
    trace = build_decision_trace(signal=_signal())
    event = as_system_event(trace)
    bad = type(event)(
        ts_ns=event.ts_ns,
        sub_kind=event.sub_kind,
        source=event.source,
        payload={},
    )
    with pytest.raises(ValueError, match="trace"):
        trace_from_system_event(bad)


def test_as_system_event_rejects_empty_source() -> None:
    trace = build_decision_trace(signal=_signal())
    with pytest.raises(ValueError, match="source"):
        as_system_event(trace, source="")


# ---------------------------------------------------------------------------
# DecisionTrace direct validation
# ---------------------------------------------------------------------------


def test_decision_trace_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="x",
            ts_ns=-1,
            symbol="BTCUSDT",
            side=Side.BUY,
            final_confidence=0.5,
            plugin_chain=(),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
        )


def test_decision_trace_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="x",
            ts_ns=0,
            symbol="",
            side=Side.BUY,
            final_confidence=0.5,
            plugin_chain=(),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
        )


def test_decision_trace_rejects_confidence_out_of_unit() -> None:
    with pytest.raises(ValueError, match="final_confidence"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="x",
            ts_ns=0,
            symbol="BTCUSDT",
            side=Side.BUY,
            final_confidence=1.5,
            plugin_chain=(),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
        )


def test_decision_trace_rejects_safety_modifier_out_of_unit() -> None:
    with pytest.raises(ValueError, match="safety_modifier"):
        DecisionTrace(
            version=DECISION_TRACE_VERSION,
            trace_id="x",
            ts_ns=0,
            symbol="BTCUSDT",
            side=Side.BUY,
            final_confidence=0.5,
            plugin_chain=(),
            regime=None,
            pressure_summary=None,
            safety_modifier=2.0,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
        )


def test_decision_trace_rejects_invalid_version() -> None:
    with pytest.raises(ValueError, match="version"):
        DecisionTrace(
            version=0,
            trace_id="x",
            ts_ns=0,
            symbol="BTCUSDT",
            side=Side.BUY,
            final_confidence=0.5,
            plugin_chain=(),
            regime=None,
            pressure_summary=None,
            safety_modifier=None,
            confidence_breakdown=(),
            active_hazards=(),
            throttle_applied=None,
            execution_outcome=None,
        )
