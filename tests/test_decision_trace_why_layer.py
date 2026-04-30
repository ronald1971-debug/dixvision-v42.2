"""Tests for the Why-layer extension of :class:`DecisionTrace` (Wave-04 PR-5).

Behaviour covered:

- :class:`BeliefReference` validates name + strength.
- :class:`WhyLayer` rejects empty-string ids (must be ``None``), duplicate
  belief names, duplicate note keys, malformed note entries.
- :class:`DecisionTrace` accepts ``why=None`` (legacy traces) and a
  populated :class:`WhyLayer`.
- :func:`build_decision_trace` threads ``why`` through.
- ``as_system_event`` / ``trace_from_system_event`` round-trip the Why
  layer byte-identically for the same input.
- The serialised ``beliefs`` and ``notes`` arrays are sorted on the wire
  so caller insertion order does not perturb replay (INV-15).
"""

from __future__ import annotations

import json

import pytest

from core.coherence.decision_trace import (
    as_system_event,
    build_decision_trace,
    trace_from_system_event,
)
from core.contracts.decision_trace import (
    DECISION_TRACE_VERSION,
    BeliefReference,
    WhyLayer,
)
from core.contracts.events import Side, SignalEvent


def _signal() -> SignalEvent:
    return SignalEvent(
        ts_ns=10,
        symbol="BTCUSDT",
        side=Side.BUY,
        confidence=0.8,
        plugin_chain=("microstructure_v1",),
    )


# ---------------------------------------------------------------------------
# BeliefReference
# ---------------------------------------------------------------------------


def test_belief_reference_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        BeliefReference(name="", strength=0.5)


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_belief_reference_rejects_out_of_unit_strength(strength: float) -> None:
    with pytest.raises(ValueError):
        BeliefReference(name="trend_following", strength=strength)


# ---------------------------------------------------------------------------
# WhyLayer constructor
# ---------------------------------------------------------------------------


def test_why_layer_defaults_are_empty() -> None:
    why = WhyLayer()
    assert why.philosophy_id is None
    assert why.beliefs == ()
    assert why.entry_logic_id is None
    assert why.exit_logic_id is None
    assert why.risk_model_id is None
    assert why.timeframe_id is None
    assert why.market_condition_id is None
    assert why.composition_id is None
    assert why.notes == ()


def test_why_layer_accepts_full_population() -> None:
    why = WhyLayer(
        philosophy_id="trader-alpha",
        beliefs=(
            BeliefReference(name="trend_following", strength=0.8),
            BeliefReference(name="risk_seeking", strength=0.6),
        ),
        entry_logic_id="entry_breakout",
        exit_logic_id="exit_fixed_target",
        risk_model_id="risk_normal",
        timeframe_id="tf_5m",
        market_condition_id="market_trending",
        composition_id="comp-001",
        notes=(("regime_check", "trending up"),),
    )
    assert why.philosophy_id == "trader-alpha"
    assert len(why.beliefs) == 2


@pytest.mark.parametrize(
    "field",
    [
        "philosophy_id",
        "entry_logic_id",
        "exit_logic_id",
        "risk_model_id",
        "timeframe_id",
        "market_condition_id",
        "composition_id",
    ],
)
def test_why_layer_rejects_empty_string_id(field: str) -> None:
    kwargs = {field: ""}
    with pytest.raises(ValueError, match=f"WhyLayer.{field}"):
        WhyLayer(**kwargs)


def test_why_layer_rejects_duplicate_belief_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        WhyLayer(
            beliefs=(
                BeliefReference(name="trend_following", strength=0.8),
                BeliefReference(name="trend_following", strength=0.6),
            ),
        )


def test_why_layer_rejects_duplicate_note_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        WhyLayer(notes=(("a", "first"), ("a", "second")))


def test_why_layer_rejects_empty_note_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        WhyLayer(notes=(("", "value"),))


# ---------------------------------------------------------------------------
# DecisionTrace integration
# ---------------------------------------------------------------------------


def test_decision_trace_accepts_why_none_for_legacy_traces() -> None:
    trace = build_decision_trace(signal=_signal())
    assert trace.why is None
    assert trace.version == DECISION_TRACE_VERSION


def test_decision_trace_accepts_populated_why() -> None:
    why = WhyLayer(
        philosophy_id="trader-alpha",
        beliefs=(BeliefReference(name="trend_following", strength=0.8),),
        composition_id="comp-001",
    )
    trace = build_decision_trace(signal=_signal(), why=why)
    assert trace.why is why


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def _full_why() -> WhyLayer:
    # Beliefs and notes are stored in alphabetical order so direct
    # equality holds across round-trip (the wire format sorts both for
    # byte-identical replay; insertion order is not semantic).
    return WhyLayer(
        philosophy_id="trader-alpha",
        beliefs=(
            BeliefReference(name="risk_seeking", strength=0.6),
            BeliefReference(name="trend_following", strength=0.8),
        ),
        entry_logic_id="entry_breakout",
        exit_logic_id="exit_fixed_target",
        risk_model_id="risk_normal",
        timeframe_id="tf_5m",
        market_condition_id="market_trending",
        composition_id="comp-001",
        notes=(("note2", "something"), ("regime", "trending up")),
    )


def test_why_round_trips_through_system_event() -> None:
    original = build_decision_trace(signal=_signal(), why=_full_why())
    event = as_system_event(original)
    restored = trace_from_system_event(event)
    assert restored == original


def test_why_round_trip_preserves_none() -> None:
    original = build_decision_trace(signal=_signal(), why=None)
    event = as_system_event(original)
    restored = trace_from_system_event(event)
    assert restored.why is None
    assert restored == original


def test_why_serialisation_is_byte_identical_across_belief_orders() -> None:
    why_ab = WhyLayer(
        beliefs=(
            BeliefReference(name="alpha", strength=0.7),
            BeliefReference(name="beta", strength=0.5),
        ),
    )
    why_ba = WhyLayer(
        beliefs=(
            BeliefReference(name="beta", strength=0.5),
            BeliefReference(name="alpha", strength=0.7),
        ),
    )
    trace_ab = build_decision_trace(signal=_signal(), why=why_ab)
    trace_ba = build_decision_trace(signal=_signal(), why=why_ba)

    bytes_ab = as_system_event(trace_ab).payload["trace"]
    bytes_ba = as_system_event(trace_ba).payload["trace"]
    assert bytes_ab == bytes_ba


def test_why_serialisation_is_byte_identical_across_note_orders() -> None:
    why_ab = WhyLayer(notes=(("a", "first"), ("b", "second")))
    why_ba = WhyLayer(notes=(("b", "second"), ("a", "first")))
    trace_ab = build_decision_trace(signal=_signal(), why=why_ab)
    trace_ba = build_decision_trace(signal=_signal(), why=why_ba)

    bytes_ab = as_system_event(trace_ab).payload["trace"]
    bytes_ba = as_system_event(trace_ba).payload["trace"]
    assert bytes_ab == bytes_ba


def test_why_serialisation_includes_all_ids() -> None:
    trace = build_decision_trace(signal=_signal(), why=_full_why())
    body = json.loads(as_system_event(trace).payload["trace"])
    why = body["why"]
    assert why["philosophy_id"] == "trader-alpha"
    assert why["entry_logic_id"] == "entry_breakout"
    assert why["exit_logic_id"] == "exit_fixed_target"
    assert why["risk_model_id"] == "risk_normal"
    assert why["timeframe_id"] == "tf_5m"
    assert why["market_condition_id"] == "market_trending"
    assert why["composition_id"] == "comp-001"
    assert {b["name"] for b in why["beliefs"]} == {"trend_following", "risk_seeking"}


def test_why_from_json_raises_on_malformed_belief_entry() -> None:
    """A corrupted ledger row with a non-dict belief must raise, not
    silently drop the entry (matches the strict contract of every
    other ``_from_json`` helper)."""
    trace = build_decision_trace(signal=_signal(), why=_full_why())
    body = json.loads(as_system_event(trace).payload["trace"])
    body["why"]["beliefs"].append(None)
    legacy_payload = {
        "trace": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    event = as_system_event(trace)
    legacy_event = type(event)(
        ts_ns=event.ts_ns,
        sub_kind=event.sub_kind,
        source=event.source,
        payload=legacy_payload,
    )
    with pytest.raises(ValueError, match="why.beliefs"):
        trace_from_system_event(legacy_event)


def test_why_from_json_raises_on_malformed_note_entry() -> None:
    """A corrupted ledger row with a malformed note entry must raise,
    not silently drop the entry."""
    trace = build_decision_trace(signal=_signal(), why=_full_why())
    body = json.loads(as_system_event(trace).payload["trace"])
    body["why"]["notes"].append(["only-one-element"])
    legacy_payload = {
        "trace": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    event = as_system_event(trace)
    legacy_event = type(event)(
        ts_ns=event.ts_ns,
        sub_kind=event.sub_kind,
        source=event.source,
        payload=legacy_payload,
    )
    with pytest.raises(ValueError, match="why.notes"):
        trace_from_system_event(legacy_event)


def test_why_legacy_event_without_why_field_round_trips() -> None:
    """A legacy DECISION_TRACE event payload missing the ``why`` key
    must deserialise to ``why=None`` instead of raising."""
    trace = build_decision_trace(signal=_signal())
    event = as_system_event(trace)
    body = json.loads(event.payload["trace"])
    body.pop("why", None)
    legacy_payload = {
        "trace": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    legacy_event = type(event)(
        ts_ns=event.ts_ns,
        sub_kind=event.sub_kind,
        source=event.source,
        payload=legacy_payload,
    )
    restored = trace_from_system_event(legacy_event)
    assert restored.why is None
