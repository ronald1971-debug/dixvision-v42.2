"""Wave-04 PR-2 — ``observation_as_system_event`` projection tests.

Pins INV-15 round-trip parity. The on-disk SystemEvent must be
byte-identical for the same input across runs, and the round-trip
projection must reconstruct an equal :class:`TraderObservation`. The
reverse path goes through the factory so B29-style validation is
re-enforced on replay.
"""

from __future__ import annotations

import pytest

from core.contracts.events import SystemEvent, SystemEventKind
from core.contracts.trader_intelligence import (
    TRADER_OBSERVATION_SIGNAL_OBSERVED,
    ConvictionStyle,
    PhilosophyProfile,
    RiskAttitude,
    TimeHorizon,
    TraderModel,
)
from intelligence_engine.trader_modeling.aggregator import (
    TRADER_MODELING_SOURCE,
    make_trader_observation,
)
from intelligence_engine.trader_modeling.observation import (
    OBSERVATION_EVENT_VERSION,
    observation_as_system_event,
    observation_from_system_event,
)


def _full_model() -> TraderModel:
    return TraderModel(
        trader_id="tv:beta",
        source_feed="SRC-TRADER-TRADINGVIEW-001",
        strategy_signatures=("sig-1", "sig-2"),
        performance_metrics={"win_rate": 0.61, "sharpe": 1.4},
        risk_profile={"avg_position_size_pct": 0.25},
        regime_performance={"trending": 0.7, "ranging": 0.4},
        behavioral_bias={"chases_breakouts": 0.3},
        philosophy=PhilosophyProfile(
            trader_id="tv:beta",
            belief_system={"trend_following": 0.8, "mean_reversion": 0.2},
            risk_attitude=RiskAttitude.BALANCED,
            time_horizon=TimeHorizon.SWING,
            conviction_style=ConvictionStyle.SYSTEMATIC,
            market_view={"regime_persists": "true"},
            decision_biases={"loss_aversion": 0.4},
        ),
        meta={"language": "en"},
    )


def test_projection_round_trip_with_philosophy():
    obs = make_trader_observation(
        ts_ns=1_700_000_000_000_000_000,
        model=_full_model(),
        observation_kind=TRADER_OBSERVATION_SIGNAL_OBSERVED,
        meta={"k": "v"},
    )
    event = observation_as_system_event(obs)
    assert event.sub_kind is SystemEventKind.TRADER_OBSERVED
    assert event.source == TRADER_MODELING_SOURCE
    assert event.produced_by_engine == "intelligence_engine"
    assert event.ts_ns == obs.ts_ns
    assert observation_from_system_event(event) == obs


def test_projection_round_trip_minimal_model():
    obs = make_trader_observation(
        ts_ns=0,
        model=TraderModel(
            trader_id="tv:gamma",
            source_feed="SRC-TRADER-TRADINGVIEW-001",
        ),
    )
    event = observation_as_system_event(obs)
    assert observation_from_system_event(event) == obs


def test_projection_is_byte_identical_for_same_input():
    obs1 = make_trader_observation(ts_ns=42, model=_full_model(), meta={"k": "v"})
    obs2 = make_trader_observation(ts_ns=42, model=_full_model(), meta={"k": "v"})
    e1 = observation_as_system_event(obs1)
    e2 = observation_as_system_event(obs2)
    # The serialised body is the canonical determinism gate (INV-15).
    assert e1.payload["observation"] == e2.payload["observation"]
    # A stable wire-format version flag rides on the body so consumers
    # can gate behaviour without unpacking the rest.
    assert f'"version":{OBSERVATION_EVENT_VERSION}' in e1.payload["observation"]


def test_projection_dict_key_order_does_not_affect_serialisation():
    # Same logical inputs supplied with different host-dict ordering
    # must produce byte-identical JSON.
    a = make_trader_observation(
        ts_ns=1,
        model=TraderModel(
            trader_id="tv:delta",
            source_feed="SRC-TRADER-TRADINGVIEW-001",
            performance_metrics={"a": 0.1, "b": 0.2},
            risk_profile={"x": 0.5, "y": 0.6},
        ),
    )
    b = make_trader_observation(
        ts_ns=1,
        model=TraderModel(
            trader_id="tv:delta",
            source_feed="SRC-TRADER-TRADINGVIEW-001",
            performance_metrics={"b": 0.2, "a": 0.1},
            risk_profile={"y": 0.6, "x": 0.5},
        ),
    )
    assert (
        observation_as_system_event(a).payload["observation"]
        == observation_as_system_event(b).payload["observation"]
    )


def test_ts_ns_override_does_not_mutate_inner_observation():
    obs = make_trader_observation(ts_ns=10, model=_full_model())
    event = observation_as_system_event(obs, ts_ns_override=99)
    assert event.ts_ns == 99
    # The inner observation timestamp survives the override.
    reconstructed = observation_from_system_event(event)
    assert reconstructed.ts_ns == 10
    assert reconstructed == obs


def test_projection_rejects_wrong_sub_kind_on_replay():
    with pytest.raises(ValueError, match="TRADER_OBSERVED"):
        observation_from_system_event(
            SystemEvent(
                ts_ns=0,
                sub_kind=SystemEventKind.SOURCE_HEARTBEAT,
                source=TRADER_MODELING_SOURCE,
                payload={"observation": "{}"},
                produced_by_engine="intelligence_engine",
            )
        )


def test_projection_rejects_missing_observation_payload():
    with pytest.raises(ValueError, match="observation"):
        observation_from_system_event(
            SystemEvent(
                ts_ns=0,
                sub_kind=SystemEventKind.TRADER_OBSERVED,
                source=TRADER_MODELING_SOURCE,
                payload={},
                produced_by_engine="intelligence_engine",
            )
        )


def test_projection_replay_revalidates_through_factory():
    # An event with an observation-kind value that isn't in the factory's
    # legal set must fail on replay even though the wire payload is
    # well-formed JSON. This is the B29-symmetric gate on the read side.
    obs = make_trader_observation(ts_ns=1, model=_full_model())
    event = observation_as_system_event(obs)
    bad_payload = event.payload["observation"].replace(
        '"observation_kind":"PROFILE_UPDATE"',
        '"observation_kind":"WHIMSY"',
    )
    bad = SystemEvent(
        ts_ns=event.ts_ns,
        sub_kind=event.sub_kind,
        source=event.source,
        payload={"observation": bad_payload},
        produced_by_engine=event.produced_by_engine,
    )
    with pytest.raises(ValueError, match="observation_kind"):
        observation_from_system_event(bad)


def test_projection_rejects_empty_source():
    obs = make_trader_observation(ts_ns=1, model=_full_model())
    with pytest.raises(ValueError, match="source"):
        observation_as_system_event(obs, source="")


def test_projection_rejects_negative_override():
    obs = make_trader_observation(ts_ns=1, model=_full_model())
    with pytest.raises(ValueError, match="ts_ns_override"):
        observation_as_system_event(obs, ts_ns_override=-1)
