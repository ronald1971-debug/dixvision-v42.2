"""Wave-04 PR-2 — TradingView trader-feed parser tests.

Pins the safe-coercion behaviour of
:func:`ui.feeds.tradingview_ideas.parse_tradingview_idea_payload`.
The parser must:

* Never raise on malformed input — return ``None`` so a webhook relay
  can keep streaming.
* Never construct a :class:`TraderObservation` directly (B29).
* Default ``source_feed`` to ``SRC-TRADER-TRADINGVIEW-001`` when the
  payload omits it.
* Reject any payload whose ``philosophy.trader_id`` disagrees with the
  enclosing model's ``trader_id``.
"""

from __future__ import annotations

from typing import Any

from core.contracts.trader_intelligence import (
    TRADER_OBSERVATION_PROFILE_UPDATE,
    TRADER_OBSERVATION_SIGNAL_OBSERVED,
    ConvictionStyle,
    RiskAttitude,
    TimeHorizon,
)
from intelligence_engine.trader_modeling.aggregator import (
    make_trader_observation,
)
from ui.feeds.tradingview_ideas import (
    TRADINGVIEW_PAYLOAD_VERSION,
    TRADINGVIEW_SOURCE_FEED,
    parse_tradingview_idea_payload,
)


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": TRADINGVIEW_PAYLOAD_VERSION,
        "trader_id": "tv:alpha",
        "observation_kind": TRADER_OBSERVATION_PROFILE_UPDATE,
    }
    base.update(overrides)
    return base


def test_parser_minimal_happy_path():
    parsed = parse_tradingview_idea_payload(_payload(), ts_ns=1)
    assert parsed is not None
    model, kind, meta = parsed
    assert model.trader_id == "tv:alpha"
    assert model.source_feed == TRADINGVIEW_SOURCE_FEED
    assert kind == TRADER_OBSERVATION_PROFILE_UPDATE
    assert meta == {}


def test_parser_accepts_signal_observed():
    parsed = parse_tradingview_idea_payload(
        _payload(observation_kind=TRADER_OBSERVATION_SIGNAL_OBSERVED),
        ts_ns=1,
    )
    assert parsed is not None
    _, kind, _ = parsed
    assert kind == TRADER_OBSERVATION_SIGNAL_OBSERVED


def test_parser_with_full_philosophy_round_trips_through_factory():
    parsed = parse_tradingview_idea_payload(
        _payload(
            strategy_signatures=["sig-1"],
            performance_metrics={"win_rate": 0.6},
            risk_profile={"avg_position_size_pct": 0.2},
            regime_performance={"trending": 0.7},
            behavioral_bias={"chases_breakouts": 0.1},
            philosophy={
                "trader_id": "tv:alpha",
                "belief_system": {"trend_following": 0.8},
                "risk_attitude": "BALANCED",
                "time_horizon": "SWING",
                "conviction_style": "SYSTEMATIC",
                "market_view": {"regime_persists": "true"},
                "decision_biases": {"loss_aversion": 0.3},
            },
            meta={"language": "en"},
        ),
        ts_ns=42,
    )
    assert parsed is not None
    model, kind, meta = parsed
    assert model.philosophy is not None
    assert model.philosophy.risk_attitude is RiskAttitude.BALANCED
    assert model.philosophy.time_horizon is TimeHorizon.SWING
    assert model.philosophy.conviction_style is ConvictionStyle.SYSTEMATIC
    # Adapter never constructs the bus record itself — the factory does.
    obs = make_trader_observation(ts_ns=42, model=model, observation_kind=kind, meta=meta)
    assert obs.trader_id == "tv:alpha"
    assert obs.meta == {"language": "en"}


def test_parser_rejects_non_mapping():
    assert parse_tradingview_idea_payload([1, 2, 3], ts_ns=1) is None
    assert parse_tradingview_idea_payload(None, ts_ns=1) is None
    assert parse_tradingview_idea_payload("string", ts_ns=1) is None


def test_parser_rejects_negative_ts_ns():
    assert parse_tradingview_idea_payload(_payload(), ts_ns=-1) is None


def test_parser_rejects_unknown_version():
    assert parse_tradingview_idea_payload(_payload(version=99), ts_ns=1) is None


def test_parser_rejects_non_int_version():
    assert parse_tradingview_idea_payload(_payload(version="latest"), ts_ns=1) is None


def test_parser_rejects_missing_trader_id():
    p = _payload()
    del p["trader_id"]
    assert parse_tradingview_idea_payload(p, ts_ns=1) is None


def test_parser_rejects_empty_trader_id():
    assert parse_tradingview_idea_payload(_payload(trader_id=""), ts_ns=1) is None


def test_parser_rejects_unknown_observation_kind():
    assert (
        parse_tradingview_idea_payload(
            _payload(observation_kind="WHIMSY"), ts_ns=1
        )
        is None
    )


def test_parser_rejects_non_str_source_feed():
    assert parse_tradingview_idea_payload(_payload(source_feed=42), ts_ns=1) is None


def test_parser_rejects_non_str_in_strategy_signatures():
    assert (
        parse_tradingview_idea_payload(
            _payload(strategy_signatures=["ok", 1]), ts_ns=1
        )
        is None
    )


def test_parser_rejects_non_float_in_performance_metrics():
    assert (
        parse_tradingview_idea_payload(
            _payload(performance_metrics={"win_rate": "high"}), ts_ns=1
        )
        is None
    )


def test_parser_rejects_non_str_in_meta():
    assert (
        parse_tradingview_idea_payload(_payload(meta={"k": 1}), ts_ns=1) is None
    )


def test_parser_drops_malformed_philosophy_silently():
    # Philosophy with invalid risk_attitude → philosophy parsed as None;
    # the model is still emitted with philosophy=None (not a hard fail).
    parsed = parse_tradingview_idea_payload(
        _payload(
            philosophy={
                "trader_id": "tv:alpha",
                "risk_attitude": "WHIMSY",
            }
        ),
        ts_ns=1,
    )
    assert parsed is not None
    model, _, _ = parsed
    assert model.philosophy is None


def test_parser_rejects_philosophy_for_other_trader():
    # Philosophy must agree with model.trader_id — otherwise an
    # adapter could ship trader-A data with trader-B's beliefs.
    assert (
        parse_tradingview_idea_payload(
            _payload(
                philosophy={
                    "trader_id": "tv:OTHER",
                    "risk_attitude": "BALANCED",
                    "time_horizon": "SWING",
                    "conviction_style": "SYSTEMATIC",
                }
            ),
            ts_ns=1,
        )
        is None
    )


def test_parser_payload_with_explicit_source_feed_is_passed_through():
    parsed = parse_tradingview_idea_payload(
        _payload(source_feed="SRC-TRADER-TRADINGVIEW-001"), ts_ns=1
    )
    assert parsed is not None
    model, _, _ = parsed
    assert model.source_feed == "SRC-TRADER-TRADINGVIEW-001"


def test_parser_is_pure():
    p = _payload(performance_metrics={"a": 0.1})
    a = parse_tradingview_idea_payload(p, ts_ns=42)
    b = parse_tradingview_idea_payload(p, ts_ns=42)
    assert a == b
