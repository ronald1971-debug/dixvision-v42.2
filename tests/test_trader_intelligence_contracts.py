"""Wave-04 PR-1 — Trader-Intelligence contracts unit tests.

Pin the structural guarantees that the rest of the wave depends on:

* Frozen / slotted instances with structural equality (replay parity,
  INV-15). Instances are intentionally NOT hashable — ``Mapping[...]``
  fields default to ``dict`` (mirrors :mod:`core.contracts.learning`).
* Structural equality (same fields → equal).
* No PII / no callables — only deterministic primitives.
* Default constructors are pure (no clock, no IO, no side effects).
* :class:`TraderObservation` carries an optional :class:`PhilosophyProfile`
  via the embedded :class:`TraderModel` so the composition engine
  (Wave-04 PR-4) can read both layers without a second event.
"""

from __future__ import annotations

import pytest

from core.contracts.trader_intelligence import (
    TRADER_OBSERVATION_PROFILE_UPDATE,
    TRADER_OBSERVATION_SIGNAL_OBSERVED,
    ConvictionStyle,
    PhilosophyProfile,
    RiskAttitude,
    TimeHorizon,
    TraderModel,
    TraderObservation,
)

# ---------------------------------------------------------------------------
# PhilosophyProfile
# ---------------------------------------------------------------------------


def test_philosophy_profile_minimal_constructor():
    """Trader id is the only required field; sane defaults elsewhere."""

    p = PhilosophyProfile(trader_id="t1")
    assert p.trader_id == "t1"
    assert p.belief_system == {}
    assert p.risk_attitude is RiskAttitude.UNKNOWN
    assert p.time_horizon is TimeHorizon.UNKNOWN
    assert p.conviction_style is ConvictionStyle.UNKNOWN
    assert p.market_view == {}
    assert p.decision_biases == {}


def test_philosophy_profile_is_frozen():
    p = PhilosophyProfile(trader_id="t1")
    with pytest.raises(AttributeError):
        p.trader_id = "t2"  # type: ignore[misc]


def test_philosophy_profile_is_slotted():
    """``slots=True`` produces a ``__slots__`` tuple on the class.

    We don't probe via ``setattr`` because frozen dataclasses raise on
    any setattr regardless of slot membership; the slot guarantee is
    verified statically here so both invariants stay distinct.
    """

    assert hasattr(PhilosophyProfile, "__slots__")
    assert "__dict__" not in PhilosophyProfile.__slots__


def test_philosophy_profile_structural_equality():
    a = PhilosophyProfile(
        trader_id="t1",
        belief_system={"trend_following": 0.8},
        risk_attitude=RiskAttitude.AGGRESSIVE,
        time_horizon=TimeHorizon.SWING,
        conviction_style=ConvictionStyle.PREDICTIVE,
        market_view={"markets_are_random": "false"},
        decision_biases={"loss_aversion": 0.4},
    )
    b = PhilosophyProfile(
        trader_id="t1",
        belief_system={"trend_following": 0.8},
        risk_attitude=RiskAttitude.AGGRESSIVE,
        time_horizon=TimeHorizon.SWING,
        conviction_style=ConvictionStyle.PREDICTIVE,
        market_view={"markets_are_random": "false"},
        decision_biases={"loss_aversion": 0.4},
    )
    assert a == b


def test_philosophy_profile_inequality_on_field_change():
    base = PhilosophyProfile(trader_id="t1", risk_attitude=RiskAttitude.BALANCED)
    other = PhilosophyProfile(
        trader_id="t1", risk_attitude=RiskAttitude.AGGRESSIVE
    )
    assert base != other


# ---------------------------------------------------------------------------
# TraderModel
# ---------------------------------------------------------------------------


def test_trader_model_minimal_constructor():
    m = TraderModel(trader_id="t1", source_feed="SRC-TRADER-TRADINGVIEW-001")
    assert m.trader_id == "t1"
    assert m.source_feed == "SRC-TRADER-TRADINGVIEW-001"
    assert m.strategy_signatures == ()
    assert m.performance_metrics == {}
    assert m.risk_profile == {}
    assert m.regime_performance == {}
    assert m.behavioral_bias == {}
    assert m.philosophy is None
    assert m.meta == {}


def test_trader_model_is_frozen():
    m = TraderModel(trader_id="t1", source_feed="SRC-A")
    with pytest.raises(AttributeError):
        m.trader_id = "t2"  # type: ignore[misc]


def test_trader_model_is_slotted():
    assert hasattr(TraderModel, "__slots__")
    assert "__dict__" not in TraderModel.__slots__


def test_trader_model_structural_equality():
    p = PhilosophyProfile(trader_id="t1")
    a = TraderModel(
        trader_id="t1",
        source_feed="SRC-A",
        strategy_signatures=("sig-1", "sig-2"),
        performance_metrics={"win_rate": 0.62},
        risk_profile={"avg_position_size_pct": 1.2},
        regime_performance={"trending": 0.7, "ranging": 0.3},
        behavioral_bias={"chases_breakouts": 0.5},
        philosophy=p,
        meta={"note": "fixture"},
    )
    b = TraderModel(
        trader_id="t1",
        source_feed="SRC-A",
        strategy_signatures=("sig-1", "sig-2"),
        performance_metrics={"win_rate": 0.62},
        risk_profile={"avg_position_size_pct": 1.2},
        regime_performance={"trending": 0.7, "ranging": 0.3},
        behavioral_bias={"chases_breakouts": 0.5},
        philosophy=p,
        meta={"note": "fixture"},
    )
    assert a == b


def test_trader_model_signature_order_matters():
    """Signature ordering is a tuple, not a set — order changes identity.

    Composition (Wave-04 PR-4) reads signatures positionally to map
    against the canonical decomposition library; reordering would
    silently swap entry/exit/risk components.
    """

    a = TraderModel(
        trader_id="t1", source_feed="SRC-A", strategy_signatures=("a", "b")
    )
    b = TraderModel(
        trader_id="t1", source_feed="SRC-A", strategy_signatures=("b", "a")
    )
    assert a != b


# ---------------------------------------------------------------------------
# TraderObservation
# ---------------------------------------------------------------------------


def _make_obs(
    *,
    ts_ns: int = 1_700_000_000_000_000_000,
    trader_id: str = "t1",
    observation_kind: str = TRADER_OBSERVATION_PROFILE_UPDATE,
) -> TraderObservation:
    return TraderObservation(
        ts_ns=ts_ns,
        trader_id=trader_id,
        observation_kind=observation_kind,
        model=TraderModel(trader_id=trader_id, source_feed="SRC-A"),
    )


def test_trader_observation_round_trips_model_id():
    obs = _make_obs()
    assert obs.trader_id == obs.model.trader_id


def test_trader_observation_is_frozen():
    obs = _make_obs()
    with pytest.raises(AttributeError):
        obs.ts_ns = 0  # type: ignore[misc]


def test_trader_observation_is_slotted():
    assert hasattr(TraderObservation, "__slots__")
    assert "__dict__" not in TraderObservation.__slots__


def test_trader_observation_structural_equality():
    a = _make_obs()
    b = _make_obs()
    assert a == b


def test_trader_observation_kinds_are_distinct():
    profile = _make_obs(observation_kind=TRADER_OBSERVATION_PROFILE_UPDATE)
    signal = _make_obs(observation_kind=TRADER_OBSERVATION_SIGNAL_OBSERVED)
    assert profile != signal


def test_trader_observation_carries_philosophy_through_model():
    """Philosophy rides on the embedded model — no second event class."""

    p = PhilosophyProfile(
        trader_id="t1",
        risk_attitude=RiskAttitude.CONSERVATIVE,
        time_horizon=TimeHorizon.POSITION,
    )
    obs = TraderObservation(
        ts_ns=1,
        trader_id="t1",
        observation_kind=TRADER_OBSERVATION_PROFILE_UPDATE,
        model=TraderModel(trader_id="t1", source_feed="SRC-A", philosophy=p),
    )
    assert obs.model.philosophy is p


# ---------------------------------------------------------------------------
# Sentinel constants — pin canonical observation-kind discriminators.
# ---------------------------------------------------------------------------


def test_observation_kind_sentinels():
    assert TRADER_OBSERVATION_PROFILE_UPDATE == "PROFILE_UPDATE"
    assert TRADER_OBSERVATION_SIGNAL_OBSERVED == "SIGNAL_OBSERVED"
