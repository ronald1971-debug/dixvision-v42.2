"""Tests for opponent_model.behavior_predictor (OPP-01).

Covers:
* Contract validation (OpponentObservation / OpponentClassification /
  BehaviorPrediction) — boundary, NaN, sign, and matched-ts_ns rules.
* Config validation — positives, NaN-safety, ordering invariants.
* Each archetype rule fires on its canonical input.
* Rule precedence (HFT_MAKER > SWEEPER > MOMENTUM_TAKER >
  SLOW_RESTING_LIQUIDITY > NOISE).
* Predicted-action mapping.
* Replay determinism (INV-15) — same input → byte-identical output
  across two predictor instances built from the same config.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.contracts.opponent import (
    BehaviorPrediction,
    OpponentArchetype,
    OpponentClassification,
    OpponentObservation,
    PredictedAction,
)
from opponent_model.behavior_predictor import (
    BehaviorPredictor,
    BehaviorPredictorConfig,
    load_behavior_predictor_config,
)


def _obs(
    *,
    ts_ns: int = 1,
    symbol: str = "BTC-USD",
    aggressor_imbalance: float = 0.0,
    avg_taker_size_usd: float = 1_000.0,
    avg_resting_size_usd: float = 20_000.0,
    cancel_to_fill_ratio: float = 2.0,
    top_of_book_refresh_rate_hz: float = 2.0,
    spread_bps: float = 10.0,
) -> OpponentObservation:
    return OpponentObservation(
        ts_ns=ts_ns,
        symbol=symbol,
        aggressor_imbalance=aggressor_imbalance,
        avg_taker_size_usd=avg_taker_size_usd,
        avg_resting_size_usd=avg_resting_size_usd,
        cancel_to_fill_ratio=cancel_to_fill_ratio,
        top_of_book_refresh_rate_hz=top_of_book_refresh_rate_hz,
        spread_bps=spread_bps,
    )


def _predictor() -> BehaviorPredictor:
    return BehaviorPredictor(load_behavior_predictor_config())


# ---------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------


def test_observation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        _obs(ts_ns=0)
    with pytest.raises(ValueError, match="symbol"):
        _obs(symbol="")
    with pytest.raises(ValueError, match="aggressor_imbalance"):
        _obs(aggressor_imbalance=1.5)
    with pytest.raises(ValueError, match="aggressor_imbalance"):
        _obs(aggressor_imbalance=float("nan"))
    with pytest.raises(ValueError, match="avg_taker_size_usd"):
        _obs(avg_taker_size_usd=-1.0)
    with pytest.raises(ValueError, match="avg_resting_size_usd"):
        _obs(avg_resting_size_usd=float("inf"))
    with pytest.raises(ValueError, match="cancel_to_fill_ratio"):
        _obs(cancel_to_fill_ratio=float("nan"))
    with pytest.raises(ValueError, match="top_of_book_refresh_rate_hz"):
        _obs(top_of_book_refresh_rate_hz=-0.5)
    with pytest.raises(ValueError, match="spread_bps"):
        _obs(spread_bps=-1.0)


def test_classification_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="confidence"):
        OpponentClassification(
            archetype=OpponentArchetype.NOISE,
            confidence=1.5,
            rule_fired="r",
            observation_ts_ns=1,
        )
    with pytest.raises(ValueError, match="rule_fired"):
        OpponentClassification(
            archetype=OpponentArchetype.NOISE,
            confidence=0.5,
            rule_fired="",
            observation_ts_ns=1,
        )
    with pytest.raises(ValueError, match="observation_ts_ns"):
        OpponentClassification(
            archetype=OpponentArchetype.NOISE,
            confidence=0.5,
            rule_fired="r",
            observation_ts_ns=0,
        )


def test_prediction_requires_matched_observation_ts_ns() -> None:
    cls = OpponentClassification(
        archetype=OpponentArchetype.NOISE,
        confidence=0.5,
        rule_fired="r",
        observation_ts_ns=42,
    )
    with pytest.raises(ValueError, match="must match"):
        BehaviorPrediction(
            symbol="BTC-USD",
            predicted_action=PredictedAction.HOLD,
            confidence=0.5,
            classification=cls,
            observation_ts_ns=43,
        )


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------


def _cfg(**overrides: float) -> BehaviorPredictorConfig:
    base = dict(
        hft_cancel_to_fill_min=8.0,
        hft_refresh_rate_hz_min=5.0,
        hft_resting_size_usd_max=5_000.0,
        sweeper_taker_size_usd_min=100_000.0,
        sweeper_spread_bps_max=5.0,
        momentum_imbalance_min=0.5,
        momentum_taker_size_usd_min=10_000.0,
        slow_cancel_to_fill_max=1.0,
        slow_refresh_rate_hz_max=1.0,
        slow_resting_size_usd_min=50_000.0,
        confidence_floor=0.55,
        confidence_ceiling=0.95,
        prediction_confidence_scale=0.85,
        noise_action_confidence=0.4,
    )
    base.update(overrides)
    return BehaviorPredictorConfig(**base)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="hft_cancel_to_fill_min"):
        _cfg(hft_cancel_to_fill_min=0.0)
    with pytest.raises(ValueError, match="sweeper_taker_size_usd_min"):
        _cfg(sweeper_taker_size_usd_min=float("nan"))
    with pytest.raises(ValueError, match="momentum_imbalance_min"):
        _cfg(momentum_imbalance_min=1.5)
    with pytest.raises(ValueError, match="floor"):
        _cfg(confidence_floor=0.0)
    with pytest.raises(ValueError, match="floor"):
        _cfg(confidence_floor=0.9, confidence_ceiling=0.5)
    with pytest.raises(ValueError, match="prediction_confidence_scale"):
        _cfg(prediction_confidence_scale=1.5)


def test_load_config_from_registry() -> None:
    cfg = load_behavior_predictor_config()
    assert isinstance(cfg, BehaviorPredictorConfig)
    assert cfg.hft_cancel_to_fill_min > 0.0
    assert cfg.confidence_ceiling >= cfg.confidence_floor


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_behavior_predictor_config(p)


# ---------------------------------------------------------------------
# Archetype rules
# ---------------------------------------------------------------------


def test_hft_maker_rule_fires() -> None:
    eng = _predictor()
    obs = _obs(
        cancel_to_fill_ratio=20.0,
        top_of_book_refresh_rate_hz=10.0,
        avg_resting_size_usd=1_000.0,
        avg_taker_size_usd=500.0,
        aggressor_imbalance=0.05,
        spread_bps=15.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.HFT_MAKER
    assert cls.rule_fired.startswith("hft_maker_")
    assert eng.config.confidence_floor <= cls.confidence <= eng.config.confidence_ceiling


def test_sweeper_rule_fires() -> None:
    eng = _predictor()
    obs = _obs(
        avg_taker_size_usd=500_000.0,
        spread_bps=2.0,
        cancel_to_fill_ratio=2.0,
        top_of_book_refresh_rate_hz=2.0,
        avg_resting_size_usd=20_000.0,
        aggressor_imbalance=0.6,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.SWEEPER
    assert cls.rule_fired.startswith("sweeper_")


def test_momentum_taker_rule_fires() -> None:
    eng = _predictor()
    obs = _obs(
        aggressor_imbalance=0.8,
        avg_taker_size_usd=20_000.0,
        spread_bps=15.0,  # too wide for SWEEPER
        cancel_to_fill_ratio=2.0,
        top_of_book_refresh_rate_hz=2.0,
        avg_resting_size_usd=20_000.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.MOMENTUM_TAKER
    assert cls.rule_fired.startswith("momentum_")


def test_momentum_taker_rule_fires_on_negative_imbalance() -> None:
    """abs(imbalance) is what matters — sell-aggression also triggers."""

    eng = _predictor()
    obs = _obs(
        aggressor_imbalance=-0.8,
        avg_taker_size_usd=20_000.0,
        spread_bps=15.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.MOMENTUM_TAKER


def test_slow_resting_liquidity_rule_fires() -> None:
    eng = _predictor()
    obs = _obs(
        cancel_to_fill_ratio=0.2,
        top_of_book_refresh_rate_hz=0.5,
        avg_resting_size_usd=200_000.0,
        avg_taker_size_usd=500.0,
        aggressor_imbalance=0.05,
        spread_bps=10.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.SLOW_RESTING_LIQUIDITY
    assert cls.rule_fired.startswith("slow_")


def test_noise_fallback_pins_confidence_at_floor() -> None:
    eng = _predictor()
    # Mid-of-the-road observation that fires no rule.
    obs = _obs(
        aggressor_imbalance=0.1,
        avg_taker_size_usd=2_000.0,
        avg_resting_size_usd=20_000.0,
        cancel_to_fill_ratio=2.0,
        top_of_book_refresh_rate_hz=2.0,
        spread_bps=10.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.NOISE
    assert cls.rule_fired == "noise_fallback"
    assert cls.confidence == pytest.approx(eng.config.confidence_floor)


def test_hft_takes_precedence_over_sweeper_when_both_qualify() -> None:
    """HFT_MAKER is rule 1 — must beat SWEEPER (rule 2) if both qualify."""

    eng = _predictor()
    obs = _obs(
        # HFT_MAKER signals
        cancel_to_fill_ratio=20.0,
        top_of_book_refresh_rate_hz=10.0,
        avg_resting_size_usd=1_000.0,
        # Also satisfies SWEEPER (very large taker on narrow book) —
        # but rule 1 wins.
        avg_taker_size_usd=500_000.0,
        spread_bps=2.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.HFT_MAKER


# ---------------------------------------------------------------------
# Predicted-action mapping
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "obs_kwargs, expected_archetype, expected_action",
    [
        (
            dict(
                cancel_to_fill_ratio=20.0,
                top_of_book_refresh_rate_hz=10.0,
                avg_resting_size_usd=1_000.0,
            ),
            OpponentArchetype.HFT_MAKER,
            PredictedAction.WITHDRAW,
        ),
        (
            dict(avg_taker_size_usd=500_000.0, spread_bps=2.0),
            OpponentArchetype.SWEEPER,
            PredictedAction.CONTINUE_AGGRESSION,
        ),
        (
            dict(
                aggressor_imbalance=0.8,
                avg_taker_size_usd=20_000.0,
                spread_bps=15.0,
            ),
            OpponentArchetype.MOMENTUM_TAKER,
            PredictedAction.CONTINUE_AGGRESSION,
        ),
        (
            dict(
                cancel_to_fill_ratio=0.2,
                top_of_book_refresh_rate_hz=0.5,
                avg_resting_size_usd=200_000.0,
            ),
            OpponentArchetype.SLOW_RESTING_LIQUIDITY,
            PredictedAction.FADE,
        ),
        (
            dict(),  # noise fallback
            OpponentArchetype.NOISE,
            PredictedAction.HOLD,
        ),
    ],
)
def test_predict_maps_archetype_to_action(
    obs_kwargs: dict[str, float],
    expected_archetype: OpponentArchetype,
    expected_action: PredictedAction,
) -> None:
    eng = _predictor()
    pred = eng.predict(_obs(**obs_kwargs))
    assert pred.classification.archetype is expected_archetype
    assert pred.predicted_action is expected_action
    assert pred.symbol == "BTC-USD"
    assert pred.observation_ts_ns == pred.classification.observation_ts_ns
    assert 0.0 <= pred.confidence <= 1.0


def test_noise_prediction_uses_flat_confidence() -> None:
    eng = _predictor()
    pred = eng.predict(_obs())
    assert pred.predicted_action is PredictedAction.HOLD
    assert pred.confidence == pytest.approx(eng.config.noise_action_confidence)


def test_non_noise_prediction_scales_classification_confidence() -> None:
    eng = _predictor()
    obs = _obs(
        aggressor_imbalance=0.8,
        avg_taker_size_usd=20_000.0,
        spread_bps=15.0,
    )
    pred = eng.predict(obs)
    assert pred.confidence == pytest.approx(
        pred.classification.confidence * eng.config.prediction_confidence_scale
    )


# ---------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------


def test_classify_is_deterministic_across_instances() -> None:
    cfg = load_behavior_predictor_config()
    a = BehaviorPredictor(cfg)
    b = BehaviorPredictor(cfg)
    obs = _obs(
        aggressor_imbalance=0.7,
        avg_taker_size_usd=15_000.0,
        spread_bps=20.0,
    )
    ca = a.classify(obs)
    cb = b.classify(obs)
    assert ca == cb


def test_predict_is_deterministic_across_calls() -> None:
    eng = _predictor()
    obs = _obs(
        cancel_to_fill_ratio=20.0,
        top_of_book_refresh_rate_hz=10.0,
        avg_resting_size_usd=1_000.0,
    )
    p1 = eng.predict(obs)
    p2 = eng.predict(obs)
    assert p1 == p2


def test_classification_carries_observation_ts_ns() -> None:
    eng = _predictor()
    obs = _obs(ts_ns=12345)
    cls = eng.classify(obs)
    assert cls.observation_ts_ns == 12345
    pred = eng.predict(obs)
    assert pred.observation_ts_ns == 12345
    assert pred.classification.observation_ts_ns == 12345


def test_confidence_scales_monotonically_within_a_rule() -> None:
    """A bigger excess on the same rule yields a >= confidence."""

    eng = _predictor()
    weak = eng.classify(
        _obs(
            aggressor_imbalance=0.55,
            avg_taker_size_usd=11_000.0,
            spread_bps=15.0,
        )
    )
    strong = eng.classify(
        _obs(
            aggressor_imbalance=0.95,
            avg_taker_size_usd=50_000.0,
            spread_bps=15.0,
        )
    )
    assert weak.archetype is OpponentArchetype.MOMENTUM_TAKER
    assert strong.archetype is OpponentArchetype.MOMENTUM_TAKER
    assert strong.confidence >= weak.confidence


def test_confidence_never_exceeds_ceiling() -> None:
    eng = _predictor()
    # Push every dimension as far past threshold as physically possible.
    obs = _obs(
        cancel_to_fill_ratio=1_000.0,
        top_of_book_refresh_rate_hz=1_000.0,
        avg_resting_size_usd=0.0,
    )
    cls = eng.classify(obs)
    assert cls.archetype is OpponentArchetype.HFT_MAKER
    assert cls.confidence <= eng.config.confidence_ceiling
    assert math.isfinite(cls.confidence)
