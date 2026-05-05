"""Tests for the IND-L09 sentiment_aggregator v1 plugin (Indira plugin #8)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.sentiment_aggregator import (
    SentimentAggregatorV1,
)


def _tick(ts: int, *, sentiment: object | None) -> MarketTick:
    meta: dict[str, object] = {}
    if sentiment is not None:
        meta["sentiment"] = sentiment
    return MarketTick(
        ts_ns=ts,
        symbol="BTC-USD",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta=meta,
    )


def test_no_signal_until_warmup_complete() -> None:
    p = SentimentAggregatorV1(
        warmup_ticks=5,
        sentiment_threshold=0.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, sentiment=0.9))
    assert out == ()


def test_sustained_bullish_emits_buy() -> None:
    p = SentimentAggregatorV1(
        alpha=0.5,
        warmup_ticks=4,
        sentiment_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, sentiment=0.8))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.plugin_chain == ("sentiment_aggregator_v1",)
    assert sig.produced_by_engine == "intelligence_engine"
    assert float(sig.meta["ema"]) > 0.3


def test_sustained_bearish_emits_sell() -> None:
    p = SentimentAggregatorV1(
        alpha=0.5,
        warmup_ticks=4,
        sentiment_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, sentiment=-0.8))
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert float(out[0].meta["ema"]) < -0.3


def test_neutral_sentiment_emits_nothing() -> None:
    p = SentimentAggregatorV1(
        alpha=0.5,
        warmup_ticks=4,
        sentiment_threshold=0.05,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, sentiment=0.0))
    assert out == ()


def test_below_threshold_emits_nothing() -> None:
    p = SentimentAggregatorV1(
        alpha=0.5,
        warmup_ticks=4,
        sentiment_threshold=0.99,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, sentiment=0.5))
    assert out == ()


def test_missing_sentiment_meta_drops() -> None:
    p = SentimentAggregatorV1(warmup_ticks=2)
    assert p.on_tick(_tick(0, sentiment=None)) == ()


def test_non_numeric_sentiment_drops() -> None:
    p = SentimentAggregatorV1(warmup_ticks=2)
    assert p.on_tick(_tick(0, sentiment="bullish")) == ()
    assert p.on_tick(_tick(1, sentiment=None)) == ()


def test_out_of_range_sentiment_drops() -> None:
    p = SentimentAggregatorV1(warmup_ticks=2)
    assert p.on_tick(_tick(0, sentiment=1.5)) == ()
    assert p.on_tick(_tick(1, sentiment=-1.5)) == ()


def test_nan_sentiment_drops() -> None:
    p = SentimentAggregatorV1(warmup_ticks=2)
    assert p.on_tick(_tick(0, sentiment=float("nan"))) == ()
    assert p.on_tick(_tick(1, sentiment=float("inf"))) == ()
    assert p.on_tick(_tick(2, sentiment=float("-inf"))) == ()


def test_replay_determinism() -> None:
    seq = [_tick(i, sentiment=0.5 if i % 2 == 0 else 0.7) for i in range(10)]
    p1 = SentimentAggregatorV1(alpha=0.3, warmup_ticks=4)
    p2 = SentimentAggregatorV1(alpha=0.3, warmup_ticks=4)
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = SentimentAggregatorV1(
        alpha=0.5,
        warmup_ticks=4,
        sentiment_threshold=0.05,
        confidence_scale=10000.0,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, sentiment=0.9))
    assert out == ()


def test_ema_reverses_with_sustained_opposite_input() -> None:
    """Strong sustained bullish then bearish should flip the signal."""
    p = SentimentAggregatorV1(
        alpha=0.7,
        warmup_ticks=2,
        sentiment_threshold=0.2,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out_a: tuple = ()
    for i in range(4):
        out_a = p.on_tick(_tick(i, sentiment=0.9))
    assert len(out_a) == 1
    assert out_a[0].side is Side.BUY
    out_b: tuple = ()
    for i in range(4, 12):
        out_b = p.on_tick(_tick(i, sentiment=-0.9))
    assert len(out_b) == 1
    assert out_b[0].side is Side.SELL


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        SentimentAggregatorV1(alpha=0.0)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(alpha=1.5)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(warmup_ticks=0)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(sentiment_threshold=-0.1)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(sentiment_threshold=1.5)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(min_confidence=1.5)
    with pytest.raises(ValueError):
        SentimentAggregatorV1(meta_key="")


def test_check_self_reports_ok() -> None:
    p = SentimentAggregatorV1()
    status = p.check_self()
    assert status.state is HealthState.OK
    assert "sentiment_aggregator_v1" in status.detail


def test_lifecycle_default_active() -> None:
    p = SentimentAggregatorV1()
    assert p.lifecycle is PluginLifecycle.ACTIVE


def test_custom_meta_key_routes_correctly() -> None:
    p = SentimentAggregatorV1(
        meta_key="news_score",
        alpha=0.5,
        warmup_ticks=2,
        sentiment_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    # Default "sentiment" key is ignored; "news_score" is consumed.
    t1 = MarketTick(
        ts_ns=0,
        symbol="X",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta={"sentiment": 0.0, "news_score": 0.9},
    )
    t2 = MarketTick(
        ts_ns=1,
        symbol="X",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta={"news_score": 0.9},
    )
    p.on_tick(t1)
    out = p.on_tick(t2)
    assert len(out) == 1
    assert out[0].side is Side.BUY
