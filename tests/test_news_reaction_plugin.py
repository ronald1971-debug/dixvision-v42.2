"""Tests for the IND-L12 news_reaction v1 plugin (Indira plugin #11)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.news_reaction import NewsReactionV1


def _tick(ts: int, *, magnitude: object | None = None) -> MarketTick:
    meta: dict[str, object] = {}
    if magnitude is not None:
        meta["news_event_magnitude"] = magnitude
    return MarketTick(
        ts_ns=ts,
        symbol="SOL-USD",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta=meta,
    )


def test_no_news_no_signal() -> None:
    p = NewsReactionV1(impulse_threshold=0.1)
    out: tuple = ()
    for i in range(5):
        out = p.on_tick(_tick(i))
    assert out == ()


def test_bullish_news_emits_buy_immediately() -> None:
    p = NewsReactionV1(
        decay_rate=0.9,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out = p.on_tick(_tick(0, magnitude=0.8))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.plugin_chain == ("news_reaction_v1",)
    assert float(sig.meta["impulse"]) == pytest.approx(0.8)


def test_bearish_news_emits_sell_immediately() -> None:
    p = NewsReactionV1(
        decay_rate=0.9,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out = p.on_tick(_tick(0, magnitude=-0.8))
    assert len(out) == 1
    assert out[0].side is Side.SELL


def test_impulse_decays_geometrically() -> None:
    p = NewsReactionV1(
        decay_rate=0.5,
        impulse_threshold=0.05,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out0 = p.on_tick(_tick(0, magnitude=0.8))
    out1 = p.on_tick(_tick(1))
    out2 = p.on_tick(_tick(2))
    out3 = p.on_tick(_tick(3))
    assert len(out0) == 1
    assert len(out1) == 1
    assert len(out2) == 1
    # Impulse halves each tick: 0.8 -> 0.4 -> 0.2 -> 0.1
    assert float(out1[0].meta["impulse"]) == pytest.approx(0.4)
    assert float(out2[0].meta["impulse"]) == pytest.approx(0.2)
    assert float(out3[0].meta["impulse"]) == pytest.approx(0.1)


def test_decay_below_threshold_silences() -> None:
    p = NewsReactionV1(
        decay_rate=0.5,
        impulse_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    p.on_tick(_tick(0, magnitude=0.8))
    p.on_tick(_tick(1))  # 0.4
    out = p.on_tick(_tick(2))  # 0.2 < 0.3
    assert out == ()


def test_news_overwrites_prior_impulse() -> None:
    p = NewsReactionV1(
        decay_rate=0.9,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    p.on_tick(_tick(0, magnitude=0.8))
    out = p.on_tick(_tick(1, magnitude=-0.5))
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert float(out[0].meta["impulse"]) == pytest.approx(-0.5)


def test_below_threshold_news_no_emit() -> None:
    p = NewsReactionV1(
        decay_rate=0.9,
        impulse_threshold=0.5,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out = p.on_tick(_tick(0, magnitude=0.3))
    assert out == ()


def test_zero_magnitude_no_emit() -> None:
    p = NewsReactionV1(impulse_threshold=0.0)
    out = p.on_tick(_tick(0, magnitude=0.0))
    # With threshold 0 and impulse exactly 0, no signal (impulse > 0 fails).
    assert out == ()


def test_non_numeric_magnitude_dropped() -> None:
    p = NewsReactionV1(impulse_threshold=0.1)
    # Non-numeric magnitude dropped silently; no impulse latched.
    out = p.on_tick(_tick(0, magnitude="hawkish"))
    assert out == ()


def test_out_of_range_magnitude_dropped() -> None:
    p = NewsReactionV1(impulse_threshold=0.1)
    out = p.on_tick(_tick(0, magnitude=1.5))
    assert out == ()
    out = p.on_tick(_tick(1, magnitude=-2.0))
    assert out == ()


def test_nan_inf_magnitude_dropped() -> None:
    p = NewsReactionV1(impulse_threshold=0.1)
    assert p.on_tick(_tick(0, magnitude=float("nan"))) == ()
    assert p.on_tick(_tick(1, magnitude=float("inf"))) == ()
    assert p.on_tick(_tick(2, magnitude=float("-inf"))) == ()


def test_replay_determinism() -> None:
    seq: list[MarketTick] = []
    for i in range(10):
        m = 0.7 if i == 2 else (-0.4 if i == 6 else None)
        seq.append(_tick(i, magnitude=m))
    p1 = NewsReactionV1(decay_rate=0.7, impulse_threshold=0.1)
    p2 = NewsReactionV1(decay_rate=0.7, impulse_threshold=0.1)
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = NewsReactionV1(
        decay_rate=0.9,
        impulse_threshold=0.05,
        confidence_scale=1e6,
        min_confidence=0.5,
    )
    out = p.on_tick(_tick(0, magnitude=0.8))
    assert out == ()


def test_no_decay_persists_forever() -> None:
    p = NewsReactionV1(
        decay_rate=1.0,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    p.on_tick(_tick(0, magnitude=0.5))
    out: tuple = ()
    for i in range(1, 20):
        out = p.on_tick(_tick(i))
    assert len(out) == 1
    assert float(out[0].meta["impulse"]) == pytest.approx(0.5)


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        NewsReactionV1(meta_key="")
    with pytest.raises(ValueError):
        NewsReactionV1(decay_rate=0.0)
    with pytest.raises(ValueError):
        NewsReactionV1(decay_rate=1.5)
    with pytest.raises(ValueError):
        NewsReactionV1(impulse_threshold=-0.1)
    with pytest.raises(ValueError):
        NewsReactionV1(impulse_threshold=1.5)
    with pytest.raises(ValueError):
        NewsReactionV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        NewsReactionV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        NewsReactionV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    status = NewsReactionV1().check_self()
    assert status.state is HealthState.OK
    assert "news_reaction_v1" in status.detail


def test_lifecycle_default_active() -> None:
    assert NewsReactionV1().lifecycle is PluginLifecycle.ACTIVE


def test_custom_meta_key_routes_correctly() -> None:
    p = NewsReactionV1(
        meta_key="regulatory_shock",
        decay_rate=0.9,
        impulse_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    t = MarketTick(
        ts_ns=0,
        symbol="X",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta={"news_event_magnitude": 0.9, "regulatory_shock": -0.6},
    )
    out = p.on_tick(t)
    assert len(out) == 1
    assert out[0].side is Side.SELL
