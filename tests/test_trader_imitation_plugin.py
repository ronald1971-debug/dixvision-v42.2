"""Tests for the IND-L11 trader_imitation v1 plugin (Indira plugin #10)."""

from __future__ import annotations

import pytest

from core.contracts.engine import HealthState, PluginLifecycle
from core.contracts.events import Side
from core.contracts.market import MarketTick
from intelligence_engine.plugins.trader_imitation import TraderImitationV1


def _tick(ts: int, *, intents: object | None) -> MarketTick:
    meta: dict[str, object] = {}
    if intents is not None:
        meta["leader_intents"] = intents
    return MarketTick(
        ts_ns=ts,
        symbol="ETH-USD",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume=1.0,
        meta=meta,
    )


def test_no_signal_until_window_full() -> None:
    p = TraderImitationV1(
        window_size=4,
        min_leaders=2,
        consensus_threshold=0.0,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(3):
        out = p.on_tick(_tick(i, intents=(0.9, 0.8)))
    assert out == ()


def test_sustained_bullish_consensus_emits_buy() -> None:
    p = TraderImitationV1(
        window_size=4,
        min_leaders=2,
        consensus_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, intents=(0.8, 0.7, 0.9)))
    assert len(out) == 1
    sig = out[0]
    assert sig.side is Side.BUY
    assert sig.plugin_chain == ("trader_imitation_v1",)
    assert float(sig.meta["rolling_consensus"]) > 0.3
    assert sig.meta["n_leaders"] == "3"


def test_sustained_bearish_consensus_emits_sell() -> None:
    p = TraderImitationV1(
        window_size=4,
        min_leaders=2,
        consensus_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, intents=(-0.8, -0.7, -0.9)))
    assert len(out) == 1
    assert out[0].side is Side.SELL
    assert float(out[0].meta["rolling_consensus"]) < -0.3


def test_balanced_leaders_no_emit() -> None:
    p = TraderImitationV1(
        window_size=4,
        min_leaders=2,
        consensus_threshold=0.05,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, intents=(0.8, -0.8)))
    # tick consensus is 0 every tick; rolling = 0
    assert out == ()


def test_below_threshold_no_emit() -> None:
    p = TraderImitationV1(
        window_size=4,
        min_leaders=1,
        consensus_threshold=0.99,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, intents=(0.5, 0.5)))
    assert out == ()


def test_too_few_leaders_drops() -> None:
    p = TraderImitationV1(window_size=2, min_leaders=3)
    # only 2 leaders provided, min is 3 → drop
    assert p.on_tick(_tick(0, intents=(0.9, 0.9))) == ()


def test_missing_meta_drops() -> None:
    p = TraderImitationV1(window_size=2, min_leaders=1)
    assert p.on_tick(_tick(0, intents=None)) == ()


def test_string_meta_drops() -> None:
    p = TraderImitationV1(window_size=2, min_leaders=1)
    assert p.on_tick(_tick(0, intents="0.9,0.8")) == ()


def test_non_numeric_in_vector_drops_entire_tick() -> None:
    p = TraderImitationV1(
        window_size=2,
        min_leaders=1,
        consensus_threshold=0.0,
    )
    assert p.on_tick(_tick(0, intents=(0.9, "bullish"))) == ()


def test_out_of_range_leader_silently_skipped() -> None:
    """Out-of-range leaders are skipped but the tick still counts."""
    p = TraderImitationV1(
        window_size=2,
        min_leaders=1,
        consensus_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    # 0.9 valid, 1.5 skipped, 2.0 skipped → consensus = 0.9
    p.on_tick(_tick(0, intents=(0.9, 1.5, 2.0)))
    out = p.on_tick(_tick(1, intents=(0.7, -1.5)))
    # rolling = (0.9 + 0.7) / 2 = 0.8
    assert len(out) == 1
    assert out[0].side is Side.BUY


def test_nan_inf_leader_silently_skipped() -> None:
    p = TraderImitationV1(
        window_size=2,
        min_leaders=1,
        consensus_threshold=0.1,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    p.on_tick(
        _tick(0, intents=(0.9, float("nan"), float("inf"), float("-inf"))),
    )
    out = p.on_tick(_tick(1, intents=(0.5,)))
    # rolling = (0.9 + 0.5) / 2 = 0.7
    assert len(out) == 1
    assert out[0].side is Side.BUY


def test_replay_determinism() -> None:
    seq = [
        _tick(i, intents=(0.6 if i % 2 == 0 else 0.4, 0.5)) for i in range(10)
    ]
    p1 = TraderImitationV1(window_size=4, min_leaders=2)
    p2 = TraderImitationV1(window_size=4, min_leaders=2)
    out1 = [p1.on_tick(t) for t in seq]
    out2 = [p2.on_tick(t) for t in seq]
    assert out1 == out2


def test_min_confidence_floor() -> None:
    p = TraderImitationV1(
        window_size=4,
        min_leaders=1,
        consensus_threshold=0.05,
        confidence_scale=1e6,
        min_confidence=0.5,
    )
    out: tuple = ()
    for i in range(4):
        out = p.on_tick(_tick(i, intents=(0.9, 0.9)))
    assert out == ()


def test_window_rotation_flips_signal() -> None:
    p = TraderImitationV1(
        window_size=2,
        min_leaders=1,
        consensus_threshold=0.3,
        confidence_scale=1.0,
        min_confidence=0.0,
    )
    out_a: tuple = ()
    for i in range(2):
        out_a = p.on_tick(_tick(i, intents=(0.9, 0.8)))
    assert len(out_a) == 1
    assert out_a[0].side is Side.BUY
    out_b: tuple = ()
    for i in range(2, 6):
        out_b = p.on_tick(_tick(i, intents=(-0.9, -0.8)))
    assert len(out_b) == 1
    assert out_b[0].side is Side.SELL


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        TraderImitationV1(meta_key="")
    with pytest.raises(ValueError):
        TraderImitationV1(window_size=1)
    with pytest.raises(ValueError):
        TraderImitationV1(min_leaders=0)
    with pytest.raises(ValueError):
        TraderImitationV1(consensus_threshold=-0.1)
    with pytest.raises(ValueError):
        TraderImitationV1(consensus_threshold=1.5)
    with pytest.raises(ValueError):
        TraderImitationV1(confidence_scale=0.0)
    with pytest.raises(ValueError):
        TraderImitationV1(min_confidence=-0.1)
    with pytest.raises(ValueError):
        TraderImitationV1(min_confidence=1.5)


def test_check_self_reports_ok() -> None:
    status = TraderImitationV1().check_self()
    assert status.state is HealthState.OK
    assert "trader_imitation_v1" in status.detail


def test_lifecycle_default_active() -> None:
    assert TraderImitationV1().lifecycle is PluginLifecycle.ACTIVE
