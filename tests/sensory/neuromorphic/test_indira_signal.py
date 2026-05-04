"""Unit tests for sensory.neuromorphic.indira_signal (NEUR-01)."""

from __future__ import annotations

import pytest

from sensory.neuromorphic.contracts import (
    POLARITY_LONG,
    POLARITY_NEUTRAL,
    POLARITY_SHORT,
)
from sensory.neuromorphic.indira_signal import (
    TradeSample,
    extract_pulse,
)


def _t(side: str, size: float) -> TradeSample:
    return TradeSample(side=side, size=size)


def test_pulse_long_dominant() -> None:
    p = extract_pulse(
        ts_ns=1,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[_t("BUY", 3.0), _t("BUY", 2.0), _t("SELL", 1.0)],
    )
    # signed_flow = 3+2-1 = 4; magnitude = 6; intensity = 4/6 ≈ 0.667
    assert p.polarity == POLARITY_LONG
    assert p.intensity == pytest.approx(4.0 / 6.0)
    assert p.sample_count == 3


def test_pulse_short_dominant() -> None:
    p = extract_pulse(
        ts_ns=2,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[_t("SELL", 5.0), _t("BUY", 1.0)],
    )
    assert p.polarity == POLARITY_SHORT
    assert p.intensity == pytest.approx(4.0 / 6.0)


def test_pulse_balanced_is_neutral() -> None:
    p = extract_pulse(
        ts_ns=3,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[_t("BUY", 2.0), _t("SELL", 2.0)],
    )
    assert p.polarity == POLARITY_NEUTRAL
    assert p.intensity == 0.0


def test_pulse_perfect_one_sided_saturates_intensity() -> None:
    p = extract_pulse(
        ts_ns=4,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[_t("BUY", 1.0), _t("BUY", 1.0)],
    )
    assert p.polarity == POLARITY_LONG
    assert p.intensity == 1.0


def test_pulse_empty_window_is_neutral_with_evidence_marker() -> None:
    """An empty window emits a NEUTRAL pulse with sample_count=1
    (clipped from 0 because PulseSignal rejects 0).

    Callers that want to distinguish "balanced flow" from "no flow"
    can pass ``evidence={"empty_window": "true"}`` themselves.
    """
    p = extract_pulse(
        ts_ns=5,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[],
        evidence={"empty_window": "true"},
    )
    assert p.polarity == POLARITY_NEUTRAL
    assert p.intensity == 0.0
    assert p.sample_count == 1
    assert p.evidence["empty_window"] == "true"


def test_pulse_is_deterministic_inv15() -> None:
    """INV-15: same input window → equal pulse."""
    window = [_t("BUY", 3.0), _t("SELL", 1.0)]
    p1 = extract_pulse(
        ts_ns=10,
        source="BINANCE",
        symbol="BTCUSDT",
        window=window,
    )
    p2 = extract_pulse(
        ts_ns=10,
        source="BINANCE",
        symbol="BTCUSDT",
        window=window,
    )
    assert p1 == p2


def test_pulse_evidence_is_copied() -> None:
    """Mutating the caller's mapping after the call must not affect
    the immutable PulseSignal.evidence (frozen-dataclass guarantee).
    """
    evidence = {"k": "v"}
    p = extract_pulse(
        ts_ns=11,
        source="BINANCE",
        symbol="BTCUSDT",
        window=[_t("BUY", 1.0)],
        evidence=evidence,
    )
    evidence["k"] = "tampered"
    assert p.evidence["k"] == "v"


def test_trade_sample_rejects_bad_side() -> None:
    with pytest.raises(ValueError, match="side"):
        TradeSample(side="HOLD", size=1.0)


@pytest.mark.parametrize(
    "bad",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_trade_sample_rejects_bad_size(bad: float) -> None:
    with pytest.raises(ValueError, match="size"):
        TradeSample(side="BUY", size=bad)
