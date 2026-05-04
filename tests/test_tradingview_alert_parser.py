"""Paper-S4 — TradingView Pine alert parser regression tests.

Pins the safe-coercion behaviour of
:func:`ui.feeds.tradingview_alert.parse_tradingview_alert_payload`.

The parser must:

* Be pure (INV-15) — two replays of the same input + ``ts_ns``
  produce byte-identical :class:`SignalEvent`.
* Never raise on malformed input — return ``None`` so the webhook
  receiver answers ``{"accepted": False, ...}`` without TradingView
  retrying on a Pine-side authoring bug.
* Stamp every accepted signal with
  :attr:`SignalTrust.EXTERNAL_LOW` and
  ``signal_source = TRADINGVIEW_ALERT_SOURCE_FEED`` so the
  governance gate (Paper-S5/S6) can clamp ``confidence`` to the cap
  registered in ``registry/external_signal_trust.yaml``.
* Tolerate the most common Pine alert shapes (lowercase actions,
  ``long``/``short`` aliases, ``ticker`` <-> ``symbol``).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.events import Side, SignalEvent
from core.contracts.signal_trust import SignalTrust
from ui.feeds.tradingview_alert import (
    TRADINGVIEW_ALERT_PRODUCED_BY_ENGINE,
    TRADINGVIEW_ALERT_SOURCE_FEED,
    parse_tradingview_alert_payload,
)


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1,
        "ticker": "BTCUSDT",
        "side": "BUY",
        "confidence": 0.62,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


def test_canonical_envelope_returns_external_low_signal() -> None:
    result = parse_tradingview_alert_payload(_payload(), ts_ns=1_700_000_000_000_000_000)
    assert result is not None
    sig = result.signal
    assert isinstance(sig, SignalEvent)
    assert sig.symbol == "BTCUSDT"
    assert sig.side is Side.BUY
    assert sig.confidence == pytest.approx(0.62)
    assert sig.signal_trust is SignalTrust.EXTERNAL_LOW
    assert sig.signal_source == TRADINGVIEW_ALERT_SOURCE_FEED
    assert sig.produced_by_engine == TRADINGVIEW_ALERT_PRODUCED_BY_ENGINE
    assert sig.plugin_chain == ("tradingview_alert",)
    assert sig.ts_ns == 1_700_000_000_000_000_000


def test_parser_is_pure_inv15() -> None:
    """Two parses of the same input + ts_ns produce byte-identical output."""

    payload = _payload(qty="0.05", strategy="pine_breakout_v3", comment="RSI<30")
    a = parse_tradingview_alert_payload(payload, ts_ns=1234)
    b = parse_tradingview_alert_payload(payload, ts_ns=1234)
    assert a is not None and b is not None
    assert a.signal == b.signal
    assert dict(a.audit_meta) == dict(b.audit_meta)


def test_audit_meta_propagates_qty_strategy_comment() -> None:
    payload = _payload(qty=0.05, strategy="pine_breakout_v3", comment="RSI < 30")
    result = parse_tradingview_alert_payload(payload, ts_ns=42)
    assert result is not None
    audit = dict(result.audit_meta)
    assert audit["qty"] == "0.05"
    assert audit["strategy"] == "pine_breakout_v3"
    assert audit["comment"] == "RSI < 30"


def test_long_alias_maps_to_buy_short_alias_maps_to_sell() -> None:
    long_r = parse_tradingview_alert_payload(_payload(side="long"), ts_ns=1)
    short_r = parse_tradingview_alert_payload(_payload(side="SHORT"), ts_ns=2)
    assert long_r is not None and long_r.signal.side is Side.BUY
    assert short_r is not None and short_r.signal.side is Side.SELL


def test_lowercase_action_alias_uppercased() -> None:
    """TradingView's ``{{strategy.order.action}}`` emits 'buy'/'sell'."""

    r = parse_tradingview_alert_payload(_payload(side="sell"), ts_ns=1)
    assert r is not None
    assert r.signal.side is Side.SELL


def test_symbol_key_accepted_in_place_of_ticker() -> None:
    payload = {"version": 1, "symbol": "ETHUSDT", "side": "BUY"}
    r = parse_tradingview_alert_payload(payload, ts_ns=1)
    assert r is not None
    assert r.signal.symbol == "ETHUSDT"


def test_action_key_accepted_in_place_of_side() -> None:
    payload = {"version": 1, "ticker": "BTCUSDT", "action": "buy"}
    r = parse_tradingview_alert_payload(payload, ts_ns=1)
    assert r is not None
    assert r.signal.side is Side.BUY


def test_confidence_default_when_absent() -> None:
    r = parse_tradingview_alert_payload({"ticker": "BTC", "side": "BUY"}, ts_ns=1)
    assert r is not None
    assert r.signal.confidence == pytest.approx(0.5)


def test_confidence_clamped_to_unit_interval() -> None:
    high = parse_tradingview_alert_payload(_payload(confidence=2.5), ts_ns=1)
    low = parse_tradingview_alert_payload(_payload(confidence=-1.0), ts_ns=2)
    assert high is not None and high.signal.confidence == 1.0
    assert low is not None and low.signal.confidence == 0.0


def test_version_omitted_is_accepted() -> None:
    r = parse_tradingview_alert_payload(
        {"ticker": "BTCUSDT", "side": "BUY"}, ts_ns=1
    )
    assert r is not None


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_none_payload_returns_none() -> None:
    assert parse_tradingview_alert_payload(None, ts_ns=1) is None


def test_non_mapping_payload_returns_none() -> None:
    assert parse_tradingview_alert_payload("not a dict", ts_ns=1) is None  # type: ignore[arg-type]
    assert parse_tradingview_alert_payload(["not", "a", "dict"], ts_ns=1) is None  # type: ignore[arg-type]


def test_unknown_version_returns_none() -> None:
    assert parse_tradingview_alert_payload(_payload(version=2), ts_ns=1) is None
    assert parse_tradingview_alert_payload(_payload(version="abc"), ts_ns=1) is None


def test_missing_ticker_returns_none() -> None:
    assert parse_tradingview_alert_payload({"version": 1, "side": "BUY"}, ts_ns=1) is None


def test_blank_ticker_returns_none() -> None:
    assert parse_tradingview_alert_payload(_payload(ticker="   "), ts_ns=1) is None


def test_missing_side_returns_none() -> None:
    payload = {"version": 1, "ticker": "BTCUSDT"}
    assert parse_tradingview_alert_payload(payload, ts_ns=1) is None


def test_unknown_side_returns_none() -> None:
    assert parse_tradingview_alert_payload(_payload(side="FLAT"), ts_ns=1) is None


def test_non_string_ticker_returns_none() -> None:
    assert parse_tradingview_alert_payload(_payload(ticker=42), ts_ns=1) is None
