"""Paper-S4 — ``POST /api/feeds/tradingview/alert`` regression tests.

Drives the in-process FastAPI ``TestClient`` so no network IO or UI
rendering is involved. The route is the *only* HTTP receiver for the
TradingView Pine-alert webhook; it must:

* Reject malformed payloads with HTTP 200 + ``{"accepted": false, ...}``
  (TradingView's webhook engine retries on 4xx/5xx, so a Pine-side
  authoring bug must not cause a retry storm).
* On success: feed the parsed :class:`SignalEvent` through the same
  Intelligence -> Execution pipeline as :func:`post_signal`, and stamp
  the resulting ledger ``feed.tradingview_alert`` row with
  :attr:`SignalTrust.EXTERNAL_LOW` and
  ``signal_source = TRADINGVIEW_ALERT_SOURCE_FEED``.
* Honour the optional caller-supplied ``ts_ns`` field; otherwise fall
  back to the ``wall_ns()`` TimeAuthority surrogate.
"""

from __future__ import annotations

import importlib

import pytest

from core.contracts.signal_trust import SignalTrust
from ui.feeds.tradingview_alert import TRADINGVIEW_ALERT_SOURCE_FEED

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client():
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


def test_accepted_envelope_runs_through_intelligence_execution(client) -> None:
    body = {
        "payload": {
            "version": 1,
            "ticker": "BTCUSDT",
            "side": "BUY",
            "confidence": 0.7,
            "qty": "0.05",
            "strategy": "pine_breakout_v3",
        },
        "ts_ns": 1_700_000_000_000_000_000,
    }
    r = client.post("/api/feeds/tradingview/alert", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["accepted"] is True
    assert payload["source_feed"] == TRADINGVIEW_ALERT_SOURCE_FEED
    sig = payload["signal"]
    assert sig["symbol"] == "BTCUSDT"
    assert sig["side"] == "BUY"
    assert sig["signal_trust"] == SignalTrust.EXTERNAL_LOW.value
    assert sig["signal_source"] == TRADINGVIEW_ALERT_SOURCE_FEED
    assert sig["ts_ns"] == 1_700_000_000_000_000_000
    assert isinstance(payload["executions"], list)


def test_malformed_payload_returns_accepted_false_http_200(client) -> None:
    body = {"payload": {"version": 1, "ticker": "BTCUSDT"}}
    r = client.post("/api/feeds/tradingview/alert", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["accepted"] is False
    assert payload["source_feed"] == TRADINGVIEW_ALERT_SOURCE_FEED
    assert "rejected" in payload["reason"].lower()


def test_unknown_version_returns_accepted_false(client) -> None:
    body = {"payload": {"version": 99, "ticker": "BTCUSDT", "side": "BUY"}}
    r = client.post("/api/feeds/tradingview/alert", json=body)
    assert r.status_code == 200
    assert r.json()["accepted"] is False


def test_non_mapping_payload_rejected_by_pydantic(client) -> None:
    """``payload`` is typed ``dict[str, Any]`` so a string trips
    Pydantic validation before the route body executes."""

    body = {"payload": "not a dict"}
    r = client.post("/api/feeds/tradingview/alert", json=body)
    assert r.status_code == 422, r.text


def test_missing_ts_ns_falls_back_to_wall_ns(client) -> None:
    body = {
        "payload": {
            "version": 1,
            "ticker": "BTCUSDT",
            "side": "BUY",
        }
    }
    r = client.post("/api/feeds/tradingview/alert", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["accepted"] is True
    assert payload["signal"]["ts_ns"] > 0


def test_long_alias_normalised_to_buy_side(client) -> None:
    body = {
        "payload": {"version": 1, "ticker": "ETHUSDT", "side": "long"},
        "ts_ns": 42,
    }
    r = client.post("/api/feeds/tradingview/alert", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["accepted"] is True
    assert payload["signal"]["side"] == "BUY"
