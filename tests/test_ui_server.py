"""Phase E1 — UI harness server tests.

Exercises the FastAPI surface in ``ui/server.py`` with the in-process
``TestClient``. No network IO; deterministic.
"""

from __future__ import annotations

import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client():
    # Reset shared state between tests so order doesn't matter.
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    ui_server._TS_COUNTER["v"] = 0  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


def test_root_routes_to_live_dashboard(client):
    """``GET /`` must take operators to the live SPA when the React build
    is present (Wave-Live PR-4) and otherwise fall back to the Phase E1
    stub. The fallback path keeps CI jobs and Node-less environments
    from receiving a hard 404 at the front door.
    """
    r = client.get("/", follow_redirects=False)
    if ui_server._DASH2_INDEX.exists():
        assert r.status_code == 307, (
            "/ must redirect to /dash2/ when the React build is present"
        )
        assert r.headers["location"] == "/dash2/"
    else:
        assert r.status_code == 200
        assert "DIX VISION" in r.text


def test_health_returns_all_six_engines(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    engines = r.json()["engines"]
    assert set(engines) == {
        "intelligence",
        "execution",
        "system",
        "governance",
        "learning",
        "evolution",
    }
    for spec in engines.values():
        assert spec["state"] == "OK"


def test_registry_endpoints(client):
    r1 = client.get("/api/registry/engines")
    r2 = client.get("/api/registry/plugins")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert "engines" in r1.json()
    assert "plugins" in r2.json()


def test_signal_without_tick_returns_failed_execution(client):
    r = client.post(
        "/api/signal",
        json={"symbol": "EURUSD", "side": "BUY", "confidence": 0.7},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["executions"]) == 1
    assert body["executions"][0]["status"] == "FAILED"


def test_tick_then_signal_flows_to_filled(client):
    tick_resp = client.post(
        "/api/tick",
        json={"symbol": "BTCUSDT", "bid": 49990, "ask": 50010, "last": 50000},
    )
    assert tick_resp.status_code == 200

    sig_resp = client.post(
        "/api/signal",
        json={"symbol": "BTCUSDT", "side": "BUY", "confidence": 0.8},
    )
    assert sig_resp.status_code == 200
    body = sig_resp.json()
    exec_evt = body["executions"][0]
    assert exec_evt["status"] == "FILLED"
    assert exec_evt["side"] == "BUY"
    assert exec_evt["price"] == 50000.0


def test_tick_emits_shadow_signal_via_microstructure_plugin(client):
    # Phase E2: a tick whose last price moves the SHADOW microstructure
    # plugin's threshold should produce a tagged shadow signal that the
    # Execution Engine rejects (no live trade in shadow mode).
    r = client.post(
        "/api/tick",
        json={"symbol": "EURUSD", "bid": 99.99, "ask": 100.01, "last": 100.10},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["signals"]) == 1
    sig = body["signals"][0]
    assert sig["side"] == "BUY"
    assert sig["meta"]["shadow"] == "true"

    assert len(body["executions"]) == 1
    exe = body["executions"][0]
    assert exe["status"] == "REJECTED"
    assert exe["meta"]["reason"] == "shadow signal"
    assert exe["order_id"] == ""


def test_events_endpoint_records_recent(client):
    client.post(
        "/api/tick",
        json={"symbol": "X", "bid": 10, "ask": 11, "last": 10.5},
    )
    client.post(
        "/api/signal",
        json={"symbol": "X", "side": "BUY", "confidence": 0.5},
    )
    r = client.get("/api/events?limit=10")
    assert r.status_code == 200
    events = r.json()["events"]
    kinds = [e.get("kind") for e in events]
    assert "MARKET_TICK" in kinds
    assert "SIGNAL_EVENT" in kinds
    assert "EXECUTION_EVENT" in kinds


def test_signal_validates_side(client):
    r = client.post(
        "/api/signal",
        json={"symbol": "X", "side": "WRONG", "confidence": 0.5},
    )
    assert r.status_code == 422


def test_tick_validates_positive_prices(client):
    r = client.post(
        "/api/tick",
        json={"symbol": "X", "bid": -1, "ask": 1, "last": 1},
    )
    assert r.status_code == 422
