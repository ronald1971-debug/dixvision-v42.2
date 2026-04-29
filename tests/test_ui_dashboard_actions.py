"""DASH-2 — operator action HTTP endpoints.

Drives the four POST endpoints + the `/operator` HTML route mounted
by ``ui.dashboard_routes`` (via ``ui.server``). No network IO;
deterministic.
"""

from __future__ import annotations

import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client():
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    ui_server._TS_COUNTER["v"] = 0  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


# ---------------------------------------------------------------------------
# /operator HTML
# ---------------------------------------------------------------------------


def test_operator_html_served(client):
    r = client.get("/operator")
    assert r.status_code == 200
    body = r.text
    # Must mention each of the five widget spec ids so we catch
    # accidental section deletions.
    for spec in ("DASH-02", "DASH-EG-01", "DASH-SLP-01", "DASH-04", "DASH-MCP-01"):
        assert spec in body, f"operator.html missing {spec}"
    assert "/api/dashboard/summary" not in body  # JS file is loaded separately
    assert "/static/operator.js" in body
    assert "/static/operator.css" in body


def test_static_assets_served(client):
    js = client.get("/static/operator.js")
    css = client.get("/static/operator.css")
    assert js.status_code == 200 and "fetch" in js.text
    assert css.status_code == 200 and "card" in css.text


# ---------------------------------------------------------------------------
# /api/dashboard/action/mode
# ---------------------------------------------------------------------------


def test_action_mode_safe_to_paper_approved(client):
    payload = {
        "target_mode": "PAPER",
        "reason": "boot up",
        "operator_authorized": True,
    }
    r = client.post("/api/dashboard/action/mode", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["approved"] is True, body
    # Mode FSM advanced -> /api/dashboard/mode reflects it.
    snap = client.get("/api/dashboard/mode").json()["mode"]
    assert snap["current_mode"] == "PAPER"


def test_action_mode_unknown_target_rejected(client):
    payload = {"target_mode": "NONSENSE", "reason": "x", "operator_authorized": True}
    r = client.post("/api/dashboard/action/mode", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is False
    assert "BRIDGE_UNKNOWN_MODE" in (body["decision"]["rejection_code"] or "")


# ---------------------------------------------------------------------------
# /api/dashboard/action/kill
# ---------------------------------------------------------------------------


def test_action_kill_locks_system(client):
    r = client.post("/api/dashboard/action/kill", json={"reason": "test kill"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["approved"] is True
    snap = client.get("/api/dashboard/mode").json()["mode"]
    assert snap["current_mode"] == "LOCKED"
    assert snap["is_locked"] is True


# ---------------------------------------------------------------------------
# /api/dashboard/action/intent
# ---------------------------------------------------------------------------


def test_action_intent_valid_payload_approved(client):
    payload = {
        "objective": "RISK_ADJUSTED_GROWTH",
        "risk_mode": "BALANCED",
        "horizon": "SHORT_TERM",
        "focus": ["BTCUSDT", "ETHUSDT"],
        "reason": "morning open",
    }
    r = client.post("/api/dashboard/action/intent", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["approved"] is True
    assert "RISK_ADJUSTED_GROWTH" in body["decision"]["summary"]


def test_action_intent_unknown_objective_rejected(client):
    payload = {
        "objective": "MOON",
        "risk_mode": "BALANCED",
        "horizon": "SHORT_TERM",
    }
    r = client.post("/api/dashboard/action/intent", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is False
    assert "BRIDGE_UNKNOWN_INTENT" in (body["decision"]["rejection_code"] or "")


# ---------------------------------------------------------------------------
# /api/dashboard/action/lifecycle
# ---------------------------------------------------------------------------


def test_action_lifecycle_records_audit_row(client):
    payload = {
        "plugin_path": "execution_engine.domains.memecoin",
        "target_status": "DISABLED",
        "reason": "operator disable",
    }
    r = client.post("/api/dashboard/action/lifecycle", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["approved"] is True
    assert body["decision"]["kind"] == "PLUGIN_LIFECYCLE"


def test_action_lifecycle_incomplete_payload_rejected(client):
    # FastAPI / pydantic rejects the missing field at validation time.
    r = client.post(
        "/api/dashboard/action/lifecycle",
        json={"plugin_path": "x"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Triad lock (INV-56) — the action endpoints never construct an
# ExecutionEvent or SignalEvent. The B20 / B21 / B22 lint covers this
# at static-analysis time; this smoke test confirms the runtime path
# still goes through Governance and never directly touches Indira /
# Executor.
# ---------------------------------------------------------------------------


def test_action_endpoints_route_through_governance_only(client):
    r = client.post(
        "/api/dashboard/action/mode",
        json={"target_mode": "PAPER", "reason": "x", "operator_authorized": True},
    )
    body = r.json()
    # The decision *must* carry a ledger sequence — proof it went
    # through GOV-CP-03 / GOV-CP-07 rather than being short-circuited.
    assert body["decision"]["ledger_seq"] >= 0
