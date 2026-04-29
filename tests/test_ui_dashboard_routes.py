"""DASH-1 — read-only widget HTTP endpoints.

Drives the FastAPI surface in :mod:`ui.dashboard_routes` (mounted by
``ui.server``) with the in-process ``TestClient``. No network IO;
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
# /api/dashboard/mode
# ---------------------------------------------------------------------------


def test_mode_endpoint_returns_safe_default(client):
    r = client.get("/api/dashboard/mode")
    assert r.status_code == 200
    body = r.json()["mode"]
    assert body["current_mode"] == "SAFE"
    assert body["is_locked"] is False
    # SAFE has at least one legal forward edge (PAPER).
    assert "PAPER" in body["legal_targets"]


# ---------------------------------------------------------------------------
# /api/dashboard/engines
# ---------------------------------------------------------------------------


def test_engines_endpoint_returns_six_rows(client):
    r = client.get("/api/dashboard/engines")
    assert r.status_code == 200
    rows = r.json()["engines"]
    names = {row["engine_name"] for row in rows}
    assert names == {
        "intelligence",
        "execution",
        "system",
        "governance",
        "learning",
        "evolution",
    }
    for row in rows:
        # Operator-facing buckets per Build Compiler Spec §6.
        assert row["bucket"] in {"alive", "degraded", "halted", "offline"}


# ---------------------------------------------------------------------------
# /api/dashboard/strategies
# ---------------------------------------------------------------------------


def test_strategies_endpoint_returns_lifecycle_columns(client):
    r = client.get("/api/dashboard/strategies")
    assert r.status_code == 200
    columns = r.json()["strategies"]
    # Canonical FSM column order (DASH-SLP-01).
    assert list(columns.keys()) == [
        "PROPOSED",
        "SHADOW",
        "CANARY",
        "LIVE",
        "RETIRED",
        "FAILED",
    ]
    # Empty by default — no strategies registered yet.
    for rows in columns.values():
        assert rows == []


# ---------------------------------------------------------------------------
# /api/dashboard/decisions
# ---------------------------------------------------------------------------


def test_decisions_endpoint_starts_empty(client):
    r = client.get("/api/dashboard/decisions")
    assert r.status_code == 200
    assert r.json()["chains"] == []


def test_decisions_endpoint_groups_after_signal(client):
    # Submit one signal to seed the in-process ledger reader via
    # ``_State.record``; the decision-trace panel should group it
    # under the symbol.
    payload = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "confidence": 0.8,
    }
    sub = client.post("/api/signal", json=payload)
    assert sub.status_code == 200, sub.text

    r = client.get("/api/dashboard/decisions")
    assert r.status_code == 200
    chains = r.json()["chains"]
    assert chains, "expected at least one decision chain after a signal"
    assert any(chain["symbol"] == "BTCUSDT" for chain in chains)


def test_decisions_endpoint_respects_limit(client):
    r = client.get("/api/dashboard/decisions?limit=10")
    assert r.status_code == 200
    # Limit out of bounds → 422 (Query validator).
    bad = client.get("/api/dashboard/decisions?limit=0")
    assert bad.status_code == 422


# ---------------------------------------------------------------------------
# /api/dashboard/memecoin
# ---------------------------------------------------------------------------


def test_memecoin_endpoint_returns_default_disabled(client):
    r = client.get("/api/dashboard/memecoin")
    assert r.status_code == 200
    body = r.json()["memecoin"]
    assert body["enabled"] is False
    assert body["killed"] is False
    assert "disabled" in body["summary"].lower()


# ---------------------------------------------------------------------------
# /api/dashboard/summary
# ---------------------------------------------------------------------------


def test_summary_endpoint_aggregates_all_widgets(client):
    r = client.get("/api/dashboard/summary")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {
        "mode",
        "engines",
        "strategies",
        "memecoin",
        "chains",
    }
    assert body["mode"]["current_mode"] == "SAFE"
    assert isinstance(body["engines"], list)
    assert isinstance(body["strategies"], dict)


# ---------------------------------------------------------------------------
# Determinism — same inputs → same JSON
# ---------------------------------------------------------------------------


def test_summary_endpoint_is_deterministic(client):
    r1 = client.get("/api/dashboard/summary")
    r2 = client.get("/api/dashboard/summary")
    assert r1.status_code == 200 and r2.status_code == 200
    # Engine status grid embeds an `is_terminal` style detail, but
    # nothing time-keyed — repeated calls without state mutation must
    # return identical bodies.
    assert r1.json() == r2.json()
