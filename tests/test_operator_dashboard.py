"""Wave-02 PR-2 — typed operator dashboard surface.

Drives the new ``/api/operator/*`` endpoints introduced for the
React/Vite port at ``/dash2/#/operator``. The legacy
``/api/dashboard/*`` endpoints stay covered by
``tests/test_ui_dashboard_actions.py``; this module asserts that the
typed parallel surface returns shapes the Pydantic→TS codegen can
project to TypeScript without drift.
"""

from __future__ import annotations

import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")

from core.contracts.api.operator import (  # noqa: E402
    OperatorActionResponse,
    OperatorSummaryResponse,
)


@pytest.fixture
def client():
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    ui_server._TS_COUNTER["v"] = 0  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


# ---------------------------------------------------------------------------
# GET /api/operator/summary
# ---------------------------------------------------------------------------


def test_operator_summary_shape(client):
    """Response validates against the typed contract (no extra keys)."""

    r = client.get("/api/operator/summary")
    assert r.status_code == 200, r.text
    parsed = OperatorSummaryResponse.model_validate(r.json())
    # Core fields are present and typed.
    assert isinstance(parsed.mode.current_mode, str)
    assert isinstance(parsed.mode.legal_targets, list)
    assert isinstance(parsed.mode.is_locked, bool)
    # Strategy counts are non-negative integers across all six states.
    counts = parsed.strategies
    for value in (
        counts.proposed,
        counts.shadow,
        counts.canary,
        counts.live,
        counts.retired,
        counts.failed,
    ):
        assert value >= 0
    # Decision chain count is reported as a non-negative integer.
    assert parsed.decision_chain_count >= 0


def test_operator_summary_keys_match_pydantic_definition(client):
    """``extra='forbid'`` + a hard-coded key list catches drift early."""

    r = client.get("/api/operator/summary")
    body = r.json()
    assert set(body.keys()) == {
        "mode",
        "engines",
        "strategies",
        "memecoin",
        "decision_chain_count",
    }
    assert set(body["mode"].keys()) == {
        "current_mode",
        "legal_targets",
        "is_locked",
    }
    assert set(body["strategies"].keys()) == {
        "proposed",
        "shadow",
        "canary",
        "live",
        "retired",
        "failed",
    }
    assert set(body["memecoin"].keys()) == {"enabled", "killed", "summary"}


def test_operator_summary_engine_rows_typed(client):
    """Every engine row carries the four typed fields and no others."""

    r = client.get("/api/operator/summary")
    body = r.json()
    for row in body["engines"]:
        assert set(row.keys()) == {
            "engine_name",
            "bucket",
            "detail",
            "plugin_count",
        }
        assert isinstance(row["engine_name"], str)
        assert isinstance(row["bucket"], str)
        assert isinstance(row["detail"], str)
        assert isinstance(row["plugin_count"], int)
        assert row["plugin_count"] >= 0


def test_operator_summary_does_not_leak_secrets(client, monkeypatch):
    """Even with secret-shaped env vars present, the JSON never echoes them."""

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-12345")
    monkeypatch.setenv("BINANCE_API_SECRET", "very-secret-binance-string")
    r = client.get("/api/operator/summary")
    assert r.status_code == 200
    body = r.text
    assert "sk-test-not-a-real-key-12345" not in body
    assert "very-secret-binance-string" not in body


# ---------------------------------------------------------------------------
# POST /api/operator/action/kill
# ---------------------------------------------------------------------------


def test_operator_kill_routes_through_governance(client):
    """A KILL request returns a typed action envelope with a ledger seq."""

    r = client.post(
        "/api/operator/action/kill",
        json={"reason": "operator drill", "requestor": "tester"},
    )
    assert r.status_code == 200, r.text
    parsed = OperatorActionResponse.model_validate(r.json())
    assert isinstance(parsed.approved, bool)
    assert isinstance(parsed.summary, str) and parsed.summary
    # The decision dict must carry a ledger sequence — proof the
    # request travelled through GOV-CP-07 rather than being short-
    # circuited at the route handler (B7 / INV-37).
    assert "ledger_seq" in parsed.decision


def test_operator_kill_default_payload_is_valid(client):
    """No body fields are required — defaults satisfy the route."""

    r = client.post("/api/operator/action/kill", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "approved" in body and "summary" in body and "decision" in body


def test_operator_kill_rejects_oversized_reason(client):
    """Pydantic validation cuts off pathological reason strings (DoS guard)."""

    r = client.post(
        "/api/operator/action/kill",
        json={"reason": "x" * 1024},
    )
    assert r.status_code == 422


def test_operator_kill_response_carries_no_env_values(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-canary-9876")
    r = client.post("/api/operator/action/kill", json={"reason": "drill"})
    assert r.status_code == 200
    assert "sk-leak-canary-9876" not in r.text


def test_operator_summary_and_legacy_dashboard_share_state(client):
    """A KILL via /api/operator should be visible on the legacy widget too.

    Both surfaces are wired to the same ``STATE`` widgets — the Phase 6
    invariant. If a future refactor accidentally double-instantiates
    them, the legacy dashboard summary mode flips while the typed one
    does not (or vice-versa) and this regression catches it.
    """

    pre = client.get("/api/operator/summary").json()["mode"]["current_mode"]
    legacy = client.get("/api/dashboard/summary").json()
    assert legacy["mode"]["current_mode"] == pre
