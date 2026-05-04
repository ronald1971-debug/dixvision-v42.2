"""AUDIT-P1.5 — regression tests for the three previously-missing
backend HTTP routes that the dashboards already call:

* ``POST /api/operator/audit`` — autonomy-mode flips and SL/TP commits
  fire-and-forget against this URL. Without the route the dashboard
  silently 404s and the authority ledger never sees the operator's
  settings transition. The route writes one
  ``OPERATOR_SETTINGS_CHANGED`` row to the SQLite-backed authority
  ledger.
* ``GET /api/feeds/memecoin/summary`` — typed memecoin subsystem
  summary (DEXtools-style ``HoldersPanel`` / ``RugScoreCard``).
  Returns the same in-process status the legacy ``MemecoinControlPanel``
  exposes.
* ``GET /api/wallet/info`` — DISCONNECTED wallet stub that the
  ``WalletInfoPage`` reads. Reports an explicit reason instead of a
  generic null so the UI can render an actionable message.

The tests drive the in-process FastAPI ``TestClient`` so no network IO
or UI rendering is involved. Each route is exercised end-to-end and
the ledger side-effect on the audit route is verified by reading
``STATE.governance.ledger`` rows after the POST.
"""

from __future__ import annotations

import importlib
import json

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client():
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


# ---------------------------------------------------------------------------
# POST /api/operator/audit
# ---------------------------------------------------------------------------


def test_operator_audit_writes_ledger_row(client):
    """An ``OPERATOR_SETTINGS_CHANGED`` row lands in the ledger."""

    rows_before = len(ui_server.STATE.governance.ledger)
    payload = {
        "kind": "OPERATOR/SETTINGS_CHANGED",
        "setting": "autonomy_mode",
        "previous": "MANUAL",
        "next": "FULL_AUTO",
        "timestamp_iso": "2026-05-04T06:30:00.000Z",
    }
    r = client.post("/api/operator/audit", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["kind"] == "OPERATOR/SETTINGS_CHANGED"
    assert body["seq"] == rows_before  # zero-indexed seq matches prior row count
    # ``persisted`` reflects whether a SQLite file is mounted; tests
    # run with ``DIXVISION_PERMIT_EPHEMERAL_LEDGER=1`` so it's False.
    assert isinstance(body["persisted"], bool)
    assert len(ui_server.STATE.governance.ledger) == rows_before + 1


def test_operator_audit_serialises_object_payloads(client):
    """SL/TP-style commits send dict ``previous`` / ``next``; the
    handler JSON-encodes them so the ledger row stays
    ``Mapping[str, str]``-shaped (the writer's contract)."""

    sl_tp_draft = {
        "form": "spot",
        "primitives": [
            {"kind": "hard", "price_pct": -2.0},
            {"kind": "trailing", "trail_pct": 1.5},
        ],
    }
    payload = {
        "kind": "OPERATOR/SETTINGS_CHANGED",
        "setting": "sl_tp/spot",
        "previous": None,
        "next": sl_tp_draft,
        "autonomy_mode": "MANUAL",
        "timestamp_iso": "2026-05-04T06:31:00.000Z",
    }
    r = client.post("/api/operator/audit", json=payload)
    assert r.status_code == 200, r.text
    rows = ui_server.STATE.governance.ledger.read()
    last = rows[-1]
    assert last.kind == "OPERATOR/SETTINGS_CHANGED"
    # next_json must round-trip back to the original draft dict.
    decoded = json.loads(last.payload["next_json"])
    assert decoded == sl_tp_draft
    assert json.loads(last.payload["previous_json"]) is None
    assert last.payload["setting"] == "sl_tp/spot"
    assert last.payload["autonomy_mode"] == "MANUAL"


def test_operator_audit_rejects_missing_kind(client):
    """``kind`` is required (non-empty)."""

    r = client.post(
        "/api/operator/audit",
        json={"setting": "autonomy_mode", "next": "FULL_AUTO"},
    )
    assert r.status_code == 422


def test_operator_audit_rejects_extra_fields(client):
    """``extra='forbid'`` keeps the audit surface tight."""

    r = client.post(
        "/api/operator/audit",
        json={
            "kind": "OPERATOR/SETTINGS_CHANGED",
            "setting": "autonomy_mode",
            "next": "FULL_AUTO",
            "rogue_field": "bypass_attempt",
        },
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/feeds/memecoin/summary
# ---------------------------------------------------------------------------


def test_feeds_memecoin_summary_default_disabled(client):
    """At boot the memecoin subsystem is disabled — the typed alias
    must reflect that and match the legacy ``/api/dashboard/memecoin``
    payload shape."""

    r = client.get("/api/feeds/memecoin/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"enabled", "killed", "summary"}
    assert body["enabled"] is False
    assert body["killed"] is False
    assert "disabled" in body["summary"].lower()


def test_feeds_memecoin_summary_matches_legacy(client):
    """The new typed alias and the legacy dashboard route must agree
    on the underlying widget snapshot — they read the same in-process
    status object."""

    legacy = client.get("/api/dashboard/memecoin").json()["memecoin"]
    typed = client.get("/api/feeds/memecoin/summary").json()
    assert typed["enabled"] == legacy["enabled"]
    assert typed["killed"] == legacy["killed"]
    assert typed["summary"] == legacy["summary"]


# ---------------------------------------------------------------------------
# GET /api/wallet/info
# ---------------------------------------------------------------------------


def test_wallet_info_disconnected_stub(client):
    """Until real wallet credentials are wired the route reports
    DISCONNECTED with an explicit reason."""

    r = client.get("/api/wallet/info")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"connected", "chain", "address", "reason"}
    assert body["connected"] is False
    assert body["chain"] == ""
    assert body["address"] == ""
    assert "credentials" in body["reason"].lower()


# ---------------------------------------------------------------------------
# Audit invariant: every route in this PR is registered on the live app
# ---------------------------------------------------------------------------


def test_routes_are_registered_on_the_fastapi_app():
    """Defensive assertion — even if ``_State`` reconstruction breaks,
    the routes themselves must be wired so the dashboard's existing
    fire-and-forget POSTs no longer 404."""

    paths = {route.path for route in ui_server.app.routes}
    assert "/api/operator/audit" in paths
    assert "/api/feeds/memecoin/summary" in paths
    assert "/api/wallet/info" in paths
