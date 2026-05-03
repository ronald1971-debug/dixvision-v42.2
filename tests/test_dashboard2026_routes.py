"""HTTP route tests for Dashboard-2026 wave-01.

Covers:

* ``GET /api/ai/providers`` — registry-driven AI provider list, with
  optional ``task`` query param filtering.
* ``GET /indira-chat`` / ``GET /dyon-chat`` / ``GET /forms-grid`` —
  legacy paths that now 307-redirect to the React SPA at ``/dash2``
  (Wave-Live PR-2 retired the vanilla HTML skeletons).

The React SPA is deliberately registry-driven; authority_lint rule
B23 enforces "no hard-coded vendor names" separately.
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
    return TestClient(ui_server.app)


def test_indira_chat_redirects_to_dash2(client) -> None:
    """Wave-Live PR-2 — ``/indira-chat`` redirects to the React SPA chat."""

    r = client.get("/indira-chat", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/dash2/#/chat"


def test_dyon_chat_redirects_to_dash2(client) -> None:
    """Wave-Live PR-2 — ``/dyon-chat`` redirects to the React SPA chat."""

    r = client.get("/dyon-chat", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/dash2/#/chat"


def test_forms_grid_redirects_to_dash2(client) -> None:
    """Wave-Live PR-2 — ``/forms-grid`` redirects to the React SPA operator."""

    r = client.get("/forms-grid", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/dash2/#/operator"


def test_ai_providers_endpoint_returns_well_formed_envelope(client) -> None:
    r = client.get("/api/ai/providers")
    assert r.status_code == 200
    body = r.json()
    # Shape contract — always present, regardless of how many rows are
    # currently enabled. Today the shipping registry has every AI row
    # ``enabled: false`` (no credentials wired yet) so providers is
    # legitimately empty; the contract is the envelope, not the count.
    assert set(body.keys()) == {"task", "providers", "task_classes"}
    assert isinstance(body["providers"], list)
    assert isinstance(body["task_classes"], list)
    assert body["task"] is None
    # task_classes must enumerate every TaskClass value.
    assert "indira_reasoning" in body["task_classes"]
    assert "dyon_coding" in body["task_classes"]
    # If any provider is enabled, it must use the public projection
    # shape — no auth / enabled / schema fields leaked.
    for p in body["providers"]:
        assert set(p.keys()) == {
            "id",
            "name",
            "provider",
            "endpoint",
            "capabilities",
        }
        assert p["id"].startswith("SRC-AI-")


def test_ai_providers_endpoint_filters_by_task(client) -> None:
    r = client.get("/api/ai/providers?task=indira_reasoning")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "indira_reasoning"
    # Every returned provider must have the ``reasoning`` capability.
    for p in body["providers"]:
        assert "reasoning" in p["capabilities"]


def test_ai_providers_endpoint_rejects_unknown_task(client) -> None:
    r = client.get("/api/ai/providers?task=does_not_exist")
    assert r.status_code in {400, 422}


def test_legacy_chat_widget_js_is_gone(client) -> None:
    """Wave-Live PR-2 — the vanilla ``chat_widget.js`` was deleted.

    The React SPA at ``/dash2`` ships its own bundle from
    ``dashboard2026/dist/assets/``; the legacy file must not still be
    served from ``ui/static/`` or it would shadow the new one.
    """

    r = client.get("/static/chat_widget.js")
    assert r.status_code == 404
