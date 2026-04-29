"""HTTP route tests for Dashboard-2026 wave-01.

Covers:

* ``GET /api/ai/providers`` — registry-driven AI provider list, with
  optional ``task`` query param filtering.
* ``GET /indira-chat`` / ``GET /dyon-chat`` / ``GET /forms-grid`` —
  vanilla skeleton HTML pages.

These pages are deliberately registry-driven; the static HTML never
names a specific AI vendor (authority_lint rule B23 enforces that
separately).
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


def test_indira_chat_page_is_served(client) -> None:
    r = client.get("/indira-chat")
    assert r.status_code == 200
    assert "Indira Chat" in r.text
    # The page must reference the provider endpoint.
    assert "/api/ai/providers" in r.text or "chat_widget.js" in r.text


def test_dyon_chat_page_is_served(client) -> None:
    r = client.get("/dyon-chat")
    assert r.status_code == 200
    assert "Dyon Chat" in r.text


def test_forms_grid_page_is_served(client) -> None:
    r = client.get("/forms-grid")
    assert r.status_code == 200
    text = r.text
    # Per-form scaffold must include all six domains.
    for domain in (
        "spot",
        "perps",
        "forex",
        "stocks",
        "defi",
        "memecoin",
    ):
        assert f'data-domain="{domain}"' in text
    # Memecoin must be tagged isolated (W1).
    assert 'class="form-card isolated"' in text


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


def test_static_chat_widget_js_has_no_vendor_strings(client) -> None:
    """Defence-in-depth — the static asset itself must be clean.

    authority_lint B23 enforces this at lint time; this test is the
    runtime verification that the file actually served by FastAPI is
    the registry-driven version (not a stale build artifact).
    """

    r = client.get("/static/chat_widget.js")
    assert r.status_code == 200
    body = r.text.lower()
    for token in (
        "openai",
        "gemini",
        "grok",
        "deepseek",
        "anthropic",
        "claude",
    ):
        assert token not in body, (
            f"served chat_widget.js contains forbidden token {token!r}"
        )
