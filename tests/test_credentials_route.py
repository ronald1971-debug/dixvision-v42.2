"""HTTP tests for /api/credentials/status and /credentials.

Read-only endpoint: zero side effects, zero outbound network calls.
The test asserts the matrix shape, summary tally, and that no
secret values leak into the JSON response (only env var *names*).
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


def test_credentials_page_is_served(client) -> None:
    r = client.get("/credentials")
    assert r.status_code == 200
    assert "Credentials" in r.text or "credentials" in r.text
    assert "/api/credentials/status" in r.text or "credentials.js" in r.text


def test_credentials_status_envelope(client, monkeypatch) -> None:
    # Clear any host-leaked env vars so the test outcome is hermetic.
    for name in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEVIN_API_KEY",
        "REUTERS_API_KEY",
        "X_BEARER_TOKEN",
        "GLASSNODE_API_KEY",
        "DUNE_API_KEY",
        "FRED_API_KEY",
        "BLS_API_KEY",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    r = client.get("/api/credentials/status")
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data and "items" in data

    # Every item is missing once env vars are scrubbed.
    summary = data["summary"]
    assert summary["total"] == len(data["items"])
    assert summary["missing"] == summary["total"]
    assert summary["present"] == 0
    assert summary["partial"] == 0


def test_credentials_status_reflects_present_var(
    client, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test")
    r = client.get("/api/credentials/status")
    assert r.status_code == 200
    data = r.json()
    openai = next(
        i for i in data["items"] if i["provider"] == "openai"
    )
    assert openai["state"] == "present"
    assert openai["missing_env_vars"] == []
    assert "OPENAI_API_KEY" in openai["env_vars"]


def test_credentials_status_does_not_leak_values(
    client, monkeypatch
) -> None:
    """The JSON response must never contain the secret value itself."""

    monkeypatch.setenv("OPENAI_API_KEY", "sk-this-must-not-leak")
    r = client.get("/api/credentials/status")
    assert r.status_code == 200
    body = r.text
    assert "sk-this-must-not-leak" not in body


def test_credentials_status_item_shape(client) -> None:
    r = client.get("/api/credentials/status")
    assert r.status_code == 200
    data = r.json()
    if not data["items"]:
        pytest.skip("no auth: required rows in registry")
    item = data["items"][0]
    expected_keys = {
        "source_id",
        "source_name",
        "category",
        "provider",
        "env_vars",
        "env_vars_present",
        "missing_env_vars",
        "signup_url",
        "free_tier",
        "notes",
        "state",
    }
    assert expected_keys.issubset(item.keys())
    assert item["state"] in {"present", "partial", "missing"}
    assert isinstance(item["env_vars"], list)
    assert isinstance(item["env_vars_present"], list)
    assert len(item["env_vars"]) == len(item["env_vars_present"])
