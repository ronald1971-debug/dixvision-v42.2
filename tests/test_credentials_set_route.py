"""Tests for the ``POST /api/credentials/set`` route + writable flag."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")
storage = importlib.import_module("system_engine.credentials.storage")


@pytest.fixture
def writable_env(monkeypatch, tmp_path: Path):
    """Force the storage shim into 'local writable' mode.

    The Devin sandbox the tests run in *is* itself a Devin session
    (``/opt/.devin`` exists); without this fixture every write
    would correctly be refused. The fixture also redirects writes
    to a per-test ``tmp_path/.env`` so we never touch the repo.
    """

    monkeypatch.delenv("DEVIN_SESSION_ID", raising=False)
    monkeypatch.delenv("DEVIN_USER_ID", raising=False)
    monkeypatch.delenv("ENVRC", raising=False)
    monkeypatch.setattr(storage, "_has_devin_install_dir", lambda: False)
    monkeypatch.setattr(
        storage, "default_dotenv_path", lambda: tmp_path / ".env",
    )
    return tmp_path / ".env"


@pytest.fixture
def client():
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    ui_server._TS_COUNTER["v"] = 0  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


# ----- /api/credentials/status carries the writable flag ------------


def test_status_writable_true_outside_devin(client, writable_env) -> None:
    r = client.get("/api/credentials/status")
    assert r.status_code == 200
    assert r.json()["writable"] is True


def test_status_writable_false_inside_devin(client, monkeypatch) -> None:
    monkeypatch.setenv("DEVIN_SESSION_ID", "abc")
    r = client.get("/api/credentials/status")
    assert r.status_code == 200
    assert r.json()["writable"] is False


# ----- /api/credentials/set -----------------------------------------


def test_set_route_writes_to_dotenv(client, writable_env) -> None:
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "OPENAI_API_KEY",
            "value": "sk-fake-12345",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "source_id": "SRC-AI-OPENAI-001",
        "env_var": "OPENAI_API_KEY",
    }
    text = writable_env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-fake-12345" in text
    # Process env updated too:
    assert os.environ.get("OPENAI_API_KEY") == "sk-fake-12345"


def test_set_route_refuses_inside_devin_session(
    client, monkeypatch
) -> None:
    monkeypatch.setenv("DEVIN_SESSION_ID", "abc")
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "OPENAI_API_KEY",
            "value": "sk-x",
        },
    )
    assert r.status_code == 409


def test_set_route_404_on_unknown_source(client, writable_env) -> None:
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-DOES-NOT-EXIST",
            "env_var": "ANY",
            "value": "x",
        },
    )
    assert r.status_code == 404


def test_set_route_422_when_env_var_not_in_blueprint(
    client, writable_env
) -> None:
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "WRONG_VAR_NAME",
            "value": "x",
        },
    )
    assert r.status_code == 422


def test_set_route_422_on_empty_value(client, writable_env) -> None:
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "OPENAI_API_KEY",
            "value": "",
        },
    )
    assert r.status_code == 422


def test_set_route_422_on_newline_in_value(client, writable_env) -> None:
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "OPENAI_API_KEY",
            "value": "line1\nline2",
        },
    )
    assert r.status_code == 422


def test_set_route_does_not_echo_value(client, writable_env) -> None:
    secret = "ABSOLUTELY-DO-NOT-LEAK-THIS"
    r = client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "OPENAI_API_KEY",
            "value": secret,
        },
    )
    assert r.status_code == 200
    assert secret not in r.text


# ----- presence shows .env-set vars without a server restart --------


def test_status_reflects_freshly_set_dotenv(client, writable_env) -> None:
    # A fresh write should flip the row to "present" on the next
    # status call without any server restart.
    client.post(
        "/api/credentials/set",
        json={
            "source_id": "SRC-AI-OPENAI-001",
            "env_var": "OPENAI_API_KEY",
            "value": "sk-fake",
        },
    )
    r = client.get("/api/credentials/status")
    items = {it["source_id"]: it for it in r.json()["items"]}
    assert items["SRC-AI-OPENAI-001"]["state"] == "present"
