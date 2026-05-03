"""HTTP route tests for ``POST /api/credentials/verify``.

The endpoint is the operator-facing trigger for the verifier module
in :mod:`system_engine.credentials.verifiers`. These tests stub out
network access at the verifier level (same monkey-patch trick used
in :mod:`tests.test_credentials_verifiers`) so the FastAPI test
client never makes a real outbound request.
"""

from __future__ import annotations

import importlib
import urllib.error
from io import BytesIO
from typing import Any

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")
verifiers = importlib.import_module("system_engine.credentials.verifiers")


@pytest.fixture
def client():
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


class _FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self._body = BytesIO(b"{}")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status


def _patch(monkeypatch, behaviour) -> list[Any]:
    calls: list[Any] = []

    def fake_open(request, timeout):
        calls.append({"url": request.full_url, "timeout": timeout})
        return behaviour(request, timeout)

    monkeypatch.setattr(verifiers, "_open", fake_open)
    return calls


def test_verify_route_ok(client, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    _patch(monkeypatch, lambda r, t: _FakeResponse(200))
    r = client.post(
        "/api/credentials/verify",
        json={"source_id": "SRC-AI-OPENAI-001"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["source_id"] == "SRC-AI-OPENAI-001"
    assert data["provider"] == "openai"
    assert data["outcome"] == "ok"
    assert data["http_status"] == 200


def test_verify_route_unauthorized(client, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")

    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, 401, "U", {}, None)

    _patch(monkeypatch, behaviour)
    r = client.post(
        "/api/credentials/verify",
        json={"source_id": "SRC-AI-OPENAI-001"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["outcome"] == "unauthorized"
    assert data["http_status"] == 401


def test_verify_route_no_verifier_for_unsupported_source(
    client, monkeypatch
) -> None:
    monkeypatch.setenv("REUTERS_API_KEY", "x")
    r = client.post(
        "/api/credentials/verify",
        json={"source_id": "SRC-NEWS-REUTERS-001"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["outcome"] == "no_verifier"
    assert data["http_status"] is None


def test_verify_route_missing_key(client, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post(
        "/api/credentials/verify",
        json={"source_id": "SRC-AI-OPENAI-001"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["outcome"] == "missing_key"
    assert "OPENAI_API_KEY" in data["detail"]


def test_verify_route_unknown_source_id(client) -> None:
    r = client.post(
        "/api/credentials/verify",
        json={"source_id": "SRC-DOES-NOT-EXIST"},
    )
    assert r.status_code == 404


def test_verify_route_rejects_non_auth_required_row(client) -> None:
    """Sources with ``auth: none`` are not in the requirement set."""

    r = client.post(
        "/api/credentials/verify",
        # SRC-MARKET-BINANCE-001 is auth: none in the registry.
        json={"source_id": "SRC-MARKET-BINANCE-001"},
    )
    assert r.status_code == 404


def test_verify_route_validates_body(client) -> None:
    r = client.post("/api/credentials/verify", json={})
    assert r.status_code == 422
    r = client.post(
        "/api/credentials/verify", json={"source_id": ""}
    )
    assert r.status_code == 422


def test_verify_route_does_not_leak_value(client, monkeypatch) -> None:
    secret = "ABSOLUTELY-MUST-NOT-LEAK"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _patch(monkeypatch, lambda r, t: _FakeResponse(200))
    r = client.post(
        "/api/credentials/verify",
        json={"source_id": "SRC-AI-OPENAI-001"},
    )
    assert r.status_code == 200
    assert secret not in r.text
