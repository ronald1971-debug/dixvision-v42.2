"""Unit tests for live credential verification.

These tests **never make real outbound HTTP requests**. The module
exposes ``_open`` as a thin indirection over
``urllib.request.urlopen`` exactly so we can monkey-patch it here
and exercise every outcome class without touching the network.
"""

from __future__ import annotations

import urllib.error
from io import BytesIO
from typing import Any

import pytest

from system_engine.credentials import verifiers
from system_engine.credentials.verifiers import (
    VERIFIERS,
    VerifyOutcome,
    verify_provider,
)

# --------------------------------------------------------------------
# Fake response objects + monkey-patch helpers
# --------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``http.client.HTTPResponse``."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self._body = BytesIO(b"{}")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status


def _patch_open(monkeypatch, behaviour) -> list[Any]:
    """Replace ``verifiers._open`` with the given callable.

    Returns a list that captures every call so tests can assert on
    URLs / headers without touching the network.
    """

    calls: list[Any] = []

    def fake_open(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "timeout": timeout,
            }
        )
        return behaviour(request, timeout)

    monkeypatch.setattr(verifiers, "_open", fake_open)
    return calls


# --------------------------------------------------------------------
# Outcome class tests
# --------------------------------------------------------------------


def test_verify_ok_for_known_provider(monkeypatch) -> None:
    _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    result = verify_provider("openai", {"OPENAI_API_KEY": "sk-fake"})
    assert result.outcome is VerifyOutcome.OK
    assert result.http_status == 200


def test_verify_unauthorized_on_401(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, 401, "Unauthorized", {}, None)

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("openai", {"OPENAI_API_KEY": "sk-bad"})
    assert result.outcome is VerifyOutcome.UNAUTHORIZED
    assert result.http_status == 401


def test_verify_unauthorized_on_403(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, 403, "Forbidden", {}, None)

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("github", {"GITHUB_TOKEN": "ghp_fake"})
    assert result.outcome is VerifyOutcome.UNAUTHORIZED
    assert result.http_status == 403


def test_verify_unauthorized_on_400_for_fred(monkeypatch) -> None:
    # FRED returns 400 + JSON error body for invalid API keys (not
    # 401). The classifier groups 400 with 401/403 so the operator
    # sees an UNAUTHORIZED outcome, not a misleading NETWORK_ERROR.
    def behaviour(r, t):
        raise urllib.error.HTTPError(
            r.full_url, 400, "Bad Request", {}, None
        )

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("fred", {"FRED_API_KEY": "fred-bad"})
    assert result.outcome is VerifyOutcome.UNAUTHORIZED
    assert result.http_status == 400


def test_verify_rate_limited_on_429(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, 429, "Too Many", {}, None)

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("xai", {"XAI_API_KEY": "xai-fake"})
    assert result.outcome is VerifyOutcome.RATE_LIMITED
    assert result.http_status == 429


def test_verify_server_error_on_5xx(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, 503, "Unavailable", {}, None)

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("deepseek", {"DEEPSEEK_API_KEY": "x"})
    assert result.outcome is VerifyOutcome.SERVER_ERROR
    assert result.http_status == 503


def test_verify_timeout_when_socket_times_out(monkeypatch) -> None:
    def behaviour(r, t):
        raise TimeoutError("connection timed out")

    _patch_open(monkeypatch, behaviour)
    result = verify_provider(
        "openai", {"OPENAI_API_KEY": "sk-x"}, timeout=2.5
    )
    assert result.outcome is VerifyOutcome.TIMEOUT
    assert result.http_status is None
    assert "2.5" in result.detail


def test_verify_timeout_when_url_error_wraps_timeout(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.URLError(TimeoutError("slow"))

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("openai", {"OPENAI_API_KEY": "sk-x"})
    assert result.outcome is VerifyOutcome.TIMEOUT


def test_verify_network_error_on_url_error(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.URLError("getaddrinfo failed")

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("openai", {"OPENAI_API_KEY": "sk-x"})
    assert result.outcome is VerifyOutcome.NETWORK_ERROR
    assert result.http_status is None


def test_verify_no_verifier_for_unsupported_provider() -> None:
    # ``reuters`` is in the registry (Reuters Connect, enterprise-only)
    # but has no public verify endpoint, so it's still NO_VERIFIER.
    result = verify_provider("reuters", {"REUTERS_API_KEY": "x"})
    assert result.outcome is VerifyOutcome.NO_VERIFIER
    assert result.http_status is None
    assert "reuters" in result.detail


def test_verify_missing_key_when_env_unset(monkeypatch) -> None:
    # Should never reach the network — assert on no calls made.
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    result = verify_provider("openai", {})
    assert result.outcome is VerifyOutcome.MISSING_KEY
    assert calls == []


def test_verify_missing_key_when_env_whitespace(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    result = verify_provider("openai", {"OPENAI_API_KEY": "   "})
    assert result.outcome is VerifyOutcome.MISSING_KEY
    assert calls == []


# --------------------------------------------------------------------
# Auth shape correctness
# --------------------------------------------------------------------


def test_bearer_auth_sends_authorization_header(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("openai", {"OPENAI_API_KEY": "sk-fake-bearer"})
    assert len(calls) == 1
    assert calls[0]["headers"].get("Authorization") == "Bearer sk-fake-bearer"


def test_query_auth_does_not_send_authorization_header(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("google", {"GEMINI_API_KEY": "gem-fake"})
    assert len(calls) == 1
    headers = calls[0]["headers"]
    # No Authorization header — Gemini uses ?key=…
    assert "Authorization" not in headers
    assert "gem-fake" in calls[0]["url"]


def test_query_auth_url_encodes_key(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("google", {"GEMINI_API_KEY": "needs/encoding=plus"})
    assert "needs%2Fencoding%3Dplus" in calls[0]["url"]


def test_query_auth_uses_api_key_param_for_glassnode(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("glassnode", {"GLASSNODE_API_KEY": "gn-fake"})
    assert len(calls) == 1
    url = calls[0]["url"]
    # Glassnode uses ?api_key= (not ?key=), and the URL must keep
    # the metric path identifier so a 200 actually means "key
    # authenticated" and not "path 404".
    assert "api_key=gn-fake" in url
    assert "/metrics/addresses/active_count" in url


def test_query_auth_uses_api_key_param_for_fred(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("fred", {"FRED_API_KEY": "fred-fake"})
    assert len(calls) == 1
    url = calls[0]["url"]
    assert "api_key=fred-fake" in url
    # ``file_type=json`` is mandatory — without it FRED returns XML
    # which we don't want to reason about.
    assert "file_type=json" in url


def test_query_auth_uses_registrationkey_param_for_bls(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("bls", {"BLS_API_KEY": "bls-fake"})
    url = calls[0]["url"]
    assert "registrationkey=bls-fake" in url


def test_header_auth_sends_custom_header_for_dune(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("dune", {"DUNE_API_KEY": "dune-fake"})
    assert len(calls) == 1
    headers = calls[0]["headers"]
    # Header names get title-cased by urllib.request.Request, so
    # match case-insensitively (Dune docs spell it X-DUNE-API-KEY).
    canonical = {k.lower(): v for k, v in headers.items()}
    assert canonical.get("x-dune-api-key") == "dune-fake"
    # Must NOT also send Authorization — Dune doesn't accept it and
    # the docs explicitly forbid mixing.
    assert "Authorization" not in headers
    # Endpoint must be the canonical /auth/me ping (not a paid path).
    assert calls[0]["url"] == "https://api.dune.com/api/v1/auth/me"


def test_bearer_auth_for_x_uses_users_by_username_endpoint(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("x", {"X_BEARER_TOKEN": "x-bearer-fake"})
    assert len(calls) == 1
    headers = calls[0]["headers"]
    assert headers.get("Authorization") == "Bearer x-bearer-fake"
    # Endpoint pinning — free-tier compatible, no tweet-cap usage.
    assert calls[0]["url"].startswith(
        "https://api.x.com/2/users/by/username/"
    )


def test_timeout_passed_through(monkeypatch) -> None:
    calls = _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    verify_provider("openai", {"OPENAI_API_KEY": "k"}, timeout=1.25)
    assert calls[0]["timeout"] == 1.25


# --------------------------------------------------------------------
# Defence-in-depth: detail strings never leak the secret value.
# --------------------------------------------------------------------


SECRET = "DO-NOT-LEAK-THIS-VALUE-1234"


@pytest.mark.parametrize(
    "provider, env_var",
    [
        ("openai", "OPENAI_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("xai", "XAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("github", "GITHUB_TOKEN"),
        ("x", "X_BEARER_TOKEN"),
        ("glassnode", "GLASSNODE_API_KEY"),
        ("dune", "DUNE_API_KEY"),
        ("fred", "FRED_API_KEY"),
        ("bls", "BLS_API_KEY"),
    ],
)
def test_no_leak_on_ok(monkeypatch, provider, env_var) -> None:
    _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    result = verify_provider(provider, {env_var: SECRET})
    assert SECRET not in result.detail


@pytest.mark.parametrize(
    "provider, env_var",
    [
        ("openai", "OPENAI_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("github", "GITHUB_TOKEN"),
        ("dune", "DUNE_API_KEY"),
        ("fred", "FRED_API_KEY"),
    ],
)
def test_no_leak_on_unauthorized(monkeypatch, provider, env_var) -> None:
    def behaviour(r, t):
        raise urllib.error.HTTPError(r.full_url, 401, "Unauthorized", {}, None)

    _patch_open(monkeypatch, behaviour)
    result = verify_provider(provider, {env_var: SECRET})
    assert SECRET not in result.detail


def test_no_leak_on_network_error(monkeypatch) -> None:
    def behaviour(r, t):
        raise urllib.error.URLError(f"failed talking to host with key={SECRET}")

    _patch_open(monkeypatch, behaviour)
    result = verify_provider("openai", {"OPENAI_API_KEY": SECRET})
    # We deliberately format ``type(exc).__name__`` only.
    assert SECRET not in result.detail
    assert result.detail == "network error: URLError"


# --------------------------------------------------------------------
# Coverage: every entry in VERIFIERS is exercised at least once.
# --------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(VERIFIERS.keys()))
def test_each_registered_verifier_can_run(monkeypatch, provider) -> None:
    spec = VERIFIERS[provider]
    _patch_open(monkeypatch, lambda r, t: _FakeResponse(200))
    result = verify_provider(provider, {spec.primary_env_var: "fake-key"})
    assert result.outcome is VerifyOutcome.OK
