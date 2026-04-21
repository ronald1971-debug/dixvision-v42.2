"""
cockpit/auth.py
DIX VISION v42.2 — Bearer-token middleware for the cockpit.

Loopback-only by default (`DIX_COCKPIT_BIND=127.0.0.1`). A bearer token is
required for every /api/* and /metrics route; static assets and /health are
public so the browser can render the login page and k8s can probe liveness.

The token is sourced from ``DIX_COCKPIT_TOKEN``. If unset on first boot a
random one is generated and printed to stderr + written to
``data/cockpit_token.txt`` (0600). The launcher reads that file and passes
``?token=...`` to the browser for one-click login.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PUBLIC_PATHS = (
    "/health",
    "/favicon.ico",
    "/static/",
    "/",            # SPA shell: JS loads and uses its own stored token
    "/pair",        # device pairing landing page (token-guarded by pairing)
    "/api/pair/claim",  # pairing claim uses its own one-time token
)


def _token_file() -> Path:
    return Path(os.environ.get("DIX_COCKPIT_TOKEN_FILE", "data/cockpit_token.txt"))


def get_or_create_token() -> str:
    """Return the current cockpit token, generating one if missing."""
    env = os.environ.get("DIX_COCKPIT_TOKEN", "").strip()
    if env:
        return env
    p = _token_file()
    if p.exists():
        txt = p.read_text().strip()
        if txt:
            return txt
    tok = secrets.token_urlsafe(32)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tok + "\n")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    except OSError as e:
        sys.stderr.write(f"[cockpit] could not persist token to {p}: {e}\n")
    sys.stderr.write(
        f"[cockpit] generated one-time token (persist via DIX_COCKPIT_TOKEN):\n"
        f"          {tok}\n"
    )
    return tok


def _extract(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    q = request.query_params.get("token")
    if q:
        return q.strip()
    c = request.cookies.get("dix_token")
    if c:
        return c.strip()
    return None


class TokenAuthMiddleware:
    """Pure-ASGI bearer-token gate (doesn't consume request body)."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        if any(path == p or path.startswith(p) for p in PUBLIC_PATHS):
            await self._app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        supplied = _extract(request)
        if not supplied or not secrets.compare_digest(supplied, self._token):
            resp = JSONResponse(
                {"error": "unauthorized",
                 "hint": "use ?token=\u2026 or Authorization: Bearer \u2026"},
                status_code=401,
            )
            await resp(scope, receive, send)
            return
        await self._app(scope, receive, send)
