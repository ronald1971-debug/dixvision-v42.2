"""Per-provider live credential verification (Dashboard-2026 wave-01.5 PR-B).

PR-A added presence detection: "is the env var set?". This module
adds the next bit: "does the key actually authenticate?". The result
is a tri-state-plus: ``ok | unauthorized | rate_limited | timeout |
network_error | not_found | no_verifier | missing_key``.

Design constraints
------------------
- **No secret values ever leave this module.** ``VerifyResult.detail``
  contains only the HTTP status, not the key. The caller surface
  (``POST /api/credentials/verify``) returns the same struct.
- **Stdlib HTTP only.** ``urllib.request`` keeps the runtime
  dependency count at 5 (PR-A held the line; we keep holding it).
  Sync calls run in FastAPI's threadpool so the operator's UI stays
  responsive.
- **Conservative coverage.** A per-provider :class:`VerifierSpec` is
  added only for endpoints whose auth shape is well-documented and
  stable (OpenAI / Gemini / Grok / DeepSeek / GitHub). Everything
  else returns :data:`VerifyOutcome.NO_VERIFIER` so the UI can show
  "verification not yet supported" without lying.
- **Bounded blast radius.** Default 5-second timeout, single
  request, no retries. Verification is operator-initiated (button
  click) — never auto-fired from the matrix render.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class VerifyOutcome(StrEnum):
    """Result classes for a single verification attempt."""

    OK = "ok"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    NO_VERIFIER = "no_verifier"
    MISSING_KEY = "missing_key"


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Result of a single verification call.

    ``detail`` is intended for the operator UI. It MUST NOT include
    the secret value or any echo of it. Tests assert on this.
    """

    outcome: VerifyOutcome
    http_status: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class VerifierSpec:
    """Static description of how to auth-ping one provider.

    Three auth shapes are supported and exhaustive for the 2026 LLM /
    market-data / news / on-chain / macro REST APIs we currently
    register:

    1. ``Authorization: Bearer <key>`` header (most LLM providers,
       X v2, etc.).
    2. ``?<param>=<key>`` query string (Google AI Studio / Gemini,
       FRED, Glassnode). The parameter name is part of
       :attr:`url` (e.g. ``...?key={key}`` vs ``...?api_key={key}``).
    3. Custom request header (Dune ``X-DUNE-API-KEY`` etc.). The
       header name is carried in :attr:`header_name`.

    A fourth shape (separate ``key`` + ``secret`` HMAC signing, as
    used by some exchange private endpoints) is intentionally
    excluded — the only registry rows that fit are off-limits to
    this PR (no exchange private endpoints are in the SCVS yet).
    """

    url: str  # may contain ``{key}`` for query-string auth
    primary_env_var: str
    auth_style: str  # "bearer" | "query" | "header"
    header_name: str | None = None  # used when auth_style == "header"


VERIFIERS: Mapping[str, VerifierSpec] = MappingProxyType(
    {
        # ----- AI providers -----
        "openai": VerifierSpec(
            url="https://api.openai.com/v1/models",
            primary_env_var="OPENAI_API_KEY",
            auth_style="bearer",
        ),
        "google": VerifierSpec(
            # Gemini / Google AI Studio uses ?key= query auth, not
            # Authorization header. The {key} placeholder is filled
            # in via :func:`urllib.parse.quote` in :func:`_build_request`.
            url=(
                "https://generativelanguage.googleapis.com/v1beta/models"
                "?key={key}"
            ),
            primary_env_var="GEMINI_API_KEY",
            auth_style="query",
        ),
        "xai": VerifierSpec(
            url="https://api.x.ai/v1/models",
            primary_env_var="XAI_API_KEY",
            auth_style="bearer",
        ),
        "deepseek": VerifierSpec(
            url="https://api.deepseek.com/v1/models",
            primary_env_var="DEEPSEEK_API_KEY",
            auth_style="bearer",
        ),
        # ----- Dev -----
        "github": VerifierSpec(
            url="https://api.github.com/user",
            primary_env_var="GITHUB_TOKEN",
            auth_style="bearer",
        ),
        # ----- Social -----
        "x": VerifierSpec(
            # X v2 with App-only Bearer Token. ``users/by/username``
            # is the cheapest authenticated GET that exists on the
            # free tier — costs no tweet-cap quota and returns 401
            # without auth, 200 with valid bearer.
            url="https://api.x.com/2/users/by/username/X",
            primary_env_var="X_BEARER_TOKEN",
            auth_style="bearer",
        ),
        # ----- On-chain -----
        "glassnode": VerifierSpec(
            # Glassnode uses ``?api_key=`` (not ``?key=``). The free
            # Tier-1 endpoint ``addresses/active_count`` returns the
            # daily count for BTC and is reachable on every plan;
            # 401 on bad key, 200 on good key.
            url=(
                "https://api.glassnode.com/v1/metrics/addresses/active_count"
                "?a=BTC&i=24h&api_key={key}"
            ),
            primary_env_var="GLASSNODE_API_KEY",
            auth_style="query",
        ),
        "dune": VerifierSpec(
            # Dune ``/auth/me`` is the canonical auth-ping endpoint;
            # uses the ``X-DUNE-API-KEY`` header (not Authorization).
            url="https://api.dune.com/api/v1/auth/me",
            primary_env_var="DUNE_API_KEY",
            auth_style="header",
            header_name="X-DUNE-API-KEY",
        ),
        # ----- Macro -----
        "fred": VerifierSpec(
            # FRED uses ``?api_key=`` and demands ``file_type=json``
            # (otherwise it returns XML). ``sources`` is a no-arg
            # endpoint that's permissioned identically to a real
            # series fetch — 200 if the key is valid, 400 + JSON
            # error body if not.
            url=(
                "https://api.stlouisfed.org/fred/sources"
                "?api_key={key}&file_type=json"
            ),
            primary_env_var="FRED_API_KEY",
            auth_style="query",
        ),
        "bls": VerifierSpec(
            # BLS exposes a public series-popularity endpoint that
            # accepts the registration key as ``?registrationkey=``.
            # Returns ``status: REQUEST_SUCCEEDED`` on a valid key,
            # ``REQUEST_NOT_PROCESSED`` on an invalid one. We treat
            # 200 as ``OK`` regardless of body text — the operator
            # still has to look at the response in the dashboard if
            # the body says NOT_PROCESSED, but at least the network
            # path is verified.
            url=(
                "https://api.bls.gov/publicAPI/v2/timeseries/popular"
                "?registrationkey={key}"
            ),
            primary_env_var="BLS_API_KEY",
            auth_style="query",
        ),
    }
)
"""Provider → :class:`VerifierSpec`. Adding a verifier here is the
*only* way to flip a provider from ``NO_VERIFIER`` to live-pingable.
This is intentional — verifiers can produce real network traffic and
must be reviewed line-by-line, not auto-derived from a YAML."""


# Default timeout (seconds). Single, conservative — every provider
# in the table responds in <1 s to a /models or /user GET on a healthy
# day, so 5 s comfortably absorbs cold starts and TLS handshakes.
DEFAULT_TIMEOUT_S: float = 5.0


def _build_request(spec: VerifierSpec, key: str) -> urllib.request.Request:
    if spec.auth_style == "query":
        url = spec.url.format(key=urllib.parse.quote(key, safe=""))
        return urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "dixvision-credential-verifier/1",
            },
            method="GET",
        )
    if spec.auth_style == "bearer":
        return urllib.request.Request(
            spec.url,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                # GitHub's API insists on a User-Agent.
                "User-Agent": "dixvision-credential-verifier/1",
            },
            method="GET",
        )
    if spec.auth_style == "header":
        if not spec.header_name:
            raise ValueError(
                "VerifierSpec.header_name is required when "
                "auth_style == 'header'"
            )
        return urllib.request.Request(
            spec.url,
            headers={
                spec.header_name: key,
                "Accept": "application/json",
                "User-Agent": "dixvision-credential-verifier/1",
            },
            method="GET",
        )
    raise ValueError(f"unknown auth_style {spec.auth_style!r}")


def _classify_http_error(status: int) -> VerifyOutcome:
    # 400 is grouped with 401/403 because some providers (notably
    # FRED) return ``400 + JSON error body`` for an invalid API key
    # rather than 401. The verifiers module only points at endpoints
    # whose 4xx semantics are documented to mean "auth-side problem"
    # (i.e. we don't send malformed bodies / unknown query params),
    # so a 400 from any registered verifier is unambiguously an
    # auth failure from the operator's point of view. Mapping it to
    # NETWORK_ERROR misled operators into chasing connectivity bugs
    # instead of fixing the key.
    if status in (400, 401, 403):
        return VerifyOutcome.UNAUTHORIZED
    if status == 404:
        return VerifyOutcome.NOT_FOUND
    if status == 429:
        return VerifyOutcome.RATE_LIMITED
    if 500 <= status <= 599:
        return VerifyOutcome.SERVER_ERROR
    return VerifyOutcome.NETWORK_ERROR


def _open(
    request: urllib.request.Request,
    timeout: float,
):
    """Indirection to make HTTP calls patchable in tests.

    Tests monkey-patch this name so we never make a real outbound
    request from the unit-test suite. Production code imports the
    module-level symbol so the patch lands on every caller.
    """

    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def verify_provider(
    provider: str,
    env: Mapping[str, str],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> VerifyResult:
    """Best-effort live auth ping for ``provider`` using ``env``.

    Pure-ish: deterministic given a fixed network response. Real
    verification is non-deterministic by definition (network state),
    which is exactly why INV-67 marks the cognitive subsystems as
    advisory-only and why this entry point lives outside the hot
    path (it's only reachable via ``POST /api/credentials/verify``).
    """

    spec = VERIFIERS.get(provider)
    if spec is None:
        return VerifyResult(
            outcome=VerifyOutcome.NO_VERIFIER,
            http_status=None,
            detail=f"no live verifier registered for provider '{provider}'",
        )

    raw = env.get(spec.primary_env_var)
    if raw is None or raw.strip() == "":
        return VerifyResult(
            outcome=VerifyOutcome.MISSING_KEY,
            http_status=None,
            detail=f"{spec.primary_env_var} is not set",
        )

    request = _build_request(spec, raw.strip())

    try:
        with _open(request, timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            return VerifyResult(
                outcome=VerifyOutcome.OK,
                http_status=int(status),
                detail=f"HTTP {status}",
            )
    except urllib.error.HTTPError as exc:
        return VerifyResult(
            outcome=_classify_http_error(exc.code),
            http_status=exc.code,
            detail=f"HTTP {exc.code}",
        )
    except TimeoutError:
        return VerifyResult(
            outcome=VerifyOutcome.TIMEOUT,
            http_status=None,
            detail=f"timed out after {timeout:.1f}s",
        )
    except urllib.error.URLError as exc:
        # Inner reason is sometimes a socket.timeout instance.
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return VerifyResult(
                outcome=VerifyOutcome.TIMEOUT,
                http_status=None,
                detail=f"timed out after {timeout:.1f}s",
            )
        return VerifyResult(
            outcome=VerifyOutcome.NETWORK_ERROR,
            http_status=None,
            # Use ``type(exc).__name__`` rather than ``str(exc)`` so
            # nothing the OS produced (DNS resolver hint, proxy URL,
            # etc.) can leak through. Keys are never in URLError
            # messages but defence-in-depth.
            detail=f"network error: {type(exc).__name__}",
        )


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "VERIFIERS",
    "VerifierSpec",
    "VerifyOutcome",
    "VerifyResult",
    "verify_provider",
]
