"""
mind.sources.providers.api_sniffer \u2014 DYON's auto-API-detective.

Given any URL, probe a bounded set of well-known endpoints and classify what
API surface the host exposes. Results feed ``SYSTEM/API_CANDIDATE`` ledger
events so DYON can propose drafting a new Provider adapter.

Never bypasses robots.txt. Never tries authenticated endpoints without a key.
Never stores or posts secrets. Probes complete within a small time budget.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SEC = 4.0
USER_AGENT = "DIX-VISION-APISniffer/42.2 (+https://dix-vision.local)"

# Probe in this order; first hit wins the classification.
PROBE_PATHS: tuple[tuple[str, str], ...] = (
    ("openapi", "/openapi.json"),
    ("openapi", "/swagger.json"),
    ("openapi", "/api-docs"),
    ("openapi", "/docs/openapi.json"),
    ("openapi", "/v3/api-docs"),
    ("graphql", "/graphql"),
    ("graphql", "/api/graphql"),
    ("rest_root", "/api"),
    ("rest_root", "/api/v1"),
    ("rest_root", "/api/v2"),
    ("rest_root", "/v1"),
    ("rest_root", "/v2"),
    ("rest_root", "/rest"),
    ("websocket", "/ws"),
    ("websocket", "/stream"),
    ("websocket", "/websocket"),
    ("rss", "/feed"),
    ("rss", "/rss"),
    ("rss", "/rss.xml"),
    ("rss", "/atom.xml"),
    ("well_known", "/.well-known/openapi.json"),
    ("sitemap", "/sitemap.xml"),
)


@dataclass
class ApiCandidate:
    """Result of sniffing a host."""
    host: str
    base_url: str
    api_surfaces: list[str] = field(default_factory=list)
    endpoints: list[tuple[str, str, int]] = field(default_factory=list)   # (kind, url, status)
    auth_required: str | None = None                                   # "bearer" | "cookie" | "none"
    rate_limit_hint: str | None = None                                 # header value if present
    robots_allows_probe: bool = True
    has_openapi: bool = False
    has_graphql: bool = False
    has_rss: bool = False
    has_websocket: bool = False
    relevance_score: float = 0.0                                          # 0..1 vs trading/news/onchain seeds
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "base_url": self.base_url,
            "api_surfaces": self.api_surfaces,
            "endpoints": [{"kind": k, "url": u, "status": s} for k, u, s in self.endpoints],
            "auth_required": self.auth_required,
            "rate_limit_hint": self.rate_limit_hint,
            "robots_allows_probe": self.robots_allows_probe,
            "has_openapi": self.has_openapi,
            "has_graphql": self.has_graphql,
            "has_rss": self.has_rss,
            "has_websocket": self.has_websocket,
            "relevance_score": round(self.relevance_score, 3),
            "errors": self.errors[:5],
        }


_RELEVANCE_KEYWORDS: tuple[str, ...] = (
    "trade", "market", "order", "orderbook", "ticker", "kline", "candle",
    "news", "sentiment", "onchain", "price", "quote", "swap", "liquidity",
    "options", "futures", "derivatives", "funding", "open-interest",
    "insider", "13f", "sec", "edgar", "filing", "dataset",
)


def _norm_base(url: str) -> tuple[str, str]:
    p = urlparse(url if "://" in url else "https://" + url)
    scheme = p.scheme or "https"
    host = p.hostname or url
    base = f"{scheme}://{host}"
    return host, base


def _check_robots(base_url: str, timeout: float) -> bool:
    try:
        req = Request(urljoin(base_url, "/robots.txt"),
                      headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as r:
            body = r.read(8192).decode("utf-8", errors="ignore")
    except Exception:
        return True                                                     # absent = allowed
    disallow: list[str] = []
    active = False
    for line in body.splitlines():
        line = line.strip()
        if line.lower().startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip()
            active = ua in ("*", USER_AGENT)
        elif active and line.lower().startswith("disallow:"):
            p = line.split(":", 1)[1].strip()
            if p == "/":
                return False
            disallow.append(p)
    return True


def _probe(url: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    req = Request(url, headers={"User-Agent": USER_AGENT,
                                "Accept": "application/json, text/xml, */*"})
    try:
        with urlopen(req, timeout=timeout) as r:
            headers = {k.lower(): v for k, v in r.getheaders()}
            body = r.read(4096)
            return int(r.status), headers, body
    except Exception:
        return 0, {}, b""


def _score_relevance(base_url: str, endpoints: list[tuple[str, str, int]]) -> float:
    joined = (base_url + " " + " ".join(u for _, u, _ in endpoints)).lower()
    hits = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in joined)
    return min(1.0, hits / 8.0)


def sniff(url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> ApiCandidate:
    """Probe URL with a small bounded budget. Returns a classifier."""
    host, base = _norm_base(url)
    cand = ApiCandidate(host=host, base_url=base)
    try:
        socket.gethostbyname(host)
    except Exception as exc:
        cand.errors.append(f"dns: {exc}")
        return cand

    cand.robots_allows_probe = _check_robots(base, timeout)
    if not cand.robots_allows_probe:
        cand.errors.append("robots.txt disallows")
        return cand

    rate_headers = ("x-ratelimit-limit", "x-ratelimit-remaining",
                    "x-rate-limit-limit", "retry-after")

    for kind, path in PROBE_PATHS:
        full = urljoin(base, path)
        status, headers, body = _probe(full, timeout)
        if status == 0:
            continue
        if status in (200, 201, 204, 301, 302, 401, 403):
            cand.endpoints.append((kind, full, status))
            for h in rate_headers:
                if h in headers and cand.rate_limit_hint is None:
                    cand.rate_limit_hint = f"{h}: {headers[h]}"
            auth = headers.get("www-authenticate", "").lower()
            if status == 401 or auth:
                if "bearer" in auth or status == 401:
                    cand.auth_required = "bearer"
                elif "basic" in auth:
                    cand.auth_required = "basic"
            if kind == "openapi" and status == 200:
                cand.has_openapi = True
                cand.api_surfaces.append("openapi")
                try:
                    json.loads(body.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
            elif kind == "graphql":
                cand.has_graphql = True
                cand.api_surfaces.append("graphql")
            elif kind == "rss":
                if b"<rss" in body or b"<feed" in body:
                    cand.has_rss = True
                    cand.api_surfaces.append("rss")
            elif kind == "rest_root" and status in (200, 401, 403):
                cand.api_surfaces.append("rest")
            elif kind == "websocket":
                cand.has_websocket = True
                cand.api_surfaces.append("websocket")
            elif kind == "sitemap" and b"<urlset" in body:
                cand.api_surfaces.append("sitemap")

    if cand.auth_required is None:
        cand.auth_required = "none" if cand.endpoints else None
    cand.api_surfaces = sorted(set(cand.api_surfaces))
    cand.relevance_score = _score_relevance(base, cand.endpoints)
    return cand


def propose_candidate(url: str, *, emit_ledger: bool = True) -> ApiCandidate:
    """Sniff + emit SYSTEM/API_CANDIDATE to the ledger (non-fatal)."""
    cand = sniff(url)
    if emit_ledger:
        try:
            from state.ledger.writer import get_writer
            get_writer().write("SYSTEM", "API_CANDIDATE", "DYON", cand.to_dict())
        except Exception:
            pass
    return cand


__all__ = ["ApiCandidate", "sniff", "propose_candidate"]
