"""HarnessRouteRegistrar — domain-organised route inventory + boot audit (P1.4).

Third and final harness god-object refactor PR after PR #346
(:class:`HarnessBootManager`) and PR #349
(:class:`HarnessBackgroundTaskManager`). The historic
``ui/server.py`` registers 47 FastAPI routes inline with
``@app.get(...)`` / ``@app.post(...)`` decorators sprinkled across
~1800 LOC between module-level helpers, with no canonical group
boundary and no audit that every expected route is actually
mounted. This manager owns the canonical domain → routes
inventory and runs a fail-closed boot-time audit asserting that
every expected route is registered on ``app``.

The route decorators themselves stay where they are in
``ui/server.py`` — moving 47 decorated routes (and their inline
helpers) would be a high-risk pure cut-and-paste edit with no
behaviour change. Instead, the manager mounts the *canonical
inventory* — the single place a reader looks to find which
endpoints belong to which domain — and runs that inventory as a
boot-time assertion. If a future PR accidentally drops a route,
boot fails fast with a clear "missing route" message naming the
domain and method+path; if a future PR adds a route, the audit
also notices and either records it under a new domain (if it
matches a domain's prefix policy) or raises so the canonical
inventory must be updated.

INV-15 byte-identical replay, B27 / B28 / INV-71 authority
symmetry, B32 single-mutator FSM, HARDEN-04 / INV-70 freeze
policy, and B7 dashboard-prefix lint are all preserved by
construction — this module is pure module-level inspection of
``app.routes`` after FastAPI's decorator pass has fired; it
never constructs typed events, never mutates ``app``, and never
opens a network port.

The thirteen canonical domains:

 1. ``core`` — bootstrap surface (``/``, ``/api/health``,
    ``/api/registry/*``, ``/api/ai/providers``, ``/api/docs``).
 2. ``credentials`` — ``/api/credentials/{status,verify,set}``
    (registry-driven API-key inventory).
 3. ``operator`` — every ``/api/operator/*`` route plus the two
    operator-flavoured side surfaces ``/api/feeds/memecoin/summary``
    and ``/api/wallet/info``.
 4. ``admin`` — env-flagged debug surface
    (``/api/admin/learning/tick``,
    ``/api/admin/route_inventory``).
 5. ``cognitive`` — ``/api/cognitive/chat/*`` (status, turn,
    approvals).
 6. ``engine`` — hot-path tick / signal / events / backtest
    (``/api/tick``, ``/api/signal``, ``/api/events``,
    ``/api/testing/backtest``).
 7. ``dashboard`` — the SSE bridge plus the dashboard router's
    read/write widget surface
    (``/api/dashboard/{stream,mode,engines,strategies,decisions,
    memecoin,summary,action/*}``).
 8. ``feeds`` — every market / news / trader feed adapter under
    ``/api/feeds/{binance,coindesk,pumpfun,raydium,tradingview}/*``.
 9. ``governance`` — ``/api/governance/*`` widget routes
    (promotion_gates / drift / sources / hazards).
10. ``execution`` — ``/api/execution/adapters``.
11. ``plugins`` — ``/api/plugins`` + per-plugin lifecycle
    (``/api/plugins/{plugin_id}/lifecycle``).
12. ``pages`` — server-rendered HTML pages
    (``/operator``, ``/credentials``, ``/indira-chat``,
    ``/dyon-chat``, ``/forms-grid``).
13. ``openapi`` — FastAPI's auto-mounted schema endpoints
    (``/openapi.json``, ``/docs/oauth2-redirect``). The Swagger
    UI is mounted at ``/api/docs`` (under ``core``) — the
    custom prefix moved the Swagger UI off the default ``/docs``
    path so the harness only owns one canonical docs entrypoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI


RouteKey = tuple[str, str]
"""Canonical (METHOD, PATH) identifier for a FastAPI route."""


_CORE_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/"),
        ("GET", "/api/health"),
        ("GET", "/api/registry/engines"),
        ("GET", "/api/registry/plugins"),
        ("GET", "/api/ai/providers"),
        ("GET", "/api/docs"),
    }
)

_GOVERNANCE_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/governance/promotion_gates"),
        ("GET", "/api/governance/drift"),
        ("GET", "/api/governance/sources"),
        ("GET", "/api/governance/hazards"),
    }
)

_EXECUTION_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/execution/adapters"),
    }
)

_PLUGINS_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/plugins"),
        ("POST", "/api/plugins/{plugin_id}/lifecycle"),
    }
)

_PAGES_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/operator"),
        ("GET", "/credentials"),
        ("GET", "/indira-chat"),
        ("GET", "/dyon-chat"),
        ("GET", "/forms-grid"),
    }
)

_OPENAPI_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/openapi.json"),
        ("GET", "/docs/oauth2-redirect"),
    }
)

_CREDENTIALS_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/credentials/status"),
        ("POST", "/api/credentials/verify"),
        ("POST", "/api/credentials/set"),
    }
)

_OPERATOR_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/operator/summary"),
        ("POST", "/api/operator/action/kill"),
        ("POST", "/api/operator/action/unlock"),
        ("POST", "/api/operator/action/mode"),
        ("POST", "/api/operator/audit"),
        ("GET", "/api/feeds/memecoin/summary"),
        ("GET", "/api/wallet/info"),
        ("GET", "/api/operator/source-trust"),
        ("POST", "/api/operator/source-trust/promote"),
        ("POST", "/api/operator/source-trust/demote"),
        ("GET", "/api/operator/learning-override"),
        ("POST", "/api/operator/learning-override"),
    }
)

_ADMIN_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("POST", "/api/admin/learning/tick"),
        ("GET", "/api/admin/route_inventory"),
    }
)

_COGNITIVE_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/cognitive/chat/status"),
        ("POST", "/api/cognitive/chat/turn"),
        ("GET", "/api/cognitive/chat/approvals"),
        ("POST", "/api/cognitive/chat/approvals/{request_id}/approve"),
        ("POST", "/api/cognitive/chat/approvals/{request_id}/reject"),
    }
)

_ENGINE_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("POST", "/api/tick"),
        ("POST", "/api/signal"),
        ("GET", "/api/events"),
        ("POST", "/api/testing/backtest"),
    }
)

_DASHBOARD_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("GET", "/api/dashboard/stream"),
        ("GET", "/api/dashboard/mode"),
        ("GET", "/api/dashboard/engines"),
        ("GET", "/api/dashboard/strategies"),
        ("GET", "/api/dashboard/decisions"),
        ("GET", "/api/dashboard/memecoin"),
        ("GET", "/api/dashboard/summary"),
        ("POST", "/api/dashboard/action/mode"),
        ("POST", "/api/dashboard/action/intent"),
        ("POST", "/api/dashboard/action/kill"),
        ("POST", "/api/dashboard/action/lifecycle"),
    }
)

_FEEDS_ROUTES: frozenset[RouteKey] = frozenset(
    {
        ("POST", "/api/feeds/binance/start"),
        ("POST", "/api/feeds/binance/stop"),
        ("GET", "/api/feeds/binance/status"),
        ("POST", "/api/feeds/coindesk/start"),
        ("POST", "/api/feeds/coindesk/stop"),
        ("GET", "/api/feeds/coindesk/status"),
        ("POST", "/api/feeds/pumpfun/start"),
        ("POST", "/api/feeds/pumpfun/stop"),
        ("GET", "/api/feeds/pumpfun/status"),
        ("GET", "/api/feeds/pumpfun/recent"),
        ("POST", "/api/feeds/raydium/start"),
        ("POST", "/api/feeds/raydium/stop"),
        ("GET", "/api/feeds/raydium/status"),
        ("GET", "/api/feeds/raydium/recent"),
        ("POST", "/api/feeds/tradingview/observation"),
        ("POST", "/api/feeds/tradingview/alert"),
    }
)


_CANONICAL_DOMAINS: tuple[str, ...] = (
    "core",
    "credentials",
    "operator",
    "admin",
    "cognitive",
    "engine",
    "dashboard",
    "governance",
    "execution",
    "plugins",
    "feeds",
    "pages",
    "openapi",
)

_DOMAIN_INVENTORY: Mapping[str, frozenset[RouteKey]] = {
    "core": _CORE_ROUTES,
    "credentials": _CREDENTIALS_ROUTES,
    "operator": _OPERATOR_ROUTES,
    "admin": _ADMIN_ROUTES,
    "cognitive": _COGNITIVE_ROUTES,
    "engine": _ENGINE_ROUTES,
    "dashboard": _DASHBOARD_ROUTES,
    "governance": _GOVERNANCE_ROUTES,
    "execution": _EXECUTION_ROUTES,
    "plugins": _PLUGINS_ROUTES,
    "feeds": _FEEDS_ROUTES,
    "pages": _PAGES_ROUTES,
    "openapi": _OPENAPI_ROUTES,
}


@dataclass(frozen=True, slots=True)
class RouteAuditReport:
    """Outcome of :meth:`HarnessRouteRegistrar.audit`.

    ``missing`` enumerates expected ``(METHOD, PATH)`` pairs that
    were not found on the supplied ``app``; an empty tuple means
    every expected route is mounted. ``unexpected`` enumerates
    ``(METHOD, PATH)`` pairs that ARE mounted on ``app`` but do
    not appear in any canonical domain — usually a sign that
    :data:`_DOMAIN_INVENTORY` needs updating after a route was
    added.

    ``by_domain`` is the inverse view: per-domain list of routes
    that are both expected AND mounted (the live inventory).
    """

    missing: tuple[tuple[str, RouteKey], ...]
    unexpected: tuple[RouteKey, ...]
    by_domain: Mapping[str, tuple[RouteKey, ...]]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected


class HarnessRouteRegistrar:
    """Owner of the canonical FastAPI route → domain inventory.

    The route handlers themselves are registered by the
    ``@app.get(...)`` / ``@app.post(...)`` decorators inline in
    ``ui/server.py``. This class does not replace those
    decorators — it inspects the decorated ``app.routes`` after
    the module body has executed and asserts that every entry in
    :data:`_DOMAIN_INVENTORY` is present.

    The class is intentionally stateless apart from the frozen
    domain mapping; it never holds a reference to ``app``,
    ``STATE``, or any engine. Methods take ``app`` as an
    explicit parameter so tests can build a small ``FastAPI``
    fixture and call :meth:`audit` directly without booting the
    harness.

    Usage from ``ui.server`` at module load (after every route
    decorator has fired)::

        _ROUTE_REGISTRAR = HarnessRouteRegistrar()
        _ROUTE_REGISTRAR.audit_or_raise(app)

    Fails closed: any drift (missing OR unexpected) raises
    :class:`RuntimeError` with a single-line diagnostic naming
    every affected route. The harness refuses to boot until the
    inventory matches.
    """

    def domains(self) -> tuple[str, ...]:
        """Canonical ordered list of domain names."""

        return _CANONICAL_DOMAINS

    def expected_routes(self, domain: str) -> frozenset[RouteKey]:
        """Expected ``(METHOD, PATH)`` set for ``domain``.

        Raises :class:`KeyError` if ``domain`` is not in
        :meth:`domains`.
        """

        if domain not in _DOMAIN_INVENTORY:
            raise KeyError(
                f"unknown route registrar domain: {domain!r}; "
                f"expected one of {self.domains()!r}"
            )
        return _DOMAIN_INVENTORY[domain]

    def expected_all(self) -> frozenset[RouteKey]:
        """Union of every expected route across every domain."""

        out: set[RouteKey] = set()
        for routes in _DOMAIN_INVENTORY.values():
            out.update(routes)
        return frozenset(out)

    def domain_for(self, key: RouteKey) -> str | None:
        """Return the canonical domain that owns ``key``, or
        ``None`` if no domain claims it."""

        for domain, routes in _DOMAIN_INVENTORY.items():
            if key in routes:
                return domain
        return None

    def mounted_routes(self, app: FastAPI) -> frozenset[RouteKey]:
        """Inspect ``app.routes`` and project the mounted FastAPI
        endpoints as ``(METHOD, PATH)`` pairs.

        Static-file mounts and non-API ``WebSocketRoute`` entries
        are ignored — only routes with a ``path`` AND a non-empty
        ``methods`` attribute are returned (i.e. APIRoute /
        Route).
        """

        out: set[RouteKey] = set()
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not path or not methods:
                continue
            for method in methods:
                if not isinstance(method, str):
                    continue
                upper = method.upper()
                if upper == "HEAD":
                    continue
                out.add((upper, path))
        return frozenset(out)

    def audit(self, app: FastAPI) -> RouteAuditReport:
        """Inspect ``app`` and return a structured audit report.

        Does NOT raise on drift; callers wanting fail-closed boot
        semantics call :meth:`audit_or_raise` instead.
        """

        mounted = self.mounted_routes(app)
        expected = self.expected_all()

        missing_pairs: list[tuple[str, RouteKey]] = []
        for domain in self.domains():
            for key in sorted(self.expected_routes(domain)):
                if key not in mounted:
                    missing_pairs.append((domain, key))

        unexpected: list[RouteKey] = []
        for key in sorted(mounted):
            if key not in expected:
                unexpected.append(key)

        by_domain: dict[str, tuple[RouteKey, ...]] = {}
        for domain in self.domains():
            present = sorted(
                key for key in self.expected_routes(domain) if key in mounted
            )
            by_domain[domain] = tuple(present)

        return RouteAuditReport(
            missing=tuple(missing_pairs),
            unexpected=tuple(unexpected),
            by_domain=by_domain,
        )

    def audit_or_raise(self, app: FastAPI) -> RouteAuditReport:
        """Run :meth:`audit` and raise :class:`RuntimeError` on
        any drift.

        The single-line diagnostic names every missing /
        unexpected route so operators see the canonical fix
        without grepping the source tree.
        """

        report = self.audit(app)
        if report.ok:
            return report
        diagnostics: list[str] = []
        if report.missing:
            missing_str = ", ".join(
                f"{domain}:{method} {path}"
                for domain, (method, path) in report.missing
            )
            diagnostics.append(f"missing routes: {missing_str}")
        if report.unexpected:
            unexpected_str = ", ".join(
                f"{method} {path}" for method, path in report.unexpected
            )
            diagnostics.append(f"unexpected routes: {unexpected_str}")
        raise RuntimeError(
            "HarnessRouteRegistrar inventory drift — "
            + "; ".join(diagnostics)
            + " — update ui/harness/route_registrar.py to match"
        )

    def inventory(self, app: FastAPI) -> Mapping[str, tuple[RouteKey, ...]]:
        """Return the live per-domain inventory (sorted).

        Convenience over :meth:`audit` for the
        ``/api/admin/route_inventory`` endpoint.
        """

        return self.audit(app).by_domain


__all__ = (
    "HarnessRouteRegistrar",
    "RouteAuditReport",
    "RouteKey",
)
