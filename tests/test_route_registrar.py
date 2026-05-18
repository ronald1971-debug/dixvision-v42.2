"""P1.4 — HarnessRouteRegistrar tests.

Pins the canonical domain inventory and the fail-closed boot
audit semantics. The boot audit is run at module load of
:mod:`ui.server`, so any drift would already raise there; these
tests are the unit-level coverage of the registrar's contract.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi import FastAPI

from ui.harness.route_registrar import (
    HarnessRouteRegistrar,
    RouteAuditReport,
    RouteKey,
)


@pytest.fixture
def registrar() -> HarnessRouteRegistrar:
    return HarnessRouteRegistrar()


def test_domains_are_a_stable_canonical_tuple(
    registrar: HarnessRouteRegistrar,
) -> None:
    """The domain list is a fixed tuple; renaming any of these
    breaks the public ``/api/admin/route_inventory`` shape and
    any dashboard widget that consumes it."""

    assert registrar.domains() == (
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


def test_expected_routes_unknown_domain_raises(
    registrar: HarnessRouteRegistrar,
) -> None:
    with pytest.raises(KeyError, match="unknown route registrar domain"):
        registrar.expected_routes("not-a-domain")


def test_expected_routes_returns_frozenset(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = registrar.expected_routes("admin")
    assert isinstance(routes, frozenset)
    assert ("POST", "/api/admin/learning/tick") in routes
    assert ("GET", "/api/admin/route_inventory") in routes


def test_expected_all_union_matches_per_domain(
    registrar: HarnessRouteRegistrar,
) -> None:
    union = set()
    for d in registrar.domains():
        union.update(registrar.expected_routes(d))
    assert registrar.expected_all() == frozenset(union)


def test_domain_for_round_trip(registrar: HarnessRouteRegistrar) -> None:
    for d in registrar.domains():
        for key in registrar.expected_routes(d):
            assert registrar.domain_for(key) == d


def test_domain_for_unknown_returns_none(
    registrar: HarnessRouteRegistrar,
) -> None:
    assert registrar.domain_for(("GET", "/no/such/route")) is None


def _stub_app(routes: list[tuple[str, str]]) -> FastAPI:
    # Match production: Swagger UI lives on /api/docs, no ReDoc,
    # and /openapi.json + /docs/oauth2-redirect remain auto-mounted.
    app = FastAPI(docs_url="/api/docs", redoc_url=None)
    for method, path in routes:
        lower = method.lower()
        decorator = getattr(app, lower)

        @decorator(path)
        def _stub() -> dict[str, Any]:  # pragma: no cover - never called
            return {}

    return app


def test_audit_clean_app_with_full_inventory(
    registrar: HarnessRouteRegistrar,
) -> None:
    """An app with exactly the canonical inventory passes audit."""

    routes = sorted(registrar.expected_all())
    app = _stub_app(routes)
    report = registrar.audit(app)
    assert report.ok is True
    assert report.missing == ()
    assert report.unexpected == ()


def test_audit_missing_route_is_reported(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = sorted(registrar.expected_all())
    # drop a known dashboard route
    dropped: RouteKey = ("GET", "/api/dashboard/stream")
    routes.remove(dropped)
    app = _stub_app(routes)
    report = registrar.audit(app)
    assert report.ok is False
    assert ("dashboard", dropped) in report.missing
    assert report.unexpected == ()


def test_audit_unexpected_route_is_reported(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = sorted(registrar.expected_all())
    routes.append(("GET", "/api/extra/route"))
    app = _stub_app(routes)
    report = registrar.audit(app)
    assert report.ok is False
    assert report.missing == ()
    assert ("GET", "/api/extra/route") in report.unexpected


def test_audit_or_raise_passes_silently_on_clean_inventory(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = sorted(registrar.expected_all())
    app = _stub_app(routes)
    report = registrar.audit_or_raise(app)
    assert isinstance(report, RouteAuditReport)
    assert report.ok is True


def test_audit_or_raise_diagnostic_lists_missing(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = sorted(registrar.expected_all())
    routes.remove(("POST", "/api/tick"))
    app = _stub_app(routes)
    with pytest.raises(RuntimeError, match="missing routes:.*engine:POST /api/tick"):
        registrar.audit_or_raise(app)


def test_audit_or_raise_diagnostic_lists_unexpected(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = sorted(registrar.expected_all())
    routes.append(("GET", "/api/unexpected/leaf"))
    app = _stub_app(routes)
    with pytest.raises(RuntimeError, match="unexpected routes:.*GET /api/unexpected/leaf"):
        registrar.audit_or_raise(app)


def test_mounted_routes_drops_head_methods(
    registrar: HarnessRouteRegistrar,
) -> None:
    """FastAPI auto-adds HEAD next to every GET; the registrar
    must collapse to (GET, path) only so the canonical inventory
    stays method-canonical."""

    app = FastAPI()

    @app.get("/probe")
    def _probe() -> dict[str, Any]:  # pragma: no cover
        return {}

    mounted = registrar.mounted_routes(app)
    assert ("GET", "/probe") in mounted
    assert ("HEAD", "/probe") not in mounted


def test_inventory_returns_sorted_per_domain(
    registrar: HarnessRouteRegistrar,
) -> None:
    routes = sorted(registrar.expected_all())
    app = _stub_app(routes)
    inv = registrar.inventory(app)
    for d, present in inv.items():
        assert list(present) == sorted(present), f"{d} not sorted"


def test_real_server_app_passes_boot_audit() -> None:
    """End-to-end: importing ui.server runs the boot audit; if it
    raised we wouldn't reach this assertion."""

    os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")
    from ui.server import _ROUTE_REGISTRAR, app

    report = _ROUTE_REGISTRAR.audit(app)
    assert report.ok is True, (
        f"server.app inventory drifted: missing={report.missing} unexpected={report.unexpected}"
    )


def test_admin_route_inventory_endpoint_groups_by_domain() -> None:
    """The /api/admin/route_inventory endpoint mirrors the
    registrar's per-domain projection."""

    os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")
    from fastapi.testclient import TestClient

    from ui.server import app

    client = TestClient(app)
    resp = client.get("/api/admin/route_inventory")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["unexpected"] == []
    names = [d["name"] for d in payload["domains"]]
    assert names == [
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
    ]
    admin_routes = next(d for d in payload["domains"] if d["name"] == "admin")["routes"]
    assert {
        "method": "GET",
        "path": "/api/admin/route_inventory",
    } in admin_routes
