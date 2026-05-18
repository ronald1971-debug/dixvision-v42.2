"""C-2 / P2-4 / R-1 — runtime_routes extraction regression pins.

Pins that the four PR-RT-4 endpoints live in the new
``ui/runtime_routes.py`` module (not inline in ``ui/server.py``)
and that the production app still mounts them at the same paths
with the same JSON shape. Any regression that puts these routes
back into the god-object would break these tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")


def test_runtime_routes_module_exposes_build_runtime_router() -> None:
    from ui.runtime_routes import build_runtime_router

    assert callable(build_runtime_router)


def test_runtime_routes_module_imports_no_engine_packages() -> None:
    """Pin the B7 contract — the route module is engine-isolated."""
    src = Path("ui/runtime_routes.py").read_text(encoding="utf-8")
    # No direct *_engine imports — the host injects the registrar via
    # the state_accessor callable, exactly like dashboard_routes does.
    for forbidden in (
        "from intelligence_engine",
        "from execution_engine",
        "from governance_engine",
        "from learning_engine",
        "from evolution_engine",
        "from system_engine",
        "from ui.harness",
        "from ui.server",
    ):
        assert forbidden not in src, (
            f"ui/runtime_routes.py must not import {forbidden!r} — it must stay engine-isolated."
        )


def test_runtime_router_mounts_four_pr_rt_4_routes() -> None:
    from fastapi import FastAPI

    from ui.runtime_routes import build_runtime_router

    class _Stub:
        @property
        def runtime_registrar(self) -> object:
            return _Reg()

    class _Reg:
        def declared_topology_view(self) -> dict[str, str]:
            return {"kind": "declared"}

        def active_view(self) -> dict[str, str]:
            return {"kind": "active"}

        def dormant_view(self) -> dict[str, str]:
            return {"kind": "dormant"}

        def capability_view(self, tag: str) -> dict[str, str]:
            return {"kind": "capability", "tag": tag}

    app = FastAPI()
    app.include_router(build_runtime_router(lambda: _Stub()))

    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/operator/runtime/topology" in paths
    assert "/api/operator/runtime/active" in paths
    assert "/api/operator/runtime/dormant" in paths
    assert "/api/operator/runtime/capability/{tag}" in paths


def test_runtime_router_handlers_return_registrar_views() -> None:
    pytest.importorskip("fastapi.testclient")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.runtime_routes import build_runtime_router

    class _Reg:
        def declared_topology_view(self) -> dict[str, str]:
            return {"kind": "declared", "digest": "abc"}

        def active_view(self) -> dict[str, str]:
            return {"kind": "active"}

        def dormant_view(self) -> dict[str, str]:
            return {"kind": "dormant"}

        def capability_view(self, tag: str) -> dict[str, str]:
            return {"kind": "capability", "tag": tag}

    class _Stub:
        @property
        def runtime_registrar(self) -> object:
            return _Reg()

    app = FastAPI()
    app.include_router(build_runtime_router(lambda: _Stub()))
    client = TestClient(app)

    r1 = client.get("/api/operator/runtime/topology")
    assert r1.status_code == 200
    assert r1.json() == {"kind": "declared", "digest": "abc"}

    r2 = client.get("/api/operator/runtime/active")
    assert r2.status_code == 200
    assert r2.json() == {"kind": "active"}

    r3 = client.get("/api/operator/runtime/dormant")
    assert r3.status_code == 200
    assert r3.json() == {"kind": "dormant"}

    r4 = client.get("/api/operator/runtime/capability/execution.dispatch")
    assert r4.status_code == 200
    assert r4.json() == {
        "kind": "capability",
        "tag": "execution.dispatch",
    }


def test_ui_server_no_longer_inlines_runtime_routes() -> None:
    """Regression: the four PR-RT-4 routes must not live in ui/server.py."""
    src = Path("ui/server.py").read_text(encoding="utf-8")
    # Inline decorators removed.
    assert '@app.get("/api/operator/runtime/topology")' not in src
    assert '@app.get("/api/operator/runtime/active")' not in src
    assert '@app.get("/api/operator/runtime/dormant")' not in src
    assert '@app.get("/api/operator/runtime/capability/{tag}")' not in src
    # Router is mounted instead.
    assert "build_runtime_router" in src
