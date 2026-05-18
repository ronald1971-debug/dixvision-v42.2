"""C-2 / P2-4 / R-1 part 4 — operator_routes extraction regression pins.

These tests pin the contract that the sixteen operator-management /
wallet / source-trust / learning-override / development-mode /
trading-allowed routes were extracted from the :mod:`ui.server`
god-object into the engine-isolated route module
:mod:`ui.operator_routes`, without changing any URL, HTTP method,
JSON shape, or operator-facing behavior.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")


def test_operator_routes_module_exposes_build_operator_router() -> None:
    from ui.operator_routes import build_operator_router

    assert callable(build_operator_router)


def test_operator_routes_module_imports_no_engine_packages() -> None:
    """Pin the B7 contract — the route module is engine-isolated.

    The route lives at the ``ui/`` tier; per ``authority_lint`` B7 it
    must not reach into engine internals or back into the harness.
    """
    src = Path("ui/operator_routes.py").read_text(encoding="utf-8")
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
        assert forbidden not in src, f"operator_routes must not import {forbidden!r}"


def test_operator_router_mounts_all_canonical_routes() -> None:
    """All sixteen endpoints must be mounted at the same URLs they
    had as inline ``@app.get/.post`` handlers in :mod:`ui.server`.
    """
    from fastapi import FastAPI

    from ui.operator_routes import build_operator_router

    app = FastAPI()
    app.include_router(build_operator_router(lambda: _StubOperatorState()))
    paths_methods: set[tuple[str, str]] = set()
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            for method in route.methods or ():
                paths_methods.add((method, route.path))
    expected: set[tuple[str, str]] = {
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
        ("GET", "/api/operator/development-mode"),
        ("POST", "/api/operator/development-mode"),
        ("GET", "/api/operator/trading-allowed"),
        ("POST", "/api/operator/trading-allowed"),
    }
    missing = expected - paths_methods
    assert not missing, f"operator router missing routes: {sorted(missing)}"


def test_operator_summary_handler_proxies_widgets() -> None:
    """``GET /api/operator/summary`` must read mode, engines, strategies
    and memecoin widgets from the state accessor — not any global.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.operator_routes import build_operator_router

    stub = _StubOperatorState()
    app = FastAPI()
    app.include_router(build_operator_router(lambda: stub))
    client = TestClient(app)

    response = client.get("/api/operator/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"]["current_mode"] == "PAPER"
    assert body["engines"] == []
    assert body["strategies"] == {
        "proposed": 0,
        "canary": 0,
        "live": 0,
        "retired": 0,
        "failed": 0,
    }
    assert body["memecoin"]["enabled"] is False
    assert body["decision_chain_count"] == 0
    assert stub.mode_widget.snapshot_calls == 1
    assert stub.engines_widget.snapshot_calls == 1


def test_operator_action_routes_validate_request_body() -> None:
    """The three action routes (kill / unlock / mode) must reject
    missing required fields with pydantic 422 — pins that the typed
    request models are still wired to FastAPI.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.operator_routes import build_operator_router

    stub = _StubOperatorState()
    app = FastAPI()
    app.include_router(build_operator_router(lambda: stub))
    client = TestClient(app)

    # /action/mode requires ``target_mode`` -> pydantic 422 on empty body.
    assert client.post("/api/operator/action/mode", json={}).status_code == 422
    # /action/kill + /action/unlock have all-defaulted bodies, so the
    # empty body validates and the stub control-plane router returns
    # a 200 with an approved=True outcome (pinned via the stub).
    for path in ("/api/operator/action/kill", "/api/operator/action/unlock"):
        response = client.post(path, json={})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["approved"] is True
        assert "summary" in body and "decision" in body


def test_source_trust_routes_validate_request_body() -> None:
    """``POST /api/operator/source-trust/promote`` + ``/demote`` must
    reject missing required fields with pydantic 422 — pins the typed
    request models stay wired to FastAPI.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.operator_routes import build_operator_router

    stub = _StubOperatorState()
    app = FastAPI()
    app.include_router(build_operator_router(lambda: stub))
    client = TestClient(app)

    assert client.post("/api/operator/source-trust/promote", json={}).status_code == 422
    assert client.post("/api/operator/source-trust/demote", json={}).status_code == 422


def test_learning_override_routes_validate_request_body() -> None:
    """``POST /api/operator/learning-override`` must reject missing
    required fields with pydantic 422 — pins LearningOverrideRequest
    is hoisted to ``core.contracts.api.operator`` and still imports.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.operator_routes import build_operator_router

    stub = _StubOperatorState()
    app = FastAPI()
    app.include_router(build_operator_router(lambda: stub))
    client = TestClient(app)

    assert client.post("/api/operator/learning-override", json={}).status_code == 422


def test_development_mode_routes_validate_request_body() -> None:
    """``POST`` on development-mode + trading-allowed must reject
    missing required fields with pydantic 422 — pins the hoisted
    DevelopmentModeRequest / TradingAllowedRequest stay wired.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.operator_routes import build_operator_router

    stub = _StubOperatorState()
    app = FastAPI()
    app.include_router(build_operator_router(lambda: stub))
    client = TestClient(app)

    assert client.post("/api/operator/development-mode", json={}).status_code == 422
    assert client.post("/api/operator/trading-allowed", json={}).status_code == 422


def test_ui_server_no_longer_inlines_operator_routes() -> None:
    """Pin the negative side of the extraction — ui/server.py must no
    longer contain inline ``@app.{get,post}("/api/operator/...")`` or
    ``@app.{get}("/api/feeds/memecoin/summary")`` /
    ``@app.{get}("/api/wallet/info")`` declarations.
    """
    src = Path("ui/server.py").read_text(encoding="utf-8")
    for forbidden in (
        '@app.get("/api/operator/summary"',
        '@app.get("/api/operator/source-trust"',
        '@app.post("/api/operator/source-trust/promote"',
        '@app.post("/api/operator/source-trust/demote"',
        '@app.get("/api/operator/learning-override"',
        '@app.post("/api/operator/learning-override"',
        '@app.get("/api/operator/development-mode"',
        '@app.post("/api/operator/development-mode"',
        '@app.get("/api/operator/trading-allowed"',
        '@app.post("/api/operator/trading-allowed"',
        '@app.get("/api/feeds/memecoin/summary"',
        '@app.get("/api/wallet/info"',
    ):
        assert forbidden not in src, (
            f"ui/server.py must no longer inline {forbidden!r}; "
            "the route belongs to ui.operator_routes"
        )
    # And it must mount the router exactly once.
    assert "build_operator_router(lambda: STATE)" in src


# ---------------------------------------------------------------------
# Stub state object — implements the read-only attributes the router
# touches plus the four mutable scalars its flip endpoints update.
# ---------------------------------------------------------------------


@dataclass
class _ModeSnap:
    current_mode: str = "PAPER"
    legal_targets: tuple[str, ...] = ()
    is_locked: bool = False


class _ModeWidget:
    snapshot_calls: int = 0

    def snapshot(self) -> _ModeSnap:
        self.snapshot_calls += 1
        return _ModeSnap()


class _EnginesWidget:
    snapshot_calls: int = 0

    def snapshot(self) -> list[Any]:
        self.snapshot_calls += 1
        return []


class _StrategiesWidget:
    def by_state(self) -> dict[str, tuple[Any, ...]]:
        return {
            "PROPOSED": (),
            "CANARY": (),
            "LIVE": (),
            "RETIRED": (),
            "FAILED": (),
        }


@dataclass
class _MemeSnap:
    enabled: bool = False
    killed: bool = False
    summary: str = ""


class _MemecoinWidget:
    def status(self) -> _MemeSnap:
        return _MemeSnap()


class _DecisionsWidget:
    def chains(self, limit: int = 200) -> list[Any]:
        _ = limit
        return []


@dataclass
class _StubOutcome:
    """Mirror :class:`dashboard_backend.control_plane.router.RouteOutcome`."""

    decision: Any = None
    summary: str = "stub"

    @property
    def approved(self) -> bool:
        return True


class _DashboardRouter:
    def submit(self, request: Any) -> _StubOutcome:
        _ = request
        return _StubOutcome(decision={"approved": True, "summary": "stub"})

    def wallet_info(self) -> dict[str, Any]:
        return {
            "address": "",
            "balance": 0.0,
            "tokens": [],
        }


class _Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def append(
        self,
        *,
        ts_ns: int,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        self.rows.append({"ts_ns": ts_ns, "kind": kind, "payload": payload})


@dataclass
class _Outcome:
    decision: Any = None
    audit_payload: dict[str, Any] = field(default_factory=dict)


class _ControlPlane:
    def submit(self, request: Any) -> _Outcome:
        _ = request
        return _Outcome(decision={"approved": True})


class _Governance:
    def __init__(self) -> None:
        self.ledger = _Ledger()
        self.control_plane = _ControlPlane()


class _Execution:
    def set_development_mode_policy(self, policy: Any) -> None:
        _ = policy


class _SignalTrustRegistry:
    def all_rows(self) -> Iterable[Any]:
        return []


class _SignalTrustPromotions:
    def promote(self, *args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        return {}

    def demote(self, *args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        return {}


class _StubOperatorState:
    """Read-only attributes + mutable scalars matching the Protocol."""

    def __init__(self) -> None:
        from core.contracts.development_mode import DevelopmentModePolicy
        from core.contracts.governance import SystemMode

        self.lock = threading.RLock()
        self.mode_widget = _ModeWidget()
        self.engines_widget = _EnginesWidget()
        self.strategies_widget = _StrategiesWidget()
        self.memecoin_widget = _MemecoinWidget()
        self.decisions_widget = _DecisionsWidget()
        self.dashboard_router = _DashboardRouter()
        self.governance = _Governance()
        self.execution = _Execution()
        self.signal_trust_registry = _SignalTrustRegistry()
        self.signal_trust_promotions = _SignalTrustPromotions()
        self.learning_override_enabled = False
        self.development_mode_enabled = True
        self.trading_allowed = False
        self.development_mode_policy = DevelopmentModePolicy(
            development_enabled=True,
            trading_allowed=False,
            mode=SystemMode.PAPER,
        )
