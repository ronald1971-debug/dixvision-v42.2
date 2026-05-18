"""C-2 / P2-4 / R-1 part 3 — cognitive_routes extraction regression pins.

Pins the contract that the five cognitive-chat operator routes
(``/api/cognitive/chat/{status,turn,approvals,approvals/.../approve,
approvals/.../reject}``) were extracted from the :mod:`ui.server`
god-object into the engine-isolated :mod:`ui.cognitive_routes`
module without changing any URL, HTTP method, request body,
response shape, or HTTP status code.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")


def test_cognitive_routes_module_exposes_build_cognitive_router() -> None:
    from ui.cognitive_routes import build_cognitive_router

    assert callable(build_cognitive_router)


def test_cognitive_routes_module_engine_isolation_contract() -> None:
    """B7-style: the route module must not import the harness or
    full ``ui.server`` _State type. It is allowed to depend on
    ``intelligence_engine.cognitive`` and the runtime façade in
    ``ui.cognitive_chat_runtime`` (its only legitimate consumers).
    """
    src = Path("ui/cognitive_routes.py").read_text(encoding="utf-8")
    for forbidden in (
        "from ui.server",
        "from ui.harness",
        "from execution_engine",
        "from governance_engine",
        "from learning_engine",
        "from evolution_engine",
        "from system_engine",
    ):
        assert forbidden not in src, f"cognitive_routes must not import {forbidden!r}"


def test_cognitive_router_mounts_all_canonical_routes() -> None:
    """All five endpoints must be mounted at the same URLs they had
    as inline ``@app.get/.post`` handlers in :mod:`ui.server`.
    """
    from fastapi import FastAPI

    from ui.cognitive_routes import build_cognitive_router

    class _StubRuntime:
        approval_queue: Any = None

        def status(self) -> Any:
            return None

        def turn(self, body: Any) -> Any:
            return None

    class _StubEdge:
        def approve(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def reject(self, *args: Any, **kwargs: Any) -> Any:
            return None

    class _Stub:
        lock = threading.Lock()
        chat_runtime = _StubRuntime()
        approval_edge = _StubEdge()

    app = FastAPI()
    app.include_router(build_cognitive_router(lambda: _Stub()))
    methods_paths = {
        (next(iter(r.methods)), r.path)
        for r in app.routes
        if hasattr(r, "path") and hasattr(r, "methods") and r.methods
    }
    expected = {
        ("GET", "/api/cognitive/chat/status"),
        ("POST", "/api/cognitive/chat/turn"),
        ("GET", "/api/cognitive/chat/approvals"),
        ("POST", "/api/cognitive/chat/approvals/{request_id}/approve"),
        ("POST", "/api/cognitive/chat/approvals/{request_id}/reject"),
    }
    missing = expected - methods_paths
    assert not missing, f"cognitive router missing routes: {sorted(missing)}"


def test_cognitive_status_handler_proxies_runtime_status() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from core.contracts.api.cognitive_chat import ChatStatusResponse
    from ui.cognitive_routes import build_cognitive_router

    @dataclass
    class _StubRuntime:
        status_calls: int = 0

        def status(self) -> ChatStatusResponse:
            self.status_calls += 1
            return ChatStatusResponse(
                enabled=True,
                eligible_providers=(),
                feature_flag_env_var="DIX_COGNITIVE_CHAT_ENABLED",
            )

        def turn(self, body: Any) -> Any:
            return None

    runtime = _StubRuntime()

    class _Stub:
        lock = threading.Lock()
        chat_runtime = runtime
        approval_edge = object()

    app = FastAPI()
    app.include_router(build_cognitive_router(lambda: _Stub()))
    client = TestClient(app)

    r = client.get("/api/cognitive/chat/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enabled"] is True
    assert runtime.status_calls == 1


def test_cognitive_turn_maps_runtime_exceptions_to_status_codes() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from core.contracts.api.cognitive_chat import ChatRoleApi
    from ui.cognitive_chat_runtime import (
        ChatTurnDisabled,
        ChatTurnNoProvider,
        ChatTurnTransportFailed,
    )
    from ui.cognitive_routes import build_cognitive_router

    class _ExcRuntime:
        def __init__(self, exc: BaseException) -> None:
            self._exc = exc

        def status(self) -> Any:
            return None

        def turn(self, body: Any) -> Any:
            raise self._exc

    class _Stub:
        def __init__(self, exc: BaseException) -> None:
            self.lock = threading.Lock()
            self.chat_runtime = _ExcRuntime(exc)
            self.approval_edge = object()

    body = {
        "messages": [
            {"role": ChatRoleApi.USER.value, "content": "hi"},
        ],
        "thread_id": "t-1",
    }

    for exc, expected in (
        (ChatTurnDisabled("off"), 503),
        (ChatTurnNoProvider("no providers"), 502),
        (ChatTurnTransportFailed("dial timeout"), 502),
        (ValueError("bad shape"), 400),
    ):
        stub = _Stub(exc)
        app = FastAPI()
        app.include_router(build_cognitive_router(lambda s=stub: s))
        client = TestClient(app)
        r = client.post("/api/cognitive/chat/turn", json=body)
        assert r.status_code == expected, (
            f"{type(exc).__name__} -> {r.status_code} (expected {expected}); body={r.text}"
        )


def test_cognitive_approve_maps_edge_exceptions_to_status_codes() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from intelligence_engine.cognitive.approval_edge import (
        ApprovalAlreadyDecidedError,
        ApprovalNotFoundError,
    )
    from ui.cognitive_routes import build_cognitive_router

    class _Edge:
        def __init__(self, exc: BaseException) -> None:
            self._exc = exc

        def approve(self, **_kwargs: Any) -> Any:
            raise self._exc

        def reject(self, **_kwargs: Any) -> Any:
            raise self._exc

    class _Stub:
        def __init__(self, exc: BaseException) -> None:
            self.lock = threading.Lock()
            self.chat_runtime = object()
            self.approval_edge = _Edge(exc)

    for exc, expected in (
        (ApprovalNotFoundError("missing"), 404),
        (ApprovalAlreadyDecidedError("dup"), 409),
    ):
        stub = _Stub(exc)
        app = FastAPI()
        app.include_router(build_cognitive_router(lambda s=stub: s))
        client = TestClient(app)
        r = client.post("/api/cognitive/chat/approvals/req-1/approve")
        assert r.status_code == expected, (
            f"approve {type(exc).__name__} -> {r.status_code} (expected {expected})"
        )
        r = client.post("/api/cognitive/chat/approvals/req-1/reject")
        assert r.status_code == expected, (
            f"reject {type(exc).__name__} -> {r.status_code} (expected {expected})"
        )


def test_ui_server_no_longer_inlines_cognitive_routes() -> None:
    """Regression: the five cognitive chat routes must not live in
    :mod:`ui.server` any more. They are mounted via
    :func:`ui.cognitive_routes.build_cognitive_router`.
    """
    src = Path("ui/server.py").read_text(encoding="utf-8")
    for forbidden in (
        '@app.get("/api/cognitive/chat/status"',
        '@app.post("/api/cognitive/chat/turn"',
        '"/api/cognitive/chat/approvals",',
        '"/api/cognitive/chat/approvals/{request_id}/approve"',
        '"/api/cognitive/chat/approvals/{request_id}/reject"',
    ):
        assert forbidden not in src, f"ui/server.py still inlines: {forbidden!r}"
    assert "build_cognitive_router" in src
