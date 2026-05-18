"""C-2 / P2-4 / R-1 part 2 — feeds_routes extraction regression pins.

These tests pin the contract that the four live-feed route families
(Binance / CoinDesk / Pump.fun / Raydium) were extracted from the
:mod:`ui.server` god-object into the engine-isolated route module
:mod:`ui.feeds_routes`, without changing any URL, HTTP method,
JSON shape, or operator-facing behavior.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")


def test_feeds_routes_module_exposes_build_feeds_router() -> None:
    from ui.feeds_routes import BinanceFeedStartIn, build_feeds_router

    assert callable(build_feeds_router)
    # BinanceFeedStartIn is the only request body and must remain a
    # pydantic v2 BaseModel so FastAPI can introspect it.
    assert hasattr(BinanceFeedStartIn, "model_validate")


def test_feeds_routes_module_imports_no_engine_packages() -> None:
    """Pin the B7 contract — the route module is engine-isolated."""
    src = Path("ui/feeds_routes.py").read_text(encoding="utf-8")
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
        assert forbidden not in src, f"feeds_routes must not import {forbidden!r}"


def test_feeds_router_mounts_all_canonical_routes() -> None:
    """The full set of feed endpoints must be mounted at the same URLs
    they had as inline ``@app.get/.post`` handlers in :mod:`ui.server`.
    """
    from fastapi import FastAPI

    from ui.feeds_routes import build_feeds_router

    @dataclass
    class _StubStatus:
        running: bool = False
        url: str = ""
        symbols: tuple[str, ...] = ()
        ticks_received: int = 0
        items_received: int = 0
        launches_received: int = 0
        snapshots_emitted: int = 0
        errors: int = 0
        last_tick_ts_ns: int = 0
        last_poll_ts_ns: int = 0
        last_event_ts_ns: int = 0

    class _StubFeed:
        def status(self) -> _StubStatus:
            return _StubStatus()

        def start(self, *args: Any, **kwargs: Any) -> None:
            return None

        def stop(self) -> None:
            return None

    class _Stub:
        lock = threading.Lock()
        binance_feed = _StubFeed()
        coindesk_feed = _StubFeed()
        pumpfun_feed = _StubFeed()
        raydium_feed = _StubFeed()
        recent_launches: deque[Any] = deque()
        recent_pool_snapshots: deque[Any] = deque()

    app = FastAPI()
    app.include_router(build_feeds_router(lambda: _Stub()))
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    expected = {
        "/api/feeds/binance/start",
        "/api/feeds/binance/stop",
        "/api/feeds/binance/status",
        "/api/feeds/coindesk/start",
        "/api/feeds/coindesk/stop",
        "/api/feeds/coindesk/status",
        "/api/feeds/pumpfun/start",
        "/api/feeds/pumpfun/stop",
        "/api/feeds/pumpfun/status",
        "/api/feeds/pumpfun/recent",
        "/api/feeds/raydium/start",
        "/api/feeds/raydium/stop",
        "/api/feeds/raydium/status",
        "/api/feeds/raydium/recent",
    }
    missing = expected - paths
    assert not missing, f"feeds router missing routes: {sorted(missing)}"


def test_feeds_router_handlers_proxy_to_runners() -> None:
    """The route handlers must read from the runner methods. Verified
    by counting calls on a stub runner per route family.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.feeds_routes import build_feeds_router

    @dataclass
    class _Status:
        running: bool = True
        url: str = "wss://stub"
        symbols: tuple[str, ...] = ("btcusdt",)
        ticks_received: int = 7
        items_received: int = 3
        launches_received: int = 11
        snapshots_emitted: int = 5
        errors: int = 0
        last_tick_ts_ns: int = 1
        last_poll_ts_ns: int = 2
        last_event_ts_ns: int = 3

    class _Counting:
        def __init__(self) -> None:
            self.start_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            self.stop_calls = 0
            self.status_calls = 0

        def status(self) -> _Status:
            self.status_calls += 1
            return _Status()

        def start(self, *args: Any, **kwargs: Any) -> None:
            self.start_calls.append((args, kwargs))

        def stop(self) -> None:
            self.stop_calls += 1

    class _Stub:
        lock = threading.Lock()
        binance_feed = _Counting()
        coindesk_feed = _Counting()
        pumpfun_feed = _Counting()
        raydium_feed = _Counting()
        recent_launches: deque[dict[str, Any]] = deque([{"mint": "A"}, {"mint": "B"}])
        recent_pool_snapshots: deque[dict[str, Any]] = deque([{"pair": "SOL-USDC"}])

    state = _Stub()
    app = FastAPI()
    app.include_router(build_feeds_router(lambda: state))
    client = TestClient(app)

    for family_attr, prefix in (
        ("binance_feed", "binance"),
        ("coindesk_feed", "coindesk"),
        ("pumpfun_feed", "pumpfun"),
        ("raydium_feed", "raydium"),
    ):
        feed: _Counting = getattr(state, family_attr)
        r = client.get(f"/api/feeds/{prefix}/status")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "feed" in body
        assert body["feed"]["running"] is True

        r = client.post(f"/api/feeds/{prefix}/start")
        assert r.status_code == 200
        assert r.json()["started"] is True
        assert len(feed.start_calls) == 1

        r = client.post(f"/api/feeds/{prefix}/stop")
        assert r.status_code == 200
        assert r.json()["stopped"] is True
        assert feed.stop_calls == 1

    r = client.get("/api/feeds/pumpfun/recent?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["launches"] == [{"mint": "A"}, {"mint": "B"}]

    r = client.get("/api/feeds/raydium/recent?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["snapshots"] == [{"pair": "SOL-USDC"}]


def test_binance_start_accepts_symbols_override() -> None:
    """The ``symbols`` override on Binance/start must round-trip through
    the FastAPI body parser into the runner ``start(symbols=...)`` call.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ui.feeds_routes import build_feeds_router

    @dataclass
    class _Status:
        running: bool = False
        url: str = "wss://stub"
        symbols: tuple[str, ...] = ()
        ticks_received: int = 0
        errors: int = 0
        last_tick_ts_ns: int = 0

    seen: dict[str, Any] = {}

    class _Binance:
        def status(self) -> _Status:
            return _Status()

        def start(self, *, symbols: Iterable[str] | None = None) -> None:
            seen["symbols"] = None if symbols is None else list(symbols)

        def stop(self) -> None:
            return None

    class _Other:
        def status(self) -> _Status:
            return _Status()

        def start(self, *args: Any, **kwargs: Any) -> None:
            return None

        def stop(self) -> None:
            return None

    class _Stub:
        lock = threading.Lock()
        binance_feed = _Binance()
        coindesk_feed = _Other()
        pumpfun_feed = _Other()
        raydium_feed = _Other()
        recent_launches: deque[Any] = deque()
        recent_pool_snapshots: deque[Any] = deque()

    app = FastAPI()
    app.include_router(build_feeds_router(lambda: _Stub()))
    client = TestClient(app)

    r = client.post(
        "/api/feeds/binance/start",
        json={"symbols": ["btcusdt", "ethusdt", "solusdt"]},
    )
    assert r.status_code == 200, r.text
    assert seen["symbols"] == ["btcusdt", "ethusdt", "solusdt"]

    seen.clear()
    r = client.post("/api/feeds/binance/start", json={})
    assert r.status_code == 200, r.text
    assert seen.get("symbols") is None


def test_ui_server_no_longer_inlines_feed_routes() -> None:
    """Regression: the four feed-route families must not live in
    :mod:`ui.server` any more. They are mounted via
    :func:`ui.feeds_routes.build_feeds_router`.
    """
    src = Path("ui/server.py").read_text(encoding="utf-8")
    for forbidden in (
        '@app.post("/api/feeds/binance/start")',
        '@app.post("/api/feeds/binance/stop")',
        '@app.get("/api/feeds/binance/status")',
        '@app.post("/api/feeds/coindesk/start")',
        '@app.post("/api/feeds/coindesk/stop")',
        '@app.get("/api/feeds/coindesk/status")',
        '@app.post("/api/feeds/pumpfun/start")',
        '@app.post("/api/feeds/pumpfun/stop")',
        '@app.get("/api/feeds/pumpfun/status")',
        '@app.get("/api/feeds/pumpfun/recent")',
        '@app.post("/api/feeds/raydium/start")',
        '@app.post("/api/feeds/raydium/stop")',
        '@app.get("/api/feeds/raydium/status")',
        '@app.get("/api/feeds/raydium/recent")',
    ):
        assert forbidden not in src, f"ui/server.py still inlines: {forbidden!r}"
    assert "build_feeds_router" in src
