"""C-2 / P2-4 / R-1 part 2 — live-feed operator HTTP surface.

The four read-only / start-stop feed adapters shipped over wave-01
through D2 (Binance public WS, CoinDesk RSS, Pump.fun launches,
Raydium pool poller) each surface three or four HTTP endpoints under
``/api/feeds/{venue}/...``. They originally lived inline in
:mod:`ui.server`; this module is the canonical extraction.

Authority constraints (B7 / dashboard isolation):

* This module does not import any ``*_engine`` package. The feed
  *runners* are SCVS-registered data adapters owned by
  :mod:`ui.feeds`; the host (``ui.server``) constructs them at boot
  and the harness ``_State`` exposes them as ``binance_feed``,
  ``coindesk_feed``, ``pumpfun_feed``, ``raydium_feed`` attributes
  plus the two recent-event rings (``recent_launches``,
  ``recent_pool_snapshots``). This module never writes the ledger
  or constructs governance decisions.
* The route module receives the harness state via a
  :class:`Protocol`-typed callable so it stays decoupled from the
  concrete ``_State`` type — same pattern as
  :mod:`ui.dashboard_routes`, :mod:`ui.governance_routes`,
  :mod:`ui.execution_routes`, and the part-1 PR-RT-4 extraction in
  :mod:`ui.runtime_routes`.

The four extracted families preserve their operator-facing URLs,
HTTP methods, and JSON shapes **verbatim**. No client-side
dashboard change is required.

Endpoints mounted:

* ``POST /api/feeds/binance/start``  — start the Binance public WS pump
* ``POST /api/feeds/binance/stop``   — stop the Binance pump
* ``GET  /api/feeds/binance/status`` — Binance pump telemetry

* ``POST /api/feeds/coindesk/start``  — start the CoinDesk RSS poller
* ``POST /api/feeds/coindesk/stop``   — stop the CoinDesk poller
* ``GET  /api/feeds/coindesk/status`` — CoinDesk poller telemetry

* ``POST /api/feeds/pumpfun/start``   — start the Pump.fun WS pump
* ``POST /api/feeds/pumpfun/stop``    — stop the Pump.fun pump
* ``GET  /api/feeds/pumpfun/status``  — Pump.fun pump telemetry
* ``GET  /api/feeds/pumpfun/recent``  — most recent launches (newest first)

* ``POST /api/feeds/raydium/start``   — start the Raydium pool poller
* ``POST /api/feeds/raydium/stop``    — stop the Raydium poller
* ``GET  /api/feeds/raydium/status``  — Raydium poller telemetry
* ``GET  /api/feeds/raydium/recent``  — most recent pool snapshots
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class BinanceFeedStartIn(BaseModel):
    """Optional override for ``POST /api/feeds/binance/start``.

    If ``symbols`` is omitted, the pump uses its module-default
    symbol set (``ui.feeds.binance_public_ws.DEFAULT_SYMBOLS``:
    BTCUSDT + ETHUSDT).
    """

    symbols: list[str] | None = Field(
        default=None,
        description="Override symbol list, e.g. ['btcusdt', 'ethusdt', 'solusdt']",
    )


class _FeedStateLike(Protocol):
    """Read-only accessor the host installs into the FastAPI app.

    Only the attributes the four feed-route families touch are
    declared here — the route module never sees the full ``_State``
    surface and therefore stays clean of the cross-cutting concerns
    held by the harness.
    """

    @property
    def lock(self) -> Lock: ...

    @property
    def binance_feed(self) -> Any: ...

    @property
    def coindesk_feed(self) -> Any: ...

    @property
    def pumpfun_feed(self) -> Any: ...

    @property
    def raydium_feed(self) -> Any: ...

    @property
    def recent_launches(self) -> Sequence[Any]: ...

    @property
    def recent_pool_snapshots(self) -> Sequence[Any]: ...


def build_feeds_router(
    state_accessor: Callable[[], _FeedStateLike],
) -> APIRouter:
    """Construct the operator ``/api/feeds`` router.

    Args:
        state_accessor: callable returning the harness state object
            that exposes the four feed runners and the two recent-event
            rings. Invoked lazily on every request so the router does
            not bind the state at module-import time.
    """

    router = APIRouter(prefix="/api/feeds", tags=["feeds"])

    # ------------------------------------------------------------------
    # Binance (SRC-MARKET-BINANCE-001) — read-only public WS pump
    # ------------------------------------------------------------------

    def _binance_status_dict() -> dict[str, Any]:
        status = state_accessor().binance_feed.status()
        return {
            "source_id": "SRC-MARKET-BINANCE-001",
            "running": status.running,
            "url": status.url,
            "symbols": list(status.symbols),
            "ticks_received": status.ticks_received,
            "errors": status.errors,
            "last_tick_ts_ns": status.last_tick_ts_ns,
        }

    @router.post("/binance/start")
    def post_binance_feed_start(
        body: BinanceFeedStartIn | None = None,
    ) -> dict[str, Any]:
        """Start the read-only Binance public WS pump (SRC-MARKET-BINANCE-001).

        Idempotent — returns the current status if already running. Pass
        ``{"symbols": ["btcusdt", "ethusdt", "solusdt"]}`` to override the
        default symbol set for this run.
        """
        symbols = body.symbols if body is not None else None
        try:
            state_accessor().binance_feed.start(symbols=symbols)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"started": True, "feed": _binance_status_dict()}

    @router.post("/binance/stop")
    def post_binance_feed_stop() -> dict[str, Any]:
        """Stop the Binance public WS pump.

        Idempotent — returns the current status if not running.
        """
        state_accessor().binance_feed.stop()
        return {"stopped": True, "feed": _binance_status_dict()}

    @router.get("/binance/status")
    def get_binance_feed_status() -> dict[str, Any]:
        """Return a telemetry snapshot of the Binance public WS pump."""
        return {"feed": _binance_status_dict()}

    # ------------------------------------------------------------------
    # CoinDesk RSS news poller (SRC-NEWS-COINDESK-001)
    # ------------------------------------------------------------------

    def _coindesk_status_dict() -> dict[str, Any]:
        status = state_accessor().coindesk_feed.status()
        return {
            "source_id": "SRC-NEWS-COINDESK-001",
            "running": status.running,
            "url": status.url,
            "items_received": status.items_received,
            "errors": status.errors,
            "last_poll_ts_ns": status.last_poll_ts_ns,
        }

    @router.post("/coindesk/start")
    def post_coindesk_feed_start() -> dict[str, Any]:
        """Start the read-only CoinDesk RSS news poller (SRC-NEWS-COINDESK-001).

        Idempotent — returns the current status if already running. The
        poller emits :class:`NewsItem` events through the NewsFanout
        pipeline (P0-5) into both the news-shock hazard sensor and the
        news-to-signal projection.
        """
        state_accessor().coindesk_feed.start()
        return {"started": True, "feed": _coindesk_status_dict()}

    @router.post("/coindesk/stop")
    def post_coindesk_feed_stop() -> dict[str, Any]:
        """Stop the CoinDesk RSS news poller.

        Idempotent — returns the current status if not running.
        """
        state_accessor().coindesk_feed.stop()
        return {"stopped": True, "feed": _coindesk_status_dict()}

    @router.get("/coindesk/status")
    def get_coindesk_feed_status() -> dict[str, Any]:
        """Return a telemetry snapshot of the CoinDesk RSS news poller."""
        return {"feed": _coindesk_status_dict()}

    # ------------------------------------------------------------------
    # Pump.fun launches (SRC-LAUNCH-PUMPFUN-001) — D2
    # ------------------------------------------------------------------

    def _pumpfun_status_dict() -> dict[str, Any]:
        status = state_accessor().pumpfun_feed.status()
        return {
            "source_id": "SRC-LAUNCH-PUMPFUN-001",
            "running": status.running,
            "url": status.url,
            "launches_received": status.launches_received,
            "errors": status.errors,
            "last_event_ts_ns": status.last_event_ts_ns,
        }

    @router.post("/pumpfun/start")
    def post_pumpfun_feed_start() -> dict[str, Any]:
        """Start the read-only Pump.fun WS launch pump (D2).

        Connects to the Pump.fun launch WS endpoint and emits one
        :class:`LaunchEvent` per new mint into the ``recent_launches``
        ring exposed by ``GET /api/feeds/pumpfun/recent``.
        """
        state_accessor().pumpfun_feed.start()
        return {"started": True, "feed": _pumpfun_status_dict()}

    @router.post("/pumpfun/stop")
    def post_pumpfun_feed_stop() -> dict[str, Any]:
        """Stop the Pump.fun launch pump. Idempotent."""
        state_accessor().pumpfun_feed.stop()
        return {"stopped": True, "feed": _pumpfun_status_dict()}

    @router.get("/pumpfun/status")
    def get_pumpfun_feed_status() -> dict[str, Any]:
        """Return a telemetry snapshot of the Pump.fun WS pump."""
        return {"feed": _pumpfun_status_dict()}

    @router.get("/pumpfun/recent")
    def get_pumpfun_recent(limit: int = 50) -> dict[str, Any]:
        """Return the most recent Pump.fun launches (newest first)."""
        cap = max(1, min(int(limit), 200))
        state = state_accessor()
        with state.lock:
            launches = list(state.recent_launches)[:cap]
        return {
            "launches": launches,
            "count": len(launches),
            "feed": _pumpfun_status_dict(),
        }

    # ------------------------------------------------------------------
    # Raydium AMM pool poller (SRC-POOL-RAYDIUM-001) — D2
    # ------------------------------------------------------------------

    def _raydium_status_dict() -> dict[str, Any]:
        status = state_accessor().raydium_feed.status()
        return {
            "source_id": "SRC-POOL-RAYDIUM-001",
            "running": status.running,
            "url": status.url,
            "snapshots_emitted": status.snapshots_emitted,
            "errors": status.errors,
            "last_poll_ts_ns": status.last_poll_ts_ns,
        }

    @router.post("/raydium/start")
    def post_raydium_feed_start() -> dict[str, Any]:
        """Start the read-only Raydium AMM pool poller (D2).

        Polls ``https://api.raydium.io/v2/main/pairs`` on a fixed
        interval and emits one :class:`PoolSnapshot` per pair into the
        ``recent_pool_snapshots`` ring exposed by
        ``GET /api/feeds/raydium/recent``.
        """
        state_accessor().raydium_feed.start()
        return {"started": True, "feed": _raydium_status_dict()}

    @router.post("/raydium/stop")
    def post_raydium_feed_stop() -> dict[str, Any]:
        """Stop the Raydium pool poller. Idempotent."""
        state_accessor().raydium_feed.stop()
        return {"stopped": True, "feed": _raydium_status_dict()}

    @router.get("/raydium/status")
    def get_raydium_feed_status() -> dict[str, Any]:
        """Return a telemetry snapshot of the Raydium pool poller."""
        return {"feed": _raydium_status_dict()}

    @router.get("/raydium/recent")
    def get_raydium_recent(limit: int = 100) -> dict[str, Any]:
        """Return the most recent Raydium pool snapshots (newest first)."""
        cap = max(1, min(int(limit), 500))
        state = state_accessor()
        with state.lock:
            snaps = list(state.recent_pool_snapshots)[:cap]
        return {
            "snapshots": snaps,
            "count": len(snaps),
            "feed": _raydium_status_dict(),
        }

    return router


__all__ = ["BinanceFeedStartIn", "build_feeds_router"]
