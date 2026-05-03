"""D1 — operator-facing execution-adapter HTTP surface (read-only).

Exposes the :class:`AdapterRegistry` snapshot as JSON so the operator
dashboard can render an :code:`AdapterStatusGrid` showing which live
venues are reachable, which are still in scaffold mode, and which have
been halted by the operator.

Authority lint: only imports :mod:`execution_engine.adapters` (no
plugin or hot-path imports). B7-clean.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from execution_engine.adapters import default_registry


def build_execution_router() -> APIRouter:
    """Construct the read-only /api/execution router."""
    router = APIRouter(prefix="/api/execution", tags=["execution"])

    @router.get("/adapters")
    def list_adapters() -> dict[str, Any]:
        reg = default_registry()
        snap = reg.snapshot()
        return {
            "count": len(snap),
            "adapters": [
                {
                    "name": s.name,
                    "venue": s.venue,
                    "state": s.state.value,
                    "detail": s.detail,
                    "last_heartbeat_ns": s.last_heartbeat_ns,
                }
                for s in snap
            ],
        }

    return router


__all__ = ["build_execution_router"]
