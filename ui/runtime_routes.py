"""C-2 / P2-4 / R-1 — runtime-topology operator HTTP surface.

Extracted from ``ui/server.py`` as the first step of the
ui-routes split (R-1). Hosts the four PR-RT-4 endpoints that
project the canonical runtime topology authority
(:class:`ui.harness.HarnessRuntimeRegistrar`) over HTTP:

* ``GET /api/operator/runtime/topology``  — the declared graph
* ``GET /api/operator/runtime/active``    — the actually-active subgraph
* ``GET /api/operator/runtime/dormant``   — declared-but-dormant nodes
* ``GET /api/operator/runtime/capability/{tag}`` — capability resolver

Authority constraints (B7 lint):

* This module imports neither ``*_engine`` packages nor any harness
  building blocks. It receives the registrar through a
  ``state_accessor`` callable that the host (``ui.server``) provides;
  the route module stays decoupled from the engine wiring exactly
  the same way :mod:`ui.dashboard_routes` and :mod:`ui.governance_routes`
  do.
* Every endpoint is GET-only. No FSM mutation, no ledger write.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from fastapi import APIRouter


class _RuntimeStateLike(Protocol):
    """Read-only accessor the host installs into the FastAPI app.

    The protocol exists so this module never imports the concrete
    harness ``_State`` type — it only knows there is an attribute
    ``runtime_registrar`` exposing the four projection methods. This
    keeps the route module decoupled from the engine wiring in
    :mod:`ui.server` and protects the B7 dashboard / B8 system-intent
    isolation rules.
    """

    @property
    def runtime_registrar(self) -> Any: ...


def build_runtime_router(
    state_accessor: Callable[[], _RuntimeStateLike],
) -> APIRouter:
    """Construct the read-only ``/api/operator/runtime`` router.

    Args:
        state_accessor: Zero-arg callable returning the host's
            ``_State`` instance. The route handlers resolve it on
            every request so the host can rebuild ``_State`` (e.g.
            in tests) without reconstructing the router.

    Returns:
        An :class:`fastapi.APIRouter` with the four PR-RT-4 routes
        mounted under ``/api/operator/runtime``.
    """

    router = APIRouter(
        prefix="/api/operator/runtime",
        tags=["operator", "runtime"],
    )

    @router.get("/topology")
    def operator_runtime_topology() -> dict[str, Any]:
        """PR-RT-4 — declared runtime topology projection.

        Returns the canonical declared topology (nodes + edges +
        INV-15 digest) as registered at boot by
        :class:`HarnessRuntimeRegistrar`. Read-only; the declared
        topology is a frozen constant of the harness build and the
        response is byte-stable across runs for a given build
        (driven by the same canonical serialization that backs the
        INV-15 digest).
        """

        return state_accessor().runtime_registrar.declared_topology_view()

    @router.get("/active")
    def operator_runtime_active() -> dict[str, Any]:
        """PR-RT-4 — actually-active runtime topology projection.

        Returns the subset of declared nodes whose ``_State``
        backing attribute resolved to a non-``None`` instance at
        boot (i.e. nodes that the harness actually wired in). This
        is the answer to "what is *actually* running right now?" —
        the inverse of :func:`operator_runtime_dormant`. Read-only.
        """

        return state_accessor().runtime_registrar.active_view()

    @router.get("/dormant")
    def operator_runtime_dormant() -> dict[str, Any]:
        """PR-RT-4 — declared-but-dormant runtime topology projection.

        Returns the subset of declared nodes whose ``_State``
        backing attribute was ``None`` (or missing) at boot. This
        is the silent-drift surface — every entry here is a node
        the architecture declares but the harness did not bring
        online. The PR-RT-5 ``tools/total_validation.py`` invariant
        pins this set against an explicit ``DECLARED_BUT_DORMANT``
        allow-list so new dormant components cannot accumulate
        without operator awareness. Read-only.
        """

        return state_accessor().runtime_registrar.dormant_view()

    @router.get("/capability/{tag}")
    def operator_runtime_capability(tag: str) -> dict[str, Any]:
        """PR-RT-4 — capability → provider resolution.

        Given a capability tag (e.g. ``intelligence.signal``,
        ``execution.dispatch``, ``learning.closed_loop``), returns
        the declared providers and which are actually active vs
        dormant. Surfaces "available-but-unwired" optimizers
        explicitly: a capability with ``declared=[X]`` but
        ``active=[]`` is the canonical silent-drift signal.
        Read-only.
        """

        return state_accessor().runtime_registrar.capability_view(tag)

    return router


__all__ = ["build_runtime_router"]
