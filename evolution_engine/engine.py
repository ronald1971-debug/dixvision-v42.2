"""EvolutionEngine — OFFLINE-ENGINE-02 (Phase E0 shell)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    OfflineEngine,
    Plugin,
)
from core.contracts.events import Event


class EvolutionEngine(OfflineEngine):
    """Phase-E0 engine shell for the evolution lane.

    The real evolution hot path is owned by
    :class:`evolution_engine.loops.structural_loop.StructuralEvolutionLoop`,
    which the harness ticks from ``POST /api/tick``. This engine
    shell never consumes runtime bus events (``process`` is a
    no-op) and primarily exists so the engine registry has a stable
    handle to the evolution lane for ``/api/health``, the runtime
    topology authority, and operator routes.

    Phase-6 P1-3 — by default ``check_self`` now reports
    :data:`HealthState.DEGRADED`. The audit flagged that returning
    :data:`HealthState.OK` while the shell is disabled was actively
    misleading: the dormancy was already exposed at
    ``/api/operator/runtime/dormant`` but ``/api/health`` was still
    reporting OK. Callers may pass an ``is_active_fn`` (e.g. a
    closure around the wired
    :class:`StructuralEvolutionLoop`) to report OK iff the loop is
    actually unfrozen, which closes the audit gap fully when the
    loop is wired.
    """

    name: str = "evolution"
    tier: EngineTier = EngineTier.OFFLINE

    def __init__(
        self,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        cron: str = "0 */6 * * *",
        is_active_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(plugin_slots or {})
        self._cron = cron
        self._is_active_fn = is_active_fn

    def schedule(self) -> str:
        return self._cron

    def process(self, event: Event) -> Sequence[Event]:
        return ()

    def check_self(self) -> HealthStatus:
        if self._is_active_fn is None:
            return HealthStatus(
                state=HealthState.DEGRADED,
                detail=(
                    "Phase E0 shell — dormant; see "
                    "/api/operator/runtime/dormant for the active "
                    "evolution loop state"
                ),
            )
        if self._is_active_fn():
            return HealthStatus(
                state=HealthState.OK,
                detail="StructuralEvolutionLoop unfrozen — active",
            )
        return HealthStatus(
            state=HealthState.DEGRADED,
            detail=(
                "StructuralEvolutionLoop wired but freeze policy "
                "is currently engaged — see /api/operator/"
                "runtime/dormant"
            ),
        )
