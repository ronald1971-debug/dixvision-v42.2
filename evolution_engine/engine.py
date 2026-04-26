"""EvolutionEngine — OFFLINE-ENGINE-02 (Phase E0 shell)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    OfflineEngine,
    Plugin,
)
from core.contracts.events import Event


class EvolutionEngine(OfflineEngine):
    name: str = "evolution"
    tier: EngineTier = EngineTier.OFFLINE

    def __init__(
        self,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        cron: str = "0 */6 * * *",
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )
        self._cron = cron

    def schedule(self) -> str:
        return self._cron

    def process(self, event: Event) -> Sequence[Event]:
        return ()

    def check_self(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.OK,
            detail="Phase E0 shell — DISABLED in v1 (no patch pipeline yet)",
        )
