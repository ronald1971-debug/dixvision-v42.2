"""LearningEngine — OFFLINE-ENGINE-01 (Phase E0 shell)."""

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


class LearningEngine(OfflineEngine):
    name: str = "learning"
    tier: EngineTier = EngineTier.OFFLINE

    def __init__(
        self,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        cron: str = "0 */1 * * *",
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )
        self._cron = cron

    def schedule(self) -> str:
        return self._cron

    def process(self, event: Event) -> Sequence[Event]:
        # Offline; runtime bus events are not consumed here.
        return ()

    def check_self(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.OK,
            detail="Phase E0 shell — DISABLED in v1 (no lanes loaded)",
        )
