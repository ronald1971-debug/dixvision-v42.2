"""SystemEngine — RUNTIME-ENGINE-03 (Phase E0 shell)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    Plugin,
    RuntimeEngine,
)
from core.contracts.events import Event


class SystemEngine(RuntimeEngine):
    name: str = "system"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )

    def process(self, event: Event) -> Sequence[Event]:
        # Phase E0: hazard/health pipelines arrive in Phase E3/E4.
        return ()

    def check_self(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.OK,
            detail="Phase E0 shell — no sensors loaded",
        )
