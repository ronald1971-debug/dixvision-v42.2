"""IntelligenceEngine — RUNTIME-ENGINE-01 (Phase E0 shell)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    Plugin,
    RuntimeEngine,
)
from core.contracts.events import Event, SignalEvent


class IntelligenceEngine(RuntimeEngine):
    """Phase E0 stub — empty plugin slots, no signal generation."""

    name: str = "intelligence"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )

    def process(self, event: Event) -> Sequence[Event]:
        # Phase E0: pass-through. Real plugins land in Phase E2.
        # Engines must be deterministic (INV-15).
        # Only SignalEvent passes through (engine produces signals);
        # other event kinds are silently ignored at the contract layer.
        if isinstance(event, SignalEvent):
            return (event,)
        return ()

    def check_self(self) -> HealthStatus:
        plugin_states = {
            slot: {p.name: HealthState.OK for p in plugins}
            for slot, plugins in self.plugin_slots.items()
        }
        return HealthStatus(
            state=HealthState.OK,
            detail="Phase E0 shell — no plugins loaded",
            plugin_states=plugin_states,
        )
