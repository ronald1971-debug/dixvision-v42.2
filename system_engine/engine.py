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
from system_engine.hazard_sensors.sensor_array import SensorArray


class SystemEngine(RuntimeEngine):
    name: str = "system"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        *,
        sensor_array: SensorArray | None = None,
    ) -> None:
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )
        # AUDIT-WIRE.4 / P1-3 — bind the SensorArray primitive that
        # carries the 12 frozen HAZ-XX sensors. Per-sensor dispatch
        # routing into ``process`` is a follow-up wave (each sensor
        # has a heterogeneous ``observe(...)`` shape, so the array
        # is held publicly for the harness + dashboard surfaces to
        # invoke individual sensors and feed results back through
        # ``sensor_array.collect(...)``); this PR closes the
        # "primitive built but never bound" gap.
        self.sensor_array: SensorArray | None = sensor_array

    def process(self, event: Event) -> Sequence[Event]:
        # Phase E0: hazard/health pipelines arrive in Phase E3/E4.
        return ()

    def check_self(self) -> HealthStatus:
        if self.sensor_array is None:
            return HealthStatus(
                state=HealthState.OK,
                detail="Phase E0 shell — no sensors loaded",
            )
        return HealthStatus(
            state=HealthState.OK,
            detail=f"sensor_array online ({len(self.sensor_array)} sensors)",
        )
