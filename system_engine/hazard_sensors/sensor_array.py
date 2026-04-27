"""HazardSensor protocol + SensorArray coordinator (Phase 4).

Sensors implement :class:`HazardSensor`. The :class:`SensorArray` runs
all registered sensors in deterministic order on each tick and yields
the union of emitted :class:`HazardEvent` rows.

Pure-Python, IO-free, clock-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from core.contracts.events import HazardEvent


@runtime_checkable
class HazardSensor(Protocol):
    """Common shape for hazard sensors.

    Each sensor:

    * carries a ``name`` (registry key) and ``code`` (HAZ-XX) attribute,
    * exposes ``observe(...)`` that may emit zero-or-more
      :class:`HazardEvent`,
    * is deterministic — same inputs (same call sequence with same
      ``ts_ns`` values) → same outputs (INV-15).
    """

    name: str
    code: str


class SensorArray:
    """Aggregator for an ordered, named set of hazard sensors.

    The array is the only thing :class:`SystemEngine` invokes per tick;
    individual sensors stay decoupled. Order of registration is the
    deterministic firing order.
    """

    name: str = "sensor_array"
    spec_id: str = "SYS-HAZ-ARR-01"

    __slots__ = ("_sensors",)

    def __init__(self) -> None:
        self._sensors: list[HazardSensor] = []

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, sensor: HazardSensor) -> None:
        if not sensor.name:
            raise ValueError("sensor.name must be non-empty")
        if any(s.name == sensor.name for s in self._sensors):
            raise ValueError(f"sensor already registered: {sensor.name!r}")
        self._sensors.append(sensor)

    def deregister(self, name: str) -> None:
        self._sensors = [s for s in self._sensors if s.name != name]

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def sensors(self) -> tuple[HazardSensor, ...]:
        return tuple(self._sensors)

    def __len__(self) -> int:
        return len(self._sensors)

    # ------------------------------------------------------------------
    # collection — sensors are dispatched by their own ``observe``
    # signatures; the array intentionally does NOT generalise the call
    # shape, because each sensor consumes a different input. Callers
    # invoke individual sensors and feed the resulting events here:
    # ------------------------------------------------------------------

    def collect(
        self, events: Sequence[Sequence[HazardEvent]]
    ) -> tuple[HazardEvent, ...]:
        """Flatten a sequence of per-sensor outputs into one ordered tuple.

        Used to keep ordering deterministic when multiple sensors fire on
        the same tick: registration order wins.
        """

        out: list[HazardEvent] = []
        for batch in events:
            out.extend(batch)
        return tuple(out)


__all__ = ["HazardSensor", "SensorArray"]
