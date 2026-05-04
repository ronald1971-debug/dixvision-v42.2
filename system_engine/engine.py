"""SystemEngine — RUNTIME-ENGINE-03.

AUDIT-P1.3 — :meth:`process` now dispatches every *pollable* hazard
sensor on the bound :class:`SensorArray` and emits the resulting
:class:`HazardEvent` rows. A pollable sensor is one whose
``observe(ts_ns: int)`` is the only signature the engine can drive
generically; sensors with bespoke ``observe(...)`` shapes (e.g.
``memory_overflow`` taking ``rss_bytes``) are dispatched through
their dedicated ingestion paths (NewsFanout, runtime monitor, etc.)
and intentionally skipped here.

Pure-Python, IO-free, clock-free — the engine reads ``ts_ns`` only
from the inbound :class:`Event`.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    Plugin,
    RuntimeEngine,
)
from core.contracts.events import Event, HazardEvent
from system_engine.hazard_sensors.sensor_array import (
    HazardSensor,
    SensorArray,
)


def _is_pollable(sensor: HazardSensor) -> bool:
    """Return ``True`` when ``sensor.observe`` has signature ``(ts_ns)``.

    The check is intentionally strict: the sole non-``self`` parameter
    must be named ``ts_ns`` and accept either positional-or-keyword
    or keyword-only binding. Sensors with multi-arg or *args/**kwargs
    signatures are dispatched through their dedicated paths and are
    *not* polled by :meth:`SystemEngine.process`. Pinning the name
    keeps INV-15 replay determinism — every poll uses the same
    canonical timestamp source the engine read off the bus event.
    """

    observe = getattr(sensor, "observe", None)
    if observe is None or not callable(observe):
        return False
    try:
        sig = inspect.signature(observe)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if len(params) != 1:
        return False
    p = params[0]
    if p.name != "ts_ns":
        return False
    return p.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


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
        # AUDIT-WIRE.4 + P1-3 — bind the SensorArray primitive that
        # carries the frozen HAZ-XX sensors. WIRE.4 closed the
        # "primitive built but never bound" gap; P1-3 closes the
        # "bound but never invoked" gap by polling the pollable
        # sensors on every :meth:`process` call.
        self.sensor_array: SensorArray | None = sensor_array
        self._pollable: tuple[HazardSensor, ...] = ()
        if sensor_array is not None:
            self._pollable = tuple(
                s for s in sensor_array.sensors if _is_pollable(s)
            )

    def process(self, event: Event) -> Sequence[HazardEvent]:
        """Dispatch pollable sensors with ``event.ts_ns``.

        Returns the union of every sensor's :class:`HazardEvent`
        emissions in registration order. Sensors arm-once (they hold
        an internal ``_armed_*`` map keyed by the offending entity)
        so repeated polls do not double-emit while the underlying
        condition persists.

        Sensors whose ``observe`` signature is not the canonical
        ``(ts_ns)`` shape — e.g. ``memory_overflow`` requiring
        ``rss_bytes``, ``clock_drift`` requiring an external clock
        sample — are *intentionally* skipped here. They are fed via
        their dedicated paths (the harness runtime monitor, the
        NewsFanout, etc.) and surface their hazards through the
        same governance ingestion seam.
        """

        if not self._pollable:
            return ()
        ts_ns = int(getattr(event, "ts_ns", 0))
        out: list[HazardEvent] = []
        for sensor in self._pollable:
            # Bind ``ts_ns`` by keyword so sensors with
            # ``def observe(self, *, ts_ns)`` (KEYWORD_ONLY) are
            # also dispatchable — the detector explicitly admits
            # both POSITIONAL_OR_KEYWORD and KEYWORD_ONLY shapes.
            out.extend(sensor.observe(ts_ns=ts_ns))
        return tuple(out)

    def check_self(self) -> HealthStatus:
        if self.sensor_array is None:
            return HealthStatus(
                state=HealthState.OK,
                detail="Phase E0 shell — no sensors loaded",
            )
        return HealthStatus(
            state=HealthState.OK,
            detail=(
                f"sensor_array online "
                f"({len(self.sensor_array)} sensors, "
                f"{len(self._pollable)} pollable)"
            ),
        )
