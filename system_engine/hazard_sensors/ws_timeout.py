"""HAZ-01 — websocket feed timeout sensor.

Pure-Python, IO-free, deterministic. The caller informs the sensor of
each tick arrival; the sensor's ``observe(ts_ns)`` checks whether the
gap since the last reported tick exceeds ``timeout_ns`` and, if so,
emits a single :class:`HazardEvent`.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class WSTimeoutSensor:
    """HAZ-01. Detects websocket feed silence past a tolerance window."""

    name: str = "ws_timeout"
    code: str = "HAZ-01"
    spec_id: str = "HAZ-01"
    source: str = "system_engine.hazard_sensors.ws_timeout"

    __slots__ = ("_timeout_ns", "_last_tick_ns", "_armed")

    def __init__(self, timeout_ns: int = 5_000_000_000) -> None:
        if timeout_ns <= 0:
            raise ValueError("timeout_ns must be positive")
        self._timeout_ns = timeout_ns
        self._last_tick_ns: int | None = None
        self._armed = False

    def on_tick(self, ts_ns: int) -> None:
        """Record that a feed tick was received at ``ts_ns``."""

        self._last_tick_ns = ts_ns
        self._armed = False

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        """Check at a clock pulse whether the feed has timed out.

        Emits exactly once per timeout episode (rearms only after the
        next ``on_tick``).
        """

        if self._last_tick_ns is None or self._armed:
            return ()
        gap = ts_ns - self._last_tick_ns
        if gap < self._timeout_ns:
            return ()
        self._armed = True
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.HIGH,
                source=self.source,
                detail=f"ws feed silent for {gap}ns",
                meta={"gap_ns": str(gap)},
                produced_by_engine="system_engine",
            ),
        )


__all__ = ["WSTimeoutSensor"]
