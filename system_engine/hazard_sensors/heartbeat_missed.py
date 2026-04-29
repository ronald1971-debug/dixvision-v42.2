"""HAZ-07 — heartbeat missed sensor (per-engine).

The Dyon health monitors emit periodic heartbeats. If an expected
heartbeat does not arrive within ``timeout_ns`` the sensor flags it.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class HeartbeatMissedSensor:
    """HAZ-07. Tracks last-seen heartbeat per engine name."""

    name: str = "heartbeat_missed"
    code: str = "HAZ-07"
    spec_id: str = "HAZ-07"
    source: str = "system_engine.hazard_sensors.heartbeat_missed"

    __slots__ = ("_timeout_ns", "_last_by_engine", "_armed_by_engine")

    def __init__(self, timeout_ns: int = 3_000_000_000) -> None:
        if timeout_ns <= 0:
            raise ValueError("timeout_ns must be positive")
        self._timeout_ns = timeout_ns
        self._last_by_engine: dict[str, int] = {}
        self._armed_by_engine: dict[str, bool] = {}

    def on_heartbeat(self, *, engine: str, ts_ns: int) -> None:
        self._last_by_engine[engine] = ts_ns
        self._armed_by_engine[engine] = False

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        out: list[HazardEvent] = []
        for engine, last in self._last_by_engine.items():
            gap = ts_ns - last
            if gap < self._timeout_ns:
                continue
            if self._armed_by_engine.get(engine, False):
                continue
            self._armed_by_engine[engine] = True
            out.append(
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.HIGH,
                    source=self.source,
                    detail=f"engine {engine!r} missed heartbeat for {gap}ns",
                    meta={"engine": engine, "gap_ns": str(gap)},
                    produced_by_engine="system_engine",
                )
            )
        return tuple(out)


__all__ = ["HeartbeatMissedSensor"]
