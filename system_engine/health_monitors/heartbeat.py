"""Heartbeat monitor — last-seen timestamp per engine."""

from __future__ import annotations


class HeartbeatMonitor:
    """Deterministic last-seen tracker. No clocks; caller passes ts_ns."""

    name: str = "heartbeat_monitor"
    spec_id: str = "SYS-HEALTH-HB-01"

    __slots__ = ("_last",)

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def record(self, *, engine: str, ts_ns: int) -> None:
        if not engine:
            raise ValueError("engine name must be non-empty")
        prev = self._last.get(engine)
        if prev is not None and ts_ns < prev:
            raise ValueError("heartbeat ts_ns must be monotonic per engine")
        self._last[engine] = ts_ns

    def last_seen(self, engine: str) -> int | None:
        return self._last.get(engine)

    def snapshot(self) -> dict[str, int]:
        return dict(self._last)


__all__ = ["HeartbeatMonitor"]
