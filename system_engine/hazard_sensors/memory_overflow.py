"""HAZ-05 — memory budget overflow sensor."""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class MemoryOverflowSensor:
    """HAZ-05. Caller passes RSS/heap samples in bytes."""

    name: str = "memory_overflow"
    code: str = "HAZ-05"
    spec_id: str = "HAZ-05"
    source: str = "system_engine.hazard_sensors.memory_overflow"

    __slots__ = ("_warn_bytes", "_critical_bytes", "_armed_warn", "_armed_critical")

    def __init__(
        self,
        *,
        warn_bytes: int = 512 * 1024 * 1024,
        critical_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        if warn_bytes <= 0 or critical_bytes <= 0:
            raise ValueError("budgets must be positive")
        if critical_bytes < warn_bytes:
            raise ValueError("critical_bytes must be >= warn_bytes")
        self._warn_bytes = warn_bytes
        self._critical_bytes = critical_bytes
        self._armed_warn = False
        self._armed_critical = False

    def observe(self, *, ts_ns: int, rss_bytes: int) -> tuple[HazardEvent, ...]:
        if rss_bytes >= self._critical_bytes:
            if self._armed_critical:
                return ()
            self._armed_critical = True
            return (
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.CRITICAL,
                    source=self.source,
                    detail=f"rss {rss_bytes} >= critical {self._critical_bytes}",
                    meta={"rss_bytes": str(rss_bytes)},
                    produced_by_engine="system_engine",
                ),
            )
        if rss_bytes >= self._warn_bytes:
            # Re-entry into a lower band must rearm the higher band so a
            # subsequent re-spike re-emits CRITICAL.
            self._armed_critical = False
            if self._armed_warn:
                return ()
            self._armed_warn = True
            return (
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.MEDIUM,
                    source=self.source,
                    detail=f"rss {rss_bytes} >= warn {self._warn_bytes}",
                    meta={"rss_bytes": str(rss_bytes)},
                    produced_by_engine="system_engine",
                ),
            )
        # Below the warn band — disarm both so a re-spike re-emits.
        self._armed_warn = False
        self._armed_critical = False
        return ()


__all__ = ["MemoryOverflowSensor"]
