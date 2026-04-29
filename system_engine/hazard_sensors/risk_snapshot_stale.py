"""HAZ-08 — fast risk cache snapshot stale sensor.

Risk snapshots monotonically advance their ``version_id``. If the
version stays unchanged longer than ``max_age_ns`` we emit HAZ-08;
the execution hot-path uses risk gates that depend on a fresh
snapshot, so a frozen cache is a high-severity hazard.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class RiskSnapshotStaleSensor:
    """HAZ-08."""

    name: str = "risk_snapshot_stale"
    code: str = "HAZ-08"
    spec_id: str = "HAZ-08"
    source: str = "system_engine.hazard_sensors.risk_snapshot_stale"

    __slots__ = ("_max_age_ns", "_last_version", "_last_change_ts", "_armed")

    def __init__(self, max_age_ns: int = 1_000_000_000) -> None:
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        self._max_age_ns = max_age_ns
        self._last_version: int | None = None
        self._last_change_ts: int | None = None
        self._armed = False

    def observe(
        self,
        *,
        ts_ns: int,
        version_id: int,
    ) -> tuple[HazardEvent, ...]:
        if self._last_version is None or version_id != self._last_version:
            self._last_version = version_id
            self._last_change_ts = ts_ns
            self._armed = False
            return ()
        assert self._last_change_ts is not None
        age = ts_ns - self._last_change_ts
        if age < self._max_age_ns or self._armed:
            return ()
        self._armed = True
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.HIGH,
                source=self.source,
                detail=f"risk snapshot v{version_id} stale for {age}ns",
                meta={"version_id": str(version_id), "age_ns": str(age)},
                produced_by_engine="system_engine",
            ),
        )


__all__ = ["RiskSnapshotStaleSensor"]
