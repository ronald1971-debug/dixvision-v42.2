"""HAZ-03 — clock drift sensor.

The TimeAuthority is the only source of monotonic ``ts_ns``. This
sensor compares two reference clocks (e.g. monotonic vs wall) and
emits HAZ-03 when their difference exceeds ``tolerance_ns``.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class ClockDriftSensor:
    """HAZ-03. Detects clock drift between two named clock readings."""

    name: str = "clock_drift"
    code: str = "HAZ-03"
    spec_id: str = "HAZ-03"
    source: str = "system_engine.hazard_sensors.clock_drift"

    __slots__ = ("_tolerance_ns", "_armed")

    def __init__(self, tolerance_ns: int = 50_000_000) -> None:
        if tolerance_ns <= 0:
            raise ValueError("tolerance_ns must be positive")
        self._tolerance_ns = tolerance_ns
        self._armed = False

    def observe(
        self,
        *,
        ts_ns: int,
        reference_ns: int,
        sample_ns: int,
    ) -> tuple[HazardEvent, ...]:
        drift = abs(sample_ns - reference_ns)
        if drift <= self._tolerance_ns:
            self._armed = False
            return ()
        if self._armed:
            return ()
        self._armed = True
        sev = (
            HazardSeverity.CRITICAL
            if drift > self._tolerance_ns * 4
            else HazardSeverity.HIGH
        )
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=sev,
                source=self.source,
                detail=f"clock drift {drift}ns > tolerance {self._tolerance_ns}ns",
                meta={"drift_ns": str(drift)},
            ),
        )


__all__ = ["ClockDriftSensor"]
