"""HAZ-10 — runtime circuit-breaker-open sensor.

Bridges the execution_engine ``RuntimeMonitor`` (Phase 2) into a
HazardEvent so Dyon can observe and Governance can decide. Sensor is
input-only; it does not call into execution_engine (INV-08/INV-11).
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class RuntimeBreakerOpenSensor:
    """HAZ-10."""

    name: str = "runtime_breaker_open"
    code: str = "HAZ-10"
    spec_id: str = "HAZ-10"
    source: str = "system_engine.hazard_sensors.runtime_breaker_open"

    __slots__ = ("_open_scopes",)

    def __init__(self) -> None:
        self._open_scopes: dict[str, bool] = {}

    def report_open(self, *, scope: str, ts_ns: int) -> tuple[HazardEvent, ...]:
        if self._open_scopes.get(scope, False):
            return ()
        self._open_scopes[scope] = True
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.CRITICAL,
                source=self.source,
                detail=f"circuit breaker OPEN for scope {scope!r}",
                meta={"scope": scope},
                produced_by_engine="system_engine",
            ),
        )

    def report_closed(self, *, scope: str) -> None:
        self._open_scopes.pop(scope, None)

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        # State is only emitted on the open transition; observe() yields
        # nothing on idle ticks. Kept on the protocol for symmetry.
        return ()


__all__ = ["RuntimeBreakerOpenSensor"]
