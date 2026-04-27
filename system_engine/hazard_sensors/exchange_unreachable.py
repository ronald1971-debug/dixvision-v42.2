"""HAZ-02 — exchange/venue unreachable sensor.

The execution engine reports adapter outcomes (``record_attempt``) and
the sensor decides when consecutive failures cross a threshold worth a
hazard. Deterministic, IO-free.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class ExchangeUnreachableSensor:
    """HAZ-02. Tracks consecutive adapter failures per venue."""

    name: str = "exchange_unreachable"
    code: str = "HAZ-02"
    spec_id: str = "HAZ-02"
    source: str = "system_engine.hazard_sensors.exchange_unreachable"

    __slots__ = ("_threshold", "_failures", "_armed")

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._failures: dict[str, int] = {}
        self._armed: dict[str, bool] = {}

    def record_attempt(self, venue: str, *, ok: bool) -> None:
        if ok:
            self._failures.pop(venue, None)
            self._armed.pop(venue, None)
        else:
            self._failures[venue] = self._failures.get(venue, 0) + 1
            # Re-arm if the streak just started fresh.
            if self._failures[venue] == 1:
                self._armed[venue] = False

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        out: list[HazardEvent] = []
        for venue, count in self._failures.items():
            if count < self._threshold or self._armed.get(venue, False):
                continue
            self._armed[venue] = True
            out.append(
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.HIGH,
                    source=self.source,
                    detail=f"venue {venue!r} unreachable: {count} consecutive failures",
                    meta={"venue": venue, "failures": str(count)},
                )
            )
        return tuple(out)


__all__ = ["ExchangeUnreachableSensor"]
