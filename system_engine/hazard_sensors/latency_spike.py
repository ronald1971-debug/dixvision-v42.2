"""HAZ-06 — latency spike sensor (rolling window over per-call samples)."""

from __future__ import annotations

from collections import deque

from core.contracts.events import HazardEvent, HazardSeverity


class LatencySpikeSensor:
    """HAZ-06. Detects sustained latency over a budget across a window."""

    name: str = "latency_spike"
    code: str = "HAZ-06"
    spec_id: str = "HAZ-06"
    source: str = "system_engine.hazard_sensors.latency_spike"

    __slots__ = ("_budget_ns", "_window", "_breach_quota", "_samples", "_armed")

    def __init__(
        self,
        *,
        budget_ns: int = 10_000_000,
        window: int = 32,
        breach_quota: int = 8,
    ) -> None:
        if budget_ns <= 0:
            raise ValueError("budget_ns must be positive")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not (0 < breach_quota <= window):
            raise ValueError("breach_quota must satisfy 0 < q <= window")
        self._budget_ns = budget_ns
        self._window = window
        self._breach_quota = breach_quota
        self._samples: deque[bool] = deque(maxlen=window)
        self._armed = False

    def record_sample(self, latency_ns: int) -> None:
        self._samples.append(latency_ns > self._budget_ns)

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        if len(self._samples) < self._window:
            return ()
        breaches = sum(1 for s in self._samples if s)
        if breaches < self._breach_quota:
            self._armed = False
            return ()
        if self._armed:
            return ()
        self._armed = True
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.MEDIUM,
                source=self.source,
                detail=f"latency breaches {breaches}/{self._window} > budget {self._budget_ns}ns",
                meta={"breaches": str(breaches), "window": str(self._window)},
            ),
        )


__all__ = ["LatencySpikeSensor"]
