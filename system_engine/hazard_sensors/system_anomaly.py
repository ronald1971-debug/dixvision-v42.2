"""HAZ-12 — system / process anomaly sensor (CPU, FD, GC).

Caller passes resource samples; the sensor enforces upper-bound
budgets. Pure-Python, IO-free.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class SystemAnomalySensor:
    """HAZ-12."""

    name: str = "system_anomaly"
    code: str = "HAZ-12"
    spec_id: str = "HAZ-12"
    source: str = "system_engine.hazard_sensors.system_anomaly"

    __slots__ = ("_max_cpu_pct", "_max_open_fds", "_armed")

    def __init__(
        self,
        *,
        max_cpu_pct: float = 90.0,
        max_open_fds: int = 4096,
    ) -> None:
        if not 0.0 < max_cpu_pct <= 100.0:
            raise ValueError("max_cpu_pct must be in (0, 100]")
        if max_open_fds <= 0:
            raise ValueError("max_open_fds must be positive")
        self._max_cpu_pct = max_cpu_pct
        self._max_open_fds = max_open_fds
        self._armed: dict[str, bool] = {"cpu": False, "fd": False}

    def observe(
        self,
        *,
        ts_ns: int,
        cpu_pct: float,
        open_fds: int,
    ) -> tuple[HazardEvent, ...]:
        out: list[HazardEvent] = []
        if cpu_pct > self._max_cpu_pct and not self._armed["cpu"]:
            self._armed["cpu"] = True
            out.append(
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.MEDIUM,
                    source=self.source,
                    detail=f"cpu {cpu_pct:.1f}% > {self._max_cpu_pct:.1f}%",
                    meta={"resource": "cpu", "value": f"{cpu_pct:.4f}"},
                )
            )
        elif cpu_pct <= self._max_cpu_pct:
            self._armed["cpu"] = False
        if open_fds > self._max_open_fds and not self._armed["fd"]:
            self._armed["fd"] = True
            out.append(
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.HIGH,
                    source=self.source,
                    detail=f"open_fds {open_fds} > {self._max_open_fds}",
                    meta={"resource": "fds", "value": str(open_fds)},
                )
            )
        elif open_fds <= self._max_open_fds:
            self._armed["fd"] = False
        return tuple(out)


__all__ = ["SystemAnomalySensor"]
