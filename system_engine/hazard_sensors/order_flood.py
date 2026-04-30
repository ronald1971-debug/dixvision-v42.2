"""HAZ-09 — order flood sensor (rate-limit guard)."""

from __future__ import annotations

from collections import deque

from core.contracts.events import HazardEvent, HazardSeverity


class OrderFloodSensor:
    """HAZ-09. Caps orders within a sliding time window."""

    name: str = "order_flood"
    code: str = "HAZ-09"
    spec_id: str = "HAZ-09"
    source: str = "system_engine.hazard_sensors.order_flood"

    __slots__ = ("_window_ns", "_max_orders", "_orders", "_armed")

    def __init__(
        self,
        *,
        window_ns: int = 1_000_000_000,
        max_orders: int = 50,
    ) -> None:
        if window_ns <= 0:
            raise ValueError("window_ns must be positive")
        if max_orders < 1:
            raise ValueError("max_orders must be >= 1")
        self._window_ns = window_ns
        self._max_orders = max_orders
        self._orders: deque[int] = deque()
        self._armed = False

    def record_order(self, ts_ns: int) -> None:
        self._orders.append(ts_ns)

    def _evict(self, ts_ns: int) -> None:
        cutoff = ts_ns - self._window_ns
        while self._orders and self._orders[0] < cutoff:
            self._orders.popleft()

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        self._evict(ts_ns)
        n = len(self._orders)
        if n <= self._max_orders:
            self._armed = False
            return ()
        if self._armed:
            return ()
        self._armed = True
        return (
            HazardEvent(
                ts_ns=ts_ns,
                code=self.code,
                severity=HazardSeverity.HIGH,
                source=self.source,
                detail=f"{n} orders in last {self._window_ns}ns > cap {self._max_orders}",
                meta={"orders": str(n), "cap": str(self._max_orders)},
                produced_by_engine="system_engine",
            ),
        )


__all__ = ["OrderFloodSensor"]
