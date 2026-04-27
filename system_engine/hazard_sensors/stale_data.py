"""HAZ-04 — stale market-data sensor.

Distinct from HAZ-01 (websocket connection silence): HAZ-04 watches a
*per-symbol* tick gap. Quote feed may still be flowing for some symbols
while a specific instrument has gone stale.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity


class StaleDataSensor:
    """HAZ-04. Flags per-symbol quote staleness."""

    name: str = "stale_data"
    code: str = "HAZ-04"
    spec_id: str = "HAZ-04"
    source: str = "system_engine.hazard_sensors.stale_data"

    __slots__ = ("_max_gap_ns", "_last_ts_by_symbol", "_armed_by_symbol")

    def __init__(self, max_gap_ns: int = 2_000_000_000) -> None:
        if max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        self._max_gap_ns = max_gap_ns
        self._last_ts_by_symbol: dict[str, int] = {}
        self._armed_by_symbol: dict[str, bool] = {}

    def on_tick(self, *, symbol: str, ts_ns: int) -> None:
        self._last_ts_by_symbol[symbol] = ts_ns
        self._armed_by_symbol[symbol] = False

    def observe(self, ts_ns: int) -> tuple[HazardEvent, ...]:
        out: list[HazardEvent] = []
        for symbol, last in self._last_ts_by_symbol.items():
            gap = ts_ns - last
            if gap < self._max_gap_ns:
                continue
            if self._armed_by_symbol.get(symbol, False):
                continue
            self._armed_by_symbol[symbol] = True
            out.append(
                HazardEvent(
                    ts_ns=ts_ns,
                    code=self.code,
                    severity=HazardSeverity.MEDIUM,
                    source=self.source,
                    detail=f"{symbol} quotes stale for {gap}ns",
                    meta={"symbol": symbol, "gap_ns": str(gap)},
                )
            )
        return tuple(out)


__all__ = ["StaleDataSensor"]
