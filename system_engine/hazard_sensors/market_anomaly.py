"""HAZ-11 — market-microstructure anomaly sensor.

Detects two cheap classes of book anomaly per-symbol:

* spread blow-out: ``(ask - bid) / mid`` exceeds ``max_spread_bps``;
* price jump: tick-over-tick mid move exceeds ``max_jump_bps``.

Stateless across symbols (each symbol has its own latched mid). All
parameters tunable; defaults conservative.
"""

from __future__ import annotations

from core.contracts.events import HazardEvent, HazardSeverity
from core.contracts.market import MarketTick


class MarketAnomalySensor:
    """HAZ-11."""

    name: str = "market_anomaly"
    code: str = "HAZ-11"
    spec_id: str = "HAZ-11"
    source: str = "system_engine.hazard_sensors.market_anomaly"

    __slots__ = ("_max_spread_bps", "_max_jump_bps", "_last_mid")

    def __init__(
        self,
        *,
        max_spread_bps: float = 50.0,
        max_jump_bps: float = 200.0,
    ) -> None:
        if max_spread_bps <= 0 or max_jump_bps <= 0:
            raise ValueError("thresholds must be positive")
        self._max_spread_bps = max_spread_bps
        self._max_jump_bps = max_jump_bps
        self._last_mid: dict[str, float] = {}

    def on_tick(self, tick: MarketTick) -> tuple[HazardEvent, ...]:
        mid = (tick.bid + tick.ask) / 2.0
        out: list[HazardEvent] = []
        if mid > 0 and tick.ask >= tick.bid:
            spread_bps = (tick.ask - tick.bid) / mid * 10_000.0
            if spread_bps > self._max_spread_bps:
                out.append(
                    HazardEvent(
                        ts_ns=tick.ts_ns,
                        code=self.code,
                        severity=HazardSeverity.MEDIUM,
                        source=self.source,
                        detail=(
                            f"{tick.symbol} spread {spread_bps:.1f}bps "
                            f"> {self._max_spread_bps:.1f}bps"
                        ),
                        meta={
                            "symbol": tick.symbol,
                            "kind": "spread",
                            "spread_bps": f"{spread_bps:.4f}",
                        },
                        produced_by_engine="system_engine",
                    )
                )
        prev = self._last_mid.get(tick.symbol)
        if prev is not None and prev > 0 and mid > 0:
            jump_bps = abs(mid - prev) / prev * 10_000.0
            if jump_bps > self._max_jump_bps:
                out.append(
                    HazardEvent(
                        ts_ns=tick.ts_ns,
                        code=self.code,
                        severity=HazardSeverity.HIGH,
                        source=self.source,
                        detail=f"{tick.symbol} mid jumped {jump_bps:.1f}bps",
                        meta={
                            "symbol": tick.symbol,
                            "kind": "jump",
                            "jump_bps": f"{jump_bps:.4f}",
                        },
                        produced_by_engine="system_engine",
                    )
                )
        self._last_mid[tick.symbol] = mid
        return tuple(out)


__all__ = ["MarketAnomalySensor"]
