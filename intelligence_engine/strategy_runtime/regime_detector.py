"""Market-regime detector — Phase 3 / v2-B.

A deterministic, IO-free classifier that maps a rolling window of
``MarketTick`` observations to one of four regimes:

* :attr:`MarketRegime.TRENDING_UP`   — sustained drift above tolerance
* :attr:`MarketRegime.TRENDING_DOWN` — sustained drift below tolerance
* :attr:`MarketRegime.RANGING`       — low realised volatility
* :attr:`MarketRegime.VOLATILE`      — high realised volatility

The detector holds **no clocks, no randomness, no IO** — the same tick
sequence always produces the same :class:`RegimeReading` (INV-15).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from core.contracts.market import MarketTick


class MarketRegime(StrEnum):
    UNKNOWN = "UNKNOWN"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """Snapshot of the detector at one tick."""

    regime: MarketRegime
    ts_ns: int
    symbol: str
    drift_bps: float
    volatility_bps: float
    sample_count: int


class RegimeDetector:
    """Rolling-window mid-price drift + volatility classifier.

    Args:
        window: Number of most recent mid prices retained per symbol.
        trend_threshold_bps: |drift| above this = TRENDING_*.
        volatility_threshold_bps: realised vol above this = VOLATILE
            (overrides trending — high vol takes precedence).
    """

    name: str = "regime_detector"
    spec_id: str = "IND-REG-01"

    def __init__(
        self,
        *,
        window: int = 32,
        trend_threshold_bps: float = 5.0,
        volatility_threshold_bps: float = 25.0,
    ) -> None:
        if window < 4:
            raise ValueError("window must be >= 4")
        if trend_threshold_bps < 0.0:
            raise ValueError("trend_threshold_bps must be >= 0")
        if volatility_threshold_bps < 0.0:
            raise ValueError("volatility_threshold_bps must be >= 0")
        self._window = window
        self._trend_threshold_bps = trend_threshold_bps
        self._volatility_threshold_bps = volatility_threshold_bps
        self._mids: dict[str, deque[float]] = {}

    # -- mutations ---------------------------------------------------------

    def observe(self, tick: MarketTick) -> RegimeReading:
        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.ask < tick.bid:
            return RegimeReading(
                regime=MarketRegime.UNKNOWN,
                ts_ns=tick.ts_ns,
                symbol=tick.symbol,
                drift_bps=0.0,
                volatility_bps=0.0,
                sample_count=0,
            )
        mid = 0.5 * (tick.bid + tick.ask)
        buf = self._mids.setdefault(
            tick.symbol, deque(maxlen=self._window)
        )
        buf.append(mid)
        return self._classify(tick, buf)

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._mids.clear()
        else:
            self._mids.pop(symbol, None)

    # -- internals ---------------------------------------------------------

    def _classify(
        self, tick: MarketTick, buf: deque[float]
    ) -> RegimeReading:
        n = len(buf)
        if n < 2:
            return RegimeReading(
                regime=MarketRegime.UNKNOWN,
                ts_ns=tick.ts_ns,
                symbol=tick.symbol,
                drift_bps=0.0,
                volatility_bps=0.0,
                sample_count=n,
            )

        first = buf[0]
        last = buf[-1]
        drift_bps = (last - first) / first * 10_000.0 if first > 0 else 0.0

        # Realised volatility = max-min range / mean, in bps.
        lo = min(buf)
        hi = max(buf)
        mean = sum(buf) / n
        volatility_bps = (
            (hi - lo) / mean * 10_000.0 if mean > 0 else 0.0
        )

        if volatility_bps >= self._volatility_threshold_bps:
            regime = MarketRegime.VOLATILE
        elif drift_bps >= self._trend_threshold_bps:
            regime = MarketRegime.TRENDING_UP
        elif drift_bps <= -self._trend_threshold_bps:
            regime = MarketRegime.TRENDING_DOWN
        else:
            regime = MarketRegime.RANGING

        return RegimeReading(
            regime=regime,
            ts_ns=tick.ts_ns,
            symbol=tick.symbol,
            drift_bps=drift_bps,
            volatility_bps=volatility_bps,
            sample_count=n,
        )


__all__ = ["MarketRegime", "RegimeDetector", "RegimeReading"]
