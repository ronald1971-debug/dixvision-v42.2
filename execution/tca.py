"""execution.tca \u2014 post-trade Transaction Cost Analysis.

Given a parent order + its child fills, attribute slippage/impact vs:
    - decision mid (what mid-price was when we decided?)
    - arrival mid (what mid-price was when we sent first child?)
    - VWAP (interval VWAP during the order's life)

Returns basis points in each bucket + total. No I/O.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    ts_ms: int
    qty: float
    price: float
    venue: str = ""


@dataclass(frozen=True)
class TCAReport:
    algo: str
    symbol: str
    side: str
    total_qty: float
    avg_fill: float
    decision_slip_bps: float
    arrival_slip_bps: float
    vwap_slip_bps: float
    fees_bps: float
    n_fills: int
    notes: str = ""

    def as_dict(self) -> dict:
        return self.__dict__


def _bps(a: float, b: float, side: str) -> float:
    if b == 0:
        return 0.0
    sign = 1.0 if side.lower() == "buy" else -1.0
    return round(sign * (a - b) / b * 10000.0, 2)


def analyze(*, algo: str, symbol: str, side: str,
            fills: Sequence[Fill],
            decision_mid: float,
            arrival_mid: float,
            interval_vwap: float,
            fees_bps: float = 0.0) -> TCAReport:
    total_qty = sum(f.qty for f in fills)
    if total_qty <= 0:
        return TCAReport(algo, symbol, side, 0.0, 0.0, 0.0, 0.0, 0.0,
                         round(fees_bps, 2), 0, "no_fills")
    avg = sum(f.qty * f.price for f in fills) / total_qty
    return TCAReport(
        algo=algo, symbol=symbol, side=side,
        total_qty=round(total_qty, 6),
        avg_fill=round(avg, 6),
        decision_slip_bps=_bps(avg, decision_mid, side),
        arrival_slip_bps=_bps(avg, arrival_mid, side),
        vwap_slip_bps=_bps(avg, interval_vwap, side),
        fees_bps=round(fees_bps, 2),
        n_fills=len(fills),
    )


__all__ = ["Fill", "TCAReport", "analyze"]
