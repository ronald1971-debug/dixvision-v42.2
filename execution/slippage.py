"""execution.slippage \u2014 pre-trade slippage + market-impact estimator.

Linear model (Almgren-Chriss permanent-impact reduced form) as default:
    exp_slip_bps = spread_bps / 2 + kappa * (order_qty / ADV)

kappa defaults to 10 bps per 1% ADV, tunable per venue. Caller may supply
volatility_bps_daily for a volatility-scaled variant:
    exp_slip_bps = spread_bps / 2 + kappa * (order_qty / ADV)
                   + lambda_vol * vol_bps * sqrt(duration_sec / 86400)

All outputs are rounded to 1 bps. Zero allocations beyond ints/floats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageEstimate:
    exp_slippage_bps: float
    spread_component_bps: float
    impact_component_bps: float
    vol_component_bps: float
    participation_pct: float
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "exp_slippage_bps": self.exp_slippage_bps,
            "spread_component_bps": self.spread_component_bps,
            "impact_component_bps": self.impact_component_bps,
            "vol_component_bps": self.vol_component_bps,
            "participation_pct": self.participation_pct,
            "note": self.note,
        }


def estimate(*, qty: float, adv_qty: float, spread_bps: float,
             kappa: float = 10.0,
             lambda_vol: float = 0.5,
             vol_bps_daily: float = 0.0,
             duration_sec: float = 0.0) -> SlippageEstimate:
    """Return an Almgren-Chriss-lite estimate in bps."""
    if qty <= 0 or adv_qty <= 0:
        return SlippageEstimate(0.0, 0.0, 0.0, 0.0, 0.0, "invalid_inputs")
    part = max(0.0, qty / adv_qty)
    spread = max(0.0, spread_bps) / 2.0
    impact = kappa * (part * 100.0) / 100.0                      # kappa per 1% ADV
    vol_comp = 0.0
    if vol_bps_daily > 0 and duration_sec > 0:
        vol_comp = lambda_vol * vol_bps_daily * math.sqrt(
            max(0.0, duration_sec / 86400.0))
    total = round(spread + impact + vol_comp, 1)
    return SlippageEstimate(
        exp_slippage_bps=total,
        spread_component_bps=round(spread, 1),
        impact_component_bps=round(impact, 1),
        vol_component_bps=round(vol_comp, 1),
        participation_pct=round(part * 100.0, 3),
    )


def min_acceptable_price(*, mid: float, side: str, exp_slip_bps: float,
                         max_extra_bps: float = 20.0) -> float:
    """Derive a minAcceptable (for buys) / maxAcceptable (for sells) price."""
    if mid <= 0:
        return 0.0
    bps_total = exp_slip_bps + max_extra_bps
    shift = mid * (bps_total / 10000.0)
    return mid + shift if side.lower() == "buy" else mid - shift


__all__ = ["SlippageEstimate", "estimate", "min_acceptable_price"]
