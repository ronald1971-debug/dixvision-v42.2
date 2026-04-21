"""execution.algos \u2014 slicing engines for parent orders.

VWAP: follow an intraday volume curve (synthetic if no history).
TWAP: equal slices over a time window.
Iceberg: show a small visible size, refill as each piece fills.
POV: participate at X% of observed market volume.

All algos are *planners* \u2014 they return a schedule of child orders. The
adapter_router actually places them via the CEX/DEX adapter. This keeps
execution/authority boundaries clean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Tuple


@dataclass(frozen=True)
class ChildOrder:
    seq: int
    offset_sec: float
    qty: float
    price_hint: float = 0.0          # 0 == market
    note: str = ""


@dataclass(frozen=True)
class AlgoPlan:
    algo: str
    symbol: str
    side: str
    total_qty: float
    window_sec: float
    children: tuple[ChildOrder, ...]

    def as_dict(self) -> dict:
        return {"algo": self.algo, "symbol": self.symbol, "side": self.side,
                "total_qty": self.total_qty, "window_sec": self.window_sec,
                "n_children": len(self.children),
                "children": [c.__dict__ for c in self.children]}


def _equal_slices(total: float, n: int) -> list[float]:
    if n <= 0:
        return []
    base = total / n
    out = [base] * n
    out[-1] = total - sum(out[:-1])
    return out


# --- VWAP ---------------------------------------------------------------

# simple U-curve volume profile (higher at open + close)
_VWAP_BUCKETS = (1.8, 1.2, 0.9, 0.7, 0.6, 0.6, 0.7, 0.9, 1.0, 1.2, 1.4, 1.8)


def plan_vwap(*, symbol: str, side: str, qty: float, window_sec: float) -> AlgoPlan:
    buckets = len(_VWAP_BUCKETS)
    weights = _VWAP_BUCKETS
    total = sum(weights)
    children: list[ChildOrder] = []
    step = window_sec / buckets
    for i, w in enumerate(weights):
        children.append(ChildOrder(
            seq=i, offset_sec=i * step, qty=qty * (w / total),
            note=f"vwap_bucket {i + 1}/{buckets}",
        ))
    return AlgoPlan("VWAP", symbol, side, qty, window_sec, tuple(children))


# --- TWAP ---------------------------------------------------------------

def plan_twap(*, symbol: str, side: str, qty: float, window_sec: float,
              slices: int = 12) -> AlgoPlan:
    slices = max(1, int(slices))
    step = window_sec / slices
    qtys = _equal_slices(qty, slices)
    children = tuple(
        ChildOrder(seq=i, offset_sec=i * step, qty=q,
                   note=f"twap_slice {i + 1}/{slices}")
        for i, q in enumerate(qtys))
    return AlgoPlan("TWAP", symbol, side, qty, window_sec, children)


# --- Iceberg ------------------------------------------------------------

def plan_iceberg(*, symbol: str, side: str, qty: float, show_size: float,
                 refill_sec: float = 3.0) -> AlgoPlan:
    if show_size <= 0:
        raise ValueError("show_size must be > 0")
    n = max(1, int(math.ceil(qty / show_size)))
    qtys = _equal_slices(qty, n)
    children = tuple(
        ChildOrder(seq=i, offset_sec=i * refill_sec, qty=q,
                   note=f"iceberg_slice {i + 1}/{n}")
        for i, q in enumerate(qtys))
    return AlgoPlan("ICEBERG", symbol, side, qty, n * refill_sec, children)


# --- POV (Percentage Of Volume) -----------------------------------------

def plan_pov(*, symbol: str, side: str, qty: float,
             observed_volume_per_sec: float,
             participation: float = 0.1,
             window_sec: float = 0.0) -> AlgoPlan:
    """Schedule child orders sized to X% of observed market volume.

    If window_sec <= 0, infer from participation: total_time = qty / (vol * p).
    """
    if observed_volume_per_sec <= 0 or participation <= 0:
        raise ValueError("volume/participation must be > 0")
    step_qty = observed_volume_per_sec * participation
    step_sec = 1.0
    n = max(1, int(math.ceil(qty / step_qty)))
    if window_sec > 0:
        step_sec = window_sec / n
    qtys = _equal_slices(qty, n)
    children = tuple(
        ChildOrder(seq=i, offset_sec=i * step_sec, qty=q,
                   note=f"pov_{participation:.0%} {i + 1}/{n}")
        for i, q in enumerate(qtys))
    return AlgoPlan("POV", symbol, side, qty, n * step_sec, children)


__all__ = ["ChildOrder", "AlgoPlan",
           "plan_vwap", "plan_twap", "plan_iceberg", "plan_pov"]
