"""
execution/confirmations/fill_tracker.py
Tracks partial + full fills and emits FILLED events to the ledger when an
order is complete.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from mind.order_manager import OrderStatus, get_order_manager
from state.ledger.writer import get_writer


@dataclass
class Fill:
    order_id: str
    filled: float
    total: float


class FillTracker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._fills: dict[str, Fill] = {}
        self._writer = get_writer()

    def record(self, order_id: str, filled: float, total: float) -> None:
        with self._lock:
            self._fills[order_id] = Fill(order_id, filled, total)
        om = get_order_manager()
        if filled >= total and total > 0:
            om.update_status(order_id, OrderStatus.FILLED, filled=filled)
            self._writer.write("MARKET", "ORDER_FILLED", "fill_tracker", {
                "order_id": order_id, "filled": filled, "total": total,
            })
        else:
            om.update_status(order_id, OrderStatus.PARTIAL, filled=filled)

    def get(self, order_id: str) -> Fill | None:
        with self._lock:
            return self._fills.get(order_id)


_ft: FillTracker | None = None
_lock = threading.Lock()


def get_fill_tracker() -> FillTracker:
    global _ft
    if _ft is None:
        with _lock:
            if _ft is None:
                _ft = FillTracker()
    return _ft
