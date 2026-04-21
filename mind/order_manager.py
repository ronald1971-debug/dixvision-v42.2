"""
mind/order_manager.py
Tracks in-flight orders and their terminal state (NEW → LIVE → FILLED/REJECTED).
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum


class OrderStatus(str, Enum):
    NEW = "NEW"
    LIVE = "LIVE"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


@dataclass
class Order:
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    client_id: str = ""
    asset: str = ""
    side: str = "BUY"
    size: float = 0.0
    price: float = 0.0
    status: OrderStatus = OrderStatus.NEW
    filled_size: float = 0.0
    created_at: str = ""
    last_update: str = ""


class OrderManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orders: dict[str, Order] = {}

    def submit(self, o: Order) -> Order:
        with self._lock:
            self._orders[o.order_id] = o
        return o

    def update_status(
        self, order_id: str, status: OrderStatus, filled: float = 0.0
    ) -> Order | None:
        with self._lock:
            o = self._orders.get(order_id)
            if o is None:
                return None
            o.status = status
            if filled > o.filled_size:
                o.filled_size = filled
            return o

    def open_orders(self) -> list[Order]:
        with self._lock:
            return [o for o in self._orders.values()
                    if o.status in {OrderStatus.NEW, OrderStatus.LIVE, OrderStatus.PARTIAL}]


_om: OrderManager | None = None
_lock = threading.Lock()


def get_order_manager() -> OrderManager:
    global _om
    if _om is None:
        with _lock:
            if _om is None:
                _om = OrderManager()
    return _om
