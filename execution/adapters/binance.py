"""
execution/adapters/binance.py
DIX VISION v42.2 — Binance Exchange Adapter

DOMAIN: INDIRA only. Dyon cannot import this module.
"""
from __future__ import annotations

from typing import Any

from execution.adapters.base import BaseAdapter
from state.ledger.event_store import append_event


class BinanceAdapter(BaseAdapter):
    """
    Exchange adapter for binance.
    All calls are logged to the event ledger.
    """
    name = "binance"
    category = "CEX"
    trading_forms = frozenset({"SPOT", "MARGIN", "PERP", "FUTURES", "OPTIONS"})
    order_types = frozenset({"MARKET", "LIMIT", "STOP", "STOP_LIMIT", "OCO", "TWAP"})

    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False

    def connect(self) -> bool:
        # TODO: establish WebSocket + REST connection
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def place_order(self, symbol: str, side: str, size: float,
                    order_type: str = "MARKET") -> dict[str, Any]:
        """Place order. Returns order result dict."""
        # TODO: implement live order placement
        result = {"order_id": f"MOCK_{symbol}_{side}_{size:.4f}",
                   "symbol": symbol, "side": side, "size": size,
                   "status": "FILLED", "filled_price": 0.0}
        append_event("MARKET", "ORDER_PLACED", "binance", result)
        return result

    def cancel_order(self, order_id: str) -> bool:
        # TODO: implement live order cancellation
        append_event("MARKET", "ORDER_CANCELLED", "binance",
                     {"order_id": order_id})
        return True

    def get_balance(self, asset: str = "USDT") -> float:
        # TODO: query live balance
        return 100_000.0

    def is_connected(self) -> bool:
        return self._connected
