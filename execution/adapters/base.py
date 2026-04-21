"""
execution/adapters/base.py
Shared protocol + base class for exchange adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Canonical trading-form identifiers (matches the manifest §2 surface).
TRADING_FORMS = (
    "SPOT",        # spot market
    "MARGIN",      # cross/isolated margin
    "PERP",        # perpetual swap / futures-linear-perp
    "FUTURES",     # dated futures
    "OPTIONS",     # options
    "DEX_SWAP",    # on-chain AMM swap
    "DEX_LP",      # liquidity provision
)


class BaseAdapter(ABC):
    """Minimal exchange adapter contract."""

    name: str = "base"
    # Which trading forms this adapter supports. Override in subclasses.
    trading_forms: frozenset[str] = frozenset({"SPOT"})
    # Which order types this adapter supports.
    order_types: frozenset[str] = frozenset({"MARKET", "LIMIT"})
    # Category: "CEX" | "DEX"
    category: str = "CEX"

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def place_order(
        self, symbol: str, side: str, size: float, order_type: str = "MARKET"
    ) -> dict[str, Any]: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> float: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    def supports(self, asset: str) -> bool:
        """Override in subclasses if asset filtering is required."""
        return True

    def meta(self) -> dict[str, Any]:
        """Introspection: used by the cockpit to render per-adapter capability."""
        return {
            "name": self.name,
            "category": self.category,
            "trading_forms": sorted(self.trading_forms),
            "order_types": sorted(self.order_types),
            "connected": bool(self.is_connected()),
        }
