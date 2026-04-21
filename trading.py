"""
trading.py
DIX VISION v42.2 — High-level trading entry (INDIRA domain)
"""
from __future__ import annotations

from enforcement.decorators import enforce_full


@enforce_full
def place_trade(symbol: str, trade_size_pct: float = 0.5) -> str:
    return f"Trade executed for {symbol} with size {trade_size_pct}%"

@enforce_full
def execute_trade(symbol: str, trade_size_pct: float = 0.5) -> str:
    return f"Trade executed for {symbol} with size {trade_size_pct}%"
