"""
mind/sources/source_types.py
Shared dataclasses for all mind.sources.* adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceKind(str, Enum):
    MARKET = "MARKET"
    NEWS = "NEWS"
    SENTIMENT = "SENTIMENT"
    ONCHAIN = "ONCHAIN"
    REST = "REST"
    WEBSOCKET = "WEBSOCKET"


@dataclass
class MarketTick:
    source: str
    asset: str
    price: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp_utc: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
