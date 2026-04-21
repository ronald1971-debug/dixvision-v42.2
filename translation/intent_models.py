"""
translation/intent_models.py
DIX VISION v42.2 — Typed Intent Schemas

All intents are typed enums. No free-text allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MarketIntentType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"

class SystemIntentType(str, Enum):
    RESTART_SERVICE = "RESTART_SERVICE"
    APPLY_PATCH = "APPLY_PATCH"
    ROLLBACK_PATCH = "ROLLBACK_PATCH"
    BACKUP = "BACKUP"
    HEALTH_CHECK = "HEALTH_CHECK"

class HazardIntentType(str, Enum):
    EXCHANGE_TIMEOUT = "EXCHANGE_TIMEOUT"
    FEED_SILENCE = "FEED_SILENCE"
    LATENCY_SPIKE = "LATENCY_SPIKE"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    SYSTEM_DEGRADATION = "SYSTEM_DEGRADATION"

@dataclass(frozen=True)
class MarketIntent:
    intent_type: MarketIntentType
    asset: str
    side: str
    size_usd: float = 0.0
    price: float | None = None
    strategy: str = ""

@dataclass(frozen=True)
class SystemIntent:
    intent_type: SystemIntentType
    target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class HazardIntent:
    intent_type: HazardIntentType
    severity: str = "MEDIUM"
    details: dict[str, Any] = field(default_factory=dict)
