"""
translation/translator.py
DIX VISION v42.2 — Deterministic Intent Translator

Converts raw payloads to typed intent schemas.
No if/elif string chains. Uses enum mapping tables.
"""
from __future__ import annotations

from typing import Any

from translation.intent_models import (
    MarketIntent,
    MarketIntentType,
    SystemIntent,
    SystemIntentType,
)


class Translator:
    def translate_market(self, payload: dict[str, Any]) -> MarketIntent:
        intent_str = str(payload.get("action", "HOLD")).upper()
        try:
            intent_type = MarketIntentType(intent_str)
        except ValueError:
            intent_type = MarketIntentType.HOLD
        return MarketIntent(
            intent_type=intent_type,
            asset=str(payload.get("asset", "")),
            side=str(payload.get("side", "NONE")),
            size_usd=float(payload.get("size_usd", 0.0)),
            price=payload.get("price"),
            strategy=str(payload.get("strategy", "")),
        )

    def translate_system(self, payload: dict[str, Any]) -> SystemIntent:
        action = str(payload.get("action", "HEALTH_CHECK")).upper()
        try:
            intent_type = SystemIntentType(action)
        except ValueError:
            intent_type = SystemIntentType.HEALTH_CHECK
        return SystemIntent(intent_type=intent_type,
                           target=str(payload.get("target", "")),
                           payload=payload.get("payload", {}))

_translator: Translator | None = None

def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator
