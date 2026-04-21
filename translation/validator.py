"""
translation/validator.py
DIX VISION v42.2 — Schema Validator

Enforces typed schema contracts.
"""
from __future__ import annotations

from translation.intent_models import MarketIntent, SystemIntent


def validate_market_intent(intent: MarketIntent) -> tuple[bool, str]:
    if not intent.asset:
        return False, "asset_empty"
    if intent.size_usd < 0:
        return False, "size_negative"
    if intent.size_usd > 1_000_000:
        return False, "size_too_large"
    return True, "valid"

def validate_system_intent(intent: SystemIntent) -> tuple[bool, str]:
    if not intent.intent_type:
        return False, "intent_type_empty"
    return True, "valid"
