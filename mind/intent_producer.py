"""
mind/intent_producer.py
DIX VISION v42.2 — Signal to Execution Event Generator

Converts Indira's analysis into typed execution events.
Output is consumed by the fast path executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    TRADE_EXECUTE = "TRADE_EXECUTE"
    HOLD = "HOLD"
    DELEGATE = "DELEGATE"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"

@dataclass
class IndiraIntent:
    intent_type: IntentType
    confidence: float
    reasoning: str = ""

class IntentProducer:
    """
    Classifies market conditions into typed intents.
    No free-text — all outputs are typed enums.
    """
    def classify(self, signal_confidence: float,
                 data_quality: float = 1.0,
                 execution_quality: float = 1.0) -> IndiraIntent:
        overall = (signal_confidence * 0.3 + data_quality * 0.4 + execution_quality * 0.3)
        if overall < 0.55:
            return IndiraIntent(IntentType.DELEGATE, overall, "confidence_gap")
        if data_quality < 0.3:
            return IndiraIntent(IntentType.HOLD, overall, "data_quality_low")
        if execution_quality < 0.3:
            return IndiraIntent(IntentType.HOLD, overall, "execution_quality_low")
        return IndiraIntent(IntentType.TRADE_EXECUTE, overall, "signals_aligned")
