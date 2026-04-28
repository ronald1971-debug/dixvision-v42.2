"""RUNTIME-ENGINE-01 Intelligence (Phase E0 shell).

Owner of all market-intelligence plugins (microstructure, alpha, alt-data,
memory, multi-timeframe, transfer, cognition, agent). Strictly deterministic.

Lint:
- B1 forbids importing from ``execution_engine``, ``system_engine``,
  ``governance_engine``, ``learning_engine``, ``evolution_engine``.
- L3 forbids importing from ``learning_engine`` or ``evolution_engine``.
"""

from intelligence_engine.engine import (
    DEFAULT_SIGNAL_WINDOW_SIZE,
    IntelligenceEngine,
)
from intelligence_engine.runtime_context import RuntimeContext

__all__ = [
    "DEFAULT_SIGNAL_WINDOW_SIZE",
    "IntelligenceEngine",
    "RuntimeContext",
]
