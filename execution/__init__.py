"""execution — Trade execution + emergency execution domain.

Canonical split:
  Indira (market) → ``trade_executor`` → adapters
  Hazard      → ``emergency_executor`` → mode transitions / kill switch

Dyon system maintenance lives under the same package but CANNOT touch
adapters or the trade_executor.
"""
from .engine import DyonEngine, get_dyon_engine

__all__ = ["DyonEngine", "get_dyon_engine"]
