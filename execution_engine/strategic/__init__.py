"""Strategic execution layer (Wave 5 / Phase 10.6).

Pure / deterministic schedulers that turn a parent execution intent
into a sequence of child slices. Owned by the executor (per the
authority matrix); the intelligence engine emits the parent intent and
the executor decides *how* to work it.

Phase 1: Almgren-Chriss closed-form. Future revisions may add VWAP,
POV, and Implementation-Shortfall variants behind the same
``ExecutionSchedule`` shape.
"""

from __future__ import annotations

from execution_engine.strategic.almgren_chriss import (
    ExecutionSchedule,
    ExecutionSlice,
    solve_almgren_chriss,
)

__all__ = [
    "ExecutionSchedule",
    "ExecutionSlice",
    "solve_almgren_chriss",
]
