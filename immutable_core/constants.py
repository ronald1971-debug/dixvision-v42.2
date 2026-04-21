"""
immutable_core/constants.py
DIX VISION v42.2 — System Invariants (LEAN4 verified floors)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyAxioms:
    MAX_DRAWDOWN_FLOOR_PCT: float = 4.0
    MAX_LOSS_PER_TRADE_FLOOR_PCT: float = 1.0
    FAIL_CLOSED: bool = True
    CREDENTIALS_LOCAL_ONLY: bool = True
    FAST_PATH_MAX_LATENCY_MS: float = 5.0

AXIOMS = SafetyAxioms()
