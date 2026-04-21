"""
core/bootstrap/lifecycle.py
Named lifecycle phases used across boot + shutdown sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LifecyclePhase(str, Enum):
    INIT = "INIT"
    BOOTING = "BOOTING"
    VERIFIED = "VERIFIED"
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    SAFE = "SAFE"
    HALTED = "HALTED"
    SHUTTING_DOWN = "SHUTTING_DOWN"


@dataclass(frozen=True)
class Lifecycle:
    phase: LifecyclePhase
    reason: str = ""

    def with_reason(self, reason: str) -> Lifecycle:
        return Lifecycle(self.phase, reason)
