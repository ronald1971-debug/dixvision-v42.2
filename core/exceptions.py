"""
core/exceptions.py
DIX VISION v42.2 — Structured Exception Hierarchy
"""
from __future__ import annotations

from typing import Any


class StructuredError(Exception):
    def __init__(self, message: str, code: str = "UNKNOWN",
                 metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}

class HardKillSwitchTriggered(StructuredError):
    def __init__(self, reason: str = "unknown") -> None:
        super().__init__(f"KILL: {reason}", "KILL_SWITCH", {"reason": reason})

class GovernanceViolation(StructuredError):
    def __init__(self, reason: str, action: str = "") -> None:
        super().__init__(f"Governance violation: {reason}", "GOV_VIOLATION",
                         {"reason": reason, "action": action})

class DomainViolation(StructuredError):
    """Raised when a component attempts to cross domain boundaries."""
    def __init__(self, component: str, target_domain: str) -> None:
        super().__init__(f"{component} cannot access {target_domain} domain",
                         "DOMAIN_VIOLATION",
                         {"component": component, "target": target_domain})
