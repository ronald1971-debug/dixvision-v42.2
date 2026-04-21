"""
security/authorization.py
Role-based authorization. Axioms (operator overrides, etc.) are hard-coded.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    OPERATOR = "OPERATOR"
    ADMIN = "ADMIN"
    AUDITOR = "AUDITOR"
    SERVICE = "SERVICE"


_ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.OPERATOR: {"trade.place", "mode.safe_mode", "mode.resume"},
    Role.ADMIN: {
        "trade.place", "mode.safe_mode", "mode.resume", "mode.halt",
        "policy.edit", "adapter.manage", "config.edit",
    },
    Role.AUDITOR: {"ledger.read", "snapshot.read", "metrics.read"},
    Role.SERVICE: {"metrics.read", "ledger.read"},
}


@dataclass
class Principal:
    name: str
    roles: set[Role] = field(default_factory=set)


class Authorizer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._principals: dict[str, Principal] = {}

    def grant(self, name: str, role: Role) -> None:
        with self._lock:
            p = self._principals.setdefault(name, Principal(name=name))
            p.roles.add(role)

    def revoke(self, name: str, role: Role) -> None:
        with self._lock:
            p = self._principals.get(name)
            if p and role in p.roles:
                p.roles.discard(role)

    def authorize(self, name: str, permission: str) -> bool:
        with self._lock:
            p = self._principals.get(name)
        if p is None:
            return False
        for r in p.roles:
            if permission in _ROLE_PERMISSIONS.get(r, set()):
                return True
        return False


_az: Authorizer | None = None
_lock = threading.Lock()


def get_authorizer() -> Authorizer:
    global _az
    if _az is None:
        with _lock:
            if _az is None:
                _az = Authorizer()
    return _az
