"""
security/secrets_manager.py
In-memory secret store. Production deployments should back this by the
Windows Credential Manager (via keyring_adapter) or a cloud KMS.

Security invariants:
  • never write plaintext to ledger / logs
  • never leave the process address space (credentials_never_leave_machine)
"""
from __future__ import annotations

import threading


class SecretsManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._store[key] = value

    def get(self, key: str, default: str = "") -> str:
        with self._lock:
            return self._store.get(key, default)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())


_sm: SecretsManager | None = None
_lock = threading.Lock()


def get_secrets_manager() -> SecretsManager:
    global _sm
    if _sm is None:
        with _lock:
            if _sm is None:
                _sm = SecretsManager()
    return _sm
