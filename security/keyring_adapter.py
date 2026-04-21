"""
security/keyring_adapter.py
Optional bridge to the OS keyring. Uses the ``keyring`` library when present;
otherwise falls back to the in-memory SecretsManager.
"""
from __future__ import annotations

import threading

try:                                # pragma: no cover - optional dependency
    import keyring  # type: ignore
    _HAS_KEYRING = True
except Exception:
    keyring = None                  # type: ignore
    _HAS_KEYRING = False

from .secrets_manager import get_secrets_manager

SERVICE = "dix_vision_v42_2"


class KeyringAdapter:
    def set(self, key: str, value: str) -> None:
        if _HAS_KEYRING:
            try:
                keyring.set_password(SERVICE, key, value)
                return
            except Exception:
                pass
        get_secrets_manager().set(key, value)

    def get(self, key: str, default: str = "") -> str:
        if _HAS_KEYRING:
            try:
                v = keyring.get_password(SERVICE, key)
                if v is not None:
                    return str(v)
            except Exception:
                pass
        return get_secrets_manager().get(key, default)

    def delete(self, key: str) -> None:
        if _HAS_KEYRING:
            try:
                keyring.delete_password(SERVICE, key)
            except Exception:
                pass
        get_secrets_manager().delete(key)


_ka: KeyringAdapter | None = None
_lock = threading.Lock()


def get_keyring_adapter() -> KeyringAdapter:
    global _ka
    if _ka is None:
        with _lock:
            if _ka is None:
                _ka = KeyringAdapter()
    return _ka
