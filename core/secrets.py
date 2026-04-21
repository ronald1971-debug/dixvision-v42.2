"""
core/secrets.py
DIX VISION v42.2 — Secure Credential Access (manifest §16).

This module is now a thin shim over the canonical
``security.secrets_manager`` so there is ONE authoritative secrets path.

Priority order for reads:
  1. env var  (for CI / Docker / local dev)
  2. OS keyring via ``security.keyring_adapter`` (DPAPI / Secret Service / Keychain)
  3. In-memory ``SecretsManager`` (populated at boot from keyring)
  4. Caller-provided default

Exchange keys NEVER leave the machine: we never write plaintext to ledger,
logs, or config files.
"""
from __future__ import annotations

import os

from security.secrets_manager import SecretsManager

_manager: SecretsManager | None = None


def _mgr() -> SecretsManager:
    global _manager
    if _manager is None:
        _manager = SecretsManager()
    return _manager


def _env_key(name: str) -> str:
    return name.upper().replace(".", "_").replace("-", "_")


def get_secret(name: str, default: str | None = None) -> str:
    env = os.environ.get(_env_key(name))
    if env:
        return env
    try:
        from security.keyring_adapter import get_from_keyring  # type: ignore[attr-defined]
        v = get_from_keyring(name)
        if v:
            return v
    except Exception:
        pass
    v = _mgr().get(name, "")
    if v:
        return v
    if default is not None:
        return default
    raise KeyError(f"Secret '{name}' not found in env/keyring/manager")


def store_secret(name: str, value: str) -> None:
    """Store a secret in the in-memory manager; persistence is via keyring_adapter."""
    _mgr().set(name, value)
    try:
        from security.keyring_adapter import set_in_keyring  # type: ignore[attr-defined]
        set_in_keyring(name, value)
    except Exception:
        pass


def delete_secret(name: str) -> None:
    _mgr().delete(name)
    try:
        from security.keyring_adapter import delete_from_keyring  # type: ignore[attr-defined]
        delete_from_keyring(name)
    except Exception:
        pass


__all__ = ["get_secret", "store_secret", "delete_secret"]
