"""security — Secrets, keyring, encryption, auth, and audit trail."""
from .audit_trail import audit
from .authentication import Authenticator, get_authenticator
from .authorization import Authorizer, Role, get_authorizer
from .encryption import decrypt_bytes, derive_key, encrypt_bytes
from .keyring_adapter import KeyringAdapter, get_keyring_adapter
from .secrets_manager import SecretsManager, get_secrets_manager

__all__ = [
    "SecretsManager",
    "get_secrets_manager",
    "KeyringAdapter",
    "get_keyring_adapter",
    "encrypt_bytes",
    "decrypt_bytes",
    "derive_key",
    "Authenticator",
    "get_authenticator",
    "Authorizer",
    "get_authorizer",
    "Role",
    "audit",
]
