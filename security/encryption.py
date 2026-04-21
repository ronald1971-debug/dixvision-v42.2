"""
security/encryption.py
Stdlib-only authenticated encryption using AES-GCM via ``cryptography`` if
available; otherwise uses HMAC+XOR KDF-based wrapping as a last-resort
obfuscation (NOT production-grade).

Prefer installing the ``cryptography`` package on Windows/Linux.
"""
from __future__ import annotations

import hashlib
import hmac
import os

try:  # pragma: no cover - optional dep
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    _HAS_CRYPTO = True
except Exception:
    AESGCM = None  # type: ignore
    _HAS_CRYPTO = False


def derive_key(passphrase: bytes, salt: bytes, iterations: int = 100_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase, salt, iterations, dklen=32)


def encrypt_bytes(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    if _HAS_CRYPTO:
        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, aad or None)
        return b"AESGCM" + nonce + ct
    nonce = os.urandom(16)
    stream = _hmac_stream(key, nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, stream, strict=False))
    tag = hmac.new(key, nonce + ct + aad, hashlib.sha256).digest()
    return b"HMACXOR" + nonce + tag + ct


def decrypt_bytes(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    if blob.startswith(b"AESGCM"):
        if not _HAS_CRYPTO:
            raise RuntimeError("AES-GCM ciphertext but cryptography lib missing")
        nonce = blob[6:18]
        ct = blob[18:]
        return AESGCM(key).decrypt(nonce, ct, aad or None)
    if blob.startswith(b"HMACXOR"):
        nonce = blob[7:23]
        tag = blob[23:55]
        ct = blob[55:]
        expected = hmac.new(key, nonce + ct + aad, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, tag):
            raise ValueError("bad_mac")
        stream = _hmac_stream(key, nonce, len(ct))
        return bytes(a ^ b for a, b in zip(ct, stream, strict=False))
    raise ValueError("unknown_ciphertext")


def _hmac_stream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])
