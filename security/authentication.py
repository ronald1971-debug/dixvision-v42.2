"""
security/authentication.py
Minimal in-process session authenticator. Uses HMAC-signed tokens with a
per-process rotating secret.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
import uuid
from dataclasses import dataclass


@dataclass
class Session:
    principal: str
    issued_at: float
    expires_at: float
    token: str


class Authenticator:
    def __init__(self, secret: bytes | None = None, ttl_s: float = 3600.0) -> None:
        self._secret = secret or os.urandom(32)
        self._ttl_s = ttl_s
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}

    def issue(self, principal: str) -> Session:
        sid = uuid.uuid4().hex
        issued = time.time()
        expires = issued + self._ttl_s
        token = hmac.new(
            self._secret,
            f"{principal}:{sid}:{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()
        sess = Session(principal=principal, issued_at=issued,
                       expires_at=expires, token=token)
        with self._lock:
            self._sessions[token] = sess
        return sess

    def verify(self, token: str) -> Session | None:
        with self._lock:
            sess = self._sessions.get(token)
        if sess is None:
            return None
        if time.time() > sess.expires_at:
            with self._lock:
                self._sessions.pop(token, None)
            return None
        return sess

    def revoke(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)


_a: Authenticator | None = None
_lock = threading.Lock()


def get_authenticator() -> Authenticator:
    global _a
    if _a is None:
        with _lock:
            if _a is None:
                _a = Authenticator()
    return _a
