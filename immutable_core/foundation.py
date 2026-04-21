"""
immutable_core/foundation.py
DIX VISION v42.2 — Foundation Integrity Verifier
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


class FoundationIntegrity:
    def __init__(self, root: Path, expected_hash: str = "") -> None:
        self.root = root
        self.expected_hash = expected_hash
        self._strict = os.environ.get("DIX_STRICT_INTEGRITY", "0") == "1"

    def compute_hash(self) -> str:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def verify(self) -> bool:
        if not self.expected_hash:
            if self._strict:
                raise RuntimeError("PROD integrity check failed: foundation.hash empty")
            return False
        actual = self.compute_hash()
        if actual != self.expected_hash:
            if self._strict:
                raise RuntimeError(f"Hash mismatch: expected={self.expected_hash} actual={actual}")
            return False
        return True
