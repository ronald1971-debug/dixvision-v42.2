"""
system/audit_logger.py
DIX VISION v42.2 — Append-Only Durable Audit Logger (fsync on every write)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from system.config import get as get_config
from system.time_source import now_with_seq


class AuditLogger:
    def __init__(self, file_path: str | None = None) -> None:
        path = file_path or get_config("data.audit_log", "data/audit.jsonl")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event_type: str, source: str, payload: dict[str, Any]) -> None:
        ts, seq = now_with_seq()
        record = json.dumps({"timestamp": ts.isoformat(), "sequence": seq,
                              "event_type": event_type, "source": source,
                              "payload": payload}, separators=(",", ":"), default=str)
        with self._lock:
            self._file.write(record + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        with self._lock:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            finally:
                self._file.close()

_audit: AuditLogger | None = None
_lock = threading.Lock()

def get_audit_logger() -> AuditLogger:
    global _audit
    if _audit is None:
        with _lock:
            if _audit is None:
                _audit = AuditLogger()
    return _audit
