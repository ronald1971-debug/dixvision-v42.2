"""
core/runtime/runtime_state.py
Lightweight process-wide runtime facts (pid, boot ts, build info).
Does NOT hold application state — that lives in ``system.state``.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


def _version() -> str:
    try:
        root = Path(os.environ.get("DIX_ROOT", "."))
        v = (root / "VERSION").read_text().strip()
        return v or "42.2.0"
    except Exception:
        return "42.2.0"


@dataclass
class RuntimeState:
    pid: int = field(default_factory=os.getpid)
    boot_ts_ns: int = field(default_factory=time.monotonic_ns)
    version: str = field(default_factory=_version)

    def uptime_ns(self) -> int:
        return time.monotonic_ns() - self.boot_ts_ns


_rs: RuntimeState | None = None
_lock = threading.Lock()


def get_runtime_state() -> RuntimeState:
    global _rs
    if _rs is None:
        with _lock:
            if _rs is None:
                _rs = RuntimeState()
    return _rs
