"""
immutable_core/kill_switch.py
DIX VISION v42.2 — Hard System Termination
stdlib ONLY — no higher-layer imports.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

_killed: bool = False
_lock = threading.Lock()

def _safe_log(reason: str, source: str) -> None:
    """Best-effort incident log. Failures are reported on stderr, never swallowed silently."""
    try:
        p = Path(os.environ.get("DIX_INCIDENTS_PATH", "data/incidents.jsonl"))
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"ts": time.time(), "reason": reason,
                                "source": source, "event": "KILL_SWITCH"}) + "\n")
    except Exception as e:
        sys.stderr.write(f"[KILL SWITCH] incident log failed: {e}\n")

def trigger_kill_switch(reason: str = "unknown", source: str = "system") -> None:
    global _killed
    with _lock:
        if _killed:
            return
        _killed = True
    _safe_log(reason, source)
    sys.stderr.write(f"[KILL SWITCH] ACTIVATED | reason={reason} | source={source}\n")
    sys.stderr.flush()
    os._exit(1)

trigger = trigger_kill_switch  # alias

def is_triggered() -> bool:
    return _killed
