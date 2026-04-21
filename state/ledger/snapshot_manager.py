"""
state/ledger/snapshot_manager.py
DIX VISION v42.2 — Snapshot Manager (Checkpoint System)

Full snapshots every 10k events. Incremental on patch/mode-change.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock

from system.config import get
from system.state import get_state_manager

_lock = Lock()

def _base() -> Path:
    return Path(get("data.snapshots", "data/snapshots"))

def save_snapshot(event_type: str = "manual", full: bool = False) -> Path:
    base = _base() / ("full" if full else "incremental")
    base.mkdir(parents=True, exist_ok=True)
    state = get_state_manager().get()
    payload = {k: getattr(state, k) for k in state.__dataclass_fields__}
    ts_ns = time.monotonic_ns()
    raw = json.dumps({"ts_ns": ts_ns, "event": event_type, "state": payload}, default=str)
    checksum = hashlib.sha256(raw.encode()).hexdigest()
    record = json.dumps({"checksum": checksum, "data": {"ts_ns": ts_ns, "event": event_type, "state": payload}}, default=str)
    path = base / f"{event_type}_{ts_ns}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(record)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return path

def restore_latest() -> dict | None:
    full_dir = _base() / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    snaps = sorted(full_dir.glob("*.json"), reverse=True)
    for p in snaps:
        try:
            data = json.loads(p.read_text())
            raw_data = json.dumps(data["data"], default=str)
            if hashlib.sha256(raw_data.encode()).hexdigest() == data["checksum"]:
                get_state_manager().restore(data["data"]["state"])
                return data["data"]["state"]
        except Exception:
            continue
    return None

def save_incremental(event_type: str = "tick") -> Path:
    return save_snapshot(event_type, full=False)
