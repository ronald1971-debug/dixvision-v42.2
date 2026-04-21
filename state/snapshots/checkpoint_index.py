"""
state/snapshots/checkpoint_index.py
Filesystem index of full snapshots under ``data/snapshots/full/``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from system.config import get


@dataclass
class Checkpoint:
    path: Path
    ts_ns: int
    event: str


class CheckpointIndex:
    def __init__(self, base: Path | None = None) -> None:
        self._base = base or Path(get("data.snapshots", "data/snapshots")) / "full"
        self._base.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Checkpoint]:
        out: list[Checkpoint] = []
        for p in sorted(self._base.glob("*.json"), reverse=True):
            stem = p.stem  # <event>_<ts_ns>
            try:
                event, ts_ns = stem.rsplit("_", 1)
                out.append(Checkpoint(path=p, ts_ns=int(ts_ns), event=event))
            except Exception:
                continue
        return out

    def latest(self) -> Checkpoint | None:
        items = self.list()
        return items[0] if items else None


_idx: CheckpointIndex | None = None
_lock = threading.Lock()


def get_checkpoint_index() -> CheckpointIndex:
    global _idx
    if _idx is None:
        with _lock:
            if _idx is None:
                _idx = CheckpointIndex()
    return _idx
