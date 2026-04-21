"""
system/config.py
DIX VISION v42.2 — Centralized Configuration (dot-path access)
"""
from __future__ import annotations

from threading import RLock
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "guardian.check_interval_seconds": 2.0,
    "guardian.heartbeat_timeout_seconds": 10.0,
    "risk.max_drawdown_pct": 4.0,
    "risk.max_loss_per_trade_pct": 1.0,
    "risk.fast_path_max_latency_ms": 5.0,
    "hazard.feed_silence_threshold_seconds": 5.0,
    "hazard.latency_spike_threshold_ms": 100.0,
    "data.audit_log": "data/audit.jsonl",
    "data.incidents": "data/incidents.jsonl",
    "data.snapshots": "data/snapshots",
    "ledger.db_path": "data/sqlite/ledger.db",
}

class Config:
    def __init__(self) -> None:
        self._lock = RLock()
        self._data: dict[str, Any] = dict(_DEFAULTS)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, _DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(values)

_cfg: Config | None = None
_lock = RLock()

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        with _lock:
            if _cfg is None:
                _cfg = Config()
    return _cfg

def get(key: str, default: Any = None) -> Any:
    return get_config().get(key, default)
