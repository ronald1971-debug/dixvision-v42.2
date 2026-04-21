"""
system_monitor/heartbeat_monitor.py
WebSocket / REST-API liveness tracker. Each endpoint has a last-seen monotonic
timestamp; silence beyond threshold → emits HazardType.API_CONNECTIVITY_FAILURE.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from execution.hazard.event_emitter import get_hazard_emitter
from system.config import get as get_config


@dataclass
class HeartbeatState:
    last_seen_mono: float = 0.0
    misses: int = 0


class HeartbeatMonitor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._endpoints: dict[str, HeartbeatState] = {}
        self._threshold_s = float(get_config("hazard.heartbeat_timeout_seconds", 10.0))
        self._emitter = get_hazard_emitter("dyon.heartbeat")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self, endpoint: str) -> None:
        with self._lock:
            st = self._endpoints.setdefault(endpoint, HeartbeatState())
            st.last_seen_mono = time.monotonic()
            st.misses = 0

    def register(self, endpoint: str) -> None:
        with self._lock:
            self._endpoints.setdefault(endpoint, HeartbeatState(last_seen_mono=time.monotonic()))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DIX-HeartbeatMonitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Lazy import to avoid a circular dependency at module load:
        # dead_man imports fast_risk_cache; heartbeat_monitor is one of
        # the first things the cockpit constructs.
        from system_monitor.dead_man import get_dead_man
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                items = list(self._endpoints.items())
            for name, st in items:
                silence = now - st.last_seen_mono
                if silence > self._threshold_s:
                    self._emitter.api_failure(name, f"no_heartbeat_{silence:.1f}s")
                    with self._lock:
                        st.misses += 1
            # Drive the dead-man switch from a single background thread.
            # status() is a pure read; check() is the only path that
            # may trip the switch and halt trading (see DeadManSwitch).
            try:
                get_dead_man().check()
            except Exception:
                pass
            time.sleep(1.0)


_hb: HeartbeatMonitor | None = None
_lock = threading.Lock()


def get_heartbeat_monitor() -> HeartbeatMonitor:
    global _hb
    if _hb is None:
        with _lock:
            if _hb is None:
                _hb = HeartbeatMonitor()
    return _hb
