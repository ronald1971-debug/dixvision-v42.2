"""
enforcement/runtime_guardian.py
DIX VISION v42.2 — Runtime Guardian

Monitors invariants. DOES NOT update heartbeat (only main loop does).
Triggers kill switch on critical breach.
"""
from __future__ import annotations

import threading
import time

from immutable_core.kill_switch import trigger_kill_switch
from system.config import get as get_config
from system.state import get_state_manager


class RuntimeGuardian:
    def __init__(self) -> None:
        self._state_mgr = get_state_manager()
        self._check_interval = float(get_config("guardian.check_interval_seconds", 2.0))
        self._heartbeat_timeout_ns = int(
            float(get_config("guardian.heartbeat_timeout_seconds", 10.0)) * 1e9
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="DIX-Guardian"
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._check()
            except Exception as e:
                try:
                    trigger_kill_switch(
                        f"guardian_unexpected_failure:{e}", "runtime_guardian"
                    )
                except Exception:
                    pass
            time.sleep(self._check_interval)

    def _check(self) -> None:
        state = self._state_mgr.get()
        if state.last_heartbeat_ns > 0:
            age_ns = time.monotonic_ns() - state.last_heartbeat_ns
            if age_ns > self._heartbeat_timeout_ns:
                self._state_mgr.set_mode("HALTED")
                trigger_kill_switch(
                    f"stale_heartbeat:{age_ns/1e9:.1f}s", "runtime_guardian"
                )
                return
        if state.health < 0.3:
            trigger_kill_switch(
                f"critical_health:{state.health:.2f}", "runtime_guardian"
            )

_guardian: RuntimeGuardian | None = None
_lock = threading.Lock()

def get_runtime_guardian() -> RuntimeGuardian:
    global _guardian
    if _guardian is None:
        with _lock:
            if _guardian is None:
                _guardian = RuntimeGuardian()
    return _guardian

def start_runtime_guardian() -> RuntimeGuardian:
    g = get_runtime_guardian()
    g.start()
    return g
