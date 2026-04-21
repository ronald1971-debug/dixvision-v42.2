"""
interrupt/dispatcher.py
Subscribes to SYSTEM_HAZARD_EVENT on the async hazard bus and routes each
hazard to Resolver → InterruptExecutor in deterministic time (< 10ms target).
"""
from __future__ import annotations

import threading
import time
from typing import Any

from .interrupt_executor import get_interrupt_executor
from .resolver import get_resolver


class Dispatcher:
    def __init__(self) -> None:
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        try:
            from execution.hazard.async_bus import get_hazard_bus

            bus = get_hazard_bus()
            # Bus may expose either subscribe(fn) (pub/sub) or has_subscribers()
            sub = getattr(bus, "subscribe", None)
            if callable(sub):
                sub(self.handle)
            else:
                # Polling fallback
                threading.Thread(
                    target=self._poll, args=(bus,), daemon=True,
                    name="DIX-Interrupt-Dispatcher",
                ).start()
        except Exception as e:  # pragma: no cover - wiring-only
            try:
                from system.logger import get_logger

                get_logger("interrupt.dispatcher").error(
                    "hazard_bus_wire_failed", error=str(e),
                )
            except Exception:
                pass

    def _poll(self, bus: Any) -> None:
        poll = getattr(bus, "drain", None)
        if not callable(poll):
            return
        while True:
            events = poll()
            if events:
                for e in events:
                    try:
                        self.handle(e)
                    except Exception:
                        pass
            time.sleep(0.005)

    def handle(self, hazard: Any) -> None:
        action = get_resolver().resolve(hazard)
        get_interrupt_executor().execute(action)


_d: Dispatcher | None = None
_lock = threading.Lock()


def get_dispatcher() -> Dispatcher:
    global _d
    if _d is None:
        with _lock:
            if _d is None:
                _d = Dispatcher()
    return _d
