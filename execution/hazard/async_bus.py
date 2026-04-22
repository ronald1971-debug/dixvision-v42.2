"""
execution/hazard/async_bus.py
DIX VISION v42.2 — Non-Blocking Hazard Event Bus

Dyon pushes SYSTEM_HAZARD_EVENTs here.
Governance pulls asynchronously. NEVER blocks the trading loop.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from system.time_source import now


class HazardSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class HazardType(str, Enum):
    EXCHANGE_TIMEOUT = "EXCHANGE_TIMEOUT"
    FEED_SILENCE = "FEED_SILENCE"
    EXECUTION_LATENCY_SPIKE = "EXECUTION_LATENCY_SPIKE"
    DATA_CORRUPTION_SUSPECTED = "DATA_CORRUPTION_SUSPECTED"
    SYSTEM_DEGRADATION = "SYSTEM_DEGRADATION"
    API_CONNECTIVITY_FAILURE = "API_CONNECTIVITY_FAILURE"
    LEDGER_INCONSISTENCY = "LEDGER_INCONSISTENCY"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    CPU_OVERLOAD = "CPU_OVERLOAD"

@dataclass
class HazardEvent:
    hazard_type: HazardType
    severity: HazardSeverity
    source: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = ""
    sequence: int = 0
    resolved: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            ts = now()
            self.timestamp_utc = ts.utc_time.isoformat()
            self.sequence = ts.sequence

class HazardBus:
    """
    Non-blocking async event bus.
    Dyon is the sole producer. Governance is the sole consumer.
    Queue never blocks the trading hot path.
    """
    def __init__(self, maxsize: int = 10_000) -> None:
        self._q: queue.Queue[HazardEvent] = queue.Queue(maxsize=maxsize)
        self._handlers: list[Callable[[HazardEvent], None]] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        # ``Thread`` objects cannot be restarted, so mint a fresh one
        # every time ``start()`` is called after ``stop()``.
        self._running = True
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._dispatch_loop,
                                        daemon=True, name="HazardBus")
        self._worker.start()

    def stop(self) -> None:
        self._running = False

    def emit(self, event: HazardEvent) -> bool:
        """
        Non-blocking emit. Returns False if queue is full (dropped).
        NEVER raises. NEVER blocks the trading loop.
        """
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            return False  # drop, continue trading

    def subscribe(self, handler: Callable[[HazardEvent], None]) -> None:
        with self._lock:
            self._handlers.append(handler)

    def _dispatch_loop(self) -> None:
        while self._running:
            try:
                event = self._q.get(timeout=0.1)
                with self._lock:
                    handlers = list(self._handlers)
                for h in handlers:
                    try:
                        h(event)
                    except Exception:
                        pass  # handler errors never crash the bus
            except queue.Empty:
                continue
            except Exception:
                continue

_bus: HazardBus | None = None
_bus_lock = threading.Lock()

def get_hazard_bus() -> HazardBus:
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = HazardBus()
                _bus.start()
    return _bus
