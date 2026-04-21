"""
observability/alerts/alert_engine.py
Rule-driven alert evaluation. Hooks into ledger events via the StreamRouter.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Alert:
    name: str
    severity: str
    message: str
    ts_ns: int


@dataclass
class AlertRule:
    name: str
    predicate: Callable[[dict[str, Any]], bool]
    severity: str = "INFO"
    message: str = ""


class AlertEngine:
    def __init__(self, maxlen: int = 1_000) -> None:
        self._lock = threading.RLock()
        self._rules: list[AlertRule] = []
        self._fired: list[Alert] = []
        self._maxlen = maxlen
        self._subscribers: list[Callable[[Alert], None]] = []

    def register(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules.append(rule)

    def subscribe(self, fn: Callable[[Alert], None]) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def evaluate(self, event: dict[str, Any]) -> list[Alert]:
        fired: list[Alert] = []
        with self._lock:
            rules = list(self._rules)
            subs = list(self._subscribers)
        for r in rules:
            try:
                if r.predicate(event):
                    a = Alert(name=r.name, severity=r.severity,
                              message=r.message or r.name, ts_ns=time.monotonic_ns())
                    fired.append(a)
            except Exception:
                continue
        if fired:
            with self._lock:
                self._fired.extend(fired)
                if len(self._fired) > self._maxlen:
                    self._fired = self._fired[-self._maxlen :]
            for a in fired:
                for s in subs:
                    try:
                        s(a)
                    except Exception:
                        continue
        return fired

    def recent(self, n: int = 50) -> list[Alert]:
        with self._lock:
            return list(self._fired[-n:])


_ae: AlertEngine | None = None
_lock = threading.Lock()


def get_alert_engine() -> AlertEngine:
    global _ae
    if _ae is None:
        with _lock:
            if _ae is None:
                _ae = AlertEngine()
    return _ae
