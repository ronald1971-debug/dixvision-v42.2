"""system_monitor.dead_man \u2014 cockpit-heartbeat-based kill switch.

If the cockpit process has not beaten within `timeout_sec`, the dead-man
trips: emits SYSTEM/DEAD_MAN_TRIPPED and arms the global kill_switch.
Then all INDIRA execution paths refuse until manual rearm.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from state.ledger.writer import get_writer
from system.time_source import utc_now

DEFAULT_TIMEOUT_SEC = 120.0


@dataclass
class DeadManStatus:
    last_beat_utc: str
    age_sec: float
    timeout_sec: float
    tripped: bool

    def as_dict(self) -> dict:
        return {
            "last_beat_utc": self.last_beat_utc,
            "age_sec": round(self.age_sec, 3),
            "timeout_sec": self.timeout_sec,
            "tripped": self.tripped,
        }


class DeadManSwitch:
    def __init__(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> None:
        self._timeout = float(timeout_sec)
        self._last: datetime | None = None
        self._tripped = False
        self._lock = threading.RLock()

    def _now(self) -> datetime:
        n = utc_now()
        return n.replace(tzinfo=timezone.utc) if n.tzinfo is None else n

    def heartbeat(self, *, source: str = "cockpit") -> None:
        with self._lock:
            self._last = self._now()
            if self._tripped:
                self._tripped = False
                get_writer().write("SYSTEM", "DEAD_MAN_REARMED", "GOVERNANCE",
                                   {"source": source})

    def status(self) -> DeadManStatus:
        with self._lock:
            if self._last is None:
                return DeadManStatus("", 0.0, self._timeout, self._tripped)
            age = (self._now() - self._last).total_seconds()
            if not self._tripped and age > self._timeout:
                self._tripped = True
                get_writer().write("SYSTEM", "DEAD_MAN_TRIPPED", "GOVERNANCE",
                                   {"age_sec": round(age, 1),
                                    "timeout_sec": self._timeout})
                try:
                    from immutable_core.kill_switch import get_kill_switch
                    get_kill_switch().arm(reason="dead_man")
                except Exception:
                    pass
            return DeadManStatus(
                last_beat_utc=self._last.isoformat(),
                age_sec=age, timeout_sec=self._timeout,
                tripped=self._tripped,
            )

    def tripped(self) -> bool:
        return self.status().tripped


_singleton: DeadManSwitch | None = None
_lock = threading.Lock()


def get_dead_man() -> DeadManSwitch:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = DeadManSwitch()
    return _singleton


__all__ = ["DeadManSwitch", "DeadManStatus", "get_dead_man"]
