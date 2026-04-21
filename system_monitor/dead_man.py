"""system_monitor.dead_man \u2014 cockpit-heartbeat-based safety halt.

If the cockpit process has not beaten within ``timeout_sec``, the dead-man
trips: emits ``SYSTEM/DEAD_MAN_TRIPPED`` and halts all live trading via
the fast-risk cache.  Then all INDIRA execution paths refuse until a
fresh heartbeat re-arms the switch.

We deliberately HALT TRADING rather than terminating the process
(``immutable_core.kill_switch.trigger_kill_switch`` calls ``os._exit``)
so the operator can rearm from the cockpit without a process restart.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from state.ledger.writer import get_writer
from system.fast_risk_cache import get_risk_cache
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
                # Re-enable trading via the risk cache.  If any other
                # halt reason is active (safe mode, circuit breaker),
                # those remain in force and block trades independently.
                get_risk_cache().resume_trading()

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
                # Halt trading via the fast-risk cache.  This is the
                # canonical "stop signing" path used by every other
                # governance halt (safe mode, emergency mode).  We do
                # NOT call immutable_core.kill_switch.trigger_kill_switch
                # here because that would os._exit() the process and
                # prevent the cockpit from ever re-arming.
                get_risk_cache().halt_trading(reason="dead_man")
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
