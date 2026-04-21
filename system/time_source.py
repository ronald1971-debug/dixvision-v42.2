"""
system/time_source.py
DIX VISION v42.2 — Strict Monotonic Time Authority

Thread-safe. UTC from anchor+monotonic delta. Sequence numbered.
No other module calls datetime.now() or time.time() directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock


@dataclass(frozen=True)
class TimeStamp:
    utc_time: datetime
    monotonic_ns: int
    sequence: int

_lock = RLock()
_anchor_mono: int = time.monotonic_ns()
_anchor_utc: datetime = datetime.now(timezone.utc)
_last_mono: int = _anchor_mono
_seq: int = 0

def now() -> TimeStamp:
    global _last_mono, _seq
    with _lock:
        cur = time.monotonic_ns()
        if cur <= _last_mono:
            cur = _last_mono + 1
        _last_mono = cur
        _seq += 1
        delta_ns = cur - _anchor_mono
        utc = _anchor_utc + timedelta(microseconds=delta_ns / 1_000)
        return TimeStamp(utc_time=utc, monotonic_ns=cur, sequence=_seq)

def now_with_seq() -> tuple[datetime, int]:
    ts = now()
    return ts.utc_time, ts.sequence

def utc_now() -> datetime:
    return now().utc_time
