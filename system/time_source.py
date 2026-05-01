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


# ─────────────────────────────────────────────────────────────────────
# T0-4 canonical hot-path API (see docs/ARCHITECTURE_V42_2_TIER0.md §6).
#
# Hot-path callers MUST use these three functions and only these three
# functions. They never allocate, never acquire locks, and never touch
# the `RLock`-guarded sequence counter above. The sequence-stamped
# ``now()`` above is for ledger / hazard / cockpit use, not the hot
# path.
#
# Contract:
#   now_ns()       — monotonic, process-local, no epoch. Use for
#                    latency measurement and fast-path ordering.
#   monotonic_ns() — alias for now_ns() (guaranteed non-decreasing).
#   wall_ns()      — wall-clock nanoseconds since the Unix epoch.
#                    Use for ledger / hazard timestamps, never for
#                    latency measurement (NTP can jump).
#
# Implementation note: the Tier-0 directive refers to
# ``time.perf_counter_ns()``. The actual Python primitive that
# CPython documents as "cannot go backward" is ``time.monotonic_ns``;
# ``time.perf_counter_ns`` is only guaranteed to have the highest
# available resolution — not monotonicity. Because the T0-4 contract
# explicitly promises non-decreasing behaviour, we bind ``now_ns`` to
# ``time.monotonic_ns`` and treat the directive's word as shorthand
# for "monotonic nanosecond clock".
#
# authority_lint rule T1 (see tools/authority_lint.py) enforces that
# ``datetime.now()`` / ``time.time()`` are not called anywhere under
# mind/*, execution/*, governance/*, system/*, except in this module
# and a documented allowlist.
# ─────────────────────────────────────────────────────────────────────


def now_ns() -> int:
    """Monotonic nanoseconds. Hot-path safe. Never goes backwards."""
    return time.monotonic_ns()


def monotonic_ns() -> int:
    """Alias for now_ns(); spelled out for T0-4 contract readability."""
    return time.monotonic_ns()


def wall_ns() -> int:
    """Wall-clock nanoseconds since Unix epoch. NEVER use for latency."""
    return time.time_ns()
