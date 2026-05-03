"""DIX VISION v42.2 — strict monotonic Time Authority (T0-4).

Restored after the rust-deletion in PR #116 stripped the polyglot
dual-backend implementation. P0-1a of the PHASE6 action plan: every
runtime caller must read time through this module so INV-15 (replay
determinism) and INV-58 (clock-drift hazard) have a single
chokepoint to instrument.

# Two API layers

The module exposes two intentionally distinct surfaces:

1. **Sequenced ledger API** (:func:`now`, :func:`now_with_seq`,
   :func:`utc_now`) — returns a :class:`TimeStamp` with a strictly
   monotonic ``monotonic_ns`` and a gap-free ``sequence``. Used by
   the ledger / hazard / cockpit. Acquires a process-wide
   ``RLock``; not allowed on the hot path.
2. **Hot-path API** (:func:`now_ns`, :func:`monotonic_ns`,
   :func:`wall_ns`) — pure wrappers around ``time.monotonic_ns`` /
   ``time.time_ns`` with no allocation, no lock, no sequence
   counter. Hot-path engines use these for latency measurement and
   bus stamping.

Both layers are byte-stable across imports — they do not branch on
configuration or environment.

# Authority lint

``tools/authority_lint.py`` rule **B-CLOCK** bans direct calls to
``datetime.now``, ``datetime.utcnow``, ``time.time``,
``time.time_ns``, ``time.monotonic_ns`` and ``time.perf_counter_ns``
in every runtime module **except** this one. Tests, the cockpit's
TOTP code, and a small documented allowlist may still read the
system clock directly; everything else routes through the functions
below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock

__all__ = (
    "TimeStamp",
    "now",
    "now_with_seq",
    "utc_now",
    "now_ns",
    "monotonic_ns",
    "wall_ns",
)


@dataclass(frozen=True)
class TimeStamp:
    """Sequenced monotonic timestamp.

    ``utc_time`` is derived from a single ``datetime.now()`` read at
    process start plus the monotonic delta, so wall-clock NTP jumps
    do not corrupt ledger ordering.
    """

    utc_time: datetime
    monotonic_ns: int
    sequence: int


_lock = RLock()
_anchor_mono: int = time.monotonic_ns()
_anchor_utc: datetime = datetime.now(UTC)
_last_mono: int = _anchor_mono
_seq: int = 0


def now() -> TimeStamp:
    """Return a sequenced :class:`TimeStamp`. Thread-safe.

    ``monotonic_ns`` is strictly increasing across threads;
    ``sequence`` increments by exactly 1 per call.
    """
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
    """Anchor-derived UTC ``datetime``. Use for ledger / hazard / news
    timestamps where wall-clock readability matters."""
    return now().utc_time


# --------------------------------------------------------------------------
# Hot-path API. Pure wrappers; no lock, no sequence counter, no allocation.
# --------------------------------------------------------------------------


def now_ns() -> int:
    """Monotonic nanoseconds for hot-path latency measurement."""
    return time.monotonic_ns()


def monotonic_ns() -> int:
    """Alias for :func:`now_ns`; spelled out for T0-4 contract readability."""
    return time.monotonic_ns()


def wall_ns() -> int:
    """Wall-clock nanoseconds since Unix epoch.

    Use for bus stamps and ledger timestamps where epoch alignment
    is needed. Never use for latency measurement (NTP can jump).
    """
    return time.time_ns()
