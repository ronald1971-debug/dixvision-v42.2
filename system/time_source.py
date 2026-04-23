"""
system/time_source.py
DIX VISION v42.2 — Strict Monotonic Time Authority (T0-4).

Thread-safe. UTC derived from anchor + monotonic delta. Sequence numbered.
No other module calls datetime.now() or time.time_ns() directly.

# Backends

This module has two interchangeable backends and picks one at import time:

* **Rust (`dixvision_py_system`)** — PyO3 extension built from
  ``rust/py_system/``. Preferred when available. Hot-path call is a
  single FFI trampoline over a mutex-guarded integer update; the
  Python side only allocates the ``TimeStamp`` dataclass wrapper.
* **Pure Python (fallback)** — the original implementation kept
  verbatim so replay of pre-polyglot ledgers still works on a box
  that cannot build the Rust wheel (no cargo, restricted glibc, etc).

Both backends implement the same guarantees:

* ``now()`` is strictly monotonic across threads; ``monotonic_ns``
  never repeats nor rewinds.
* ``sequence`` is gap-free and strictly increasing by exactly 1 per
  call. The Python backend starts at 1; the Rust backend is a
  process-wide singleton whose counter survives reloads.
* ``utc_time`` == ``anchor_utc + (monotonic_ns - anchor_mono_ns)``,
  i.e. UTC drift is bounded by the anchor error; there is no second
  wall-clock read after process start.

Selection is observable via ``backend()``. Tests in
``tests/test_time_source_parity.py`` assert both backends satisfy
the invariants by calling ``make_python_backend()`` and
``make_rust_backend()`` directly, so neither test has to reload this
module or poke an environment variable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class TimeStamp:
    utc_time: datetime
    monotonic_ns: int
    sequence: int


# ---------------------------------------------------------------------------
# Pure-Python backend (also the reference implementation the parity
# suite validates the Rust port against).
# ---------------------------------------------------------------------------

def make_python_backend() -> Callable[[], TimeStamp]:
    """Construct a fresh Python-backed ``now()`` callable with its
    own anchor + sequence counter. Callers: the module-level default
    when the Rust wheel is unavailable, and the parity test suite."""
    lock = RLock()
    anchor_mono = time.monotonic_ns()
    anchor_utc = datetime.now(timezone.utc)
    state = {"last_mono": anchor_mono, "seq": 0}

    def _now() -> TimeStamp:
        with lock:
            cur = time.monotonic_ns()
            if cur <= state["last_mono"]:
                cur = state["last_mono"] + 1
            state["last_mono"] = cur
            state["seq"] += 1
            delta_ns = cur - anchor_mono
            utc = anchor_utc + timedelta(microseconds=delta_ns / 1_000)
            return TimeStamp(utc_time=utc, monotonic_ns=cur, sequence=state["seq"])

    return _now


# ---------------------------------------------------------------------------
# Rust backend wrapper. Imported lazily so a missing wheel falls through
# cleanly to the Python backend without an ImportError at module load.
# ---------------------------------------------------------------------------

def make_rust_backend() -> Optional[Callable[[], TimeStamp]]:
    """Construct a fresh Rust-backed ``now()`` callable if the
    ``dixvision_py_system`` wheel is importable; otherwise return
    ``None``. The underlying Rust ``TimeSource`` is a process-wide
    singleton, so repeated calls to this factory share the same
    sequence counter — the Python anchor captured here is private
    per-factory-call for UTC derivation only."""
    try:
        import dixvision_py_system as _rust  # type: ignore[import-not-found]
    except ImportError:
        return None

    # Independent anchor for UTC derivation. The Rust side reports
    # its own utc_nanos (derived from its own anchor); we ignore it
    # to keep exactly one ``datetime.now()`` read per backend
    # construction on the Python side.
    rust_anchor_utc = datetime.now(timezone.utc)
    rust_anchor_mono = int(_rust.now_mono_ns())

    def _now() -> TimeStamp:
        _, mono_ns, seq = _rust.now()
        delta_ns = int(mono_ns) - rust_anchor_mono
        utc = rust_anchor_utc + timedelta(microseconds=delta_ns / 1_000)
        return TimeStamp(utc_time=utc, monotonic_ns=int(mono_ns), sequence=int(seq))

    return _now


# ---------------------------------------------------------------------------
# Module-level default. Picks the Rust backend if the wheel was built
# and installed; otherwise the pure-Python reference. Both are always
# available via the ``make_*_backend`` factories above.
# ---------------------------------------------------------------------------

_rust_impl = make_rust_backend()
_BACKEND_NAME = "rust" if _rust_impl is not None else "python"
_now_impl: Callable[[], TimeStamp] = _rust_impl if _rust_impl is not None else make_python_backend()


def now() -> TimeStamp:
    return _now_impl()


def now_with_seq() -> Tuple[datetime, int]:
    ts = now()
    return ts.utc_time, ts.sequence


def utc_now() -> datetime:
    return now().utc_time


def backend() -> str:
    """Which backend is active in this process. Test-hook only —
    runtime code MUST NOT branch on the return value. Returns either
    ``"rust"`` or ``"python"``."""
    return _BACKEND_NAME
