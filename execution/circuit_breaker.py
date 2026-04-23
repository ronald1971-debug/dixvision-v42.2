"""
execution/circuit_breaker.py
DIX VISION v42.2 — Exchange-adapter circuit-breaker (T0-8)

Three-state circuit-breaker used by every exchange adapter under
``execution/adapters/``. Fails fast after a burst of exchange errors
and self-recovers without operator intervention.

Polyglot dual-backend
---------------------
When the Rust extension ``dixvision_py_system`` is importable **and**
exports the full ``circuit_breaker_*`` surface, this module proxies
to Rust (canonical, thread-safe via ``parking_lot::Mutex``). Otherwise
it falls back to the pure-Python reference below. Both backends
satisfy the identical invariants exercised by
``tests/test_circuit_breaker_parity.py``.

State machine
-------------
* **closed** — normal operation; every ``allow()`` returns True.
  Consecutive ``record_failure()`` calls increment a counter; when
  it reaches ``failure_threshold`` the breaker transitions to open.
* **open** — ``allow()`` returns False. After ``reset_timeout_ms``
  elapses the breaker enters ``half_open`` on the next ``allow()``.
* **half_open** — exactly ONE probe call is allowed; every subsequent
  ``allow()`` returns False until the probe resolves.
  ``record_success()`` transitions to ``closed``;
  ``record_failure()`` re-opens for another ``reset_timeout_ms``.

Public surface
--------------
The module-level ``CircuitBreaker`` class is the dual-backend
selector. Construction:

    cb = CircuitBreaker("binance-adapter",
                        failure_threshold=3,
                        reset_timeout_ms=30_000)
    if cb.allow():
        try:
            send_order(...)
            cb.record_success()
        except ExchangeError:
            cb.record_failure()

Thread-safety: all methods are serialised internally; safe to share
a single breaker across adapter worker threads.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

# ---------------------------------------------------------------- Rust backend
try:
    import dixvision_py_system as _rs  # type: ignore[import-not-found]

    _HAVE_RUST = all(
        hasattr(_rs, fn)
        for fn in (
            "circuit_breaker_register",
            "circuit_breaker_allow",
            "circuit_breaker_record_success",
            "circuit_breaker_record_failure",
            "circuit_breaker_reset",
            "circuit_breaker_state",
            "circuit_breaker_failure_count",
        )
    )
except ImportError:  # pragma: no cover - backend selection branch
    _rs = None
    _HAVE_RUST = False


# -------------------------------------------------------------- Python backend


class _PythonCircuitBreaker:
    """Pure-CPython reference breaker.

    Mirrors :class:`_RustCircuitBreaker` bit-for-bit. All state
    transitions are serialised through a single :class:`threading.Lock`.
    """

    __slots__ = (
        "_name",
        "_threshold",
        "_reset_s",
        "_lock",
        "_state",
        "_failures",
        "_open_until",
        "_probe_in_flight",
    )

    _STATES = ("closed", "open", "half_open")

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        reset_timeout_ms: int,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._name = name
        self._threshold = int(failure_threshold)
        self._reset_s = float(reset_timeout_ms) / 1000.0
        self._lock = threading.Lock()
        self._state: str = "closed"
        self._failures: int = 0
        self._open_until: Optional[float] = None
        self._probe_in_flight: bool = False

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if self._open_until is not None and now >= self._open_until:
                    self._state = "half_open"
                    self._open_until = None
                    self._probe_in_flight = True
                    return True
                return False
            # half_open
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state == "closed":
                self._failures = 0
            elif self._state == "half_open":
                self._state = "closed"
                self._failures = 0
                self._probe_in_flight = False

    def record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._state == "closed":
                self._failures += 1
                if self._failures >= self._threshold:
                    self._state = "open"
                    self._open_until = now + self._reset_s
            elif self._state == "half_open":
                self._state = "open"
                self._open_until = now + self._reset_s
                self._probe_in_flight = False

    def reset(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._open_until = None
            self._probe_in_flight = False

    def state(self) -> str:
        with self._lock:
            return self._state

    def failure_count(self) -> int:
        with self._lock:
            return self._failures


# ---------------------------------------------------------------- Rust backend


class _RustCircuitBreaker:
    """Thin adapter over ``dixvision_py_system.circuit_breaker_*``.

    The Rust extension owns a process-wide registry keyed by name, so
    this wrapper only carries the name string. Construction registers
    the breaker on the Rust side; methods forward to the registry.
    """

    __slots__ = ("_name",)

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        reset_timeout_ms: int,
    ) -> None:
        assert _rs is not None  # invariant of the selector below
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._name = name
        _rs.circuit_breaker_register(
            name, int(failure_threshold), int(reset_timeout_ms)
        )

    def allow(self) -> bool:
        assert _rs is not None
        return _rs.circuit_breaker_allow(self._name)

    def record_success(self) -> None:
        assert _rs is not None
        _rs.circuit_breaker_record_success(self._name)

    def record_failure(self) -> None:
        assert _rs is not None
        _rs.circuit_breaker_record_failure(self._name)

    def reset(self) -> None:
        assert _rs is not None
        _rs.circuit_breaker_reset(self._name)

    def state(self) -> str:
        assert _rs is not None
        return _rs.circuit_breaker_state(self._name)

    def failure_count(self) -> int:
        assert _rs is not None
        return _rs.circuit_breaker_failure_count(self._name)


# --------------------------------------------------------------- dual-backend

# The active breaker class. Selection happens at import time.
CircuitBreaker = _RustCircuitBreaker if (_HAVE_RUST and _rs is not None) else _PythonCircuitBreaker


def make_python_breaker(
    name: str, failure_threshold: int, reset_timeout_ms: int
) -> _PythonCircuitBreaker:
    """Construct a Python-reference breaker. Used by the parity suite."""
    return _PythonCircuitBreaker(name, failure_threshold, reset_timeout_ms)


def make_rust_breaker(
    name: str, failure_threshold: int, reset_timeout_ms: int
) -> "_RustCircuitBreaker":
    """Construct a Rust-backed breaker. Parity suite helper."""
    if not (_HAVE_RUST and _rs is not None):
        raise RuntimeError("dixvision_py_system not available")
    return _RustCircuitBreaker(name, failure_threshold, reset_timeout_ms)


__all__ = [
    "CircuitBreaker",
    "make_python_breaker",
    "make_rust_breaker",
]
