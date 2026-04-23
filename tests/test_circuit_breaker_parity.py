"""
tests/test_circuit_breaker_parity.py

Parity suite for the T0-8 circuit-breaker (execution/circuit_breaker.py).
Each backend (pure-Python reference + Rust via dixvision_py_system)
must satisfy identical invariants. The Rust test class is skipped
when the extension is not importable in the current environment.
"""
from __future__ import annotations

import threading
import time
import uuid

import pytest

import execution.circuit_breaker as cb_mod

# -- backend availability probe ----------------------------------------------

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
except ImportError:  # pragma: no cover
    _HAVE_RUST = False


def _unique(prefix: str) -> str:
    """Unique name per-test so the Rust process-global registry
    never leaks state between tests."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# -- shared contract ---------------------------------------------------------


class _BackendContract:
    """Invariants every backend must satisfy. Concrete subclasses
    override :meth:`make_breaker` to pick a backend."""

    def make_breaker(self, threshold: int = 3, reset_ms: int = 50):
        raise NotImplementedError

    # -- state machine -------------------------------------------------------

    def test_default_state_is_closed(self) -> None:
        cb = self.make_breaker()
        assert cb.state() == "closed"
        assert cb.allow() is True
        assert cb.failure_count() == 0

    def test_trips_after_threshold_failures(self) -> None:
        cb = self.make_breaker(threshold=3, reset_ms=10_000)
        for _ in range(3):
            cb.record_failure()
        assert cb.state() == "open"
        assert cb.allow() is False

    def test_success_resets_counter_in_closed(self) -> None:
        cb = self.make_breaker(threshold=3, reset_ms=10_000)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count() == 2
        cb.record_success()
        assert cb.failure_count() == 0
        assert cb.state() == "closed"

    def test_open_transitions_to_half_open_after_timeout(self) -> None:
        cb = self.make_breaker(threshold=1, reset_ms=30)
        cb.record_failure()
        assert cb.state() == "open"
        # Sleep past the reset window; next allow() reserves the probe.
        time.sleep(0.06)
        assert cb.allow() is True
        assert cb.state() == "half_open"
        # Only one probe slot per recovery window.
        assert cb.allow() is False

    def test_half_open_success_closes_breaker(self) -> None:
        cb = self.make_breaker(threshold=1, reset_ms=30)
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow() is True  # probe
        cb.record_success()
        assert cb.state() == "closed"
        assert cb.failure_count() == 0
        assert cb.allow() is True

    def test_half_open_failure_reopens_for_new_window(self) -> None:
        cb = self.make_breaker(threshold=1, reset_ms=30)
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow() is True  # probe
        cb.record_failure()
        assert cb.state() == "open"
        # Immediate re-allow must be rejected; must wait a full
        # new reset window.
        assert cb.allow() is False
        time.sleep(0.06)
        assert cb.allow() is True  # next probe

    def test_reset_clears_all_state(self) -> None:
        cb = self.make_breaker(threshold=2, reset_ms=10_000)
        cb.record_failure()
        cb.record_failure()
        assert cb.state() == "open"
        cb.reset()
        assert cb.state() == "closed"
        assert cb.failure_count() == 0
        assert cb.allow() is True

    def test_zero_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            self.make_breaker(threshold=0, reset_ms=10)

    # -- concurrency ---------------------------------------------------------

    def test_concurrent_failures_trip_exactly_once(self) -> None:
        cb = self.make_breaker(threshold=10, reset_ms=10_000)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(25):
                    cb.record_failure()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cb.state() == "open"
        # saturating add semantics — counter must not wrap regardless
        # of interleaving.
        assert cb.failure_count() >= 10

    def test_half_open_grants_only_one_probe_under_contention(self) -> None:
        cb = self.make_breaker(threshold=1, reset_ms=20)
        cb.record_failure()
        time.sleep(0.04)
        granted: list[bool] = []
        gate = threading.Event()
        lock = threading.Lock()

        def worker() -> None:
            gate.wait()
            got = cb.allow()
            with lock:
                granted.append(got)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join()
        assert granted.count(True) == 1, "exactly one probe slot"


# -- concrete backends -------------------------------------------------------


class TestPythonBackend(_BackendContract):
    def make_breaker(self, threshold: int = 3, reset_ms: int = 50):
        return cb_mod.make_python_breaker(_unique("py"), threshold, reset_ms)


@pytest.mark.skipif(not _HAVE_RUST, reason="dixvision_py_system not importable")
class TestRustBackend(_BackendContract):
    def make_breaker(self, threshold: int = 3, reset_ms: int = 50):
        return cb_mod.make_rust_breaker(_unique("rs"), threshold, reset_ms)


# -- module-level selector ---------------------------------------------------


def test_module_level_selector_picks_rust_when_available() -> None:
    """Confirm ``CircuitBreaker`` points at the Rust class when the
    extension is importable; otherwise at the Python reference."""
    if _HAVE_RUST:
        assert cb_mod.CircuitBreaker is cb_mod._RustCircuitBreaker
    else:
        assert cb_mod.CircuitBreaker is cb_mod._PythonCircuitBreaker
