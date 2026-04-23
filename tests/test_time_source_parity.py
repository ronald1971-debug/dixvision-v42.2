"""
tests/test_time_source_parity.py
DIX VISION v42.2 — T0-4 TimeAuthority parity suite (Rust port canary).

Purpose: prove the Rust-backed time source (``dixvision_py_system``)
satisfies the identical invariants to the pure-Python reference.

Structure:
    * ``_BackendContract`` — abstract base exercising the six
      guarantees documented in ``system/time_source.py``. Applied to
      both backends via the ``make_python_backend`` /
      ``make_rust_backend`` factories.
    * ``TestPythonBackend`` — exercises the pure-Python reference.
    * ``TestRustBackend`` — runs against the Rust backend. Skipped
      cleanly when the ``dixvision_py_system`` wheel is not installed.
    * ``test_backend_selector_*`` — asserts the module-level
      ``backend()`` reports the selected implementation and that
      the public API surface is unchanged.

Each test builds its own backend via the factory, so there is no
cross-test state leakage on the Python side. The Rust
``TimeSource`` is a process-wide singleton whose sequence counter
survives between tests — the invariants are phrased as "gap-free
and strictly increasing", not "starts at 1", to accommodate that.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from datetime import datetime, timezone

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from system import time_source as ts_mod  # noqa: E402


def _rust_wheel_available() -> bool:
    try:
        importlib.import_module("dixvision_py_system")
        return True
    except ImportError:
        return False


class _BackendContract:
    """Six invariants every backend must satisfy. Subclasses set
    ``make_backend``."""

    make_backend = staticmethod(ts_mod.make_python_backend)

    def setup_method(self) -> None:  # pytest hook
        self.now = self.make_backend()  # type: ignore[misc]
        assert self.now is not None, "backend factory returned None"

    def test_sequence_is_gap_free_and_strictly_increasing(self) -> None:
        # The Rust backend owns a process-wide singleton whose
        # sequence counter survives test-level teardown, so we
        # cannot assert an absolute starting value. What we CAN
        # (and do) assert is the property that matters for ledger
        # ordering: the counter is gap-free and strictly increasing
        # by exactly 1 per call.
        a = self.now()
        b = self.now()
        c = self.now()
        assert b.sequence == a.sequence + 1
        assert c.sequence == b.sequence + 1
        assert a.sequence >= 1

    def test_monotonic_ns_strictly_increasing_serial(self) -> None:
        last = -1
        for _ in range(5_000):
            m = self.now().monotonic_ns
            assert m > last, f"not strictly monotonic: {last} then {m}"
            last = m

    def test_utc_tracks_monotonic_delta_within_microsecond(self) -> None:
        a = self.now()
        # Busy a little so the deltas are non-trivial.
        for _ in range(1_000):
            self.now()
        b = self.now()
        mono_delta_ns = b.monotonic_ns - a.monotonic_ns
        utc_delta = (b.utc_time - a.utc_time).total_seconds() * 1e9
        # Python's datetime arithmetic rounds to microseconds, so
        # allow 1.5 us of slop on either side. The Rust backend goes
        # through the same datetime round-trip on the Python side,
        # so the same tolerance applies.
        assert abs(mono_delta_ns - utc_delta) < 1_500, (
            f"utc drift too large: mono_ns={mono_delta_ns} utc_ns={utc_delta}"
        )

    def test_utc_time_is_plausible_unix_window(self) -> None:
        t = self.now()
        jan_2020 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        jan_2100 = datetime(2100, 1, 1, tzinfo=timezone.utc)
        assert jan_2020 < t.utc_time < jan_2100, (
            f"utc_time out of plausible range: {t.utc_time}"
        )

    def test_sequence_gap_free_under_thread_contention(self) -> None:
        # Pin the starting sequence *before* the workers spawn so the
        # expected range is relative to wherever the singleton is
        # right now. Without this the Rust backend test flakes if
        # another test in the suite has already bumped the counter.
        start = self.now().sequence

        results: list[int] = []
        results_lock = threading.Lock()

        def worker() -> None:
            local = [self.now().sequence for _ in range(500)]
            with results_lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results.sort()
        # If the backend has a race the set will have duplicates or
        # gaps relative to ``start``; either shows up here.
        expected = list(range(start + 1, start + 1 + len(results)))
        assert results == expected


class TestPythonBackend(_BackendContract):
    make_backend = staticmethod(ts_mod.make_python_backend)


@pytest.mark.skipif(
    not _rust_wheel_available(),
    reason="dixvision_py_system wheel not installed; build via `maturin develop` in rust/py_system",
)
class TestRustBackend(_BackendContract):
    make_backend = staticmethod(ts_mod.make_rust_backend)


def test_backend_selector_returns_expected_name() -> None:
    name = ts_mod.backend()
    assert name in ("rust", "python")
    if _rust_wheel_available():
        assert name == "rust"
    else:
        assert name == "python"


def test_public_api_surface_unchanged() -> None:
    """Port must not break any existing caller. Exported names and
    ``TimeStamp`` field layout are part of the T0-4 public contract."""
    assert hasattr(ts_mod, "TimeStamp")
    assert hasattr(ts_mod, "now")
    assert hasattr(ts_mod, "now_with_seq")
    assert hasattr(ts_mod, "utc_now")
    assert hasattr(ts_mod, "backend")
    assert hasattr(ts_mod, "make_python_backend")
    assert hasattr(ts_mod, "make_rust_backend")

    ts = ts_mod.now()
    assert ts.__class__.__name__ == "TimeStamp"
    assert {"utc_time", "monotonic_ns", "sequence"} <= set(ts.__dataclass_fields__)
    utc, seq = ts_mod.now_with_seq()
    assert isinstance(utc, datetime) and isinstance(seq, int)
    assert isinstance(ts_mod.utc_now(), datetime)


def test_make_rust_backend_returns_none_when_wheel_missing() -> None:
    """Factory contract: returns ``None`` (never raises) when the
    PyO3 wheel is not importable."""
    if _rust_wheel_available():
        assert ts_mod.make_rust_backend() is not None
    else:
        assert ts_mod.make_rust_backend() is None
