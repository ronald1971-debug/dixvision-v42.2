"""
tests/test_metrics_parity.py
DIX VISION v42.2 — Parity suite for the two MetricsSink backends.

Both backends (pure Python + Rust via ``dixvision_py_system``) MUST
satisfy the same public-API invariants. The Rust suite is skipped
cleanly when the wheel is not present in the environment.

Skeleton:

* ``_BackendContract`` defines every invariant as a method that uses
  ``self.sink`` — a fresh sink built by the subclass's ``make_sink``.
* ``TestPythonBackend`` instantiates the contract against the pure
  Python implementation.
* ``TestRustBackend`` instantiates the contract against the Rust
  implementation. Because the Rust sink is a process-global singleton
  we cannot reset it between tests; the Rust tests therefore use
  unique metric-name prefixes per test class instance to avoid
  interference.
"""
from __future__ import annotations

import threading
import uuid

import pytest

import system.metrics as metrics_mod


def _rust_wheel_available() -> bool:
    """True iff ``dixvision_py_system`` was importable AND exposed the
    metrics FFI surface. Mirrors the probe inside ``system.metrics``
    so Rust tests skip cleanly on a no-cargo box."""
    return metrics_mod._HAVE_RUST and metrics_mod._rs is not None


# -------------------------------------------------------------- Contract


class _BackendContract:
    """Invariants every backend must satisfy. Subclasses override
    ``make_sink`` (a classmethod-equivalent staticmethod) and
    ``_prefix`` (for Rust-singleton isolation)."""

    make_sink: staticmethod = staticmethod(metrics_mod.make_python_sink)
    _prefix: str = ""  # Rust backend overrides with a unique prefix

    @pytest.fixture(autouse=True)
    def _setup_sink(self) -> None:
        # ``make_sink`` returns a fresh Python instance on the Python
        # path; on the Rust path all instances share a singleton so
        # we instead route each test's metric names through a unique
        # prefix. The prefix is generated per-test to guarantee
        # isolation even when tests run in parallel.
        self.sink = type(self).make_sink()
        self._prefix = uuid.uuid4().hex[:12] + "_"

    # Helpers so test bodies stay concise.
    def _n(self, name: str) -> str:
        return self._prefix + name

    # -------------------------------------------------------- invariants

    def test_increment_without_labels_accumulates(self) -> None:
        n = self._n("trades")
        self.sink.increment(n, 1.0)
        self.sink.increment(n, 2.5)
        snap = self.sink.snapshot()
        assert snap["counters"][n] == pytest.approx(3.5)

    def test_increment_default_value_is_one(self) -> None:
        n = self._n("ticks")
        self.sink.increment(n)
        self.sink.increment(n)
        self.sink.increment(n)
        snap = self.sink.snapshot()
        assert snap["counters"][n] == pytest.approx(3.0)

    def test_increment_with_labels_keys_separately(self) -> None:
        n = self._n("trades_l")
        self.sink.increment(n, 1.0, labels={"side": "buy"})
        self.sink.increment(n, 2.0, labels={"side": "sell"})
        snap = self.sink.snapshot()
        # Labels are folded into the counter key as "{name}:{labels}".
        # str({}) differs from str({"side": "buy"}) between calls, but
        # the two distinct labels must produce two distinct keys.
        keys = [k for k in snap["counters"] if k.startswith(n)]
        assert len(keys) == 2, keys
        total = sum(snap["counters"][k] for k in keys)
        assert total == pytest.approx(3.0)

    def test_observe_reports_p99_across_buffer(self) -> None:
        n = self._n("latency_ms")
        for v in range(1, 101):
            self.sink.observe(n, float(v))
        # int(100 * 0.99) = 99, clamped to len-1=99, sorted[99] = 100.0
        assert self.sink.p99(n) == pytest.approx(100.0)

    def test_p99_of_never_observed_metric_is_zero(self) -> None:
        n = self._n("never_observed")
        assert self.sink.p99(n) == 0.0

    def test_snapshot_reports_p99_per_metric(self) -> None:
        fast = self._n("fast")
        slow = self._n("slow")
        self.sink.observe(fast, 1.0)
        self.sink.observe(slow, 1000.0)
        snap = self.sink.snapshot()
        assert snap["p99"][fast] == pytest.approx(1.0)
        assert snap["p99"][slow] == pytest.approx(1000.0)

    def test_snapshot_is_detached_from_sink(self) -> None:
        n = self._n("a")
        self.sink.increment(n, 1.0)
        before = self.sink.snapshot()
        self.sink.increment(n, 100.0)
        # The earlier snapshot must not reflect the later increment.
        assert before["counters"][n] == pytest.approx(1.0)
        assert self.sink.snapshot()["counters"][n] == pytest.approx(101.0)

    def test_observe_ring_buffer_trims_on_overflow(self) -> None:
        n = self._n("ringbuf")
        # Push 10_100 samples to overflow the 10_000 cap; after trim
        # the backend retains the tail 5_000 + the 100 newer samples.
        # Verify p99 still reports a plausible value and no exception.
        for v in range(10_100):
            self.sink.observe(n, float(v))
        val = self.sink.p99(n)
        assert val > 0.0
        snap = self.sink.snapshot()
        assert n in snap["p99"]

    def test_concurrent_writers_do_not_lose_counts(self) -> None:
        n = self._n("ctr")
        workers = 8
        per_worker = 1_000

        def bump() -> None:
            for _ in range(per_worker):
                self.sink.increment(n, 1.0)

        threads = [threading.Thread(target=bump) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = self.sink.snapshot()
        assert snap["counters"][n] == pytest.approx(float(workers * per_worker))

    def test_concurrent_snapshot_never_tears(self) -> None:
        n_ctr = self._n("w")
        n_lat = self._n("lat")
        stop = threading.Event()

        def writer() -> None:
            while not stop.is_set():
                self.sink.increment(n_ctr, 1.0)
                self.sink.observe(n_lat, 2.0)

        t = threading.Thread(target=writer)
        t.start()
        try:
            for _ in range(500):
                snap = self.sink.snapshot()
                # Every counter value in the snapshot must be a finite
                # float; torn reads would produce garbage types.
                for v in snap["counters"].values():
                    assert isinstance(v, float)
                    assert v == v  # not NaN
        finally:
            stop.set()
            t.join()


# -------------------------------------------------------------- Python backend


class TestPythonBackend(_BackendContract):
    """Pure-Python reference implementation."""

    make_sink = staticmethod(metrics_mod.make_python_sink)


# ---------------------------------------------------------------- Rust backend


@pytest.mark.skipif(
    not _rust_wheel_available(),
    reason="dixvision_py_system wheel not built in this environment",
)
class TestRustBackend(_BackendContract):
    """Rust-backed implementation via PyO3. Skipped cleanly when the
    wheel is absent."""

    make_sink = staticmethod(
        lambda: metrics_mod.make_rust_sink() if _rust_wheel_available() else None
    )
