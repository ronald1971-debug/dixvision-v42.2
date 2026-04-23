"""
system/metrics.py
DIX VISION v42.2 — Prometheus Metrics

Tracks: trade_latency_ms, hazard_detection_time_ms,
        governance_decision_time_ms, circuit_breaker_triggers.

# Backends

This module has two interchangeable backends and picks one at import
time:

* **Rust (``dixvision_py_system``)** — PyO3 extension built from
  ``rust/py_system/``. Preferred when the wheel is installed. Counters
  + histograms live in Rust behind a single ``parking_lot::Mutex``;
  the FFI surface takes plain primitives on every call. Histograms
  are stored as ``Vec<f64>`` with the same 10_000 → 5_000 ring-buffer
  trim as the Python reference.
* **Pure Python (fallback)** — the original threading-lock
  implementation kept verbatim so pre-polyglot ledgers still replay
  on a box without the Rust wheel.

Both backends satisfy the same invariants, validated by
``tests/test_metrics_parity.py``:

* ``increment(name, value, labels)`` bumps a counter. Labels are
  folded into the counter key as ``"{name}:{labels}"`` when non-empty.
* ``observe(name, value_ms)`` appends a histogram sample with the
  same 10_000 → 5_000 ring-buffer trim on both sides.
* ``p99(name)`` is the 99th-percentile sample across the current
  buffer, matching the Python reference's indexing scheme
  (``int(len * 0.99)`` clamped to ``len - 1``).
* ``snapshot()`` returns ``{"counters": {..}, "p99": {..}}`` as a
  plain ``dict`` with identical keys on both sides.

Selection is observable via ``backend()``. Both implementations are
importable directly via ``make_python_sink()`` and ``make_rust_sink()``
so parity tests can exercise them without reloading this module or
mutating the environment.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------- Rust backend
# Probe the Rust wheel at import time. The wheel is optional; when it
# is missing we fall back to the Python implementation below without
# raising. This keeps `pytest` green on boxes without cargo.
try:
    import dixvision_py_system as _rs  # type: ignore[import-not-found]

    _HAVE_RUST = all(
        hasattr(_rs, fn)
        for fn in (
            "metrics_increment",
            "metrics_observe",
            "metrics_p99",
            "metrics_snapshot",
        )
    )
except ImportError:  # pragma: no cover — covered by the no-wheel path
    _rs = None  # type: ignore[assignment]
    _HAVE_RUST = False


# -------------------------------------------------------------- Python backend


class _PythonMetricsSink:
    """Pure-CPython reference sink.

    Preserves the exact public API of the pre-polyglot
    ``MetricsSink`` class so callers importing ``MetricsSink`` from
    this module keep working when no Rust wheel is available.
    """

    __slots__ = ("_lock", "_counters", "_histograms")

    _HISTOGRAM_CAP = 10_000
    _HISTOGRAM_KEEP = 5_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict] = None,
    ) -> None:
        key = _compose_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            buf = self._histograms[name]
            buf.append(value_ms)
            if len(buf) > self._HISTOGRAM_CAP:
                self._histograms[name] = buf[-self._HISTOGRAM_KEEP :]

    def p99(self, name: str) -> float:
        with self._lock:
            vals = sorted(self._histograms.get(name, []))
            if not vals:
                return 0.0
            idx = int(len(vals) * 0.99)
            return vals[min(idx, len(vals) - 1)]

    def snapshot(self) -> dict:
        with self._lock:
            # Read p99 inline under the same lock so the snapshot is
            # internally consistent even if writers keep going.
            counters = dict(self._counters)
            p99 = {}
            for name, buf in self._histograms.items():
                if not buf:
                    p99[name] = 0.0
                    continue
                sorted_buf = sorted(buf)
                idx = int(len(sorted_buf) * 0.99)
                p99[name] = sorted_buf[min(idx, len(sorted_buf) - 1)]
            return {"counters": counters, "p99": p99}


def _compose_key(name: str, labels: Optional[object]) -> str:
    """Match the Rust seam exactly: empty / ``None`` labels collapse
    to the bare name; otherwise the key becomes ``"{name}:{labels}"``."""
    if labels is None:
        return name
    return f"{name}:{labels}" if labels else name


# ---------------------------------------------------------------- Rust wrapper


class _RustMetricsSink:
    """Thin adapter delegating to the process-global Rust sink.

    All instances share state because the FFI seam exposes a single
    ``OnceLock<MetricsSink>``. That mirrors the Python module-level
    ``get_metrics()`` singleton so migration is transparent. Tests
    that want isolation must use :class:`_PythonMetricsSink` directly.
    """

    __slots__ = ("_rs",)

    def __init__(self, rust_module: object) -> None:
        self._rs = rust_module

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict] = None,
    ) -> None:
        # Preserve the Python reference's key format by pre-folding
        # labels into the name on the Python side. The Rust seam does
        # the same fold, but only when the caller passes a non-empty
        # label string; routing it here keeps one code path.
        if labels is None:
            self._rs.metrics_increment(name, float(value), None)
            return
        label_str = "" if not labels else str(labels)
        self._rs.metrics_increment(name, float(value), label_str or None)

    def observe(self, name: str, value_ms: float) -> None:
        self._rs.metrics_observe(name, float(value_ms))

    def p99(self, name: str) -> float:
        return float(self._rs.metrics_p99(name))

    def snapshot(self) -> dict:
        counters_list, p99_list = self._rs.metrics_snapshot()
        return {
            "counters": {k: float(v) for k, v in counters_list},
            "p99": {k: float(v) for k, v in p99_list},
        }


# -------------------------------------------------------- Public compatibility


# Preserve the pre-polyglot public name. When the Rust wheel is
# available, ``MetricsSink()`` constructions point to the shared Rust
# sink; otherwise they construct a fresh Python sink. Callers using
# the canonical ``get_metrics()`` accessor below get the same
# behaviour on both paths — parallel ``MetricsSink()`` instantiations
# are discouraged but remain compatible with the Python reference.
def make_python_sink() -> _PythonMetricsSink:
    """Force-construct a fresh pure-Python sink. Used by the parity
    test suite; callers should prefer :func:`get_metrics`."""
    return _PythonMetricsSink()


def make_rust_sink() -> _RustMetricsSink:
    """Force-construct a Rust-backed wrapper. Raises ``RuntimeError``
    when the Rust wheel was not imported — tests gate on
    ``_rust_wheel_available()`` before calling this."""
    if _rs is None or not _HAVE_RUST:
        raise RuntimeError("dixvision_py_system wheel not available")
    return _RustMetricsSink(_rs)


def backend() -> str:
    """Return ``"rust"`` or ``"python"`` depending on which
    implementation backs the process-global sink. Observable for
    logging / health checks."""
    return "rust" if (_HAVE_RUST and _rs is not None) else "python"


MetricsSink = _RustMetricsSink if (_HAVE_RUST and _rs is not None) else _PythonMetricsSink


class LatencyTimer:
    """Context manager for measuring latency.

    Backend-agnostic — takes any sink object that exposes
    ``observe(name, value_ms)``.
    """

    __slots__ = ("_sink", "_name", "_start")

    def __init__(self, sink: object, metric_name: str) -> None:
        self._sink = sink
        self._name = metric_name
        self._start = 0.0

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._sink.observe(self._name, elapsed_ms)


# -------------------------------------------------------------- Singleton path


_metrics: object = None
_init_lock = threading.Lock()


def get_metrics() -> object:
    """Process-wide metrics sink. First call constructs the sink using
    whichever backend was selected at import time; subsequent calls
    return the same instance."""
    global _metrics
    if _metrics is None:
        with _init_lock:
            if _metrics is None:
                if _HAVE_RUST and _rs is not None:
                    _metrics = _RustMetricsSink(_rs)
                else:
                    _metrics = _PythonMetricsSink()
    return _metrics
