"""Execution runtime monitor — EXEC-08 / Phase 2.

A deterministic counter-based monitor that classifies the execution
engine's health from observed :class:`ExecutionEvent` outcomes plus
caller-provided latency samples. No IO, no clocks: callers feed
``ts_ns`` from the source event.

Outputs feed:

* ``ExecutionEngine.check_self`` (rolled into the engine's
  :class:`HealthStatus`)
* ``governance_engine`` slow-path audit via the canonical event bus
  (the monitor never writes to the ledger directly — that is the
  Governance authority's job).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from core.contracts.engine import HealthState
from core.contracts.events import ExecutionEvent, ExecutionStatus


class RuntimeMonitorState(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class RuntimeMonitorReport:
    """Snapshot of the monitor's view at one instant."""

    state: RuntimeMonitorState
    health: HealthState
    submitted: int
    filled: int
    rejected: int
    failed: int
    fill_rate: float
    reject_rate: float
    fail_rate: float
    p50_latency_ns: int
    p95_latency_ns: int
    p99_latency_ns: int
    queue_depth: int
    detail: str


@dataclass(slots=True)
class _Counters:
    submitted: int = 0
    filled: int = 0
    rejected: int = 0
    failed: int = 0


class RuntimeMonitor:
    """Rolling window monitor.

    Args:
        window: Number of most recent execution outcomes to retain.
        latency_window: Number of most recent latency samples to retain.
        degraded_reject_rate: Reject rate above this is DEGRADED.
        fail_reject_rate: Reject rate above this is FAIL.
        max_queue_depth: Above this, monitor reports DEGRADED.
    """

    name: str = "runtime_monitor"
    spec_id: str = "EXEC-08"

    def __init__(
        self,
        *,
        window: int = 1024,
        latency_window: int = 1024,
        degraded_reject_rate: float = 0.10,
        fail_reject_rate: float = 0.30,
        degraded_fail_rate: float = 0.02,
        fail_fail_rate: float = 0.10,
        max_queue_depth: int = 256,
    ) -> None:
        if window <= 0 or latency_window <= 0:
            raise ValueError("windows must be > 0")
        self._window: deque[ExecutionStatus] = deque(maxlen=window)
        self._latencies_ns: deque[int] = deque(maxlen=latency_window)
        self._counters = _Counters()
        self._queue_depth: int = 0
        self._degraded_reject_rate = degraded_reject_rate
        self._fail_reject_rate = fail_reject_rate
        self._degraded_fail_rate = degraded_fail_rate
        self._fail_fail_rate = fail_fail_rate
        self._max_queue_depth = max_queue_depth

    # -- queries -----------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    # -- mutations ---------------------------------------------------------

    def set_queue_depth(self, depth: int) -> None:
        if depth < 0:
            raise ValueError("queue depth must be >= 0")
        self._queue_depth = depth

    def record(self, event: ExecutionEvent, *, latency_ns: int = 0) -> None:
        """Feed one outcome.

        ``latency_ns`` is the wall-time delta between signal arrival
        and execution-event emission, computed by the caller from the
        source ``SignalEvent.ts_ns``.
        """
        if latency_ns < 0:
            raise ValueError("latency_ns must be >= 0")
        self._counters.submitted += 1
        self._window.append(event.status)
        if event.status is ExecutionStatus.FILLED:
            self._counters.filled += 1
        elif event.status is ExecutionStatus.REJECTED:
            self._counters.rejected += 1
        elif event.status is ExecutionStatus.FAILED:
            self._counters.failed += 1
        if latency_ns > 0:
            self._latencies_ns.append(latency_ns)

    def record_batch(
        self,
        events: Iterable[ExecutionEvent],
        *,
        latency_ns: int = 0,
    ) -> None:
        for ev in events:
            self.record(ev, latency_ns=latency_ns)

    def report(self) -> RuntimeMonitorReport:
        c = self._counters
        n = max(1, c.submitted)
        fill_rate = c.filled / n
        reject_rate = c.rejected / n
        fail_rate = c.failed / n

        if (
            fail_rate >= self._fail_fail_rate
            or reject_rate >= self._fail_reject_rate
        ):
            state = RuntimeMonitorState.FAIL
            health = HealthState.FAIL
        elif (
            fail_rate >= self._degraded_fail_rate
            or reject_rate >= self._degraded_reject_rate
            or self._queue_depth > self._max_queue_depth
        ):
            state = RuntimeMonitorState.DEGRADED
            health = HealthState.DEGRADED
        else:
            state = RuntimeMonitorState.OK
            health = HealthState.OK

        p50 = self._percentile(self._latencies_ns, 0.50)
        p95 = self._percentile(self._latencies_ns, 0.95)
        p99 = self._percentile(self._latencies_ns, 0.99)

        return RuntimeMonitorReport(
            state=state,
            health=health,
            submitted=c.submitted,
            filled=c.filled,
            rejected=c.rejected,
            failed=c.failed,
            fill_rate=fill_rate,
            reject_rate=reject_rate,
            fail_rate=fail_rate,
            p50_latency_ns=p50,
            p95_latency_ns=p95,
            p99_latency_ns=p99,
            queue_depth=self._queue_depth,
            detail=(
                f"submitted={c.submitted} filled={c.filled} "
                f"rejected={c.rejected} failed={c.failed}"
            ),
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _percentile(samples: deque[int], q: float) -> int:
        if not samples:
            return 0
        ordered = sorted(samples)
        if q <= 0.0:
            return ordered[0]
        if q >= 1.0:
            return ordered[-1]
        # Nearest-rank — deterministic, no interpolation surprises.
        rank = max(0, min(len(ordered) - 1, int(q * len(ordered))))
        return ordered[rank]


__all__ = [
    "RuntimeMonitor",
    "RuntimeMonitorReport",
    "RuntimeMonitorState",
]
