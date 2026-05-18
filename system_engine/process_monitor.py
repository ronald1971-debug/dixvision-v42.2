"""I-11 psutil — canonical process-health monitor (RUNTIME_SAFE).

# ADAPTED FROM: psutil
#   https://github.com/giampaolo/psutil
#   Cross-platform process & system utilities; we adopt the field shape of
#   `psutil.Process.as_dict()` (cpu_percent / memory_info.rss / num_fds /
#   num_threads / status) so a future operator-gated swap to a real psutil
#   backend is a byte-equivalent change.

Canonical ledger row (per DIX_MASTER_CANONICAL.md, TIER I, I-11):

    I-11 psutil → system_engine/process_monitor.py
        - Real process metrics for HAZ sensors (cpu / rss / fd / threads / status)
        - RUNTIME_SAFE: never reads wall clock; ts_ns is caller-supplied
        - Lazy seam: `psutil` is NEVER imported at module level
        - stdlib fallback always available (pure value-object surface)

Authority constraints (pinned by AST guardrail tests in tests/test_process_monitor.py):

    INV-15  No top-level forbidden imports
            ({psutil, time, datetime, random, asyncio, os, numpy, torch,
              polars, requests}).  Caller-supplied monotone `ts_ns`.
    B1      No imports from runtime engine tiers (execution / intelligence /
            governance / learning / evolution).
    B27/28/INV-71  No typed-event constructors
            ({PatchProposal, HazardEvent, SignalEvent, ExecutionEvent,
              SystemEvent, LearningUpdate}).  This module returns read-side
            value objects only; downstream HAZ sensors map them to
            `HazardEvent` instances.

Design:

    `ProcessMetrics`        frozen+slotted snapshot of one PID at one ts_ns.
    `ProcessHealthPolicy`   frozen+slotted threshold table (warn / crit per
                            metric).  Defaults pinned at canonical values.
    `ProcessHealthLevel`    enum: OK / WARN / CRIT.
    `ProcessHealth`         frozen+slotted evaluation result.
    `evaluate_metrics`      pure function: ProcessMetrics + policy ⇒ ProcessHealth.
    `ProcessMonitor`        bounded-history accumulator keyed by pid.
    `stdlib_process_monitor_factory`  always-available production default.
    `enable_psutil_factory` lazy seam (imports psutil INSIDE function body
                            only) returning a sampler that reads live OS state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("psutil",)

# ---------------------------------------------------------------------------
# Canonical defaults (pinned by tests).
# ---------------------------------------------------------------------------

DEFAULT_CPU_WARN_PCT: float = 80.0
DEFAULT_CPU_CRIT_PCT: float = 95.0
DEFAULT_RSS_WARN_BYTES: int = 1_073_741_824  # 1 GiB
DEFAULT_RSS_CRIT_BYTES: int = 2_147_483_648  # 2 GiB
DEFAULT_FD_WARN_COUNT: int = 768
DEFAULT_FD_CRIT_COUNT: int = 960
DEFAULT_THREAD_WARN_COUNT: int = 256
DEFAULT_THREAD_CRIT_COUNT: int = 384
DEFAULT_HISTORY_MAXSIZE: int = 256
ALLOWED_STATUSES: tuple[str, ...] = (
    "running",
    "sleeping",
    "disk-sleep",
    "stopped",
    "tracing-stop",
    "zombie",
    "dead",
    "wake-kill",
    "waking",
    "idle",
    "locked",
    "waiting",
    "suspended",
    "parked",
    "unknown",
)


# ---------------------------------------------------------------------------
# Policy.
# ---------------------------------------------------------------------------


def _validate_pos_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_pos_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True, slots=True)
class ProcessHealthPolicy:
    """Threshold table — warn / crit per metric."""

    cpu_warn_pct: float = DEFAULT_CPU_WARN_PCT
    cpu_crit_pct: float = DEFAULT_CPU_CRIT_PCT
    rss_warn_bytes: int = DEFAULT_RSS_WARN_BYTES
    rss_crit_bytes: int = DEFAULT_RSS_CRIT_BYTES
    fd_warn_count: int = DEFAULT_FD_WARN_COUNT
    fd_crit_count: int = DEFAULT_FD_CRIT_COUNT
    thread_warn_count: int = DEFAULT_THREAD_WARN_COUNT
    thread_crit_count: int = DEFAULT_THREAD_CRIT_COUNT
    history_maxsize: int = DEFAULT_HISTORY_MAXSIZE

    def __post_init__(self) -> None:
        _validate_pos_float("cpu_warn_pct", self.cpu_warn_pct)
        _validate_pos_float("cpu_crit_pct", self.cpu_crit_pct)
        _validate_pos_int("rss_warn_bytes", self.rss_warn_bytes)
        _validate_pos_int("rss_crit_bytes", self.rss_crit_bytes)
        _validate_pos_int("fd_warn_count", self.fd_warn_count)
        _validate_pos_int("fd_crit_count", self.fd_crit_count)
        _validate_pos_int("thread_warn_count", self.thread_warn_count)
        _validate_pos_int("thread_crit_count", self.thread_crit_count)
        _validate_pos_int("history_maxsize", self.history_maxsize)
        if self.cpu_crit_pct <= self.cpu_warn_pct:
            raise ValueError("cpu_crit_pct must be > cpu_warn_pct")
        if self.rss_crit_bytes <= self.rss_warn_bytes:
            raise ValueError("rss_crit_bytes must be > rss_warn_bytes")
        if self.fd_crit_count <= self.fd_warn_count:
            raise ValueError("fd_crit_count must be > fd_warn_count")
        if self.thread_crit_count <= self.thread_warn_count:
            raise ValueError("thread_crit_count must be > thread_warn_count")


# ---------------------------------------------------------------------------
# Metrics snapshot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProcessMetrics:
    """Read-side snapshot of one PID at one caller-supplied `ts_ns`.

    Field shape mirrors `psutil.Process.as_dict()` for byte-equivalent swap.
    """

    ts_ns: int
    pid: int
    cpu_pct: float
    rss_bytes: int
    vms_bytes: int
    num_fds: int
    num_threads: int
    status: str

    def __post_init__(self) -> None:
        _validate_pos_int("ts_ns", self.ts_ns)
        if isinstance(self.pid, bool) or not isinstance(self.pid, int):
            raise TypeError(f"pid must be int, got {type(self.pid).__name__}")
        if self.pid <= 0:
            raise ValueError(f"pid must be > 0, got {self.pid}")
        if isinstance(self.cpu_pct, bool) or not isinstance(self.cpu_pct, (int, float)):
            raise TypeError("cpu_pct must be number")
        if self.cpu_pct < 0.0:
            raise ValueError("cpu_pct must be >= 0")
        for fname in ("rss_bytes", "vms_bytes", "num_fds", "num_threads"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{fname} must be int")
            if v < 0:
                raise ValueError(f"{fname} must be >= 0")
        if not isinstance(self.status, str):
            raise TypeError("status must be str")
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {ALLOWED_STATUSES}, got {self.status!r}")


# ---------------------------------------------------------------------------
# Health evaluation.
# ---------------------------------------------------------------------------


class ProcessHealthLevel(str, Enum):  # noqa: UP042 - need str subclass for byte-stable JSON
    OK = "ok"
    WARN = "warn"
    CRIT = "crit"


@dataclass(frozen=True, slots=True)
class ProcessHealth:
    """Result of evaluating one `ProcessMetrics` against a policy."""

    ts_ns: int
    pid: int
    level: ProcessHealthLevel
    breaches: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_pos_int("ts_ns", self.ts_ns)
        if isinstance(self.pid, bool) or not isinstance(self.pid, int):
            raise TypeError("pid must be int")
        if self.pid <= 0:
            raise ValueError("pid must be > 0")
        if not isinstance(self.level, ProcessHealthLevel):
            raise TypeError("level must be ProcessHealthLevel")
        if not isinstance(self.breaches, tuple):
            raise TypeError("breaches must be tuple")
        for b in self.breaches:
            if not isinstance(b, str):
                raise TypeError("each breach must be str")


_DEAD_STATUSES: frozenset[str] = frozenset({"zombie", "dead"})


def evaluate_metrics(metrics: ProcessMetrics, policy: ProcessHealthPolicy) -> ProcessHealth:
    """Pure function: snapshot + policy ⇒ deterministic health verdict.

    Iteration order over checks is fixed (cpu → rss → fds → threads → status)
    so `breaches` is byte-identical across runs.
    """
    if not isinstance(metrics, ProcessMetrics):
        raise TypeError("metrics must be ProcessMetrics")
    if not isinstance(policy, ProcessHealthPolicy):
        raise TypeError("policy must be ProcessHealthPolicy")

    breaches: list[str] = []
    level = ProcessHealthLevel.OK

    if metrics.cpu_pct >= policy.cpu_crit_pct:
        breaches.append(f"cpu>={policy.cpu_crit_pct}:crit")
        level = ProcessHealthLevel.CRIT
    elif metrics.cpu_pct >= policy.cpu_warn_pct:
        breaches.append(f"cpu>={policy.cpu_warn_pct}:warn")
        if level == ProcessHealthLevel.OK:
            level = ProcessHealthLevel.WARN

    if metrics.rss_bytes >= policy.rss_crit_bytes:
        breaches.append(f"rss>={policy.rss_crit_bytes}:crit")
        level = ProcessHealthLevel.CRIT
    elif metrics.rss_bytes >= policy.rss_warn_bytes:
        breaches.append(f"rss>={policy.rss_warn_bytes}:warn")
        if level == ProcessHealthLevel.OK:
            level = ProcessHealthLevel.WARN

    if metrics.num_fds >= policy.fd_crit_count:
        breaches.append(f"fd>={policy.fd_crit_count}:crit")
        level = ProcessHealthLevel.CRIT
    elif metrics.num_fds >= policy.fd_warn_count:
        breaches.append(f"fd>={policy.fd_warn_count}:warn")
        if level == ProcessHealthLevel.OK:
            level = ProcessHealthLevel.WARN

    if metrics.num_threads >= policy.thread_crit_count:
        breaches.append(f"threads>={policy.thread_crit_count}:crit")
        level = ProcessHealthLevel.CRIT
    elif metrics.num_threads >= policy.thread_warn_count:
        breaches.append(f"threads>={policy.thread_warn_count}:warn")
        if level == ProcessHealthLevel.OK:
            level = ProcessHealthLevel.WARN

    if metrics.status in _DEAD_STATUSES:
        breaches.append(f"status={metrics.status}:crit")
        level = ProcessHealthLevel.CRIT

    return ProcessHealth(
        ts_ns=metrics.ts_ns,
        pid=metrics.pid,
        level=level,
        breaches=tuple(breaches),
    )


# ---------------------------------------------------------------------------
# Bounded-history monitor.
# ---------------------------------------------------------------------------


class ProcessMonitor:
    """Bounded per-pid history of `ProcessMetrics` + latest-health lookup.

    The monitor is a pure value-object aggregator — it never samples the OS.
    Callers feed in `ProcessMetrics` from whichever source they choose (the
    stdlib factory accepts synthetic samples; the psutil seam provides a live
    sampler).
    """

    __slots__ = ("_policy", "_history")

    def __init__(self, *, policy: ProcessHealthPolicy) -> None:
        if not isinstance(policy, ProcessHealthPolicy):
            raise TypeError("policy must be ProcessHealthPolicy")
        self._policy = policy
        self._history: dict[int, list[ProcessMetrics]] = {}

    @property
    def policy(self) -> ProcessHealthPolicy:
        return self._policy

    def observe(self, metrics: ProcessMetrics) -> ProcessHealth:
        if not isinstance(metrics, ProcessMetrics):
            raise TypeError("metrics must be ProcessMetrics")
        ring = self._history.setdefault(metrics.pid, [])
        if ring and metrics.ts_ns < ring[-1].ts_ns:
            raise ValueError(
                f"ts_ns must be monotone per pid; got {metrics.ts_ns} "
                f"after {ring[-1].ts_ns} for pid={metrics.pid}"
            )
        ring.append(metrics)
        if len(ring) > self._policy.history_maxsize:
            del ring[0 : len(ring) - self._policy.history_maxsize]
        return evaluate_metrics(metrics, self._policy)

    def latest(self, pid: int) -> ProcessMetrics | None:
        ring = self._history.get(pid)
        if not ring:
            return None
        return ring[-1]

    def history(self, pid: int) -> tuple[ProcessMetrics, ...]:
        return tuple(self._history.get(pid, ()))

    def pids(self) -> tuple[int, ...]:
        return tuple(sorted(self._history.keys()))

    def clear(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Factories.
# ---------------------------------------------------------------------------


def stdlib_process_monitor_factory(*, policy: ProcessHealthPolicy | None = None) -> ProcessMonitor:
    """Always-available production default (no psutil dependency)."""
    return ProcessMonitor(policy=policy or ProcessHealthPolicy())


PsutilSampleFn = Callable[[int, int], ProcessMetrics]


def enable_psutil_factory(
    *, policy: ProcessHealthPolicy | None = None
) -> tuple[ProcessMonitor, PsutilSampleFn]:
    """Lazy seam — imports psutil INSIDE the function body only.

    Returns (monitor, sample_fn) where `sample_fn(pid, ts_ns) -> ProcessMetrics`
    reads live OS state via psutil.  The monitor itself is stdlib-only; the
    seam exists so an operator can wire `monitor.observe(sample_fn(pid, ts))`
    into the harness without the module ever depending on psutil at import time.
    """
    import psutil  # noqa: PLC0415 - lazy seam (function-local by design, INV-15)

    monitor = stdlib_process_monitor_factory(policy=policy)

    def _sample(pid: int, ts_ns: int) -> ProcessMetrics:
        proc = psutil.Process(pid)
        with proc.oneshot():
            cpu = float(proc.cpu_percent(interval=None))
            mem = proc.memory_info()
            try:
                fds = int(proc.num_fds())
            except (AttributeError, psutil.AccessDenied):
                fds = 0
            threads = int(proc.num_threads())
            status = str(proc.status())
        return ProcessMetrics(
            ts_ns=ts_ns,
            pid=pid,
            cpu_pct=cpu,
            rss_bytes=int(mem.rss),
            vms_bytes=int(mem.vms),
            num_fds=fds,
            num_threads=threads,
            status=status if status in ALLOWED_STATUSES else "unknown",
        )

    return monitor, _sample


__all__ = (
    "ALLOWED_STATUSES",
    "DEFAULT_CPU_CRIT_PCT",
    "DEFAULT_CPU_WARN_PCT",
    "DEFAULT_FD_CRIT_COUNT",
    "DEFAULT_FD_WARN_COUNT",
    "DEFAULT_HISTORY_MAXSIZE",
    "DEFAULT_RSS_CRIT_BYTES",
    "DEFAULT_RSS_WARN_BYTES",
    "DEFAULT_THREAD_CRIT_COUNT",
    "DEFAULT_THREAD_WARN_COUNT",
    "NEW_PIP_DEPENDENCIES",
    "ProcessHealth",
    "ProcessHealthLevel",
    "ProcessHealthPolicy",
    "ProcessMetrics",
    "ProcessMonitor",
    "PsutilSampleFn",
    "enable_psutil_factory",
    "evaluate_metrics",
    "stdlib_process_monitor_factory",
)


# Unused imports keep this module honest about its top-level surface.
_: Mapping[str, object] = {}
