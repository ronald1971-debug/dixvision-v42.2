"""Tests for I-11 psutil process monitor.

Authority pins:
    INV-15  No top-level forbidden imports.  Caller-supplied `ts_ns`.
            3-run byte-identical replay under fixed sample sequence.
    B1      No cross-runtime-tier imports.
    B27/28/INV-71  No typed-event constructors.
"""

from __future__ import annotations

import ast
import dataclasses
from importlib import import_module
from pathlib import Path

import pytest

import system_engine.process_monitor as pm_mod
from system_engine.process_monitor import (
    ALLOWED_STATUSES,
    DEFAULT_CPU_CRIT_PCT,
    DEFAULT_CPU_WARN_PCT,
    DEFAULT_FD_CRIT_COUNT,
    DEFAULT_FD_WARN_COUNT,
    DEFAULT_HISTORY_MAXSIZE,
    DEFAULT_RSS_CRIT_BYTES,
    DEFAULT_RSS_WARN_BYTES,
    DEFAULT_THREAD_CRIT_COUNT,
    DEFAULT_THREAD_WARN_COUNT,
    NEW_PIP_DEPENDENCIES,
    ProcessHealth,
    ProcessHealthLevel,
    ProcessHealthPolicy,
    ProcessMetrics,
    ProcessMonitor,
    enable_psutil_factory,
    evaluate_metrics,
    stdlib_process_monitor_factory,
)

# ---------------------------------------------------------------------------
# Module surface.
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_pin() -> None:
    assert NEW_PIP_DEPENDENCIES == ("psutil",)


def test_canonical_defaults_pin() -> None:
    assert DEFAULT_CPU_WARN_PCT == 80.0
    assert DEFAULT_CPU_CRIT_PCT == 95.0
    assert DEFAULT_RSS_WARN_BYTES == 1_073_741_824
    assert DEFAULT_RSS_CRIT_BYTES == 2_147_483_648
    assert DEFAULT_FD_WARN_COUNT == 768
    assert DEFAULT_FD_CRIT_COUNT == 960
    assert DEFAULT_THREAD_WARN_COUNT == 256
    assert DEFAULT_THREAD_CRIT_COUNT == 384
    assert DEFAULT_HISTORY_MAXSIZE == 256


def test_allowed_statuses_contains_canonical_set() -> None:
    for s in ("running", "sleeping", "zombie", "dead", "stopped", "idle", "unknown"):
        assert s in ALLOWED_STATUSES


# ---------------------------------------------------------------------------
# Policy validation.
# ---------------------------------------------------------------------------


def test_policy_defaults_construct() -> None:
    p = ProcessHealthPolicy()
    assert p.cpu_warn_pct == DEFAULT_CPU_WARN_PCT
    assert p.cpu_crit_pct == DEFAULT_CPU_CRIT_PCT


def test_policy_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        ProcessHealthPolicy(cpu_warn_pct=0.0)
    with pytest.raises(ValueError):
        ProcessHealthPolicy(rss_warn_bytes=0)
    with pytest.raises(ValueError):
        ProcessHealthPolicy(fd_warn_count=-1)
    with pytest.raises(ValueError):
        ProcessHealthPolicy(history_maxsize=0)


def test_policy_rejects_non_int_fields() -> None:
    with pytest.raises(TypeError):
        ProcessHealthPolicy(rss_warn_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ProcessHealthPolicy(fd_warn_count="100")  # type: ignore[arg-type]


def test_policy_rejects_crit_le_warn() -> None:
    with pytest.raises(ValueError):
        ProcessHealthPolicy(cpu_warn_pct=90.0, cpu_crit_pct=80.0)
    with pytest.raises(ValueError):
        ProcessHealthPolicy(rss_warn_bytes=2_000_000, rss_crit_bytes=1_000_000)
    with pytest.raises(ValueError):
        ProcessHealthPolicy(fd_warn_count=500, fd_crit_count=500)
    with pytest.raises(ValueError):
        ProcessHealthPolicy(thread_warn_count=400, thread_crit_count=300)


def test_policy_frozen() -> None:
    p = ProcessHealthPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.cpu_warn_pct = 50.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProcessMetrics validation.
# ---------------------------------------------------------------------------


def _mk_metrics(**over: object) -> ProcessMetrics:
    base: dict[str, object] = dict(
        ts_ns=1_000_000_000,
        pid=1234,
        cpu_pct=10.0,
        rss_bytes=100_000_000,
        vms_bytes=200_000_000,
        num_fds=32,
        num_threads=8,
        status="running",
    )
    base.update(over)
    return ProcessMetrics(**base)  # type: ignore[arg-type]


def test_metrics_construct_canonical() -> None:
    m = _mk_metrics()
    assert m.pid == 1234
    assert m.cpu_pct == 10.0
    assert m.status == "running"


def test_metrics_rejects_bad_ts_ns() -> None:
    with pytest.raises(ValueError):
        _mk_metrics(ts_ns=0)
    with pytest.raises(ValueError):
        _mk_metrics(ts_ns=-1)
    with pytest.raises(TypeError):
        _mk_metrics(ts_ns=True)


def test_metrics_rejects_bad_pid() -> None:
    with pytest.raises(ValueError):
        _mk_metrics(pid=0)
    with pytest.raises(TypeError):
        _mk_metrics(pid="x")


def test_metrics_rejects_negative_numerics() -> None:
    with pytest.raises(ValueError):
        _mk_metrics(cpu_pct=-1.0)
    with pytest.raises(ValueError):
        _mk_metrics(rss_bytes=-1)
    with pytest.raises(ValueError):
        _mk_metrics(num_fds=-1)
    with pytest.raises(ValueError):
        _mk_metrics(num_threads=-1)


def test_metrics_rejects_bad_status() -> None:
    with pytest.raises(ValueError):
        _mk_metrics(status="bogus")
    with pytest.raises(TypeError):
        _mk_metrics(status=42)  # type: ignore[arg-type]


def test_metrics_frozen() -> None:
    m = _mk_metrics()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.cpu_pct = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evaluate_metrics — pure threshold function.
# ---------------------------------------------------------------------------


def test_evaluate_ok_path() -> None:
    h = evaluate_metrics(_mk_metrics(), ProcessHealthPolicy())
    assert h.level == ProcessHealthLevel.OK
    assert h.breaches == ()
    assert h.ts_ns == 1_000_000_000
    assert h.pid == 1234


def test_evaluate_cpu_warn() -> None:
    h = evaluate_metrics(_mk_metrics(cpu_pct=80.0), ProcessHealthPolicy())
    assert h.level == ProcessHealthLevel.WARN
    assert any("cpu>=" in b and ":warn" in b for b in h.breaches)


def test_evaluate_cpu_crit() -> None:
    h = evaluate_metrics(_mk_metrics(cpu_pct=99.0), ProcessHealthPolicy())
    assert h.level == ProcessHealthLevel.CRIT
    assert any("cpu>=" in b and ":crit" in b for b in h.breaches)


def test_evaluate_rss_warn_then_crit() -> None:
    p = ProcessHealthPolicy()
    h_warn = evaluate_metrics(_mk_metrics(rss_bytes=p.rss_warn_bytes), p)
    assert h_warn.level == ProcessHealthLevel.WARN
    h_crit = evaluate_metrics(_mk_metrics(rss_bytes=p.rss_crit_bytes), p)
    assert h_crit.level == ProcessHealthLevel.CRIT


def test_evaluate_fd_thresholds() -> None:
    p = ProcessHealthPolicy()
    assert (
        evaluate_metrics(_mk_metrics(num_fds=p.fd_warn_count), p).level == ProcessHealthLevel.WARN
    )
    assert (
        evaluate_metrics(_mk_metrics(num_fds=p.fd_crit_count), p).level == ProcessHealthLevel.CRIT
    )


def test_evaluate_thread_thresholds() -> None:
    p = ProcessHealthPolicy()
    assert (
        evaluate_metrics(_mk_metrics(num_threads=p.thread_warn_count), p).level
        == ProcessHealthLevel.WARN
    )
    assert (
        evaluate_metrics(_mk_metrics(num_threads=p.thread_crit_count), p).level
        == ProcessHealthLevel.CRIT
    )


def test_evaluate_dead_statuses_force_crit() -> None:
    for status in ("zombie", "dead"):
        h = evaluate_metrics(_mk_metrics(status=status), ProcessHealthPolicy())
        assert h.level == ProcessHealthLevel.CRIT
        assert any(f"status={status}" in b for b in h.breaches)


def test_evaluate_warn_does_not_downgrade_after_crit() -> None:
    h = evaluate_metrics(
        _mk_metrics(cpu_pct=99.0, rss_bytes=DEFAULT_RSS_WARN_BYTES),
        ProcessHealthPolicy(),
    )
    assert h.level == ProcessHealthLevel.CRIT


def test_evaluate_breach_order_is_canonical() -> None:
    p = ProcessHealthPolicy()
    h = evaluate_metrics(
        _mk_metrics(
            cpu_pct=p.cpu_warn_pct,
            rss_bytes=p.rss_warn_bytes,
            num_fds=p.fd_warn_count,
            num_threads=p.thread_warn_count,
        ),
        p,
    )
    assert h.level == ProcessHealthLevel.WARN
    keys = [b.split(">=", 1)[0].split("=", 1)[0] for b in h.breaches]
    assert keys == ["cpu", "rss", "fd", "threads"]


def test_evaluate_rejects_bad_args() -> None:
    p = ProcessHealthPolicy()
    with pytest.raises(TypeError):
        evaluate_metrics("not-metrics", p)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_metrics(_mk_metrics(), "not-policy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProcessHealth value object.
# ---------------------------------------------------------------------------


def test_process_health_frozen_and_typed() -> None:
    h = ProcessHealth(ts_ns=1, pid=1, level=ProcessHealthLevel.OK, breaches=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.level = ProcessHealthLevel.CRIT  # type: ignore[misc]


def test_process_health_validation() -> None:
    with pytest.raises(ValueError):
        ProcessHealth(ts_ns=0, pid=1, level=ProcessHealthLevel.OK, breaches=())
    with pytest.raises(ValueError):
        ProcessHealth(ts_ns=1, pid=0, level=ProcessHealthLevel.OK, breaches=())
    with pytest.raises(TypeError):
        ProcessHealth(ts_ns=1, pid=1, level="ok", breaches=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ProcessHealth(
            ts_ns=1,
            pid=1,
            level=ProcessHealthLevel.OK,
            breaches=["x"],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# ProcessMonitor.
# ---------------------------------------------------------------------------


def test_monitor_constructs_and_typed() -> None:
    mon = ProcessMonitor(policy=ProcessHealthPolicy())
    assert mon.pids() == ()
    with pytest.raises(TypeError):
        ProcessMonitor(policy="bad")  # type: ignore[arg-type]


def test_monitor_observe_returns_health() -> None:
    mon = stdlib_process_monitor_factory()
    h = mon.observe(_mk_metrics())
    assert isinstance(h, ProcessHealth)
    assert h.level == ProcessHealthLevel.OK


def test_monitor_history_grouped_by_pid() -> None:
    mon = stdlib_process_monitor_factory()
    mon.observe(_mk_metrics(pid=1, ts_ns=1_000))
    mon.observe(_mk_metrics(pid=2, ts_ns=1_000))
    mon.observe(_mk_metrics(pid=1, ts_ns=2_000))
    assert mon.pids() == (1, 2)
    assert len(mon.history(1)) == 2
    assert len(mon.history(2)) == 1
    assert mon.latest(1) is not None
    assert mon.latest(1).ts_ns == 2_000  # type: ignore[union-attr]
    assert mon.latest(99) is None


def test_monitor_rejects_non_monotone_ts() -> None:
    mon = stdlib_process_monitor_factory()
    mon.observe(_mk_metrics(ts_ns=2_000))
    with pytest.raises(ValueError):
        mon.observe(_mk_metrics(ts_ns=1_000))


def test_monitor_bounded_history_evicts_oldest() -> None:
    mon = ProcessMonitor(policy=ProcessHealthPolicy(history_maxsize=3))
    for i in range(5):
        mon.observe(_mk_metrics(ts_ns=1_000 + i))
    hist = mon.history(1234)
    assert len(hist) == 3
    assert [m.ts_ns for m in hist] == [1_002, 1_003, 1_004]


def test_monitor_clear() -> None:
    mon = stdlib_process_monitor_factory()
    mon.observe(_mk_metrics())
    mon.clear()
    assert mon.pids() == ()


def test_monitor_observe_typed() -> None:
    mon = stdlib_process_monitor_factory()
    with pytest.raises(TypeError):
        mon.observe("not-metrics")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay.
# ---------------------------------------------------------------------------


def _replay_sequence() -> tuple[tuple[int, ...], tuple[str, ...]]:
    mon = stdlib_process_monitor_factory()
    levels: list[str] = []
    breach_counts: list[int] = []
    samples = [
        (1_000, 1, 10.0, 50_000_000, 32, 8, "running"),
        (2_000, 1, 85.0, 50_000_000, 32, 8, "running"),
        (3_000, 1, 99.0, 50_000_000, 32, 8, "running"),
        (4_000, 2, 5.0, DEFAULT_RSS_WARN_BYTES, 32, 8, "sleeping"),
        (5_000, 2, 5.0, DEFAULT_RSS_CRIT_BYTES, 32, 8, "sleeping"),
        (6_000, 3, 5.0, 50_000_000, 32, 8, "zombie"),
    ]
    for ts, pid, cpu, rss, fds, threads, status in samples:
        h = mon.observe(
            ProcessMetrics(
                ts_ns=ts,
                pid=pid,
                cpu_pct=cpu,
                rss_bytes=rss,
                vms_bytes=rss * 2,
                num_fds=fds,
                num_threads=threads,
                status=status,
            )
        )
        levels.append(h.level.value)
        breach_counts.append(len(h.breaches))
    return tuple(breach_counts), tuple(levels)


def test_inv15_three_run_replay_byte_identical() -> None:
    run1 = _replay_sequence()
    run2 = _replay_sequence()
    run3 = _replay_sequence()
    assert run1 == run2 == run3
    counts, levels = run1
    assert levels == ("ok", "warn", "crit", "warn", "crit", "crit")
    assert counts == (0, 1, 1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Factories.
# ---------------------------------------------------------------------------


def test_stdlib_factory_returns_monitor() -> None:
    mon = stdlib_process_monitor_factory()
    assert isinstance(mon, ProcessMonitor)
    assert mon.policy == ProcessHealthPolicy()


def test_stdlib_factory_accepts_custom_policy() -> None:
    p = ProcessHealthPolicy(cpu_warn_pct=10.0, cpu_crit_pct=20.0)
    mon = stdlib_process_monitor_factory(policy=p)
    assert mon.policy is p


def test_enable_psutil_factory_seam_skips_or_runs() -> None:
    try:
        import psutil  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        pytest.skip("psutil not installed; lazy seam tested by AST guards")
    mon, sample = enable_psutil_factory()
    assert isinstance(mon, ProcessMonitor)
    metrics = sample(1, 1_000_000_000)  # init PID 1 (always present on Linux)
    assert isinstance(metrics, ProcessMetrics)
    assert metrics.pid == 1
    assert metrics.ts_ns == 1_000_000_000


# ---------------------------------------------------------------------------
# AST guardrails — INV-15 / B1 / B27/28/INV-71.
# ---------------------------------------------------------------------------


_FORBIDDEN_TOPLEVEL: frozenset[str] = frozenset(
    {
        "psutil",
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "requests",
    }
)


def _toplevel_imports(src: str) -> set[str]:
    tree = ast.parse(src)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".", 1)[0])
    return names


def _function_local_imports(src: str, func_names: set[str]) -> set[str]:
    """Return module names imported INSIDE the body of any function in `func_names`."""
    tree = ast.parse(src)
    found: set[str] = set()
    for fdef in ast.walk(tree):
        if isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)) and fdef.name in func_names:
            for node in ast.walk(fdef):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found.add(alias.name.split(".", 1)[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        found.add(node.module.split(".", 1)[0])
    return found


def _module_src() -> str:
    return Path(pm_mod.__file__).read_text(encoding="utf-8")


def test_no_forbidden_toplevel_imports_inv15() -> None:
    src = _module_src()
    bad = _toplevel_imports(src) & _FORBIDDEN_TOPLEVEL
    assert bad == set(), f"forbidden top-level imports: {bad}"


def test_psutil_imported_only_inside_enable_seam() -> None:
    src = _module_src()
    locals_in_seam = _function_local_imports(src, {"enable_psutil_factory"})
    assert "psutil" in locals_in_seam, "psutil must be imported inside enable_psutil_factory"
    assert "psutil" not in _toplevel_imports(src)


def test_no_typed_event_constructors() -> None:
    """B27 / B28 / INV-71 — module never constructs typed events."""
    src = _module_src()
    tree = ast.parse(src)
    forbidden = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "LearningUpdate",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden, (
                f"forbidden typed-event constructor at line {node.lineno}: {node.func.id}"
            )


def test_no_cross_runtime_tier_imports_b1() -> None:
    """B1 — system_engine must not import from peer runtime tiers."""
    src = _module_src()
    forbidden = {
        "execution_engine",
        "intelligence_engine",
        "governance_engine",
        "learning_engine",
        "evolution_engine",
    }
    bad = _toplevel_imports(src) & forbidden
    assert bad == set(), f"B1 cross-tier imports: {bad}"


def test_no_wall_clock_reads() -> None:
    """No `time.time()` / `time.monotonic()` / `datetime.now()` etc. anywhere."""
    src = _module_src()
    tree = ast.parse(src)
    forbidden_attrs = {
        ("time", "time"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("time", "time_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                key = (node.func.value.id, node.func.attr)
                assert key not in forbidden_attrs, f"wall-clock read at line {node.lineno}: {key}"


def test_module_importable_without_psutil() -> None:
    """Re-import the module fresh — must succeed even if psutil is unavailable."""
    mod = import_module("system_engine.process_monitor")
    assert mod is pm_mod
