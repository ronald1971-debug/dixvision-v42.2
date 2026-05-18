"""Tests for I-08 — canonical tenacity-shape retry mixin."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from execution_engine.adapters import _retry_mixin_tenacity as mod
from execution_engine.adapters._retry_mixin import (
    NonRecoverableError,
    RetryExhausted,
    RetryOutcome,
)
from execution_engine.adapters._retry_mixin_tenacity import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DELAY_SEC,
    DEFAULT_MIN_DELAY_SEC,
    DEFAULT_MULTIPLIER_SEC,
    NEW_PIP_DEPENDENCIES,
    TenacityRetryExecutor,
    TenacityRetryPolicy,
    compute_tenacity_wait_sec,
    enable_tenacity_factory,
    stdlib_executor_factory,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _module_tree(rel: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_module_surface() -> None:
    assert NEW_PIP_DEPENDENCIES == ("tenacity",)
    assert DEFAULT_MAX_ATTEMPTS == 5
    assert DEFAULT_MULTIPLIER_SEC == 1.0
    assert DEFAULT_MIN_DELAY_SEC == 1.0
    assert DEFAULT_MAX_DELAY_SEC == 60.0


def test_policy_defaults_match_canonical_doc() -> None:
    p = TenacityRetryPolicy()
    assert p.max_attempts == 5  # stop_after_attempt(5)
    assert p.multiplier_sec == 1.0  # wait_random_exponential(min=1, ...)
    assert p.max_delay_sec == 60.0  # wait_random_exponential(max=60)


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": -1},
        {"multiplier_sec": -0.1},
        {"min_delay_sec": -0.1},
        {"max_delay_sec": -0.1},
        {"jitter_factor": -0.1},
        {"jitter_factor": 1.5},
        {"min_delay_sec": 10.0, "max_delay_sec": 5.0},
    ],
)
def test_policy_rejects_invalid(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        TenacityRetryPolicy(**kwargs)


@pytest.mark.parametrize("bad", [True, False, 1.5, "5", None])
def test_policy_rejects_non_int_max_attempts(bad: Any) -> None:
    with pytest.raises(TypeError):
        TenacityRetryPolicy(max_attempts=bad)


def test_policy_frozen() -> None:
    p = TenacityRetryPolicy()
    with pytest.raises((AttributeError, TypeError)):
        p.max_attempts = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Wait math — wait_random_exponential semantics
# ---------------------------------------------------------------------------


def test_wait_attempt_zero_is_zero() -> None:
    assert compute_tenacity_wait_sec(0, TenacityRetryPolicy()) == 0.0


def test_wait_no_jitter_is_pure_exponential() -> None:
    p = TenacityRetryPolicy(
        multiplier_sec=1.0,
        min_delay_sec=0.0,
        max_delay_sec=1000.0,
        jitter_factor=0.0,
    )
    # attempt 1 → 1 * 2^0 = 1
    # attempt 2 → 1 * 2^1 = 2
    # attempt 3 → 1 * 2^2 = 4
    assert compute_tenacity_wait_sec(1, p) == 1.0
    assert compute_tenacity_wait_sec(2, p) == 2.0
    assert compute_tenacity_wait_sec(3, p) == 4.0
    assert compute_tenacity_wait_sec(4, p) == 8.0


def test_wait_clamped_to_max() -> None:
    p = TenacityRetryPolicy(
        multiplier_sec=1.0,
        min_delay_sec=0.0,
        max_delay_sec=10.0,
        jitter_factor=0.0,
    )
    # 2^10 = 1024 → clamped to 10
    assert compute_tenacity_wait_sec(11, p) == 10.0


def test_wait_clamped_to_min() -> None:
    p = TenacityRetryPolicy(
        multiplier_sec=0.001,
        min_delay_sec=0.5,
        max_delay_sec=10.0,
        jitter_factor=0.0,
    )
    # 0.001 * 2^0 = 0.001 → clamped up to 0.5
    assert compute_tenacity_wait_sec(1, p) == 0.5


def test_wait_jitter_within_range() -> None:
    p = TenacityRetryPolicy(
        multiplier_sec=1.0,
        min_delay_sec=0.0,
        max_delay_sec=1000.0,
        jitter_factor=1.0,
    )
    raw = 4.0  # attempt 3 → 2^2 = 4
    for seed in range(20):
        result = compute_tenacity_wait_sec(3, p, seed=seed)
        assert 0.0 <= result <= raw


def test_wait_deterministic_under_same_seed() -> None:
    p = TenacityRetryPolicy(jitter_factor=1.0)
    for attempt in range(1, 8):
        a = compute_tenacity_wait_sec(attempt, p, seed=42)
        b = compute_tenacity_wait_sec(attempt, p, seed=42)
        assert a == b


def test_wait_different_seeds_diverge() -> None:
    p = TenacityRetryPolicy(jitter_factor=1.0)
    samples = {compute_tenacity_wait_sec(3, p, seed=s) for s in range(10)}
    assert len(samples) > 1  # not all identical


def test_wait_rejects_negative_attempt() -> None:
    with pytest.raises(ValueError):
        compute_tenacity_wait_sec(-1, TenacityRetryPolicy())


def test_wait_rejects_bool_attempt() -> None:
    with pytest.raises(TypeError):
        compute_tenacity_wait_sec(True, TenacityRetryPolicy())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Executor — success path
# ---------------------------------------------------------------------------


def test_executor_succeeds_on_first_attempt() -> None:
    sleeps: list[float] = []
    executor = TenacityRetryExecutor(sleep_fn=sleeps.append)
    result, record = executor.run(lambda: 42, callable_name="ok")
    assert result == 42
    assert record.outcome == RetryOutcome.SUCCESS
    assert record.total_attempts == 1
    assert sleeps == []


def test_executor_succeeds_after_recoverable_retries() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("timeout")
        return 99

    executor = TenacityRetryExecutor(
        policy=TenacityRetryPolicy(max_attempts=4, jitter_factor=0.0),
        is_recoverable=lambda exc: isinstance(exc, ConnectionError),
        sleep_fn=sleeps.append,
    )
    result, record = executor.run(flaky)
    assert result == 99
    assert record.outcome == RetryOutcome.SUCCESS
    assert record.total_attempts == 3
    # Two sleeps before the two retries
    assert len(sleeps) == 2
    assert sleeps[0] == 1.0  # attempt 1 → 2^0 = 1
    assert sleeps[1] == 2.0  # attempt 2 → 2^1 = 2


def test_executor_non_recoverable_raises_immediately() -> None:
    sleeps: list[float] = []
    executor = TenacityRetryExecutor(
        is_recoverable=lambda exc: False,
        sleep_fn=sleeps.append,
    )
    with pytest.raises(NonRecoverableError) as ei:
        executor.run(lambda: (_ for _ in ()).throw(ValueError("bad key")))
    assert ei.value.record.outcome == RetryOutcome.NON_RECOVERABLE
    assert ei.value.record.total_attempts == 1
    assert sleeps == []


def test_executor_exhaustion_raises() -> None:
    sleeps: list[float] = []
    executor = TenacityRetryExecutor(
        policy=TenacityRetryPolicy(max_attempts=2, jitter_factor=0.0),
        is_recoverable=lambda exc: True,
        sleep_fn=sleeps.append,
    )

    def always_fail() -> int:
        raise ConnectionError("flaky")

    with pytest.raises(RetryExhausted) as ei:
        executor.run(always_fail)
    record = ei.value.record
    assert record.outcome == RetryOutcome.EXHAUSTED
    assert record.total_attempts == 3  # 1 initial + 2 retries
    assert len(sleeps) == 2


def test_executor_record_meta_is_frozen_and_sorted() -> None:
    executor = TenacityRetryExecutor()
    _, record = executor.run(lambda: 1, meta={"b": "2", "a": "1"})
    assert list(record.meta.keys()) == ["a", "b"]
    with pytest.raises(TypeError):
        record.meta["c"] = "3"  # type: ignore[index]


def test_executor_record_attempts_are_tuple() -> None:
    executor = TenacityRetryExecutor()
    _, record = executor.run(lambda: 1)
    assert isinstance(record.attempts, tuple)


def test_executor_replay_byte_identical() -> None:
    """INV-15 — three independent runs over the same inputs produce
    byte-identical records (when jitter is on but seeded the same)."""

    def make() -> RetryOutcome:
        calls = {"n": 0}

        def flaky() -> int:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("x")
            return 7

        executor = TenacityRetryExecutor(
            policy=TenacityRetryPolicy(max_attempts=4, jitter_factor=1.0),
            is_recoverable=lambda exc: True,
            sleep_fn=lambda d: None,
            prng_seed=7,
        )
        _, rec = executor.run(flaky, callable_name="flaky")
        return rec

    a = make()
    b = make()
    c = make()

    # Convert to comparable tuples (RetryRecord is a frozen dataclass).
    def _key(r: Any) -> tuple[Any, ...]:
        return (
            r.callable_name,
            r.outcome,
            r.total_attempts,
            tuple(
                (att.index, att.succeeded, att.error_class, att.delay_before_retry_sec)
                for att in r.attempts
            ),
        )

    assert _key(a) == _key(b) == _key(c)


# ---------------------------------------------------------------------------
# Executor — constructor validation
# ---------------------------------------------------------------------------


def test_executor_rejects_bool_seed() -> None:
    with pytest.raises(TypeError):
        TenacityRetryExecutor(prng_seed=True)  # type: ignore[arg-type]


def test_executor_seed_masked_to_64_bits() -> None:
    executor = TenacityRetryExecutor(prng_seed=(1 << 70) | 5)
    assert executor.prng_seed < (1 << 64)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def test_stdlib_factory_returns_executor() -> None:
    ex = stdlib_executor_factory()
    assert isinstance(ex, TenacityRetryExecutor)


def test_stdlib_factory_carries_custom_policy() -> None:
    p = TenacityRetryPolicy(max_attempts=2)
    ex = stdlib_executor_factory(policy=p)
    assert ex.policy is p


# ---------------------------------------------------------------------------
# Lazy seam — tenacity import gated inside function body only
# ---------------------------------------------------------------------------


def test_enable_tenacity_factory_skips_when_tenacity_absent() -> None:
    try:
        import tenacity  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError):
            enable_tenacity_factory()
        return
    ex = enable_tenacity_factory()
    assert hasattr(ex, "run")
    assert hasattr(ex, "policy")


def test_enable_tenacity_factory_runs_when_present() -> None:
    try:
        import tenacity  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("tenacity not installed in this environment")
    ex = enable_tenacity_factory(
        policy=TenacityRetryPolicy(max_attempts=1, jitter_factor=0.0),
        sleep_fn=lambda d: None,
    )
    result, record = ex.run(lambda: "ok", callable_name="probe")
    assert result == "ok"
    assert record.outcome == RetryOutcome.SUCCESS


# ---------------------------------------------------------------------------
# AST guards — no top-level forbidden imports, lazy seam pinned
# ---------------------------------------------------------------------------


_FORBIDDEN_TOPLEVEL_IMPORTS: frozenset[str] = frozenset(
    {
        "tenacity",
        "random",
        "datetime",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "requests",
    }
)


def _toplevel_imports(tree: ast.Module) -> Iterable[str]:
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


def test_no_forbidden_toplevel_imports() -> None:
    tree = _module_tree("execution_engine/adapters/_retry_mixin_tenacity.py")
    for name in _toplevel_imports(tree):
        assert name not in _FORBIDDEN_TOPLEVEL_IMPORTS, name


def test_tenacity_imported_only_inside_enable_seam() -> None:
    """``import tenacity`` must appear *inside* the function body only,
    never at module top level."""

    source = inspect.getsource(mod)
    tree = ast.parse(source)
    # Module body: must have NO tenacity import.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            assert "tenacity" not in names
    # Function body of enable_tenacity_factory: must contain it.
    enable_fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "enable_tenacity_factory"
    )
    inner_imports: list[str] = []
    for node in ast.walk(enable_fn):
        if isinstance(node, ast.Import):
            inner_imports.extend(a.name.split(".")[0] for a in node.names)
    assert "tenacity" in inner_imports


# ---------------------------------------------------------------------------
# Authority guards — B1, B27/B28/INV-71, no wall-clock reads
# ---------------------------------------------------------------------------


_FORBIDDEN_EVENT_CTORS: frozenset[str] = frozenset(
    {
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "GovernanceDecision",
        "LearningUpdate",
        "PatchProposal",
    }
)


def test_no_typed_event_constructors() -> None:
    tree = _module_tree("execution_engine/adapters/_retry_mixin_tenacity.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = ""
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            assert name not in _FORBIDDEN_EVENT_CTORS, name


def test_no_wall_clock_reads() -> None:
    tree = _module_tree("execution_engine/adapters/_retry_mixin_tenacity.py")
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
            target = node.func
            if isinstance(target.value, ast.Name):
                pair = (target.value.id, target.attr)
                assert pair not in forbidden_attrs, pair


def test_no_engine_cross_imports() -> None:
    """B1 — must not import from sibling runtime engines."""

    tree = _module_tree("execution_engine/adapters/_retry_mixin_tenacity.py")
    forbidden_roots = {
        "intelligence_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
        "learning_engine",
        "ui",
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in forbidden_roots, node.module
