"""Tests for the A-20.1 freqtrade-adapted retry mixin.

Coverage:

* :class:`RetryPolicy` validation (negative / non-numeric / inverted bounds).
* :func:`compute_backoff_sec` faithful to freqtrade's quadratic formula,
  clamps, and deterministic jitter.
* :class:`RetryExecutor` happy path, EXHAUSTED, NON_RECOVERABLE, sleep
  schedule, byte-identical 3-run determinism, meta key sorting,
  ``callable_name`` resolution, ``RetryMixin`` integration.
* AST guards:
    - no engine cross-imports (B1)
    - no typed bus event construction (B27 / B28 / INV-71)
    - no clock reads (``time.time`` / ``time.time_ns`` / ``datetime.now`` /
      ``datetime.utcnow``) (B-CLOCK / INV-15)
    - no ``random`` / ``asyncio`` / ``os`` / ``numpy`` / ``torch`` imports
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from execution_engine.adapters._retry_mixin import (
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_JITTER_FACTOR,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DELAY_SEC,
    NEW_PIP_DEPENDENCIES,
    NonRecoverableError,
    RetryAttempt,
    RetryExecutor,
    RetryExhausted,
    RetryMixin,
    RetryOutcome,
    RetryPolicy,
    RetryRecord,
    compute_backoff_sec,
    default_is_recoverable,
)

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "execution_engine" / "adapters" / "_retry_mixin.py"
)


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_default_constants() -> None:
    assert DEFAULT_MAX_ATTEMPTS == 4
    assert DEFAULT_BASE_DELAY_SEC == 1.0
    assert DEFAULT_MAX_DELAY_SEC == 60.0
    assert DEFAULT_JITTER_FACTOR == 0.0


# ---------------------------------------------------------------------------
# RetryPolicy validation
# ---------------------------------------------------------------------------


def test_policy_defaults() -> None:
    p = RetryPolicy()
    assert p.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert p.base_delay_sec == DEFAULT_BASE_DELAY_SEC
    assert p.max_delay_sec == DEFAULT_MAX_DELAY_SEC
    assert p.jitter_factor == DEFAULT_JITTER_FACTOR


def test_policy_frozen_is_immutable() -> None:
    p = RetryPolicy()
    with pytest.raises(dataclass_error()):
        p.max_attempts = 1  # type: ignore[misc]


def dataclass_error() -> type[Exception]:
    # Python's frozen dataclasses raise FrozenInstanceError but it lives
    # in dataclasses module; assert via attribute presence is safer.
    import dataclasses

    return dataclasses.FrozenInstanceError


def test_policy_rejects_bool_max_attempts() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(max_attempts=True)  # type: ignore[arg-type]


def test_policy_rejects_negative_max_attempts() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=-1)


def test_policy_rejects_string_max_attempts() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(max_attempts="3")  # type: ignore[arg-type]


def test_policy_rejects_negative_base_delay() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_sec=-0.1)


def test_policy_rejects_bool_base_delay() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(base_delay_sec=True)  # type: ignore[arg-type]


def test_policy_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_sec=10.0, max_delay_sec=5.0)


def test_policy_rejects_jitter_out_of_range() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(jitter_factor=-0.1)
    with pytest.raises(ValueError):
        RetryPolicy(jitter_factor=1.5)


def test_policy_zero_attempts_is_valid() -> None:
    p = RetryPolicy(max_attempts=0)
    assert p.max_attempts == 0


# ---------------------------------------------------------------------------
# compute_backoff_sec — freqtrade math
# ---------------------------------------------------------------------------


def test_backoff_attempt_zero_is_zero() -> None:
    assert compute_backoff_sec(0, RetryPolicy()) == 0.0


def test_backoff_matches_freqtrade_formula_clamped() -> None:
    # freqtrade: calculate_backoff(retrycount, max_retries) = (max - retrycount)**2 + 1
    # With our re-indexing (attempt = max - retrycount): delay(attempt) = attempt**2 + 1.
    # Default policy clamps to [1.0, 60.0]: delays 2, 5, 10, 17 all within range.
    p = RetryPolicy()
    assert compute_backoff_sec(1, p) == 2.0
    assert compute_backoff_sec(2, p) == 5.0
    assert compute_backoff_sec(3, p) == 10.0
    assert compute_backoff_sec(4, p) == 17.0


def test_backoff_respects_max_delay_clamp() -> None:
    p = RetryPolicy(max_delay_sec=5.0)
    # attempt 3 raw = 10 → clamped to 5
    assert compute_backoff_sec(3, p) == 5.0
    assert compute_backoff_sec(10, p) == 5.0


def test_backoff_respects_base_delay_clamp() -> None:
    # No real attempt gives < 2 except attempt 1 (which is exactly 2),
    # so push base_delay up to verify clamp engages.
    p = RetryPolicy(base_delay_sec=10.0, max_delay_sec=60.0)
    assert compute_backoff_sec(1, p) == 10.0
    assert compute_backoff_sec(2, p) == 10.0
    assert compute_backoff_sec(3, p) == 10.0


def test_backoff_with_jitter_is_deterministic() -> None:
    p = RetryPolicy(jitter_factor=0.5)
    a = compute_backoff_sec(2, p, seed=42)
    b = compute_backoff_sec(2, p, seed=42)
    assert a == b


def test_backoff_with_different_seeds_diverges() -> None:
    p = RetryPolicy(jitter_factor=0.5)
    a = compute_backoff_sec(2, p, seed=42)
    b = compute_backoff_sec(2, p, seed=43)
    assert a != b


def test_backoff_with_jitter_stays_finite_and_nonnegative() -> None:
    p = RetryPolicy(jitter_factor=1.0)
    for attempt in range(1, 6):
        for seed in (0, 1, 999, 2**63):
            v = compute_backoff_sec(attempt, p, seed=seed)
            assert math.isfinite(v)
            assert v >= 0.0


def test_backoff_rejects_negative_attempt() -> None:
    with pytest.raises(ValueError):
        compute_backoff_sec(-1, RetryPolicy())


def test_backoff_rejects_bool_attempt() -> None:
    with pytest.raises(TypeError):
        compute_backoff_sec(True, RetryPolicy())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# default_is_recoverable
# ---------------------------------------------------------------------------


def test_default_is_recoverable_always_false() -> None:
    assert default_is_recoverable(RuntimeError("x")) is False
    assert default_is_recoverable(TimeoutError("y")) is False


# ---------------------------------------------------------------------------
# RetryExecutor happy path
# ---------------------------------------------------------------------------


class _RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, sec: float) -> None:
        self.delays.append(sec)


def test_executor_first_attempt_succeeds_no_sleep() -> None:
    sleeps = _RecordingSleep()
    exec_ = RetryExecutor(sleep_fn=sleeps)
    result, record = exec_.run(lambda: 7, callable_name="ok")
    assert result == 7
    assert record.outcome is RetryOutcome.SUCCESS
    assert record.total_attempts == 1
    assert record.callable_name == "ok"
    assert sleeps.delays == []
    assert record.attempts[0].succeeded is True
    assert record.attempts[0].delay_before_retry_sec == 0.0


def test_executor_retries_then_succeeds() -> None:
    sleeps = _RecordingSleep()
    counter = {"n": 0}

    def flaky() -> int:
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("transient")
        return 42

    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=4, jitter_factor=0.0),
        is_recoverable=lambda exc: isinstance(exc, RuntimeError),
        sleep_fn=sleeps,
    )
    result, record = exec_.run(flaky, callable_name="flaky")
    assert result == 42
    assert record.outcome is RetryOutcome.SUCCESS
    assert record.total_attempts == 3
    # First call: no sleep. Retry 1 → delay 2. Retry 2 → delay 5.
    assert sleeps.delays == [2.0, 5.0]
    assert record.attempts[-1].succeeded is True


# ---------------------------------------------------------------------------
# RetryExecutor exhausted
# ---------------------------------------------------------------------------


def test_executor_exhausts_after_max_attempts() -> None:
    sleeps = _RecordingSleep()

    def always_fail() -> int:
        raise RuntimeError("nope")

    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=2, jitter_factor=0.0),
        is_recoverable=lambda exc: True,
        sleep_fn=sleeps,
    )
    with pytest.raises(RetryExhausted) as info:
        exec_.run(always_fail, callable_name="bust")
    record = info.value.record
    assert record.outcome is RetryOutcome.EXHAUSTED
    assert record.total_attempts == 3
    assert record.callable_name == "bust"
    assert record.final_error_class == "RuntimeError"
    assert record.final_error_message == "nope"
    # 3 attempts: first no sleep, retry 1 → 2.0, retry 2 → 5.0
    assert sleeps.delays == [2.0, 5.0]


def test_executor_zero_max_attempts_runs_once() -> None:
    sleeps = _RecordingSleep()

    def always_fail() -> int:
        raise RuntimeError("nope")

    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=0),
        is_recoverable=lambda exc: True,
        sleep_fn=sleeps,
    )
    with pytest.raises(RetryExhausted) as info:
        exec_.run(always_fail)
    assert info.value.record.total_attempts == 1
    assert sleeps.delays == []


# ---------------------------------------------------------------------------
# RetryExecutor non-recoverable
# ---------------------------------------------------------------------------


def test_executor_non_recoverable_raises_immediately() -> None:
    sleeps = _RecordingSleep()
    calls = {"n": 0}

    def bad_key() -> int:
        calls["n"] += 1
        raise PermissionError("invalid api key")

    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=4, jitter_factor=0.0),
        is_recoverable=lambda exc: not isinstance(exc, PermissionError),
        sleep_fn=sleeps,
    )
    with pytest.raises(NonRecoverableError) as info:
        exec_.run(bad_key, callable_name="auth")
    record = info.value.record
    assert record.outcome is RetryOutcome.NON_RECOVERABLE
    assert record.total_attempts == 1
    assert record.final_error_class == "PermissionError"
    assert calls["n"] == 1
    assert sleeps.delays == []


def test_executor_recoverable_then_non_recoverable() -> None:
    sleeps = _RecordingSleep()
    seq: list[type[Exception]] = [RuntimeError, RuntimeError, PermissionError]

    def stepped() -> int:
        cls = seq.pop(0)
        raise cls("boom")

    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=4, jitter_factor=0.0),
        is_recoverable=lambda exc: not isinstance(exc, PermissionError),
        sleep_fn=sleeps,
    )
    with pytest.raises(NonRecoverableError) as info:
        exec_.run(stepped)
    record = info.value.record
    assert record.outcome is RetryOutcome.NON_RECOVERABLE
    assert record.total_attempts == 3
    assert record.final_error_class == "PermissionError"
    # Two sleeps for the two recoverable retries.
    assert sleeps.delays == [2.0, 5.0]


# ---------------------------------------------------------------------------
# INV-15 — 3-run byte-identical determinism
# ---------------------------------------------------------------------------


def _run_executor_scenario(
    seed: int,
) -> tuple[RetryRecord, tuple[float, ...]]:
    sleeps = _RecordingSleep()
    counter = {"n": 0}

    def flaky() -> int:
        counter["n"] += 1
        if counter["n"] < 4:
            raise RuntimeError("t")
        return 1

    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=4, jitter_factor=0.4),
        is_recoverable=lambda exc: isinstance(exc, RuntimeError),
        sleep_fn=sleeps,
        prng_seed=seed,
    )
    _, record = exec_.run(flaky, callable_name="flaky")
    return record, tuple(sleeps.delays)


def test_three_run_byte_identical_replay() -> None:
    r1, s1 = _run_executor_scenario(seed=0xC0FFEE)
    r2, s2 = _run_executor_scenario(seed=0xC0FFEE)
    r3, s3 = _run_executor_scenario(seed=0xC0FFEE)
    assert r1 == r2 == r3
    assert s1 == s2 == s3


def test_three_run_byte_identical_with_failures() -> None:
    def scenario(seed: int) -> RetryRecord:
        def always_fail() -> int:
            raise RuntimeError("x")

        exec_ = RetryExecutor(
            policy=RetryPolicy(max_attempts=3, jitter_factor=0.7),
            is_recoverable=lambda exc: True,
            sleep_fn=lambda _: None,
            prng_seed=seed,
        )
        try:
            exec_.run(always_fail, callable_name="x")
        except RetryExhausted as exc:
            return exc.record
        raise AssertionError("expected RetryExhausted")

    r1 = scenario(0xBEEF)
    r2 = scenario(0xBEEF)
    r3 = scenario(0xBEEF)
    assert r1 == r2 == r3
    assert r1.outcome is RetryOutcome.EXHAUSTED


def test_different_seeds_give_different_delays() -> None:
    def delays(seed: int) -> tuple[float, ...]:
        sleeps = _RecordingSleep()

        def always_fail() -> int:
            raise RuntimeError("x")

        exec_ = RetryExecutor(
            policy=RetryPolicy(max_attempts=3, jitter_factor=0.9),
            is_recoverable=lambda exc: True,
            sleep_fn=sleeps,
            prng_seed=seed,
        )
        try:
            exec_.run(always_fail)
        except RetryExhausted:
            pass
        return tuple(sleeps.delays)

    a = delays(1)
    b = delays(2)
    assert a != b


# ---------------------------------------------------------------------------
# meta key sorting
# ---------------------------------------------------------------------------


def test_meta_keys_are_sorted() -> None:
    exec_ = RetryExecutor(sleep_fn=lambda _: None)
    _, record = exec_.run(lambda: 1, meta={"z": "1", "a": "2", "m": "3"})
    assert list(record.meta.keys()) == ["a", "m", "z"]


def test_meta_defaults_to_empty() -> None:
    exec_ = RetryExecutor(sleep_fn=lambda _: None)
    _, record = exec_.run(lambda: 1)
    assert dict(record.meta) == {}


# ---------------------------------------------------------------------------
# callable_name resolution
# ---------------------------------------------------------------------------


def test_callable_name_falls_back_to_fn_name() -> None:
    def named_fn() -> int:
        return 0

    exec_ = RetryExecutor(sleep_fn=lambda _: None)
    _, record = exec_.run(named_fn)
    assert record.callable_name == "named_fn"


def test_callable_name_falls_back_to_anonymous_for_lambdas() -> None:
    exec_ = RetryExecutor(sleep_fn=lambda _: None)
    _, record = exec_.run(lambda: 0)
    # Lambdas have __name__ == "<lambda>"; we still surface it.
    assert record.callable_name == "<lambda>"


def test_explicit_callable_name_wins() -> None:
    def named_fn() -> int:
        return 0

    exec_ = RetryExecutor(sleep_fn=lambda _: None)
    _, record = exec_.run(named_fn, callable_name="override")
    assert record.callable_name == "override"


# ---------------------------------------------------------------------------
# RetryAttempt / RetryRecord invariants
# ---------------------------------------------------------------------------


def test_retry_attempt_is_frozen() -> None:
    a = RetryAttempt(
        index=0,
        succeeded=True,
        error_class="",
        error_message="",
        delay_before_retry_sec=0.0,
    )
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        a.index = 1  # type: ignore[misc]


def test_retry_record_is_frozen() -> None:
    r = RetryRecord(
        callable_name="x",
        outcome=RetryOutcome.SUCCESS,
        total_attempts=1,
        attempts=(),
        final_error_class="",
        final_error_message="",
    )
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        r.outcome = RetryOutcome.EXHAUSTED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetryMixin integration
# ---------------------------------------------------------------------------


class _FakeAdapter(RetryMixin):
    def __init__(self, executor: RetryExecutor) -> None:
        super().__init__(retry_executor=executor)

    def call(self, fn) -> tuple[int, RetryRecord]:
        return self._retry_call(fn, callable_name="venue.call")


def test_mixin_routes_through_executor() -> None:
    sleeps = _RecordingSleep()
    exec_ = RetryExecutor(
        policy=RetryPolicy(max_attempts=2, jitter_factor=0.0),
        is_recoverable=lambda exc: True,
        sleep_fn=sleeps,
    )
    adapter = _FakeAdapter(exec_)
    result, record = adapter.call(lambda: 99)
    assert result == 99
    assert record.callable_name == "venue.call"
    assert adapter.retry_policy.max_attempts == 2


def test_mixin_defaults_when_no_executor_passed() -> None:
    class _AdapterDefault(RetryMixin):
        pass

    adapter = _AdapterDefault()
    assert adapter.retry_policy.max_attempts == DEFAULT_MAX_ATTEMPTS


def test_executor_seed_normalized_to_uint64() -> None:
    exec_ = RetryExecutor(prng_seed=-1)
    assert exec_.prng_seed == (1 << 64) - 1


def test_executor_rejects_bool_seed() -> None:
    with pytest.raises(TypeError):
        RetryExecutor(prng_seed=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AST guards — module is hot-path-safe and authority-respecting
# ---------------------------------------------------------------------------


def _module_ast() -> ast.AST:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _import_names() -> set[str]:
    tree = _module_ast()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.add(mod)
    return out


def test_no_top_level_random_or_asyncio_imports() -> None:
    imports = _import_names()
    forbidden = {"random", "asyncio", "os", "datetime"}
    assert imports.isdisjoint(forbidden), f"forbidden imports present: {imports & forbidden}"


def test_no_numpy_or_torch_imports() -> None:
    imports = _import_names()
    forbidden = {"numpy", "torch", "scipy", "polars", "pandas"}
    assert imports.isdisjoint(forbidden)


def test_no_engine_cross_imports() -> None:
    imports = _import_names()
    for mod in imports:
        for forbidden_prefix in (
            "governance_engine",
            "system_engine",
            "intelligence_engine",
            "evolution_engine",
        ):
            assert not mod.startswith(forbidden_prefix), f"forbidden engine import: {mod}"


def test_no_typed_bus_event_construction() -> None:
    tree = _module_ast()
    forbidden = {
        "HazardEvent",
        "SignalEvent",
        "PatchProposal",
        "GovernanceDecision",
        "ExecutionEvent",
        "SystemEvent",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                name = None
            assert name not in forbidden, f"typed bus event constructed: {name}"


def test_no_clock_reads_b_clock() -> None:
    tree = _module_ast()
    forbidden = {
        ("time", "time"),
        ("time", "time_ns"),
        ("time", "monotonic_ns"),
        ("time", "perf_counter_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name):
            continue
        pair = (func.value.id, func.attr)
        assert pair not in forbidden, f"forbidden clock read: {pair}"


def test_module_does_not_import_external_freqtrade() -> None:
    imports = _import_names()
    for mod in imports:
        assert not mod.startswith("freqtrade"), f"freqtrade import leaked: {mod}"


def test_adapted_from_header_present() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: freqtrade/exchange/common.py" in text
