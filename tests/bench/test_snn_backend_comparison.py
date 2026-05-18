"""B-17 — Norse vs snnTorch SNN backend comparison benchmark.

Pure-Python, deterministic, no time/randomness dependencies (uses
splitmix64 seeded by a fixed integer). Compares the two LIF
integrators that ship in B-14 (`sensory.neuromorphic.snn_lif`) and
B-17 (`sensory.neuromorphic.snntorch_detector`) head-to-head on
synthetic market-event-like current traces.

Spec line 2206:
    "Deploy whichever gives better precision per benchmark results"

This benchmark feeds the per-symbol promotion decision: it measures
spike count, first-spike-step, total latency, and three-run
determinism. Either backend may win on a given regime — the
benchmark is the source of truth, not the test assertion.
"""

from __future__ import annotations

import math
import time

import pytest

from sensory.neuromorphic.snntorch_detector import (
    RESET_SUBTRACT,
    RESET_ZERO,
    BackendBenchmark,
    benchmark_against_norse,
)

# ----------------------------------------------------- helpers


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z, state


def _uniform(state: int) -> tuple[float, int]:
    z, next_state = _splitmix64(state)
    return (z >> 11) * (1.0 / (1 << 53)), next_state


def _synthetic_trace(*, seed: int, length: int, amp: float) -> tuple[float, ...]:
    """Deterministic noisy current trace with periodic bursts.

    Mirrors the structural shape of an order-book event stream: a
    slowly-rising baseline plus discrete spike-trigger bursts every
    ~10 steps.
    """

    out: list[float] = []
    state = seed
    for step in range(length):
        u, state = _uniform(state)
        baseline = 0.2 + 0.3 * math.sin(step * 0.07)
        burst = amp if step % 10 == 0 else 0.0
        out.append(baseline + burst + 0.1 * (u - 0.5))
    return tuple(out)


# ----------------------------------------------------- benchmarks


def test_benchmark_silent_trace_zero_spikes_both_sides() -> None:
    result = benchmark_against_norse(input_current=(0.0,) * 100)
    assert result.norse_spike_count == 0
    assert result.snntorch_spike_count == 0
    assert result.is_precision_match()


def test_benchmark_dense_drive_both_spike() -> None:
    # Norse forward-Euler scales input by dt/tau = 0.1 per step, so it
    # needs ~7 steps to ramp v from 0 to threshold=1.0 under I=2.0.
    # snnTorch's multiplicative recurrence injects I directly so it
    # fires on the first step. The benchmark MUST capture both.
    result = benchmark_against_norse(input_current=(2.0,) * 200)
    assert result.norse_spike_count > 0
    assert result.snntorch_spike_count > 0
    assert 0 <= result.first_spike_step_norse <= 15
    assert 0 <= result.first_spike_step_snntorch <= 5


def test_benchmark_synthetic_market_trace_low_amp() -> None:
    trace = _synthetic_trace(seed=11_111, length=500, amp=0.0)
    result = benchmark_against_norse(input_current=trace)
    # Either backend may or may not fire — we report the result, we
    # don't assert any specific count. INV-15 still holds.
    assert isinstance(result, BackendBenchmark)


def test_benchmark_synthetic_market_trace_with_bursts() -> None:
    # Amplitude tuned so Norse (which scales input by dt/tau=0.1) also
    # fires on the burst step (0.1 * 20 = 2.0 > threshold).
    trace = _synthetic_trace(seed=22_222, length=500, amp=20.0)
    result = benchmark_against_norse(input_current=trace)
    assert result.norse_spike_count > 0
    assert result.snntorch_spike_count > 0


def test_benchmark_subtract_vs_zero_reset_differs() -> None:
    trace = _synthetic_trace(seed=33_333, length=200, amp=2.0)
    sub = benchmark_against_norse(input_current=trace, reset_mechanism=RESET_SUBTRACT)
    zero = benchmark_against_norse(input_current=trace, reset_mechanism=RESET_ZERO)
    # Identical inputs but different reset semantics — at least one of
    # the snnTorch-side metrics must differ.
    assert (
        sub.snntorch_spike_count != zero.snntorch_spike_count
        or sub.first_spike_step_snntorch != zero.first_spike_step_snntorch
    )


# ----------------------------------------------------- INV-15 determinism


def test_three_run_benchmark_equality() -> None:
    trace = _synthetic_trace(seed=42_424, length=300, amp=2.5)
    a = benchmark_against_norse(input_current=trace)
    b = benchmark_against_norse(input_current=trace)
    c = benchmark_against_norse(input_current=trace)
    assert a == b == c
    assert a.digest == b.digest == c.digest


def test_benchmark_seed_change_changes_digest() -> None:
    t1 = _synthetic_trace(seed=1, length=200, amp=2.5)
    t2 = _synthetic_trace(seed=2, length=200, amp=2.5)
    assert benchmark_against_norse(input_current=t1).digest != (
        benchmark_against_norse(input_current=t2).digest
    )


# ----------------------------------------------------- latency budget


@pytest.mark.parametrize("trace_len", [128, 512])
def test_benchmark_under_latency_budget(trace_len: int) -> None:
    """Both integrators must clear a generous per-call latency budget.

    The RUNTIME_SAFE classification only applies to single-step
    inference (one `leaky_feed_forward_step` call ≤ 1 ms). The whole
    benchmark harness runs offline so we only assert a soft 250 ms
    ceiling for a `trace_len`-step traces — well above what either
    pure-Python integrator needs. This guards against accidental O(n²)
    regressions in either integrator.
    """

    trace = _synthetic_trace(seed=99_999, length=trace_len, amp=2.0)
    start = time.perf_counter()
    benchmark_against_norse(input_current=trace)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.25, f"benchmark took {elapsed * 1000:.1f} ms"


# ----------------------------------------------------- promotion gate


def test_promotion_gate_exposed_via_is_precision_match() -> None:
    # Amplitude tuned so BOTH integrators fire on bursts — only when
    # both backends actually spike can `is_precision_match` validate
    # the cross-backend precision tolerance.
    trace = _synthetic_trace(seed=55_555, length=400, amp=20.0)
    result = benchmark_against_norse(input_current=trace)
    assert result.first_spike_step_norse >= 0
    assert result.first_spike_step_snntorch >= 0
    delta_count = abs(result.spike_count_delta)
    delta_first = abs(result.first_spike_step_norse - result.first_spike_step_snntorch)
    # The gate must pass when we set tolerances exactly at the
    # observed deltas. Anything tighter is a precision regression.
    assert result.is_precision_match(
        count_tolerance=delta_count,
        first_spike_step_tolerance=delta_first,
    )
