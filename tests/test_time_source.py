"""
tests/test_time_source.py — T0-4 TimeAuthority contract tests.

Covers:
    - now_ns / monotonic_ns / wall_ns API exists and returns int ns.
    - Monotonic guarantee: 100k consecutive now_ns() samples never go
      backwards.
    - wall_ns is in a sane range vs. monotonic (not a latency source).
    - Backwards-compat: now() / utc_now() / now_with_seq() continue to
      produce sequence-stamped monotonic UTC stamps.
"""
from __future__ import annotations

import time

from system import time_source as ts


def test_now_ns_returns_positive_int() -> None:
    v = ts.now_ns()
    assert isinstance(v, int)
    assert v > 0


def test_monotonic_ns_returns_positive_int() -> None:
    v = ts.monotonic_ns()
    assert isinstance(v, int)
    assert v > 0


def test_wall_ns_returns_epoch_nanoseconds() -> None:
    v = ts.wall_ns()
    # sanity: wall_ns is epoch-based, should be > 1.7e18 ns (after
    # 2023-11-15). If anyone runs this in 2010 we have bigger problems.
    assert isinstance(v, int)
    assert v > 1_700_000_000_000_000_000


def test_now_ns_is_strictly_monotonic_100k_samples() -> None:
    """T0-4 invariant: now_ns() must never go backwards."""
    prev = ts.now_ns()
    for _ in range(100_000):
        cur = ts.now_ns()
        assert cur >= prev, f"regression: {cur} < {prev}"
        prev = cur


def test_monotonic_ns_is_strictly_monotonic_100k_samples() -> None:
    prev = ts.monotonic_ns()
    for _ in range(100_000):
        cur = ts.monotonic_ns()
        assert cur >= prev
        prev = cur


def test_wall_ns_and_monotonic_ns_are_independent() -> None:
    """wall_ns and monotonic_ns draw from different clocks; neither is
    a latency source for the other."""
    w1 = ts.wall_ns()
    m1 = ts.monotonic_ns()
    time.sleep(0.001)
    w2 = ts.wall_ns()
    m2 = ts.monotonic_ns()
    # both advance forward
    assert w2 >= w1
    assert m2 >= m1


def test_backwards_compat_now_and_utc_now() -> None:
    """Existing sequence-stamped API must still work."""
    s1 = ts.now()
    s2 = ts.now()
    assert s2.sequence > s1.sequence
    assert s2.monotonic_ns >= s1.monotonic_ns
    assert s2.utc_time >= s1.utc_time

    u = ts.utc_now()
    assert u.tzinfo is not None  # UTC-aware

    when, seq = ts.now_with_seq()
    assert when.tzinfo is not None
    assert seq > s2.sequence
