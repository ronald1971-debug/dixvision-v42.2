"""P0-1a — system.time_source restoration tests.

Re-verifies the canonical TimeAuthority surface re-introduced after
the rust-deletion (PR #116). Mirrors the contract from PR #7:

* ``now()`` is strictly monotonic and gap-free across threads.
* ``sequence`` increments by exactly 1 per call.
* ``utc_time`` is anchor-derived (no second wall-clock read).
* Hot-path API (``now_ns`` / ``monotonic_ns`` / ``wall_ns``) returns
  positive integers and is non-decreasing.
"""

from __future__ import annotations

import threading

from system.time_source import (
    TimeStamp,
    monotonic_ns,
    now,
    now_ns,
    now_with_seq,
    utc_now,
    wall_ns,
)


def test_now_returns_timestamp() -> None:
    ts = now()
    assert isinstance(ts, TimeStamp)
    assert ts.monotonic_ns > 0
    assert ts.sequence >= 1


def test_now_sequence_strictly_increases() -> None:
    a = now()
    b = now()
    c = now()
    assert b.sequence == a.sequence + 1
    assert c.sequence == b.sequence + 1


def test_now_monotonic_strictly_increases() -> None:
    samples = [now() for _ in range(100)]
    for i in range(1, len(samples)):
        assert samples[i].monotonic_ns > samples[i - 1].monotonic_ns


def test_now_with_seq_matches_now() -> None:
    utc, seq = now_with_seq()
    assert isinstance(seq, int)
    assert seq >= 1
    follow = now()
    assert follow.sequence > seq
    assert follow.utc_time >= utc


def test_utc_now_is_non_decreasing() -> None:
    a = utc_now()
    b = utc_now()
    assert b >= a


def test_hot_path_api_returns_positive_ints() -> None:
    assert now_ns() > 0
    assert monotonic_ns() > 0
    assert wall_ns() > 0


def test_now_ns_non_decreasing_across_calls() -> None:
    samples = [now_ns() for _ in range(1000)]
    for i in range(1, len(samples)):
        assert samples[i] >= samples[i - 1]


def test_concurrent_now_sequence_gap_free() -> None:
    """Across threads, ``sequence`` is contiguous (no duplicates, no gaps)."""

    barrier = threading.Barrier(8)
    seqs: list[int] = []
    seqs_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        local: list[int] = []
        for _ in range(125):
            local.append(now().sequence)
        with seqs_lock:
            seqs.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seqs) == 1000
    seqs.sort()
    assert len(set(seqs)) == 1000
    assert seqs[-1] - seqs[0] == 999
