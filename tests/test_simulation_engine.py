"""Tests for ``simulation/engine.py`` (S-03.1 nautilus_trader)."""

from __future__ import annotations

import pytest

from simulation.engine import (
    NEW_PIP_DEPENDENCIES,
    DeterministicEventLoop,
    ScheduledEvent,
    SimulatedClock,
    TestClock,
)

# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_no_pip_dependency_added() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_public_surface_is_documented() -> None:
    import simulation.engine as m

    assert set(m.__all__) == {
        "DeterministicEventLoop",
        "EventHandler",
        "NEW_PIP_DEPENDENCIES",
        "ScheduledEvent",
        "SimulatedClock",
        "TestClock",
    }


# ---------------------------------------------------------------------------
# TestClock
# ---------------------------------------------------------------------------


def test_test_clock_default_is_zero() -> None:
    c = TestClock()
    assert c.now_ns() == 0


def test_test_clock_initial_ts() -> None:
    c = TestClock(initial_ts_ns=1_000)
    assert c.now_ns() == 1_000


def test_test_clock_advance_to_moves_forward() -> None:
    c = TestClock(initial_ts_ns=0)
    c.advance_to(2_000)
    assert c.now_ns() == 2_000
    c.advance_to(3_000)
    assert c.now_ns() == 3_000


def test_test_clock_advance_to_same_ts_is_noop() -> None:
    c = TestClock(initial_ts_ns=2_000)
    c.advance_to(2_000)
    assert c.now_ns() == 2_000


def test_test_clock_rejects_backwards() -> None:
    c = TestClock(initial_ts_ns=2_000)
    with pytest.raises(ValueError, match="backwards"):
        c.advance_to(1_999)


def test_test_clock_rejects_negative_init() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TestClock(initial_ts_ns=-1)


def test_test_clock_rejects_non_int_init() -> None:
    with pytest.raises(TypeError, match="must be int"):
        TestClock(initial_ts_ns=1.0)  # type: ignore[arg-type]


def test_test_clock_rejects_non_int_advance() -> None:
    c = TestClock()
    with pytest.raises(TypeError, match="must be int"):
        c.advance_to(1.5)  # type: ignore[arg-type]


def test_test_clock_satisfies_simulated_clock_protocol() -> None:
    c = TestClock()
    assert isinstance(c, SimulatedClock)


# ---------------------------------------------------------------------------
# ScheduledEvent
# ---------------------------------------------------------------------------


def test_scheduled_event_basic() -> None:
    e = ScheduledEvent(ts_ns=1_000, sequence=0, payload="x")
    assert e.ts_ns == 1_000
    assert e.sequence == 0
    assert e.payload == "x"


def test_scheduled_event_rejects_non_int_ts() -> None:
    with pytest.raises(TypeError, match="ts_ns"):
        ScheduledEvent(ts_ns=1.0, sequence=0, payload=None)  # type: ignore[arg-type]


def test_scheduled_event_rejects_negative_ts() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        ScheduledEvent(ts_ns=-1, sequence=0, payload=None)


def test_scheduled_event_rejects_negative_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        ScheduledEvent(ts_ns=0, sequence=-1, payload=None)


# ---------------------------------------------------------------------------
# DeterministicEventLoop — construction
# ---------------------------------------------------------------------------


def test_loop_default_clock_is_test_clock_zero() -> None:
    loop = DeterministicEventLoop()
    assert loop.clock.now_ns() == 0
    assert loop.pending_count == 0
    assert loop.dispatched_count == 0


def test_loop_accepts_supplied_clock() -> None:
    c = TestClock(initial_ts_ns=5_000)
    loop = DeterministicEventLoop(clock=c)
    assert loop.clock is c


def test_loop_rejects_non_clock() -> None:
    with pytest.raises(TypeError, match="SimulatedClock"):
        DeterministicEventLoop(clock="not a clock")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def test_schedule_at_assigns_sequence_in_order() -> None:
    loop = DeterministicEventLoop()
    a = loop.schedule_at(ts_ns=1_000, payload="a")
    b = loop.schedule_at(ts_ns=2_000, payload="b")
    c = loop.schedule_at(ts_ns=1_500, payload="c")
    assert a.sequence == 0
    assert b.sequence == 1
    assert c.sequence == 2
    assert loop.pending_count == 3


def test_schedule_at_rejects_past_ts() -> None:
    c = TestClock(initial_ts_ns=2_000)
    loop = DeterministicEventLoop(clock=c)
    with pytest.raises(ValueError, match="past timestamp"):
        loop.schedule_at(ts_ns=1_999, payload=None)


def test_schedule_at_rejects_non_int_ts() -> None:
    loop = DeterministicEventLoop()
    with pytest.raises(TypeError, match="ts_ns"):
        loop.schedule_at(ts_ns=1.0, payload=None)  # type: ignore[arg-type]


def test_schedule_at_rejects_negative_ts() -> None:
    loop = DeterministicEventLoop()
    with pytest.raises(ValueError, match="non-negative"):
        loop.schedule_at(ts_ns=-1, payload=None)


def test_schedule_many_preserves_insertion_order() -> None:
    loop = DeterministicEventLoop()
    out = loop.schedule_many(
        [
            (1_000, "a"),
            (2_000, "b"),
            (1_500, "c"),
        ]
    )
    assert tuple(e.sequence for e in out) == (0, 1, 2)
    assert tuple(e.payload for e in out) == ("a", "b", "c")


def test_schedule_many_rejects_bad_entry() -> None:
    loop = DeterministicEventLoop()
    with pytest.raises(TypeError, match="ts_ns, payload"):
        loop.schedule_many([(1_000,)])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Run — strict ordering
# ---------------------------------------------------------------------------


def test_run_dispatches_in_ts_order() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=3_000, payload="c")
    loop.schedule_at(ts_ns=1_000, payload="a")
    loop.schedule_at(ts_ns=2_000, payload="b")
    out = loop.run()
    assert tuple(e.payload for e in out) == ("a", "b", "c")
    assert tuple(e.ts_ns for e in out) == (1_000, 2_000, 3_000)


def test_run_tie_breaks_by_insertion_sequence() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1_000, payload="a")
    loop.schedule_at(ts_ns=1_000, payload="b")
    loop.schedule_at(ts_ns=1_000, payload="c")
    out = loop.run()
    # All tied at 1_000 — insertion order wins.
    assert tuple(e.payload for e in out) == ("a", "b", "c")


def test_run_advances_clock_to_each_event_ts() -> None:
    clock = TestClock()
    loop = DeterministicEventLoop(clock=clock)
    loop.schedule_at(ts_ns=1_000, payload=None)
    loop.schedule_at(ts_ns=5_000, payload=None)
    seen: list[int] = []

    def handler(event: ScheduledEvent) -> None:
        # Clock must be advanced to event.ts_ns *before* the handler runs.
        seen.append(clock.now_ns())

    loop.run(handler=handler)
    assert seen == [1_000, 5_000]
    assert clock.now_ns() == 5_000


def test_run_calls_handler_exactly_once_per_event() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1, payload="a")
    loop.schedule_at(ts_ns=2, payload="b")
    loop.schedule_at(ts_ns=3, payload="c")
    seen: list[str] = []
    loop.run(handler=lambda e: seen.append(str(e.payload)))
    assert seen == ["a", "b", "c"]
    assert loop.dispatched_count == 3
    assert loop.pending_count == 0


def test_run_with_no_handler_still_dispatches() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1, payload="a")
    out = loop.run()
    assert len(out) == 1
    assert loop.pending_count == 0


def test_run_max_events_caps_dispatch() -> None:
    loop = DeterministicEventLoop()
    for ts in range(1, 11):
        loop.schedule_at(ts_ns=ts, payload=ts)
    out = loop.run(max_events=3)
    assert len(out) == 3
    assert tuple(e.payload for e in out) == (1, 2, 3)
    assert loop.pending_count == 7


def test_run_max_events_zero_is_noop() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1, payload="a")
    out = loop.run(max_events=0)
    assert out == ()
    assert loop.pending_count == 1


def test_run_max_events_rejects_negative() -> None:
    loop = DeterministicEventLoop()
    with pytest.raises(ValueError, match="non-negative"):
        loop.run(max_events=-1)


def test_handler_can_schedule_future_event() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1_000, payload="trigger")

    def handler(event: ScheduledEvent) -> None:
        if event.payload == "trigger":
            loop.schedule_at(ts_ns=2_000, payload="follow_up")

    out = loop.run(handler=handler)
    assert tuple(e.payload for e in out) == ("trigger", "follow_up")


def test_handler_cannot_schedule_into_past() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=2_000, payload="trigger")

    def handler(event: ScheduledEvent) -> None:
        # Clock is now 2_000; scheduling at 1_500 must fail.
        with pytest.raises(ValueError, match="past timestamp"):
            loop.schedule_at(ts_ns=1_500, payload="bad")

    loop.run(handler=handler)


# ---------------------------------------------------------------------------
# run_until
# ---------------------------------------------------------------------------


def test_run_until_dispatches_inclusive() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1_000, payload="a")
    loop.schedule_at(ts_ns=2_000, payload="b")
    loop.schedule_at(ts_ns=3_000, payload="c")
    out = loop.run_until(ts_ns=2_000)
    assert tuple(e.payload for e in out) == ("a", "b")
    assert loop.pending_count == 1
    # Subsequent run picks up the rest.
    out2 = loop.run_until(ts_ns=10_000)
    assert tuple(e.payload for e in out2) == ("c",)


def test_run_until_no_events_in_range_is_noop() -> None:
    clock = TestClock(initial_ts_ns=0)
    loop = DeterministicEventLoop(clock=clock)
    loop.schedule_at(ts_ns=10_000, payload="a")
    out = loop.run_until(ts_ns=5_000)
    assert out == ()
    # Clock should NOT advance if no events fired.
    assert clock.now_ns() == 0
    assert loop.pending_count == 1


def test_run_until_rejects_negative() -> None:
    loop = DeterministicEventLoop()
    with pytest.raises(ValueError, match="non-negative"):
        loop.run_until(ts_ns=-1)


def test_run_until_rejects_non_int() -> None:
    loop = DeterministicEventLoop()
    with pytest.raises(TypeError, match="ts_ns"):
        loop.run_until(ts_ns=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# peek / drain
# ---------------------------------------------------------------------------


def test_peek_empty_queue_is_none() -> None:
    loop = DeterministicEventLoop()
    assert loop.peek() is None


def test_peek_returns_next_without_dispatching() -> None:
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=2_000, payload="b")
    loop.schedule_at(ts_ns=1_000, payload="a")
    head = loop.peek()
    assert head is not None
    assert head.payload == "a"
    # Peek must not consume.
    assert loop.pending_count == 2


def test_drain_clears_queue_in_order_without_advancing_clock() -> None:
    clock = TestClock(initial_ts_ns=0)
    loop = DeterministicEventLoop(clock=clock)
    loop.schedule_at(ts_ns=2_000, payload="b")
    loop.schedule_at(ts_ns=1_000, payload="a")
    out = loop.drain()
    assert tuple(e.payload for e in out) == ("a", "b")
    assert loop.pending_count == 0
    # Clock must not advance.
    assert clock.now_ns() == 0
    assert loop.dispatched_count == 0


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def _build_and_run(seed_pairs: list[tuple[int, str]]) -> tuple[ScheduledEvent, ...]:
    loop = DeterministicEventLoop()
    for ts, pay in seed_pairs:
        loop.schedule_at(ts_ns=ts, payload=pay)
    return loop.run()


def test_replay_byte_identical_across_runs() -> None:
    pairs = [
        (3_000, "c"),
        (1_000, "a"),
        (2_000, "b"),
        (1_000, "a2"),  # tied with "a"; must dispatch second.
        (2_000, "b2"),  # tied with "b"; must dispatch second.
    ]
    out_a = _build_and_run(pairs)
    out_b = _build_and_run(pairs)
    out_c = _build_and_run(pairs)
    # Same input → same dispatched sequence, byte for byte.
    assert out_a == out_b == out_c
    assert tuple(e.payload for e in out_a) == ("a", "a2", "b", "b2", "c")


def test_payload_unorderable_does_not_break_heap() -> None:
    # Two distinct objects of the same uncomparable type tied at ts_ns.
    # Without the (ts_ns, sequence, event) heap keying, Python's heapq
    # would attempt to compare payloads and raise TypeError.
    class Opaque:
        pass

    a, b, c = Opaque(), Opaque(), Opaque()
    loop = DeterministicEventLoop()
    loop.schedule_at(ts_ns=1_000, payload=a)
    loop.schedule_at(ts_ns=1_000, payload=b)
    loop.schedule_at(ts_ns=1_000, payload=c)
    out = loop.run()
    assert [e.payload for e in out] == [a, b, c]
