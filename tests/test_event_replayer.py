"""Tests for ``simulation/event_replayer.py`` (S-03.2 nautilus_trader)."""

from __future__ import annotations

import pytest

from simulation.engine import ScheduledEvent, TestClock
from simulation.event_replayer import (
    NEW_PIP_DEPENDENCIES,
    EventLogEntry,
    EventReplayer,
    ReplayResult,
)

# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_no_pip_dependency_added() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_public_surface_is_documented() -> None:
    import simulation.event_replayer as m

    assert set(m.__all__) == {
        "EventLogEntry",
        "EventReplayer",
        "NEW_PIP_DEPENDENCIES",
        "ReplayResult",
    }


# ---------------------------------------------------------------------------
# EventLogEntry
# ---------------------------------------------------------------------------


def test_event_log_entry_basic() -> None:
    e = EventLogEntry(ts_ns=1_000, payload="x")
    assert e.ts_ns == 1_000
    assert e.payload == "x"


def test_event_log_entry_zero_ts_is_valid() -> None:
    e = EventLogEntry(ts_ns=0, payload=None)
    assert e.ts_ns == 0


def test_event_log_entry_rejects_non_int_ts() -> None:
    with pytest.raises(TypeError, match="ts_ns"):
        EventLogEntry(ts_ns=1.0, payload=None)  # type: ignore[arg-type]


def test_event_log_entry_rejects_negative_ts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        EventLogEntry(ts_ns=-1, payload=None)


def test_event_log_entries_with_same_inputs_are_structurally_equal() -> None:
    a = EventLogEntry(ts_ns=1_000, payload="x")
    b = EventLogEntry(ts_ns=1_000, payload="x")
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------


def test_replay_result_basic() -> None:
    r = ReplayResult(
        scenario_id="sc",
        seed=0,
        events_dispatched=3,
        start_ts_ns=1_000,
        end_ts_ns=3_000,
        final_clock_ns=3_000,
    )
    assert r.scenario_id == "sc"
    assert r.events_dispatched == 3


def test_replay_result_rejects_empty_scenario_id() -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        ReplayResult(
            scenario_id="",
            seed=0,
            events_dispatched=0,
            start_ts_ns=0,
            end_ts_ns=0,
            final_clock_ns=0,
        )


def test_replay_result_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        ReplayResult(
            scenario_id="sc",
            seed=-1,
            events_dispatched=0,
            start_ts_ns=0,
            end_ts_ns=0,
            final_clock_ns=0,
        )


def test_replay_result_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_ts_ns"):
        ReplayResult(
            scenario_id="sc",
            seed=0,
            events_dispatched=0,
            start_ts_ns=2_000,
            end_ts_ns=1_000,
            final_clock_ns=1_000,
        )


def test_replay_result_rejects_clock_before_end() -> None:
    with pytest.raises(ValueError, match="final_clock_ns"):
        ReplayResult(
            scenario_id="sc",
            seed=0,
            events_dispatched=0,
            start_ts_ns=0,
            end_ts_ns=2_000,
            final_clock_ns=1_500,
        )


def test_replay_result_rejects_negative_dispatched() -> None:
    with pytest.raises(ValueError, match="events_dispatched"):
        ReplayResult(
            scenario_id="sc",
            seed=0,
            events_dispatched=-1,
            start_ts_ns=0,
            end_ts_ns=0,
            final_clock_ns=0,
        )


# ---------------------------------------------------------------------------
# EventReplayer — construction
# ---------------------------------------------------------------------------


def test_from_iterable_basic() -> None:
    r = EventReplayer.from_iterable(
        [
            EventLogEntry(ts_ns=1_000, payload="a"),
            EventLogEntry(ts_ns=2_000, payload="b"),
        ]
    )
    assert len(r) == 2
    assert r.start_ts_ns == 1_000
    assert r.end_ts_ns == 2_000
    assert r.time_span_ns == 1_000


def test_from_pairs_basic() -> None:
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b")])
    assert len(r) == 2
    assert r.entries[0].payload == "a"
    assert r.entries[1].payload == "b"


def test_from_pairs_rejects_bad_entry() -> None:
    with pytest.raises(TypeError, match="ts_ns, payload"):
        EventReplayer.from_pairs([(1_000,)])  # type: ignore[list-item]


def test_constructor_rejects_non_tuple() -> None:
    with pytest.raises(TypeError, match="must be tuple"):
        EventReplayer([EventLogEntry(ts_ns=0, payload=None)])  # type: ignore[arg-type]


def test_constructor_rejects_non_log_entry_in_tuple() -> None:
    with pytest.raises(TypeError, match="EventLogEntry"):
        EventReplayer((EventLogEntry(ts_ns=0, payload=None), "bad"))  # type: ignore[arg-type]


def test_constructor_rejects_decreasing_ts_order() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        EventReplayer.from_pairs([(2_000, "b"), (1_000, "a")])


def test_constructor_accepts_equal_ts_order() -> None:
    # Equal timestamps are fine (loop tie-breaks by insertion order).
    r = EventReplayer.from_pairs([(1_000, "a"), (1_000, "b"), (1_000, "c")])
    assert len(r) == 3


def test_empty_log_is_valid() -> None:
    r = EventReplayer.from_iterable([])
    assert r.is_empty
    assert len(r) == 0
    assert r.start_ts_ns == 0
    assert r.end_ts_ns == 0
    assert r.time_span_ns == 0


# ---------------------------------------------------------------------------
# EventReplayer — read-only accessors
# ---------------------------------------------------------------------------


def test_iteration_preserves_input_order() -> None:
    entries = [
        EventLogEntry(ts_ns=1_000, payload="a"),
        EventLogEntry(ts_ns=1_000, payload="b"),
        EventLogEntry(ts_ns=2_000, payload="c"),
    ]
    r = EventReplayer.from_iterable(entries)
    assert list(r) == entries


def test_entries_property_returns_frozen_tuple() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    assert isinstance(r.entries, tuple)
    assert len(r.entries) == 1


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


def test_slice_inclusive_on_both_ends() -> None:
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b"), (3_000, "c"), (4_000, "d")])
    sub = r.slice(2_000, 3_000)
    assert tuple(e.payload for e in sub) == ("b", "c")


def test_slice_outside_range_returns_empty() -> None:
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b")])
    sub = r.slice(5_000, 9_000)
    assert sub.is_empty


def test_slice_rejects_end_before_start() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(ValueError, match="end_ts_ns"):
        r.slice(2_000, 1_000)


def test_slice_rejects_negative_start() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(ValueError, match="non-negative"):
        r.slice(-1, 1_000)


def test_slice_rejects_non_int() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(TypeError, match="start_ts_ns"):
        r.slice(1.0, 2_000)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_dispatches_in_log_order() -> None:
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b"), (3_000, "c")])
    result, dispatched = r.replay(scenario_id="sc")
    assert tuple(e.payload for e in dispatched) == ("a", "b", "c")
    assert result.events_dispatched == 3
    assert result.start_ts_ns == 1_000
    assert result.end_ts_ns == 3_000
    assert result.final_clock_ns == 3_000


def test_replay_calls_handler_once_per_event() -> None:
    r = EventReplayer.from_pairs([(1, "a"), (2, "b"), (3, "c")])
    seen: list[str] = []
    r.replay(
        scenario_id="sc",
        handler=lambda e: seen.append(str(e.payload)),
    )
    assert seen == ["a", "b", "c"]


def test_replay_handler_sees_advanced_clock() -> None:
    clock = TestClock(initial_ts_ns=0)
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b")])
    seen: list[int] = []

    def handler(_: ScheduledEvent) -> None:
        seen.append(clock.now_ns())

    r.replay(scenario_id="sc", handler=handler, clock=clock)
    assert seen == [1_000, 2_000]


def test_replay_default_clock_starts_at_log_start() -> None:
    r = EventReplayer.from_pairs([(5_000, "a"), (6_000, "b")])
    seen: list[int] = []
    # Without a caller-supplied clock the replayer builds a TestClock
    # at the log's start_ts_ns; the first handler invocation must
    # therefore see the clock at the *first* event's ts_ns.
    result, _ = r.replay(
        scenario_id="sc",
        handler=lambda e: seen.append(e.ts_ns),
    )
    assert seen == [5_000, 6_000]
    assert result.final_clock_ns == 6_000


def test_replay_empty_log_is_noop() -> None:
    r = EventReplayer.from_iterable([])
    result, dispatched = r.replay(scenario_id="sc")
    assert dispatched == ()
    assert result.events_dispatched == 0
    assert result.final_clock_ns == 0


def test_replay_seed_is_carried_to_result() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    result, _ = r.replay(scenario_id="sc", seed=42)
    assert result.seed == 42


def test_replay_rejects_empty_scenario_id() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(ValueError, match="scenario_id"):
        r.replay(scenario_id="")


def test_replay_rejects_negative_seed() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(ValueError, match="seed"):
        r.replay(scenario_id="sc", seed=-1)


def test_replay_rejects_non_clock() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(TypeError, match="SimulatedClock"):
        r.replay(scenario_id="sc", clock="not a clock")  # type: ignore[arg-type]


def test_replay_rejects_non_str_scenario_id() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(TypeError, match="scenario_id"):
        r.replay(scenario_id=123)  # type: ignore[arg-type]


def test_replay_rejects_non_int_seed() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(TypeError, match="seed"):
        r.replay(scenario_id="sc", seed=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# replay_until
# ---------------------------------------------------------------------------


def test_replay_until_inclusive() -> None:
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b"), (3_000, "c")])
    result, dispatched = r.replay_until(scenario_id="sc", until_ts_ns=2_000)
    assert tuple(e.payload for e in dispatched) == ("a", "b")
    assert result.events_dispatched == 2
    assert result.end_ts_ns == 2_000


def test_replay_until_zero_drops_everything() -> None:
    r = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b")])
    result, dispatched = r.replay_until(scenario_id="sc", until_ts_ns=0)
    assert dispatched == ()
    assert result.events_dispatched == 0


def test_replay_until_rejects_negative() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(ValueError, match="non-negative"):
        r.replay_until(scenario_id="sc", until_ts_ns=-1)


def test_replay_until_rejects_non_int() -> None:
    r = EventReplayer.from_pairs([(1_000, "a")])
    with pytest.raises(TypeError, match="until_ts_ns"):
        r.replay_until(scenario_id="sc", until_ts_ns=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def _build(pairs: list[tuple[int, str]]) -> EventReplayer:
    return EventReplayer.from_pairs(pairs)


def test_replay_byte_identical_across_three_runs() -> None:
    pairs = [
        (1_000, "a"),
        (1_000, "a2"),  # tied with "a"; insertion order wins.
        (2_000, "b"),
        (2_000, "b2"),
        (3_000, "c"),
    ]
    r = _build(pairs)
    out_a = r.replay(scenario_id="sc", seed=7)
    out_b = r.replay(scenario_id="sc", seed=7)
    out_c = r.replay(scenario_id="sc", seed=7)
    assert out_a == out_b == out_c
    result, dispatched = out_a
    assert tuple(e.payload for e in dispatched) == (
        "a",
        "a2",
        "b",
        "b2",
        "c",
    )
    assert result.events_dispatched == 5


def test_two_replayers_with_equal_input_replay_byte_identical() -> None:
    r1 = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b")])
    r2 = EventReplayer.from_pairs([(1_000, "a"), (2_000, "b")])
    assert r1.entries == r2.entries
    assert r1.replay(scenario_id="sc") == r2.replay(scenario_id="sc")


def test_handler_observes_dispatched_events_in_input_order() -> None:
    pairs = [(1_000, "a"), (1_000, "b"), (1_000, "c")]
    r = _build(pairs)
    seen: list[str] = []
    r.replay(
        scenario_id="sc",
        handler=lambda e: seen.append(str(e.payload)),
    )
    assert seen == ["a", "b", "c"]


def test_payload_unorderable_does_not_break_replay() -> None:
    # Three opaque uncomparable payloads at the same ts_ns. Without
    # the loop's (ts_ns, sequence)-keyed heap this would raise
    # TypeError from inside heapq.
    class Opaque:
        pass

    a, b, c = Opaque(), Opaque(), Opaque()
    r = EventReplayer.from_pairs([(1_000, a), (1_000, b), (1_000, c)])
    _result, dispatched = r.replay(scenario_id="sc")
    assert [e.payload for e in dispatched] == [a, b, c]
