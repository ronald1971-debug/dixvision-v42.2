# ADAPTED FROM: nautilus_trader/nautilus_trader/backtest/engine.py
# ADAPTED FROM: nautilus_trader/nautilus_trader/common/clock.py (TestClock)
"""Deterministic simulation event loop (S-03.1) — adapted from
``nautilus_trader``.

What survives from upstream
---------------------------
* ``nautilus_trader``'s **deterministic backtest event-loop architecture**:
  events are kept in a strict ``(ts_ns, sequence)`` priority queue, the
  simulated clock is *advanced to* each event's timestamp before
  dispatch, and there is no wall-clock read anywhere. This is the
  property nautilus is famous for and the only reason this module
  exists in DIX at all.
* ``TestClock``-style **clock injection**: the engine owns a
  :class:`SimulatedClock` instance and never imports
  ``time``/``datetime``. Every "what time is it?" query goes through
  the injected clock — replays are byte-identical because the clock
  is a deterministic function of the events that have been dispatched
  so far.
* Stable **tie-breaking**: events at identical ``ts_ns`` dispatch in
  insertion order, exactly like nautilus's ``UUID4``-tied event queue
  but using a monotonic in-process sequence counter (no UUID library
  dependency, fully deterministic across replays).

What does NOT survive
---------------------
* The ``nautilus_trader`` runtime classes themselves. nautilus is
  LGPL-3.0; PART 1 of ``DIX_MASTER_CANONICAL.md`` requires the
  *pattern* to be re-implemented in pure Python rather than the LGPL
  classes ported. We import nothing from ``nautilus_trader``.
* Live-trading paths. nautilus's ``LiveEngine`` reaches the wall clock
  and writes to real exchanges. This module is OFFLINE-tier only —
  there is **no live mode** and authority lint blocks
  :mod:`hot_path` from importing :mod:`simulation`.
* Specific event types. nautilus has its own ``OrderFilled`` /
  ``BarUpdated`` / ``QuoteTick`` taxonomy. The DIX engine is event-
  type-agnostic: callers schedule whatever ``payload`` they want
  (typically a :class:`core.contracts.events.SignalEvent` /
  :class:`core.contracts.events.ExecutionEvent`) and supply a
  ``handler`` callable that knows how to interpret it.

Authority
---------
* OFFLINE tier (slow cadence). Never called from
  :mod:`hot_path` — authority lint enforces this with the
  ``simulation*`` package-name boundary check.
* No clock. ``SimulatedClock`` is the only time source; the engine
  never imports ``datetime`` / ``time`` / ``system.time_source``.
* No PRNG. The dispatch order is fully determined by the
  ``(ts_ns, sequence)`` of scheduled events — no random tie-breaking.
* No IO. The engine is a pure in-memory state machine.
* No mutation of inputs. ``schedule_at`` / ``run`` / ``run_until``
  return frozen records and never mutate caller state.

INV-15 (replay determinism)
---------------------------
Two engines that receive the same sequence of ``schedule_at`` calls
followed by the same ``run`` / ``run_until`` calls produce
byte-identical :class:`ScheduledEvent` outputs in byte-identical order.
The :class:`SimulatedClock` they expose advances to byte-identical
``ts_ns`` values. This is verified by
``tests/test_simulation_engine.py``.

References
----------
* ``DIX_MASTER_CANONICAL.md`` §S-03 (canonical adaptation prompt).
* ``nautilus_trader/backtest/engine.py`` —
  ``BacktestEngine.add_data`` and ``BacktestEngine.run`` for the
  scheduling + dispatch pattern.
* ``nautilus_trader/common/clock.py`` — ``TestClock.advance_time``
  for the clock-advance-then-dispatch ordering.
"""

from __future__ import annotations

import dataclasses
import heapq
from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No new pip dependencies — the nautilus pattern is reproduced in pure
Python with stdlib ``heapq`` + ``dataclasses``."""

__all__ = (
    "DeterministicEventLoop",
    "EventHandler",
    "NEW_PIP_DEPENDENCIES",
    "ScheduledEvent",
    "SimulatedClock",
    "TestClock",
)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


@runtime_checkable
class SimulatedClock(Protocol):
    """The injected clock contract — no wall-clock reads anywhere.

    The engine reads ``now_ns()`` and calls ``advance_to(ts_ns)`` once
    per dispatched event. A ``SimulatedClock`` *only* moves forward;
    rewinding raises :class:`ValueError`.
    """

    def now_ns(self) -> int: ...

    def advance_to(self, ts_ns: int) -> None: ...


@dataclasses.dataclass(slots=True)
class TestClock:
    """Default :class:`SimulatedClock` — adapted from
    ``nautilus_trader::common::clock::TestClock``.

    Holds a single monotonic ``ts_ns`` cursor. ``advance_to`` is the
    only mutator; it rejects backwards moves so a caller can never
    accidentally reorder events. ``now_ns`` is a cheap read.

    The clock is intentionally *not* frozen so the engine can advance
    it; everything else about it is deterministic — given the same
    sequence of ``advance_to`` calls, ``now_ns`` returns byte-identical
    values across replays.
    """

    initial_ts_ns: int = 0
    _now_ns: int = dataclasses.field(init=False)

    # ``TestClock`` is the canonical name from
    # ``nautilus_trader.common.clock``. Tell pytest not to collect this
    # class as a test container — the name has nothing to do with
    # pytest test classes.
    __test__ = False

    def __post_init__(self) -> None:
        if not isinstance(self.initial_ts_ns, int):
            raise TypeError(
                f"TestClock.initial_ts_ns must be int, got {type(self.initial_ts_ns).__name__}"
            )
        if self.initial_ts_ns < 0:
            raise ValueError(
                f"TestClock.initial_ts_ns must be non-negative, got {self.initial_ts_ns!r}"
            )
        self._now_ns = self.initial_ts_ns

    def now_ns(self) -> int:
        return self._now_ns

    def advance_to(self, ts_ns: int) -> None:
        if not isinstance(ts_ns, int):
            raise TypeError(f"TestClock.advance_to ts_ns must be int, got {type(ts_ns).__name__}")
        if ts_ns < self._now_ns:
            raise ValueError(
                "TestClock.advance_to refuses backwards move: "
                f"current={self._now_ns!r}, requested={ts_ns!r}"
            )
        self._now_ns = ts_ns


# ---------------------------------------------------------------------------
# Scheduled event
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """One event in the priority queue.

    ``ts_ns`` is the simulated dispatch time; ``sequence`` is a
    monotonic in-process counter assigned by the engine on
    :meth:`DeterministicEventLoop.schedule_at`. Together they form the
    total ordering: events sort by ``ts_ns`` first, then by
    ``sequence`` (insertion order). Two engines that schedule the
    same payloads in the same order at the same timestamps produce
    byte-identical orderings (INV-15).

    ``payload`` is intentionally :class:`object` — the engine is event-
    type-agnostic. Callers typically pass typed DIX events
    (``SignalEvent`` / ``ExecutionEvent``) but the loop itself does not
    care; the registered handler is the only place that needs to know
    the payload type.
    """

    ts_ns: int
    sequence: int
    payload: object

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int):
            raise TypeError(f"ScheduledEvent.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"ScheduledEvent.ts_ns must be non-negative, got {self.ts_ns!r}")
        if not isinstance(self.sequence, int):
            raise TypeError(
                f"ScheduledEvent.sequence must be int, got {type(self.sequence).__name__}"
            )
        if self.sequence < 0:
            raise ValueError(f"ScheduledEvent.sequence must be non-negative, got {self.sequence!r}")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


EventHandler = Callable[[ScheduledEvent], None]
"""Caller-supplied callback. Called once per dispatched event, after
the engine has advanced the clock to ``event.ts_ns``. The handler may
:meth:`DeterministicEventLoop.schedule_at` further events at later
timestamps; any attempt to schedule into the past raises
:class:`ValueError`."""


# ---------------------------------------------------------------------------
# Deterministic event loop
# ---------------------------------------------------------------------------


class DeterministicEventLoop:
    """Deterministic event loop adapted from
    ``nautilus_trader.backtest.engine.BacktestEngine``.

    Usage::

        clock = TestClock(initial_ts_ns=1_000)
        loop = DeterministicEventLoop(clock=clock)
        loop.schedule_at(ts_ns=2_000, payload=signal_event_a)
        loop.schedule_at(ts_ns=2_000, payload=signal_event_b)
        loop.schedule_at(ts_ns=3_000, payload=signal_event_c)

        dispatched = loop.run(handler=my_handler)
        # → (a, b, c) in that exact order: a, b are tied at ts_ns=2_000
        # so the loop falls back to insertion order.

    Replays are byte-identical because:
      * The priority queue is keyed by ``(ts_ns, sequence)`` only —
        no PRNG, no hash randomisation.
      * The clock is injected and only ever moves forward via
        :meth:`SimulatedClock.advance_to`, called by the loop itself
        once per dispatch.
      * The handler is the only place caller-supplied logic runs;
        if the handler is deterministic, the whole run is.
    """

    __slots__ = ("_clock", "_heap", "_next_seq", "_dispatched")

    def __init__(self, clock: SimulatedClock | None = None) -> None:
        if clock is None:
            clock = TestClock()
        if not isinstance(clock, SimulatedClock):
            raise TypeError(
                "DeterministicEventLoop.clock must satisfy SimulatedClock, "
                f"got {type(clock).__name__}"
            )
        self._clock: SimulatedClock = clock
        # Min-heap of (ts_ns, sequence, ScheduledEvent). The
        # ScheduledEvent itself is at the third slot so heap ordering
        # never needs to compare payloads (which may be unorderable).
        self._heap: list[tuple[int, int, ScheduledEvent]] = []
        self._next_seq: int = 0
        self._dispatched: int = 0

    # -- read-only accessors -------------------------------------------------

    @property
    def clock(self) -> SimulatedClock:
        return self._clock

    @property
    def pending_count(self) -> int:
        """How many events are still queued (not yet dispatched)."""

        return len(self._heap)

    @property
    def dispatched_count(self) -> int:
        """How many events have been dispatched so far this run."""

        return self._dispatched

    # -- scheduling ----------------------------------------------------------

    def schedule_at(self, ts_ns: int, payload: object) -> ScheduledEvent:
        """Schedule a single event. Returns the frozen
        :class:`ScheduledEvent` so callers can correlate dispatches to
        their original schedules.

        Scheduling at ``ts_ns < clock.now_ns()`` raises
        :class:`ValueError` — the loop never dispatches into the past
        (this is also how nautilus's ``BacktestEngine`` behaves).
        """

        if not isinstance(ts_ns, int):
            raise TypeError(
                f"DeterministicEventLoop.schedule_at ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise ValueError(
                f"DeterministicEventLoop.schedule_at ts_ns must be non-negative, got {ts_ns!r}"
            )
        now = self._clock.now_ns()
        if ts_ns < now:
            raise ValueError(
                "DeterministicEventLoop.schedule_at refuses past "
                f"timestamp: ts_ns={ts_ns!r}, clock_now_ns={now!r}"
            )
        seq = self._next_seq
        self._next_seq += 1
        event = ScheduledEvent(ts_ns=ts_ns, sequence=seq, payload=payload)
        heapq.heappush(self._heap, (ts_ns, seq, event))
        return event

    def schedule_many(self, events: Iterable[tuple[int, object]]) -> tuple[ScheduledEvent, ...]:
        """Convenience: schedule a list of ``(ts_ns, payload)`` pairs in
        order. Equivalent to repeated :meth:`schedule_at` calls and
        preserves the same insertion-order tie-breaking.
        """

        scheduled: list[ScheduledEvent] = []
        for item in events:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "DeterministicEventLoop.schedule_many entries must be "
                    f"(ts_ns, payload) tuples, got {item!r}"
                )
            ts_ns, payload = item
            scheduled.append(self.schedule_at(ts_ns=ts_ns, payload=payload))
        return tuple(scheduled)

    # -- run loops -----------------------------------------------------------

    def run(
        self,
        handler: EventHandler | None = None,
        max_events: int | None = None,
    ) -> tuple[ScheduledEvent, ...]:
        """Drain the queue.

        Dispatches events in strict ``(ts_ns, sequence)`` order. For
        each event it: (1) advances the clock to ``event.ts_ns``, (2)
        calls ``handler(event)`` if supplied, (3) appends the event to
        the returned tuple. Stops after ``max_events`` dispatches if
        given.
        """

        if max_events is not None:
            if not isinstance(max_events, int):
                raise TypeError(
                    "DeterministicEventLoop.run max_events must be int, "
                    f"got {type(max_events).__name__}"
                )
            if max_events < 0:
                raise ValueError(
                    "DeterministicEventLoop.run max_events must be "
                    f"non-negative, got {max_events!r}"
                )

        dispatched: list[ScheduledEvent] = []
        while self._heap:
            if max_events is not None and len(dispatched) >= max_events:
                break
            _ts_ns, _seq, event = heapq.heappop(self._heap)
            self._clock.advance_to(event.ts_ns)
            if handler is not None:
                handler(event)
            dispatched.append(event)
            self._dispatched += 1
        return tuple(dispatched)

    def run_until(
        self,
        ts_ns: int,
        handler: EventHandler | None = None,
    ) -> tuple[ScheduledEvent, ...]:
        """Dispatch every queued event with ``event.ts_ns <= ts_ns``,
        then stop. The clock is left at the last dispatched event's
        ``ts_ns`` (or unchanged if no events fired).
        """

        if not isinstance(ts_ns, int):
            raise TypeError(
                f"DeterministicEventLoop.run_until ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise ValueError(
                f"DeterministicEventLoop.run_until ts_ns must be non-negative, got {ts_ns!r}"
            )

        dispatched: list[ScheduledEvent] = []
        while self._heap and self._heap[0][0] <= ts_ns:
            _ts_ns, _seq, event = heapq.heappop(self._heap)
            self._clock.advance_to(event.ts_ns)
            if handler is not None:
                handler(event)
            dispatched.append(event)
            self._dispatched += 1
        return tuple(dispatched)

    # -- introspection -------------------------------------------------------

    def peek(self) -> ScheduledEvent | None:
        """Return the next event without dispatching, or ``None`` if
        the queue is empty. Useful for unit tests and for callers that
        need to gate on the next event's ``ts_ns``.
        """

        if not self._heap:
            return None
        return self._heap[0][2]

    def drain(
        self,
        _: Any = None,  # noqa: ANN401 (intentional opaque sink)
    ) -> tuple[ScheduledEvent, ...]:
        """Clear the queue without dispatching. Returns the events in
        ``(ts_ns, sequence)`` order so the caller can audit what was
        thrown away. Does NOT advance the clock.
        """

        ordered = sorted(self._heap, key=lambda triple: (triple[0], triple[1]))
        self._heap.clear()
        return tuple(triple[2] for triple in ordered)
