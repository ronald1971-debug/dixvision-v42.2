# ADAPTED FROM: nautilus_trader/nautilus_trader/backtest/engine.py
# ADAPTED FROM: nautilus_trader/nautilus_trader/backtest/data_client.py
"""Event-log replayer (S-03.2) — adapted from ``nautilus_trader``.

This is the second and final file of the S-03 nautilus_trader canonical
pair. The first (``simulation/engine.py``) ships the deterministic
event-loop primitive; this file ships the higher-level wrapper that
loads a pre-recorded event log and pumps it through the loop.

What survives from upstream
---------------------------
* ``nautilus_trader``'s **canonical input contract for backtests**:
  every replayable run is described by a list of
  ``(ts_ns, payload)`` rows in **non-decreasing timestamp order**, fed
  in their natural order to a deterministic event-loop. This is the
  only shape that gives byte-identical replays across runs.
* The **validate-then-replay** split: ``BacktestEngine.add_data`` /
  ``BacktestNode.run`` first checks the input is well-ordered, then
  schedules every row, then drains the loop. We do the same — load
  rejects malformed logs *before* any handler observes a single
  event, which prevents a bad log from leaving partial state behind.
* The **slice-then-replay** pattern: ``BacktestEngine`` exposes
  ``backtest_start`` / ``backtest_end`` boundaries on the historical
  feed. We expose :meth:`EventReplayer.slice` for the same purpose.

What does NOT survive
---------------------
* The ``nautilus_trader`` runtime classes themselves. nautilus is
  LGPL-3.0; PART 1 of ``DIX_MASTER_CANONICAL.md`` requires the
  *pattern* to be re-implemented in pure Python rather than the LGPL
  classes ported. We import nothing from ``nautilus_trader``.
* Live-feed drivers. nautilus's data-client classes wrap WebSocket
  feeds, exchange REST adapters, etc. The DIX replayer only operates
  on **recorded** logs (in-memory iterables, JSONL files, fixture
  generators). There is no live mode — replay is offline-tier only,
  and the authority lint already blocks :mod:`hot_path` from
  importing :mod:`simulation`.
* nautilus-specific event types (``QuoteTick`` / ``OrderFilled`` /
  ``BarUpdated`` / ``UUID4`` keys). The replayer is event-type-
  agnostic: ``payload`` is :class:`object` so callers can replay
  typed DIX events (``SignalEvent`` / ``ExecutionEvent``) without the
  replayer importing :mod:`core.events`.

Authority
---------
* OFFLINE tier only. Authority lint blocks :mod:`hot_path` from
  importing :mod:`simulation`.
* No clock. The replayer accepts a :class:`SimulatedClock` (defaults
  to :class:`TestClock` from :mod:`simulation.engine`) and never
  imports ``time`` / ``datetime``.
* No PRNG. Dispatch order is fully ``(ts_ns, sequence)`` from the
  loop; ``seed`` is carried through to :class:`ReplayResult` for
  caller-side correlation but the replayer itself does not consume
  it.
* No IO. Loaders accept iterables; concrete file readers are out of
  scope for this leaf and live in callers.
* No global mutable state. ``EventReplayer`` is constructed from a
  frozen tuple of :class:`EventLogEntry` rows.

INV-15 (replay determinism)
---------------------------
Two ``EventReplayer.replay(...)`` calls with byte-identical
constructor input + byte-identical ``scenario_id``/``seed``/handler
produce byte-identical dispatched event sequences and byte-identical
:class:`ReplayResult` records. Verified by
``tests/test_event_replayer.py``.

References
----------
* ``DIX_MASTER_CANONICAL.md`` §S-03 (canonical adaptation prompt).
* ``nautilus_trader/backtest/engine.py`` —
  ``BacktestEngine.add_data`` for the schedule-from-iterable pattern
  and ``BacktestEngine.run`` for the run-then-summarise pattern.
* ``nautilus_trader/backtest/data_client.py`` —
  ``BacktestDataClient`` for the validate-then-replay split.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator

from simulation.engine import (
    DeterministicEventLoop,
    EventHandler,
    ScheduledEvent,
    SimulatedClock,
    TestClock,
)

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No new pip dependencies — the replayer is a pure-stdlib wrapper
around :mod:`simulation.engine` (which itself is pure-stdlib)."""

__all__ = (
    "EventLogEntry",
    "EventReplayer",
    "NEW_PIP_DEPENDENCIES",
    "ReplayResult",
)


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EventLogEntry:
    """One row in a recorded event log.

    Frozen + slotted so two ``EventReplayer`` instances built from the
    same input tuples are structurally equal — required for the INV-15
    replay-byte-identity property to hold across processes.

    ``ts_ns`` must be a non-negative ``int``. ``payload`` is :class:`object`
    so callers can carry typed DIX events without forcing the replayer
    to import :mod:`core.events`.
    """

    ts_ns: int
    payload: object

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int):
            raise TypeError(f"EventLogEntry.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"EventLogEntry.ts_ns must be non-negative, got {self.ts_ns!r}")


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayResult:
    """Frozen summary of one :meth:`EventReplayer.replay` invocation.

    All fields are deterministic functions of the input log + clock +
    handler-effects-on-loop-state. Two replays of the same log with
    the same handler produce byte-identical :class:`ReplayResult`
    instances (INV-15).
    """

    scenario_id: str
    seed: int
    events_dispatched: int
    start_ts_ns: int
    end_ts_ns: int
    final_clock_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str):
            raise TypeError(
                f"ReplayResult.scenario_id must be str, got {type(self.scenario_id).__name__}"
            )
        if not self.scenario_id:
            raise ValueError("ReplayResult.scenario_id must be non-empty")
        if not isinstance(self.seed, int):
            raise TypeError(f"ReplayResult.seed must be int, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"ReplayResult.seed must be non-negative, got {self.seed!r}")
        if not isinstance(self.events_dispatched, int):
            raise TypeError(
                "ReplayResult.events_dispatched must be int, "
                f"got {type(self.events_dispatched).__name__}"
            )
        if self.events_dispatched < 0:
            raise ValueError(
                "ReplayResult.events_dispatched must be non-negative, "
                f"got {self.events_dispatched!r}"
            )
        for name in ("start_ts_ns", "end_ts_ns", "final_clock_ns"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"ReplayResult.{name} must be int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"ReplayResult.{name} must be non-negative, got {value!r}")
        if self.end_ts_ns < self.start_ts_ns:
            raise ValueError(
                "ReplayResult.end_ts_ns must be >= start_ts_ns, "
                f"got start={self.start_ts_ns!r}, end={self.end_ts_ns!r}"
            )
        if self.final_clock_ns < self.end_ts_ns:
            raise ValueError(
                "ReplayResult.final_clock_ns must be >= end_ts_ns, "
                f"got final={self.final_clock_ns!r}, end={self.end_ts_ns!r}"
            )


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------


class EventReplayer:
    """Loads + validates + replays a pre-recorded event log through a
    :class:`DeterministicEventLoop`.

    Adapted from ``nautilus_trader.backtest.engine.BacktestEngine.add_data``
    + ``BacktestEngine.run`` — the same validate-then-schedule-then-drain
    flow, with the LGPL nautilus classes replaced by pure Python.

    Construction is via :meth:`from_iterable` /
    :meth:`from_pairs` so input always passes a single validation
    chokepoint before any handler runs.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[EventLogEntry, ...]) -> None:
        # Internal constructor — callers should use the factory class
        # methods so input validation is centralised. This still
        # validates the tuple itself so a misuse fails loudly.
        if not isinstance(entries, tuple):
            raise TypeError(f"EventReplayer.entries must be tuple, got {type(entries).__name__}")
        for index, entry in enumerate(entries):
            if not isinstance(entry, EventLogEntry):
                raise TypeError(
                    "EventReplayer.entries must hold EventLogEntry rows, "
                    f"got {type(entry).__name__} at index {index}"
                )
        for index in range(1, len(entries)):
            prev_ts = entries[index - 1].ts_ns
            cur_ts = entries[index].ts_ns
            if cur_ts < prev_ts:
                raise ValueError(
                    "EventReplayer.entries must be non-decreasing in ts_ns, "
                    f"got prev={prev_ts!r} at {index - 1}, "
                    f"cur={cur_ts!r} at {index}"
                )
        self._entries: tuple[EventLogEntry, ...] = entries

    # -- factories -----------------------------------------------------------

    @classmethod
    def from_iterable(cls, entries: Iterable[EventLogEntry]) -> EventReplayer:
        """Build a replayer from an iterable of
        :class:`EventLogEntry` rows. The iterable is consumed
        eagerly so any validation error is raised before
        :meth:`replay` is called.
        """

        materialised = tuple(entries)
        return cls(materialised)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, object]]) -> EventReplayer:
        """Build a replayer from an iterable of ``(ts_ns, payload)``
        tuples. Convenience for callers that don't want to construct
        :class:`EventLogEntry` rows themselves; equivalent to
        :meth:`from_iterable` after wrapping each pair.
        """

        materialised: list[EventLogEntry] = []
        for index, item in enumerate(pairs):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "EventReplayer.from_pairs entries must be "
                    f"(ts_ns, payload) tuples, got {item!r} at index {index}"
                )
            ts_ns, payload = item
            materialised.append(EventLogEntry(ts_ns=ts_ns, payload=payload))
        return cls(tuple(materialised))

    # -- read-only accessors -------------------------------------------------

    @property
    def entries(self) -> tuple[EventLogEntry, ...]:
        """Frozen tuple of log entries in input order."""

        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[EventLogEntry]:
        return iter(self._entries)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    @property
    def start_ts_ns(self) -> int:
        """First entry's ``ts_ns``. Zero if the log is empty."""

        if not self._entries:
            return 0
        return self._entries[0].ts_ns

    @property
    def end_ts_ns(self) -> int:
        """Last entry's ``ts_ns``. Zero if the log is empty."""

        if not self._entries:
            return 0
        return self._entries[-1].ts_ns

    @property
    def time_span_ns(self) -> int:
        """``end_ts_ns - start_ts_ns``. Zero if the log is empty."""

        return self.end_ts_ns - self.start_ts_ns

    # -- transformations -----------------------------------------------------

    def slice(self, start_ts_ns: int, end_ts_ns: int) -> EventReplayer:
        """Return a new :class:`EventReplayer` covering only entries
        with ``start_ts_ns <= ts_ns <= end_ts_ns``. Inclusive on both
        ends to mirror nautilus's ``backtest_start`` / ``backtest_end``
        semantics.
        """

        if not isinstance(start_ts_ns, int):
            raise TypeError(
                f"EventReplayer.slice start_ts_ns must be int, got {type(start_ts_ns).__name__}"
            )
        if not isinstance(end_ts_ns, int):
            raise TypeError(
                f"EventReplayer.slice end_ts_ns must be int, got {type(end_ts_ns).__name__}"
            )
        if start_ts_ns < 0:
            raise ValueError(
                f"EventReplayer.slice start_ts_ns must be non-negative, got {start_ts_ns!r}"
            )
        if end_ts_ns < start_ts_ns:
            raise ValueError(
                "EventReplayer.slice end_ts_ns must be >= start_ts_ns, "
                f"got start={start_ts_ns!r}, end={end_ts_ns!r}"
            )
        kept = tuple(entry for entry in self._entries if start_ts_ns <= entry.ts_ns <= end_ts_ns)
        return EventReplayer(kept)

    # -- replay --------------------------------------------------------------

    def replay(
        self,
        scenario_id: str,
        seed: int = 0,
        handler: EventHandler | None = None,
        clock: SimulatedClock | None = None,
    ) -> tuple[ReplayResult, tuple[ScheduledEvent, ...]]:
        """Replay the entire log through a fresh
        :class:`DeterministicEventLoop`.

        Returns a ``(ReplayResult, dispatched_events)`` pair. The
        dispatched-events tuple is the loop's verbatim output —
        callers that only need the summary can discard it.

        The clock is *not* shared between calls. Each replay builds a
        fresh :class:`TestClock` (starting at the log's
        ``start_ts_ns``) unless the caller passes one in. INV-15
        guarantees byte-identity across replays of the same input
        log + same handler.
        """

        if not isinstance(scenario_id, str):
            raise TypeError(
                f"EventReplayer.replay scenario_id must be str, got {type(scenario_id).__name__}"
            )
        if not scenario_id:
            raise ValueError("EventReplayer.replay scenario_id must be non-empty")
        if not isinstance(seed, int):
            raise TypeError(f"EventReplayer.replay seed must be int, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"EventReplayer.replay seed must be non-negative, got {seed!r}")

        if clock is None:
            clock = TestClock(initial_ts_ns=self.start_ts_ns)
        elif not isinstance(clock, SimulatedClock):
            raise TypeError(
                "EventReplayer.replay clock must satisfy SimulatedClock, "
                f"got {type(clock).__name__}"
            )

        loop = DeterministicEventLoop(clock=clock)
        for entry in self._entries:
            loop.schedule_at(ts_ns=entry.ts_ns, payload=entry.payload)
        dispatched = loop.run(handler=handler)

        events_dispatched = len(dispatched)
        start_ts_ns = self.start_ts_ns
        end_ts_ns = self.end_ts_ns
        final_clock_ns = clock.now_ns()
        # Empty-log edge: clock never advanced, but ReplayResult
        # validators require final_clock_ns >= end_ts_ns. start/end
        # are both 0 for empty logs so this holds.

        result = ReplayResult(
            scenario_id=scenario_id,
            seed=seed,
            events_dispatched=events_dispatched,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            final_clock_ns=final_clock_ns,
        )
        return result, dispatched

    def replay_until(
        self,
        scenario_id: str,
        until_ts_ns: int,
        seed: int = 0,
        handler: EventHandler | None = None,
        clock: SimulatedClock | None = None,
    ) -> tuple[ReplayResult, tuple[ScheduledEvent, ...]]:
        """Replay only entries with ``ts_ns <= until_ts_ns`` through a
        fresh loop. Equivalent to ``self.slice(0, until_ts_ns).replay(...)``
        but does not allocate the intermediate sliced replayer.
        """

        if not isinstance(until_ts_ns, int):
            raise TypeError(
                "EventReplayer.replay_until until_ts_ns must be int, "
                f"got {type(until_ts_ns).__name__}"
            )
        if until_ts_ns < 0:
            raise ValueError(
                f"EventReplayer.replay_until until_ts_ns must be non-negative, got {until_ts_ns!r}"
            )
        return self.slice(0, until_ts_ns).replay(
            scenario_id=scenario_id,
            seed=seed,
            handler=handler,
            clock=clock,
        )
