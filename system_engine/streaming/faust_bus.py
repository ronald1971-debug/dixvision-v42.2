"""C-02 faust-streaming — CEP-Capable Event Stream.

# ADAPTED FROM: robinhood/faust — ``faust/app/base.py`` (``App``
# top-level container), ``faust/models/record.py`` (``Record``
# dataclass-shaped schema), ``faust/agents/agent.py`` (``@app.agent``
# coroutine-shape consumer pattern), ``faust/tables/wrappers.py``
# (windowed-table ``Table.tumbling`` shape), ``faust/types/windows.py``
# (``TumblingWindow`` event-time emission).
#
# Tier: OFFLINE_ONLY — the faust pipeline runs as a SEPARATE process
# feeding DIX via a :mod:`multiprocessing` queue. The pipeline never
# imports any RUNTIME tier (no ``intelligence_engine``,
# ``execution_engine``, ``governance_engine`` imports). The pipeline
# emits ``FaustResult`` advisory records that *carry* already-constructed
# DIX events (``SignalEvent`` / ``HazardEvent`` / ``ExecutionEvent`` /
# ``SystemEvent``); the bus NEVER constructs new typed events itself.
# This preserves B27 / B28 / INV-71 authority symmetry — only the
# engine that produced an event may construct it, never the transport.

This module is the canonical alternative to :mod:`system_engine.streaming.event_fabric`
(C-01 bytewax). The two transports expose the same contract surface so
production callers can swap between them based on deployment needs:

* :mod:`event_fabric` (bytewax) — dataflow-style operator chain.
* :mod:`faust_bus` (faust-streaming) — agent/topic CEP-style routing
  with first-class event-time tumbling-window tables.

Faust's selling point over bytewax is the CEP ``WindowedTable``
abstraction — keyed state with declarative event-time windowing
where downstream agents read live aggregates as ordinary ``table[key]``
lookups during window iteration. This module emulates that surface in
deterministic pure Python; the lazy seam :func:`faust_app_factory`
gates activation of the real :mod:`faust` PyPI package behind a future
research-acceptance PR.

Determinism (INV-15):

* No top-level imports of :mod:`time` / :mod:`datetime` / :mod:`random`
  / :mod:`asyncio` / :mod:`os` / :mod:`faust` / :mod:`numpy` /
  :mod:`torch` / :mod:`polars`.
* All windowing is event-time over a caller-supplied ``ts_fn``
  extractor. No wall-clock reads.
* Topic fan-out order: agents fire in registration order. Multiple
  agents on the same topic see events in source insertion order.
* Windowed-table emission: ``(bucket_idx ascending, key ascending)``
  on flush. Two byte-identical input streams produce two byte-identical
  output streams.
* Frozen, slotted dataclasses everywhere. The :class:`App` itself is a
  value object — registering an agent / topic / table returns a *new*
  ``App`` (immutable builder, mirroring :class:`Dataflow` in C-01).
* BLAKE2b-16 ``app_digest`` over the topic / agent / table spec gives
  byte-identical replay equality.

Worker bridge:

* :func:`spawn_faust_worker` uses ``multiprocessing.get_context("spawn")``
  so the child process has an independent interpreter (no inherited
  module state). Callbacks passed to agents / tables must be top-level
  module-importable callables for cross-process use; lambdas and
  closures are fine for the in-process replay tests.
* The worker terminates cleanly on a :class:`FaustBusSentinel` on the
  inbound queue.

Authority discipline:

* B27 / B28 / INV-71: this module does **not** call ``PatchProposal(...)``,
  ``HazardEvent(...)``, ``SignalEvent(...)``, ``ExecutionEvent(...)``
  or ``SystemEvent(...)`` directly. AST tests pin the constraint.
* B1 isolation: no imports from ``intelligence_engine``,
  ``execution_engine``, ``governance_engine``, ``evolution_engine``.
  The bus is a leaf transport.

Outputs declared by canonical block C-02:

1. ``system_engine/streaming/faust_bus.py`` (this file)
2. ``tests/test_faust_bus.py``
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Generic, TypeVar

FAUST_BUS_VERSION: int = 1

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("faust-streaming",)
"""Declared so the canonical pin-set is complete.

The package itself is NEVER imported in this module — see the module
docstring for the rationale and :func:`faust_app_factory` for the lazy
seam where a future PR can wire it up after the research-acceptance
gate is documented.
"""


T = TypeVar("T")
A = TypeVar("A")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Record — Faust-style typed envelope.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Record:
    """Marker base for Faust-style typed envelopes.

    Concrete subclasses are frozen, slotted dataclasses defined by the
    caller. Mirrors ``faust.Record`` — a dataclass-shaped value object
    whose fields are the wire schema for a topic. The bus itself never
    constructs typed DIX events (B27 / B28 / INV-71); ``Record`` is a
    *user-level* schema base that callers may extend, not a DIX typed
    event constructor.

    # ADAPTED FROM: faust/models/record.py — ``Record``.
    """


# ---------------------------------------------------------------------------
# Topic — named typed channel.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Topic:
    """Named typed channel.

    Routes events from producers to all agents registered on this
    topic. ``schema`` is advisory — runtime-shape checking is left to
    the caller's agent body so the bus stays a pure transport.

    # ADAPTED FROM: faust/topics.py — ``Topic``.
    """

    name: str
    schema: str = ""


# ---------------------------------------------------------------------------
# Agent — coroutine-shape consumer (here: synchronous callable).
# ---------------------------------------------------------------------------

AgentFn = Callable[["AgentContext", Any], "Iterable[SendOp] | None"]


@dataclass(frozen=True, slots=True)
class Agent:
    """Bind a callable to a topic.

    The callable receives an :class:`AgentContext` (table handles +
    ``send`` helper) and the inbound event payload. It may either
    return ``None`` (pure side-effect on a table) or an iterable of
    :class:`SendOp` to fan more events onto downstream topics. In
    Faust's async surface this maps to::

        @app.agent(topic)
        async def my_agent(stream):
            async for event in stream:
                await downstream.send(value=...)

    The synchronous shape here keeps INV-15 byte-identical replay
    intact — async event-loop scheduling is non-deterministic and is
    forbidden at module-import time anyway.

    # ADAPTED FROM: faust/agents/agent.py — ``@app.agent``.
    """

    topic_name: str
    fn: AgentFn
    name: str


# ---------------------------------------------------------------------------
# Table — keyed state with optional event-time tumbling window.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TumblingWindowSpec:
    """Event-time tumbling-window spec for a :class:`Table`.

    ``window_ns`` is the bucket size in nanoseconds. ``ts_fn`` extracts
    the event-time nanoseconds from the inbound payload. Buckets are
    ``floor(ts_ns / window_ns)`` — closed-open on the lower edge.

    # ADAPTED FROM: faust/types/windows.py — ``TumblingWindow``.
    """

    window_ns: int
    ts_fn: Callable[[Any], int]


@dataclass(frozen=True, slots=True)
class Table(Generic[V]):
    """Named keyed-state container, optionally event-time windowed.

    Mirrors ``app.Table(name, default=int)`` and the
    ``.tumbling(size)`` builder. The table is a *spec* — runtime state
    lives on :class:`AppState` so the :class:`App` itself stays a pure
    value object (immutable builder mirroring C-01's :class:`Dataflow`).

    # ADAPTED FROM: faust/tables/table.py + faust/tables/wrappers.py.
    """

    name: str
    default: Callable[[], V]
    window: TumblingWindowSpec | None = None

    def tumbling(
        self,
        window_ns: int,
        ts_fn: Callable[[Any], int],
    ) -> Table[V]:
        if window_ns <= 0:
            raise ValueError("window_ns must be > 0")
        return Table(
            name=self.name,
            default=self.default,
            window=TumblingWindowSpec(window_ns=window_ns, ts_fn=ts_fn),
        )


# ---------------------------------------------------------------------------
# SendOp — fan-out instruction returned by an agent body.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SendOp:
    """Instruct the bus to forward ``payload`` onto ``topic_name``.

    Agents return an iterable of these to fan more events. The
    forwarded events join the inbound queue tail in the order yielded;
    bus dispatch order is therefore fully determined by source
    insertion order plus agent emission order (INV-15).
    """

    topic_name: str
    payload: Any


# ---------------------------------------------------------------------------
# AgentContext — per-call handle passed to agent bodies.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Runtime context handed to an agent on every call.

    ``tables`` is a name → mutable-state mapping owned by :class:`AppState`.
    Windowed tables expose a ``(bucket_idx, key) → value`` dict; flat
    tables expose a ``key → value`` dict.

    ``current_ts_ns`` is the event-time of the payload being processed
    when the table is windowed. ``None`` for flat tables / unwindowed
    contexts. Used by agent bodies to address the correct bucket.
    """

    tables: Mapping[str, dict[Any, Any]]
    current_ts_ns: int | None = None


# ---------------------------------------------------------------------------
# App — immutable builder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class App:
    """Faust-style application container.

    Tracks the registered topics / agents / tables. Builder helpers
    return *new* :class:`App` instances (the existing instance is
    never mutated), mirroring C-01's :class:`Dataflow`.

    # ADAPTED FROM: faust/app/base.py — ``App``.
    """

    id: str
    topics: tuple[Topic, ...] = ()
    agents: tuple[Agent, ...] = ()
    tables: tuple[Table[Any], ...] = ()

    def topic(self, name: str, schema: str = "") -> App:
        if any(t.name == name for t in self.topics):
            raise ValueError(f"topic already registered: {name}")
        return App(
            id=self.id,
            topics=self.topics + (Topic(name=name, schema=schema),),
            agents=self.agents,
            tables=self.tables,
        )

    def table(
        self,
        name: str,
        default: Callable[[], V],
        window: TumblingWindowSpec | None = None,
    ) -> App:
        if any(t.name == name for t in self.tables):
            raise ValueError(f"table already registered: {name}")
        return App(
            id=self.id,
            topics=self.topics,
            agents=self.agents,
            tables=self.tables + (Table(name=name, default=default, window=window),),
        )

    def agent(self, topic_name: str, fn: AgentFn, name: str = "") -> App:
        if not any(t.name == topic_name for t in self.topics):
            raise ValueError(f"topic not registered: {topic_name}")
        agent_name = name or f"agent_{len(self.agents)}"
        return App(
            id=self.id,
            topics=self.topics,
            agents=self.agents
            + (Agent(topic_name=topic_name, fn=fn, name=agent_name),),
            tables=self.tables,
        )

    def app_digest(self) -> str:
        """Stable 16-hex BLAKE2b digest over the app spec.

        Encodes the topic / agent / table *names* and shapes (window
        size for windowed tables) — i.e. the parts of the spec that
        are serialisable without inspecting closure cells. This is
        sufficient for INV-15 replay equality testing because two
        apps constructed from the same code path produce identical
        topic / agent / table tuples by construction.
        """

        spec: list[Mapping[str, object]] = [{"id": self.id}]
        for t in self.topics:
            spec.append({"kind": "topic", "name": t.name, "schema": t.schema})
        for ag in self.agents:
            spec.append(
                {"kind": "agent", "name": ag.name, "topic": ag.topic_name}
            )
        for tbl in self.tables:
            entry: dict[str, object] = {"kind": "table", "name": tbl.name}
            if tbl.window is not None:
                entry["window_ns"] = tbl.window.window_ns
            spec.append(entry)
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# FaustResult — advisory output record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaustResult:
    """One emitted output from the bus.

    The bus is purely a transport — ``payload`` carries already-
    constructed DIX events (``SignalEvent`` / ``HazardEvent`` /
    ``ExecutionEvent`` / ``SystemEvent``) or downstream aggregates.
    The bus never constructs typed events itself; that authority
    stays with the producing engine (B27 / B28 / INV-71).

    ``kind`` is ``"event"`` for raw agent-fanout outputs and
    ``"window"`` for flushed tumbling-window aggregates.
    """

    seq: int
    kind: str
    topic_name: str
    payload: object
    key: str = ""
    bucket_idx: int = -1
    table_name: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InboundEvent — what callers feed into the bus.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InboundEvent:
    """One inbound event for :func:`run_app`.

    ``topic_name`` selects which agents fire. ``payload`` is the
    already-constructed DIX event (or any value object the caller's
    agent body knows how to interpret).
    """

    topic_name: str
    payload: Any


# ---------------------------------------------------------------------------
# run_app — in-process execution.
# ---------------------------------------------------------------------------


def run_app(
    app: App,
    events: Iterable[InboundEvent],
) -> tuple[FaustResult, ...]:
    """Execute ``app`` over ``events`` and return ``FaustResult`` tuple.

    Pure function — given identical inputs, returns byte-identical
    outputs (INV-15). No clock reads, no random, no I/O. Emission
    order is fully determined by:

    * Source events are processed in input order.
    * For each source event, agents on its topic fire in registration
      order. Each agent's ``SendOp`` outputs are queued in yield order
      and appended to the tail of the inbound queue (BFS fan-out).
    * Forwarded events are processed after all source events drain,
      again in arrival order.
    * On stream end, every windowed table flushes ``(bucket_idx, key)``
      pairs sorted by ``(bucket_idx ascending, key ascending)``.

    The caller's ``events`` iterable is materialised once; downstream
    state lives on :class:`AppState` (created here per-run so callers
    can re-run the same :class:`App` over different event streams).
    """

    queued: list[InboundEvent] = list(events)
    tables: dict[str, dict[Any, Any]] = {tbl.name: {} for tbl in app.tables}
    table_specs: dict[str, Table[Any]] = {tbl.name: tbl for tbl in app.tables}
    agents_by_topic: dict[str, list[Agent]] = {}
    for ag in app.agents:
        agents_by_topic.setdefault(ag.topic_name, []).append(ag)

    results: list[FaustResult] = []
    seq = 0
    cursor = 0
    while cursor < len(queued):
        ev = queued[cursor]
        cursor += 1
        agents = agents_by_topic.get(ev.topic_name, [])
        for ag in agents:
            ctx = _agent_context(tables, table_specs, ev.payload)
            out = ag.fn(ctx, ev.payload)
            if out is None:
                continue
            for send in out:
                if not isinstance(send, SendOp):
                    raise TypeError(
                        "agent must yield SendOp instances; "
                        f"got {type(send).__name__}"
                    )
                queued.append(
                    InboundEvent(topic_name=send.topic_name, payload=send.payload)
                )
                results.append(
                    FaustResult(
                        seq=seq,
                        kind="event",
                        topic_name=send.topic_name,
                        payload=send.payload,
                    )
                )
                seq += 1

    for tbl in app.tables:
        if tbl.window is None:
            for key in sorted(tables[tbl.name].keys(), key=_key_sort):
                results.append(
                    FaustResult(
                        seq=seq,
                        kind="table",
                        topic_name="",
                        payload=tables[tbl.name][key],
                        key=_key_str(key),
                        table_name=tbl.name,
                    )
                )
                seq += 1
            continue
        cells = tables[tbl.name]
        for cell in sorted(cells.keys(), key=_cell_sort):
            if not (isinstance(cell, tuple) and len(cell) == 2):
                raise TypeError(
                    "windowed table cells must be (bucket_idx, key) tuples; "
                    f"got {type(cell).__name__}"
                )
            bucket_idx, key = cell
            results.append(
                FaustResult(
                    seq=seq,
                    kind="window",
                    topic_name="",
                    payload=cells[cell],
                    key=_key_str(key),
                    bucket_idx=int(bucket_idx),
                    table_name=tbl.name,
                )
            )
            seq += 1

    return tuple(results)


def _agent_context(
    tables: Mapping[str, dict[Any, Any]],
    table_specs: Mapping[str, Table[Any]],
    payload: Any,
) -> AgentContext:
    """Build the per-call :class:`AgentContext`.

    ``current_ts_ns`` is derived by inspecting any windowed table on
    the app: if exactly one windowed table exists, its ``ts_fn`` is
    consulted. Otherwise ``current_ts_ns`` is ``None`` and the agent
    body must address windowed cells explicitly via ``(bucket, key)``.
    """

    windowed = [
        spec for spec in table_specs.values() if spec.window is not None
    ]
    current_ts_ns: int | None = None
    if len(windowed) == 1:
        spec = windowed[0]
        assert spec.window is not None
        try:
            ts = spec.window.ts_fn(payload)
        except (AttributeError, KeyError, TypeError):
            ts = None
        if isinstance(ts, int):
            current_ts_ns = ts
    return AgentContext(tables=tables, current_ts_ns=current_ts_ns)


def _key_sort(key: object) -> tuple[int, str]:
    """Stable key-sort across heterogeneous flat-table keys."""

    return (0, _key_str(key))


def _cell_sort(cell: object) -> tuple[int, str]:
    """Stable cell-sort: ``(bucket asc, key asc)``."""

    if not (isinstance(cell, tuple) and len(cell) == 2):
        raise TypeError(
            f"windowed cell must be (bucket_idx, key); got {type(cell).__name__}"
        )
    bucket, key = cell
    if not isinstance(bucket, int):
        raise TypeError(f"bucket_idx must be int; got {type(bucket).__name__}")
    return (bucket, _key_str(key))


def _key_str(key: object) -> str:
    """Coerce keys to ``str`` for stable lexical ordering."""

    if isinstance(key, str):
        return key
    return repr(key)


def bucket_index(ts_ns: int, window_ns: int) -> int:
    """``floor(ts_ns / window_ns)`` helper for caller agent bodies."""

    if window_ns <= 0:
        raise ValueError("window_ns must be > 0")
    return ts_ns // window_ns


# ---------------------------------------------------------------------------
# Cross-process worker bridge.
# ---------------------------------------------------------------------------


class FaustBusSentinel:
    """Sentinel placed on the inbound queue to terminate the worker.

    A distinct class (not a string) so payloads that happen to be
    strings can never accidentally trip the shutdown path.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return "FaustBusSentinel()"


def faust_worker_main(
    app: App,
    inbound: multiprocessing.Queue[object],
    outbound: multiprocessing.Queue[object],
    batch_size: int = 1,
) -> None:
    """Worker entrypoint.

    Pulls batches of :class:`InboundEvent` off ``inbound`` until a
    :class:`FaustBusSentinel` is received, then runs ``app`` over each
    batch and pushes the resulting :class:`FaustResult` tuple onto
    ``outbound`` (preserving the within-batch emission order).

    A second sentinel is then pushed onto ``outbound`` so the parent
    can detect a clean shutdown.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch: list[InboundEvent] = []
    while True:
        item = inbound.get()
        if isinstance(item, FaustBusSentinel):
            if batch:
                outbound.put(run_app(app, batch))
                batch.clear()
            outbound.put(FaustBusSentinel())
            return
        if not isinstance(item, InboundEvent):
            raise TypeError(
                "faust worker expects InboundEvent items; "
                f"got {type(item).__name__}"
            )
        batch.append(item)
        if len(batch) >= batch_size:
            outbound.put(run_app(app, batch))
            batch.clear()


def spawn_faust_worker(
    app: App,
    inbound: multiprocessing.Queue[object],
    outbound: multiprocessing.Queue[object],
    batch_size: int = 1,
) -> multiprocessing.Process:
    """Spawn a child process running :func:`faust_worker_main`.

    Uses the ``spawn`` start method so the child has an independent
    interpreter (no inherited module state from the parent). Matches
    faust's production deployment model (separate worker processes
    per partition) and avoids fork determinism hazards on Linux.

    Returns the started :class:`multiprocessing.Process` so the caller
    can ``join()`` it. The caller is responsible for placing a
    :class:`FaustBusSentinel` on ``inbound`` to terminate the worker.
    """

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=faust_worker_main,
        args=(app, inbound, outbound, batch_size),
        daemon=False,
    )
    proc.start()
    return proc


def drain_queue(
    outbound: multiprocessing.Queue[object],
    *,
    timeout_s: float | None = None,
) -> tuple[FaustResult, ...]:
    """Drain ``outbound`` until the worker's terminating sentinel.

    Concatenates each ``FaustResult`` tuple in arrival order and
    returns the flat tuple. The ``timeout_s`` parameter is a per-``get``
    deadline; pass ``None`` (default) for an unbounded blocking get.

    Raises ``TimeoutError`` if a ``get`` times out before the
    terminating sentinel arrives.
    """

    drained: list[FaustResult] = []
    while True:
        try:
            chunk = outbound.get(timeout=timeout_s)
        except Empty as exc:  # pragma: no cover - timeout path
            raise TimeoutError("faust bus outbound drain timed out") from exc
        if isinstance(chunk, FaustBusSentinel):
            return tuple(drained)
        if not isinstance(chunk, tuple):
            raise TypeError(
                "faust outbound expects FaustResult tuples; "
                f"got {type(chunk).__name__}"
            )
        drained.extend(chunk)


# ---------------------------------------------------------------------------
# faust lazy factory — research-acceptance gate.
# ---------------------------------------------------------------------------


def faust_app_factory(*args: object, **kwargs: object) -> object:
    """Lazy faust bridge — pinned ``NotImplementedError``.

    Wiring the real :mod:`faust` PyPI package is OUT OF SCOPE for
    C-02. The canonical block declares faust-streaming as a research
    source; activation is gated by a future PR that:

    1. Documents a shadow-equivalence harness comparing the in-process
       Python execution above against the real faust-streaming backend
       over a fixed event log.
    2. Demonstrates byte-identical aggregates between the two backends
       across at least one full replay cycle.
    3. Pins the faust operator surface to the subset adapted here
       (``App`` / ``Topic`` / ``Record`` / ``@agent`` / ``Table`` +
       ``TumblingWindow``).

    Until that PR lands, this factory raises so any accidental
    production import is loud rather than silent.
    """

    raise NotImplementedError(
        "faust_app_factory: gated until research-acceptance "
        "shadow-equivalence harness lands"
    )
