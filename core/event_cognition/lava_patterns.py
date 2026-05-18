# ADAPTED FROM: lava-nc/lava (BSD-3-Clause)
# Pattern-only distillation — no hardware dependency, no lava-nc
# import. See ``docs/lava_event_driven_patterns.md`` for the
# mapping rationale and the *exact* Lava primitives this module
# mirrors.

"""Compute-on-spike asyncio primitives — Lava-style event cognition.

This module distils the *behavioural* contract of Intel's Lava
event-driven neuromorphic framework
(`lava-nc/lava <https://github.com/lava-nc/lava>`_) into a pure
asyncio + stdlib pattern. Three discrete abstractions:

* :class:`LavaInPort` / :class:`LavaOutPort` — typed bounded async
  ports that mirror Lava's ``InPort`` / ``OutPort`` blocking
  send/recv semantics.
* :class:`LavaProcess` — an actor that *only computes when a spike
  arrives on an input port*. No polling, no clock reads. The
  contract is purely event-driven: ``on_spike`` is invoked exactly
  once per delivered :class:`LavaSpike` in deterministic per-port
  arrival order.
* :class:`LavaGraph` + :class:`LavaScheduler` — declarative wiring
  layer that connects ``out_port → in_port`` edges and drives the
  asyncio loop until the producer side completes.

Why a pattern, not a port
-------------------------

Lava ships a hardware-targeted runtime that talks to Loihi 2 and
fans out to the underlying neuromorphic substrate. None of that is
useful inside DIX, where the runtime is plain Python + asyncio.
What *is* useful is the architectural discipline Lava enforces:

* **Sensory nodes do not poll.** A sensor only produces work when an
  upstream event arrives. Idle = literally idle.
* **Inputs are typed and bounded.** Backpressure is explicit; a
  slow consumer blocks its producers, never silently buffers
  unbounded.
* **Computation is functional.** Each ``on_spike`` call takes one
  input spike (or one synchronously-batched group) and emits zero
  or more output spikes — no shared mutable state between actors,
  no clock reads.

This module ports those three properties into a tier-safe (no
``time``, no ``random``, no global state, no cross-engine imports)
form that downstream sensors can subclass directly.

Tier classification
-------------------

``OFFLINE_ONLY | ADVISORY (PATTERN_ONLY)`` — the abstractions are
free for any tier to subclass, but the module itself is forbidden
from importing vendor packages, runtime engines, or the registry.
:func:`tools.authority_lint` enforces the import discipline.

Determinism
-----------

* :class:`LavaInPort` is FIFO per producer.
* :class:`LavaProcess.run` consumes one spike at a time from a
  user-supplied port-ordering tuple. Given the same input
  sequence per port, every actor produces byte-identical output
  spikes across runs (INV-15).
* :class:`LavaScheduler` schedules concurrent tasks via
  :func:`asyncio.gather`, but each :class:`LavaProcess` has its
  own coroutine and never reads from a shared mutable list — so
  ordering inside a process is deterministic even if the scheduler
  ordering between processes is not.

Example
-------

::

    from core.event_cognition.lava_patterns import (
        LavaGraph,
        LavaInPort,
        LavaOutPort,
        LavaScheduler,
        LavaSpike,
        PassthroughProcess,
    )

    async def example() -> list[LavaSpike[float]]:
        source_out = LavaOutPort[float](port_id="source.out")
        midway_in = LavaInPort[float](port_id="midway.in", capacity=4)
        midway_out = LavaOutPort[float](port_id="midway.out")
        sink_in = LavaInPort[float](port_id="sink.in", capacity=4)

        midway = PassthroughProcess[float, float](
            process_id="midway",
            in_ports=(midway_in,),
            out_ports=(midway_out,),
        )
        sink = PassthroughProcess[float, float](
            process_id="sink",
            in_ports=(sink_in,),
            out_ports=(),
        )

        graph = LavaGraph(processes=(midway, sink))
        graph.connect(source_out, midway_in)
        graph.connect(midway_out, sink_in)
        graph.validate()

        scheduler = LavaScheduler(graph=graph)
        await source_out.send(LavaSpike(ts_ns=1, payload=1.5))
        await source_out.send(LavaSpike(ts_ns=2, payload=2.5))
        await source_out.close()
        return await scheduler.run()
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# =========================================================== Version

LAVA_PATTERNS_VERSION: str = "lava-patterns/v1"
"""Module-level version tag for ledger / audit projections."""

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""No external dependencies — pure stdlib pattern (PATTERN_ONLY)."""


# =========================================================== Errors


class LavaPortError(ValueError):
    """A :class:`LavaInPort` / :class:`LavaOutPort` invariant was
    violated.

    Examples: capacity ≤ 0, duplicate connection, send-after-close.
    """


class LavaCompositionError(ValueError):
    """A :class:`LavaGraph` wiring invariant was violated.

    Examples: cycle, type mismatch, dangling input, double-edge.
    """


# =========================================================== Spike

PayloadT = TypeVar("PayloadT")
"""Type variable for a :class:`LavaSpike` payload."""


@dataclass(frozen=True, slots=True)
class LavaSpike(Generic[PayloadT]):
    """Typed event carried over a port.

    Spikes are immutable, monotonic-timestamped messages. The
    only required fields are :attr:`ts_ns` (caller-supplied — no
    clock reads inside the module, per CONST-04) and
    :attr:`payload`. Optional :attr:`source_port_id` lets a
    downstream actor identify the producing edge.
    """

    ts_ns: int
    payload: PayloadT
    source_port_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise LavaPortError("LavaSpike.ts_ns must be an int")
        if self.ts_ns < 0:
            raise LavaPortError("LavaSpike.ts_ns must be ≥ 0")
        if not isinstance(self.source_port_id, str):
            raise LavaPortError("LavaSpike.source_port_id must be a str")


# =========================================================== Ports


_PORT_ID_MAX_LEN = 128
_PORT_CAPACITY_MAX = 4_096


def _validate_port_id(port_id: str) -> None:
    if not isinstance(port_id, str):
        raise LavaPortError("port_id must be a str")
    if not port_id:
        raise LavaPortError("port_id must be non-empty")
    if len(port_id) > _PORT_ID_MAX_LEN:
        raise LavaPortError(f"port_id length must be ≤ {_PORT_ID_MAX_LEN}")


class LavaInPort(Generic[PayloadT]):
    """Typed bounded async receive port (mirror of Lava ``InPort``).

    Backed by an :class:`asyncio.Queue` with a hard capacity ceiling
    so a slow consumer applies real backpressure. Connection is a
    one-time bind: a port may be connected to at most one upstream
    :class:`LavaOutPort` (enforced by :meth:`bind`).

    The port is **not** thread-safe — it relies on a single asyncio
    loop, like every other primitive in this module.
    """

    __slots__ = (
        "port_id",
        "capacity",
        "_queue",
        "_bound_out_port_id",
        "_closed",
    )

    def __init__(self, *, port_id: str, capacity: int = 32) -> None:
        _validate_port_id(port_id)
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise LavaPortError("capacity must be an int")
        if capacity < 1:
            raise LavaPortError("capacity must be ≥ 1")
        if capacity > _PORT_CAPACITY_MAX:
            raise LavaPortError(f"capacity must be ≤ {_PORT_CAPACITY_MAX}")
        self.port_id: str = port_id
        self.capacity: int = capacity
        self._queue: asyncio.Queue[LavaSpike[PayloadT] | None] = asyncio.Queue(maxsize=capacity)
        self._bound_out_port_id: str = ""
        self._closed: bool = False

    @property
    def is_bound(self) -> bool:
        """Whether an upstream :class:`LavaOutPort` is connected."""

        return bool(self._bound_out_port_id)

    @property
    def bound_out_port_id(self) -> str:
        """Identifier of the upstream port (or ``""`` when unbound)."""

        return self._bound_out_port_id

    @property
    def is_closed(self) -> bool:
        """Whether the upstream side has signalled end-of-stream."""

        return self._closed

    def bind(self, out_port_id: str) -> None:
        """Bind this port to a producing :class:`LavaOutPort`.

        Raises:
            LavaPortError: When the port is already bound.
        """

        _validate_port_id(out_port_id)
        if self._bound_out_port_id:
            raise LavaPortError(
                f"in_port {self.port_id!r} already bound to {self._bound_out_port_id!r}"
            )
        self._bound_out_port_id = out_port_id

    async def push(self, spike: LavaSpike[PayloadT]) -> None:
        """Enqueue a spike for downstream consumption.

        Called by the producing :class:`LavaOutPort`; consumers
        should not call this directly.
        """

        if not isinstance(spike, LavaSpike):
            raise LavaPortError("LavaInPort.push expects a LavaSpike instance")
        if self._closed:
            raise LavaPortError(f"in_port {self.port_id!r} is closed")
        await self._queue.put(spike)

    async def receive(self) -> LavaSpike[PayloadT] | None:
        """Await a single spike. Returns ``None`` on end-of-stream."""

        return await self._queue.get()

    async def close(self) -> None:
        """Signal end-of-stream — receive() will eventually drain to None."""

        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"LavaInPort(port_id={self.port_id!r}, capacity={self.capacity})"


class LavaOutPort(Generic[PayloadT]):
    """Typed async send port (mirror of Lava ``OutPort``).

    An out-port multicasts to every connected in-port: each
    downstream port receives an independent enqueue. The port owns
    no state beyond its connected sinks; it is purely a fan-out
    seam.
    """

    __slots__ = ("port_id", "_sinks", "_closed")

    def __init__(self, *, port_id: str) -> None:
        _validate_port_id(port_id)
        self.port_id: str = port_id
        self._sinks: list[LavaInPort[PayloadT]] = []
        self._closed: bool = False

    @property
    def sink_ids(self) -> tuple[str, ...]:
        """Identifiers of every connected :class:`LavaInPort`."""

        return tuple(s.port_id for s in self._sinks)

    @property
    def fanout(self) -> int:
        """Number of connected downstream ports."""

        return len(self._sinks)

    @property
    def is_closed(self) -> bool:
        """Whether end-of-stream has been signalled."""

        return self._closed

    def connect(self, in_port: LavaInPort[PayloadT]) -> None:
        """Connect a downstream :class:`LavaInPort`.

        Raises:
            LavaPortError: When the same in-port is connected twice
                or when the in-port is already bound elsewhere.
        """

        if not isinstance(in_port, LavaInPort):
            raise LavaPortError("connect expects a LavaInPort instance")
        if in_port in self._sinks:
            raise LavaPortError(
                f"in_port {in_port.port_id!r} already connected to out_port {self.port_id!r}"
            )
        in_port.bind(self.port_id)
        self._sinks.append(in_port)

    async def send(self, spike: LavaSpike[PayloadT]) -> None:
        """Broadcast a spike to every connected in-port.

        Spikes are re-tagged with this out-port's id so a downstream
        actor can identify the producing edge.
        """

        if not isinstance(spike, LavaSpike):
            raise LavaPortError("LavaOutPort.send expects a LavaSpike instance")
        if self._closed:
            raise LavaPortError(f"out_port {self.port_id!r} is closed")
        retagged = LavaSpike[PayloadT](
            ts_ns=spike.ts_ns,
            payload=spike.payload,
            source_port_id=self.port_id,
        )
        for sink in self._sinks:
            await sink.push(retagged)

    async def close(self) -> None:
        """Signal end-of-stream on every connected in-port."""

        if self._closed:
            return
        self._closed = True
        for sink in self._sinks:
            await sink.close()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"LavaOutPort(port_id={self.port_id!r}, fanout={self.fanout})"


# =========================================================== Process


_PROCESS_ID_MAX_LEN = 128


@runtime_checkable
class _LavaProcessProtocol(Protocol):
    """Structural contract for any actor a :class:`LavaScheduler`
    can drive."""

    process_id: str

    async def run(self) -> None: ...


class LavaProcess(Generic[PayloadT]):
    """Compute-on-spike asyncio actor.

    Subclass and override :meth:`on_spike` to implement the actor
    behaviour. The runtime contract is:

    1. :meth:`run` awaits on each declared in-port *in declaration
       order*. There is **no polling** — the actor blocks until a
       spike arrives.
    2. On each spike, :meth:`on_spike` is invoked. The base
       implementation does nothing; subclasses can return ``None``
       (no output) or emit spikes by calling
       :meth:`LavaOutPort.send` on a registered out-port.
    3. When all in-ports drain (each receives ``None``), the actor
       gracefully closes every out-port and returns.

    The actor is **stateless from the framework's perspective**. Any
    state a subclass needs (e.g. running statistics) lives on
    ``self`` and is the subclass's responsibility — but it must
    remain isolated to a single actor instance to preserve INV-15
    determinism.
    """

    __slots__ = (
        "process_id",
        "in_ports",
        "out_ports",
        "_consumed_count",
        "_emitted_count",
        "_spike_listeners",
    )

    def __init__(
        self,
        *,
        process_id: str,
        in_ports: tuple[LavaInPort[Any], ...] = (),
        out_ports: tuple[LavaOutPort[Any], ...] = (),
    ) -> None:
        if not isinstance(process_id, str):
            raise LavaCompositionError("process_id must be a str")
        if not process_id:
            raise LavaCompositionError("process_id must be non-empty")
        if len(process_id) > _PROCESS_ID_MAX_LEN:
            raise LavaCompositionError(f"process_id length must be ≤ {_PROCESS_ID_MAX_LEN}")
        if not isinstance(in_ports, tuple):
            raise LavaCompositionError("in_ports must be a tuple")
        if not isinstance(out_ports, tuple):
            raise LavaCompositionError("out_ports must be a tuple")
        for port in in_ports:
            if not isinstance(port, LavaInPort):
                raise LavaCompositionError("in_ports entries must be LavaInPort instances")
        for port in out_ports:
            if not isinstance(port, LavaOutPort):
                raise LavaCompositionError("out_ports entries must be LavaOutPort instances")
        # Disallow duplicate port ids inside a single process; the
        # downstream graph cycle check assumes unique identifiers.
        seen_in: set[str] = set()
        for port in in_ports:
            if port.port_id in seen_in:
                raise LavaCompositionError(f"duplicate in_port id {port.port_id!r}")
            seen_in.add(port.port_id)
        seen_out: set[str] = set()
        for port in out_ports:
            if port.port_id in seen_out:
                raise LavaCompositionError(f"duplicate out_port id {port.port_id!r}")
            seen_out.add(port.port_id)
        self.process_id: str = process_id
        self.in_ports: tuple[LavaInPort[Any], ...] = in_ports
        self.out_ports: tuple[LavaOutPort[Any], ...] = out_ports
        self._consumed_count: int = 0
        self._emitted_count: int = 0
        self._spike_listeners: list[Any] = []  # callables (in_port, spike) -> Awaitable[None]

    @property
    def consumed_count(self) -> int:
        """Number of spikes ``on_spike`` has been invoked with."""

        return self._consumed_count

    @property
    def emitted_count(self) -> int:
        """Number of spikes :meth:`emit` has dispatched."""

        return self._emitted_count

    async def on_spike(
        self,
        in_port: LavaInPort[Any],
        spike: LavaSpike[Any],
    ) -> None:
        """Override to react to a single input spike.

        The base implementation is a no-op — useful for null sinks
        in tests and graph validation.
        """

        return None

    async def emit(
        self,
        out_port: LavaOutPort[Any],
        spike: LavaSpike[Any],
    ) -> None:
        """Helper: dispatch ``spike`` on a registered ``out_port``.

        Raises:
            LavaCompositionError: When ``out_port`` is not registered
                on this process.
        """

        if out_port not in self.out_ports:
            raise LavaCompositionError(
                f"out_port {out_port.port_id!r} not registered on process {self.process_id!r}"
            )
        await out_port.send(spike)
        self._emitted_count += 1

    def add_spike_listener(
        self,
        listener: Any,
    ) -> None:
        """Register a coroutine ``(in_port, spike) -> Awaitable[None]``
        invoked after :meth:`on_spike` for every delivered spike.

        Used by :class:`LavaScheduler` to collect sink output
        without monkey-patching :meth:`on_spike`. Subclasses may
        also use this for observability.
        """

        if not callable(listener):
            raise LavaCompositionError("spike listener must be callable")
        self._spike_listeners.append(listener)

    def remove_spike_listener(
        self,
        listener: Any,
    ) -> None:
        """Remove a previously-registered spike listener."""

        try:
            self._spike_listeners.remove(listener)
        except ValueError as exc:
            raise LavaCompositionError("spike listener not registered") from exc

    async def _drive_port(self, in_port: LavaInPort[Any]) -> None:
        while True:
            spike = await in_port.receive()
            if spike is None:
                return
            self._consumed_count += 1
            await self.on_spike(in_port, spike)
            for listener in self._spike_listeners:
                await listener(in_port, spike)

    async def run(self) -> None:
        """Drive every in-port until it drains, then close out-ports.

        When the process has no in-ports it returns immediately.
        With one in-port, drives it sequentially. With multiple,
        every in-port is driven concurrently — but each in-port is
        FIFO within itself.
        """

        if not self.in_ports:
            for out_port in self.out_ports:
                await out_port.close()
            return
        if len(self.in_ports) == 1:
            try:
                await self._drive_port(self.in_ports[0])
            finally:
                for out_port in self.out_ports:
                    await out_port.close()
            return
        try:
            await asyncio.gather(*(self._drive_port(p) for p in self.in_ports))
        finally:
            for out_port in self.out_ports:
                await out_port.close()


# =========================================================== Passthrough


PassPayloadT = TypeVar("PassPayloadT")


class PassthroughProcess(LavaProcess[PassPayloadT], Generic[PassPayloadT]):
    """Reference :class:`LavaProcess` that forwards every spike
    on its first out-port unchanged.

    Used in tests and as a canonical example of the
    compute-on-spike pattern.
    """

    __slots__ = ()

    async def on_spike(
        self,
        in_port: LavaInPort[Any],
        spike: LavaSpike[Any],
    ) -> None:
        if not self.out_ports:
            return
        await self.emit(self.out_ports[0], spike)


# =========================================================== Graph


@dataclass(frozen=True, slots=True)
class LavaEdge:
    """An out-port → in-port edge inside a :class:`LavaGraph`."""

    out_port_id: str
    in_port_id: str
    producer_process_id: str
    consumer_process_id: str


@dataclass(slots=True)
class LavaGraph:
    """Declarative wiring of :class:`LavaProcess` instances.

    The graph holds processes and the edges between their out- and
    in-ports. Edges are added via :meth:`connect`; the structural
    invariants are checked in :meth:`validate`:

    * Every in-port owned by a registered process is connected to
      *exactly one* out-port (also owned by a registered process).
    * No cycles. ``run`` requires a DAG — a cycle deadlocks the
      asyncio scheduler.
    """

    processes: tuple[LavaProcess[Any], ...]
    edges: tuple[LavaEdge, ...] = field(default_factory=tuple)
    _process_by_in_port: dict[str, LavaProcess[Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _process_by_out_port: dict[str, LavaProcess[Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.processes, tuple):
            raise LavaCompositionError("processes must be a tuple")
        seen_processes: set[str] = set()
        for process in self.processes:
            if not isinstance(process, LavaProcess):
                raise LavaCompositionError("processes entries must be LavaProcess instances")
            if process.process_id in seen_processes:
                raise LavaCompositionError(f"duplicate process_id {process.process_id!r}")
            seen_processes.add(process.process_id)
            for in_port in process.in_ports:
                if in_port.port_id in self._process_by_in_port:
                    raise LavaCompositionError(
                        f"in_port {in_port.port_id!r} owned by multiple processes"
                    )
                self._process_by_in_port[in_port.port_id] = process
            for out_port in process.out_ports:
                if out_port.port_id in self._process_by_out_port:
                    raise LavaCompositionError(
                        f"out_port {out_port.port_id!r} owned by multiple processes"
                    )
                self._process_by_out_port[out_port.port_id] = process

    def process_count(self) -> int:
        """Number of registered processes."""

        return len(self.processes)

    def edge_count(self) -> int:
        """Number of declared edges."""

        return len(self.edges)

    def connect(
        self,
        out_port: LavaOutPort[Any],
        in_port: LavaInPort[Any],
    ) -> None:
        """Wire ``out_port → in_port`` and record the edge.

        Performs the actual port binding; cycle / dangling-input
        checks are deferred to :meth:`validate`.
        """

        if not isinstance(out_port, LavaOutPort):
            raise LavaCompositionError("connect: out_port must be a LavaOutPort")
        if not isinstance(in_port, LavaInPort):
            raise LavaCompositionError("connect: in_port must be a LavaInPort")
        producer = self._process_by_out_port.get(out_port.port_id)
        consumer = self._process_by_in_port.get(in_port.port_id)
        if producer is None and out_port.port_id not in _external_out_ports(self.edges):
            # External producer (e.g. driving the graph from
            # outside) — still allowed, but recorded with empty
            # producer_process_id.
            pass
        if consumer is None:
            raise LavaCompositionError(
                f"in_port {in_port.port_id!r} not owned by any registered process"
            )
        out_port.connect(in_port)
        self.edges = self.edges + (
            LavaEdge(
                out_port_id=out_port.port_id,
                in_port_id=in_port.port_id,
                producer_process_id=(producer.process_id if producer else ""),
                consumer_process_id=consumer.process_id,
            ),
        )

    def validate(self) -> None:
        """Verify wiring invariants. Raises on first violation."""

        # Every in-port owned by a process must be bound.
        for in_port_id, process in self._process_by_in_port.items():
            in_port = next(p for p in process.in_ports if p.port_id == in_port_id)
            if not in_port.is_bound:
                raise LavaCompositionError(
                    f"in_port {in_port_id!r} on process {process.process_id!r} is dangling"
                )
        # No process-to-itself or larger cycles. Build a directed
        # graph keyed on process_id.
        adjacency: dict[str, set[str]] = {p.process_id: set() for p in self.processes}
        for edge in self.edges:
            if not edge.producer_process_id:
                continue
            adjacency[edge.producer_process_id].add(edge.consumer_process_id)
        if _has_cycle(adjacency):
            raise LavaCompositionError("LavaGraph contains a cycle — required to be a DAG")


def _external_out_ports(edges: tuple[LavaEdge, ...]) -> set[str]:
    """Return the set of out-port ids whose producer is external."""

    return {edge.out_port_id for edge in edges if not edge.producer_process_id}


def _has_cycle(adjacency: Mapping[str, set[str]]) -> bool:
    """Iterative DFS cycle detection over a process adjacency map."""

    WHITE, GRAY, BLACK = 0, 1, 2
    state: dict[str, int] = {node: WHITE for node in adjacency}
    for root in adjacency:
        if state[root] != WHITE:
            continue
        stack: list[tuple[str, list[str]]] = [(root, list(adjacency[root]))]
        state[root] = GRAY
        while stack:
            node, neighbours = stack[-1]
            if not neighbours:
                state[node] = BLACK
                stack.pop()
                continue
            nxt = neighbours.pop()
            if state.get(nxt, BLACK) == GRAY:
                return True
            if state.get(nxt, BLACK) == WHITE:
                state[nxt] = GRAY
                stack.append((nxt, list(adjacency.get(nxt, set()))))
    return False


# =========================================================== Scheduler


_DEFAULT_SCHEDULER_TIMEOUT_S = 5.0


@dataclass(slots=True)
class LavaScheduler:
    """Asyncio driver for a validated :class:`LavaGraph`.

    The scheduler launches a coroutine per process and gathers them
    until each one's in-ports drain. It does **not** generate any
    spikes itself — that's the caller's job (either from outside
    the graph or from a process with no in-ports).
    """

    graph: LavaGraph

    def __post_init__(self) -> None:
        if not isinstance(self.graph, LavaGraph):
            raise LavaCompositionError("LavaScheduler.graph must be a LavaGraph")

    async def run(self, *, timeout_s: float = _DEFAULT_SCHEDULER_TIMEOUT_S) -> list[LavaSpike[Any]]:
        """Drive every process to completion.

        Returns the *collected sink spikes* — every spike delivered
        to an in-port whose owning process has no out-ports. Useful
        for test assertions; production callers usually ignore the
        return value and read state from registered processes.
        """

        if not isinstance(timeout_s, int | float) or isinstance(timeout_s, bool):
            raise LavaCompositionError("timeout_s must be a number")
        if math.isnan(timeout_s) or math.isinf(timeout_s):
            raise LavaCompositionError("timeout_s must be finite")
        if timeout_s <= 0:
            raise LavaCompositionError("timeout_s must be positive")
        sink_collector: list[LavaSpike[Any]] = []
        sink_processes = tuple(p for p in self.graph.processes if not p.out_ports)

        async def _collect(_in_port: LavaInPort[Any], spike: LavaSpike[Any]) -> None:
            sink_collector.append(spike)

        for process in sink_processes:
            process.add_spike_listener(_collect)
        try:
            await asyncio.wait_for(
                asyncio.gather(*(p.run() for p in self.graph.processes)),
                timeout=timeout_s,
            )
        finally:
            for process in sink_processes:
                process.remove_spike_listener(_collect)
        return sink_collector


# =========================================================== __all__

__all__ = [
    "LAVA_PATTERNS_VERSION",
    "LavaCompositionError",
    "LavaEdge",
    "LavaGraph",
    "LavaInPort",
    "LavaOutPort",
    "LavaPortError",
    "LavaProcess",
    "LavaScheduler",
    "LavaSpike",
    "NEW_PIP_DEPENDENCIES",
    "PassthroughProcess",
]
