"""C-01 bytewax — Python-Native Stream Processing.

# ADAPTED FROM: bytewax/bytewax — ``bytewax/dataflow.py`` (Dataflow
# builder) and ``bytewax/operators/`` (``map``, ``filter``,
# ``window``, ``reduce``).
#
# Tier: OFFLINE_ONLY — the bytewax pipeline runs as a SEPARATE
# process feeding DIX via a :mod:`multiprocessing` queue. The
# pipeline never imports any RUNTIME tier (no
# ``intelligence_engine``, ``execution_engine``,
# ``governance_engine`` imports). The pipeline emits ``FabricResult``
# advisory records that *carry* already-constructed DIX events
# (``SignalEvent`` / ``HazardEvent`` / ``ExecutionEvent`` /
# ``SystemEvent``); the fabric NEVER constructs new typed events
# itself. This preserves B27 / B28 / INV-71 authority symmetry —
# only the engine that produced an event may construct it, never the
# transport.

The DIX in-process Python translation below is sufficient for offline
replay tests and for the cross-process queue bridge that production
callers wire up. The real :mod:`bytewax` PyPI package is *only*
imported inside :func:`bytewax_dataflow_factory` and only after the
research-acceptance gate documented there. Importing the package at
module load time would (a) break determinism — bytewax depends on
``time`` for windowing by default — and (b) violate the OFFLINE_ONLY
contract for downstream importers.

Determinism (INV-15):

* No top-level imports of :mod:`time` / :mod:`datetime` /
  :mod:`random` / :mod:`asyncio` / :mod:`os` / :mod:`bytewax`.
* All windowing is event-time over a caller-supplied ``ts_ns``
  extractor. No wall-clock reads.
* Operator output order: ``Map`` and ``Filter`` are insertion-order.
  ``TumblingWindow`` is ``(bucket_index ascending, key ascending)``.
  ``Reduce`` is per-key insertion-order keyed by first-seen.
* Frozen, slotted dataclasses everywhere. The dataflow itself is a
  ``Dataflow`` value object — appending an operator returns a *new*
  ``Dataflow`` (immutable builder).
* BLAKE2b-16 ``dataflow_digest`` over the operator chain spec gives
  byte-identical replay equality.

Worker bridge:

* :func:`spawn_fabric_worker` uses ``multiprocessing.get_context(
  "spawn")`` so the child process has an independent interpreter
  (no inherited module state). Callbacks passed to operators must
  be top-level module-importable callables for cross-process use;
  lambdas / closures are fine for the in-process replay tests.
* The worker terminates cleanly on a :class:`EventFabricSentinel`
  on the inbound queue.

Authority discipline:

* B27 / B28 / INV-71: this module does **not** call
  ``PatchProposal(...)``, ``HazardEvent(...)``, ``SignalEvent(...)``,
  ``ExecutionEvent(...)`` or ``SystemEvent(...)`` directly. AST tests
  pin the constraint.
* B1 isolation: no imports from ``intelligence_engine``,
  ``execution_engine``, ``governance_engine``, ``evolution_engine``.
  The fabric is a leaf transport.

Outputs declared by canonical block C-01:

1. ``system_engine/streaming/event_fabric.py`` (this file)
2. ``tests/test_event_fabric.py``
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Generic, TypeVar

EVENT_FABRIC_VERSION: int = 1

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("bytewax",)
"""Declared so the canonical pin-set is complete.

The package itself is NEVER imported in this module — see the
module docstring for the rationale and :func:`bytewax_dataflow_factory`
for the lazy seam where a future PR can wire it up after the
research-acceptance gate is documented.
"""


T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
A = TypeVar("A")


# ---------------------------------------------------------------------------
# Operator value objects.
# ---------------------------------------------------------------------------


class Operator:
    """Marker base for all operator value objects.

    Concrete subclasses are frozen, slotted dataclasses below.
    Using a marker class (rather than a ``Union`` type alias) lets
    :func:`run_dataflow` dispatch via ``isinstance`` without
    importing the concrete types eagerly.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class MapOp(Operator, Generic[T, U]):
    """Element-wise transform.

    # ADAPTED FROM: bytewax/operators/__init__.py — ``map``.
    """

    fn: Callable[[T], U]
    name: str = "map"


@dataclass(frozen=True, slots=True)
class FilterOp(Operator, Generic[T]):
    """Drop elements where ``predicate(x)`` is falsy.

    # ADAPTED FROM: bytewax/operators/__init__.py — ``filter``.
    """

    predicate: Callable[[T], bool]
    name: str = "filter"


@dataclass(frozen=True, slots=True)
class KeyByOp(Operator, Generic[T, K]):
    """Re-key the stream so downstream stateful operators can group.

    Produces ``(key, item)`` tuples. The bytewax equivalent is the
    ``key_on`` operator. Downstream ``ReduceOp`` / ``TumblingWindowOp``
    consume the keyed pairs.

    # ADAPTED FROM: bytewax/operators/__init__.py — ``key_on``.
    """

    key_fn: Callable[[T], str]
    name: str = "key_by"


@dataclass(frozen=True, slots=True)
class ReduceOp(Operator, Generic[A, T]):
    """Per-key stateful reducer.

    Input must be keyed ``(key, item)`` pairs (produced by a prior
    :class:`KeyByOp`). State per key starts from ``init()``. Each
    element advances state via ``step(acc, item)``. On stream end the
    reducer emits one tuple ``(key, final_state)`` per key, in
    first-seen key order.

    # ADAPTED FROM: bytewax/operators/__init__.py — ``reduce_final``.
    """

    init: Callable[[], A]
    step: Callable[[A, T], A]
    name: str = "reduce"


@dataclass(frozen=True, slots=True)
class TumblingWindowOp(Operator, Generic[T, A, U]):
    """Event-time tumbling window with deterministic emission.

    Input must be keyed ``(key, item)`` pairs (produced by a prior
    :class:`KeyByOp`). ``ts_fn`` extracts the event-time nanoseconds
    from the inner item. Buckets are
    ``floor(ts_ns / window_ns)`` — closed-open on the lower edge.

    On stream end the operator finalises each ``(bucket, key)``
    cell via ``finalize(acc, key, bucket_idx, bucket_start_ns)`` and
    emits the results sorted by ``(bucket_idx ascending,
    key ascending)`` so two byte-identical input streams produce
    two byte-identical output streams (INV-15).

    # ADAPTED FROM: bytewax/operators/window.py — ``TumblingWindow``.
    """

    window_ns: int
    ts_fn: Callable[[T], int]
    init: Callable[[], A]
    step: Callable[[A, T], A]
    finalize: Callable[[A, str, int, int], U]
    name: str = "tumbling_window"


# ---------------------------------------------------------------------------
# Dataflow — immutable builder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataflow:
    """Immutable operator-chain value object.

    Append a new operator by calling :meth:`then`, which returns a
    *new* :class:`Dataflow` (the existing instance is never mutated).
    The empty dataflow is the identity transform.

    # ADAPTED FROM: bytewax/dataflow.py — ``Dataflow`` builder.
    """

    name: str
    operators: tuple[Operator, ...] = ()

    def then(self, op: Operator) -> Dataflow:
        return Dataflow(name=self.name, operators=self.operators + (op,))

    def map(self, fn: Callable[[Any], Any]) -> Dataflow:
        return self.then(MapOp(fn=fn))

    def filter(self, predicate: Callable[[Any], bool]) -> Dataflow:
        return self.then(FilterOp(predicate=predicate))

    def key_by(self, key_fn: Callable[[Any], str]) -> Dataflow:
        return self.then(KeyByOp(key_fn=key_fn))

    def reduce(
        self,
        init: Callable[[], Any],
        step: Callable[[Any, Any], Any],
    ) -> Dataflow:
        return self.then(ReduceOp(init=init, step=step))

    def tumbling_window(
        self,
        window_ns: int,
        ts_fn: Callable[[Any], int],
        init: Callable[[], Any],
        step: Callable[[Any, Any], Any],
        finalize: Callable[[Any, str, int, int], Any],
    ) -> Dataflow:
        if window_ns <= 0:
            raise ValueError("window_ns must be > 0")
        return self.then(
            TumblingWindowOp(
                window_ns=window_ns,
                ts_fn=ts_fn,
                init=init,
                step=step,
                finalize=finalize,
            )
        )

    def dataflow_digest(self) -> str:
        """Stable 16-hex BLAKE2b digest over the operator chain spec.

        Encodes only operator *types*, declared names, and (where
        applicable) window sizes — i.e. the parts of the chain that
        are serialisable without inspecting closure cells. This is
        sufficient for INV-15 replay equality testing because two
        dataflows constructed from the same code path produce
        identical operator chains by construction.
        """

        spec: list[Mapping[str, object]] = [{"name": self.name}]
        for op in self.operators:
            entry: dict[str, object] = {
                "type": type(op).__name__,
                "name": op.name,
            }
            if isinstance(op, TumblingWindowOp):
                entry["window_ns"] = op.window_ns
            spec.append(entry)
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# FabricResult — advisory output record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FabricResult:
    """One emitted output from the fabric.

    The fabric is purely a transport — ``payload`` carries
    already-constructed DIX events (``SignalEvent`` / ``HazardEvent``
    / ``ExecutionEvent`` / ``SystemEvent``) or downstream aggregates.
    The fabric never constructs typed events itself; that authority
    stays with the producing engine (B27 / B28 / INV-71).
    """

    seq: int
    operator_name: str
    payload: object
    key: str = ""
    bucket_idx: int = -1
    meta: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# run_dataflow — in-process execution.
# ---------------------------------------------------------------------------


def run_dataflow(
    df: Dataflow,
    events: Iterable[object],
) -> tuple[FabricResult, ...]:
    """Execute ``df`` over ``events`` and return ``FabricResult`` tuple.

    Pure function — given identical inputs, returns byte-identical
    outputs (INV-15). No clock reads, no random, no I/O. The
    emission order is fully determined by the operator chain spec
    documented in each operator's docstring.

    The caller's ``events`` iterable is materialised once; downstream
    operators run over the in-memory list to keep the emission order
    well-defined.
    """

    stream: list[Any] = list(events)
    final_results: list[FabricResult] = []

    for op in df.operators:
        if isinstance(op, MapOp):
            stream = [op.fn(item) for item in stream]
        elif isinstance(op, FilterOp):
            stream = [item for item in stream if op.predicate(item)]
        elif isinstance(op, KeyByOp):
            stream = [(op.key_fn(item), item) for item in stream]
        elif isinstance(op, ReduceOp):
            stream = _apply_reduce(op, stream)
        elif isinstance(op, TumblingWindowOp):
            stream = _apply_tumbling_window(op, stream)
        else:  # pragma: no cover - exhaustiveness guard
            raise TypeError(f"unknown operator: {type(op).__name__}")

    seq = 0
    for item in stream:
        key = ""
        bucket_idx = -1
        payload: object = item
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            key, payload = item
        if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], int):
            bucket_idx, key, payload = item
        final_results.append(
            FabricResult(
                seq=seq,
                operator_name=df.operators[-1].name if df.operators else "identity",
                payload=payload,
                key=key,
                bucket_idx=bucket_idx,
            )
        )
        seq += 1

    return tuple(final_results)


def _apply_reduce(
    op: ReduceOp[Any, Any],
    stream: Sequence[Any],
) -> list[Any]:
    """Per-key fold with first-seen key order preservation.

    Input is a sequence of ``(key, item)`` tuples; output is one
    ``(key, acc)`` tuple per distinct key in first-seen order.
    """

    order: list[str] = []
    state: dict[str, Any] = {}
    for entry in stream:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise TypeError(f"ReduceOp expects (key, item) tuples; got {type(entry).__name__}")
        key, item = entry
        if not isinstance(key, str):
            raise TypeError(f"ReduceOp keys must be str; got {type(key).__name__}")
        if key not in state:
            state[key] = op.init()
            order.append(key)
        state[key] = op.step(state[key], item)
    return [(key, state[key]) for key in order]


def _apply_tumbling_window(
    op: TumblingWindowOp[Any, Any, Any],
    stream: Sequence[Any],
) -> list[Any]:
    """Event-time tumbling window with sorted-key emission.

    Input is a sequence of ``(key, item)`` tuples; output is one
    ``(bucket_idx, key, finalize(...))`` tuple per non-empty
    ``(bucket, key)`` cell, ordered by ``(bucket_idx ascending,
    key ascending)``.
    """

    cells: dict[tuple[int, str], Any] = {}
    for entry in stream:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise TypeError(
                f"TumblingWindowOp expects (key, item) tuples; got {type(entry).__name__}"
            )
        key, item = entry
        if not isinstance(key, str):
            raise TypeError(f"TumblingWindowOp keys must be str; got {type(key).__name__}")
        ts_ns = op.ts_fn(item)
        if not isinstance(ts_ns, int):
            raise TypeError(f"ts_fn must return int; got {type(ts_ns).__name__}")
        bucket = ts_ns // op.window_ns
        cell = (bucket, key)
        if cell not in cells:
            cells[cell] = op.init()
        cells[cell] = op.step(cells[cell], item)

    emitted: list[Any] = []
    for bucket, key in sorted(cells.keys()):
        acc = cells[(bucket, key)]
        bucket_start_ns = bucket * op.window_ns
        emitted.append((bucket, key, op.finalize(acc, key, bucket, bucket_start_ns)))
    return emitted


# ---------------------------------------------------------------------------
# Cross-process worker bridge.
# ---------------------------------------------------------------------------


class EventFabricSentinel:
    """Sentinel placed on the inbound queue to terminate the worker.

    A distinct class (not a string) so payloads that happen to be
    strings can never accidentally trip the shutdown path.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return "EventFabricSentinel()"


def fabric_worker_main(
    df: Dataflow,
    inbound: multiprocessing.Queue[object],
    outbound: multiprocessing.Queue[object],
    batch_size: int = 1,
) -> None:
    """Worker entrypoint.

    Pulls batches of events off ``inbound`` until a
    :class:`EventFabricSentinel` is received, then runs ``df`` over
    each batch and pushes the resulting :class:`FabricResult` tuple
    onto ``outbound`` (preserving operator-chain ordering).

    A second sentinel is then pushed onto ``outbound`` so the parent
    can detect a clean shutdown.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch: list[object] = []
    while True:
        item = inbound.get()
        if isinstance(item, EventFabricSentinel):
            if batch:
                outbound.put(run_dataflow(df, batch))
                batch.clear()
            outbound.put(EventFabricSentinel())
            return
        batch.append(item)
        if len(batch) >= batch_size:
            outbound.put(run_dataflow(df, batch))
            batch.clear()


def spawn_fabric_worker(
    df: Dataflow,
    inbound: multiprocessing.Queue[object],
    outbound: multiprocessing.Queue[object],
    batch_size: int = 1,
) -> multiprocessing.Process:
    """Spawn a child process running :func:`fabric_worker_main`.

    Uses the ``spawn`` start method so the child has an independent
    interpreter (no inherited module state from the parent). This
    matches bytewax's production deployment model and avoids fork
    determinism hazards on Linux.

    Returns the started :class:`multiprocessing.Process` so the
    caller can ``join()`` it. The caller is responsible for placing
    an :class:`EventFabricSentinel` on ``inbound`` to terminate the
    worker.
    """

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=fabric_worker_main,
        args=(df, inbound, outbound, batch_size),
        daemon=False,
    )
    proc.start()
    return proc


def drain_queue(
    outbound: multiprocessing.Queue[object],
    *,
    timeout_s: float | None = None,
) -> tuple[FabricResult, ...]:
    """Drain ``outbound`` until the worker's terminating sentinel.

    Concatenates each ``FabricResult`` tuple in arrival order and
    returns the flat tuple. The ``timeout_s`` parameter is a
    per-``get`` deadline; pass ``None`` (default) for an unbounded
    blocking get.

    Raises ``TimeoutError`` if a ``get`` times out before the
    terminating sentinel arrives.
    """

    drained: list[FabricResult] = []
    while True:
        try:
            chunk = outbound.get(timeout=timeout_s)
        except Empty as exc:  # pragma: no cover - timeout path
            raise TimeoutError("event fabric outbound drain timed out") from exc
        if isinstance(chunk, EventFabricSentinel):
            return tuple(drained)
        if not isinstance(chunk, tuple):
            raise TypeError(
                f"fabric outbound expects FabricResult tuples; got {type(chunk).__name__}"
            )
        drained.extend(chunk)


# ---------------------------------------------------------------------------
# bytewax lazy factory — research-acceptance gate.
# ---------------------------------------------------------------------------


def bytewax_dataflow_factory(*args: object, **kwargs: object) -> object:
    """Lazy bytewax bridge — pinned ``NotImplementedError``.

    Wiring the real :mod:`bytewax` PyPI package is OUT OF SCOPE for
    C-01. The canonical block declares bytewax as a research source;
    activation is gated by a future PR that:

    1. Documents a shadow-equivalence harness comparing the
       in-process Python execution above against the real bytewax
       backend over a fixed event log.
    2. Demonstrates byte-identical aggregates between the two
       backends across at least one full replay cycle.
    3. Pins the bytewax operator surface to the subset adapted here
       (``map`` / ``filter`` / ``key_on`` / ``reduce_final`` /
       ``TumblingWindow``).

    Until that PR lands, this factory raises so any accidental
    production import is loud rather than silent.
    """

    raise NotImplementedError(
        "bytewax_dataflow_factory: gated until research-acceptance "
        "PR documents shadow-equivalence + byte-identical replay "
        "against in-process Python execution"
    )
