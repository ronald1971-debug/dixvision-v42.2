"""C-07 streamz — Lightweight In-Process Complex Event Processing.

# ADAPTED FROM: python-streamz/streamz — ``streamz/core.py``
# (``Stream`` node graph, ``map`` / ``filter`` / ``accumulate`` /
# ``zip`` / ``combine_latest`` operators) and ``streamz/batch.py``
# (``sliding_window`` windowed batch).
#
# Tier: OFFLINE_ONLY — the streamz graph runs in-process under a
# deterministic executor; the pipeline never imports any RUNTIME
# tier (no ``intelligence_engine``, ``execution_engine``,
# ``governance_engine`` imports). The graph emits ``GraphResult``
# advisory records carrying already-constructed values; the CEP
# transport NEVER constructs DIX typed events itself. This preserves
# B27 / B28 / INV-71 authority symmetry — only the engine that
# produced an event may construct it, never the transport.

The DIX in-process Python translation below is sufficient for offline
replay tests and for the in-process pump that production callers wire
into ``system_engine/hazard_sensors/sensor_array.py``. The real
:mod:`streamz` PyPI package is *only* imported inside
:func:`streamz_stream_factory` and only after the research-acceptance
gate documented there. Importing the package at module load time would
(a) break determinism — streamz depends on :mod:`asyncio` for its async
sink mode — and (b) violate the OFFLINE_ONLY contract for downstream
importers.

Differentiator over C-01 ``event_fabric`` (bytewax) and C-02
``faust_bus`` (Faust):

* **Node graph, not linear dataflow.** A streamz graph is a DAG —
  nodes can have multiple downstream subscribers, branches can be
  zipped together, and a node may feed both a sink and another
  operator. Bytewax / Faust operate on a single linear pipeline.
* **In-process only.** No cross-process worker — streamz is the
  lightweight CEP option intended for single-process callers
  (sensor_array, hazard fan-out). Cross-process bridges live in
  C-01 / C-02 / C-03 / C-05 / C-06.
* **Sinks are explicit value-object records.** Each terminal node
  records into ``GraphResult.sink_outputs[name]`` so callers reason
  about the full snapshot of every sink after a deterministic run.

Determinism (INV-15):

* No top-level imports of :mod:`time` / :mod:`datetime` /
  :mod:`random` / :mod:`asyncio` / :mod:`os` / :mod:`streamz`.
* All windowing is over caller-supplied ordered input — there is no
  wall-clock notion of "now"; ``sliding_window(n)`` slides over the
  last ``n`` events delivered to the node by the executor.
* Operator output order is deterministic: emission follows the
  topological order of the graph (computed once via Kahn's algorithm
  with ties broken by lexicographic node name) and, within a node,
  the insertion order of inputs.
* Frozen, slotted dataclasses everywhere. The graph itself is a
  ``StreamGraph`` value object — adding a node or an edge returns a
  *new* ``StreamGraph`` (immutable builder).
* BLAKE2b-16 ``graph_digest`` over the node + edge spec gives
  byte-identical replay equality.

Authority discipline:

* B27 / B28 / INV-71: this module does **not** call
  ``PatchProposal(...)``, ``HazardEvent(...)``, ``SignalEvent(...)``,
  ``ExecutionEvent(...)`` or ``SystemEvent(...)`` directly. AST tests
  pin the constraint.
* B1 isolation: no imports from ``intelligence_engine``,
  ``execution_engine``, ``governance_engine``, ``evolution_engine``,
  ``learning_engine``. The CEP graph is a leaf transport.

Outputs declared by canonical block C-07:

1. ``system_engine/streaming/streamz_cep.py`` (this file)
2. ``tests/test_streamz_cep.py``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

STREAMZ_CEP_VERSION: int = 1

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("streamz",)
"""Declared so the canonical pin-set is complete.

The package itself is NEVER imported in this module — see the module
docstring for the rationale and :func:`streamz_stream_factory` for the
lazy seam where a future PR can wire it up after the
research-acceptance gate is documented.
"""


T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")


# ---------------------------------------------------------------------------
# Node value-objects.
# ---------------------------------------------------------------------------


class Node:
    """Marker base for all node value-objects in the graph.

    Concrete subclasses are frozen + slotted dataclasses below. Using a
    marker class (rather than a ``Union`` type alias) lets
    :func:`run_graph` dispatch via ``isinstance`` without importing the
    concrete types eagerly.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class SourceNode(Node):
    """Entry-point node — receives inputs at the start of each run.

    # ADAPTED FROM: streamz/core.py — ``Stream`` root node.
    """

    name: str


@dataclass(frozen=True, slots=True)
class MapNode(Node, Generic[T, U]):
    """Element-wise transform on a single upstream.

    # ADAPTED FROM: streamz/core.py — ``Stream.map``.
    """

    name: str
    upstream: str
    fn: Callable[[T], U]


@dataclass(frozen=True, slots=True)
class FilterNode(Node, Generic[T]):
    """Drop elements where ``predicate(x)`` is falsy.

    # ADAPTED FROM: streamz/core.py — ``Stream.filter``.
    """

    name: str
    upstream: str
    predicate: Callable[[T], bool]


@dataclass(frozen=True, slots=True)
class AccumulateNode(Node, Generic[A, T, U]):
    """Stateful scan — emits one output per input.

    State starts from ``init()``. Each element advances state via
    ``step(acc, item) -> (new_acc, emit)``. The emitted value is
    forwarded downstream; the new accumulator is kept for the next
    input.

    # ADAPTED FROM: streamz/core.py — ``Stream.accumulate``.
    """

    name: str
    upstream: str
    init: Callable[[], A]
    step: Callable[[A, T], tuple[A, U]]


@dataclass(frozen=True, slots=True)
class ZipNode(Node):
    """Synchronous zip — emits one tuple per matched index across upstreams.

    The node buffers per-upstream queues; once *every* upstream has at
    least one queued element, the head of each queue is popped and
    emitted as a tuple in the declared upstream order.

    # ADAPTED FROM: streamz/core.py — ``Stream.zip``.
    """

    name: str
    upstreams: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CombineLatestNode(Node):
    """Combine-latest — emits whenever any upstream produces, using the
    most recent value seen on every other upstream.

    No output is produced until every upstream has emitted at least
    once (matches streamz's ``combine_latest`` semantic).

    # ADAPTED FROM: streamz/core.py — ``Stream.combine_latest``.
    """

    name: str
    upstreams: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SlidingWindowNode(Node, Generic[T]):
    """Sliding window of fixed size ``n`` over the upstream.

    Emits a tuple of the last ``n`` elements on every input after the
    window has filled. ``return_partial=True`` emits partial windows
    while filling.

    # ADAPTED FROM: streamz/batch.py — ``Stream.sliding_window``.
    """

    name: str
    upstream: str
    n: int
    return_partial: bool = False


@dataclass(frozen=True, slots=True)
class SinkNode(Node, Generic[T]):
    """Terminal node — records every input into ``GraphResult.sink_outputs``.

    The executor records emissions in arrival order. ``sink_fn`` is an
    optional callable invoked for each element (allowing side-effecting
    sinks); the return value is ignored.

    # ADAPTED FROM: streamz/core.py — ``Stream.sink``.
    """

    name: str
    upstream: str
    sink_fn: Callable[[T], None] | None = None


# ---------------------------------------------------------------------------
# StreamGraph — immutable builder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamGraph:
    """Immutable node-graph value object.

    The graph is a DAG of named nodes; the executor follows topological
    order with lexicographic tie-breaking on node name. The empty graph
    has zero nodes.

    # ADAPTED FROM: streamz/core.py — ``Stream`` graph builder.
    """

    name: str
    nodes: tuple[Node, ...] = ()

    # ------------------------------------------------------------------
    # Builders — every one returns a NEW StreamGraph.
    # ------------------------------------------------------------------

    def _with(self, node: Node) -> StreamGraph:
        if any(_node_name(n) == _node_name(node) for n in self.nodes):
            raise ValueError(f"duplicate node name {_node_name(node)!r}")
        return StreamGraph(name=self.name, nodes=self.nodes + (node,))

    def source(self, name: str) -> StreamGraph:
        return self._with(SourceNode(name=name))

    def map(self, name: str, upstream: str, fn: Callable[[Any], Any]) -> StreamGraph:
        self._require_upstream(upstream)
        return self._with(MapNode(name=name, upstream=upstream, fn=fn))

    def filter(
        self,
        name: str,
        upstream: str,
        predicate: Callable[[Any], bool],
    ) -> StreamGraph:
        self._require_upstream(upstream)
        return self._with(FilterNode(name=name, upstream=upstream, predicate=predicate))

    def accumulate(
        self,
        name: str,
        upstream: str,
        init: Callable[[], Any],
        step: Callable[[Any, Any], tuple[Any, Any]],
    ) -> StreamGraph:
        self._require_upstream(upstream)
        return self._with(AccumulateNode(name=name, upstream=upstream, init=init, step=step))

    def zip(self, name: str, upstreams: Sequence[str]) -> StreamGraph:
        if len(upstreams) < 2:
            raise ValueError("zip requires at least 2 upstreams")
        for up in upstreams:
            self._require_upstream(up)
        return self._with(ZipNode(name=name, upstreams=tuple(upstreams)))

    def combine_latest(self, name: str, upstreams: Sequence[str]) -> StreamGraph:
        if len(upstreams) < 2:
            raise ValueError("combine_latest requires at least 2 upstreams")
        for up in upstreams:
            self._require_upstream(up)
        return self._with(CombineLatestNode(name=name, upstreams=tuple(upstreams)))

    def sliding_window(
        self,
        name: str,
        upstream: str,
        n: int,
        *,
        return_partial: bool = False,
    ) -> StreamGraph:
        if n <= 0:
            raise ValueError("sliding_window n must be > 0")
        self._require_upstream(upstream)
        return self._with(
            SlidingWindowNode(
                name=name,
                upstream=upstream,
                n=n,
                return_partial=return_partial,
            )
        )

    def sink(
        self,
        name: str,
        upstream: str,
        sink_fn: Callable[[Any], None] | None = None,
    ) -> StreamGraph:
        self._require_upstream(upstream)
        return self._with(SinkNode(name=name, upstream=upstream, sink_fn=sink_fn))

    # ------------------------------------------------------------------
    # Read-only helpers.
    # ------------------------------------------------------------------

    def _require_upstream(self, upstream: str) -> None:
        if not any(_node_name(n) == upstream for n in self.nodes):
            raise KeyError(f"unknown upstream {upstream!r}")

    def node_names(self) -> tuple[str, ...]:
        return tuple(_node_name(n) for n in self.nodes)

    def source_names(self) -> tuple[str, ...]:
        return tuple(_node_name(n) for n in self.nodes if isinstance(n, SourceNode))

    def sink_names(self) -> tuple[str, ...]:
        return tuple(_node_name(n) for n in self.nodes if isinstance(n, SinkNode))

    def graph_digest(self) -> str:
        """Stable 16-hex BLAKE2b digest over the node + edge spec.

        Encodes only node *types*, names, declared upstreams, and (where
        applicable) static parameters (``n``, ``return_partial``). It is
        sufficient for INV-15 replay equality because two graphs that
        share a digest will produce byte-identical outputs given the
        same source inputs and the same callable bodies.
        """
        encoded: list[dict[str, Any]] = []
        for n in self.nodes:
            entry: dict[str, Any] = {
                "type": type(n).__name__,
                "name": _node_name(n),
            }
            if isinstance(n, MapNode | FilterNode | AccumulateNode | SinkNode):
                entry["upstream"] = n.upstream
            if isinstance(n, ZipNode | CombineLatestNode):
                entry["upstreams"] = list(n.upstreams)
            if isinstance(n, SlidingWindowNode):
                entry["upstream"] = n.upstream
                entry["n"] = n.n
                entry["return_partial"] = n.return_partial
            encoded.append(entry)
        encoded.sort(key=lambda e: e["name"])
        payload = json.dumps(
            {"graph": self.name, "nodes": encoded},
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _node_name(node: Node) -> str:
    """Stable name accessor that doesn't require ``isinstance``."""
    name = getattr(node, "name", None)
    if not isinstance(name, str):
        raise AssertionError(f"node has no string ``name``: {node!r}")
    return name


# ---------------------------------------------------------------------------
# GraphResult value-object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphResult:
    """Pure-value output of :func:`run_graph`.

    * ``graph_name`` — the graph that produced this result.
    * ``graph_digest`` — digest of the graph spec (matches
      :meth:`StreamGraph.graph_digest`).
    * ``sink_outputs`` — for each sink node, a tuple of every value
      that reached it, in arrival order.
    * ``run_digest`` — BLAKE2b-16 over ``graph_digest`` + the
      canonicalised ``sink_outputs``. Two byte-identical runs of the
      same graph over the same inputs MUST produce the same
      ``run_digest`` (INV-15).
    """

    graph_name: str
    graph_digest: str
    sink_outputs: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    run_digest: str = ""


# ---------------------------------------------------------------------------
# run_graph — deterministic in-process executor.
# ---------------------------------------------------------------------------


def _topological_order(graph: StreamGraph) -> tuple[Node, ...]:
    """Kahn's algorithm with lexicographic tie-breaking on node name."""
    names = list(graph.node_names())
    by_name: dict[str, Node] = {_node_name(n): n for n in graph.nodes}
    in_deps: dict[str, set[str]] = {name: set() for name in names}
    out_deps: dict[str, set[str]] = {name: set() for name in names}
    for n in graph.nodes:
        ups: tuple[str, ...]
        if isinstance(n, SourceNode):
            ups = ()
        elif isinstance(n, ZipNode | CombineLatestNode):
            ups = n.upstreams
        elif isinstance(
            n,
            MapNode | FilterNode | AccumulateNode | SlidingWindowNode | SinkNode,
        ):
            ups = (n.upstream,)
        else:
            raise AssertionError(f"unknown node type: {type(n).__name__}")
        nm = _node_name(n)
        for u in ups:
            in_deps[nm].add(u)
            out_deps[u].add(nm)
    # Kahn's algorithm with lexicographic tie-break.
    ready = sorted(name for name, deps in in_deps.items() if not deps)
    order: list[Node] = []
    while ready:
        nm = ready.pop(0)
        order.append(by_name[nm])
        for downstream in sorted(out_deps[nm]):
            in_deps[downstream].discard(nm)
            if not in_deps[downstream]:
                # Insert maintaining sorted order so ties break lexicographically.
                bisect_insort(ready, downstream)
    if len(order) != len(graph.nodes):
        raise ValueError("graph has a cycle")
    return tuple(order)


def bisect_insort(seq: list[str], item: str) -> None:
    """Tiny inline ``bisect.insort`` so we don't add a top-level import."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < item:
            lo = mid + 1
        else:
            hi = mid
    seq.insert(lo, item)


def run_graph(
    graph: StreamGraph,
    sources: Mapping[str, Sequence[Any]],
) -> GraphResult:
    """Run ``graph`` against ``sources`` to completion, returning a
    fully-materialised :class:`GraphResult`.

    ``sources`` maps each :class:`SourceNode` name to an ordered
    sequence of input values. Every source declared on the graph MUST
    appear in ``sources``. Unknown source names raise ``KeyError`` so a
    typo doesn't silently drop data.

    The executor is single-pass: emissions from each upstream node are
    fully drained into per-edge queues before the executor advances to
    the next node in topological order. This makes the run trivially
    deterministic — every node sees its inputs in the exact order the
    upstream produced them.
    """
    if not isinstance(graph, StreamGraph):
        raise TypeError("graph must be StreamGraph")
    # Validate source coverage.
    expected_sources = set(graph.source_names())
    declared = set(sources.keys())
    missing = expected_sources - declared
    if missing:
        raise KeyError(f"sources missing for nodes: {sorted(missing)}")
    extra = declared - expected_sources
    if extra:
        raise KeyError(f"sources for unknown nodes: {sorted(extra)}")

    order = _topological_order(graph)
    # Per-node outbox of emissions in order produced.
    outbox: dict[str, list[Any]] = {_node_name(n): [] for n in graph.nodes}
    sink_outputs: dict[str, list[Any]] = {}

    for node in order:
        nm = _node_name(node)
        if isinstance(node, SourceNode):
            for item in sources[nm]:
                outbox[nm].append(item)
            continue
        if isinstance(node, MapNode):
            for item in outbox[node.upstream]:
                outbox[nm].append(node.fn(item))
            continue
        if isinstance(node, FilterNode):
            for item in outbox[node.upstream]:
                if node.predicate(item):
                    outbox[nm].append(item)
            continue
        if isinstance(node, AccumulateNode):
            acc = node.init()
            for item in outbox[node.upstream]:
                acc, emit = node.step(acc, item)
                outbox[nm].append(emit)
            continue
        if isinstance(node, SlidingWindowNode):
            buf: list[Any] = []
            for item in outbox[node.upstream]:
                buf.append(item)
                if len(buf) > node.n:
                    buf.pop(0)
                if len(buf) == node.n or node.return_partial:
                    outbox[nm].append(tuple(buf))
            continue
        if isinstance(node, ZipNode):
            # Synchronous zip — one tuple per index across upstreams.
            queues = [list(outbox[u]) for u in node.upstreams]
            limit = min(len(q) for q in queues)
            for i in range(limit):
                outbox[nm].append(tuple(q[i] for q in queues))
            continue
        if isinstance(node, CombineLatestNode):
            # Process all upstream emissions in upstream-declaration
            # order (deterministic). Each emission first refreshes the
            # state for that upstream; once every upstream has at least
            # one value, the combined tuple is emitted.
            latest: dict[str, Any] = {}
            sentinel = object()
            for up in node.upstreams:
                latest[up] = sentinel
            for up in node.upstreams:
                for item in outbox[up]:
                    latest[up] = item
                    if all(latest[u] is not sentinel for u in node.upstreams):
                        outbox[nm].append(tuple(latest[u] for u in node.upstreams))
            continue
        if isinstance(node, SinkNode):
            recorded: list[Any] = []
            for item in outbox[node.upstream]:
                if node.sink_fn is not None:
                    node.sink_fn(item)
                recorded.append(item)
            sink_outputs[nm] = recorded
            continue
        raise AssertionError(f"unhandled node type: {type(node).__name__}")

    finalized = {nm: tuple(vals) for nm, vals in sink_outputs.items()}
    graph_dig = graph.graph_digest()
    run_dig = _run_digest(graph_dig, finalized)
    return GraphResult(
        graph_name=graph.name,
        graph_digest=graph_dig,
        sink_outputs=finalized,
        run_digest=run_dig,
    )


def _run_digest(graph_digest: str, sink_outputs: Mapping[str, tuple[Any, ...]]) -> str:
    """Stable BLAKE2b-16 digest over ``(graph_digest, canonicalised sinks)``.

    Sink names are sorted lexicographically; each sink's outputs are
    serialised in arrival order. Non-JSON-serialisable values are
    coerced through ``repr`` so the digest still pins them — two runs
    that produce the same ``repr`` per element produce the same digest.
    """
    encoded_sinks: list[tuple[str, list[str]]] = []
    for name in sorted(sink_outputs):
        encoded_sinks.append((name, [_safe_repr(v) for v in sink_outputs[name]]))
    payload = json.dumps(
        {"graph_digest": graph_digest, "sinks": encoded_sinks},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _safe_repr(v: Any) -> str:
    """``repr`` that prefers JSON for primitive containers — bytes, ints,
    floats, str, None, tuples/lists/dicts of primitives — and falls back
    to plain ``repr`` for everything else.

    This keeps the digest stable across CPython versions for the common
    primitive case while still pinning richer objects deterministically.
    """
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=True, default=repr)
    except TypeError:
        return repr(v)


# ---------------------------------------------------------------------------
# Lazy seam — real streamz hookup gate.
# ---------------------------------------------------------------------------


def streamz_stream_factory(graph: StreamGraph) -> Any:
    """Lazy seam — would construct a real :class:`streamz.Stream` graph.

    Intentionally raises :class:`NotImplementedError` on current main.
    A future PR may import :mod:`streamz` inside this function (NOT at
    module level) once:

    1. A shadow-equivalence proof is checked in showing
       :func:`run_graph` matches the live ``Stream`` semantics across
       every operator we use, AND
    2. A governance research-acceptance entry covers the import —
       streamz uses :mod:`asyncio` internally and pulls in
       :mod:`tornado`, so the import-time effects must be approved.

    Until then this seam is the canonical "we declared the dependency"
    marker for the ``NEW_PIP_DEPENDENCIES`` audit.
    """
    raise NotImplementedError(
        "live streamz.Stream graphs are gated on a shadow-equivalence "
        "research-acceptance PR; use run_graph(...) for offline replay"
    )
