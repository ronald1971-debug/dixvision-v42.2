"""A-12 networkx → causal dependency graph.

# ADAPTED FROM: networkx/networkx
#   - networkx/classes/digraph.py — DiGraph API surface (add_node /
#     add_edge / nodes / edges / predecessors / successors)
#   - networkx/algorithms/dag.py — is_directed_acyclic_graph,
#     topological_sort, descendants, ancestors
#   - networkx/algorithms/shortest_paths/generic.py — shortest_path,
#     all_simple_paths
# BSD-3-Clause license; no networkx source is reproduced verbatim — only
# the public algorithmic contract (call signature, return shape,
# topological-sort ordering, DAG-required pre-conditions) is mirrored.

The :class:`CausalGraph` is a thin, pure-Python adaptation of the
``networkx.DiGraph`` surface that the rest of DIX needs for causal
analysis of governance decisions (spec lines 1226–1230). Nodes and
edges carry typed attribute payloads; the graph enforces a DAG
invariant at every mutation; topological sort, ancestor / descendant
walks, and all-simple-paths enumeration are exposed for runtime
read consumers.

Tier discipline (spec lines 1233–1236):

* **OFFLINE writes** — only ``learning_engine.*`` / ``governance_engine.*``
  causal-analysis paths may mutate the graph (add nodes / edges, set
  attributes). The mutation surface is named ``add_node`` /
  ``add_edge`` / ``set_node_attr`` / ``set_edge_attr`` /
  ``remove_edge`` / ``clear``.
* **Runtime reads** — every read surface (``has_node``, ``has_edge``,
  ``predecessors``, ``successors``, ``ancestors``, ``descendants``,
  ``topological_sort``, ``shortest_path``, ``all_simple_paths``,
  ``is_dag``) is O(V+E) at worst against the in-memory adjacency
  lists and is therefore allowed in the hot path under RUNTIME_SAFE
  classification.

Determinism (INV-15):

* No clock reads — every ``ts_ns`` is supplied by the caller.
* No randomness — adjacency lists are stored as ``dict`` and iterated
  in sorted-key order on every traversal so two graphs with the same
  logical contents enumerate identically.
* Sorted-key JSON projection for ledger checkpointing.

Authority symmetry (B27 / B28 / INV-71):

This module does **not** construct typed bus events. It is a passive
analysis structure. ``CausalGraph`` rows describing governance
decisions are produced by the caller and projected here for offline
introspection; the runtime never writes back through this module.

Fallback: a ``networkx``-backed equivalent is exposed via
``networkx_export()`` for callers that want to plug the graph into the
broader ``networkx`` algorithm suite. The export is lazy — the
``networkx`` import is confined to that function body.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

# ---------------------------------------------------------------------------
# Module identity / dependency declaration.
# ---------------------------------------------------------------------------

CAUSAL_GRAPH_VERSION: str = "1"
"""Serialisation version pin (incremented on breaking schema changes)."""

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("networkx",)
"""Optional — required only for :func:`CausalGraph.networkx_export`."""


__all__ = (
    "CAUSAL_GRAPH_VERSION",
    "CausalGraph",
    "CausalGraphError",
    "CycleError",
    "EdgeView",
    "NEW_PIP_DEPENDENCIES",
    "NodeView",
)


class CausalGraphError(RuntimeError):
    """Raised on contract / schema violations."""


class CycleError(CausalGraphError):
    """Raised when a mutation would introduce a cycle.

    The DAG invariant is the central correctness property of the graph
    (topological_sort, ancestors, descendants are all DAG-only). A
    failed mutation leaves the graph unchanged.
    """


def _validate_prop_value(value: Any) -> None:
    """Reject non-JSON-primitive attribute values.

    INV-15: keeping every payload value JSON-primitive ensures the
    serialisation projection is byte-stable across Python versions.
    """
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float | str):
        return
    raise CausalGraphError(f"attr value must be JSON-primitive, got {type(value).__name__}")


def _freeze(props: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a key-sorted read-only projection."""
    return MappingProxyType(dict(sorted(props.items())))


@dataclass(frozen=True, slots=True)
class NodeView:
    """One frozen graph node.

    Attributes:
        node_id: Stable, unique identifier.
        attrs: Frozen, key-sorted attribute payload mapping. Values
            must be JSON-primitive
            (``int`` / ``float`` / ``str`` / ``bool`` / ``None``).
    """

    node_id: str
    attrs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EdgeView:
    """One frozen directed edge.

    Attributes:
        source_id: Origin node ID.
        target_id: Destination node ID.
        attrs: Frozen, key-sorted edge-attribute mapping.
    """

    source_id: str
    target_id: str
    attrs: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Causal graph.
# ---------------------------------------------------------------------------


class CausalGraph:
    """A pure-Python DAG with a ``networkx.DiGraph``-shaped surface.

    The graph maintains:

    * ``self._nodes`` — ``dict[node_id, dict[attr_key, attr_value]]``,
      iterated in sorted ``node_id`` order on every traversal.
    * ``self._out`` — ``dict[node_id, dict[node_id, dict[attr]]]`` —
      forward adjacency.
    * ``self._in`` — ``dict[node_id, set[node_id]]`` — reverse
      adjacency (used for ``predecessors`` / ``ancestors``).

    Both forward and reverse adjacency are maintained in lockstep on
    every mutation. Sorted iteration on every read pins INV-15.
    """

    __slots__ = ("_in", "_nodes", "_out")

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._out: dict[str, dict[str, dict[str, Any]]] = {}
        self._in: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # OFFLINE write surface.
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, /, **attrs: Any) -> None:
        """Idempotent node insert.

        Re-adding an existing node updates its attribute payload
        (caller-supplied attrs replace existing keys; unspecified keys
        are preserved). Mirrors ``networkx.DiGraph.add_node``.
        """
        self._require_nonempty_str("node_id", node_id)
        for value in attrs.values():
            _validate_prop_value(value)
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = dict(attrs)
            self._out.setdefault(node_id, {})
            self._in.setdefault(node_id, set())
        else:
            existing.update(attrs)

    def add_edge(self, source_id: str, target_id: str, /, **attrs: Any) -> None:
        """Idempotent directed-edge insert with cycle detection.

        Both endpoints are auto-created if absent (mirrors networkx).
        Re-adding an existing edge updates its attribute payload. If
        the edge would create a cycle, :class:`CycleError` is raised
        and the graph is left unchanged.
        """
        self._require_nonempty_str("source_id", source_id)
        self._require_nonempty_str("target_id", target_id)
        if source_id == target_id:
            raise CycleError(f"self-edge not permitted: {source_id!r}")
        for value in attrs.values():
            _validate_prop_value(value)
        # Auto-create endpoints (preserves existing attrs).
        if source_id not in self._nodes:
            self.add_node(source_id)
        if target_id not in self._nodes:
            self.add_node(target_id)
        # Cycle check: a cycle exists iff ``source`` is reachable from
        # ``target`` (the new edge would close the loop).
        if self._reachable(target_id, source_id):
            raise CycleError(f"adding edge {source_id!r}->{target_id!r} would create a cycle")
        existing = self._out[source_id].get(target_id)
        if existing is None:
            self._out[source_id][target_id] = dict(attrs)
            self._in[target_id].add(source_id)
        else:
            existing.update(attrs)

    def set_node_attr(self, node_id: str, key: str, value: Any) -> None:
        """Set a single node attribute."""
        self._require_node_exists(node_id)
        self._require_nonempty_str("key", key)
        _validate_prop_value(value)
        self._nodes[node_id][key] = value

    def set_edge_attr(self, source_id: str, target_id: str, key: str, value: Any) -> None:
        """Set a single edge attribute."""
        self._require_edge_exists(source_id, target_id)
        self._require_nonempty_str("key", key)
        _validate_prop_value(value)
        self._out[source_id][target_id][key] = value

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove a directed edge; raises if absent."""
        self._require_edge_exists(source_id, target_id)
        del self._out[source_id][target_id]
        self._in[target_id].discard(source_id)

    def clear(self) -> None:
        """Remove every node and edge."""
        self._nodes.clear()
        self._out.clear()
        self._in.clear()

    # ------------------------------------------------------------------
    # Runtime read surface.
    # ------------------------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, source_id: str, target_id: str) -> bool:
        return source_id in self._out and target_id in self._out[source_id]

    def node(self, node_id: str) -> NodeView:
        """Return a :class:`NodeView` for ``node_id`` (KeyError if missing)."""
        self._require_node_exists(node_id)
        return NodeView(node_id=node_id, attrs=_freeze(self._nodes[node_id]))

    def edge(self, source_id: str, target_id: str) -> EdgeView:
        """Return an :class:`EdgeView` (KeyError if missing)."""
        self._require_edge_exists(source_id, target_id)
        return EdgeView(
            source_id=source_id,
            target_id=target_id,
            attrs=_freeze(self._out[source_id][target_id]),
        )

    def predecessors(self, node_id: str) -> tuple[str, ...]:
        """Direct predecessors of ``node_id``, sorted."""
        self._require_node_exists(node_id)
        return tuple(sorted(self._in[node_id]))

    def successors(self, node_id: str) -> tuple[str, ...]:
        """Direct successors of ``node_id``, sorted."""
        self._require_node_exists(node_id)
        return tuple(sorted(self._out[node_id].keys()))

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        """All transitive predecessors of ``node_id``, sorted.

        Excludes ``node_id`` itself. Mirrors
        ``networkx.algorithms.dag.ancestors``.
        """
        self._require_node_exists(node_id)
        seen: set[str] = set()
        stack = list(self._in[node_id])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._in.get(current, ()))
        return tuple(sorted(seen))

    def descendants(self, node_id: str) -> tuple[str, ...]:
        """All transitive successors of ``node_id``, sorted.

        Excludes ``node_id`` itself. Mirrors
        ``networkx.algorithms.dag.descendants``.
        """
        self._require_node_exists(node_id)
        seen: set[str] = set()
        stack = list(self._out[node_id].keys())
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._out.get(current, {}).keys())
        return tuple(sorted(seen))

    def is_dag(self) -> bool:
        """Always ``True`` — the DAG invariant is enforced at write time."""
        return True

    def topological_sort(self) -> tuple[str, ...]:
        """Return one valid topological ordering of every node.

        Uses Kahn's algorithm with a sorted ready-queue so the
        enumeration is INV-15 deterministic across runs / machines.
        Mirrors ``networkx.algorithms.dag.topological_sort``.
        """
        indegree = {nid: len(self._in[nid]) for nid in self._nodes}
        ready = sorted(nid for nid, d in indegree.items() if d == 0)
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for succ in sorted(self._out[nid].keys()):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    # Maintain sorted invariant on the ready queue.
                    self._sorted_insert(ready, succ)
        if len(order) != len(self._nodes):
            # Should never happen — the DAG invariant pins this.
            raise CausalGraphError(  # pragma: no cover - defensive
                "topological_sort failed (graph is not a DAG)"
            )
        return tuple(order)

    def shortest_path(self, source_id: str, target_id: str) -> tuple[str, ...] | None:
        """Shortest directed path from ``source_id`` to ``target_id``.

        Returns ``None`` if no path exists. BFS over forward adjacency
        with sorted neighbour enumeration. Mirrors
        ``networkx.algorithms.shortest_paths.generic.shortest_path``.
        """
        self._require_node_exists(source_id)
        self._require_node_exists(target_id)
        if source_id == target_id:
            return (source_id,)
        # BFS with predecessor map.
        predecessor: dict[str, str] = {}
        visited: set[str] = {source_id}
        frontier: list[str] = [source_id]
        while frontier:
            next_frontier: list[str] = []
            for current in frontier:
                for nxt in sorted(self._out[current].keys()):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    predecessor[nxt] = current
                    if nxt == target_id:
                        return self._reconstruct_path(predecessor, source_id, target_id)
                    next_frontier.append(nxt)
            frontier = next_frontier
        return None

    def all_simple_paths(
        self, source_id: str, target_id: str, *, max_depth: int
    ) -> tuple[tuple[str, ...], ...]:
        """Enumerate every acyclic path from ``source_id`` to ``target_id``.

        ``max_depth`` bounds path length (in edges). Returned paths are
        sorted by ``(length, lexicographic node sequence)`` to pin
        INV-15. Mirrors
        ``networkx.algorithms.simple_paths.all_simple_paths``.
        """
        self._require_node_exists(source_id)
        self._require_node_exists(target_id)
        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise CausalGraphError("max_depth must be int")
        if max_depth <= 0:
            raise CausalGraphError(f"max_depth must be positive: {max_depth!r}")
        paths: list[tuple[str, ...]] = []
        # DFS enumerating every acyclic path up to ``max_depth`` hops.
        stack: list[tuple[str, tuple[str, ...]]] = [(source_id, (source_id,))]
        while stack:
            current, path = stack.pop()
            if current == target_id and len(path) > 1:
                paths.append(path)
                continue
            if len(path) - 1 >= max_depth:
                continue
            for nxt in sorted(self._out[current].keys(), reverse=True):
                if nxt in path:
                    continue
                stack.append((nxt, (*path, nxt)))
        paths.sort(key=lambda p: (len(p), p))
        return tuple(paths)

    # ------------------------------------------------------------------
    # Iteration.
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[NodeView]:
        for nid in sorted(self._nodes.keys()):
            yield NodeView(node_id=nid, attrs=_freeze(self._nodes[nid]))

    def iter_edges(self) -> Iterator[EdgeView]:
        for src in sorted(self._out.keys()):
            for dst in sorted(self._out[src].keys()):
                yield EdgeView(
                    source_id=src,
                    target_id=dst,
                    attrs=_freeze(self._out[src][dst]),
                )

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and node_id in self._nodes

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self._out.values())

    # ------------------------------------------------------------------
    # Serialisation.
    # ------------------------------------------------------------------

    def serialize(self) -> bytes:
        """INV-15 byte-stable JSON projection."""
        nodes = [
            {"id": n.node_id, "attrs": dict(sorted(n.attrs.items()))} for n in self.iter_nodes()
        ]
        edges = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "attrs": dict(sorted(e.attrs.items())),
            }
            for e in self.iter_edges()
        ]
        payload = {
            "version": CAUSAL_GRAPH_VERSION,
            "nodes": nodes,
            "edges": edges,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, blob: bytes) -> CausalGraph:
        """Reconstruct a graph from :meth:`serialize` output."""
        if not isinstance(blob, bytes | bytearray):
            raise CausalGraphError("serialize blob must be bytes")
        try:
            payload = json.loads(blob.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CausalGraphError(f"corrupt blob: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise CausalGraphError("blob root must be an object")
        if payload.get("version") != CAUSAL_GRAPH_VERSION:
            raise CausalGraphError(f"unsupported version: {payload.get('version')!r}")
        graph = cls()
        for raw_node in payload.get("nodes", ()):
            graph.add_node(str(raw_node["id"]), **dict(raw_node.get("attrs", {})))
        for raw_edge in payload.get("edges", ()):
            graph.add_edge(
                str(raw_edge["source"]),
                str(raw_edge["target"]),
                **dict(raw_edge.get("attrs", {})),
            )
        return graph

    # ------------------------------------------------------------------
    # networkx export (lazy import).
    # ------------------------------------------------------------------

    def networkx_export(self) -> Any:
        """Return an equivalent ``networkx.DiGraph`` (lazy import).

        Spec line 1228 requires runtime reads to be in pure Python;
        callers that want to plug the graph into the broader networkx
        algorithm suite can call this method to materialise a real
        ``DiGraph``. The ``networkx`` import is confined to this
        function body so the rest of the module remains importable
        without the dep.
        """
        try:
            import networkx  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised when dep absent
            raise CausalGraphError("networkx is not installed; see NEW_PIP_DEPENDENCIES") from exc
        g = networkx.DiGraph()
        for n in self.iter_nodes():
            g.add_node(n.node_id, **dict(n.attrs))
        for e in self.iter_edges():
            g.add_edge(e.source_id, e.target_id, **dict(e.attrs))
        return g

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _reachable(self, src: str, dst: str) -> bool:
        """``True`` iff ``dst`` is reachable from ``src`` via forward edges."""
        if src == dst:
            return True
        if src not in self._out:
            return False
        seen: set[str] = set()
        stack = [src]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for nxt in self._out.get(current, {}):
                if nxt == dst:
                    return True
                stack.append(nxt)
        return False

    @staticmethod
    def _reconstruct_path(
        predecessor: Mapping[str, str], source: str, target: str
    ) -> tuple[str, ...]:
        chain: list[str] = [target]
        current = target
        while current != source:
            current = predecessor[current]
            chain.append(current)
        chain.reverse()
        return tuple(chain)

    @staticmethod
    def _sorted_insert(seq: list[str], value: str) -> None:
        """Insert ``value`` into ``seq`` keeping the list sorted."""
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        seq.insert(lo, value)

    def _require_node_exists(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"node not found: {node_id!r}")

    def _require_edge_exists(self, source_id: str, target_id: str) -> None:
        if not self.has_edge(source_id, target_id):
            raise KeyError(f"edge not found: {source_id!r}->{target_id!r}")

    @staticmethod
    def _require_nonempty_str(name: str, value: Any) -> None:
        if not isinstance(value, str) or not value:
            raise CausalGraphError(f"{name} must be a non-empty str")
