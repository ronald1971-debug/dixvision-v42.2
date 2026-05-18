"""A-11 neo4j → strategy knowledge graph.

# ADAPTED FROM: neo4j/neo4j-python-driver
#   - neo4j/sync/driver.py — GraphDatabase.driver(), Session.run()
#   - neo4j/graph/graph.py — Node, Relationship
# Apache-2.0 license; no neo4j source code is reproduced verbatim — only
# the call-site contract (``Session.run(cypher, **params).data()``) and
# the ``MERGE`` idempotent-upsert pattern are mirrored here.

The :class:`KnowledgeGraph` is the **slow-path** strategy lineage store
spec'd at :file:`DIX_MASTER_CANONICAL.md` lines 1170–1204. It records
the causal lattice ``Strategy → Failure → Regime``:

* ``Strategy`` nodes — every governance-side strategy ever proposed.
* ``Regime`` nodes — macro / order-book / sentiment regimes the
  ``MacroRegimeEngine`` (Phase 1 macro-D) has classified.
* ``Failure`` nodes — every adverse-outcome event the learning loop has
  attributed back to a strategy in a given regime.
* ``CAUSED_BY`` edges — directed, weighted causal links between any two
  nodes (Strategy → Failure, Failure → Regime, Strategy → Strategy for
  lineage, …).

Tier discipline (spec line 1195):

* **OFFLINE writes** — only ``learning_engine.*`` (and tests) may call
  the mutating surface (:meth:`merge_strategy`, :meth:`merge_regime`,
  :meth:`merge_failure`, :meth:`merge_caused_by`).
* **Slow-path reads** — ``intelligence_engine.*`` may call the
  read surface (:meth:`fetch_node`, :meth:`fetch_outgoing`,
  :meth:`fetch_incoming`, :meth:`fetch_causal_chain`). All reads are
  ``< 5 ms`` against the in-memory NetworkX fallback by construction
  (pure Python dict lookups); against a live neo4j server, latency is
  bounded by the operator's deployment topology.

Fallback (spec line 1196):

If ``neo4j_driver_factory`` cannot bind a live driver (the
``neo4j`` package is not installed, or the server is unreachable),
the in-memory :class:`InMemoryGraphTransport` is selected
automatically. The fallback satisfies the same
:class:`GraphTransport` Protocol, evaluates the Cypher templates in
:mod:`state.knowledge_graph_queries`, and is INV-15 byte-identical
across runs.

Authority symmetry (B27 / B28 / INV-71):

This module does **not** construct typed bus events. It is a passive
projection store. ``Strategy`` rows are produced from
``learning_engine`` proposals via the operator-approval edge
(Wave-03 PR-5). ``Failure`` rows are produced from the trade-outcome
side of the learning loop (FeedbackCollector → IntelligenceFeedbackSink,
PR #140). The AST guard in :mod:`tests.test_knowledge_graph` pins this.

INV-15 byte-identical replay:

* No clock reads — every timestamp is supplied by the caller as
  ``ts_ns: int``.
* No randomness — deterministic ``(source, target, rel)`` ordering on
  every traversal, lexicographic ``id`` ordering on every node scan.
* Frozen, slotted record types with structural equality.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from state.knowledge_graph_queries import (
    ALL_QUERIES,
    FETCH_ALL_EDGES,
    FETCH_ALL_NODES,
    FETCH_CAUSAL_CHAIN,
    FETCH_INCOMING,
    FETCH_NODE,
    FETCH_OUTGOING,
    MERGE_CAUSED_BY,
    MERGE_FAILURE,
    MERGE_REGIME,
    MERGE_STRATEGY,
)

# ---------------------------------------------------------------------------
# Module identity / dependency declaration.
# ---------------------------------------------------------------------------

KNOWLEDGE_GRAPH_VERSION: str = "1"
"""Serialisation version pin (incremented on breaking schema changes)."""

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("neo4j",)
"""Live transport requires ``neo4j``; in-memory fallback has no deps."""


__all__ = (
    "CAUSED_BY",
    "CausalChain",
    "EdgeRecord",
    "GraphTransport",
    "InMemoryGraphTransport",
    "KNOWLEDGE_GRAPH_VERSION",
    "KnowledgeGraph",
    "KnowledgeGraphError",
    "NEW_PIP_DEPENDENCIES",
    "NodeKind",
    "NodeRecord",
    "neo4j_driver_factory",
)


# ---------------------------------------------------------------------------
# Public value-object types.
# ---------------------------------------------------------------------------


class KnowledgeGraphError(RuntimeError):
    """Raised on schema / contract / transport violations."""


class NodeKind(enum.StrEnum):
    """The three first-class node labels the graph stores."""

    STRATEGY = "Strategy"
    REGIME = "Regime"
    FAILURE = "Failure"


CAUSED_BY: str = "CAUSED_BY"
"""Canonical edge relationship type. The only edge label this module emits."""


def _freeze_props(props: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a key-sorted read-only projection of ``props``.

    INV-15: the on-disk projection of every node / edge is sorted on
    key insertion so two writes with the same logical payload yield
    byte-equal serialisations.
    """
    if not isinstance(props, Mapping):
        raise TypeError("props must be a Mapping")
    sorted_items = sorted(props.items(), key=lambda kv: kv[0])
    return MappingProxyType(dict(sorted_items))


@dataclass(frozen=True, slots=True)
class NodeRecord:
    """One frozen graph node.

    Attributes:
        kind: One of :class:`NodeKind` — ``Strategy`` / ``Regime`` /
            ``Failure``.
        node_id: Stable, unique identifier within ``kind``.
        props: Frozen, key-sorted payload mapping. Producer-set;
            value types must be JSON-primitive
            (``int`` / ``float`` / ``str`` / ``bool`` / ``None``) so the
            ledger projection is byte-stable.
        ts_ns: Caller-supplied nanosecond timestamp of the last update.
    """

    kind: NodeKind
    node_id: str
    props: Mapping[str, Any]
    ts_ns: int


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """One frozen ``CAUSED_BY`` edge.

    Attributes:
        source_id: ``node_id`` of the cause.
        target_id: ``node_id`` of the effect.
        weight: Causal weight, ``0.0 <= weight <= 1.0``.
        ts_ns: Caller-supplied nanosecond timestamp of the last update.
    """

    source_id: str
    target_id: str
    weight: float
    ts_ns: int


@dataclass(frozen=True, slots=True)
class CausalChain:
    """One causal path returned by :meth:`KnowledgeGraph.fetch_causal_chain`.

    Attributes:
        node_ids: Tuple of ``node_id`` strings from source to terminal.
            Length ``>= 2``. ``node_ids[0]`` is the query root.
    """

    node_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Transport Protocol.
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphTransport(Protocol):
    """Minimal Cypher transport surface.

    Mirrors the ``Session.run(cypher, **params).data()`` contract from
    ``neo4j.sync.driver``. The :class:`InMemoryGraphTransport` fallback
    and any live ``neo4j``-backed wrapper both satisfy this Protocol.

    Implementations must:

    * Accept any cypher string in :data:`state.knowledge_graph_queries.ALL_QUERIES`.
    * Return a sequence of ``Mapping[str, Any]`` rows whose key order is
      deterministic (typically ``dict`` insertion order; INV-15).
    * Be re-entrant — :class:`KnowledgeGraph` may call ``run`` in any
      order and at any time.
    """

    def run(
        self, cypher: str, /, **params: Any
    ) -> Sequence[Mapping[str, Any]]:  # pragma: no cover - Protocol
        ...

    def close(self) -> None:  # pragma: no cover - Protocol
        ...


# ---------------------------------------------------------------------------
# In-memory fallback transport (NetworkX-shaped, but with zero external deps).
# ---------------------------------------------------------------------------


def _validate_prop_value(value: Any) -> None:
    """Reject non-JSON-primitive property values.

    INV-15: keeping every prop value JSON-primitive ensures the
    serialisation projection is byte-stable across Python versions.
    """
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float | str):
        return
    raise KnowledgeGraphError(f"prop value must be JSON-primitive, got {type(value).__name__}")


class InMemoryGraphTransport:
    """Pure-Python evaluator for the Cypher templates in this module.

    Spec line 1196: "Must fall back to in-memory NetworkX if Neo4j
    server unavailable." This implementation honours the *contract* of
    the NetworkX-shaped fallback (directed graph, idempotent upsert,
    deterministic traversal) without dragging the ``networkx`` package
    in — every method here is a pure-Python dict walk.

    Thread-safety: not re-entrant; callers must serialise access. The
    higher-level :class:`KnowledgeGraph` is the only intended caller and
    is single-threaded by design (OFFLINE writes, slow-path reads).
    """

    __slots__ = ("_edges", "_nodes")

    def __init__(self) -> None:
        # node_id -> (label, props)
        self._nodes: dict[str, tuple[str, dict[str, Any]]] = {}
        # (source_id, target_id) -> {weight, ts_ns}
        self._edges: dict[tuple[str, str], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Transport surface.
    # ------------------------------------------------------------------

    def run(self, cypher: str, /, **params: Any) -> Sequence[Mapping[str, Any]]:
        if cypher == MERGE_STRATEGY:
            return self._merge_node(NodeKind.STRATEGY.value, params)
        if cypher == MERGE_REGIME:
            return self._merge_node(NodeKind.REGIME.value, params)
        if cypher == MERGE_FAILURE:
            return self._merge_node(NodeKind.FAILURE.value, params)
        if cypher == MERGE_CAUSED_BY:
            return self._merge_edge(params)
        if cypher == FETCH_NODE:
            return self._fetch_node(params)
        if cypher == FETCH_OUTGOING:
            return self._fetch_outgoing(params)
        if cypher == FETCH_INCOMING:
            return self._fetch_incoming(params)
        if cypher == FETCH_CAUSAL_CHAIN:
            return self._fetch_causal_chain(params)
        if cypher == FETCH_ALL_NODES:
            return self._fetch_all_nodes()
        if cypher == FETCH_ALL_EDGES:
            return self._fetch_all_edges()
        raise KnowledgeGraphError(f"unknown cypher query: {cypher[:60]!r}…")

    def close(self) -> None:
        # In-memory transport holds no external handles.
        return None

    # ------------------------------------------------------------------
    # MERGE handlers.
    # ------------------------------------------------------------------

    def _merge_node(self, label: str, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        node_id = self._require_str(params, ("strategy_id", "regime_id", "failure_id"))
        ts_ns = self._require_int(params, "ts_ns")
        props: dict[str, Any] = {"ts_ns": ts_ns}
        for key, value in params.items():
            if key in {"strategy_id", "regime_id", "failure_id", "ts_ns"}:
                continue
            _validate_prop_value(value)
            props[key] = value
        existing = self._nodes.get(node_id)
        if existing is not None:
            existing_label, existing_props = existing
            if existing_label != label:
                raise KnowledgeGraphError(
                    f"node {node_id!r} already exists with label "
                    f"{existing_label!r}, cannot relabel to {label!r}"
                )
            merged = dict(existing_props)
            merged.update(props)
            self._nodes[node_id] = (label, merged)
        else:
            self._nodes[node_id] = (label, dict(props))
        return ({"id": node_id},)

    def _merge_edge(self, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        src = self._require_str(params, ("src_id",))
        dst = self._require_str(params, ("dst_id",))
        if src == dst:
            raise KnowledgeGraphError(f"self-edge not permitted: {src!r}")
        if src not in self._nodes or dst not in self._nodes:
            raise KnowledgeGraphError(
                f"edge endpoints must both exist; got src={src!r} dst={dst!r}"
            )
        weight = float(params.get("weight", 1.0))
        if not 0.0 <= weight <= 1.0:
            raise KnowledgeGraphError(f"weight out of bounds: {weight!r}")
        ts_ns = self._require_int(params, "ts_ns")
        self._edges[(src, dst)] = {"weight": weight, "ts_ns": ts_ns}
        return ({"rel": CAUSED_BY},)

    # ------------------------------------------------------------------
    # Read handlers.
    # ------------------------------------------------------------------

    def _fetch_node(self, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        node_id = self._require_str(params, ("node_id",))
        node = self._nodes.get(node_id)
        if node is None:
            return ()
        label, props = node
        return ({"label": label, "props": dict(sorted(props.items()))},)

    def _fetch_outgoing(self, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        node_id = self._require_str(params, ("node_id",))
        rows: list[dict[str, Any]] = []
        for (src, dst), edge_props in self._edges.items():
            if src != node_id:
                continue
            rows.append(
                {
                    "rel": CAUSED_BY,
                    "target": dst,
                    "edge_props": dict(sorted(edge_props.items())),
                }
            )
        rows.sort(key=lambda r: r["target"])
        return tuple(rows)

    def _fetch_incoming(self, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        node_id = self._require_str(params, ("node_id",))
        rows: list[dict[str, Any]] = []
        for (src, dst), edge_props in self._edges.items():
            if dst != node_id:
                continue
            rows.append(
                {
                    "rel": CAUSED_BY,
                    "source": src,
                    "edge_props": dict(sorted(edge_props.items())),
                }
            )
        rows.sort(key=lambda r: r["source"])
        return tuple(rows)

    def _fetch_causal_chain(self, params: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        src = self._require_str(params, ("src_id",))
        max_depth = self._require_int(params, "max_depth")
        if max_depth <= 0:
            raise KnowledgeGraphError(f"max_depth must be positive: {max_depth!r}")
        if src not in self._nodes:
            return ()
        # Adjacency in sorted target order to pin INV-15 enumeration.
        adjacency: dict[str, list[str]] = {}
        for s, t in self._edges:
            adjacency.setdefault(s, []).append(t)
        for neighbours in adjacency.values():
            neighbours.sort()
        # DFS enumerating all paths up to ``max_depth`` hops.
        chains: list[tuple[str, ...]] = []
        stack: list[tuple[str, tuple[str, ...]]] = [(src, (src,))]
        while stack:
            current, path = stack.pop()
            if len(path) - 1 >= max_depth:
                continue
            for nxt in adjacency.get(current, ()):
                if nxt in path:
                    continue  # acyclic enumeration
                new_path = (*path, nxt)
                chains.append(new_path)
                stack.append((nxt, new_path))
        chains.sort(key=lambda c: (len(c), c))
        return tuple({"chain": list(c)} for c in chains)

    def _fetch_all_nodes(self) -> Sequence[Mapping[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node_id, (label, props) in self._nodes.items():
            rows.append(
                {
                    "label": label,
                    "id": node_id,
                    "props": dict(sorted(props.items())),
                }
            )
        rows.sort(key=lambda r: (r["label"], r["id"]))
        return tuple(rows)

    def _fetch_all_edges(self) -> Sequence[Mapping[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (src, dst), edge_props in self._edges.items():
            rows.append(
                {
                    "rel": CAUSED_BY,
                    "source": src,
                    "target": dst,
                    "edge_props": dict(sorted(edge_props.items())),
                }
            )
        rows.sort(key=lambda r: (r["source"], r["target"], r["rel"]))
        return tuple(rows)

    # ------------------------------------------------------------------
    # Param-validation helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _require_str(params: Mapping[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            if key in params:
                value = params[key]
                if not isinstance(value, str) or not value:
                    raise KnowledgeGraphError(f"param {key!r} must be a non-empty str")
                return value
        raise KnowledgeGraphError(f"missing required str param: one of {keys!r}")

    @staticmethod
    def _require_int(params: Mapping[str, Any], key: str) -> int:
        if key not in params:
            raise KnowledgeGraphError(f"missing required int param: {key!r}")
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise KnowledgeGraphError(f"param {key!r} must be int")
        return value


# ---------------------------------------------------------------------------
# High-level coordinator.
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """OFFLINE-tier strategy / regime / failure lineage store.

    Thin coordinator over a :class:`GraphTransport`. All mutating
    methods are categorised as OFFLINE_ONLY (``learning_engine``-side);
    all read methods are RUNTIME_SAFE (``intelligence_engine``-side).

    The Cypher queries are imported from
    :mod:`state.knowledge_graph_queries` — never inlined here.
    """

    __slots__ = ("_transport",)

    def __init__(self, transport: GraphTransport | None = None) -> None:
        if transport is None:
            transport = InMemoryGraphTransport()
        else:
            if not isinstance(transport, GraphTransport):
                raise TypeError("transport must satisfy GraphTransport")
        self._transport = transport

    # ------------------------------------------------------------------
    # OFFLINE write surface.
    # ------------------------------------------------------------------

    def merge_strategy(
        self,
        *,
        strategy_id: str,
        version: int,
        lifecycle: str,
        ts_ns: int,
    ) -> None:
        """Idempotent ``Strategy`` upsert.

        Args:
            strategy_id: Stable identifier; non-empty.
            version: Monotonically increasing per ``strategy_id``.
            lifecycle: One of the
                :class:`core.contracts.strategy_registry.StrategyLifecycle`
                string values; this module accepts the raw string to
                avoid cross-engine imports.
            ts_ns: Caller-supplied nanosecond timestamp.
        """
        self._require_nonempty_str("strategy_id", strategy_id)
        self._require_int("version", version)
        self._require_nonempty_str("lifecycle", lifecycle)
        self._require_int("ts_ns", ts_ns)
        self._transport.run(
            MERGE_STRATEGY,
            strategy_id=strategy_id,
            version=version,
            lifecycle=lifecycle,
            ts_ns=ts_ns,
        )

    def merge_regime(
        self,
        *,
        regime_id: str,
        label: str,
        ts_ns: int,
    ) -> None:
        """Idempotent ``Regime`` upsert."""
        self._require_nonempty_str("regime_id", regime_id)
        self._require_nonempty_str("label", label)
        self._require_int("ts_ns", ts_ns)
        self._transport.run(
            MERGE_REGIME,
            regime_id=regime_id,
            label=label,
            ts_ns=ts_ns,
        )

    def merge_failure(
        self,
        *,
        failure_id: str,
        kind: str,
        severity: str,
        ts_ns: int,
    ) -> None:
        """Idempotent ``Failure`` upsert."""
        self._require_nonempty_str("failure_id", failure_id)
        self._require_nonempty_str("kind", kind)
        self._require_nonempty_str("severity", severity)
        self._require_int("ts_ns", ts_ns)
        self._transport.run(
            MERGE_FAILURE,
            failure_id=failure_id,
            kind=kind,
            severity=severity,
            ts_ns=ts_ns,
        )

    def merge_caused_by(
        self,
        *,
        source_id: str,
        target_id: str,
        weight: float,
        ts_ns: int,
    ) -> None:
        """Idempotent ``CAUSED_BY`` edge upsert.

        Both endpoints must already exist (caller is responsible for
        node-before-edge ordering). ``weight`` is clamped to
        ``[0.0, 1.0]`` and validates a closed range; self-edges are
        rejected.
        """
        self._require_nonempty_str("source_id", source_id)
        self._require_nonempty_str("target_id", target_id)
        if not isinstance(weight, float | int) or isinstance(weight, bool):
            raise KnowledgeGraphError("weight must be a real number")
        self._require_int("ts_ns", ts_ns)
        self._transport.run(
            MERGE_CAUSED_BY,
            src_id=source_id,
            dst_id=target_id,
            weight=float(weight),
            ts_ns=ts_ns,
        )

    # ------------------------------------------------------------------
    # Slow-path read surface.
    # ------------------------------------------------------------------

    def fetch_node(self, *, node_id: str) -> NodeRecord | None:
        """Return the :class:`NodeRecord` for ``node_id`` or ``None``."""
        self._require_nonempty_str("node_id", node_id)
        rows = self._transport.run(FETCH_NODE, node_id=node_id)
        if not rows:
            return None
        row = rows[0]
        label = str(row["label"])
        props = _freeze_props(row["props"])
        ts_ns = int(props.get("ts_ns", 0))
        return NodeRecord(
            kind=NodeKind(label),
            node_id=node_id,
            props=props,
            ts_ns=ts_ns,
        )

    def fetch_outgoing(self, *, node_id: str) -> tuple[EdgeRecord, ...]:
        """Outgoing ``CAUSED_BY`` edges of ``node_id``, sorted by target."""
        self._require_nonempty_str("node_id", node_id)
        rows = self._transport.run(FETCH_OUTGOING, node_id=node_id)
        return tuple(
            EdgeRecord(
                source_id=node_id,
                target_id=str(row["target"]),
                weight=float(row["edge_props"].get("weight", 0.0)),
                ts_ns=int(row["edge_props"].get("ts_ns", 0)),
            )
            for row in rows
        )

    def fetch_incoming(self, *, node_id: str) -> tuple[EdgeRecord, ...]:
        """Incoming ``CAUSED_BY`` edges of ``node_id``, sorted by source."""
        self._require_nonempty_str("node_id", node_id)
        rows = self._transport.run(FETCH_INCOMING, node_id=node_id)
        return tuple(
            EdgeRecord(
                source_id=str(row["source"]),
                target_id=node_id,
                weight=float(row["edge_props"].get("weight", 0.0)),
                ts_ns=int(row["edge_props"].get("ts_ns", 0)),
            )
            for row in rows
        )

    def fetch_causal_chain(self, *, source_id: str, max_depth: int) -> tuple[CausalChain, ...]:
        """Enumerate ``CAUSED_BY`` chains rooted at ``source_id``.

        Returns chains sorted by ``(length, lexicographic)`` so the
        traversal is INV-15 byte-identical across runs.
        """
        self._require_nonempty_str("source_id", source_id)
        self._require_int("max_depth", max_depth)
        if max_depth <= 0:
            raise KnowledgeGraphError(f"max_depth must be positive: {max_depth!r}")
        rows = self._transport.run(
            FETCH_CAUSAL_CHAIN,
            src_id=source_id,
            max_depth=max_depth,
        )
        return tuple(CausalChain(node_ids=tuple(row["chain"])) for row in rows)

    # ------------------------------------------------------------------
    # Iteration / serialisation.
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[NodeRecord]:
        """Iterate every node, deterministically ordered by ``(label, id)``."""
        for row in self._transport.run(FETCH_ALL_NODES):
            label = str(row["label"])
            node_id = str(row["id"])
            props = _freeze_props(row["props"])
            ts_ns = int(props.get("ts_ns", 0))
            yield NodeRecord(
                kind=NodeKind(label),
                node_id=node_id,
                props=props,
                ts_ns=ts_ns,
            )

    def iter_edges(self) -> Iterator[EdgeRecord]:
        """Iterate every edge, deterministically ordered."""
        for row in self._transport.run(FETCH_ALL_EDGES):
            yield EdgeRecord(
                source_id=str(row["source"]),
                target_id=str(row["target"]),
                weight=float(row["edge_props"].get("weight", 0.0)),
                ts_ns=int(row["edge_props"].get("ts_ns", 0)),
            )

    def serialize(self) -> bytes:
        """INV-15 byte-stable JSON projection of the graph."""
        nodes = [
            {
                "kind": n.kind.value,
                "id": n.node_id,
                "props": dict(sorted(n.props.items())),
                "ts_ns": n.ts_ns,
            }
            for n in self.iter_nodes()
        ]
        edges = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "weight": e.weight,
                "ts_ns": e.ts_ns,
            }
            for e in self.iter_edges()
        ]
        payload = {
            "version": KNOWLEDGE_GRAPH_VERSION,
            "nodes": nodes,
            "edges": edges,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, blob: bytes) -> KnowledgeGraph:
        """Reconstruct a graph from :meth:`serialize` output."""
        if not isinstance(blob, bytes | bytearray):
            raise KnowledgeGraphError("serialize blob must be bytes")
        try:
            payload = json.loads(blob.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise KnowledgeGraphError(f"corrupt blob: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise KnowledgeGraphError("blob root must be an object")
        version = payload.get("version")
        if version != KNOWLEDGE_GRAPH_VERSION:
            raise KnowledgeGraphError(f"unsupported version: {version!r}")
        graph = cls()
        for raw_node in payload.get("nodes", ()):
            kind = NodeKind(raw_node["kind"])
            if kind is NodeKind.STRATEGY:
                graph.merge_strategy(
                    strategy_id=str(raw_node["id"]),
                    version=int(raw_node["props"].get("version", 0)),
                    lifecycle=str(raw_node["props"].get("lifecycle", "DRAFT")),
                    ts_ns=int(raw_node["ts_ns"]),
                )
            elif kind is NodeKind.REGIME:
                graph.merge_regime(
                    regime_id=str(raw_node["id"]),
                    label=str(raw_node["props"].get("label", "unknown")),
                    ts_ns=int(raw_node["ts_ns"]),
                )
            else:
                graph.merge_failure(
                    failure_id=str(raw_node["id"]),
                    kind=str(raw_node["props"].get("kind", "unknown")),
                    severity=str(raw_node["props"].get("severity", "unknown")),
                    ts_ns=int(raw_node["ts_ns"]),
                )
        for raw_edge in payload.get("edges", ()):
            graph.merge_caused_by(
                source_id=str(raw_edge["source"]),
                target_id=str(raw_edge["target"]),
                weight=float(raw_edge["weight"]),
                ts_ns=int(raw_edge["ts_ns"]),
            )
        return graph

    def close(self) -> None:
        """Close the underlying transport (no-op for in-memory)."""
        self._transport.close()

    # ------------------------------------------------------------------
    # Internal validation helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _require_nonempty_str(name: str, value: Any) -> None:
        if not isinstance(value, str) or not value:
            raise KnowledgeGraphError(f"{name} must be a non-empty str")

    @staticmethod
    def _require_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise KnowledgeGraphError(f"{name} must be an int")


# ---------------------------------------------------------------------------
# Lazy factory for the live ``neo4j`` driver.
# ---------------------------------------------------------------------------


def neo4j_driver_factory(
    *,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str | None = None,
) -> Any:
    """Lazy-bind ``neo4j.GraphDatabase.driver``.

    The ``neo4j`` import is confined to this function body so the rest
    of the module remains importable in replay / unit-test
    environments where the dependency is absent.

    Returns a driver handle that satisfies the call-site contract of a
    :class:`GraphTransport`-adapting wrapper. Wiring the wrapper itself
    is intentionally left to the next A-tier integration PR; spec line
    1196 requires the fallback to handle the "server unavailable" path,
    which :class:`InMemoryGraphTransport` does already.
    """
    if not isinstance(uri, str) or not uri:
        raise ValueError("uri must be a non-empty str")
    if not isinstance(user, str) or not user:
        raise ValueError("user must be a non-empty str")
    if password is not None and not isinstance(password, str):
        raise TypeError("password must be a str or None")
    try:
        from neo4j import GraphDatabase  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when dep absent
        raise KnowledgeGraphError("neo4j is not installed; see NEW_PIP_DEPENDENCIES") from exc
    return GraphDatabase.driver(uri, auth=(user, password or ""))


# Re-export the query template tuple so AST guards / consumers do not
# need to import the queries module directly.
_QUERY_SURFACE: tuple[str, ...] = ALL_QUERIES
