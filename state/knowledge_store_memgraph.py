"""C-22 gqlalchemy → Memgraph streaming-graph knowledge store.

# ADAPTED FROM: memgraph/gqlalchemy
#   - gqlalchemy/memgraph.py — Memgraph(), execute_and_fetch()
#   - gqlalchemy/models.py  — Node, Relationship
# Apache-2.0 license; no gqlalchemy source code is reproduced verbatim — only
# the call-site contract (``Memgraph().execute_and_fetch(cypher, params)``)
# and the ``MERGE`` idempotent-upsert pattern are mirrored here.

The :class:`MemgraphKnowledgeStore` is the streaming-graph alternative to
the A-11 Neo4j-backed :class:`state.knowledge_graph.KnowledgeGraph`.
Both stores share the identical Cypher-compatible interface
(:class:`GraphTransport` Protocol) and the same node / edge value-objects
(:class:`NodeRecord`, :class:`EdgeRecord`, :class:`CausalChain`).

Records the same causal lattice ``Strategy → Failure → Regime``:

* ``Strategy`` nodes — every governance-side strategy ever proposed.
* ``Regime`` nodes — macro / order-book / sentiment regimes from
  ``MacroRegimeEngine``.
* ``Failure`` nodes — adverse-outcome events the learning loop has
  attributed back to a strategy in a given regime.
* ``CAUSED_BY`` edges — directed, weighted causal links.

Tier discipline:

* **OFFLINE writes** — only ``learning_engine.*`` (and tests) may call
  the mutating surface.
* **Slow-path reads** — ``intelligence_engine.*`` may call the read
  surface. All reads are ``< 5 ms`` against the in-memory fallback by
  construction.

Fallback:

If ``memgraph_client_factory`` cannot bind a live client (``gqlalchemy``
is not installed or the Memgraph server is unreachable), the in-memory
:class:`InMemoryMemgraphTransport` is selected automatically. The
fallback satisfies the same :class:`MemgraphTransport` Protocol, evaluates
the openCypher templates in
:mod:`state.knowledge_store_memgraph_queries`, and is INV-15
byte-identical across runs.

Authority symmetry (B27 / B28 / INV-71):

This module does **not** construct typed bus events. It is a passive
projection store. The AST guard in :mod:`tests.test_memgraph_store` pins
this.

INV-15 byte-identical replay:

* No clock reads — every timestamp is supplied by the caller as
  ``ts_ns: int``.
* No randomness — deterministic ordering on every traversal.
* Frozen, slotted record types with structural equality.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from state.knowledge_store_memgraph_queries import (
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

MEMGRAPH_STORE_VERSION: str = "1"
"""Serialisation version pin (incremented on breaking schema changes)."""

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("gqlalchemy",)
"""Live transport requires ``gqlalchemy``; in-memory fallback has no deps."""


__all__ = (
    "CAUSED_BY",
    "CausalChain",
    "EdgeRecord",
    "InMemoryMemgraphTransport",
    "MEMGRAPH_STORE_VERSION",
    "MemgraphKnowledgeStore",
    "MemgraphKnowledgeStoreError",
    "MemgraphTransport",
    "NEW_PIP_DEPENDENCIES",
    "NodeKind",
    "NodeRecord",
    "memgraph_client_factory",
)


# ---------------------------------------------------------------------------
# Public value-object types.
# ---------------------------------------------------------------------------


class MemgraphKnowledgeStoreError(RuntimeError):
    """Raised on schema / contract / transport violations."""


class NodeKind(enum.StrEnum):
    """The three first-class node labels the graph stores."""

    STRATEGY = "Strategy"
    REGIME = "Regime"
    FAILURE = "Failure"


CAUSED_BY: str = "CAUSED_BY"
"""Canonical edge relationship type. The only edge label this module emits."""


def _freeze_props(props: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a key-sorted read-only projection of ``props``."""
    if not isinstance(props, Mapping):
        raise TypeError("props must be a Mapping")
    sorted_items = sorted(props.items(), key=lambda kv: kv[0])
    return MappingProxyType(dict(sorted_items))


@dataclass(frozen=True, slots=True)
class NodeRecord:
    """One frozen graph node."""

    kind: NodeKind
    node_id: str
    props: Mapping[str, Any]
    ts_ns: int


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """One frozen ``CAUSED_BY`` edge."""

    source_id: str
    target_id: str
    weight: float
    ts_ns: int


@dataclass(frozen=True, slots=True)
class CausalChain:
    """One causal path."""

    node_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Transport Protocol.
# ---------------------------------------------------------------------------


@runtime_checkable
class MemgraphTransport(Protocol):
    """Minimal openCypher transport surface.

    Mirrors ``Memgraph().execute_and_fetch(cypher, params)`` from
    ``gqlalchemy``. The :class:`InMemoryMemgraphTransport` fallback and
    any live ``gqlalchemy``-backed wrapper both satisfy this Protocol.
    """

    def run(
        self, cypher: str, /, **params: Any
    ) -> Sequence[Mapping[str, Any]]:  # pragma: no cover - Protocol
        ...

    def close(self) -> None:  # pragma: no cover - Protocol
        ...


# ---------------------------------------------------------------------------
# In-memory fallback transport.
# ---------------------------------------------------------------------------


def _validate_prop_value(value: Any) -> None:
    """Reject non-JSON-primitive property values."""
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float | str):
        return
    raise MemgraphKnowledgeStoreError(
        f"prop value must be JSON-primitive, got {type(value).__name__}"
    )


class InMemoryMemgraphTransport:
    """Pure-Python evaluator for the openCypher templates.

    Deterministic, OFFLINE-tier fallback when ``gqlalchemy`` is absent
    or Memgraph is unreachable.
    """

    __slots__ = ("_edges", "_nodes")

    def __init__(self) -> None:
        self._nodes: dict[str, tuple[str, dict[str, Any]]] = {}
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
        raise MemgraphKnowledgeStoreError(f"unknown cypher query: {cypher[:60]!r}…")

    def close(self) -> None:
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
                raise MemgraphKnowledgeStoreError(
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
            raise MemgraphKnowledgeStoreError(f"self-edge not permitted: {src!r}")
        if src not in self._nodes or dst not in self._nodes:
            raise MemgraphKnowledgeStoreError(
                f"edge endpoints must both exist; got src={src!r} dst={dst!r}"
            )
        weight = float(params.get("weight", 1.0))
        if not 0.0 <= weight <= 1.0:
            raise MemgraphKnowledgeStoreError(f"weight out of bounds: {weight!r}")
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
            raise MemgraphKnowledgeStoreError(f"max_depth must be positive: {max_depth!r}")
        if src not in self._nodes:
            return ()
        adjacency: dict[str, list[str]] = {}
        for s, t in self._edges:
            adjacency.setdefault(s, []).append(t)
        for neighbours in adjacency.values():
            neighbours.sort()
        chains: list[tuple[str, ...]] = []
        stack: list[tuple[str, tuple[str, ...]]] = [(src, (src,))]
        while stack:
            current, path = stack.pop()
            if len(path) - 1 >= max_depth:
                continue
            for nxt in adjacency.get(current, ()):
                if nxt in path:
                    continue
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
                    raise MemgraphKnowledgeStoreError(f"param {key!r} must be a non-empty str")
                return value
        raise MemgraphKnowledgeStoreError(f"missing required str param: one of {keys!r}")

    @staticmethod
    def _require_int(params: Mapping[str, Any], key: str) -> int:
        if key not in params:
            raise MemgraphKnowledgeStoreError(f"missing required int param: {key!r}")
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise MemgraphKnowledgeStoreError(f"param {key!r} must be int")
        return value


# ---------------------------------------------------------------------------
# High-level coordinator.
# ---------------------------------------------------------------------------


class MemgraphKnowledgeStore:
    """OFFLINE-tier strategy / regime / failure lineage store (Memgraph).

    Streaming-graph alternative to A-11 Neo4j
    :class:`state.knowledge_graph.KnowledgeGraph`.
    Implements the identical Cypher-compatible interface.
    """

    __slots__ = ("_transport",)

    def __init__(self, transport: MemgraphTransport | None = None) -> None:
        if transport is None:
            transport = InMemoryMemgraphTransport()
        else:
            if not isinstance(transport, MemgraphTransport):
                raise TypeError("transport must satisfy MemgraphTransport")
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
        """Idempotent ``Strategy`` upsert."""
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
        """Idempotent ``CAUSED_BY`` edge upsert."""
        self._require_nonempty_str("source_id", source_id)
        self._require_nonempty_str("target_id", target_id)
        if not isinstance(weight, float | int) or isinstance(weight, bool):
            raise MemgraphKnowledgeStoreError("weight must be a real number")
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
        """Outgoing ``CAUSED_BY`` edges, sorted by target."""
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
        """Incoming ``CAUSED_BY`` edges, sorted by source."""
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
        """Enumerate ``CAUSED_BY`` chains rooted at ``source_id``."""
        self._require_nonempty_str("source_id", source_id)
        self._require_int("max_depth", max_depth)
        if max_depth <= 0:
            raise MemgraphKnowledgeStoreError(f"max_depth must be positive: {max_depth!r}")
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
            "version": MEMGRAPH_STORE_VERSION,
            "nodes": nodes,
            "edges": edges,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, blob: bytes) -> MemgraphKnowledgeStore:
        """Reconstruct a graph from :meth:`serialize` output."""
        if not isinstance(blob, bytes | bytearray):
            raise MemgraphKnowledgeStoreError("serialize blob must be bytes")
        try:
            payload = json.loads(blob.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise MemgraphKnowledgeStoreError(f"corrupt blob: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise MemgraphKnowledgeStoreError("blob root must be an object")
        version = payload.get("version")
        if version != MEMGRAPH_STORE_VERSION:
            raise MemgraphKnowledgeStoreError(f"unsupported version: {version!r}")
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
        """Close the underlying transport."""
        self._transport.close()

    # ------------------------------------------------------------------
    # Internal validation helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _require_nonempty_str(name: str, value: Any) -> None:
        if not isinstance(value, str) or not value:
            raise MemgraphKnowledgeStoreError(f"{name} must be a non-empty str")

    @staticmethod
    def _require_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise MemgraphKnowledgeStoreError(f"{name} must be an int")


# ---------------------------------------------------------------------------
# Lazy factory for the live ``gqlalchemy`` Memgraph client.
# ---------------------------------------------------------------------------


def memgraph_client_factory(
    *,
    host: str = "localhost",
    port: int = 7687,
    username: str = "",
    password: str = "",
) -> Any:
    """Lazy-bind ``gqlalchemy.Memgraph``.

    The ``gqlalchemy`` import is confined to this function body so the
    rest of the module remains importable in replay / unit-test
    environments where the dependency is absent.
    """
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a non-empty str")
    if not isinstance(port, int) or isinstance(port, bool):
        raise TypeError("port must be an int")
    if port < 1 or port > 65535:
        raise ValueError(f"port out of range: {port!r}")
    if not isinstance(username, str):
        raise TypeError("username must be a str")
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    try:
        from gqlalchemy import Memgraph  # noqa: PLC0415
    except ImportError as exc:
        raise MemgraphKnowledgeStoreError(
            "gqlalchemy is not installed; see NEW_PIP_DEPENDENCIES"
        ) from exc
    return Memgraph(host=host, port=port, username=username, password=password)


_QUERY_SURFACE: tuple[str, ...] = ALL_QUERIES
