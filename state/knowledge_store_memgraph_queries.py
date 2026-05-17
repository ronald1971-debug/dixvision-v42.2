"""C-22 — openCypher query constants for :mod:`state.knowledge_store_memgraph`.

# ADAPTED FROM: memgraph/gqlalchemy
#   - gqlalchemy/memgraph.py (Memgraph connection, execute_query)
#   - openCypher 9 specification (same parameterised-template surface as neo4j)
# Apache-2.0 license; only the query *shapes* and call-site contract are
# reproduced — no gqlalchemy source code is copied into this repository.

This module is intentionally a flat namespace of named openCypher templates.
Memgraph implements the same parameterised-query surface as Neo4j
(``MERGE … SET … RETURN``), so these templates mirror
:mod:`state.knowledge_graph_queries` with only name-reference changes
(``gqlalchemy`` vs ``neo4j``).

All parameters use the ``$param`` binding convention. Callers bind
parameters via :class:`state.knowledge_store_memgraph.MemgraphTransport.run`
rather than f-string interpolation.

INV-15: queries are pure string constants — no clock reads, no random
state, no dict iteration order.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Node ``MERGE`` queries (idempotent upserts).
# ---------------------------------------------------------------------------

MERGE_STRATEGY: Final[str] = (
    "MERGE (s:Strategy {id: $strategy_id}) "
    "SET s.version = $version, "
    "    s.lifecycle = $lifecycle, "
    "    s.last_ts_ns = $ts_ns "
    "RETURN s.id AS id"
)
"""Upsert a :class:`Strategy` node keyed by ``id``."""

MERGE_REGIME: Final[str] = (
    "MERGE (r:Regime {id: $regime_id}) "
    "SET r.label = $label, "
    "    r.last_ts_ns = $ts_ns "
    "RETURN r.id AS id"
)
"""Upsert a :class:`Regime` node keyed by ``id``."""

MERGE_FAILURE: Final[str] = (
    "MERGE (f:Failure {id: $failure_id}) "
    "SET f.kind = $kind, "
    "    f.severity = $severity, "
    "    f.ts_ns = $ts_ns "
    "RETURN f.id AS id"
)
"""Upsert a :class:`Failure` node keyed by ``id``."""

# ---------------------------------------------------------------------------
# Edge ``MERGE`` queries (idempotent causal edges).
# ---------------------------------------------------------------------------

MERGE_CAUSED_BY: Final[str] = (
    "MATCH (src {id: $src_id}), (dst {id: $dst_id}) "
    "MERGE (src)-[e:CAUSED_BY]->(dst) "
    "SET e.weight = $weight, e.ts_ns = $ts_ns "
    "RETURN type(e) AS rel"
)
"""Upsert a ``CAUSED_BY`` edge between two existing nodes."""

# ---------------------------------------------------------------------------
# Read-only traversal queries.
# ---------------------------------------------------------------------------

FETCH_NODE: Final[str] = (
    "MATCH (n {id: $node_id}) RETURN labels(n)[0] AS label, properties(n) AS props"
)
"""Read a node by ``id``; returns ``(label, props)``."""

FETCH_OUTGOING: Final[str] = (
    "MATCH (n {id: $node_id})-[e]->(m) "
    "RETURN type(e) AS rel, m.id AS target, properties(e) AS edge_props "
    "ORDER BY target ASC"
)
"""Outgoing edges of a node, deterministically ordered."""

FETCH_INCOMING: Final[str] = (
    "MATCH (n {id: $node_id})<-[e]-(m) "
    "RETURN type(e) AS rel, m.id AS source, properties(e) AS edge_props "
    "ORDER BY source ASC"
)
"""Incoming edges of a node, deterministically ordered."""

FETCH_CAUSAL_CHAIN: Final[str] = (
    "MATCH path = (src {id: $src_id})-[:CAUSED_BY*1..$max_depth]->(dst) "
    "RETURN [n IN nodes(path) | n.id] AS chain "
    "ORDER BY length(path) ASC, chain ASC"
)
"""All ``CAUSED_BY`` chains rooted at ``src_id`` up to ``max_depth`` hops."""

FETCH_ALL_NODES: Final[str] = (
    "MATCH (n) RETURN labels(n)[0] AS label, n.id AS id, properties(n) AS props "
    "ORDER BY label ASC, id ASC"
)
"""All nodes in the graph, deterministically ordered."""

FETCH_ALL_EDGES: Final[str] = (
    "MATCH (s)-[e]->(t) "
    "RETURN type(e) AS rel, s.id AS source, t.id AS target, properties(e) AS edge_props "
    "ORDER BY source ASC, target ASC, rel ASC"
)
"""All edges in the graph, deterministically ordered."""

ALL_QUERIES: Final[tuple[str, ...]] = (
    MERGE_STRATEGY,
    MERGE_REGIME,
    MERGE_FAILURE,
    MERGE_CAUSED_BY,
    FETCH_NODE,
    FETCH_OUTGOING,
    FETCH_INCOMING,
    FETCH_CAUSAL_CHAIN,
    FETCH_ALL_NODES,
    FETCH_ALL_EDGES,
)
