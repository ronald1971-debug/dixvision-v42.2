"""A-11 — Cypher query constants for :mod:`state.knowledge_graph`.

# ADAPTED FROM: neo4j/neo4j-python-driver
#   - neo4j/sync/driver.py (Session.run patterns)
#   - neo4j/graph/graph.py (Node / Relationship surface)
# Apache-2.0 license; only the query *shapes* and call-site contract are
# reproduced — no neo4j source code is copied into this repository.

This module is intentionally a flat namespace of named Cypher templates.
Spec line 1197 mandates that all Cypher queries live in this query file
and are **never** inlined inside :mod:`state.knowledge_graph` logic.

Every query is a parameterised template (``$param``). Callers bind
parameters via :class:`state.knowledge_graph.GraphTransport.run` rather
than f-string interpolation; this preserves the parameter / value-object
separation neo4j enforces server-side and keeps the queries reusable
across the live neo4j transport and the offline NetworkX fallback.

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

# Tuple of every query exposed by this module — pinned by the AST guard
# in :mod:`tests.test_knowledge_graph` so the symbol surface cannot
# drift accidentally.
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
