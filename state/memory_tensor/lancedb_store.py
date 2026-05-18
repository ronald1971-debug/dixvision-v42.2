# ADAPTED FROM: lancedb/lancedb
#   lancedb/db.py — ``connect(uri)``;
#   lancedb/table.py — ``Table.add(data)`` / ``Table.search(query)`` /
#   ``Table.delete(where=...)`` / ``Table.create_index(...)``;
#   lancedb/query.py — Query builder ``.limit(k).where(expr).to_list()``.
#
# License: Apache-2.0.
"""Embedded zero-config semantic vector store — LanceDB backend (C-24).

Pure-Python reproduction of the LanceDB ``connect()`` + ``Table`` API
surface behind the
:class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol.
Lightweight, embedded alternative to A-10 qdrant (server) and C-23 milvus
(massive scale): no server, no daemon, no network. Persists alongside the
audit ledger on disk.

Algorithmic surface ported from lancedb:

* ``db = lancedb.connect(uri)`` →
  :func:`lancedb_connect` (lazy, factory only).
* ``table = db.create_table(name, data, schema)`` → store constructor
  (the in-memory backend treats the table as an unordered collection
  with a primary key on ``episode_id``).
* ``table.add(rows)`` → :meth:`add` / :meth:`insert`.
* ``table.search(vector).limit(k).where(expr).to_list()`` →
  :meth:`search` / :meth:`search_with_filter`.
* ``table.delete(where=...)`` → :meth:`delete`.
* ``table.create_index(metric, vector_column_name)`` →
  :meth:`create_index` (audit-only — in-memory does exact search).

Metric types mirror LanceDB's ``metric=`` argument:

* ``L2``: Euclidean L2 distance.
* ``COSINE``: cosine **distance** (``1 - cos``).
* ``DOT``: negated inner product, shifted to non-negative.

Index types mirror LanceDB's ``index_type=`` argument:

* ``IVF_PQ``: inverted-file with product quantisation — recorded only.
* ``BTREE``: scalar btree — recorded only.
* ``BITMAP``: scalar bitmap — recorded only.

The in-memory backend evaluates search exactly regardless of the chosen
index type; the recorded ``IndexParams`` are kept for audit / replay.

Authority constraints (C-24, OFFLINE_ONLY):

* **OFFLINE tier write** — :meth:`add` / :meth:`delete` /
  :meth:`create_index` are never called from the hot path.
* **RUNTIME-SAFE read** — :meth:`search` runs in <5 ms for typical N.
* **B27 / B28 / INV-71 authority symmetry** — this module does NOT
  construct typed bus events.
* **INV-15 replay determinism** — same inputs → same outputs
  byte-identical. Tie-breaking on ``(distance, ts_ns, episode_id)``;
  serialisation sorts episodes by ``(ts_ns, episode_id)``;
  :meth:`serialize` round-trips byte-equal via :meth:`deserialize`.
* **No clock, no PRNG, no IO** — every timestamp comes from caller
  supplied ``Episode.ts_ns`` / ``MemoryQuery.ts_ns``.
* **Pure stdlib** — no lancedb, no numpy. ``NEW_PIP_DEPENDENCIES =
  ("lancedb",)`` only for the lazy factory.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from state.memory_tensor.contracts import (
    Episode,
    MemoryHit,
    MemoryQuery,
    MemoryResult,
    validate_embedding,
)

# ---------------------------------------------------------------------------
# Module-level metadata
# ---------------------------------------------------------------------------
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("lancedb",)
LANCEDB_ADAPTER_VERSION: str = "1"
_SERIALIZATION_VERSION: int = 1


# ---------------------------------------------------------------------------
# Metric type enum
# ---------------------------------------------------------------------------
class MetricType(enum.Enum):
    """Mirrors LanceDB's ``metric=`` argument."""

    L2 = "l2"
    COSINE = "cosine"
    DOT = "dot"


# ---------------------------------------------------------------------------
# Index type enum
# ---------------------------------------------------------------------------
class IndexType(enum.Enum):
    """Mirrors LanceDB's ``index_type=`` argument (closed subset)."""

    IVF_PQ = "IVF_PQ"
    BTREE = "BTREE"
    BITMAP = "BITMAP"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LanceDBStoreError(RuntimeError):
    """Raised when the lancedb adapter rejects an input."""


# ---------------------------------------------------------------------------
# WhereClause — Python form of lancedb's SQL-ish ``where=`` predicate.
# ---------------------------------------------------------------------------
class WhereClause:
    """Equality / membership filter over payload fields.

    Mirrors LanceDB's ``table.search(...).where("strategy = 'alpha'")``
    surface (closed subset: equality / non-equality + AND of clauses).

    Construction is keyword-only:

    * ``equals``: every key/value pair must match exactly.
    * ``not_equals``: none of the key/value pairs may match.
    """

    __slots__ = ("_equals", "_not_equals")

    def __init__(
        self,
        *,
        equals: Mapping[str, str] | None = None,
        not_equals: Mapping[str, str] | None = None,
    ) -> None:
        self._equals = self._freeze(equals, "equals")
        self._not_equals = self._freeze(not_equals, "not_equals")

    @staticmethod
    def _freeze(m: Mapping[str, str] | None, name: str) -> tuple[tuple[str, str], ...]:
        if m is None:
            return ()
        if not isinstance(m, Mapping):
            raise TypeError(f"WhereClause.{name} must be a Mapping or None")
        pairs: list[tuple[str, str]] = []
        for key in sorted(m):
            if not isinstance(key, str):
                raise TypeError(f"WhereClause.{name} keys must be str")
            value = m[key]
            if not isinstance(value, str):
                raise TypeError(f"WhereClause.{name}[{key!r}] must be str")
            pairs.append((key, value))
        return tuple(pairs)

    @property
    def equals(self) -> tuple[tuple[str, str], ...]:
        return self._equals

    @property
    def not_equals(self) -> tuple[tuple[str, str], ...]:
        return self._not_equals

    def matches(self, payload: Mapping[str, str]) -> bool:
        for key, value in self._equals:
            if payload.get(key) != value:
                return False
        for key, value in self._not_equals:
            if payload.get(key) == value:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WhereClause):
            return NotImplemented
        return self._equals == other._equals and self._not_equals == other._not_equals

    def __hash__(self) -> int:
        return hash((self._equals, self._not_equals))

    def __repr__(self) -> str:
        return f"WhereClause(equals={self._equals!r}, not_equals={self._not_equals!r})"


# ---------------------------------------------------------------------------
# IndexParams — audit-only record of create_index() call.
# ---------------------------------------------------------------------------
class IndexParams:
    """Mirrors the kwargs lancedb accepts under ``table.create_index(...)``.

    Records the requested index type / metric type / params. The
    in-memory backend evaluates search exactly regardless; params are
    kept for audit / replay.
    """

    __slots__ = ("_index_type", "_metric_type", "_params")

    def __init__(
        self,
        *,
        index_type: IndexType,
        metric_type: MetricType,
        params: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(index_type, IndexType):
            raise TypeError("index_type must be an IndexType")
        if not isinstance(metric_type, MetricType):
            raise TypeError("metric_type must be a MetricType")
        frozen: tuple[tuple[str, int], ...] = ()
        if params is not None:
            if not isinstance(params, Mapping):
                raise TypeError("params must be a Mapping or None")
            pairs: list[tuple[str, int]] = []
            for k in sorted(params):
                if not isinstance(k, str):
                    raise TypeError("params keys must be str")
                v = params[k]
                if not isinstance(v, int) or isinstance(v, bool):
                    raise TypeError(f"params[{k!r}] must be int")
                if v <= 0:
                    raise ValueError(f"params[{k!r}] must be positive, got {v}")
                pairs.append((k, v))
            frozen = tuple(pairs)
        self._index_type = index_type
        self._metric_type = metric_type
        self._params = frozen

    @property
    def index_type(self) -> IndexType:
        return self._index_type

    @property
    def metric_type(self) -> MetricType:
        return self._metric_type

    @property
    def params(self) -> tuple[tuple[str, int], ...]:
        return self._params

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexParams):
            return NotImplemented
        return (
            self._index_type is other._index_type
            and self._metric_type is other._metric_type
            and self._params == other._params
        )

    def __hash__(self) -> int:
        return hash((self._index_type, self._metric_type, self._params))

    def __repr__(self) -> str:
        return (
            f"IndexParams(index_type={self._index_type!r}, "
            f"metric_type={self._metric_type!r}, "
            f"params={self._params!r})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in vec))


def _inner_product(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = _inner_product(a, b) / (na * nb)
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return 1.0 - cos


def _l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _distance(
    metric: MetricType,
    a: Sequence[float],
    b: Sequence[float],
    *,
    dot_offset: float = 0.0,
) -> float:
    if metric is MetricType.COSINE:
        return _cosine_distance(a, b)
    if metric is MetricType.L2:
        return _l2_distance(a, b)
    if metric is MetricType.DOT:
        ip = _inner_product(a, b)
        d = dot_offset - ip
        if d < 0.0:
            d = 0.0
        return d
    raise LanceDBStoreError(f"unknown metric type: {metric!r}")


# ---------------------------------------------------------------------------
# LanceDBStore — primary backend
# ---------------------------------------------------------------------------
class LanceDBStore:
    """Embedded LanceDB-style vector store (in-memory mode).

    Implements :class:`~state.memory_tensor.contracts.MemoryStoreBase`.
    """

    __slots__ = (
        "_dim",
        "_dot_offset",
        "_episodes",
        "_index_params",
        "_max_size",
        "_metric",
        "_table_name",
    )

    def __init__(
        self,
        *,
        dim: int,
        max_size: int,
        metric_type: MetricType = MetricType.COSINE,
        table_name: str = "dix_lancedb",
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive int")
        if not isinstance(metric_type, MetricType):
            raise TypeError("metric_type must be a MetricType")
        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty str")
        self._dim = dim
        self._max_size = max_size
        self._metric = metric_type
        self._table_name = table_name
        self._episodes: dict[str, Episode] = {}
        self._dot_offset: float = 0.0
        self._index_params: IndexParams | None = None

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def metric_type(self) -> MetricType:
        return self._metric

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def index_params(self) -> IndexParams | None:
        return self._index_params

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, episode_id: object) -> bool:
        if not isinstance(episode_id, str):
            return False
        return episode_id in self._episodes

    def __iter__(self) -> Iterator[Episode]:
        for eid in sorted(self._episodes):
            yield self._episodes[eid]

    # ------------------------------------------------------------------
    # Write surface
    # ------------------------------------------------------------------
    def add(self, episode: Episode) -> None:
        if not isinstance(episode, Episode):
            raise TypeError("episode must be an Episode")
        if episode.dim != self._dim:
            raise ValueError(f"episode.dim={episode.dim} != store.dim={self._dim}")
        if episode.episode_id in self._episodes:
            raise ValueError(f"episode_id={episode.episode_id!r} already in store")
        if len(self._episodes) >= self._max_size:
            self._evict_oldest()
        self._episodes[episode.episode_id] = episode
        if self._metric is MetricType.DOT:
            ip = _inner_product(episode.embedding, episode.embedding)
            if ip > self._dot_offset:
                self._dot_offset = ip

    def insert(self, episodes: Iterable[Episode]) -> None:
        """LanceDB ``Table.add`` shape."""
        for episode in episodes:
            self.add(episode)

    def upsert(self, episodes: Iterable[Episode]) -> None:
        for episode in episodes:
            if episode.episode_id in self._episodes:
                self.delete(episode.episode_id)
            self.add(episode)

    def delete(self, episode_id: str) -> bool:
        if not isinstance(episode_id, str):
            raise TypeError("episode_id must be a str")
        return self._episodes.pop(episode_id, None) is not None

    def _evict_oldest(self) -> None:
        if not self._episodes:
            return
        oldest = min(
            self._episodes.values(),
            key=lambda e: (e.ts_ns, e.episode_id),
        )
        del self._episodes[oldest.episode_id]

    # ------------------------------------------------------------------
    # Index creation (audit-only)
    # ------------------------------------------------------------------
    def create_index(self, params: IndexParams) -> None:
        if not isinstance(params, IndexParams):
            raise TypeError("params must be IndexParams")
        if params.metric_type is not self._metric:
            raise ValueError(
                f"index metric_type={params.metric_type!r} != store metric_type={self._metric!r}"
            )
        self._index_params = params

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: MemoryQuery) -> MemoryResult:
        return self.search_with_filter(query, where=None)

    def search_with_filter(
        self,
        query: MemoryQuery,
        *,
        where: WhereClause | None,
    ) -> MemoryResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        if query.dim != self._dim:
            raise ValueError(f"query.dim={query.dim} != store.dim={self._dim}")
        if where is not None and not isinstance(where, WhereClause):
            raise TypeError("where must be a WhereClause or None")
        scored: list[tuple[float, int, str, Episode]] = []
        for episode in self._episodes.values():
            if where is not None and not where.matches(episode.payload):
                continue
            d = _distance(
                self._metric,
                query.embedding,
                episode.embedding,
                dot_offset=self._dot_offset,
            )
            scored.append((d, episode.ts_ns, episode.episode_id, episode))
        scored.sort(key=lambda row: (row[0], row[1], row[2]))
        top = scored[: query.k]
        hits = tuple(
            MemoryHit(
                episode_id=ep.episode_id,
                distance=d,
                ts_ns=ep.ts_ns,
                payload=ep.payload,
            )
            for d, _ts, _eid, ep in top
        )
        return MemoryResult(
            ts_ns=query.ts_ns,
            query_id=query.query_id,
            hits=hits,
        )

    # ------------------------------------------------------------------
    # Scan (LanceDB ``Table.to_list()`` shape with optional ``where``)
    # ------------------------------------------------------------------
    def to_list(
        self,
        *,
        where: WhereClause | None = None,
    ) -> tuple[Episode, ...]:
        if where is not None and not isinstance(where, WhereClause):
            raise TypeError("where must be a WhereClause or None")
        matches: list[Episode] = []
        for episode in self._episodes.values():
            if where is None or where.matches(episode.payload):
                matches.append(episode)
        matches.sort(key=lambda e: (e.ts_ns, e.episode_id))
        return tuple(matches)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def serialize(self) -> bytes:
        episodes_sorted = sorted(
            self._episodes.values(),
            key=lambda e: (e.ts_ns, e.episode_id),
        )
        index_payload: dict[str, Any] | None
        if self._index_params is None:
            index_payload = None
        else:
            index_payload = {
                "index_type": self._index_params.index_type.value,
                "metric_type": self._index_params.metric_type.value,
                "params": [{"key": k, "value": v} for k, v in self._index_params.params],
            }
        payload = {
            "version": _SERIALIZATION_VERSION,
            "table": self._table_name,
            "dim": self._dim,
            "max_size": self._max_size,
            "metric": self._metric.value,
            "dot_offset": self._dot_offset,
            "index": index_payload,
            "episodes": [
                {
                    "ts_ns": e.ts_ns,
                    "episode_id": e.episode_id,
                    "embedding": list(e.embedding),
                    "payload": dict(e.payload),
                }
                for e in episodes_sorted
            ],
        }
        return json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("ascii")

    @classmethod
    def deserialize(cls, raw: bytes) -> LanceDBStore:
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw must be bytes")
        try:
            obj = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LanceDBStoreError(f"corrupt serialised blob: {exc!s}") from exc
        if not isinstance(obj, dict):
            raise LanceDBStoreError("blob root must be an object")
        version = obj.get("version")
        if version != _SERIALIZATION_VERSION:
            raise LanceDBStoreError(f"unsupported version: {version!r}")
        try:
            metric = MetricType(obj["metric"])
        except (KeyError, ValueError) as exc:
            raise LanceDBStoreError(f"bad metric: {exc!s}") from exc
        store = cls(
            dim=int(obj["dim"]),
            max_size=int(obj["max_size"]),
            metric_type=metric,
            table_name=str(obj["table"]),
        )
        store._dot_offset = float(obj.get("dot_offset", 0.0))
        index_raw = obj.get("index")
        if index_raw is not None:
            try:
                idx_type = IndexType(index_raw["index_type"])
                idx_metric = MetricType(index_raw["metric_type"])
            except (KeyError, ValueError) as exc:
                raise LanceDBStoreError(f"bad index: {exc!s}") from exc
            params_raw = index_raw.get("params", [])
            if not isinstance(params_raw, list):
                raise LanceDBStoreError("index params must be list")
            params_dict: dict[str, int] = {}
            for row in params_raw:
                if not isinstance(row, dict):
                    raise LanceDBStoreError("index param row must be object")
                params_dict[str(row["key"])] = int(row["value"])
            store._index_params = IndexParams(
                index_type=idx_type,
                metric_type=idx_metric,
                params=params_dict if params_dict else None,
            )
        for row in obj.get("episodes", []):
            embedding = tuple(float(x) for x in row["embedding"])
            validate_embedding(embedding, field="Episode.embedding")
            payload_raw = row.get("payload", {})
            if not isinstance(payload_raw, dict):
                raise LanceDBStoreError("payload must be an object")
            payload = MappingProxyType({str(k): str(v) for k, v in payload_raw.items()})
            episode = Episode(
                ts_ns=int(row["ts_ns"]),
                episode_id=str(row["episode_id"]),
                embedding=embedding,
                payload=payload,
            )
            store._episodes[episode.episode_id] = episode
        return store


# ---------------------------------------------------------------------------
# LanceDB binding (lazy)
# ---------------------------------------------------------------------------
def lancedb_connect(*, uri: str) -> Any:
    """Lazy-bind the ``lancedb`` package.

    The ``lancedb`` import is confined to this function body.

    Returns the live ``lancedb.DBConnection`` instance; the caller is
    responsible for binding it to a wrapper that satisfies
    :class:`~state.memory_tensor.contracts.MemoryStoreBase`.
    """
    if not isinstance(uri, str) or not uri:
        raise ValueError("uri must be a non-empty str")
    try:
        import lancedb  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when dep absent
        raise LanceDBStoreError("lancedb is not installed; see NEW_PIP_DEPENDENCIES") from exc
    return lancedb.connect(uri)


__all__ = [
    "IndexParams",
    "IndexType",
    "LANCEDB_ADAPTER_VERSION",
    "LanceDBStore",
    "LanceDBStoreError",
    "MetricType",
    "NEW_PIP_DEPENDENCIES",
    "WhereClause",
    "lancedb_connect",
]
