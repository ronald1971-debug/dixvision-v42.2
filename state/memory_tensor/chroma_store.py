# ADAPTED FROM: chroma-core/chroma
#   chromadb/api/client.py — ``Client.create_collection`` /
#   ``Client.get_or_create_collection``;
#   chromadb/api/models/Collection.py — ``Collection.add(ids,
#   embeddings, metadatas, documents)`` / ``Collection.query(
#   query_embeddings, n_results, where, where_document)`` /
#   ``Collection.delete(ids, where)``;
#   chromadb/db/impl/sqlite.py — persistent SQLite backend (audit
#   reference only — the in-memory backend is the C-24 leaf).
#
# License: Apache-2.0.
"""Developer-friendly RAG vector store — chromadb backend (C-25).

Pure-Python reproduction of the ``chromadb.Client`` + ``Collection`` API
surface behind the
:class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol.
Chromadb is the developer-friendly RAG-pipeline backend with optional
persistent SQLite storage — this leaf ships the algorithmic surface plus
an in-memory pure-Python evaluator; the real ``chromadb.PersistentClient``
kicks in only when the operator provisions a SQLite store and uses
:func:`chroma_client_factory`.

Algorithmic surface ported from chromadb:

* ``client = chromadb.Client()`` →
  :func:`chroma_client_factory` (lazy, factory only).
* ``collection = client.create_collection(name, metadata={"hnsw:space":
  metric})`` → store constructor.
* ``collection.add(ids=[...], embeddings=[...], metadatas=[...])`` →
  :meth:`add` / :meth:`insert` / :meth:`upsert`.
* ``collection.query(query_embeddings=[...], n_results=k, where={...},
  where_document={...})`` → :meth:`search` / :meth:`search_with_filter`.
* ``collection.delete(ids=[...], where={...})`` → :meth:`delete` /
  :meth:`delete_where`.
* ``collection.get(ids=[...], where={...})`` → :meth:`get`.

Distance metrics mirror chromadb's ``hnsw:space`` collection metadata:

* ``L2``: Euclidean L2 distance (default in chromadb).
* ``COSINE``: cosine **distance** (``1 - cos``).
* ``IP``: negated inner product, shifted to non-negative.

Where-clauses mirror chromadb's metadata filter dict shape:

* ``{"key": "value"}`` → equals match (must).
* ``{"key": {"$ne": "value"}}`` → not-equals match (must_not).
* Multiple keys → conjunction (all must hold).

Authority constraints (C-25, OFFLINE_ONLY):

* **OFFLINE tier write** — :meth:`add` / :meth:`delete` /
  :meth:`upsert` are never called from the hot path.
* **RUNTIME-SAFE read** — :meth:`search` / :meth:`get` run in <5 ms
  for typical N.
* **B27 / B28 / INV-71 authority symmetry** — this module does NOT
  construct typed bus events.
* **INV-15 replay determinism** — same inputs → same outputs
  byte-identical. Tie-breaking on ``(distance, ts_ns, episode_id)``;
  serialisation sorts episodes by ``(ts_ns, episode_id)``;
  :meth:`serialize` round-trips byte-equal via :meth:`deserialize`.
* **No clock, no PRNG, no IO** — every timestamp comes from caller
  supplied ``Episode.ts_ns`` / ``MemoryQuery.ts_ns``.
* **Pure stdlib** — no chromadb, no numpy. ``NEW_PIP_DEPENDENCIES =
  ("chromadb",)`` only for the lazy factory.
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
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("chromadb",)
CHROMA_ADAPTER_VERSION: str = "1"
_SERIALIZATION_VERSION: int = 1


# ---------------------------------------------------------------------------
# Distance metric enum — mirrors chromadb's ``hnsw:space``.
# ---------------------------------------------------------------------------
class DistanceMetric(enum.Enum):
    L2 = "l2"
    COSINE = "cosine"
    IP = "ip"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ChromaStoreError(RuntimeError):
    """Raised when the chromadb adapter rejects an input."""


# ---------------------------------------------------------------------------
# WhereFilter — Python form of chromadb's metadata filter dict.
# ---------------------------------------------------------------------------
class WhereFilter:
    """Metadata filter over payload fields.

    Mirrors chromadb's ``where=`` dict shape (closed subset):

    * ``equals``: every key must equal the value.
    * ``not_equals``: none of the keys may equal the value.
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
            raise TypeError(f"WhereFilter.{name} must be a Mapping or None")
        pairs: list[tuple[str, str]] = []
        for key in sorted(m):
            if not isinstance(key, str):
                raise TypeError(f"WhereFilter.{name} keys must be str")
            value = m[key]
            if not isinstance(value, str):
                raise TypeError(f"WhereFilter.{name}[{key!r}] must be str")
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
        if not isinstance(other, WhereFilter):
            return NotImplemented
        return self._equals == other._equals and self._not_equals == other._not_equals

    def __hash__(self) -> int:
        return hash((self._equals, self._not_equals))

    def __repr__(self) -> str:
        return f"WhereFilter(equals={self._equals!r}, not_equals={self._not_equals!r})"


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
    metric: DistanceMetric,
    a: Sequence[float],
    b: Sequence[float],
    *,
    ip_offset: float = 0.0,
) -> float:
    if metric is DistanceMetric.COSINE:
        return _cosine_distance(a, b)
    if metric is DistanceMetric.L2:
        return _l2_distance(a, b)
    if metric is DistanceMetric.IP:
        ip = _inner_product(a, b)
        d = ip_offset - ip
        if d < 0.0:
            d = 0.0
        return d
    raise ChromaStoreError(f"unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# ChromaCollectionStore — primary backend
# ---------------------------------------------------------------------------
class ChromaCollectionStore:
    """Chromadb-style RAG semantic vector store (in-memory mode).

    Implements :class:`~state.memory_tensor.contracts.MemoryStoreBase`.
    """

    __slots__ = (
        "_collection_name",
        "_dim",
        "_episodes",
        "_ip_offset",
        "_max_size",
        "_metric",
    )

    def __init__(
        self,
        *,
        dim: int,
        max_size: int,
        metric: DistanceMetric = DistanceMetric.L2,
        collection_name: str = "dix_chroma",
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive int")
        if not isinstance(metric, DistanceMetric):
            raise TypeError("metric must be a DistanceMetric")
        if not isinstance(collection_name, str) or not collection_name:
            raise ValueError("collection_name must be a non-empty str")
        self._dim = dim
        self._max_size = max_size
        self._metric = metric
        self._collection_name = collection_name
        self._episodes: dict[str, Episode] = {}
        self._ip_offset: float = 0.0

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
    def metric(self) -> DistanceMetric:
        return self._metric

    @property
    def collection_name(self) -> str:
        return self._collection_name

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
        if self._metric is DistanceMetric.IP:
            ip = _inner_product(episode.embedding, episode.embedding)
            if ip > self._ip_offset:
                self._ip_offset = ip

    def insert(self, episodes: Iterable[Episode]) -> None:
        """Chromadb ``Collection.add(ids=[...], embeddings=[...])``."""
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

    def delete_where(self, where: WhereFilter) -> int:
        """Chromadb ``Collection.delete(where=...)``."""
        if not isinstance(where, WhereFilter):
            raise TypeError("where must be a WhereFilter")
        to_remove = [eid for eid, ep in self._episodes.items() if where.matches(ep.payload)]
        for eid in to_remove:
            del self._episodes[eid]
        return len(to_remove)

    def _evict_oldest(self) -> None:
        if not self._episodes:
            return
        oldest = min(
            self._episodes.values(),
            key=lambda e: (e.ts_ns, e.episode_id),
        )
        del self._episodes[oldest.episode_id]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: MemoryQuery) -> MemoryResult:
        return self.search_with_filter(query, where=None)

    def search_with_filter(
        self,
        query: MemoryQuery,
        *,
        where: WhereFilter | None,
    ) -> MemoryResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        if query.dim != self._dim:
            raise ValueError(f"query.dim={query.dim} != store.dim={self._dim}")
        if where is not None and not isinstance(where, WhereFilter):
            raise TypeError("where must be a WhereFilter or None")
        scored: list[tuple[float, int, str, Episode]] = []
        for episode in self._episodes.values():
            if where is not None and not where.matches(episode.payload):
                continue
            d = _distance(
                self._metric,
                query.embedding,
                episode.embedding,
                ip_offset=self._ip_offset,
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
    # Get (chromadb's ``Collection.get(ids=[...], where=...)`` shape)
    # ------------------------------------------------------------------
    def get(
        self,
        *,
        ids: Iterable[str] | None = None,
        where: WhereFilter | None = None,
    ) -> tuple[Episode, ...]:
        if where is not None and not isinstance(where, WhereFilter):
            raise TypeError("where must be a WhereFilter or None")
        if ids is not None:
            id_set: set[str] = set()
            for x in ids:
                if not isinstance(x, str):
                    raise TypeError("ids must be an iterable of str")
                id_set.add(x)
        else:
            id_set = set(self._episodes)
        matches: list[Episode] = []
        for eid in id_set:
            ep = self._episodes.get(eid)
            if ep is None:
                continue
            if where is not None and not where.matches(ep.payload):
                continue
            matches.append(ep)
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
        payload = {
            "version": _SERIALIZATION_VERSION,
            "collection_name": self._collection_name,
            "dim": self._dim,
            "max_size": self._max_size,
            "metric": self._metric.value,
            "ip_offset": self._ip_offset,
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
    def deserialize(cls, raw: bytes) -> ChromaCollectionStore:
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw must be bytes")
        try:
            obj = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChromaStoreError(f"corrupt serialised blob: {exc!s}") from exc
        if not isinstance(obj, dict):
            raise ChromaStoreError("blob root must be an object")
        version = obj.get("version")
        if version != _SERIALIZATION_VERSION:
            raise ChromaStoreError(f"unsupported version: {version!r}")
        try:
            metric = DistanceMetric(obj["metric"])
        except (KeyError, ValueError) as exc:
            raise ChromaStoreError(f"bad metric: {exc!s}") from exc
        store = cls(
            dim=int(obj["dim"]),
            max_size=int(obj["max_size"]),
            metric=metric,
            collection_name=str(obj["collection_name"]),
        )
        store._ip_offset = float(obj.get("ip_offset", 0.0))
        for row in obj.get("episodes", []):
            embedding = tuple(float(x) for x in row["embedding"])
            validate_embedding(embedding, field="Episode.embedding")
            payload_raw = row.get("payload", {})
            if not isinstance(payload_raw, dict):
                raise ChromaStoreError("payload must be an object")
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
# Chromadb binding (lazy)
# ---------------------------------------------------------------------------
def chroma_client_factory(
    *,
    persist_directory: str | None = None,
) -> Any:
    """Lazy-bind the ``chromadb`` package.

    The ``chromadb`` import is confined to this function body.

    When ``persist_directory`` is ``None``, the ephemeral in-memory
    client is returned (``chromadb.Client()``); otherwise the SQLite
    persistent client is returned (``chromadb.PersistentClient(path=...)``).
    """
    if persist_directory is not None and (
        not isinstance(persist_directory, str) or not persist_directory
    ):
        raise ValueError("persist_directory must be a non-empty str or None")
    try:
        import chromadb  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when dep absent
        raise ChromaStoreError("chromadb is not installed; see NEW_PIP_DEPENDENCIES") from exc
    if persist_directory is None:
        return chromadb.Client()
    return chromadb.PersistentClient(path=persist_directory)


__all__ = [
    "CHROMA_ADAPTER_VERSION",
    "ChromaCollectionStore",
    "ChromaStoreError",
    "DistanceMetric",
    "NEW_PIP_DEPENDENCIES",
    "WhereFilter",
    "chroma_client_factory",
]
