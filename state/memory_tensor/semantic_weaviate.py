# ADAPTED FROM: weaviate/weaviate-python-client
#   weaviate_client/v4/collections.py — Collection / data.insert /
#   query.near_vector / data.delete_by_id;
#   weaviate_client/v4/data.py — DataObject with `properties` + `vector`;
#   weaviate_client/v4/query.py — `near_vector(vector, limit, filters)`,
#   `Filter.by_property(...).equal(...)`;
#   weaviate_client/v4/connect.py — embedded vs server connection split.
#
# License: BSD-3-Clause.
"""Persistent semantic vector store — weaviate-client-shape backend (C-21).

Pure-Python reproduction of the ``weaviate-client v4`` collection +
query surface behind the
:class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol. This
module is a drop-in alternative to the FAISS-IP store (S-08.3) and the
Qdrant-style store (A-10); engines depend only on the Protocol in
:mod:`state.memory_tensor.contracts`.

Algorithmic surface ported from weaviate-client v4:

* ``client.collections.create(name, vectorizer_config, properties)`` —
  collection schema is bound at constructor time (``dim`` +
  ``distance_metric`` + ``collection_name``).
* ``collection.data.insert(properties, vector, uuid)`` — :meth:`add`.
* ``collection.query.near_vector(vector, limit, filters)`` —
  :meth:`search_with_filter` (un-filtered → :meth:`search`).
* ``collection.data.delete_by_id(uuid)`` — :meth:`delete`.
* ``Filter.by_property(name).equal(value)`` — :class:`Filter` primitive
  with ``must`` / ``should`` / ``must_not`` clauses, matching against
  :attr:`Episode.payload` (string-only, per the contracts).
* Embedded mode (``WeaviateClient.connect_to_embedded()``) — the only
  mode this adapter implements; the persistent
  ``WeaviateClient.connect_to_local(...)`` / ``connect_to_wcs(...)``
  paths are reached via :func:`weaviate_client_factory`.

The official ``weaviate-client`` package is lazy-imported *only* inside
:func:`weaviate_client_factory` — the top-level module imports remain
empty so this leaf is importable in replay / test environments where
the dependency is absent.

Distance lanes (mirroring weaviate's ``VectorDistances`` enum):

* ``COSINE``: cosine **distance** ``d = 1 - cos(a, b)`` (weaviate's
  default; zero-norm fallback ``d = 1``).
* ``DOT``: negated inner product shifted by the running maximum
  positive self-IP, identical to qdrant's DOT lane so callers can swap
  backends without rewriting their scoring.
* ``L2_SQUARED``: weaviate's squared-Euclidean distance (``≥ 0`` by
  construction — weaviate stores L2² for speed; we mirror exactly).

Each :class:`~state.memory_tensor.contracts.MemoryHit` carries a
non-negative scalar so the contracts invariant always holds.

Authority constraints (C-21, OFFLINE_ONLY):

* **OFFLINE tier write** — :meth:`add` / :meth:`delete` are never
  called from the hot path. Authority-lint pins.
* **RUNTIME-SAFE read path** — :meth:`search` may be invoked from
  runtime engines, but only if it returns in < 5 ms for the typical
  store size (mirrors the S-08.3 / A-10 contract).
* **B27 / B28 / INV-71 authority symmetry** — this module does NOT
  construct typed bus events (``SignalEvent`` / ``ExecutionEvent`` /
  ``SystemEvent`` / ``HazardEvent`` / ``GovernanceDecision`` /
  ``PatchProposal``). Pinned by AST guard.
* **INV-15 replay determinism** — same inputs → same outputs
  byte-identical. Tie-breaking on ``(distance, ts_ns, episode_id)``;
  serialisation sorts episodes by ``(ts_ns, episode_id)``;
  :meth:`serialize` round-trips byte-equal via :meth:`deserialize`.
* **No clock, no PRNG, no IO** — every timestamp comes from
  caller-supplied ``Episode.ts_ns`` / ``MemoryQuery.ts_ns``. No
  ``random`` / ``time`` / ``datetime`` / ``os`` / ``socket`` /
  ``requests`` imports.
* **Pure stdlib** — no weaviate-client, no numpy. ``NEW_PIP_DEPENDENCIES
  = ("weaviate-client",)`` only for the lazy factory wrapper.
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
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("weaviate-client",)
WEAVIATE_ADAPTER_VERSION: str = "1"
_SERIALIZATION_VERSION: int = 1


# ---------------------------------------------------------------------------
# Distance metric enum
# ---------------------------------------------------------------------------
class DistanceMetric(enum.Enum):
    """Mirrors weaviate's ``VectorDistances`` enum (subset).

    Weaviate names the squared-Euclidean distance ``l2-squared`` (not
    ``euclid``) because the server stores the squared value for speed.
    This adapter preserves the upstream naming verbatim.
    """

    COSINE = "cosine"
    DOT = "dot"
    L2_SQUARED = "l2-squared"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class SemanticWeaviateError(RuntimeError):
    """Raised when the weaviate adapter rejects an input or capacity error."""


# ---------------------------------------------------------------------------
# Filter primitives — Python form of weaviate's `Filter.by_property` clauses.
# ---------------------------------------------------------------------------
class Filter:
    """Match-all / match-any payload filter.

    Mirrors weaviate's ``Filter.by_property(name).equal(value)`` chain:

    * ``must`` — every condition must hold (intersection),
    * ``should`` — at least one must hold,
    * ``must_not`` — none may hold.

    Payload values are strings (see :class:`Episode.payload`); the
    filter compares verbatim.
    """

    __slots__ = ("_must", "_must_not", "_should")

    def __init__(
        self,
        *,
        must: Mapping[str, str] | None = None,
        should: Mapping[str, str] | None = None,
        must_not: Mapping[str, str] | None = None,
    ) -> None:
        self._must = self._freeze(must, "must")
        self._should = self._freeze(should, "should")
        self._must_not = self._freeze(must_not, "must_not")

    @staticmethod
    def _freeze(m: Mapping[str, str] | None, name: str) -> tuple[tuple[str, str], ...]:
        if m is None:
            return ()
        if not isinstance(m, Mapping):
            raise TypeError(f"Filter.{name} must be a Mapping or None")
        pairs: list[tuple[str, str]] = []
        for key in sorted(m):
            if not isinstance(key, str):
                raise TypeError(f"Filter.{name} keys must be str")
            value = m[key]
            if not isinstance(value, str):
                raise TypeError(f"Filter.{name}[{key!r}] must be str")
            pairs.append((key, value))
        return tuple(pairs)

    @property
    def must(self) -> tuple[tuple[str, str], ...]:
        return self._must

    @property
    def should(self) -> tuple[tuple[str, str], ...]:
        return self._should

    @property
    def must_not(self) -> tuple[tuple[str, str], ...]:
        return self._must_not

    def matches(self, payload: Mapping[str, str]) -> bool:
        for key, value in self._must:
            if payload.get(key) != value:
                return False
        for key, value in self._must_not:
            if payload.get(key) == value:
                return False
        if self._should:
            if not any(payload.get(k) == v for k, v in self._should):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Filter):
            return NotImplemented
        return (
            self._must == other._must
            and self._should == other._should
            and self._must_not == other._must_not
        )

    def __hash__(self) -> int:
        return hash((self._must, self._should, self._must_not))

    def __repr__(self) -> str:
        return f"Filter(must={self._must!r}, should={self._should!r}, must_not={self._must_not!r})"


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


def _l2_squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _distance(
    metric: DistanceMetric,
    a: Sequence[float],
    b: Sequence[float],
    *,
    dot_offset: float = 0.0,
) -> float:
    if metric is DistanceMetric.COSINE:
        return _cosine_distance(a, b)
    if metric is DistanceMetric.L2_SQUARED:
        return _l2_squared_distance(a, b)
    if metric is DistanceMetric.DOT:
        ip = _inner_product(a, b)
        d = dot_offset - ip
        if d < 0.0:
            d = 0.0
        return d
    raise SemanticWeaviateError(f"unknown distance metric: {metric!r}")


# ---------------------------------------------------------------------------
# SemanticWeaviateStore — primary backend
# ---------------------------------------------------------------------------
class SemanticWeaviateStore:
    """Weaviate-collection-shape semantic vector store (embedded mode).

    Implements :class:`~state.memory_tensor.contracts.MemoryStoreBase`.
    """

    __slots__ = (
        "_collection",
        "_dim",
        "_dot_offset",
        "_episodes",
        "_max_size",
        "_metric",
    )

    def __init__(
        self,
        *,
        dim: int,
        max_size: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        collection: str = "DIXMemoryEpisode",
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive int")
        if not isinstance(distance_metric, DistanceMetric):
            raise TypeError("distance_metric must be a DistanceMetric")
        if not isinstance(collection, str) or not collection:
            raise ValueError("collection must be a non-empty str")
        self._dim = dim
        self._max_size = max_size
        self._metric = distance_metric
        self._collection = collection
        self._episodes: dict[str, Episode] = {}
        self._dot_offset: float = 0.0

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
    def distance_metric(self) -> DistanceMetric:
        return self._metric

    @property
    def collection(self) -> str:
        return self._collection

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, episode_id: object) -> bool:
        if not isinstance(episode_id, str):
            return False
        return episode_id in self._episodes

    def __iter__(self) -> Iterator[Episode]:
        for eid in sorted(self._episodes):
            yield self._episodes[eid]

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
        if self._metric is DistanceMetric.DOT:
            ip = _inner_product(episode.embedding, episode.embedding)
            if ip > self._dot_offset:
                self._dot_offset = ip

    def upsert(self, episodes: Iterable[Episode]) -> None:
        """Insert-or-replace a batch (mirrors ``data.insert_many``).

        Existing ``episode_id`` rows are deleted before re-insertion so
        the embedding and payload are atomically refreshed.
        """
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
    # Search
    # ------------------------------------------------------------------
    def search(self, query: MemoryQuery) -> MemoryResult:
        return self.search_with_filter(query, query_filter=None)

    def search_with_filter(
        self,
        query: MemoryQuery,
        *,
        query_filter: Filter | None,
    ) -> MemoryResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        if query.dim != self._dim:
            raise ValueError(f"query.dim={query.dim} != store.dim={self._dim}")
        if query_filter is not None and not isinstance(query_filter, Filter):
            raise TypeError("query_filter must be a Filter or None")
        scored: list[tuple[float, int, str, Episode]] = []
        for episode in self._episodes.values():
            if query_filter is not None and not query_filter.matches(episode.payload):
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
    # Serialisation
    # ------------------------------------------------------------------
    def serialize(self) -> bytes:
        episodes_sorted = sorted(
            self._episodes.values(),
            key=lambda e: (e.ts_ns, e.episode_id),
        )
        payload = {
            "version": _SERIALIZATION_VERSION,
            "collection": self._collection,
            "dim": self._dim,
            "max_size": self._max_size,
            "metric": self._metric.value,
            "dot_offset": self._dot_offset,
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
    def deserialize(cls, raw: bytes) -> SemanticWeaviateStore:
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw must be bytes")
        try:
            obj = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SemanticWeaviateError(f"corrupt serialised blob: {exc!s}") from exc
        if not isinstance(obj, dict):
            raise SemanticWeaviateError("blob root must be an object")
        version = obj.get("version")
        if version != _SERIALIZATION_VERSION:
            raise SemanticWeaviateError(f"unsupported version: {version!r}")
        try:
            metric = DistanceMetric(obj["metric"])
        except (KeyError, ValueError) as exc:
            raise SemanticWeaviateError(f"bad metric: {exc!s}") from exc
        store = cls(
            dim=int(obj["dim"]),
            max_size=int(obj["max_size"]),
            distance_metric=metric,
            collection=str(obj["collection"]),
        )
        store._dot_offset = float(obj.get("dot_offset", 0.0))
        for row in obj.get("episodes", []):
            embedding = tuple(float(x) for x in row["embedding"])
            validate_embedding(embedding, field="Episode.embedding")
            payload_raw = row.get("payload", {})
            if not isinstance(payload_raw, dict):
                raise SemanticWeaviateError("payload must be an object")
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
# Weaviate binding (lazy)
# ---------------------------------------------------------------------------
def weaviate_client_factory(
    *,
    url: str = ":embedded:",
    api_key: str | None = None,
) -> Any:
    """Lazy-bind the ``weaviate-client`` package.

    The ``weaviate`` import is confined to this function body; the
    rest of the module remains importable without the dependency.

    Returns the live ``WeaviateClient`` instance; the caller is
    responsible for binding it to a wrapper that satisfies
    :class:`~state.memory_tensor.contracts.MemoryStoreBase` (the
    wrapper is out-of-scope for this leaf — C-21 ships the embedded
    pure-Python backend; a follow-up wiring PR can swap in the real
    client when the operator provisions a weaviate server).

    Parameters:
        url: ``":embedded:"`` for ``connect_to_embedded()``, otherwise
            the HTTP URL of a running weaviate instance.
        api_key: optional Bearer key for Weaviate Cloud Services.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty str")
    if api_key is not None and not isinstance(api_key, str):
        raise TypeError("api_key must be a str or None")
    try:
        import weaviate  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when dep absent
        raise SemanticWeaviateError(
            "weaviate-client is not installed; see NEW_PIP_DEPENDENCIES"
        ) from exc
    if url == ":embedded:":
        return weaviate.connect_to_embedded()
    if api_key is None:
        return weaviate.connect_to_local(url=url)
    return weaviate.connect_to_wcs(cluster_url=url, auth_credentials=api_key)


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "WEAVIATE_ADAPTER_VERSION",
    "DistanceMetric",
    "Filter",
    "SemanticWeaviateError",
    "SemanticWeaviateStore",
    "weaviate_client_factory",
]
