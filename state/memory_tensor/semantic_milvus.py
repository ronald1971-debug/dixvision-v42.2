# ADAPTED FROM: milvus-io/pymilvus
#   pymilvus/client/stub.py — Collection.insert / Collection.search /
#   Collection.create_index / Collection.delete / Collection.query;
#   pymilvus/orm/types.py — DataType (FLOAT_VECTOR, VARCHAR);
#   pymilvus/orm/constants.py — IndexType (HNSW, IVF_FLAT, FLAT),
#   MetricType (L2, IP, COSINE).
#
# License: Apache-2.0.
"""Massive-scale semantic vector store — Milvus-style backend (C-23).

Pure-Python reproduction of the pymilvus ``Collection`` + ``insert`` +
``search`` + ``create_index`` surface behind the
:class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol. This is
a drop-in swap for the qdrant-style store from A-10
(:mod:`state.memory_tensor.semantic_qdrant`); engines depend only on the
Protocol in :mod:`state.memory_tensor.contracts`.

Milvus is intended for the ``>10M vectors`` regime — the live backend
ships an HNSW graph index over a sharded segment store. This adapter
ships the algorithmic surface plus an in-memory pure-Python evaluator,
so authority lint / replay determinism / serialise round-trip all hold
without the live dependency.

Algorithmic surface ported from pymilvus:

* ``Collection(name, schema)`` — collection schema is bound at
  constructor time (``dim`` + ``metric_type``).
* ``collection.insert([(id, vector, payload_json), ...])`` —
  :meth:`add` / :meth:`upsert`.
* ``collection.search(vectors, anns_field, param, limit, expr)`` —
  :meth:`search_with_filter` (un-filtered search → :meth:`search`).
* ``collection.delete(expr=f'id in ["{id}"]')`` — :meth:`delete`.
* ``collection.create_index(field_name, index_params)`` —
  :meth:`create_index`; the in-memory backend records the chosen
  index for the audit trail and uses exact search regardless (HNSW
  approximation is the live-client concern).

The official ``pymilvus`` package is lazy-imported *only* inside
:func:`milvus_client_factory` — the top-level module imports remain
empty so this leaf is importable in replay / test environments where
the dependency is absent.

Metric types (mirroring ``pymilvus.MetricType``):

* ``L2``: Euclidean L2 distance (``≥ 0`` by construction).
* ``IP``: negated inner product, shifted to non-negative —
  ``d = max_ip - <a, b>`` where ``max_ip`` is the running maximum
  positive inner product seen at index time; zero-norm fallback: the
  shifted value.
* ``COSINE``: cosine **distance** ``d = 1 - cos(a, b)`` (unit-normalised
  in-flight; identical embeddings → ``d = 0``; orthogonal → ``d = 1``;
  opposing → ``d = 2``). Zero-norm fallback: ``cos = 0`` (``d = 1``).

Index types (mirroring ``pymilvus.IndexType``):

* ``FLAT``: exact brute-force search (the in-memory default).
* ``IVF_FLAT``: inverted-file index — recorded only; in-memory falls
  back to exact.
* ``HNSW``: hierarchical small-world graph — recorded only; in-memory
  falls back to exact.

The hit's ``distance`` field always carries a non-negative scalar so
the :class:`~state.memory_tensor.contracts.MemoryHit` invariant holds.

Authority constraints (C-23, OFFLINE_ONLY):

* **OFFLINE tier write** — :meth:`add` / :meth:`delete` /
  :meth:`create_index` are never called from the hot path.
  Authority-lint pins.
* **RUNTIME-SAFE read path** — :meth:`search` may be invoked from
  runtime engines, but only if it returns in < 5 ms for the typical
  store size (mirrors the S-08.3 contract).
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
* **Pure stdlib** — no pymilvus, no numpy. ``NEW_PIP_DEPENDENCIES =
  ("pymilvus",)`` only for the lazy factory wrapper.
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
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("pymilvus",)
MILVUS_ADAPTER_VERSION: str = "1"
_SERIALIZATION_VERSION: int = 1


# ---------------------------------------------------------------------------
# Metric type enum
# ---------------------------------------------------------------------------
class MetricType(enum.Enum):
    """Mirrors :class:`pymilvus.MetricType`."""

    L2 = "L2"
    IP = "IP"
    COSINE = "COSINE"


# ---------------------------------------------------------------------------
# Index type enum
# ---------------------------------------------------------------------------
class IndexType(enum.Enum):
    """Mirrors :class:`pymilvus.IndexType` (closed subset)."""

    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    HNSW = "HNSW"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class SemanticMilvusError(RuntimeError):
    """Raised when the milvus adapter rejects an input or capacity error."""


# ---------------------------------------------------------------------------
# Boolean expression — Python form of pymilvus ``expr`` filter string.
# ---------------------------------------------------------------------------
class BooleanExpression:
    """Equality / membership filter over payload fields.

    Mirrors pymilvus's ``expr='field == "value"'`` and ``expr='field in
    ["a","b"]'`` boolean expressions. Payload values are strings (see
    :class:`Episode.payload`); the filter compares verbatim.

    Construction is keyword-only:

    * ``must``: every key/value pair must match exactly.
    * ``should``: at least one of the key/value pairs must match.
    * ``must_not``: none of the key/value pairs may match.
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
            raise TypeError(f"BooleanExpression.{name} must be a Mapping or None")
        pairs: list[tuple[str, str]] = []
        for key in sorted(m):
            if not isinstance(key, str):
                raise TypeError(f"BooleanExpression.{name} keys must be str")
            value = m[key]
            if not isinstance(value, str):
                raise TypeError(f"BooleanExpression.{name}[{key!r}] must be str")
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
        if not isinstance(other, BooleanExpression):
            return NotImplemented
        return (
            self._must == other._must
            and self._should == other._should
            and self._must_not == other._must_not
        )

    def __hash__(self) -> int:
        return hash((self._must, self._should, self._must_not))

    def __repr__(self) -> str:
        return (
            f"BooleanExpression(must={self._must!r}, "
            f"should={self._should!r}, must_not={self._must_not!r})"
        )


# ---------------------------------------------------------------------------
# Index params record (audit-only)
# ---------------------------------------------------------------------------
class IndexParams:
    """Mirrors the dict pymilvus accepts under ``create_index(...)``.

    Records the requested index type / metric type / construction-time
    parameters (``M`` / ``efConstruction`` for HNSW, ``nlist`` for
    IVF_FLAT). The in-memory backend evaluates search exactly regardless
    of the recorded index — the parameters are kept for audit / replay.
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
    ip_offset: float = 0.0,
) -> float:
    if metric is MetricType.COSINE:
        return _cosine_distance(a, b)
    if metric is MetricType.L2:
        return _l2_distance(a, b)
    if metric is MetricType.IP:
        ip = _inner_product(a, b)
        d = ip_offset - ip
        if d < 0.0:
            d = 0.0
        return d
    raise SemanticMilvusError(f"unknown metric type: {metric!r}")


# ---------------------------------------------------------------------------
# SemanticMilvusStore — primary backend
# ---------------------------------------------------------------------------
class SemanticMilvusStore:
    """Milvus-style massive-scale semantic vector store (in-memory mode).

    Implements :class:`~state.memory_tensor.contracts.MemoryStoreBase`.
    """

    __slots__ = (
        "_collection",
        "_dim",
        "_episodes",
        "_index_params",
        "_ip_offset",
        "_max_size",
        "_metric",
    )

    def __init__(
        self,
        *,
        dim: int,
        max_size: int,
        metric_type: MetricType = MetricType.COSINE,
        collection: str = "dix_milvus",
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive int")
        if not isinstance(metric_type, MetricType):
            raise TypeError("metric_type must be a MetricType")
        if not isinstance(collection, str) or not collection:
            raise ValueError("collection must be a non-empty str")
        self._dim = dim
        self._max_size = max_size
        self._metric = metric_type
        self._collection = collection
        self._episodes: dict[str, Episode] = {}
        self._ip_offset: float = 0.0
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
    def collection(self) -> str:
        return self._collection

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
        if self._metric is MetricType.IP:
            ip = _inner_product(episode.embedding, episode.embedding)
            if ip > self._ip_offset:
                self._ip_offset = ip

    def insert(self, episodes: Iterable[Episode]) -> None:
        """Pymilvus ``Collection.insert`` shape.

        Raises on duplicate ids — callers wanting overwrite semantics
        should use :meth:`upsert`.
        """
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
        """Record an index for the audit trail.

        The in-memory backend evaluates search exactly regardless of the
        chosen index type. The live ``pymilvus`` backend would actually
        build the HNSW / IVF graph here.
        """
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
        return self.search_with_filter(query, expr=None)

    def search_with_filter(
        self,
        query: MemoryQuery,
        *,
        expr: BooleanExpression | None,
    ) -> MemoryResult:
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a MemoryQuery")
        if query.dim != self._dim:
            raise ValueError(f"query.dim={query.dim} != store.dim={self._dim}")
        if expr is not None and not isinstance(expr, BooleanExpression):
            raise TypeError("expr must be a BooleanExpression or None")
        scored: list[tuple[float, int, str, Episode]] = []
        for episode in self._episodes.values():
            if expr is not None and not expr.matches(episode.payload):
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
    # Pymilvus-shape ``query`` — exact lookup by expression
    # ------------------------------------------------------------------
    def query(
        self,
        *,
        expr: BooleanExpression,
    ) -> tuple[Episode, ...]:
        """Pymilvus ``Collection.query(expr=...)`` shape — exact lookup."""
        if not isinstance(expr, BooleanExpression):
            raise TypeError("expr must be a BooleanExpression")
        matches: list[Episode] = []
        for episode in self._episodes.values():
            if expr.matches(episode.payload):
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
            "collection": self._collection,
            "dim": self._dim,
            "max_size": self._max_size,
            "metric": self._metric.value,
            "ip_offset": self._ip_offset,
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
    def deserialize(cls, raw: bytes) -> SemanticMilvusStore:
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw must be bytes")
        try:
            obj = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SemanticMilvusError(f"corrupt serialised blob: {exc!s}") from exc
        if not isinstance(obj, dict):
            raise SemanticMilvusError("blob root must be an object")
        version = obj.get("version")
        if version != _SERIALIZATION_VERSION:
            raise SemanticMilvusError(f"unsupported version: {version!r}")
        try:
            metric = MetricType(obj["metric"])
        except (KeyError, ValueError) as exc:
            raise SemanticMilvusError(f"bad metric: {exc!s}") from exc
        store = cls(
            dim=int(obj["dim"]),
            max_size=int(obj["max_size"]),
            metric_type=metric,
            collection=str(obj["collection"]),
        )
        store._ip_offset = float(obj.get("ip_offset", 0.0))
        index_raw = obj.get("index")
        if index_raw is not None:
            try:
                idx_type = IndexType(index_raw["index_type"])
                idx_metric = MetricType(index_raw["metric_type"])
            except (KeyError, ValueError) as exc:
                raise SemanticMilvusError(f"bad index: {exc!s}") from exc
            params_raw = index_raw.get("params", [])
            if not isinstance(params_raw, list):
                raise SemanticMilvusError("index params must be list")
            params_dict: dict[str, int] = {}
            for row in params_raw:
                if not isinstance(row, dict):
                    raise SemanticMilvusError("index param row must be object")
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
                raise SemanticMilvusError("payload must be an object")
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
# Milvus binding (lazy)
# ---------------------------------------------------------------------------
def milvus_client_factory(
    *,
    host: str = "localhost",
    port: int = 19530,
    user: str | None = None,
    password: str | None = None,
    secure: bool = False,
) -> Any:
    """Lazy-bind the ``pymilvus`` package.

    The ``pymilvus`` import is confined to this function body; the rest
    of the module remains importable without the dependency.

    Returns the live ``MilvusClient`` instance; the caller is
    responsible for binding it to a wrapper that satisfies
    :class:`~state.memory_tensor.contracts.MemoryStoreBase` (the wrapper
    is out-of-scope for this leaf — C-23 ships the in-memory pure-Python
    backend; a follow-up wiring PR can swap in the real client when the
    operator provisions a milvus server).
    """
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a non-empty str")
    if not isinstance(port, int) or isinstance(port, bool) or port <= 0:
        raise ValueError("port must be a positive int")
    if user is not None and not isinstance(user, str):
        raise TypeError("user must be a str or None")
    if password is not None and not isinstance(password, str):
        raise TypeError("password must be a str or None")
    if not isinstance(secure, bool):
        raise TypeError("secure must be a bool")
    try:
        from pymilvus import MilvusClient  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when dep absent
        raise SemanticMilvusError("pymilvus is not installed; see NEW_PIP_DEPENDENCIES") from exc
    scheme = "https" if secure else "http"
    uri = f"{scheme}://{host}:{port}"
    token: str | None = None
    if user is not None and password is not None:
        token = f"{user}:{password}"
    if token is None:
        return MilvusClient(uri=uri)
    return MilvusClient(uri=uri, token=token)


__all__ = [
    "BooleanExpression",
    "IndexParams",
    "IndexType",
    "MILVUS_ADAPTER_VERSION",
    "MetricType",
    "NEW_PIP_DEPENDENCIES",
    "SemanticMilvusError",
    "SemanticMilvusStore",
    "milvus_client_factory",
]
