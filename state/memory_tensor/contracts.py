# ADAPTED FROM: facebookresearch/faiss Python interface
# (faiss/python/faiss/__init__.py — IndexFlatL2, IndexIVFFlat, index_factory;
#  faiss/python/swigfaiss.py — index.add(), index.search(),
#  serialize/deserialize)
"""Memory-tensor contracts (S-08, OFFLINE).

Pure value-object surface for the vector memory tier. Concrete backends
(FAISS exact, FAISS-IVF, Qdrant, brute-force, ...) implement the
:class:`MemoryStoreBase` Protocol; engines and lanes only ever depend
on the contracts in this module.

Authority constraints (manifest §H1, INV-15, INV-08):

* Pure data + a Protocol — no engine cross-imports, no clock, no PRNG,
  no IO. The whole module is reachable from any tier without violating
  the authority-lint rules T1 / B1 / B20 / L2 / L3.
* Every dataclass is frozen + slotted so memory records are immutable
  audit rows.
* Every field is range-checked in ``__post_init__`` so corrupt rows
  cannot enter the store.
* Embeddings are stored as ``tuple[float, ...]`` so:

    * Records hash structurally (replay determinism, INV-15).
    * Records cross domain boundaries without numpy in scope (INV-08).
    * Backends that prefer numpy convert at the edge.

* No floating-point tolerance lives in the contracts; the backends
  decide what "near" means.

Refs:
  - manifest §H1 (state/memory_tensor tree)
  - DIX_MASTER_CANONICAL.md S-08 (faiss-cpu, lines 562–600)
  - reference patterns: ``core.contracts.simulation``,
    ``core.contracts.portfolio``
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Embedding validator (shared by the contracts and the backends)
# ---------------------------------------------------------------------------


def validate_embedding(
    embedding: tuple[float, ...],
    *,
    field: str,
) -> None:
    """Raise :class:`ValueError` unless ``embedding`` is a clean vector.

    A clean vector is:

    * a tuple (not a list, not a numpy array — those are caller-side),
    * non-empty,
    * built from finite ``float`` components only (no ``NaN`` /
      ``±inf`` slipping into the index — corrupt rows poison every
      neighbour they ever return).

    The ``field`` argument is folded into the exception message so the
    caller knows which field tripped (``Episode.embedding`` vs.
    ``MemoryQuery.embedding``).
    """

    if not isinstance(embedding, tuple):
        raise TypeError(f"{field} must be a tuple of floats, got {type(embedding).__name__}")
    if len(embedding) == 0:
        raise ValueError(f"{field} must not be empty")
    for i, v in enumerate(embedding):
        if not isinstance(v, float):
            raise TypeError(f"{field}[{i}] must be float, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ValueError(f"{field}[{i}] must be finite, got {v!r}")


# ---------------------------------------------------------------------------
# Episode — one row in the memory tensor.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Episode:
    """One frozen point in the memory tensor.

    Stores an embedding vector keyed by an opaque ``episode_id``. The
    payload carries auxiliary stringified metadata (regime tag, source
    strategy, post-trade summary, ...). The store does NOT interpret
    payload contents — they round-trip verbatim.

    Fields:
        ts_ns: nanosecond timestamp of the underlying observation.
            Eviction is keyed by this field (oldest first).
        episode_id: opaque, non-empty identifier. Used as the primary
            key for lookups, deletes, and hit-row referencing.
        embedding: dense vector. Must be a tuple of finite floats. The
            tuple's length is the dimension; the store keeps it stable
            across all rows.
        payload: ledger-stringified metadata. Frozen at construction.
    """

    ts_ns: int
    episode_id: str
    embedding: tuple[float, ...]
    payload: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError(f"Episode.ts_ns must be positive, got {self.ts_ns!r}")
        if not self.episode_id:
            raise ValueError("Episode.episode_id must be non-empty")
        validate_embedding(self.embedding, field="Episode.embedding")
        for k, v in self.payload.items():
            if not isinstance(k, str):
                raise TypeError(f"Episode.payload keys must be str, got {type(k).__name__}")
            if not isinstance(v, str):
                raise TypeError(f"Episode.payload[{k!r}] must be str, got {type(v).__name__}")
        # Freeze the payload so callers cannot mutate it after the
        # episode is stored. Re-uses the same MappingProxyType the
        # default factory would have built.
        if not isinstance(self.payload, MappingProxyType):
            object.__setattr__(
                self,
                "payload",
                MappingProxyType(dict(self.payload)),
            )

    @property
    def dim(self) -> int:
        """Dimensionality of :attr:`embedding`."""
        return len(self.embedding)


# ---------------------------------------------------------------------------
# MemoryQuery — a search request.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryQuery:
    """A nearest-neighbour search request against a memory store.

    Fields:
        ts_ns: nanosecond timestamp of the query (caller-supplied).
        query_id: opaque, non-empty correlation id. The store echoes
            this back on :class:`MemoryResult` so async callers can
            join responses to requests.
        embedding: dense query vector. Same dimensional contract as
            :attr:`Episode.embedding`.
        k: top-k neighbours requested. Must be a positive int. The
            store may return fewer rows if it holds fewer than ``k``
            episodes; it must never return more.
    """

    ts_ns: int
    query_id: str
    embedding: tuple[float, ...]
    k: int

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError(f"MemoryQuery.ts_ns must be positive, got {self.ts_ns!r}")
        if not self.query_id:
            raise ValueError("MemoryQuery.query_id must be non-empty")
        validate_embedding(
            self.embedding,
            field="MemoryQuery.embedding",
        )
        if self.k <= 0:
            raise ValueError(f"MemoryQuery.k must be positive, got {self.k!r}")

    @property
    def dim(self) -> int:
        """Dimensionality of :attr:`embedding`."""
        return len(self.embedding)


# ---------------------------------------------------------------------------
# MemoryHit / MemoryResult — the search response.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryHit:
    """One row in a :class:`MemoryResult`.

    Fields:
        episode_id: id of the matched :class:`Episode`.
        distance: backend-defined distance metric. Must be finite and
            non-negative (L2, cosine-distance, inner-product distance
            all satisfy this; raw inner-product would not, so backends
            that work in similarity space must convert at the edge).
        ts_ns: timestamp of the matched episode (so consumers can
            order hits by recency without a second lookup).
        payload: copy of the matched episode's payload, frozen.
    """

    episode_id: str
    distance: float
    ts_ns: int
    payload: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("MemoryHit.episode_id must be non-empty")
        if not isinstance(self.distance, float):
            raise TypeError(f"MemoryHit.distance must be float, got {type(self.distance).__name__}")
        if not math.isfinite(self.distance):
            raise ValueError(f"MemoryHit.distance must be finite, got {self.distance!r}")
        if self.distance < 0.0:
            raise ValueError(f"MemoryHit.distance must be non-negative, got {self.distance!r}")
        if self.ts_ns <= 0:
            raise ValueError(f"MemoryHit.ts_ns must be positive, got {self.ts_ns!r}")
        for k, v in self.payload.items():
            if not isinstance(k, str):
                raise TypeError(f"MemoryHit.payload keys must be str, got {type(k).__name__}")
            if not isinstance(v, str):
                raise TypeError(f"MemoryHit.payload[{k!r}] must be str, got {type(v).__name__}")
        if not isinstance(self.payload, MappingProxyType):
            object.__setattr__(
                self,
                "payload",
                MappingProxyType(dict(self.payload)),
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryResult:
    """A search response.

    Fields:
        ts_ns: timestamp at which the search ran (caller-supplied so
            replays stay byte-identical, INV-15).
        query_id: id of the originating :class:`MemoryQuery`.
        hits: tuple of :class:`MemoryHit`, **sorted by ascending
            distance** (best match first). Length is ``min(k,
            store_size)``; never exceeds ``k``.
    """

    ts_ns: int
    query_id: str
    hits: tuple[MemoryHit, ...]

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError(f"MemoryResult.ts_ns must be positive, got {self.ts_ns!r}")
        if not self.query_id:
            raise ValueError("MemoryResult.query_id must be non-empty")
        if not isinstance(self.hits, tuple):
            raise TypeError(f"MemoryResult.hits must be a tuple, got {type(self.hits).__name__}")
        previous: float = -1.0
        for i, h in enumerate(self.hits):
            if not isinstance(h, MemoryHit):
                raise TypeError(f"MemoryResult.hits[{i}] must be MemoryHit, got {type(h).__name__}")
            if h.distance < previous:
                raise ValueError(
                    "MemoryResult.hits must be sorted by ascending "
                    "distance, but "
                    f"hits[{i}].distance={h.distance!r} < "
                    f"hits[{i - 1}].distance={previous!r}"
                )
            previous = h.distance


# ---------------------------------------------------------------------------
# MemoryStoreBase — Protocol satisfied by every backend.
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStoreBase(Protocol):
    """Protocol for vector-memory backends.

    Concrete backends land in :mod:`state.memory_tensor.episodic` (S-08.2,
    FAISS exact / IVF) and :mod:`state.memory_tensor.semantic` (S-08.3,
    same Protocol with a different distance metric).

    Implementations must satisfy:

    * **Replay determinism (INV-15):** ``add`` then ``search`` with the
      same arguments returns the same :class:`MemoryResult` byte-for-byte
      across runs and across machines. No clock reads, no PRNG.
    * **Bounded growth:** ``len(store)`` ≤ ``max_size``. When ``add``
      would exceed the cap, the oldest episode by ``ts_ns`` is evicted
      first; ties broken by ``episode_id`` (lexicographic). Eviction
      is deterministic.
    * **OFFLINE write path:** :meth:`add` is *not* permitted on the
      hot path. Authority-lint will not let an execution-tier module
      import a concrete backend.
    * **RUNTIME-SAFE read path:** :meth:`search` may be called from
      runtime engines but must complete in <5 ms for typical N
      (per S-08 spec).
    * **Checkpointable:** :meth:`serialize` returns the full state as
      bytes; :meth:`load` round-trips byte-equal.
    """

    @property
    def dim(self) -> int:
        """Embedding dimension of every stored episode."""
        ...

    @property
    def max_size(self) -> int:
        """Hard upper bound on stored episode count."""
        ...

    def __len__(self) -> int:
        """Number of episodes currently in the store."""
        ...

    def __contains__(self, episode_id: str) -> bool:
        """Whether ``episode_id`` is in the store."""
        ...

    def add(self, episode: Episode) -> None:
        """Insert one episode into the store.

        If ``len(self) == self.max_size`` then the oldest episode (by
        ``ts_ns`` ascending, ``episode_id`` ascending on tie) is
        evicted before the new one is inserted.

        Raises :class:`ValueError` if ``episode.dim != self.dim`` or
        if ``episode.episode_id`` is already in the store.
        """
        ...

    def search(self, query: MemoryQuery) -> MemoryResult:
        """Return up to ``query.k`` nearest neighbours.

        Hits are sorted by ascending distance. The returned
        :class:`MemoryResult.ts_ns` echoes the query's ``ts_ns``.
        """
        ...

    def serialize(self) -> bytes:
        """Return a deterministic byte representation of the store."""
        ...


# ---------------------------------------------------------------------------
# Public surface.
# ---------------------------------------------------------------------------


__all__ = [
    "Episode",
    "MemoryHit",
    "MemoryQuery",
    "MemoryResult",
    "MemoryStoreBase",
    "validate_embedding",
]
