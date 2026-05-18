# ADAPTED FROM: facebookresearch/faiss Python interface
# (faiss/python/faiss/__init__.py — IndexFlatIP (inner-product flat),
#  IndexIVFFlat (Voronoi-cell approximate index), Index.add /
#  Index.search / Index.serialize / Index.deserialize on the
#  swigfaiss.py surface)
"""Semantic memory store — FAISS-style cosine / IVF backend (S-08.3).

Pure-Python reproduction of the faiss inner-product semantic-search
path behind the
:class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol. This
is the third and final leaf of S-08; it complements the L2-distance
:class:`~state.memory_tensor.episodic.EpisodicMemoryStore` from S-08.2.

Algorithmic surface ported from faiss:

* ``IndexFlatIP.add(vec)`` and ``.search(query, k)`` — exact inner-
  product NN over **L2-normalised** embeddings, which is equivalent to
  cosine similarity. The flat path is the default backend.
* ``IndexIVFFlat`` — Voronoi-cell partitioning. Each stored embedding
  is bucketed at insert time to the nearest caller-supplied centroid;
  search ranks centroids by cosine to the query and exhaustively
  scans only the top ``nprobe`` buckets. We do **not** train
  centroids at runtime (faiss's k-means is non-deterministic without
  fixed seeding). Instead the caller passes a frozen tuple of
  centroids — typically clustered offline by an OFFLINE-tier
  pipeline.
* ``faiss.write_index`` / ``faiss.read_index`` (bytes form) →
  :meth:`SemanticMemoryStore.serialize` /
  :meth:`SemanticMemoryStore.deserialize`.

Distance lane: cosine **distance** ``d = 1 - cos(a, b)``. For unit
vectors ``cos`` ∈ ``[-1, 1]`` so ``d`` ∈ ``[0, 2]`` — naturally non-
negative, which satisfies the ``MemoryHit.distance >= 0`` invariant
from S-08.1. Identical embeddings have ``d == 0``; perfectly opposing
embeddings have ``d == 2``. Zero-norm embeddings are handled by
falling back to ``cos = 0`` (i.e. ``d = 1``) — they never participate
in the ranking with any defined direction.

Authority constraints (manifest §H1 / S-08 spec):

* **OFFLINE tier write** — :meth:`add` is never called from the hot
  path. Authority-lint will not let an execution-tier module import
  this module.
* **RUNTIME-SAFE read path** — :meth:`search` is permitted from
  runtime engines and must complete in <5 ms for typical N. The
  pure-Python brute-force flat scan over <10⁴ vectors of dim ≤256
  fits inside that budget; the IVF path tightens that further by
  ``nprobe / nlist``.
* **INV-15 replay determinism** — same inputs → same outputs
  byte-identical. Tie-breaking on
  ``(distance, ts_ns, episode_id)``; serialization sorts episodes
  by ``(ts_ns, episode_id)``; ``serialize / deserialize`` round-trips
  byte-equal.
* **No clock, no PRNG** — every timestamp comes from
  caller-supplied ``Episode.ts_ns`` / ``MemoryQuery.ts_ns``. No
  ``random`` / ``secrets`` — centroids are caller-supplied, not
  trained at runtime.
* **Pure stdlib** — no numpy, no faiss. ``NEW_PIP_DEPENDENCIES =
  ()``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence

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

#: pip dependencies introduced by this module. Strict canonical rule
#: (PART 1 §10) — flag every new dep. Empty here: pure stdlib.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()

#: Serialization-format version. Bumped on any wire-format change.
_SERIALIZATION_VERSION = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _l2_norm(vec: Sequence[float]) -> float:
    """Return the L2 norm of ``vec``. ``math.fsum`` for replay stability."""
    return math.sqrt(math.fsum(x * x for x in vec))


def _cosine_distance(
    a: Sequence[float],
    b: Sequence[float],
    norm_a: float,
    norm_b: float,
) -> float:
    """Return cosine distance ``1 - cos(a, b)``.

    Zero-norm guard: if either norm is zero, ``cos`` is undefined; we
    treat the pair as orthogonal (``cos = 0``, ``d = 1``). This keeps
    the value within ``[0, 2]`` and never feeds NaN into the result.
    """
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 1.0
    ip = math.fsum(x * y for x, y in zip(a, b, strict=True))
    cos = ip / (norm_a * norm_b)
    # Clamp against rounding overshoot (cos can numerically reach
    # 1.000000001 for unit-pair embeddings due to FP rounding).
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return 1.0 - cos


# ---------------------------------------------------------------------------
# SemanticMemoryStore
# ---------------------------------------------------------------------------


class SemanticMemoryStore:
    """FAISS-style ``IndexFlatIP`` + optional ``IndexIVFFlat`` backend.

    Operating modes:

    * **Flat (default)** — brute-force cosine over every stored
      embedding. Activated when ``centroids is None``.
    * **IVF (approximate)** — caller supplies a frozen tuple of
      ``centroids``; each ``add`` buckets the episode to its closest
      centroid; each ``search`` ranks centroids and scans only the
      top ``nprobe`` buckets. Activated when ``centroids`` is non-
      empty.

    The two modes share storage (``_episodes``) and the bucket map
    (``_buckets``). Switching modes after construction is not
    supported — re-create the store with a different ``centroids``
    argument.
    """

    __slots__ = (
        "_dim",
        "_max_size",
        "_episodes",
        "_norms",
        "_centroids",
        "_centroid_norms",
        "_nprobe",
        "_buckets",
    )

    # --------------------------------------------------------------- ctor

    def __init__(
        self,
        *,
        dim: int,
        max_size: int,
        centroids: Iterable[tuple[float, ...]] | None = None,
        nprobe: int = 1,
    ) -> None:
        if not isinstance(dim, int):
            raise TypeError(f"SemanticMemoryStore.dim must be int, got {type(dim).__name__}")
        if dim <= 0:
            raise ValueError(f"SemanticMemoryStore.dim must be positive, got {dim!r}")
        if not isinstance(max_size, int):
            raise TypeError(
                f"SemanticMemoryStore.max_size must be int, got {type(max_size).__name__}"
            )
        if max_size <= 0:
            raise ValueError(f"SemanticMemoryStore.max_size must be positive, got {max_size!r}")
        if not isinstance(nprobe, int):
            raise TypeError(f"SemanticMemoryStore.nprobe must be int, got {type(nprobe).__name__}")
        if nprobe <= 0:
            raise ValueError(f"SemanticMemoryStore.nprobe must be positive, got {nprobe!r}")

        # Centroids: validate every row (same guards as Episode.embedding).
        validated_centroids: tuple[tuple[float, ...], ...]
        if centroids is None:
            validated_centroids = ()
        else:
            tmp: list[tuple[float, ...]] = []
            for i, c in enumerate(centroids):
                if not isinstance(c, tuple):
                    raise TypeError(
                        f"SemanticMemoryStore.centroids[{i}] must be tuple, got {type(c).__name__}"
                    )
                validate_embedding(c, field=f"SemanticMemoryStore.centroids[{i}]")
                if len(c) != dim:
                    raise ValueError(
                        f"SemanticMemoryStore.centroids[{i}] has dim {len(c)}, expected {dim}"
                    )
                tmp.append(c)
            validated_centroids = tuple(tmp)

        if validated_centroids and nprobe > len(validated_centroids):
            raise ValueError(
                "SemanticMemoryStore.nprobe must be ≤ len(centroids); "
                f"got nprobe={nprobe}, len(centroids)={len(validated_centroids)}"
            )

        self._dim = dim
        self._max_size = max_size
        self._episodes: dict[str, Episode] = {}
        self._norms: dict[str, float] = {}
        self._centroids = validated_centroids
        self._centroid_norms: tuple[float, ...] = tuple(_l2_norm(c) for c in validated_centroids)
        self._nprobe = nprobe
        # episode_id -> centroid index (only populated in IVF mode)
        self._buckets: dict[str, int] = {}

    # --------------------------------------------------------------- props

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def centroids(self) -> tuple[tuple[float, ...], ...]:
        return self._centroids

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @property
    def is_ivf(self) -> bool:
        return bool(self._centroids)

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, episode_id: object) -> bool:
        return isinstance(episode_id, str) and episode_id in self._episodes

    def __iter__(self) -> Iterator[Episode]:
        """Iterate episodes in deterministic ``(ts_ns, episode_id)`` order."""
        for ep_id in sorted(
            self._episodes,
            key=lambda eid: (self._episodes[eid].ts_ns, eid),
        ):
            yield self._episodes[ep_id]

    # --------------------------------------------------------------- write

    def add(self, episode: Episode) -> None:
        """Insert ``episode``; evict the oldest by ``(ts_ns, episode_id)``."""
        if not isinstance(episode, Episode):
            raise TypeError(
                f"SemanticMemoryStore.add expects Episode, got {type(episode).__name__}"
            )
        if episode.dim != self._dim:
            raise ValueError(
                "SemanticMemoryStore.add dim mismatch: "
                f"store dim={self._dim}, episode dim={episode.dim}"
            )
        if episode.episode_id in self._episodes:
            raise ValueError(
                f"SemanticMemoryStore.add: episode_id already present: {episode.episode_id!r}"
            )

        if len(self._episodes) >= self._max_size:
            oldest_id = self._oldest_episode_id()
            del self._episodes[oldest_id]
            del self._norms[oldest_id]
            self._buckets.pop(oldest_id, None)

        self._episodes[episode.episode_id] = episode
        self._norms[episode.episode_id] = _l2_norm(episode.embedding)
        if self._centroids:
            self._buckets[episode.episode_id] = self._best_centroid(episode.embedding)

    def delete(self, episode_id: str) -> bool:
        """Remove an episode by id. Returns ``True`` if it was present."""
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            del self._norms[episode_id]
            self._buckets.pop(episode_id, None)
            return True
        return False

    def _oldest_episode_id(self) -> str:
        return min(
            self._episodes,
            key=lambda eid: (self._episodes[eid].ts_ns, eid),
        )

    def _best_centroid(self, vec: tuple[float, ...]) -> int:
        """Return the index of the centroid closest to ``vec`` by cosine.

        Ties on cosine distance are broken by centroid index ascending,
        which mirrors faiss's stable ordering and pins INV-15.
        """
        norm_v = _l2_norm(vec)
        ranked: list[tuple[float, int]] = []
        for i, c in enumerate(self._centroids):
            d = _cosine_distance(vec, c, norm_v, self._centroid_norms[i])
            ranked.append((d, i))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return ranked[0][1]

    # --------------------------------------------------------------- read

    def search(self, query: MemoryQuery) -> MemoryResult:
        """Return up to ``query.k`` nearest neighbours by cosine distance."""
        if not isinstance(query, MemoryQuery):
            raise TypeError(
                f"SemanticMemoryStore.search expects MemoryQuery, got {type(query).__name__}"
            )
        if query.dim != self._dim:
            raise ValueError(
                "SemanticMemoryStore.search dim mismatch: "
                f"store dim={self._dim}, query dim={query.dim}"
            )

        q_emb = query.embedding
        norm_q = _l2_norm(q_emb)

        # Decide which episode_ids to score.
        if self._centroids:
            # Rank centroids by cosine to the query, take top nprobe.
            centroid_rank: list[tuple[float, int]] = []
            for i, c in enumerate(self._centroids):
                d = _cosine_distance(q_emb, c, norm_q, self._centroid_norms[i])
                centroid_rank.append((d, i))
            centroid_rank.sort(key=lambda t: (t[0], t[1]))
            probed = {idx for _, idx in centroid_rank[: self._nprobe]}
            candidates = [ep_id for ep_id, c_idx in self._buckets.items() if c_idx in probed]
        else:
            candidates = list(self._episodes.keys())

        scored: list[tuple[float, int, str, Episode]] = []
        for ep_id in candidates:
            ep = self._episodes[ep_id]
            d = _cosine_distance(ep.embedding, q_emb, self._norms[ep_id], norm_q)
            scored.append((d, ep.ts_ns, ep_id, ep))

        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        top_k = scored[: query.k]

        hits = tuple(
            MemoryHit(
                episode_id=ep_id,
                distance=d,
                ts_ns=ts_ns,
                payload=ep.payload,
            )
            for d, ts_ns, ep_id, ep in top_k
        )
        return MemoryResult(
            ts_ns=query.ts_ns,
            query_id=query.query_id,
            hits=hits,
        )

    # --------------------------------------------------------------- check

    def serialize(self) -> bytes:
        """Return a deterministic byte representation.

        Format (UTF-8 JSON, ``sort_keys=True``):

        .. code-block:: json

            {
              "version": 1,
              "dim": <int>,
              "max_size": <int>,
              "nprobe": <int>,
              "centroids": [[<float>, ...], ...],
              "episodes": [
                {
                  "ts_ns": <int>,
                  "episode_id": <str>,
                  "embedding": [<float>, ...],
                  "payload": {<str>: <str>, ...}
                },
                ...
              ]
            }

        Episodes are sorted by ``(ts_ns, episode_id)`` ascending —
        byte-identical across runs / machines / Python instances
        (INV-15). Centroids preserve their original order so bucket
        indices remain stable on round-trip.
        """
        episodes_payload = [
            {
                "ts_ns": ep.ts_ns,
                "episode_id": ep.episode_id,
                "embedding": list(ep.embedding),
                "payload": dict(ep.payload),
            }
            for ep in self
        ]
        blob = {
            "version": _SERIALIZATION_VERSION,
            "dim": self._dim,
            "max_size": self._max_size,
            "nprobe": self._nprobe,
            "centroids": [list(c) for c in self._centroids],
            "episodes": episodes_payload,
        }
        return json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, blob: bytes) -> SemanticMemoryStore:
        """Round-trip the bytes produced by :meth:`serialize`."""
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(
                f"SemanticMemoryStore.deserialize expects bytes, got {type(blob).__name__}"
            )
        try:
            obj = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"SemanticMemoryStore.deserialize: invalid blob: {exc}") from exc

        if not isinstance(obj, dict):
            raise ValueError("SemanticMemoryStore.deserialize: top-level must be object")
        version = obj.get("version")
        if version != _SERIALIZATION_VERSION:
            raise ValueError(
                "SemanticMemoryStore.deserialize: unsupported version "
                f"{version!r} (expected {_SERIALIZATION_VERSION})"
            )
        dim = obj.get("dim")
        max_size = obj.get("max_size")
        nprobe = obj.get("nprobe", 1)
        centroids_raw = obj.get("centroids", [])
        episodes = obj.get("episodes")
        if not isinstance(dim, int):
            raise ValueError(
                f"SemanticMemoryStore.deserialize: 'dim' must be int, got {type(dim).__name__}"
            )
        if not isinstance(max_size, int):
            raise ValueError(
                "SemanticMemoryStore.deserialize: 'max_size' must be int, "
                f"got {type(max_size).__name__}"
            )
        if not isinstance(nprobe, int):
            raise ValueError(
                "SemanticMemoryStore.deserialize: 'nprobe' must be int, "
                f"got {type(nprobe).__name__}"
            )
        if not isinstance(centroids_raw, list):
            raise ValueError("SemanticMemoryStore.deserialize: 'centroids' must be list")
        if not isinstance(episodes, list):
            raise ValueError("SemanticMemoryStore.deserialize: 'episodes' must be list")

        centroids: list[tuple[float, ...]] = []
        for i, row in enumerate(centroids_raw):
            if not isinstance(row, list):
                raise ValueError(f"SemanticMemoryStore.deserialize: centroids[{i}] must be list")
            centroids.append(tuple(float(x) for x in row))

        store = cls(
            dim=dim,
            max_size=max_size,
            centroids=tuple(centroids) if centroids else None,
            nprobe=nprobe,
        )
        for i, row in enumerate(episodes):
            if not isinstance(row, dict):
                raise ValueError(f"SemanticMemoryStore.deserialize: episodes[{i}] must be object")
            ts_ns = row.get("ts_ns")
            episode_id = row.get("episode_id")
            embedding = row.get("embedding")
            payload = row.get("payload", {})
            if not isinstance(ts_ns, int):
                raise ValueError(f"episodes[{i}].ts_ns must be int, got {type(ts_ns).__name__}")
            if not isinstance(episode_id, str):
                raise ValueError(
                    f"episodes[{i}].episode_id must be str, got {type(episode_id).__name__}"
                )
            if not isinstance(embedding, list):
                raise ValueError(f"episodes[{i}].embedding must be list")
            emb_tuple = tuple(float(x) for x in embedding)
            validate_embedding(emb_tuple, field=f"episodes[{i}].embedding")
            if not isinstance(payload, dict):
                raise ValueError(f"episodes[{i}].payload must be object")
            ep = Episode(
                ts_ns=ts_ns,
                episode_id=episode_id,
                embedding=emb_tuple,
                payload={str(k): str(v) for k, v in payload.items()},
            )
            store.add(ep)
        return store


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "SemanticMemoryStore",
    "NEW_PIP_DEPENDENCIES",
]
