# ADAPTED FROM: facebookresearch/faiss Python interface
# (faiss/python/faiss/__init__.py — IndexFlatL2 (exact L2 nearest-neighbour),
#  Index.add / Index.search / Index.serialize / Index.deserialize on the
#  swigfaiss.py surface)
"""Episodic memory store — FAISS-style exact-L2 backend (S-08.2).

Pure-Python reproduction of the faiss ``IndexFlatL2`` algorithm behind
the :class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol.

Algorithmic surface ported from faiss:

* ``IndexFlatL2.add(vec)`` → :meth:`EpisodicMemoryStore.add`
* ``IndexFlatL2.search(query, k)`` → :meth:`EpisodicMemoryStore.search`
* ``faiss.write_index(idx, fname)`` (bytes form) →
  :meth:`EpisodicMemoryStore.serialize`
* ``faiss.read_index(fname)`` (bytes form) →
  :meth:`EpisodicMemoryStore.deserialize`

We intentionally implement the **flat / exact** path only here. The
approximate IVF path (``IndexIVFFlat``) lives in
:mod:`state.memory_tensor.semantic` (S-08.3) where the cosine /
inner-product distance lane is a better fit. The flat backend is the
right default for stores up to ~10⁴ vectors per the master canonical
doc (S-08, lines 581–583).

Authority constraints (manifest §H1 / S-08 spec):

* **OFFLINE tier** — :meth:`add` is never called from the hot path.
  Authority-lint will not let an execution-tier module import this
  module.
* **RUNTIME-SAFE read path** — :meth:`search` is permitted from
  runtime engines but must complete in <5 ms for typical N. The
  pure-Python brute-force scan over <10⁴ vectors of dim ≤256 fits
  inside that budget on commodity hardware (no numpy import in the
  hot lane keeps cold-start latency low).
* **INV-15 replay determinism** — same inputs → same outputs
  byte-identical. We sort search ties by ``(distance, ts_ns,
  episode_id)``; serialization sorts episodes by ``(ts_ns,
  episode_id)``; ``serialize / deserialize`` round-trips byte-equal.
* **Bounded growth** — :meth:`add` evicts the oldest episode by
  ``(ts_ns, episode_id)`` ascending when the cap would be exceeded.
* **Pure stdlib** — no numpy, no faiss. ``NEW_PIP_DEPENDENCIES = ()``.
  numpy is permitted by the S-08 spec for this tier but adds an
  import-cost / replay-determinism risk we can avoid for the flat
  path.
* **No clock, no PRNG** — every timestamp comes from caller-supplied
  ``Episode.ts_ns`` / ``MemoryQuery.ts_ns``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator

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
# EpisodicMemoryStore
# ---------------------------------------------------------------------------


class EpisodicMemoryStore:
    """FAISS-style ``IndexFlatL2`` episodic memory backend.

    Internal storage:

    * ``_episodes`` — dict keyed by ``episode_id``. Insertion-ordered
      since Python 3.7 but eviction depends on ``(ts_ns, episode_id)``,
      not insertion order, so we re-derive the eviction key every time.

    Search algorithm (the flat path):

    * Compute squared L2 distance from the query to every stored
      embedding. ``math.fsum`` is used over the per-component squared
      diffs so accumulation is order-independent and replay-stable.
    * Take ``sqrt`` to get the L2 distance.
    * Sort hits by ``(distance, ts_ns, episode_id)`` ascending so ties
      are broken deterministically.
    * Truncate to ``query.k``.
    """

    __slots__ = ("_dim", "_max_size", "_episodes")

    # --------------------------------------------------------------- ctor

    def __init__(self, *, dim: int, max_size: int) -> None:
        if not isinstance(dim, int):
            raise TypeError(f"EpisodicMemoryStore.dim must be int, got {type(dim).__name__}")
        if dim <= 0:
            raise ValueError(f"EpisodicMemoryStore.dim must be positive, got {dim!r}")
        if not isinstance(max_size, int):
            raise TypeError(
                f"EpisodicMemoryStore.max_size must be int, got {type(max_size).__name__}"
            )
        if max_size <= 0:
            raise ValueError(f"EpisodicMemoryStore.max_size must be positive, got {max_size!r}")
        self._dim = dim
        self._max_size = max_size
        self._episodes: dict[str, Episode] = {}

    # --------------------------------------------------------------- props

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_size(self) -> int:
        return self._max_size

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
        """Insert ``episode`` into the store, evicting the oldest if full.

        Eviction key: ``(ts_ns, episode_id)`` ascending. The pair
        guarantees deterministic ties — INV-15 byte-identical replay
        across runs and machines.

        Raises:
            ValueError: if ``episode.dim != self.dim``.
            ValueError: if ``episode.episode_id`` is already in the
                store (no overwrite — callers must delete first).
        """

        if not isinstance(episode, Episode):
            raise TypeError(
                f"EpisodicMemoryStore.add expects Episode, got {type(episode).__name__}"
            )
        if episode.dim != self._dim:
            raise ValueError(
                "EpisodicMemoryStore.add dim mismatch: "
                f"store dim={self._dim}, episode dim={episode.dim}"
            )
        if episode.episode_id in self._episodes:
            raise ValueError(
                f"EpisodicMemoryStore.add: episode_id already present: {episode.episode_id!r}"
            )

        if len(self._episodes) >= self._max_size:
            oldest_id = self._oldest_episode_id()
            del self._episodes[oldest_id]

        self._episodes[episode.episode_id] = episode

    def delete(self, episode_id: str) -> bool:
        """Remove an episode by id. Returns ``True`` if it was present."""
        return self._episodes.pop(episode_id, None) is not None

    def _oldest_episode_id(self) -> str:
        """Return the eviction-key-minimum episode id.

        Pulled out as its own method so tests can pin the eviction order
        without reaching into private state.
        """

        return min(
            self._episodes,
            key=lambda eid: (self._episodes[eid].ts_ns, eid),
        )

    # --------------------------------------------------------------- read

    def search(self, query: MemoryQuery) -> MemoryResult:
        """Return up to ``query.k`` nearest neighbours by L2 distance."""

        if not isinstance(query, MemoryQuery):
            raise TypeError(
                f"EpisodicMemoryStore.search expects MemoryQuery, got {type(query).__name__}"
            )
        if query.dim != self._dim:
            raise ValueError(
                "EpisodicMemoryStore.search dim mismatch: "
                f"store dim={self._dim}, query dim={query.dim}"
            )

        # Compute (distance, ts_ns, episode_id, episode) for every row.
        # Tuple key sort gives deterministic ordering on ties.
        scored: list[tuple[float, int, str, Episode]] = []
        q_emb = query.embedding
        for ep_id, ep in self._episodes.items():
            d2 = math.fsum((a - b) * (a - b) for a, b in zip(ep.embedding, q_emb, strict=True))
            d = math.sqrt(d2) if d2 > 0.0 else 0.0
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

        Format (UTF-8 JSON, ``sort_keys=True`` for stability):

        .. code-block:: json

            {
              "version": 1,
              "dim": <int>,
              "max_size": <int>,
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

        Episodes are sorted by ``(ts_ns, episode_id)`` ascending so the
        serialized blob is byte-identical across runs / machines /
        Python instances (INV-15). Payload keys are also sorted via
        ``sort_keys=True``.

        Floats are encoded by ``json.dumps`` using Python's ``repr``,
        which is locale-independent and bit-stable per Python version.
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
            "episodes": episodes_payload,
        }
        return json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, blob: bytes) -> EpisodicMemoryStore:
        """Round-trip the bytes produced by :meth:`serialize`.

        Raises :class:`ValueError` on any structural / type mismatch.
        Validation reuses the same per-field guards the contracts apply
        so partial / corrupt blobs never produce a half-built store.
        """

        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(
                f"EpisodicMemoryStore.deserialize expects bytes, got {type(blob).__name__}"
            )
        try:
            obj = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"EpisodicMemoryStore.deserialize: invalid blob: {exc}") from exc

        if not isinstance(obj, dict):
            raise ValueError("EpisodicMemoryStore.deserialize: top-level must be object")
        version = obj.get("version")
        if version != _SERIALIZATION_VERSION:
            raise ValueError(
                "EpisodicMemoryStore.deserialize: unsupported version "
                f"{version!r} (expected {_SERIALIZATION_VERSION})"
            )
        dim = obj.get("dim")
        max_size = obj.get("max_size")
        episodes = obj.get("episodes")
        if not isinstance(dim, int):
            raise ValueError(
                f"EpisodicMemoryStore.deserialize: 'dim' must be int, got {type(dim).__name__}"
            )
        if not isinstance(max_size, int):
            raise ValueError(
                "EpisodicMemoryStore.deserialize: 'max_size' must be int, "
                f"got {type(max_size).__name__}"
            )
        if not isinstance(episodes, list):
            raise ValueError("EpisodicMemoryStore.deserialize: 'episodes' must be list")

        store = cls(dim=dim, max_size=max_size)
        for i, row in enumerate(episodes):
            if not isinstance(row, dict):
                raise ValueError(f"EpisodicMemoryStore.deserialize: episodes[{i}] must be object")
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
            # Bypass the cap — deserialization restores a previously
            # valid store, so it can never exceed ``max_size``. We still
            # use the public add() so duplicate-id guards run.
            store.add(ep)
        return store


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "EpisodicMemoryStore",
    "NEW_PIP_DEPENDENCIES",
]
