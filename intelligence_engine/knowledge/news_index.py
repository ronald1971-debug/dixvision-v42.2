"""Deterministic in-memory news similarity index (D4).

The index stores one row per :class:`core.contracts.news.NewsItem`
(keyed by ``(source, guid)``), backed by a bag-of-words token vector
extracted from ``title + " " + summary``. Queries score every stored
row against a query token vector via sparse cosine similarity.

Determinism contract:

* No clocks. The "newness" axis comes from caller-supplied
  ``NewsItem.ts_ns``.
* No PRNG. Tie-breaking on equal cosine score sorts by ``ts_ns``
  descending, then ``guid`` ascending — both stable.
* No I/O. The index never reads or writes disk on its own.

Bounded growth: ``max_items`` (default 4096) caps memory; the oldest
row by ``ts_ns`` is evicted when the cap is reached. The eviction set
is reproducible because ``ts_ns`` ties are broken by ``guid`` ascending.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from core.contracts.news import NewsItem

#: Bumped on any change to tokenization / scoring so callers can pin
#: a known-good projection version in their audit ledger.
KNOWLEDGE_INDEX_VERSION = "v1"

#: Soft default — operators can override per-instance.
DEFAULT_MAX_ITEMS = 4096

#: Tokens shorter than this are dropped to suppress single-letter noise
#: (``a``, ``i``, …) without inviting locale-specific stopword lists
#: that would couple the index to one language.
MIN_TOKEN_LEN = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    lowered = text.lower()
    out: list[str] = []
    for match in _TOKEN_RE.finditer(lowered):
        tok = match.group(0)
        if len(tok) < MIN_TOKEN_LEN:
            continue
        out.append(tok)
    return tuple(out)


def _vectorize(tokens: Iterable[str]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def _norm(vec: Mapping[str, int]) -> float:
    if not vec:
        return 0.0
    return math.sqrt(sum(c * c for c in vec.values()))


def _cosine(
    a: Mapping[str, int],
    a_norm: float,
    b: Mapping[str, int],
    b_norm: float,
) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    # Iterate the smaller map for the dot product.
    if len(a) > len(b):
        a, b = b, a
    dot = 0
    for tok, ca in a.items():
        cb = b.get(tok)
        if cb is None:
            continue
        dot += ca * cb
    return dot / (a_norm * b_norm)


@dataclass(frozen=True, slots=True)
class SimilarityHit:
    """One row of a :meth:`NewsKnowledgeIndex.query` result.

    Attributes:
        score: Cosine similarity in ``[0.0, 1.0]``. ``0.0`` means the
            query and stored row share no tokens.
        item: The matched :class:`NewsItem` (frozen — same row that
            was added).
    """

    score: float
    item: NewsItem


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Cheap snapshot of the index for ops dashboards.

    Attributes:
        size: Number of stored rows.
        max_items: Upper bound on size (eviction threshold).
        version: :data:`KNOWLEDGE_INDEX_VERSION` at construction time.
        unique_tokens: Number of distinct tokens across all stored rows.
        unique_sources: Number of distinct ``NewsItem.source`` values.
    """

    size: int
    max_items: int
    version: str
    unique_tokens: int
    unique_sources: int


@dataclass(slots=True)
class _Row:
    item: NewsItem
    vec: Mapping[str, int]
    norm: float
    sort_key: tuple[int, str] = field(init=False)

    def __post_init__(self) -> None:
        # Ascending eviction order: oldest ts_ns first; on tie, lower
        # guid first. Stored once because every comparison would
        # otherwise rebuild the tuple.
        self.sort_key = (self.item.ts_ns, self.item.guid)


class NewsKnowledgeIndex:
    """Deterministic bag-of-words similarity index for news headlines.

    Thread safety: not safe — the intended call site is single-threaded
    per replay tick. The slow-loop learner owns its own instance.

    Args:
        max_items: Upper bound on stored rows. When exceeded, the
            oldest row by ``(ts_ns, guid)`` is evicted (reproducible
            because both fields are deterministic).
    """

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        if max_items <= 0:
            raise ValueError(f"max_items must be > 0, got {max_items}")
        self._max_items = max_items
        # Keyed by (source, guid) so the same headline cannot land twice.
        self._rows: dict[tuple[str, str], _Row] = {}

    @property
    def max_items(self) -> int:
        return self._max_items

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, item: NewsItem) -> bool:
        """Index one news item.

        Returns ``True`` if the item was newly added, ``False`` if a row
        with the same ``(source, guid)`` already exists. Identical
        re-adds are a no-op so the slow-loop learner can replay a
        ledger window without double-counting.
        """

        if not item.source:
            raise ValueError("NewsItem.source must be non-empty")
        if not item.guid:
            raise ValueError("NewsItem.guid must be non-empty")
        key = (item.source, item.guid)
        if key in self._rows:
            return False
        tokens = _tokenize(item.title + " " + item.summary)
        vec = _vectorize(tokens)
        row = _Row(item=item, vec=vec, norm=_norm(vec))
        self._rows[key] = row
        if len(self._rows) > self._max_items:
            self._evict_oldest()
        return True

    def add_many(self, items: Iterable[NewsItem]) -> int:
        added = 0
        for it in items:
            if self.add(it):
                added += 1
        return added

    def drop(self, source: str, guid: str) -> bool:
        return self._rows.pop((source, guid), None) is not None

    def clear(self) -> None:
        self._rows.clear()

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        source: str | None = None,
    ) -> tuple[SimilarityHit, ...]:
        """Return the top-``k`` matches for ``text``.

        Args:
            text: Free-form query string. Tokenized with the same rules
                used for indexed items so vocabulary matches.
            top_k: Maximum number of hits to return. Must be ``> 0``.
            min_score: Drop hits with cosine score below this threshold.
            source: Optional filter restricting matches to one news
                source (e.g. ``"COINDESK"``).

        Returns:
            Tuple of :class:`SimilarityHit` ordered by
            ``(-score, -ts_ns, guid)``. Empty when no row scores above
            ``min_score``.
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        q_tokens = _tokenize(text)
        if not q_tokens:
            return ()
        q_vec = _vectorize(q_tokens)
        q_norm = _norm(q_vec)
        if q_norm == 0.0:
            return ()
        scored: list[tuple[float, int, str, NewsItem]] = []
        for (src, guid), row in self._rows.items():
            if source is not None and src != source:
                continue
            score = _cosine(q_vec, q_norm, row.vec, row.norm)
            if score < min_score:
                continue
            # Negative ts_ns so sort ascending == newest first; guid
            # tie-break sorts ascending, deterministic.
            scored.append((-score, -row.item.ts_ns, guid, row.item))
        scored.sort(key=lambda r: (r[0], r[1], r[2]))
        out: list[SimilarityHit] = []
        for neg_score, _neg_ts, _guid, item in scored[:top_k]:
            out.append(SimilarityHit(score=-neg_score, item=item))
        return tuple(out)

    def sources(self) -> tuple[str, ...]:
        """Return distinct ``NewsItem.source`` values, sorted ascending."""
        return tuple(sorted({src for (src, _guid) in self._rows.keys()}))

    def stats(self) -> IndexStats:
        unique_tokens: set[str] = set()
        unique_sources: set[str] = set()
        for (src, _guid), row in self._rows.items():
            unique_sources.add(src)
            unique_tokens.update(row.vec.keys())
        return IndexStats(
            size=len(self._rows),
            max_items=self._max_items,
            version=KNOWLEDGE_INDEX_VERSION,
            unique_tokens=len(unique_tokens),
            unique_sources=len(unique_sources),
        )

    # -- internal ----------------------------------------------------------

    def _evict_oldest(self) -> None:
        # O(n) but only runs when len > max_items; replay determinism is
        # preserved because sort_key is fully derived from NewsItem.
        oldest_key: tuple[str, str] | None = None
        oldest_sort: tuple[int, str] | None = None
        for key, row in self._rows.items():
            if oldest_sort is None or row.sort_key < oldest_sort:
                oldest_sort = row.sort_key
                oldest_key = key
        if oldest_key is not None:
            del self._rows[oldest_key]


__all__ = [
    "DEFAULT_MAX_ITEMS",
    "KNOWLEDGE_INDEX_VERSION",
    "MIN_TOKEN_LEN",
    "IndexStats",
    "NewsKnowledgeIndex",
    "SimilarityHit",
]
