"""News knowledge subsystem (D4).

Deterministic, replay-safe news indexing + similarity search. The index
is a pure-Python bag-of-words cosine engine — no FAISS C++ binding, no
embedding model, no random init. Every state mutation is anchored on
caller-supplied :class:`core.contracts.news.NewsItem` rows, so two
replays of the same NewsItem stream produce byte-identical query
results.

Authority constraints (lint-checked):

* No clocks (``time.time``, ``time.monotonic``, ``wall_ns()``) — all
  timestamps come from the caller-supplied :class:`NewsItem.ts_ns`.
* No PRNG — tie-breaking uses lexicographic ordering on ``guid`` so
  the result of ``query()`` is fully deterministic.
* No I/O — index lives in memory; persistence is the caller's concern.

Public surface:

* :class:`NewsKnowledgeIndex` — add / query / drop with bounded size.
* :class:`SimilarityHit` — frozen result row carrying the matched
  NewsItem + cosine score.
* :class:`IndexStats` — cheap snapshot for ops dashboards.

INV-15 (replay determinism) is preserved end-to-end.
"""

from intelligence_engine.knowledge.news_index import (
    KNOWLEDGE_INDEX_VERSION,
    IndexStats,
    NewsKnowledgeIndex,
    SimilarityHit,
)

__all__ = [
    "KNOWLEDGE_INDEX_VERSION",
    "IndexStats",
    "NewsKnowledgeIndex",
    "SimilarityHit",
]
