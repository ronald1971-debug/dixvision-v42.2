"""Value types for the web autolearn pipeline.

All types are frozen + slotted dataclasses so they are hashable,
immutable, and safe to record into the deterministic-replay ledger
(INV-15).

The flow is::

    Crawler   ----(RawDocument)----> AIFilter
    AIFilter  ----(FilteredItem)---> Curator
    Curator   ----(CuratedItem)----> PendingBuffer

:class:`NewsItem` is re-exported from :mod:`core.contracts.news` to
satisfy ``registry/data_source_registry.yaml`` references like
``sensory.web_autolearn.contracts.NewsItem`` (Reuters row). The
canonical home for ``NewsItem`` remains :mod:`core.contracts.news`;
this module is a thin re-export so the registry path resolves at
import time.

:class:`SocialPost` is the analogous value for the X / Reddit /
Telegram social rows in the same registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.contracts.news import NewsItem


@dataclass(frozen=True, slots=True)
class SocialPost:
    """One post / tweet snapshot from a social source.

    Mirrors the shape of :class:`NewsItem` so the SCVS validator can
    apply identical schema/staleness checks without a special-case.
    The distinction is semantic: a NewsItem is editorialized, a
    SocialPost is user-generated.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds (caller-
            supplied, never derived from the payload — INV-15).
        source: Stable source identifier matching the SCVS registry
            row (e.g. ``"X"``, ``"REDDIT"``). Empty string is rejected.
        post_id: Per-source unique identifier for deduplication.
        author: Author handle / username. Empty string is rejected
            because anonymous posts have no signal authority.
        body: Post text. Sanitized (no HTML tags, no leading/trailing
            whitespace). Empty string is rejected.
        url: Optional canonical URL.
        published_ts_ns: Optional publication timestamp from the
            payload. ``None`` when the source omits it. Never ``0``.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    source: str
    post_id: str
    author: str
    body: str
    url: str = ""
    published_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("SocialPost.source must be non-empty")
        if not self.post_id:
            raise ValueError("SocialPost.post_id must be non-empty")
        if not self.author:
            raise ValueError("SocialPost.author must be non-empty")
        if not self.body:
            raise ValueError("SocialPost.body must be non-empty")
        if (
            self.published_ts_ns is not None
            and self.published_ts_ns <= 0
        ):
            raise ValueError(
                "SocialPost.published_ts_ns must be positive or None"
            )


@dataclass(frozen=True, slots=True)
class RawDocument:
    """A document as fetched from a seed URL — pre-filter, pre-curation.

    This is the Crawler's output and the AIFilter's input. It is
    intentionally generic: the crawler may produce HTML pages, JSON
    feeds, or RSS items, and the AIFilter inspects the structured
    fields rather than parsing again.

    Attributes:
        ts_ns: Ingestion timestamp (caller-supplied).
        seed_id: Stable identifier of the seed that produced this
            document (matches a row in :file:`seeds.yaml`).
        url: Canonical URL the document was fetched from.
        title: Document title / headline. Empty string allowed for
            sources without titles.
        body: Document body text. Empty string allowed.
        fetched_ok: Whether the fetch reached HTTP 200 (or feed
            equivalent). False indicates a partial/error fetch — the
            AIFilter will reject these without scoring.
        meta: Free-form payload metadata (status code, content type,
            etc.). No PII, no secrets.
    """

    ts_ns: int
    seed_id: str
    url: str
    title: str = ""
    body: str = ""
    fetched_ok: bool = True
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("RawDocument.seed_id must be non-empty")
        if not self.url:
            raise ValueError("RawDocument.url must be non-empty")


@dataclass(frozen=True, slots=True)
class FilteredItem:
    """The AIFilter's output: a RawDocument that passed the relevance
    score plus its score and reason.

    Attributes:
        ts_ns: Carry-through from the source RawDocument.
        seed_id: Carry-through from the source RawDocument.
        url: Carry-through from the source RawDocument.
        title: Carry-through.
        body: Carry-through.
        score: Filter score in ``[0.0, 1.0]``. Higher = more relevant
            to the seed's declared topic. Below the curator's
            ``min_score`` the item is dropped.
        reason: Short human-readable label for the filter decision
            (e.g. ``"keyword:bitcoin"``, ``"recency:fresh"``). Empty
            string is rejected — every passed item must explain
            *why* it passed.
        meta: Carry-through from the source RawDocument.
    """

    ts_ns: int
    seed_id: str
    url: str
    title: str
    body: str
    score: float
    reason: str
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("FilteredItem.seed_id must be non-empty")
        if not self.url:
            raise ValueError("FilteredItem.url must be non-empty")
        if not self.reason:
            raise ValueError("FilteredItem.reason must be non-empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "FilteredItem.score must be in [0.0, 1.0]"
            )


@dataclass(frozen=True, slots=True)
class CuratedItem:
    """The Curator's output: a FilteredItem that survived the seed-
    specific allow/deny rules and is ready for HITL review.

    Attributes:
        ts_ns: Carry-through from the source FilteredItem.
        seed_id: Carry-through.
        url: Carry-through.
        title: Carry-through.
        body: Carry-through.
        score: Carry-through filter score.
        seed_topic: The topic label declared by the seed in
            ``seeds.yaml`` (e.g. ``"crypto"``, ``"macro"``). Used by
            the operator dashboard to group pending items.
        curator_tags: Sorted, deduplicated tuple of tags applied by
            the curator (e.g. ``("rate-decision", "fed")``). Empty
            tuple is allowed.
        meta: Carry-through.
    """

    ts_ns: int
    seed_id: str
    url: str
    title: str
    body: str
    score: float
    seed_topic: str
    curator_tags: tuple[str, ...] = ()
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("CuratedItem.seed_id must be non-empty")
        if not self.url:
            raise ValueError("CuratedItem.url must be non-empty")
        if not self.seed_topic:
            raise ValueError(
                "CuratedItem.seed_topic must be non-empty"
            )
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "CuratedItem.score must be in [0.0, 1.0]"
            )


__all__ = [
    "CuratedItem",
    "FilteredItem",
    "NewsItem",
    "RawDocument",
    "SocialPost",
]
