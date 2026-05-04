"""WEBLEARN-02 — AI relevance filter.

The filter receives :class:`RawDocument` instances and returns a
:class:`FilterDecision` that either promotes them to a
:class:`FilteredItem` or drops them with a reason. It is a pure
function — no I/O, no clock reads, no global state — so the
deterministic-replay invariant (INV-15) is preserved when the same
documents are re-fed during replay.

This module ships :class:`KeywordAIFilter`, a simple keyword-scoring
implementation that is sufficient for the initial sensory build. A
real LLM-backed filter (e.g. ``LLMRelevanceFilter`` calling the
existing ``RegistryDrivenChatModel`` adapter) can replace it later
behind the same :class:`AIFilter` :class:`Protocol`.

Authority discipline: the filter never imports an engine, never
mutates the SystemMode FSM, and never writes to the audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sensory.web_autolearn.contracts import FilteredItem, RawDocument


@dataclass(frozen=True, slots=True)
class FilterDecision:
    """The outcome of running one document through the filter.

    Exactly one of ``item`` and ``drop_reason`` is non-None. When
    ``item`` is set, the document passed and was promoted to a
    :class:`FilteredItem`. When ``drop_reason`` is set, the document
    was rejected (e.g. ``"fetch_failed"``, ``"score_below_min"``,
    ``"no_keywords_matched"``).
    """

    item: FilteredItem | None = None
    drop_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.item is None) == (self.drop_reason is None):
            raise ValueError(
                "FilterDecision: exactly one of item/drop_reason"
                " must be set"
            )

    @property
    def passed(self) -> bool:
        return self.item is not None


@runtime_checkable
class AIFilter(Protocol):
    """A relevance filter for web autolearn documents."""

    def evaluate(self, document: RawDocument) -> FilterDecision:
        """Return a :class:`FilterDecision` for ``document``."""
        ...


def _normalize_keywords(keywords: tuple[str, ...]) -> tuple[str, ...]:
    """Lowercase + strip + dedupe + sort for deterministic matching."""

    cleaned: set[str] = set()
    for kw in keywords:
        norm = kw.strip().lower()
        if norm:
            cleaned.add(norm)
    return tuple(sorted(cleaned))


@dataclass(frozen=True, slots=True)
class KeywordAIFilter:
    """Score documents by counting per-seed keyword hits.

    The filter has no global keyword list. Instead, callers provide a
    ``seed_keywords`` map keyed by seed_id; this keeps the filter
    decision local to each seed and matches the seeds.yaml structure
    where every seed declares its own topic vocabulary.

    Score formula::

        matches  = number of distinct keywords found in
                   (title + ' ' + body), case-insensitive
        score    = min(1.0, matches / max(1, len(keywords)))

    Documents with ``fetched_ok=False`` are dropped without scoring
    (``drop_reason='fetch_failed'``). Documents whose seed has no
    declared keywords are dropped with ``drop_reason='no_keywords'``.
    Score below ``min_score`` is dropped with
    ``drop_reason='score_below_min'``.

    The ``min_score`` default (0.0) is intentional: by default *any*
    keyword match passes, and the curator (WEBLEARN-03) is responsible
    for the stricter score threshold. This keeps the filter cheap and
    delegates policy to the curator (which is the layer the operator
    can tune via seeds.yaml).
    """

    seed_keywords: dict[str, tuple[str, ...]] = field(default_factory=dict)
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                "KeywordAIFilter.min_score must be in [0.0, 1.0]"
            )

    def evaluate(self, document: RawDocument) -> FilterDecision:
        """Score a single document; return a :class:`FilterDecision`."""

        if not document.fetched_ok:
            return FilterDecision(drop_reason="fetch_failed")
        keywords = _normalize_keywords(
            self.seed_keywords.get(document.seed_id, ())
        )
        if not keywords:
            return FilterDecision(drop_reason="no_keywords")
        haystack = f"{document.title} {document.body}".lower()
        matched = tuple(kw for kw in keywords if kw in haystack)
        if not matched:
            return FilterDecision(drop_reason="no_keywords_matched")
        score = min(1.0, len(matched) / max(1, len(keywords)))
        if score < self.min_score:
            return FilterDecision(drop_reason="score_below_min")
        item = FilteredItem(
            ts_ns=document.ts_ns,
            seed_id=document.seed_id,
            url=document.url,
            title=document.title,
            body=document.body,
            score=score,
            reason=f"keyword:{matched[0]}",
            meta=dict(document.meta),
        )
        return FilterDecision(item=item)


__all__ = [
    "AIFilter",
    "FilterDecision",
    "KeywordAIFilter",
]
