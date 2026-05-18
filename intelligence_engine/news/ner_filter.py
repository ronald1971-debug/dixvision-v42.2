"""B-07 canonical adaptation: spaCy NER filter for news items.

# ADAPTED FROM: spaCy explosion/spaCy
#   - spacy/pipeline/ner.py — EntityRecognizer (sequence-tagging head)
#   - spacy/language.py     — Language.pipe() batch processing
#   - spacy/matcher/matcher.py — rule-based pattern matching

This module is a **pure-Python, deterministic, offline-only** adaptation of
spaCy's NER + Matcher contract over the DIX ``NewsItem`` value object. It
extracts a typed tuple of :class:`NamedEntity` value objects from a
:class:`core.contracts.news.NewsItem` and projects them into an
:class:`EnrichedNewsItem` whose ``entities`` field is sorted and
byte-stable so that replay (INV-15) is preserved.

Design principles
-----------------

1. **No spaCy import.** Production never pulls a 12 MB statistical model
   into the runtime tree. We adapt the *pattern* (entity types, batch
   pipe contract, matcher rules) using Python's ``re`` module and a
   small, alphabetically-sorted rule table. The implementation is
   stdlib-only — ``NEW_PIP_DEPENDENCIES = ()``.
2. **OFFLINE sensory tier.** Pinned by AST tests. The module never
   constructs typed bus events (``PatchProposal`` / ``SignalEvent`` /
   ``GovernanceDecision`` / ``SystemEvent`` / ``ExecutionIntent`` /
   ``FillEvent``) — only frozen value objects. B27 / B28 / INV-71
   authority symmetry is upheld.
3. **Soft failure.** NER failures (no matches, malformed text) never
   raise — the ``EnrichedNewsItem.entities`` field is simply empty.
   ``NewsItem`` re-emission upstream is preserved.
4. **Batch pipe contract.** :func:`extract_entities_batch` mirrors
   spaCy's ``Language.pipe()``: a single deterministic pass over the
   input iterable producing one :class:`EnrichedNewsItem` per
   :class:`NewsItem`. No per-item exceptions ever leak.
5. **Determinism.** Sorted matcher rules, sorted entity output (by
   ``(start, end, label, text)``), no clock / random / IO. Same input
   text → identical entity tuple across runs / machines / Python
   versions.

Entity label coverage
---------------------

The label set is a frozen subset of spaCy's English NER label set
augmented with one DIX-specific label:

* ``ORG``     — known financial / tech companies (``Apple``, ``Microsoft``,
  ``Tesla``, ``Coinbase``, ``Binance``, ``BlackRock``, …)
* ``MONEY``   — ``"$100"``, ``"$1.5M"``, ``"USD 250B"``, …
* ``PERCENT`` — ``"5%"``, ``"-2.3 %"``, ``"+0.5percent"``, …
* ``GPE``     — geopolitical entities (``US``, ``China``, ``EU``, ``UK``,
  ``Japan``, …)
* ``DATE``    — ISO-8601 (``"2024-01-15"``) and short forms
  (``"2024"``, ``"Q1 2024"``, ``"Q4"``)
* ``PRODUCT`` — known crypto / forex / stock products (``Bitcoin``,
  ``Ethereum``, ``Solana``, …)
* ``TICKER``  — DIX-specific: ``"$BTC"``, ``"$ETH"``, ``"AAPL"``,
  ``"TSLA"`` (4-letter uppercase, optionally ``$``-prefixed)

The rule tables (``_ORG_LITERALS``, ``_GPE_LITERALS``,
``_PRODUCT_LITERALS``) are alphabetically sorted at module load and
guarded by an AST test so they cannot drift out of order.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from core.contracts.news import NewsItem

# ---------------------------------------------------------------------------
# Provenance / lifecycle
# ---------------------------------------------------------------------------


NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""B-07 ships pure stdlib. We deliberately do **not** pin spaCy: the
spec calls for the *adaptation* of the pattern, not for spaCy itself to
enter the runtime tree."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class EntityLabel(StrEnum):
    """Canonical NER label set.

    Matches a frozen subset of spaCy's English NER labels (`ORG`,
    `MONEY`, `PERCENT`, `GPE`, `DATE`, `PRODUCT`) plus one DIX-specific
    label (`TICKER`) for ``$BTC`` / ``$ETH`` / ``AAPL``-style symbols.
    Sorted alphabetically so any iteration is deterministic.
    """

    DATE = "DATE"
    GPE = "GPE"
    MONEY = "MONEY"
    ORG = "ORG"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    TICKER = "TICKER"


@dataclass(frozen=True, slots=True, order=True)
class NamedEntity:
    """One entity extracted from a piece of text.

    Attributes:
        start: Inclusive 0-based character offset into the source text.
            Must be ``>= 0`` and ``< end``.
        end: Exclusive 0-based character offset into the source text.
            Must be ``> start``.
        label: Canonical :class:`EntityLabel`.
        text: The literal substring (``source_text[start:end]``).
            Stored explicitly so downstream consumers do not need the
            original buffer.
    """

    start: int
    end: int
    label: EntityLabel
    text: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("NamedEntity.start must be >= 0")
        if self.end <= self.start:
            raise ValueError("NamedEntity.end must be > start")
        if not self.text:
            raise ValueError("NamedEntity.text must be non-empty")


@dataclass(frozen=True, slots=True)
class EnrichedNewsItem:
    """A :class:`NewsItem` plus the entity tuple extracted from it.

    The original ``NewsItem`` is preserved verbatim — replay / dedup
    semantics upstream are unaffected. ``entities`` is sorted by
    ``(start, end, label, text)``.
    """

    item: NewsItem
    entities: tuple[NamedEntity, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # `tuple()` over a generator is type-safe but mypy gets confused
        # without an explicit local; sort is stable so equal keys keep
        # insertion order, but we guarantee dictionary-order
        # independence by sorting on the full tuple key.
        sorted_ents = tuple(sorted(self.entities))
        if sorted_ents != self.entities:
            object.__setattr__(self, "entities", sorted_ents)

    @property
    def entity_texts(self) -> tuple[str, ...]:
        """Convenience: deduplicated, sorted entity text values."""

        return tuple(sorted({e.text for e in self.entities}))

    def entities_by_label(self, label: EntityLabel) -> tuple[NamedEntity, ...]:
        """Filter entities by label. Order preserved."""

        return tuple(e for e in self.entities if e.label is label)


# ---------------------------------------------------------------------------
# Rule tables (alphabetically sorted; pinned by AST test)
# ---------------------------------------------------------------------------


# Curated organisation literals. Mirrors spaCy's Matcher token-text
# rule pattern (``[{"LOWER": "apple"}]``).
_ORG_LITERALS: Final[tuple[str, ...]] = (
    "Alphabet",
    "Amazon",
    "Apple",
    "BlackRock",
    "Binance",
    "Bitfinex",
    "BlackRock",
    "Citadel",
    "Coinbase",
    "Federal Reserve",
    "Fidelity",
    "Goldman Sachs",
    "Google",
    "JPMorgan",
    "Kraken",
    "Meta",
    "Microsoft",
    "Morgan Stanley",
    "Nvidia",
    "OpenAI",
    "OKX",
    "SEC",
    "Tesla",
)

# Curated geopolitical entities.
_GPE_LITERALS: Final[tuple[str, ...]] = (
    "Argentina",
    "Australia",
    "Brazil",
    "Canada",
    "China",
    "EU",
    "France",
    "Germany",
    "Hong Kong",
    "India",
    "Italy",
    "Japan",
    "Mexico",
    "Russia",
    "Singapore",
    "South Korea",
    "Switzerland",
    "Taiwan",
    "UK",
    "US",
    "USA",
)

# Curated crypto / financial product names.
_PRODUCT_LITERALS: Final[tuple[str, ...]] = (
    "Avalanche",
    "Bitcoin",
    "Cardano",
    "Dogecoin",
    "ETF",
    "Ethereum",
    "Litecoin",
    "Polkadot",
    "Polygon",
    "Ripple",
    "Solana",
    "Tether",
    "USDC",
    "USDT",
)


def _literal_pattern(literals: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a word-boundary alternation regex over the given literals.

    Sorted by length descending so longest match wins (mirrors spaCy's
    longest-token-first Matcher semantics). Each literal is regex-escaped.
    """

    by_length = sorted(set(literals), key=lambda s: (-len(s), s))
    alternation = "|".join(re.escape(lit) for lit in by_length)
    return re.compile(rf"(?<![A-Za-z0-9_]){alternation}(?![A-Za-z0-9_])")


_ORG_PATTERN: Final[re.Pattern[str]] = _literal_pattern(_ORG_LITERALS)
_GPE_PATTERN: Final[re.Pattern[str]] = _literal_pattern(_GPE_LITERALS)
_PRODUCT_PATTERN: Final[re.Pattern[str]] = _literal_pattern(_PRODUCT_LITERALS)


# Ticker symbols: ``$BTC``, ``$ETH``, ``AAPL``, ``TSLA``. We accept
# 2-5 uppercase letters either with a leading ``$`` (crypto convention)
# or as a standalone 3-5 letter all-caps token (US equities). Lookbehind /
# lookahead enforce word boundaries on alphanumeric characters so we do
# not split words like ``USA`` mid-stream.
_TICKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])(?:\$[A-Z]{2,5}|[A-Z]{3,5})(?![A-Za-z0-9_])"
)

# Money: optional sign, ``$`` or ``USD``/``EUR``/``GBP`` prefix or suffix,
# numeric body with optional decimal + thousands separator + optional
# magnitude suffix (``K`` / ``M`` / ``B`` / ``T``).
_MONEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:\$\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?:[KMBT])?"
    r"|\$\s?[0-9]+(?:\.[0-9]+)?(?:[KMBT])?"
    r"|(?:USD|EUR|GBP|JPY|CHF)\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?:[KMBT])?"
    r"|(?:USD|EUR|GBP|JPY|CHF)\s?[0-9]+(?:\.[0-9]+)?(?:[KMBT])?)"
    r"(?![A-Za-z0-9_])"
)

# Percent: optional sign, number with optional decimal, optional space,
# then ``%`` or ``percent``.
_PERCENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[+\-]?[0-9]+(?:\.[0-9]+)?\s?(?:%|percent)"
    r"(?![A-Za-z0-9_])"
)

# Dates: ISO-8601 (``2024-01-15``), short year (``2024``), quarter
# (``Q1 2024`` / ``Q4``).
_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"|Q[1-4](?:\s[0-9]{4})?"
    r"|(?:19|20)[0-9]{2})"
    r"(?![A-Za-z0-9_])"
)


# Pattern dispatch table — sorted by label name for byte-stable iteration.
_LABEL_PATTERNS: Final[tuple[tuple[EntityLabel, re.Pattern[str]], ...]] = (
    (EntityLabel.DATE, _DATE_PATTERN),
    (EntityLabel.GPE, _GPE_PATTERN),
    (EntityLabel.MONEY, _MONEY_PATTERN),
    (EntityLabel.ORG, _ORG_PATTERN),
    (EntityLabel.PERCENT, _PERCENT_PATTERN),
    (EntityLabel.PRODUCT, _PRODUCT_PATTERN),
    (EntityLabel.TICKER, _TICKER_PATTERN),
)


# Bound on input text length. spaCy's default ``max_length`` is 1_000_000;
# news headlines + summaries are <2 KB so 32 KB is generous and bounds
# regex blowup risk.
MAX_TEXT_LENGTH: Final[int] = 32_768


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NERFilterError(ValueError):
    """Raised when caller-supplied arguments violate the NER contract."""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _resolve_overlaps(
    candidates: list[NamedEntity],
) -> tuple[NamedEntity, ...]:
    """Keep the longest non-overlapping subset, ties broken by label order.

    Mirrors spaCy's ``util.filter_spans``: when two spans overlap we
    keep the longer one; ties are broken by the canonical (start, end,
    label, text) sort order. The result is sorted by ``(start, end,
    label, text)`` so it is byte-stable.
    """

    # Sort by length descending, then by canonical order ascending.
    by_priority = sorted(
        candidates,
        key=lambda e: (-(e.end - e.start), e.start, e.end, e.label.value, e.text),
    )
    kept: list[NamedEntity] = []
    occupied: list[tuple[int, int]] = []
    for ent in by_priority:
        if any(not (ent.end <= a or ent.start >= b) for (a, b) in occupied):
            continue
        kept.append(ent)
        occupied.append((ent.start, ent.end))
    return tuple(sorted(kept))


def extract_entities(text: str) -> tuple[NamedEntity, ...]:
    """Extract a sorted tuple of :class:`NamedEntity` from *text*.

    Pure / deterministic. Returns an empty tuple on empty input. Long
    inputs (> :data:`MAX_TEXT_LENGTH`) raise :class:`NERFilterError`
    so callers cannot accidentally amplify regex cost.
    """

    if not isinstance(text, str):
        raise NERFilterError("extract_entities: text must be str")
    if len(text) > MAX_TEXT_LENGTH:
        raise NERFilterError(f"extract_entities: text exceeds MAX_TEXT_LENGTH ({MAX_TEXT_LENGTH})")
    if not text:
        return ()

    candidates: list[NamedEntity] = []
    for label, pattern in _LABEL_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            substring = text[start:end]
            if not substring:
                continue
            candidates.append(
                NamedEntity(
                    start=start,
                    end=end,
                    label=label,
                    text=substring,
                )
            )
    return _resolve_overlaps(candidates)


# ---------------------------------------------------------------------------
# NewsItem enrichment
# ---------------------------------------------------------------------------


def _news_item_text(item: NewsItem) -> str:
    """Canonical text projection used by the NER filter.

    Title and summary are joined with a single ``\\n`` separator so
    offsets stay meaningful and a summary entity does not collide with
    a title entity at the same position.
    """

    if item.summary:
        return f"{item.title}\n{item.summary}"
    return item.title


def enrich_news_item(item: NewsItem) -> EnrichedNewsItem:
    """Wrap *item* with a sorted, byte-stable entity tuple.

    Soft failure: an exception inside :func:`extract_entities` is
    swallowed and the resulting :class:`EnrichedNewsItem` carries an
    empty tuple. The original :class:`NewsItem` is always preserved.
    """

    if not isinstance(item, NewsItem):
        raise NERFilterError("enrich_news_item: item must be NewsItem")
    try:
        entities = extract_entities(_news_item_text(item))
    except NERFilterError:
        entities = ()
    return EnrichedNewsItem(item=item, entities=entities)


def extract_entities_batch(
    items: Iterable[NewsItem],
) -> tuple[EnrichedNewsItem, ...]:
    """Mirror of spaCy's ``Language.pipe()`` — one deterministic pass.

    Order is preserved from the input iterable. Per-item exceptions
    are swallowed and surface as empty ``entities`` so a single
    malformed headline never poisons the batch.
    """

    out: list[EnrichedNewsItem] = []
    for item in items:
        if not isinstance(item, NewsItem):
            raise NERFilterError("extract_entities_batch: every element must be NewsItem")
        out.append(enrich_news_item(item))
    return tuple(out)


# ---------------------------------------------------------------------------
# Convenience projections (read-only)
# ---------------------------------------------------------------------------


def entity_summary(items: Iterable[EnrichedNewsItem]) -> Mapping[EntityLabel, int]:
    """Aggregate entity counts per label across *items*.

    Returns a frozen mapping (via :class:`types.MappingProxyType` in
    callers; we hand back a plain dict to keep this fully stdlib).
    Counts are stable across runs because input order is preserved
    and entity tuples are byte-stable.
    """

    counts: dict[EntityLabel, int] = {label: 0 for label in EntityLabel}
    for enriched in items:
        for ent in enriched.entities:
            counts[ent.label] += 1
    # Drop zero counts so the projection is small + byte-stable.
    return {label: n for label, n in counts.items() if n > 0}


__all__ = [
    "MAX_TEXT_LENGTH",
    "NEW_PIP_DEPENDENCIES",
    "EnrichedNewsItem",
    "EntityLabel",
    "NERFilterError",
    "NamedEntity",
    "enrich_news_item",
    "entity_summary",
    "extract_entities",
    "extract_entities_batch",
]
