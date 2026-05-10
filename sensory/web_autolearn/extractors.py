# ADAPTED FROM: adbar/trafilatura + codelucas/newspaper + buriy/python-readability
"""A-19.1 — Pure HTML → text extraction pipeline for the web-autolearn stack.

This module is the *extraction* half of the A-19 deep news crawler pair
(:mod:`sensory.web_autolearn.crawler_scrapy` is the *fetch* half, A-19.2).
It takes already-fetched raw HTML (the body of a :class:`RawDocument`
that came out of a crawler) and produces a cleaned title + main-text
projection via a deterministic fall-through cascade:

    trafilatura.extract()  →  newspaper3k.Article.parse()  →  readability.Document.summary()

The first extractor that returns a non-empty body wins; later
extractors are not invoked. If all three return empty the
:class:`ExtractorResult` carries ``succeeded=False`` and an empty body
so downstream consumers can drop the document without raising.

License posture
---------------
Upstream licenses:

* ``adbar/trafilatura``                 — **Apache-2.0**
* ``codelucas/newspaper`` (newspaper3k) — **MIT**
* ``buriy/python-readability`` (readability-lxml) — **Apache-2.0**

All three are compatible with the DIX project. The adapter only
consumes documented public APIs (``trafilatura.extract``,
``newspaper.Article.set_html`` / ``parse``, ``readability.Document.title``
/ ``summary``); no upstream code is copied or vendored.

Tier discipline
---------------
* **RUNTIME_SAFE — pure CPU-bound, no clock, no IO, no random.** This
  module never reaches into the runtime hot path itself (sensory layer
  is leaf-level by construction), but the extractors are deterministic
  enough that a hot-path caller could in principle invoke them. They
  are documented as RUNTIME_SAFE so a future analytics path can
  consume them without an authority breach. Lazy-imported deps stay
  out of every site-packages until the operator explicitly enables
  them via :func:`build_default_pipeline`.
* **Authority symmetry (B27 / B28 / INV-71).** This module never
  constructs typed bus events. It returns frozen :class:`ExtractorResult`
  value objects only; callers that re-materialise documents (e.g.
  the AI filter / curator) are responsible for any typed-event
  emission through their canonical producer.
* **No engine import.** Lint rule B1 / T1 / W1 pins the absence of
  ``governance_engine`` / ``system_engine`` / ``execution_engine`` /
  ``evolution_engine`` / ``intelligence_engine`` imports — verified
  by an AST test alongside the playwright + firecrawl peers.
* **INV-15 determinism.** No ``random``, ``time``, ``datetime``,
  ``os``, ``asyncio``, ``websockets``, ``numpy``, ``torch``,
  ``polars``, ``langsmith`` imports. All extractor inputs and
  outputs are pure data — the same HTML always produces the same
  :class:`ExtractorResult`. Pinned by AST tests.

Pip dependencies
----------------

``NEW_PIP_DEPENDENCIES = ("trafilatura", "newspaper3k", "readability-lxml")``
— declared at the module level for ops tooling; **lazy-imported only
inside the factory function bodies**. The module is importable with
zero external deps installed; tests inject ``extractor_factory`` so
they never need the real packages.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

__all__ = (
    "MAX_TITLE_LEN",
    "MAX_BODY_LEN",
    "NEW_PIP_DEPENDENCIES",
    "Extractor",
    "ExtractorResult",
    "ExtractorPipeline",
    "build_default_pipeline",
    "newspaper3k_extractor_factory",
    "readability_extractor_factory",
    "trafilatura_extractor_factory",
)


# ---------------------------------------------------------------------------
# Public sentinels
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = (
    "trafilatura",
    "newspaper3k",
    "readability-lxml",
)
"""Lazy pip dependencies surfaced for ops tooling.

None of the three packages is imported at module top level; each lives
inside its own factory function body so the module is importable in
environments where the extractors are not installed.
"""

MAX_TITLE_LEN: Final[int] = 512
"""Hard cap on extracted title length.

The extractors do not enforce length on their own — different
implementations may yield very long titles when fed pathological HTML.
Truncating here keeps the pipeline output bounded for the audit
ledger and downstream embeddings.
"""

MAX_BODY_LEN: Final[int] = 200_000
"""Hard cap on extracted body length.

Same rationale as :data:`MAX_TITLE_LEN`. Articles longer than this
are truncated; the meta record carries ``"body_truncated": "1"`` so
the auditor can tell.
"""


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExtractorResult:
    """Outcome of a single extractor invocation.

    All extractors return one of these regardless of underlying impl,
    so :class:`ExtractorPipeline` can compose them uniformly.

    Attributes:
        title: Extracted document title. Empty string when the
            extractor produced no title (still allowed even when
            ``succeeded=True`` since some sources have no title).
        body: Extracted main-text body. Empty string when the
            extractor failed to find any content — pinned to empty
            via :attr:`succeeded` for an unambiguous fail-soft signal.
        language: Two-letter ISO 639-1 language code if the extractor
            detected one, empty string otherwise. Lower-case.
        succeeded: ``True`` iff the extractor produced a non-empty
            body. The pipeline cascade only consults this flag to
            decide whether to try the next extractor.
        source: Stable identifier of which extractor produced this
            result. One of ``"trafilatura"``, ``"newspaper3k"``,
            ``"readability"``, or ``"none"`` for the empty
            fall-through.
        meta: Free-form structural metadata (no PII, no secrets).
            Keys are sorted lexicographically when serialised so the
            digest is order-independent.
    """

    title: str
    body: str
    language: str
    succeeded: bool
    source: str
    meta: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: MappingProxyType({}),
    )

    def __post_init__(self) -> None:
        if self.source not in {
            "trafilatura",
            "newspaper3k",
            "readability",
            "none",
        }:
            raise ValueError(
                "ExtractorResult.source must be one of "
                f"{{'trafilatura', 'newspaper3k', 'readability', 'none'}}; "
                f"got {self.source!r}"
            )
        if self.succeeded and not self.body:
            raise ValueError(
                "ExtractorResult.succeeded=True requires non-empty body"
            )
        if not self.succeeded and self.body:
            raise ValueError(
                "ExtractorResult.succeeded=False requires empty body"
            )
        if len(self.title) > MAX_TITLE_LEN:
            raise ValueError(
                "ExtractorResult.title exceeds "
                f"{MAX_TITLE_LEN} chars; truncate before construction"
            )
        if len(self.body) > MAX_BODY_LEN:
            raise ValueError(
                "ExtractorResult.body exceeds "
                f"{MAX_BODY_LEN} chars; truncate before construction"
            )
        # Force a frozen MappingProxyType view with sorted keys so two
        # results that differ only in dict insertion order compare equal
        # and serialise identically.
        if not isinstance(self.meta, MappingProxyType):
            frozen = MappingProxyType(
                {str(k): str(v) for k, v in sorted(self.meta.items())},
            )
            object.__setattr__(self, "meta", frozen)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Extractor(Protocol):
    """Pure HTML → :class:`ExtractorResult` projection.

    Implementations must be:

      * **stateless across calls** — :meth:`extract` is a pure function
        of its inputs;
      * **deterministic** — identical ``(html, url)`` arguments always
        produce identical :class:`ExtractorResult` (INV-15);
      * **fail-soft** — any internal exception is caught and converted
        into a :class:`ExtractorResult` with ``succeeded=False`` and an
        empty body. Extractors must never raise out of ``extract``.
    """

    def extract(
        self,
        html: str,
        *,
        url: str = "",
    ) -> ExtractorResult:
        """Project ``html`` (and optional ``url`` context) into a result.

        Args:
            html: The raw HTML body. May be empty — extractors must
                return ``ExtractorResult(succeeded=False, ...)``
                in that case rather than raising.
            url: Optional canonical URL the HTML came from. Some
                extractors (notably newspaper3k) use it to resolve
                relative links; readability/trafilatura ignore it.

        Returns:
            One :class:`ExtractorResult` describing what the
            extractor found. Empty input → empty result, never an
            exception.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete factories — each lazy-imports its own package.
# ---------------------------------------------------------------------------


def _truncate(value: str, *, limit: int) -> tuple[str, bool]:
    """Clip ``value`` to ``limit`` chars, returning ``(clipped, did_truncate)``."""

    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _empty_result(source: str, *, meta: Mapping[str, str]) -> ExtractorResult:
    return ExtractorResult(
        title="",
        body="",
        language="",
        succeeded=False,
        source=source,
        meta=meta,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _TrafilaturaExtractor:
    """Internal :class:`Extractor` impl wrapping ``trafilatura.extract``.

    Only constructed by :func:`trafilatura_extractor_factory`; kept
    file-private so callers route through the factory (which performs
    the lazy import).
    """

    _extract: Callable[..., str | None]

    def extract(
        self,
        html: str,
        *,
        url: str = "",
    ) -> ExtractorResult:
        if not html:
            return _empty_result(
                "trafilatura",
                meta=MappingProxyType({"reason": "empty_html"}),
            )
        try:
            body_raw = self._extract(
                html,
                url=url or None,
                output_format="txt",
                include_comments=False,
                include_tables=False,
                no_fallback=True,
                deduplicate=True,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft contract
            return _empty_result(
                "trafilatura",
                meta=MappingProxyType(
                    {
                        "reason": "exception",
                        "exception_class": exc.__class__.__name__,
                    },
                ),
            )
        if not body_raw:
            return _empty_result(
                "trafilatura",
                meta=MappingProxyType({"reason": "no_content"}),
            )
        body_clipped, body_trunc = _truncate(
            body_raw.strip(),
            limit=MAX_BODY_LEN,
        )
        if not body_clipped:
            return _empty_result(
                "trafilatura",
                meta=MappingProxyType({"reason": "whitespace_only"}),
            )
        meta_pairs: dict[str, str] = {}
        if body_trunc:
            meta_pairs["body_truncated"] = "1"
        return ExtractorResult(
            title="",
            body=body_clipped,
            language="",
            succeeded=True,
            source="trafilatura",
            meta=meta_pairs,
        )


def trafilatura_extractor_factory(
    *,
    extract_callable: Callable[..., str | None] | None = None,
) -> Extractor:
    """Build a :class:`Extractor` backed by ``trafilatura.extract``.

    ``trafilatura`` is *lazy-imported* inside this factory; the
    module top-level has no reference to it. Tests inject
    ``extract_callable`` so they never need the real package.
    """

    if extract_callable is None:
        import trafilatura  # noqa: PLC0415 — lazy import is the point

        extract_callable = trafilatura.extract
    return _TrafilaturaExtractor(_extract=extract_callable)


@dataclasses.dataclass(frozen=True, slots=True)
class _Newspaper3kExtractor:
    """Internal :class:`Extractor` impl wrapping ``newspaper.Article``."""

    _article_factory: Callable[[str], object]

    def extract(
        self,
        html: str,
        *,
        url: str = "",
    ) -> ExtractorResult:
        if not html:
            return _empty_result(
                "newspaper3k",
                meta=MappingProxyType({"reason": "empty_html"}),
            )
        try:
            article = self._article_factory(url or "https://example.invalid/")
            # newspaper3k requires set_html() then parse() — no network
            # call when html is already in hand.
            set_html = article.set_html  # type: ignore[attr-defined]
            set_html(html)
            parse = article.parse  # type: ignore[attr-defined]
            parse()
            title_raw: str = str(getattr(article, "title", "") or "")
            body_raw: str = str(getattr(article, "text", "") or "")
            meta_lang_raw = getattr(article, "meta_lang", "") or ""
            language = (
                str(meta_lang_raw).strip().lower()
                if isinstance(meta_lang_raw, str)
                else ""
            )
        except Exception as exc:  # noqa: BLE001
            return _empty_result(
                "newspaper3k",
                meta=MappingProxyType(
                    {
                        "reason": "exception",
                        "exception_class": exc.__class__.__name__,
                    },
                ),
            )
        body_clipped, body_trunc = _truncate(
            body_raw.strip(),
            limit=MAX_BODY_LEN,
        )
        if not body_clipped:
            return _empty_result(
                "newspaper3k",
                meta=MappingProxyType({"reason": "no_content"}),
            )
        title_clipped, title_trunc = _truncate(
            title_raw.strip(),
            limit=MAX_TITLE_LEN,
        )
        meta_pairs: dict[str, str] = {}
        if body_trunc:
            meta_pairs["body_truncated"] = "1"
        if title_trunc:
            meta_pairs["title_truncated"] = "1"
        # Guard the two-letter ISO-639-1 contract — newspaper3k can
        # sometimes return junk like "zh-CN"; we keep only the prefix
        # so downstream rules stay simple.
        if language and len(language) >= 2 and language[:2].isalpha():
            language = language[:2]
        else:
            language = ""
        return ExtractorResult(
            title=title_clipped,
            body=body_clipped,
            language=language,
            succeeded=True,
            source="newspaper3k",
            meta=meta_pairs,
        )


def newspaper3k_extractor_factory(
    *,
    article_factory: Callable[[str], object] | None = None,
) -> Extractor:
    """Build a :class:`Extractor` backed by ``newspaper.Article``.

    ``newspaper3k`` is *lazy-imported* inside this factory. The
    factory accepts ``article_factory(url) -> Article`` so tests can
    inject a fake article without the real package. The default
    factory builds ``newspaper.Article(url)`` with
    ``fetch_images=False`` to avoid any network IO during parse —
    extraction is HTML → text only.
    """

    if article_factory is None:
        from newspaper import Article  # noqa: PLC0415 — lazy import

        def _factory(url: str) -> object:
            return Article(url, fetch_images=False)

        article_factory = _factory
    return _Newspaper3kExtractor(_article_factory=article_factory)


@dataclasses.dataclass(frozen=True, slots=True)
class _ReadabilityExtractor:
    """Internal :class:`Extractor` impl wrapping ``readability.Document``."""

    _document_factory: Callable[[str], object]
    _strip_tags: Callable[[str], str]

    def extract(
        self,
        html: str,
        *,
        url: str = "",
    ) -> ExtractorResult:
        if not html:
            return _empty_result(
                "readability",
                meta=MappingProxyType({"reason": "empty_html"}),
            )
        try:
            doc = self._document_factory(html)
            title_method = doc.title  # type: ignore[attr-defined]
            summary_method = doc.summary  # type: ignore[attr-defined]
            title_raw: str = str(title_method() or "")
            summary_html: str = str(summary_method() or "")
            body_raw = self._strip_tags(summary_html)
        except Exception as exc:  # noqa: BLE001
            return _empty_result(
                "readability",
                meta=MappingProxyType(
                    {
                        "reason": "exception",
                        "exception_class": exc.__class__.__name__,
                    },
                ),
            )
        body_clipped, body_trunc = _truncate(
            body_raw.strip(),
            limit=MAX_BODY_LEN,
        )
        if not body_clipped:
            return _empty_result(
                "readability",
                meta=MappingProxyType({"reason": "no_content"}),
            )
        title_clipped, title_trunc = _truncate(
            title_raw.strip(),
            limit=MAX_TITLE_LEN,
        )
        meta_pairs: dict[str, str] = {}
        if body_trunc:
            meta_pairs["body_truncated"] = "1"
        if title_trunc:
            meta_pairs["title_truncated"] = "1"
        return ExtractorResult(
            title=title_clipped,
            body=body_clipped,
            language="",
            succeeded=True,
            source="readability",
            meta=meta_pairs,
        )


def _strip_html_tags(html: str) -> str:
    """Minimal HTML → text fallback used when the readability summary
    is a fragment of HTML.

    Pure-Python (stdlib :mod:`html.parser`). No bs4, no lxml. Keeps
    text content only; collapses whitespace runs to a single space.
    """

    from html.parser import HTMLParser  # noqa: PLC0415 — stdlib lazy import

    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.chunks: list[str] = []

        def handle_data(self, data: str) -> None:
            self.chunks.append(data)

    parser = _Stripper()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.chunks)
    # Collapse all whitespace runs to a single space, then strip — the
    # extractor result post-init handles outer strip but body
    # comparison is whitespace-sensitive so we normalise here.
    parts = raw.split()
    return " ".join(parts)


def readability_extractor_factory(
    *,
    document_factory: Callable[[str], object] | None = None,
    strip_tags: Callable[[str], str] | None = None,
) -> Extractor:
    """Build a :class:`Extractor` backed by ``readability.Document``.

    ``readability-lxml`` is *lazy-imported* inside this factory. The
    factory accepts ``document_factory(html) -> Document`` so tests
    can inject a fake document without the real package; the default
    builds ``readability.Document(html)``.
    """

    if document_factory is None:
        from readability import Document  # noqa: PLC0415 — lazy import

        def _doc_factory(html: str) -> object:
            return Document(html)

        document_factory = _doc_factory
    return _ReadabilityExtractor(
        _document_factory=document_factory,
        _strip_tags=strip_tags or _strip_html_tags,
    )


# ---------------------------------------------------------------------------
# Pipeline cascade
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExtractorPipeline:
    """Run a tuple of :class:`Extractor` in declared order, first hit wins.

    The default order — ``trafilatura → newspaper3k → readability`` —
    follows the spec line 1502 ranking ("trafilatura best, newspaper3k
    fallback, readability last") and is enforced by
    :func:`build_default_pipeline`.

    The pipeline itself is fail-soft: an extractor that returns
    ``succeeded=False`` is dropped and the next one is tried. If
    *all* extractors fail, the pipeline returns a synthetic
    ``ExtractorResult(source="none")`` with empty body so the caller
    has a single uniform shape to consume.
    """

    extractors: tuple[Extractor, ...]

    def __post_init__(self) -> None:
        if not self.extractors:
            raise ValueError(
                "ExtractorPipeline requires at least one extractor"
            )

    def extract(
        self,
        html: str,
        *,
        url: str = "",
    ) -> ExtractorResult:
        """Run the cascade and return the first successful result.

        Args:
            html: Raw HTML body. Empty input always cascades through
                every extractor (each returns
                ``succeeded=False, reason="empty_html"``) and
                produces the synthetic ``source="none"`` result.
            url: Optional URL context — passed through to each
                extractor unchanged.
        """

        attempted: list[str] = []
        for extractor in self.extractors:
            result = extractor.extract(html, url=url)
            attempted.append(result.source)
            if result.succeeded:
                return result
        return ExtractorResult(
            title="",
            body="",
            language="",
            succeeded=False,
            source="none",
            meta={"attempted": ",".join(attempted)},
        )


def build_default_pipeline(
    *,
    trafilatura: Extractor | None = None,
    newspaper3k: Extractor | None = None,
    readability: Extractor | None = None,
) -> ExtractorPipeline:
    """Build the canonical 3-stage cascade.

    Order is fixed: ``trafilatura → newspaper3k → readability``.
    Callers may inject any subset of test doubles — the rest are
    built via their lazy-import factory.
    """

    return ExtractorPipeline(
        extractors=(
            trafilatura
            if trafilatura is not None
            else trafilatura_extractor_factory(),
            newspaper3k
            if newspaper3k is not None
            else newspaper3k_extractor_factory(),
            readability
            if readability is not None
            else readability_extractor_factory(),
        ),
    )


# Re-export for callers that build their own pipelines from a custom
# extractor ordering (e.g. readability-only sandboxes).
_DEFAULT_ORDER: Final[Sequence[str]] = (
    "trafilatura",
    "newspaper3k",
    "readability",
)
