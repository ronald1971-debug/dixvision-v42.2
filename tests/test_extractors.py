"""Tests for :mod:`sensory.web_autolearn.extractors` (A-19.1).

Covers:
* :class:`ExtractorResult` invariants + frozen-meta projection.
* Each extractor factory (trafilatura / newspaper3k / readability)
  via injected fakes — never imports the real package.
* :class:`ExtractorPipeline` cascade order + fail-soft + synthetic
  ``source="none"`` fall-through.
* :func:`build_default_pipeline` honours the canonical
  trafilatura → newspaper3k → readability order.
* INV-15 determinism — 3-run equality on byte projection.
* AST guards: no top-level external imports, no engine cross-imports,
  no typed-event construction, no clock / random / IO.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Mapping

import pytest

from sensory.web_autolearn.extractors import (
    MAX_BODY_LEN,
    MAX_TITLE_LEN,
    NEW_PIP_DEPENDENCIES,
    Extractor,
    ExtractorPipeline,
    ExtractorResult,
    build_default_pipeline,
    newspaper3k_extractor_factory,
    readability_extractor_factory,
    trafilatura_extractor_factory,
)

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "sensory" / "web_autolearn" / "extractors.py"
)
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SAMPLE_HTML = (
    "<html><head><title>Sample Story</title></head>"
    "<body><article><p>Hello world.</p>"
    "<p>Second paragraph for context.</p></article></body></html>"
)


def _fake_trafilatura_ok(
    html: str,
    *,
    url: str | None = None,
    output_format: str = "txt",
    include_comments: bool = False,
    include_tables: bool = False,
    no_fallback: bool = True,
    deduplicate: bool = True,
) -> str | None:
    return "Hello world.\n\nSecond paragraph for context."


def _fake_trafilatura_none(
    html: str,
    *,
    url: str | None = None,
    output_format: str = "txt",
    include_comments: bool = False,
    include_tables: bool = False,
    no_fallback: bool = True,
    deduplicate: bool = True,
) -> str | None:
    return None


def _fake_trafilatura_raises(html: str, **_kwargs: object) -> str | None:
    raise RuntimeError("upstream blew up")


class _FakeArticle:
    def __init__(self, url: str) -> None:
        self.url = url
        self.title = ""
        self.text = ""
        self.meta_lang = ""
        self._html = ""

    def set_html(self, html: str) -> None:
        self._html = html
        # Toy "parser": treat anything between <article>...</article>
        # as the body; pretend the first <p>...</p> is the title.
        self.title = "Sample Story"
        self.text = "Hello world. Second paragraph for context."
        self.meta_lang = "en"

    def parse(self) -> None:
        pass


class _FakeArticleEmpty(_FakeArticle):
    def set_html(self, html: str) -> None:
        self._html = html
        self.title = ""
        self.text = ""
        self.meta_lang = ""


class _FakeArticleRaises(_FakeArticle):
    def parse(self) -> None:
        raise RuntimeError("boom")


class _FakeDocument:
    def __init__(self, html: str) -> None:
        self._html = html

    def title(self) -> str:
        return "Sample Story"

    def summary(self) -> str:
        return "<div><p>Hello world.</p><p>Second paragraph for context.</p></div>"


class _FakeDocumentEmpty(_FakeDocument):
    def summary(self) -> str:
        return "<div></div>"


class _FakeDocumentRaises(_FakeDocument):
    def title(self) -> str:
        raise RuntimeError("title blew up")


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


class TestSentinels:
    def test_new_pip_dependencies_declared(self) -> None:
        assert NEW_PIP_DEPENDENCIES == (
            "trafilatura",
            "newspaper3k",
            "readability-lxml",
        )

    def test_caps_present(self) -> None:
        assert MAX_TITLE_LEN > 0
        assert MAX_BODY_LEN > MAX_TITLE_LEN


# ---------------------------------------------------------------------------
# ExtractorResult invariants
# ---------------------------------------------------------------------------


class TestExtractorResult:
    def test_succeeded_requires_body(self) -> None:
        with pytest.raises(ValueError, match="succeeded=True requires"):
            ExtractorResult(
                title="t",
                body="",
                language="",
                succeeded=True,
                source="trafilatura",
            )

    def test_unsucceeded_must_have_empty_body(self) -> None:
        with pytest.raises(ValueError, match="succeeded=False requires"):
            ExtractorResult(
                title="",
                body="hi",
                language="",
                succeeded=False,
                source="trafilatura",
            )

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source must be one of"):
            ExtractorResult(
                title="",
                body="",
                language="",
                succeeded=False,
                source="bogus",
            )

    def test_title_cap_enforced(self) -> None:
        with pytest.raises(ValueError, match="title exceeds"):
            ExtractorResult(
                title="x" * (MAX_TITLE_LEN + 1),
                body="b",
                language="",
                succeeded=True,
                source="trafilatura",
            )

    def test_body_cap_enforced(self) -> None:
        with pytest.raises(ValueError, match="body exceeds"):
            ExtractorResult(
                title="",
                body="x" * (MAX_BODY_LEN + 1),
                language="",
                succeeded=True,
                source="trafilatura",
            )

    def test_meta_is_sorted_frozen_view(self) -> None:
        r = ExtractorResult(
            title="t",
            body="b",
            language="en",
            succeeded=True,
            source="trafilatura",
            meta={"z": "1", "a": "0"},
        )
        # Sorted key order
        assert list(r.meta.keys()) == ["a", "z"]
        # Frozen view
        from types import MappingProxyType

        assert isinstance(r.meta, MappingProxyType)


# ---------------------------------------------------------------------------
# Trafilatura extractor (via injected fake)
# ---------------------------------------------------------------------------


class TestTrafilaturaExtractor:
    def test_protocol_conformance(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_ok,
        )
        assert isinstance(ext, Extractor)

    def test_happy_path(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_ok,
        )
        result = ext.extract(_SAMPLE_HTML, url="https://example.com/a")
        assert result.succeeded is True
        assert result.source == "trafilatura"
        assert "Hello world." in result.body
        assert result.title == ""  # trafilatura does not emit titles

    def test_empty_html_short_circuits(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_ok,
        )
        result = ext.extract("")
        assert result.succeeded is False
        assert result.source == "trafilatura"
        assert result.meta["reason"] == "empty_html"

    def test_none_return_becomes_unsucceeded(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_none,
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "no_content"

    def test_exception_becomes_unsucceeded(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_raises,
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "exception"
        assert result.meta["exception_class"] == "RuntimeError"

    def test_whitespace_only_becomes_unsucceeded(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=lambda html, **_kw: "   \n  \t   ",
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "whitespace_only"

    def test_truncation_flagged(self) -> None:
        big = "z" * (MAX_BODY_LEN + 100)
        ext = trafilatura_extractor_factory(
            extract_callable=lambda html, **_kw: big,
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is True
        assert len(result.body) == MAX_BODY_LEN
        assert result.meta["body_truncated"] == "1"


# ---------------------------------------------------------------------------
# Newspaper3k extractor (via injected fake)
# ---------------------------------------------------------------------------


class TestNewspaper3kExtractor:
    def test_protocol_conformance(self) -> None:
        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticle(url),
        )
        assert isinstance(ext, Extractor)

    def test_happy_path(self) -> None:
        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticle(url),
        )
        result = ext.extract(_SAMPLE_HTML, url="https://example.com/a")
        assert result.succeeded is True
        assert result.source == "newspaper3k"
        assert result.title == "Sample Story"
        assert result.body == "Hello world. Second paragraph for context."
        assert result.language == "en"

    def test_empty_html(self) -> None:
        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticle(url),
        )
        result = ext.extract("")
        assert result.succeeded is False
        assert result.meta["reason"] == "empty_html"

    def test_empty_article_unsucceeded(self) -> None:
        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticleEmpty(url),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "no_content"

    def test_exception_fail_soft(self) -> None:
        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticleRaises(url),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "exception"
        assert result.meta["exception_class"] == "RuntimeError"

    def test_language_normalised(self) -> None:
        class _ZhArticle(_FakeArticle):
            def set_html(self, html: str) -> None:
                super().set_html(html)
                self.meta_lang = "zh-CN"

        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _ZhArticle(url),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.language == "zh"

    def test_language_garbage_dropped(self) -> None:
        class _JunkArticle(_FakeArticle):
            def set_html(self, html: str) -> None:
                super().set_html(html)
                self.meta_lang = "1!"

        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _JunkArticle(url),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.language == ""

    def test_title_truncation_flagged(self) -> None:
        class _BigTitleArticle(_FakeArticle):
            def set_html(self, html: str) -> None:
                super().set_html(html)
                self.title = "T" * (MAX_TITLE_LEN + 5)

        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _BigTitleArticle(url),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert len(result.title) == MAX_TITLE_LEN
        assert result.meta["title_truncated"] == "1"


# ---------------------------------------------------------------------------
# Readability extractor (via injected fake)
# ---------------------------------------------------------------------------


class TestReadabilityExtractor:
    def test_protocol_conformance(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        assert isinstance(ext, Extractor)

    def test_happy_path(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        result = ext.extract(_SAMPLE_HTML, url="https://example.com/a")
        assert result.succeeded is True
        assert result.source == "readability"
        assert result.title == "Sample Story"
        assert "Hello world." in result.body
        assert "Second paragraph" in result.body
        assert result.language == ""  # readability never returns lang

    def test_empty_html_short_circuit(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        result = ext.extract("")
        assert result.succeeded is False
        assert result.meta["reason"] == "empty_html"

    def test_empty_summary_unsucceeded(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocumentEmpty(html),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "no_content"

    def test_exception_fail_soft(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocumentRaises(html),
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is False
        assert result.meta["reason"] == "exception"

    def test_strip_tags_injectable(self) -> None:
        seen: list[str] = []

        def _strip(html: str) -> str:
            seen.append(html)
            return "stripped-body"

        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
            strip_tags=_strip,
        )
        result = ext.extract(_SAMPLE_HTML)
        assert result.succeeded is True
        assert result.body == "stripped-body"
        assert len(seen) == 1


# ---------------------------------------------------------------------------
# Pipeline cascade
# ---------------------------------------------------------------------------


class _StubExtractor:
    def __init__(
        self,
        *,
        source: str,
        succeed: bool,
        body: str = "",
        title: str = "",
    ) -> None:
        self._source = source
        self._succeed = succeed
        self._body = body
        self._title = title
        self.calls = 0

    def extract(self, html: str, *, url: str = "") -> ExtractorResult:
        self.calls += 1
        if self._succeed:
            return ExtractorResult(
                title=self._title,
                body=self._body or "default",
                language="",
                succeeded=True,
                source=self._source,
            )
        return ExtractorResult(
            title="",
            body="",
            language="",
            succeeded=False,
            source=self._source,
            meta={"reason": "stub_unsucceeded"},
        )


class TestExtractorPipeline:
    def test_requires_at_least_one_extractor(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ExtractorPipeline(extractors=())

    def test_first_hit_wins(self) -> None:
        a = _StubExtractor(source="trafilatura", succeed=True, body="A")
        b = _StubExtractor(source="newspaper3k", succeed=True, body="B")
        c = _StubExtractor(source="readability", succeed=True, body="C")
        pipeline = ExtractorPipeline(extractors=(a, b, c))
        result = pipeline.extract(_SAMPLE_HTML)
        assert result.source == "trafilatura"
        assert result.body == "A"
        # b and c never invoked
        assert a.calls == 1
        assert b.calls == 0
        assert c.calls == 0

    def test_cascade_to_second(self) -> None:
        a = _StubExtractor(source="trafilatura", succeed=False)
        b = _StubExtractor(source="newspaper3k", succeed=True, body="B")
        c = _StubExtractor(source="readability", succeed=True, body="C")
        pipeline = ExtractorPipeline(extractors=(a, b, c))
        result = pipeline.extract(_SAMPLE_HTML)
        assert result.source == "newspaper3k"
        assert result.body == "B"
        assert a.calls == 1
        assert b.calls == 1
        assert c.calls == 0

    def test_cascade_to_third(self) -> None:
        a = _StubExtractor(source="trafilatura", succeed=False)
        b = _StubExtractor(source="newspaper3k", succeed=False)
        c = _StubExtractor(source="readability", succeed=True, body="C")
        pipeline = ExtractorPipeline(extractors=(a, b, c))
        result = pipeline.extract(_SAMPLE_HTML)
        assert result.source == "readability"
        assert result.body == "C"

    def test_all_fail_returns_none(self) -> None:
        a = _StubExtractor(source="trafilatura", succeed=False)
        b = _StubExtractor(source="newspaper3k", succeed=False)
        c = _StubExtractor(source="readability", succeed=False)
        pipeline = ExtractorPipeline(extractors=(a, b, c))
        result = pipeline.extract(_SAMPLE_HTML)
        assert result.source == "none"
        assert result.succeeded is False
        assert result.body == ""
        assert result.meta["attempted"] == "trafilatura,newspaper3k,readability"

    def test_url_forwarded_to_each_extractor(self) -> None:
        seen: list[str] = []

        class _UrlSpy:
            def extract(
                self,
                html: str,
                *,
                url: str = "",
            ) -> ExtractorResult:
                seen.append(url)
                return ExtractorResult(
                    title="",
                    body="",
                    language="",
                    succeeded=False,
                    source="trafilatura",
                    meta={"reason": "spy"},
                )

        pipeline = ExtractorPipeline(extractors=(_UrlSpy(),))
        pipeline.extract(_SAMPLE_HTML, url="https://example.com/x")
        assert seen == ["https://example.com/x"]


class TestBuildDefaultPipeline:
    def test_order_is_trafilatura_then_newspaper_then_readability(self) -> None:
        traf = _StubExtractor(source="trafilatura", succeed=False)
        news = _StubExtractor(source="newspaper3k", succeed=False)
        read = _StubExtractor(source="readability", succeed=True, body="R")
        pipeline = build_default_pipeline(
            trafilatura=traf,
            newspaper3k=news,
            readability=read,
        )
        result = pipeline.extract(_SAMPLE_HTML)
        assert result.source == "readability"
        # Order: trafilatura first, then newspaper3k, then readability.
        assert traf.calls == 1
        assert news.calls == 1
        assert read.calls == 1


# ---------------------------------------------------------------------------
# Real readability HTML-strip fallback (stdlib only — no readability needed)
# ---------------------------------------------------------------------------


class TestReadabilityHtmlStrip:
    def test_default_strip_removes_tags(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        result = ext.extract(_SAMPLE_HTML)
        # No leading "<", no trailing ">" — tags are stripped, plain
        # text remains.
        assert "<" not in result.body
        assert ">" not in result.body
        assert "Hello world." in result.body


# ---------------------------------------------------------------------------
# INV-15 determinism: 3-run equality on byte projection
# ---------------------------------------------------------------------------


def _project(result: ExtractorResult) -> tuple[object, ...]:
    """Stable byte projection — used for 3-run equality."""

    return (
        result.title,
        result.body,
        result.language,
        result.succeeded,
        result.source,
        tuple(sorted(result.meta.items())),
    )


class TestInv15Determinism:
    def test_trafilatura_three_run_equality(self) -> None:
        ext = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_ok,
        )
        projections = [_project(ext.extract(_SAMPLE_HTML, url="https://x/a")) for _ in range(3)]
        assert projections[0] == projections[1] == projections[2]

    def test_newspaper3k_three_run_equality(self) -> None:
        ext = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticle(url),
        )
        projections = [_project(ext.extract(_SAMPLE_HTML, url="https://x/a")) for _ in range(3)]
        assert projections[0] == projections[1] == projections[2]

    def test_readability_three_run_equality(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        projections = [_project(ext.extract(_SAMPLE_HTML, url="https://x/a")) for _ in range(3)]
        assert projections[0] == projections[1] == projections[2]

    def test_pipeline_three_run_equality(self) -> None:
        # New stubs per run so call counters do not interfere.
        runs: list[tuple[object, ...]] = []
        for _ in range(3):
            a2 = _StubExtractor(source="trafilatura", succeed=False)
            b2 = _StubExtractor(source="newspaper3k", succeed=True, body="B")
            p = ExtractorPipeline(extractors=(a2, b2))
            runs.append(_project(p.extract(_SAMPLE_HTML)))
        assert runs[0] == runs[1] == runs[2]

    def test_meta_key_order_independent(self) -> None:
        r1 = ExtractorResult(
            title="t",
            body="b",
            language="en",
            succeeded=True,
            source="trafilatura",
            meta={"a": "0", "z": "1"},
        )
        r2 = ExtractorResult(
            title="t",
            body="b",
            language="en",
            succeeded=True,
            source="trafilatura",
            meta={"z": "1", "a": "0"},
        )
        assert _project(r1) == _project(r2)


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _imported_modules(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def _top_level_imported_modules(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


class TestAstGuards:
    def test_no_top_level_external_imports(self) -> None:
        top = _top_level_imported_modules(_MODULE_AST)
        forbidden = {"trafilatura", "newspaper", "readability", "lxml"}
        assert forbidden.isdisjoint(top), f"top-level imports forbidden: {top & forbidden}"

    def test_no_engine_imports(self) -> None:
        imported = _imported_modules(_MODULE_AST)
        forbidden = {
            "governance_engine",
            "system_engine",
            "execution_engine",
            "evolution_engine",
            "intelligence_engine",
            "dyon",
        }
        assert forbidden.isdisjoint(imported), (
            f"engine cross-imports forbidden: {imported & forbidden}"
        )

    def test_no_clock_or_random_or_io_imports(self) -> None:
        imported = _imported_modules(_MODULE_AST)
        forbidden = {
            "time",
            "datetime",
            "random",
            "asyncio",
            "os",
            "websockets",
            "numpy",
            "torch",
            "polars",
            "langsmith",
        }
        assert forbidden.isdisjoint(imported), (
            f"clock/random/IO forbidden imports: {imported & forbidden}"
        )

    def test_no_typed_event_construction(self) -> None:
        banned = {
            "SignalEvent",
            "ExecutionEvent",
            "SystemEvent",
            "HazardEvent",
            "PatchProposal",
            "GovernanceDecision",
            "OperatorDirective",
            "ExecutionIntent",
        }
        for node in ast.walk(_MODULE_AST):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in banned, (
                    f"extractors.py must not construct {node.func.id!r}"
                )

    def test_adapted_from_header_present(self) -> None:
        assert _MODULE_SOURCE.startswith(
            "# ADAPTED FROM: adbar/trafilatura + codelucas/newspaper + buriy/python-readability"
        )

    def test_lazy_imports_only_inside_function_bodies(self) -> None:
        # Walk function bodies; collect ImportFrom/Import that name
        # the three external packages. They must exist (factories
        # rely on them) but never appear at module top level.
        external = {"trafilatura", "newspaper", "readability"}

        def _walk_func_imports(
            tree: ast.AST,
        ) -> set[str]:
            seen: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Import):
                            for alias in inner.names:
                                head = alias.name.split(".")[0]
                                if head in external:
                                    seen.add(head)
                        elif isinstance(inner, ast.ImportFrom):
                            if inner.module:
                                head = inner.module.split(".")[0]
                                if head in external:
                                    seen.add(head)
            return seen

        func_imports = _walk_func_imports(_MODULE_AST)
        assert "trafilatura" in func_imports, "trafilatura must be lazy-imported in a function body"
        assert "newspaper" in func_imports, "newspaper must be lazy-imported in a function body"
        assert "readability" in func_imports, "readability must be lazy-imported in a function body"

    def test_extractor_factory_protocol_returns(self) -> None:
        # Sanity: each factory advertises return type :class:`Extractor`.
        # We don't AST-walk return annotations; instead exercise the
        # runtime_checkable Protocol with the test doubles to confirm
        # the contract holds end-to-end.
        traf = trafilatura_extractor_factory(
            extract_callable=_fake_trafilatura_ok,
        )
        news = newspaper3k_extractor_factory(
            article_factory=lambda url: _FakeArticle(url),
        )
        read = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        for ext in (traf, news, read):
            assert isinstance(ext, Extractor)

    def test_meta_immutable_after_construction(self) -> None:
        r = ExtractorResult(
            title="t",
            body="b",
            language="en",
            succeeded=True,
            source="trafilatura",
            meta={"a": "0"},
        )
        from types import MappingProxyType

        assert isinstance(r.meta, MappingProxyType)
        assert isinstance(r.meta, Mapping)


# ---------------------------------------------------------------------------
# Real readability strip-tags integration (no external deps)
# ---------------------------------------------------------------------------


class TestRealStripTags:
    """The default strip_tags helper is stdlib-only and shipped in the
    module; exercise it via the readability extractor with a fake
    document so we run the real :func:`_strip_html_tags`.
    """

    def test_default_strip_collapses_whitespace(self) -> None:
        ext = readability_extractor_factory(
            document_factory=lambda html: _FakeDocument(html),
        )
        result = ext.extract(_SAMPLE_HTML)
        # No double spaces, no tags
        assert "  " not in result.body
        assert result.body.startswith("Hello world.")
