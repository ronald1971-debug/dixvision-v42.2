"""Tests for S-05 :mod:`sensory.web_autolearn.crawler_firecrawl`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from sensory.web_autolearn.contracts import RawDocument
from sensory.web_autolearn.crawler import Crawler
from sensory.web_autolearn.crawler_firecrawl import (
    NEW_PIP_DEPENDENCIES,
    CrawlerStatus,
    FirecrawlCrawler,
    FirecrawlCredentials,
)

# --------------------------------------------------------------------
# Stubs for the injected SDK client
# --------------------------------------------------------------------


class _RecordingClient:
    """Stub Firecrawl client that records calls + returns canned payloads.

    ``payloads`` maps URL → payload (dict or exception class). When a
    URL maps to an exception class, ``scrape_url`` raises an instance.
    """

    def __init__(self, payloads: Mapping[str, Any]) -> None:
        self._payloads = dict(payloads)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def scrape_url(
        self,
        url: str,
        *,
        formats: list[str],
    ) -> Any:
        self.calls.append((url, tuple(formats)))
        try:
            payload = self._payloads[url]
        except KeyError as exc:
            raise RuntimeError(f"_RecordingClient: no canned payload for {url!r}") from exc
        if isinstance(payload, type) and issubclass(payload, BaseException):
            raise payload(f"stub error for {url}")
        return payload


def _ok_payload(title: str, body: str, status: int = 200) -> dict[str, Any]:
    """Shape of a successful Firecrawl scrape response."""

    return {
        "markdown": body,
        "html": "<html></html>",
        "metadata": {
            "title": title,
            "description": "stub description",
            "sourceURL": "https://stub.example/canonical",
            "statusCode": status,
        },
    }


def _wrapped_payload(title: str, body: str) -> dict[str, Any]:
    """Newer-SDK shape that nests the payload under ``"data"``."""

    return {"data": _ok_payload(title, body)}


# --------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("firecrawl-py",)


def test_satisfies_crawler_protocol() -> None:
    crawler = FirecrawlCrawler(
        seed_urls={"a": "https://a.example"},
        credentials=FirecrawlCredentials(api_key="fc-test"),
        client_factory=lambda _c: _RecordingClient({}),
    )
    assert isinstance(crawler, Crawler)


# --------------------------------------------------------------------
# FirecrawlCredentials validation
# --------------------------------------------------------------------


def test_credentials_reject_empty_key() -> None:
    with pytest.raises(ValueError):
        FirecrawlCredentials(api_key="")


def test_credentials_reject_non_str_key() -> None:
    with pytest.raises(TypeError):
        FirecrawlCredentials(api_key=123)  # type: ignore[arg-type]


def test_credentials_frozen() -> None:
    creds = FirecrawlCredentials(api_key="fc-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        creds.api_key = "fc-2"  # type: ignore[misc]


def test_credentials_hashable_and_eq() -> None:
    a = FirecrawlCredentials(api_key="fc-1")
    b = FirecrawlCredentials(api_key="fc-1")
    assert a == b
    assert hash(a) == hash(b)
    assert FirecrawlCredentials(api_key="fc-2") != a


# --------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------


def test_constructor_rejects_empty_seed_urls() -> None:
    with pytest.raises(ValueError):
        FirecrawlCrawler(seed_urls={})


def test_constructor_rejects_empty_seed_id() -> None:
    with pytest.raises(ValueError):
        FirecrawlCrawler(seed_urls={"": "https://x.example"})


def test_constructor_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        FirecrawlCrawler(seed_urls={"a": ""})


def test_constructor_rejects_non_str_url() -> None:
    with pytest.raises(ValueError):
        FirecrawlCrawler(seed_urls={"a": 5})  # type: ignore[dict-item]


def test_constructor_rejects_empty_request_formats() -> None:
    with pytest.raises(ValueError):
        FirecrawlCrawler(
            seed_urls={"a": "https://a.example"},
            request_formats=(),
        )


def test_constructor_rejects_blank_request_format_entry() -> None:
    with pytest.raises(ValueError):
        FirecrawlCrawler(
            seed_urls={"a": "https://a.example"},
            request_formats=("markdown", ""),
        )


def test_seed_urls_property_returns_immutable_view() -> None:
    src = {"a": "https://a.example"}
    crawler = FirecrawlCrawler(seed_urls=src)
    view = crawler.seed_urls
    # Mutating the source dict must not affect the crawler.
    src["b"] = "https://b.example"
    assert dict(view) == {"a": "https://a.example"}
    # The exposed mapping itself must be read-only.
    with pytest.raises(TypeError):
        view["c"] = "https://c.example"  # type: ignore[index]


def test_request_formats_default_is_markdown_only() -> None:
    crawler = FirecrawlCrawler(seed_urls={"a": "https://a.example"})
    assert crawler.request_formats == ("markdown",)


# --------------------------------------------------------------------
# Lifecycle: scaffold mode
# --------------------------------------------------------------------


def test_scaffold_mode_status_is_disconnected() -> None:
    crawler = FirecrawlCrawler(seed_urls={"a": "https://a.example"})
    assert crawler.status is CrawlerStatus.DISCONNECTED
    assert not crawler.is_ready


def test_scaffold_mode_connect_raises() -> None:
    crawler = FirecrawlCrawler(seed_urls={"a": "https://a.example"})
    with pytest.raises(RuntimeError, match="scaffold mode"):
        crawler.connect()


def test_scaffold_mode_fetch_raises() -> None:
    crawler = FirecrawlCrawler(seed_urls={"a": "https://a.example"})
    with pytest.raises(RuntimeError, match="adapter_not_ready"):
        crawler.fetch(["a"], ts_ns=1)


# --------------------------------------------------------------------
# Lifecycle: connect / disconnect with injected factory
# --------------------------------------------------------------------


def test_connect_uses_injected_factory() -> None:
    captured: list[FirecrawlCredentials] = []

    def factory(creds: FirecrawlCredentials) -> _RecordingClient:
        captured.append(creds)
        return _RecordingClient({})

    creds = FirecrawlCredentials(api_key="fc-injected")
    crawler = FirecrawlCrawler(
        seed_urls={"a": "https://a.example"},
        credentials=creds,
        client_factory=factory,
    )
    assert not crawler.is_ready
    crawler.connect()
    assert crawler.is_ready
    assert crawler.status is CrawlerStatus.CONNECTED
    assert captured == [creds]


def test_disconnect_returns_to_disconnected() -> None:
    crawler = FirecrawlCrawler(
        seed_urls={"a": "https://a.example"},
        credentials=FirecrawlCredentials(api_key="fc"),
        client_factory=lambda _c: _RecordingClient({}),
    )
    crawler.connect()
    crawler.disconnect()
    assert crawler.status is CrawlerStatus.DISCONNECTED
    assert not crawler.is_ready


def test_disconnect_is_idempotent() -> None:
    crawler = FirecrawlCrawler(seed_urls={"a": "https://a.example"})
    crawler.disconnect()
    crawler.disconnect()
    assert crawler.status is CrawlerStatus.DISCONNECTED


def test_fetch_after_disconnect_raises() -> None:
    crawler = FirecrawlCrawler(
        seed_urls={"a": "https://a.example"},
        credentials=FirecrawlCredentials(api_key="fc"),
        client_factory=lambda _c: _RecordingClient({"https://a.example": _ok_payload("t", "b")}),
    )
    crawler.connect()
    crawler.disconnect()
    with pytest.raises(RuntimeError, match="adapter_not_ready"):
        crawler.fetch(["a"], ts_ns=1)


# --------------------------------------------------------------------
# fetch() — happy path
# --------------------------------------------------------------------


def _connected(
    seed_urls: Mapping[str, str],
    payloads: Mapping[str, Any],
    *,
    request_formats: tuple[str, ...] = ("markdown",),
) -> tuple[FirecrawlCrawler, _RecordingClient]:
    """Build a CONNECTED crawler with a stub client. Helper for tests."""

    client = _RecordingClient(payloads)
    crawler = FirecrawlCrawler(
        seed_urls=seed_urls,
        credentials=FirecrawlCredentials(api_key="fc"),
        client_factory=lambda _c: client,
        request_formats=request_formats,
    )
    crawler.connect()
    return crawler, client


def test_fetch_single_seed_returns_one_document() -> None:
    crawler, _client = _connected(
        {"news": "https://news.example/feed"},
        {"https://news.example/feed": _ok_payload("Headline", "Body text")},
    )
    docs = crawler.fetch(["news"], ts_ns=10_000)
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, RawDocument)
    assert doc.seed_id == "news"
    assert doc.url == "https://news.example/feed"
    assert doc.title == "Headline"
    assert doc.body == "Body text"
    assert doc.fetched_ok is True
    assert doc.ts_ns == 10_000
    # Metadata projection
    assert doc.meta["status_code"] == "200"
    assert doc.meta["description"] == "stub description"
    assert doc.meta["source_url"] == "https://stub.example/canonical"


def test_fetch_preserves_caller_seed_order() -> None:
    crawler, _client = _connected(
        {
            "a": "https://a.example",
            "b": "https://b.example",
            "c": "https://c.example",
        },
        {
            "https://a.example": _ok_payload("A", "body-a"),
            "https://b.example": _ok_payload("B", "body-b"),
            "https://c.example": _ok_payload("C", "body-c"),
        },
    )
    docs = crawler.fetch(["c", "a", "b"], ts_ns=1)
    assert [d.seed_id for d in docs] == ["c", "a", "b"]
    assert [d.title for d in docs] == ["C", "A", "B"]


def test_fetch_handles_duplicate_seeds() -> None:
    crawler, client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "body-a")},
    )
    docs = crawler.fetch(["a", "a", "a"], ts_ns=2)
    assert len(docs) == 3
    # Each duplicate triggers its own scrape (no implicit cache).
    assert client.calls == [
        ("https://a.example", ("markdown",)),
        ("https://a.example", ("markdown",)),
        ("https://a.example", ("markdown",)),
    ]
    # All three carry the same caller-supplied ts_ns.
    assert {d.ts_ns for d in docs} == {2}


def test_fetch_passes_request_formats_to_sdk() -> None:
    crawler, client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "B")},
        request_formats=("markdown", "html"),
    )
    crawler.fetch(["a"], ts_ns=1)
    assert client.calls == [
        ("https://a.example", ("markdown", "html")),
    ]


def test_fetch_accepts_wrapped_data_payload() -> None:
    """Newer-SDK responses nest the body under ``data`` — handle both."""

    crawler, _client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _wrapped_payload("Wrapped", "wrapped body")},
    )
    docs = crawler.fetch(["a"], ts_ns=5)
    assert docs[0].fetched_ok is True
    assert docs[0].title == "Wrapped"
    assert docs[0].body == "wrapped body"


# --------------------------------------------------------------------
# fetch() — fail-soft
# --------------------------------------------------------------------


def test_fetch_unknown_seed_returns_failsoft_doc() -> None:
    crawler, client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "B")},
    )
    docs = crawler.fetch(["unknown_seed"], ts_ns=3)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.fetched_ok is False
    assert doc.seed_id == "unknown_seed"
    assert doc.url == "about:unknown/unknown_seed"
    assert doc.meta["error"] == "unknown_seed"
    # Unknown seeds must not hit the SDK at all.
    assert client.calls == []


def test_fetch_sdk_error_returns_failsoft_doc() -> None:
    crawler, client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": RuntimeError},
    )
    docs = crawler.fetch(["a"], ts_ns=4)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.fetched_ok is False
    assert doc.seed_id == "a"
    assert doc.url == "https://a.example"
    assert doc.meta["error"] == "scrape_failed"
    assert doc.meta["error_class"] == "RuntimeError"
    # The SDK was actually called once before the failure.
    assert client.calls == [("https://a.example", ("markdown",))]


def test_fetch_continues_after_per_seed_error() -> None:
    crawler, _client = _connected(
        {
            "a": "https://a.example",
            "b": "https://b.example",
            "c": "https://c.example",
        },
        {
            "https://a.example": _ok_payload("A", "body-a"),
            "https://b.example": ValueError,
            "https://c.example": _ok_payload("C", "body-c"),
        },
    )
    docs = crawler.fetch(["a", "b", "c"], ts_ns=7)
    assert [d.fetched_ok for d in docs] == [True, False, True]
    assert docs[1].meta["error_class"] == "ValueError"
    # Successful neighbours unaffected.
    assert docs[0].title == "A"
    assert docs[2].title == "C"


def test_fetch_non_mapping_payload_returns_failsoft() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": "not-a-mapping"},
    )
    docs = crawler.fetch(["a"], ts_ns=8)
    assert docs[0].fetched_ok is False
    assert docs[0].meta["error"] == "non_mapping_response"


def test_fetch_payload_missing_metadata_falls_back_to_blank_title() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {
            "https://a.example": {
                "markdown": "just a body",
                # No metadata key at all.
            }
        },
    )
    docs = crawler.fetch(["a"], ts_ns=9)
    assert docs[0].fetched_ok is True
    assert docs[0].title == ""
    assert docs[0].body == "just a body"


def test_fetch_payload_with_none_fields_coerces_to_empty_str() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {
            "https://a.example": {
                "markdown": None,
                "metadata": {"title": None, "statusCode": 200},
            }
        },
    )
    docs = crawler.fetch(["a"], ts_ns=10)
    assert docs[0].fetched_ok is True
    assert docs[0].title == ""
    assert docs[0].body == ""
    assert docs[0].meta["status_code"] == "200"


def test_fetch_payload_with_non_int_status_code_omits_field() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {
            "https://a.example": {
                "markdown": "x",
                "metadata": {"title": "T", "statusCode": "200-ish"},
            }
        },
    )
    docs = crawler.fetch(["a"], ts_ns=11)
    assert "status_code" not in docs[0].meta


# --------------------------------------------------------------------
# fetch() — argument validation
# --------------------------------------------------------------------


def test_fetch_rejects_zero_ts_ns() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "B")},
    )
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        crawler.fetch(["a"], ts_ns=0)


def test_fetch_rejects_negative_ts_ns() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "B")},
    )
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        crawler.fetch(["a"], ts_ns=-1)


def test_fetch_empty_seed_list_returns_empty_tuple() -> None:
    crawler, client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "B")},
    )
    docs = crawler.fetch([], ts_ns=1)
    assert docs == ()
    # Empty seed list never touches the SDK.
    assert client.calls == []


def test_fetch_returns_tuple_not_list() -> None:
    crawler, _client = _connected(
        {"a": "https://a.example"},
        {"https://a.example": _ok_payload("A", "B")},
    )
    docs = crawler.fetch(["a"], ts_ns=1)
    assert isinstance(docs, tuple)


# --------------------------------------------------------------------
# Replay determinism (INV-15)
# --------------------------------------------------------------------


def _build_replay_crawler() -> FirecrawlCrawler:
    payloads = {
        "https://a.example": _ok_payload("A", "body-a"),
        "https://b.example": _ok_payload("B", "body-b"),
        "https://c.example": ValueError,  # deterministic failure
    }
    return _connected(
        {
            "a": "https://a.example",
            "b": "https://b.example",
            "c": "https://c.example",
        },
        payloads,
    )[0]


def test_replay_determinism_byte_identical() -> None:
    seeds = ["b", "a", "c", "a"]
    runs: list[tuple[RawDocument, ...]] = []
    for _ in range(3):
        crawler = _build_replay_crawler()
        runs.append(tuple(crawler.fetch(seeds, ts_ns=12_345_678)))
    # All three replays must be byte-identical.
    assert runs[0] == runs[1] == runs[2]


def test_replay_determinism_meta_dict_stable() -> None:
    crawler1 = _build_replay_crawler()
    crawler2 = _build_replay_crawler()
    docs1 = crawler1.fetch(["a", "b"], ts_ns=1)
    docs2 = crawler2.fetch(["a", "b"], ts_ns=1)
    # Frozen RawDocument equality covers meta-mapping equality too.
    assert docs1 == docs2


# --------------------------------------------------------------------
# No clock / no os.environ
# --------------------------------------------------------------------


def _module_ast() -> Any:
    import ast

    import sensory.web_autolearn.crawler_firecrawl as mod

    src = mod.__file__
    assert src is not None
    with open(src, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def test_module_does_not_import_os_or_time() -> None:
    """No top-level `import os` / `import time` / `import datetime`.

    Pure AST walk so docstring mentions of these names don't false-
    positive (the module's own docstring talks about ``os.environ``
    in prose to explain the design).
    """

    import ast

    tree = _module_ast()
    forbidden = {"os", "time", "datetime"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden, f"forbidden top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            assert root not in forbidden, f"forbidden from-import: {node.module}"


def test_module_does_not_call_environ_or_clock() -> None:
    """No `os.environ`, `os.getenv`, `time.time`, `datetime.now`, etc.

    AST attribute walk — robust against docstring text that mentions
    these names in prose.
    """

    import ast

    tree = _module_ast()
    forbidden_chains = {
        ("os", "environ"),
        ("os", "getenv"),
        ("time", "time"),
        ("time", "sleep"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("asyncio", "sleep"),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            chain = (node.value.id, node.attr)
            assert chain not in forbidden_chains, (
                f"forbidden attribute access: {chain[0]}.{chain[1]}"
            )


# --------------------------------------------------------------------
# AGPL mitigation: only `scrape_url` is consumed
# --------------------------------------------------------------------


def test_agpl_mitigation_only_scrape_url_call() -> None:
    """The adapter must not reach into the SDK beyond ``scrape_url``.

    AGPL exposure scales with how much of the AGPL code we touch. The
    mitigation strategy in the master canonical doc is to consume
    only the public ``FirecrawlApp.scrape_url`` interface. This test
    pins that contract by walking the AST for the actual call sites.
    """

    import ast

    tree = _module_ast()
    sdk_attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "client":
                sdk_attrs.add(node.func.attr)
    # The only Firecrawl SDK method we call is scrape_url.
    assert sdk_attrs == {"scrape_url"}, f"unexpected SDK call sites: {sdk_attrs}"
