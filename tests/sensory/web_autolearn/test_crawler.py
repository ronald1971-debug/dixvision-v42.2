"""Unit tests for sensory.web_autolearn.crawler."""

from __future__ import annotations

import pytest

from sensory.web_autolearn.crawler import (
    Crawler,
    DeterministicCrawler,
    _PreparedDocument,
)


def test_deterministic_crawler_satisfies_protocol() -> None:
    crawler = DeterministicCrawler.from_pairs(("s", "https://x"))
    assert isinstance(crawler, Crawler)


def test_deterministic_crawler_returns_one_per_seed_in_order() -> None:
    crawler = DeterministicCrawler.from_pairs(
        ("a", "https://a"),
        ("b", "https://b"),
        ("c", "https://c"),
    )
    out = crawler.fetch(("c", "a", "b"), ts_ns=42)
    urls = [d.url for d in out]
    assert urls == ["https://c", "https://a", "https://b"]
    assert all(d.ts_ns == 42 for d in out)
    assert all(d.fetched_ok for d in out)


def test_deterministic_crawler_unknown_seed_is_failsoft() -> None:
    crawler = DeterministicCrawler.from_pairs(("a", "https://a"))
    out = crawler.fetch(("a", "missing"), ts_ns=10)

    assert out[0].fetched_ok is True
    assert out[0].url == "https://a"
    assert out[1].fetched_ok is False
    assert out[1].seed_id == "missing"
    assert out[1].url.startswith("about:unknown/")
    assert out[1].meta["error"] == "unknown_seed"


def test_deterministic_crawler_preserves_duplicate_seeds() -> None:
    """If the caller asks for the same seed twice, both copies appear."""

    crawler = DeterministicCrawler.from_pairs(("a", "https://a"))
    out = crawler.fetch(("a", "a", "a"), ts_ns=1)
    assert len(out) == 3
    assert all(d.url == "https://a" for d in out)


def test_deterministic_crawler_from_documents_preserves_fields() -> None:
    crawler = DeterministicCrawler.from_documents(
        [
            _PreparedDocument(
                seed_id="s",
                url="https://x",
                title="T",
                body="B",
                fetched_ok=True,
                meta={"k": "v"},
            )
        ]
    )
    out = crawler.fetch(("s",), ts_ns=99)
    assert out[0].title == "T"
    assert out[0].body == "B"
    assert dict(out[0].meta) == {"k": "v"}


def test_deterministic_crawler_rejects_nonpositive_ts_ns() -> None:
    crawler = DeterministicCrawler.from_pairs(("a", "https://a"))
    with pytest.raises(ValueError, match="ts_ns"):
        crawler.fetch(("a",), ts_ns=0)
    with pytest.raises(ValueError, match="ts_ns"):
        crawler.fetch(("a",), ts_ns=-1)


def test_deterministic_crawler_replay_determinism() -> None:
    """Same inputs -> same outputs (TEST-01 / INV-15 invariant)."""

    crawler = DeterministicCrawler.from_pairs(
        ("a", "https://a"),
        ("b", "https://b"),
    )
    out1 = crawler.fetch(("a", "b"), ts_ns=100)
    out2 = crawler.fetch(("a", "b"), ts_ns=100)
    assert out1 == out2
