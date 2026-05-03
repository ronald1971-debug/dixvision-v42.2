"""Tests for the deterministic news knowledge index (D4)."""

from __future__ import annotations

import pytest

from core.contracts.news import NewsItem
from intelligence_engine.knowledge import (
    KNOWLEDGE_INDEX_VERSION,
    NewsKnowledgeIndex,
)


def _item(
    *,
    ts_ns: int,
    source: str = "COINDESK",
    guid: str = "g1",
    title: str = "",
    summary: str = "",
) -> NewsItem:
    return NewsItem(
        ts_ns=ts_ns,
        source=source,
        guid=guid,
        title=title,
        url="",
        summary=summary,
    )


def test_add_and_query_returns_higher_score_for_better_match() -> None:
    idx = NewsKnowledgeIndex()
    idx.add(
        _item(
            ts_ns=1,
            guid="a",
            title="Bitcoin rallies on ETF inflows",
            summary="institutional demand surges",
        )
    )
    idx.add(
        _item(
            ts_ns=2,
            guid="b",
            title="Ethereum staking yield drops",
            summary="validators rotate offline",
        )
    )
    hits = idx.query("bitcoin etf", top_k=2)
    assert len(hits) >= 1
    assert hits[0].item.guid == "a"
    if len(hits) > 1:
        assert hits[0].score >= hits[1].score


def test_query_top_k_caps_results() -> None:
    idx = NewsKnowledgeIndex()
    for i in range(5):
        idx.add(
            _item(
                ts_ns=i,
                guid=f"g{i}",
                title="bitcoin rally" if i % 2 == 0 else "rally",
            )
        )
    hits = idx.query("bitcoin rally", top_k=2)
    assert len(hits) == 2


def test_query_with_min_score_drops_unrelated() -> None:
    idx = NewsKnowledgeIndex()
    idx.add(_item(ts_ns=1, guid="a", title="bitcoin etf"))
    idx.add(_item(ts_ns=2, guid="b", title="weather report cloudy"))
    hits = idx.query("bitcoin", top_k=5, min_score=0.1)
    assert len(hits) == 1
    assert hits[0].item.guid == "a"


def test_query_filtered_by_source() -> None:
    idx = NewsKnowledgeIndex()
    idx.add(
        _item(
            ts_ns=1, source="COINDESK", guid="a", title="ethereum upgrade"
        )
    )
    idx.add(
        _item(
            ts_ns=2, source="REUTERS", guid="a", title="ethereum upgrade"
        )
    )
    hits = idx.query("ethereum", top_k=5, source="REUTERS")
    assert len(hits) == 1
    assert hits[0].item.source == "REUTERS"


def test_add_duplicate_returns_false() -> None:
    idx = NewsKnowledgeIndex()
    item = _item(ts_ns=1, guid="x", title="hello world")
    assert idx.add(item) is True
    assert idx.add(item) is False
    assert len(idx) == 1


def test_eviction_kicks_out_oldest_when_full() -> None:
    idx = NewsKnowledgeIndex(max_items=3)
    idx.add(_item(ts_ns=1, guid="a", title="alpha"))
    idx.add(_item(ts_ns=2, guid="b", title="beta"))
    idx.add(_item(ts_ns=3, guid="c", title="gamma"))
    idx.add(_item(ts_ns=4, guid="d", title="delta"))
    assert len(idx) == 3
    # The oldest (ts_ns=1, guid="a") must be gone.
    hits = idx.query("alpha", top_k=5)
    assert all(h.item.guid != "a" for h in hits)


def test_drop_removes_row() -> None:
    idx = NewsKnowledgeIndex()
    idx.add(_item(ts_ns=1, guid="x", title="hello"))
    assert idx.drop("COINDESK", "x") is True
    assert idx.drop("COINDESK", "x") is False
    assert len(idx) == 0


def test_query_is_deterministic_across_calls() -> None:
    idx_a = NewsKnowledgeIndex()
    idx_b = NewsKnowledgeIndex()
    items = [
        _item(ts_ns=i, guid=f"g{i}", title=f"bitcoin item {i}")
        for i in range(10)
    ]
    # Insert in two different orders — the index is keyed on
    # (source, guid) so insertion order must not affect the result.
    for it in items:
        idx_a.add(it)
    for it in reversed(items):
        idx_b.add(it)
    hits_a = idx_a.query("bitcoin", top_k=5)
    hits_b = idx_b.query("bitcoin", top_k=5)
    assert tuple(h.item.guid for h in hits_a) == tuple(
        h.item.guid for h in hits_b
    )


def test_empty_query_text_returns_no_hits() -> None:
    idx = NewsKnowledgeIndex()
    idx.add(_item(ts_ns=1, guid="x", title="hello world"))
    assert idx.query("", top_k=5) == ()
    # All-whitespace and pure punctuation also tokenize to nothing.
    assert idx.query("   ", top_k=5) == ()
    assert idx.query("!@#$%", top_k=5) == ()


def test_query_zero_top_k_raises() -> None:
    idx = NewsKnowledgeIndex()
    with pytest.raises(ValueError):
        idx.query("hello", top_k=0)


def test_max_items_must_be_positive() -> None:
    with pytest.raises(ValueError):
        NewsKnowledgeIndex(max_items=0)


def test_empty_source_or_guid_rejected() -> None:
    idx = NewsKnowledgeIndex()
    with pytest.raises(ValueError):
        idx.add(_item(ts_ns=1, source="", guid="x"))
    with pytest.raises(ValueError):
        idx.add(_item(ts_ns=1, source="X", guid=""))


def test_stats_reflects_index_state() -> None:
    idx = NewsKnowledgeIndex(max_items=10)
    idx.add(
        _item(
            ts_ns=1,
            source="COINDESK",
            guid="a",
            title="bitcoin etf flows",
        )
    )
    idx.add(
        _item(
            ts_ns=2,
            source="REUTERS",
            guid="b",
            title="ethereum staking",
        )
    )
    stats = idx.stats()
    assert stats.size == 2
    assert stats.max_items == 10
    assert stats.version == KNOWLEDGE_INDEX_VERSION
    assert stats.unique_sources == 2
    assert stats.unique_tokens >= 4


def test_sources_returns_sorted_distinct_values() -> None:
    idx = NewsKnowledgeIndex()
    idx.add(_item(ts_ns=1, source="REUTERS", guid="a", title="hello"))
    idx.add(_item(ts_ns=2, source="COINDESK", guid="b", title="hello"))
    idx.add(_item(ts_ns=3, source="COINDESK", guid="c", title="world"))
    assert idx.sources() == ("COINDESK", "REUTERS")
