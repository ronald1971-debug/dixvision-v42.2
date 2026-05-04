"""Unit tests for sensory.web_autolearn.contracts."""

from __future__ import annotations

import pytest

from sensory.web_autolearn.contracts import (
    CuratedItem,
    FilteredItem,
    NewsItem,
    RawDocument,
    SocialPost,
)


def test_news_item_reexport_is_core_contract() -> None:
    """The ``sensory.web_autolearn.contracts.NewsItem`` symbol is the
    same class as :class:`core.contracts.news.NewsItem`.

    The ``data_source_registry.yaml`` Reuters row references the
    sensory path; the canonical home stays in core.
    """

    from core.contracts.news import NewsItem as CoreNewsItem

    assert NewsItem is CoreNewsItem


def test_social_post_minimal_valid() -> None:
    post = SocialPost(
        ts_ns=10,
        source="X",
        post_id="123",
        author="user",
        body="hello",
    )
    assert post.published_ts_ns is None
    assert post.url == ""
    assert dict(post.meta) == {}


@pytest.mark.parametrize(
    "field, kwargs",
    [
        ("source", {"source": ""}),
        ("post_id", {"post_id": ""}),
        ("author", {"author": ""}),
        ("body", {"body": ""}),
    ],
)
def test_social_post_required_fields(
    field: str, kwargs: dict[str, str]
) -> None:
    base = dict(
        ts_ns=10,
        source="X",
        post_id="1",
        author="u",
        body="b",
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=field):
        SocialPost(**base)


def test_social_post_published_ts_ns_zero_rejected() -> None:
    with pytest.raises(ValueError, match="published_ts_ns"):
        SocialPost(
            ts_ns=10,
            source="X",
            post_id="1",
            author="u",
            body="b",
            published_ts_ns=0,
        )


def test_raw_document_minimal_valid() -> None:
    doc = RawDocument(ts_ns=1, seed_id="s", url="https://x")
    assert doc.fetched_ok is True
    assert doc.title == ""
    assert doc.body == ""


@pytest.mark.parametrize("attr", ["seed_id", "url"])
def test_raw_document_required_fields(attr: str) -> None:
    base = dict(ts_ns=1, seed_id="s", url="https://x")
    base[attr] = ""
    with pytest.raises(ValueError, match=attr):
        RawDocument(**base)


def test_filtered_item_score_bounds() -> None:
    base = dict(
        ts_ns=1,
        seed_id="s",
        url="https://x",
        title="t",
        body="b",
        score=0.5,
        reason="keyword:x",
    )
    FilteredItem(**base)  # should not raise

    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="score"):
            FilteredItem(**{**base, "score": bad})


def test_filtered_item_reason_required() -> None:
    with pytest.raises(ValueError, match="reason"):
        FilteredItem(
            ts_ns=1,
            seed_id="s",
            url="https://x",
            title="t",
            body="b",
            score=0.5,
            reason="",
        )


def test_curated_item_score_bounds() -> None:
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="score"):
            CuratedItem(
                ts_ns=1,
                seed_id="s",
                url="https://x",
                title="t",
                body="b",
                score=bad,
                seed_topic="crypto",
            )


def test_curated_item_seed_topic_required() -> None:
    with pytest.raises(ValueError, match="seed_topic"):
        CuratedItem(
            ts_ns=1,
            seed_id="s",
            url="https://x",
            title="t",
            body="b",
            score=0.5,
            seed_topic="",
        )


def test_value_types_are_frozen() -> None:
    """Frozen dataclasses cannot be mutated post-construction."""

    from dataclasses import FrozenInstanceError

    doc = RawDocument(ts_ns=1, seed_id="s", url="https://x")
    with pytest.raises(FrozenInstanceError):
        doc.url = "changed"  # type: ignore[misc]
