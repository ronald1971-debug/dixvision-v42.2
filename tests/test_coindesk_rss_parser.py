"""Tests for the pure CoinDesk RSS parser layer.

These tests exercise :func:`ui.feeds.coindesk_rss.parse_rss_feed` and
:class:`core.contracts.news.NewsItem` validation. No I/O, no clock, no
network — the parser is pure and the caller supplies ``ts_ns``.
"""

from __future__ import annotations

import pytest

from core.contracts.news import NewsItem
from ui.feeds.coindesk_rss import (
    SOURCE_TAG,
    make_coindesk_rss_url,
    parse_rss_feed,
)

# ---------------------------------------------------------------------------
# Fixture documents
# ---------------------------------------------------------------------------

_RSS_TWO_ITEMS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>CoinDesk</title>
    <item>
      <title>BTC tops $200k after ETF inflows</title>
      <link>https://example.invalid/news/btc-200k</link>
      <guid isPermaLink="false">guid-001</guid>
      <description>Spot ETFs absorbed a record $4.1bn in the last week.</description>
      <pubDate>Sat, 19 Apr 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <title>SEC clarifies staking guidance</title>
      <link>https://example.invalid/news/sec-staking</link>
      <description><![CDATA[<p>Statement issued <em>today</em>.</p>]]></description>
      <pubDate>Sun, 20 Apr 2026 14:30:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

_RSS_MALFORMED = b"<rss><channel><not-closed>"
_RSS_EMPTY = b""
_RSS_NO_ITEMS = (
    b"<rss version='2.0'><channel><title>Empty</title></channel></rss>"
)
_RSS_BLANK_TITLE = b"""<?xml version="1.0"?>
<rss><channel>
  <item><title></title><link>https://example.invalid/x</link></item>
  <item><title>Real headline</title><link>https://example.invalid/y</link></item>
</channel></rss>
"""
_RSS_NO_GUID_NO_LINK = b"""<?xml version="1.0"?>
<rss><channel>
  <item><title>orphan</title></item>
  <item><title>has-link</title><link>https://example.invalid/ok</link></item>
</channel></rss>
"""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parse_rss_feed_emits_items_in_order() -> None:
    items = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=42)
    assert len(items) == 2
    first, second = items
    assert first.title == "BTC tops $200k after ETF inflows"
    assert first.guid == "guid-001"
    assert first.url == "https://example.invalid/news/btc-200k"
    assert first.summary.startswith("Spot ETFs")
    assert second.title == "SEC clarifies staking guidance"
    # No <guid> on the second item — link substitutes for it.
    assert second.guid == "https://example.invalid/news/sec-staking"


def test_parse_rss_feed_strips_html_from_summary() -> None:
    items = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=1)
    assert items[1].summary == "Statement issued today ."


def test_parse_rss_feed_uses_caller_ts_ns_for_every_item() -> None:
    """INV-15: the parser must never read its own clock."""
    items = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=12345)
    assert all(item.ts_ns == 12345 for item in items)


def test_parse_rss_feed_uses_caller_source_tag() -> None:
    items = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=1, source="DESK-A")
    assert all(item.source == "DESK-A" for item in items)


def test_parse_rss_feed_default_source_is_coindesk() -> None:
    assert SOURCE_TAG == "COINDESK"
    items = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=1)
    assert items[0].source == "COINDESK"


def test_parse_rss_feed_parses_pub_date_to_ns() -> None:
    items = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=1)
    assert items[0].published_ts_ns is not None
    # Sanity check: the second item is later than the first.
    assert items[1].published_ts_ns is not None
    assert items[1].published_ts_ns > items[0].published_ts_ns


def test_parse_rss_feed_accepts_str_payload() -> None:
    items = parse_rss_feed(_RSS_TWO_ITEMS.decode("utf-8"), ts_ns=1)
    assert len(items) == 2


# ---------------------------------------------------------------------------
# Tolerance / boundary cases — parser must never raise
# ---------------------------------------------------------------------------


def test_parse_rss_feed_returns_empty_on_malformed_xml() -> None:
    assert parse_rss_feed(_RSS_MALFORMED, ts_ns=1) == ()


def test_parse_rss_feed_returns_empty_on_empty_payload() -> None:
    assert parse_rss_feed(_RSS_EMPTY, ts_ns=1) == ()


def test_parse_rss_feed_returns_empty_when_no_items() -> None:
    assert parse_rss_feed(_RSS_NO_ITEMS, ts_ns=1) == ()


def test_parse_rss_feed_skips_items_with_blank_title() -> None:
    items = parse_rss_feed(_RSS_BLANK_TITLE, ts_ns=1)
    assert len(items) == 1
    assert items[0].title == "Real headline"


def test_parse_rss_feed_skips_items_with_no_guid_and_no_link() -> None:
    items = parse_rss_feed(_RSS_NO_GUID_NO_LINK, ts_ns=1)
    assert len(items) == 1
    assert items[0].title == "has-link"


def test_parse_rss_feed_returns_tuple_not_list() -> None:
    """Frozen-output contract: callers may not mutate parser results."""
    out = parse_rss_feed(_RSS_TWO_ITEMS, ts_ns=1)
    assert isinstance(out, tuple)


def test_parse_rss_feed_with_unparseable_pubdate_returns_none() -> None:
    payload = b"""<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>weird date</title>
    <link>https://example.invalid/x</link>
    <pubDate>not a date</pubDate>
  </item>
</channel></rss>
"""
    items = parse_rss_feed(payload, ts_ns=1)
    assert len(items) == 1
    assert items[0].published_ts_ns is None


# ---------------------------------------------------------------------------
# NewsItem validation
# ---------------------------------------------------------------------------


def test_news_item_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        NewsItem(ts_ns=1, source="", guid="g", title="t")


def test_news_item_rejects_empty_guid() -> None:
    with pytest.raises(ValueError, match="guid"):
        NewsItem(ts_ns=1, source="X", guid="", title="t")


def test_news_item_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        NewsItem(ts_ns=1, source="X", guid="g", title="")


def test_news_item_rejects_negative_published_ts() -> None:
    with pytest.raises(ValueError, match="published_ts_ns"):
        NewsItem(
            ts_ns=1, source="X", guid="g", title="t", published_ts_ns=-1
        )


def test_news_item_allows_none_published_ts() -> None:
    item = NewsItem(
        ts_ns=1, source="X", guid="g", title="t", published_ts_ns=None
    )
    assert item.published_ts_ns is None


def test_news_item_is_frozen() -> None:
    item = NewsItem(ts_ns=1, source="X", guid="g", title="t")
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass error
        item.title = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# URL accessor
# ---------------------------------------------------------------------------


def test_make_coindesk_rss_url_returns_canonical_endpoint() -> None:
    assert make_coindesk_rss_url() == (
        "https://www.coindesk.com/arc/outboundfeeds/rss/"
    )
