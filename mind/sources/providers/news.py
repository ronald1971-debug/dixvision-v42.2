"""
mind.sources.providers.news — news feed providers (JSON APIs + RSS).

RSS feeds are fetched via stdlib ``urllib`` + regex parsing to avoid
pulling in ``feedparser`` as a hard dep. JSON APIs (CryptoPanic / Messari
/ NewsAPI) require keys and stay disabled until keys are configured.
"""
from __future__ import annotations

import re

from mind.sources.news_streams import NewsItem
from mind.sources.provider_base import Provider
from mind.sources.rest_client import get as http_get
from mind.sources.source_types import SourceKind
from system.time_source import utc_now


def _now_iso() -> str:
    return utc_now().isoformat()


_RSS_ITEM_RE = re.compile(r"<item[^>]*>(.*?)</item>", re.DOTALL | re.IGNORECASE)
_RSS_TITLE_RE = re.compile(r"<title(?:[^>]*)>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</title>", re.DOTALL | re.IGNORECASE)
_ATOM_ENTRY_RE = re.compile(r"<entry[^>]*>(.*?)</entry>", re.DOTALL | re.IGNORECASE)


def _parse_rss_titles(xml: str, limit: int = 20) -> list[str]:
    out: list[str] = []
    for item in _RSS_ITEM_RE.findall(xml)[:limit]:
        m = _RSS_TITLE_RE.search(item)
        if m:
            out.append(re.sub(r"<[^>]+>", "", m.group(1)).strip())
    if not out:  # atom fallback
        for entry in _ATOM_ENTRY_RE.findall(xml)[:limit]:
            m = _RSS_TITLE_RE.search(entry)
            if m:
                out.append(re.sub(r"<[^>]+>", "", m.group(1)).strip())
    return [t for t in out if t]


class _NewsProviderBase(Provider):
    kind = SourceKind.NEWS
    rate_limit_rps = 0.5

    def _item(self, headline: str, polarity: float = 0.0, confidence: float = 0.2) -> NewsItem:
        return NewsItem(source=self.name, headline=headline[:280],
                        polarity=polarity, confidence=confidence,
                        timestamp_utc=_now_iso())


class CryptoPanicProvider(_NewsProviderBase):
    name = "cryptopanic"
    api_key_env = "CRYPTOPANIC_API_KEY"
    api_key_required = True

    def poll(self) -> list[NewsItem]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get(
                f"https://cryptopanic.com/api/v1/posts/?auth_token={self._api_key()}&kind=news&public=true",
                timeout_s=4.0,
            )
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            results = (r.json() or {}).get("results", []) or []
            out = []
            for row in results[:40]:
                title = str(row.get("title", "")).strip()
                if not title:
                    continue
                votes = row.get("votes", {}) or {}
                pos = float(votes.get("positive", 0))
                neg = float(votes.get("negative", 0))
                pol = 0.0 if (pos + neg) == 0 else (pos - neg) / (pos + neg)
                out.append(self._item(title, polarity=pol, confidence=0.3))
            self._mark_ok(len(out))
            return out
        except Exception as e:
            self._mark_err(repr(e))
            return []


class MessariProvider(_NewsProviderBase):
    name = "messari"
    api_key_env = "MESSARI_API_KEY"
    api_key_required = True

    def poll(self) -> list[NewsItem]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get("https://data.messari.io/api/v1/news",
                         headers={"x-messari-api-key": self._api_key()}, timeout_s=4.0)
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            data = (r.json() or {}).get("data", []) or []
            out = [self._item(str(row.get("title", "")).strip()) for row in data[:40]
                   if row.get("title")]
            self._mark_ok(len(out))
            return out
        except Exception as e:
            self._mark_err(repr(e))
            return []


class NewsAPIProvider(_NewsProviderBase):
    name = "newsapi"
    api_key_env = "NEWSAPI_API_KEY"
    api_key_required = True

    def poll(self) -> list[NewsItem]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get(
                f"https://newsapi.org/v2/everything?q=crypto+OR+bitcoin+OR+ethereum&sortBy=publishedAt&pageSize=40&apiKey={self._api_key()}",
                timeout_s=4.0,
            )
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            articles = (r.json() or {}).get("articles", []) or []
            out = [self._item(str(a.get("title", "")).strip()) for a in articles if a.get("title")]
            self._mark_ok(len(out))
            return out
        except Exception as e:
            self._mark_err(repr(e))
            return []


class _RSSProvider(_NewsProviderBase):
    api_key_required = False
    rss_url = ""

    def poll(self) -> list[NewsItem]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get(self.rss_url, timeout_s=4.0)
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            titles = _parse_rss_titles(r.text)
            out = [self._item(t) for t in titles]
            self._mark_ok(len(out))
            return out
        except Exception as e:
            self._mark_err(repr(e))
            return []


class CoinDeskRSSProvider(_RSSProvider):
    name = "coindesk_rss"
    rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"


class TheBlockRSSProvider(_RSSProvider):
    name = "theblock_rss"
    rss_url = "https://www.theblock.co/rss.xml"


class DecryptRSSProvider(_RSSProvider):
    name = "decrypt_rss"
    rss_url = "https://decrypt.co/feed"


class CointelegraphRSSProvider(_RSSProvider):
    name = "cointelegraph_rss"
    rss_url = "https://cointelegraph.com/rss"
