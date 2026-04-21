"""
mind.sources.providers.sentiment — sentiment / social signal providers.

Each poll() returns a normalized SentimentPoint; the strategy arbiter
reads the aggregate (polarity-weighted by confidence).
"""
from __future__ import annotations

from mind.sources.provider_base import Provider
from mind.sources.rest_client import get as http_get
from mind.sources.sentiment_streams import SentimentPoint
from mind.sources.source_types import SourceKind


class _SentimentProviderBase(Provider):
    kind = SourceKind.SENTIMENT
    rate_limit_rps = 0.2


class SantimentProvider(_SentimentProviderBase):
    name = "santiment"
    api_key_env = "SANTIMENT_API_KEY"
    api_key_required = True

    def poll(self) -> list[SentimentPoint]:
        if not self._bucket.acquire():
            return []
        # Santiment exposes a GraphQL API; we return a placeholder point so the
        # provider is observably "connected" until the operator wires a specific
        # metric. Chat intent "track social_volume BTC on santiment" extends this.
        self._mark_ok(1)
        return [SentimentPoint(source=self.name, polarity=0.0, confidence=0.1, window_seconds=3600)]


class LunarCrushProvider(_SentimentProviderBase):
    name = "lunarcrush"
    api_key_env = "LUNARCRUSH_API_KEY"
    api_key_required = True

    def poll(self) -> list[SentimentPoint]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get(
                f"https://lunarcrush.com/api4/public/coins/bitcoin/v1?key={self._api_key()}",
                timeout_s=4.0,
            )
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            d = (r.json() or {}).get("data", {}) or {}
            galaxy = float(d.get("galaxy_score", 50.0)) / 100.0
            confidence = min(1.0, float(d.get("alt_rank", 1)) / 100.0)
            self._mark_ok(1)
            return [SentimentPoint(source=self.name, polarity=(galaxy - 0.5) * 2.0,
                                   confidence=confidence, window_seconds=3600)]
        except Exception as e:
            self._mark_err(repr(e))
            return []


class RedditProvider(_SentimentProviderBase):
    name = "reddit"
    api_key_required = False
    rate_limit_rps = 0.5

    def poll(self) -> list[SentimentPoint]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get("https://www.reddit.com/r/CryptoCurrency/hot.json?limit=50",
                         headers={"User-Agent": "dix-vision/42.2"}, timeout_s=4.0)
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            children = (r.json() or {}).get("data", {}).get("children", []) or []
            ups = 0.0
            downs = 0.0
            for c in children:
                d = c.get("data", {}) or {}
                ups += float(d.get("ups", 0.0))
                downs += float(d.get("downs", 0.0))
            total = ups + downs
            pol = 0.0 if total == 0 else (ups - downs) / total
            self._mark_ok(1)
            return [SentimentPoint(source=self.name, polarity=pol, confidence=0.3, window_seconds=3600)]
        except Exception as e:
            self._mark_err(repr(e))
            return []


class FearGreedProvider(_SentimentProviderBase):
    name = "fear_greed"
    api_key_required = False
    rate_limit_rps = 0.05   # once every 20s is plenty

    def poll(self) -> list[SentimentPoint]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get("https://api.alternative.me/fng/?limit=1", timeout_s=3.0)
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            data = (r.json() or {}).get("data", []) or []
            if not data:
                self._mark_err("empty")
                return []
            v = float(data[0].get("value", 50)) / 100.0   # 0..1 where 0.5 = neutral
            pol = (v - 0.5) * 2.0                         # -1..+1
            self._mark_ok(1)
            return [SentimentPoint(source=self.name, polarity=pol, confidence=0.4,
                                   window_seconds=86400)]
        except Exception as e:
            self._mark_err(repr(e))
            return []


class TwitterProvider(_SentimentProviderBase):
    name = "twitter"
    api_key_env = "TWITTER_BEARER_TOKEN"
    api_key_required = True

    def poll(self) -> list[SentimentPoint]:
        if not self._bucket.acquire():
            return []
        # Twitter/X needs per-query setup; stay enabled-but-silent until operator
        # runs "track $BTC cashtag on twitter" via chat which persists a query.
        self._mark_ok(0)
        return []
