"""
mind.sources.providers.news_expanded \u2014 ~25 additional news providers.

All follow the same contract as news.py: subclass Provider, declare
rate_limit_rps / api_key_env / api_key_required, implement poll().
Missing API key = silently disabled. Never throws into the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass

from mind.sources.provider_base import Provider
from mind.sources.source_types import SourceKind


@dataclass(frozen=True)
class NewsSpec:
    name: str
    url: str
    key_env: str = ""
    key_required: bool = False
    rps: float = 0.2
    kind: SourceKind = SourceKind.NEWS


_SPECS: tuple[NewsSpec, ...] = (
    # General financial
    NewsSpec("bloomberg_rss", "https://feeds.bloomberg.com/markets/news.rss"),
    NewsSpec("reuters_rss", "https://www.reutersagency.com/feed/?best-topics=business-finance"),
    NewsSpec("wsj_rss", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    NewsSpec("ft_rss", "https://www.ft.com/markets?format=rss"),
    NewsSpec("cnbc_rss", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    NewsSpec("marketwatch_rss", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    NewsSpec("yahoo_finance_rss", "https://finance.yahoo.com/news/rssindex"),
    NewsSpec("investing_rss", "https://www.investing.com/rss/news.rss"),
    NewsSpec("business_insider_rss", "https://markets.businessinsider.com/rss/news"),
    NewsSpec("seeking_alpha_rss", "https://seekingalpha.com/market_currents.xml"),
    NewsSpec("benzinga_rss", "https://www.benzinga.com/feed"),
    # Macro / Fed
    NewsSpec("fred_api", "https://api.stlouisfed.org/fred/series/observations",
             key_env="FRED_API_KEY", key_required=True, rps=1.0, kind=SourceKind.REST),
    NewsSpec("treasury_xml", "https://home.treasury.gov/resource-center/data-chart-center/"
                               "interest-rates/daily-treasury-rates.csv"),
    NewsSpec("bls_api", "https://api.bls.gov/publicAPI/v2/timeseries/data/",
             key_env="BLS_API_KEY", rps=0.5, kind=SourceKind.REST),
    NewsSpec("bea_api", "https://apps.bea.gov/api/data/",
             key_env="BEA_API_KEY", key_required=True, rps=0.5, kind=SourceKind.REST),
    NewsSpec("fomc_rss", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    # Crypto-native
    NewsSpec("bitcoinmagazine_rss", "https://bitcoinmagazine.com/feed"),
    NewsSpec("coinjournal_rss", "https://coinjournal.net/feed/"),
    NewsSpec("dlnews_rss", "https://www.dlnews.com/arc/outboundfeeds/rss/"),
    NewsSpec("blockworks_rss", "https://blockworks.co/feed"),
    NewsSpec("messari_research_rss", "https://messari.io/rss"),
    NewsSpec("glassnode_insights_rss", "https://insights.glassnode.com/rss/"),
    # Alt-data
    NewsSpec("gdelt_api", "https://api.gdeltproject.org/api/v2/doc/doc",
             rps=0.5, kind=SourceKind.REST),
    NewsSpec("reddit_wsb", "https://www.reddit.com/r/wallstreetbets/new.json",
             rps=0.25, kind=SourceKind.REST),
    NewsSpec("reddit_investing", "https://www.reddit.com/r/investing/new.json",
             rps=0.25, kind=SourceKind.REST),
    NewsSpec("reddit_cryptocurrency", "https://www.reddit.com/r/cryptocurrency/new.json",
             rps=0.25, kind=SourceKind.REST),
    NewsSpec("stocktwits_trending", "https://api.stocktwits.com/api/2/streams/trending.json",
             rps=0.25, kind=SourceKind.REST),
    NewsSpec("hn_frontpage", "https://hacker-news.firebaseio.com/v0/topstories.json",
             rps=0.5, kind=SourceKind.REST),
)


def _make_class(spec: NewsSpec) -> type:
    cls_name = "".join(p.capitalize() for p in spec.name.split("_")) + "Provider"
    attrs: dict[str, object] = {
        "name": spec.name,
        "kind": spec.kind,
        "api_key_env": spec.key_env,
        "api_key_required": spec.key_required,
        "rate_limit_rps": spec.rps,
        "rate_limit_burst": max(2.0, spec.rps * 4),
        "homepage": spec.url,
        "poll": lambda self: [],                                                 # real poll wired later
    }
    return type(cls_name, (Provider,), attrs)


EXPANDED_NEWS_PROVIDERS: list[type] = [_make_class(s) for s in _SPECS]
__all__ = ["EXPANDED_NEWS_PROVIDERS"]
