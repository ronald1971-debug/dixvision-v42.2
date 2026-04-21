"""
mind.sources.providers — concrete provider implementations.

Import ``bootstrap_all_providers()`` to auto-register every provider whose
API key is present (or which is key-free) into the relevant registry.
Providers without keys are skipped silently but remain discoverable via
``list_all_providers()`` so the cockpit can show them as disabled.
"""
from __future__ import annotations

import threading
from typing import List

from mind.sources.market_streams import get_market_streams
from mind.sources.news_streams import get_news_streams
from mind.sources.onchain_streams import get_onchain_streams
from mind.sources.provider_base import Provider, ProviderStatus
from mind.sources.providers.code_search import CODE_SEARCH_PROVIDERS

# Providers
from mind.sources.providers.market_cex import (
    BinanceProvider,
    BitfinexProvider,
    BybitProvider,
    CoinbaseProvider,
    KrakenProvider,
    KuCoinProvider,
    OKXProvider,
)
from mind.sources.providers.market_expanded import EXPANDED_MARKET_PROVIDERS
from mind.sources.providers.news import (
    CoinDeskRSSProvider,
    CointelegraphRSSProvider,
    CryptoPanicProvider,
    DecryptRSSProvider,
    MessariProvider,
    NewsAPIProvider,
    TheBlockRSSProvider,
)
from mind.sources.providers.news_expanded import EXPANDED_NEWS_PROVIDERS
from mind.sources.providers.onchain import (
    BitqueryProvider,
    DuneProvider,
    EtherscanProvider,
    EthplorerProvider,
    GlassnodeProvider,
    SolanaRPCProvider,
)
from mind.sources.providers.sentiment import (
    FearGreedProvider,
    LunarCrushProvider,
    RedditProvider,
    SantimentProvider,
    TwitterProvider,
)
from mind.sources.sentiment_streams import get_sentiment_streams

_ALL_PROVIDER_CLASSES = [
    # market / CEX
    BinanceProvider, CoinbaseProvider, KrakenProvider, BybitProvider,
    OKXProvider, KuCoinProvider, BitfinexProvider,
    # news
    CryptoPanicProvider, MessariProvider, NewsAPIProvider,
    CoinDeskRSSProvider, TheBlockRSSProvider, DecryptRSSProvider,
    CointelegraphRSSProvider,
    # sentiment
    SantimentProvider, LunarCrushProvider, RedditProvider,
    FearGreedProvider, TwitterProvider,
    # onchain
    EtherscanProvider, SolanaRPCProvider, BitqueryProvider,
    GlassnodeProvider, DuneProvider, EthplorerProvider,
    # expanded
    *EXPANDED_NEWS_PROVIDERS,
    *EXPANDED_MARKET_PROVIDERS,
    *CODE_SEARCH_PROVIDERS,
]

_lock = threading.Lock()
_bootstrapped = False
_instances: list[Provider] = []


def bootstrap_all_providers() -> list[Provider]:
    """Instantiate every provider and register the enabled ones with the
    appropriate registry. Safe to call multiple times."""
    global _bootstrapped, _instances
    with _lock:
        if _bootstrapped:
            return list(_instances)
        market = get_market_streams()
        news = get_news_streams()
        sent = get_sentiment_streams()
        chain = get_onchain_streams()
        for cls in _ALL_PROVIDER_CLASSES:
            try:
                p = cls()
            except Exception:
                continue
            _instances.append(p)
            if not p.enabled():
                continue
            try:
                if isinstance(p, (BinanceProvider, CoinbaseProvider, KrakenProvider,
                                  BybitProvider, OKXProvider, KuCoinProvider,
                                  BitfinexProvider)):
                    market.register(p.name, p.poll)
                elif isinstance(p, (CryptoPanicProvider, MessariProvider, NewsAPIProvider,
                                    CoinDeskRSSProvider, TheBlockRSSProvider,
                                    DecryptRSSProvider, CointelegraphRSSProvider)):
                    news.register(p.name, p.poll)
                elif isinstance(p, (SantimentProvider, LunarCrushProvider, RedditProvider,
                                    FearGreedProvider, TwitterProvider)):
                    sent.register(p.name, p.poll)
                elif isinstance(p, (EtherscanProvider, SolanaRPCProvider, BitqueryProvider,
                                    GlassnodeProvider, DuneProvider, EthplorerProvider)):
                    chain.register(p.name, p.poll)
            except Exception:
                continue
        _bootstrapped = True
        return list(_instances)


def list_all_providers() -> list[ProviderStatus]:
    """Return status for every provider (enabled or not)."""
    bootstrap_all_providers()
    return [p.status() for p in _instances]


def provider_summary() -> list[dict]:
    """JSON-serialisable view of every provider for the cockpit."""
    out: list[dict] = []
    for s in list_all_providers():
        out.append({
            "name": s.name,
            "kind": s.kind,
            "enabled": s.enabled,
            "has_key": s.has_key,
            "last_poll_ok": s.last_poll_ok,
            "last_poll_count": s.last_poll_count,
            "last_error": s.last_error,
        })
    return out


__all__ = ["bootstrap_all_providers", "list_all_providers", "provider_summary"]
