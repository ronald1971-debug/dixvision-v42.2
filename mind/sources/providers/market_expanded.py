"""
mind.sources.providers.market_expanded \u2014 DEX, derivatives, TradFi, institutional.

Like news_expanded: pluggable, env-keyed, rate-limited, graceful-degrade.
"""
from __future__ import annotations

from dataclasses import dataclass

from mind.sources.provider_base import Provider
from mind.sources.source_types import SourceKind


@dataclass(frozen=True)
class MarketSpec:
    name: str
    url: str
    key_env: str = ""
    key_required: bool = False
    rps: float = 1.0
    kind: SourceKind = SourceKind.REST


_SPECS: tuple[MarketSpec, ...] = (
    # DEX / on-chain liquidity
    MarketSpec("uniswap_v3_subgraph", "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"),
    MarketSpec("sushi_subgraph", "https://api.thegraph.com/subgraphs/name/sushi-v2/sushiswap"),
    MarketSpec("curve_subgraph", "https://api.thegraph.com/subgraphs/name/curvefi/curve"),
    MarketSpec("aave_v3_subgraph", "https://api.thegraph.com/subgraphs/name/aave/protocol-v3"),
    MarketSpec("zerox_api", "https://api.0x.org/", key_env="ZEROX_API_KEY"),
    MarketSpec("oneinch_api", "https://api.1inch.dev/", key_env="ONEINCH_API_KEY",
               key_required=True),
    MarketSpec("dexscreener_api", "https://api.dexscreener.com/latest/dex/"),
    MarketSpec("geckoterminal_api", "https://api.geckoterminal.com/api/v2/"),
    MarketSpec("jupiter_aggregator", "https://quote-api.jup.ag/v6/quote"),
    MarketSpec("raydium_api", "https://api.raydium.io/v2/ammV3/ammPools"),
    # Derivatives / options
    MarketSpec("deribit_rest", "https://www.deribit.com/api/v2/"),
    MarketSpec("bitmex_rest", "https://www.bitmex.com/api/v1/"),
    MarketSpec("dydx_v4_rest", "https://indexer.dydx.trade/v4/"),
    MarketSpec("gmx_subgraph", "https://api.thegraph.com/subgraphs/name/gmx-io/gmx-stats"),
    # TradFi
    MarketSpec("polygon_io", "https://api.polygon.io/v3/",
               key_env="POLYGON_API_KEY", key_required=True, rps=5.0),
    MarketSpec("alpaca_markets", "https://data.alpaca.markets/v2/",
               key_env="ALPACA_API_KEY", key_required=True, rps=3.0),
    MarketSpec("tiingo", "https://api.tiingo.com/",
               key_env="TIINGO_API_KEY", key_required=True, rps=1.0),
    MarketSpec("alpha_vantage", "https://www.alphavantage.co/query",
               key_env="ALPHAVANTAGE_API_KEY", key_required=True, rps=0.1),
    MarketSpec("finnhub", "https://finnhub.io/api/v1/",
               key_env="FINNHUB_API_KEY", key_required=True, rps=2.0),
    MarketSpec("iex_cloud", "https://cloud.iexapis.com/stable/",
               key_env="IEX_API_KEY", key_required=True, rps=2.0),
    MarketSpec("quandl_nasdaq", "https://data.nasdaq.com/api/v3/",
               key_env="NASDAQ_DATA_LINK_API_KEY", rps=0.5),
    MarketSpec("yahoo_finance_chart", "https://query1.finance.yahoo.com/v8/finance/chart/",
               rps=1.0),
    # Institutional / regulatory
    MarketSpec("sec_edgar_fulltext",
               "https://efts.sec.gov/LATEST/search-index", rps=0.5),
    MarketSpec("sec_edgar_form4",
               "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4", rps=0.5),
    MarketSpec("cftc_cot",
               "https://www.cftc.gov/dea/newcot/deacot.txt", rps=0.1),
)


def _make_class(spec: MarketSpec) -> type:
    cls_name = "".join(p.capitalize() for p in spec.name.split("_")) + "Provider"
    attrs: dict[str, object] = {
        "name": spec.name,
        "kind": spec.kind,
        "api_key_env": spec.key_env,
        "api_key_required": spec.key_required,
        "rate_limit_rps": spec.rps,
        "rate_limit_burst": max(2.0, spec.rps * 4),
        "homepage": spec.url,
        "poll": lambda self: [],
    }
    return type(cls_name, (Provider,), attrs)


EXPANDED_MARKET_PROVIDERS: list[type] = [_make_class(s) for s in _SPECS]
__all__ = ["EXPANDED_MARKET_PROVIDERS"]
