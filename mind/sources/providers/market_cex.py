"""
mind.sources.providers.market_cex — public CEX price/ticker providers.

All implementations use the REST ticker endpoint (no key required for
public price data). Streaming WS hooks live on the individual exchange
adapters in ``execution/adapters/``; this module only provides pull-based
market-data for the INDIRA knowledge layer.

Providers auto-register a default symbol set; additions are made via the
chat interface (DYON: "track XYZ on Kraken") which persists the change.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mind.sources.provider_base import Provider
from mind.sources.rest_client import get as http_get
from mind.sources.source_types import MarketTick, SourceKind


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _CEXProviderBase(Provider):
    kind = SourceKind.MARKET
    symbols: list[str] = []

    def _tick(self, asset: str, price: float, bid: float = 0.0, ask: float = 0.0,
              extra: dict[str, Any] | None = None) -> MarketTick:
        return MarketTick(
            source=self.name, asset=asset, price=price,
            bid=bid, ask=ask, timestamp_utc=_now_iso(),
            extra=extra or {},
        )


class BinanceProvider(_CEXProviderBase):
    name = "binance"
    api_key_env = "BINANCE_API_KEY"
    api_key_required = False   # public tickers work without a key
    rate_limit_rps = 5.0
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            r = http_get("https://api.binance.com/api/v3/ticker/bookTicker", timeout_s=3.0)
            if r.status == 200:
                data = r.json() or []
                want = set(self.symbols)
                for row in data:
                    sym = str(row.get("symbol", ""))
                    if sym not in want:
                        continue
                    bid = float(row.get("bidPrice", 0.0))
                    ask = float(row.get("askPrice", 0.0))
                    mid = (bid + ask) / 2.0 if (bid and ask) else bid or ask
                    out.append(self._tick(sym, mid, bid, ask))
                self._mark_ok(len(out))
            else:
                self._mark_err(f"http={r.status}")
        except Exception as e:
            self._mark_err(repr(e))
        return out


class CoinbaseProvider(_CEXProviderBase):
    name = "coinbase"
    api_key_env = "COINBASE_API_KEY"
    api_key_required = False
    rate_limit_rps = 4.0
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            for sym in self.symbols:
                r = http_get(f"https://api.exchange.coinbase.com/products/{sym}/ticker", timeout_s=3.0)
                if r.status != 200:
                    continue
                d = r.json() or {}
                price = float(d.get("price", 0.0))
                bid = float(d.get("bid", 0.0))
                ask = float(d.get("ask", 0.0))
                out.append(self._tick(sym, price, bid, ask))
            self._mark_ok(len(out))
        except Exception as e:
            self._mark_err(repr(e))
        return out


class KrakenProvider(_CEXProviderBase):
    name = "kraken"
    api_key_env = "KRAKEN_API_KEY"
    api_key_required = False
    rate_limit_rps = 1.0
    symbols = ["XBTUSD", "ETHUSD", "SOLUSD"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            pair = ",".join(self.symbols)
            r = http_get(f"https://api.kraken.com/0/public/Ticker?pair={pair}", timeout_s=3.0)
            if r.status == 200:
                d = (r.json() or {}).get("result", {}) or {}
                for sym, v in d.items():
                    c = v.get("c", ["0"])
                    b = v.get("b", ["0"])
                    a = v.get("a", ["0"])
                    out.append(self._tick(sym, float(c[0]), float(b[0]), float(a[0])))
                self._mark_ok(len(out))
            else:
                self._mark_err(f"http={r.status}")
        except Exception as e:
            self._mark_err(repr(e))
        return out


class BybitProvider(_CEXProviderBase):
    name = "bybit"
    api_key_env = "BYBIT_API_KEY"
    api_key_required = False
    rate_limit_rps = 5.0
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            r = http_get("https://api.bybit.com/v5/market/tickers?category=spot", timeout_s=3.0)
            if r.status == 200:
                data = (r.json() or {}).get("result", {}).get("list", []) or []
                want = set(self.symbols)
                for row in data:
                    sym = str(row.get("symbol", ""))
                    if sym not in want:
                        continue
                    bid = float(row.get("bid1Price", 0.0))
                    ask = float(row.get("ask1Price", 0.0))
                    last = float(row.get("lastPrice", 0.0))
                    out.append(self._tick(sym, last, bid, ask))
                self._mark_ok(len(out))
            else:
                self._mark_err(f"http={r.status}")
        except Exception as e:
            self._mark_err(repr(e))
        return out


class OKXProvider(_CEXProviderBase):
    name = "okx"
    api_key_env = "OKX_API_KEY"
    api_key_required = False
    rate_limit_rps = 5.0
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            r = http_get("https://www.okx.com/api/v5/market/tickers?instType=SPOT", timeout_s=3.0)
            if r.status == 200:
                data = (r.json() or {}).get("data", []) or []
                want = set(self.symbols)
                for row in data:
                    sym = str(row.get("instId", ""))
                    if sym not in want:
                        continue
                    bid = float(row.get("bidPx", 0.0) or 0.0)
                    ask = float(row.get("askPx", 0.0) or 0.0)
                    last = float(row.get("last", 0.0) or 0.0)
                    out.append(self._tick(sym, last, bid, ask))
                self._mark_ok(len(out))
            else:
                self._mark_err(f"http={r.status}")
        except Exception as e:
            self._mark_err(repr(e))
        return out


class KuCoinProvider(_CEXProviderBase):
    name = "kucoin"
    api_key_env = "KUCOIN_API_KEY"
    api_key_required = False
    rate_limit_rps = 5.0
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            r = http_get("https://api.kucoin.com/api/v1/market/allTickers", timeout_s=3.0)
            if r.status == 200:
                data = (r.json() or {}).get("data", {}).get("ticker", []) or []
                want = set(self.symbols)
                for row in data:
                    sym = str(row.get("symbol", ""))
                    if sym not in want:
                        continue
                    bid = float(row.get("buy", 0.0) or 0.0)
                    ask = float(row.get("sell", 0.0) or 0.0)
                    last = float(row.get("last", 0.0) or 0.0)
                    out.append(self._tick(sym, last, bid, ask))
                self._mark_ok(len(out))
            else:
                self._mark_err(f"http={r.status}")
        except Exception as e:
            self._mark_err(repr(e))
        return out


class BitfinexProvider(_CEXProviderBase):
    name = "bitfinex"
    api_key_env = "BITFINEX_API_KEY"
    api_key_required = False
    rate_limit_rps = 2.0
    symbols = ["tBTCUSD", "tETHUSD", "tSOLUSD"]

    def poll(self) -> list[MarketTick]:
        if not self._bucket.acquire():
            return []
        out: list[MarketTick] = []
        try:
            syms = ",".join(self.symbols)
            r = http_get(f"https://api-pub.bitfinex.com/v2/tickers?symbols={syms}", timeout_s=3.0)
            if r.status == 200:
                data = r.json() or []
                # Ticker format: [SYMBOL, BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, ..., LAST_PRICE, ...]
                for row in data:
                    if not isinstance(row, list) or len(row) < 8:
                        continue
                    sym = str(row[0])
                    bid = float(row[1] or 0.0)
                    ask = float(row[3] or 0.0)
                    last = float(row[7] or 0.0)
                    out.append(self._tick(sym, last, bid, ask))
                self._mark_ok(len(out))
            else:
                self._mark_err(f"http={r.status}")
        except Exception as e:
            self._mark_err(repr(e))
        return out
