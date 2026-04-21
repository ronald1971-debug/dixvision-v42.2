"""
mind.sources.providers.onchain — on-chain data providers.

Key-free:
    SolanaRPCProvider  (public RPC)
    EthplorerProvider  (freekey tier)
Keyed:
    EtherscanProvider
    BitqueryProvider
    GlassnodeProvider
    DuneProvider
"""
from __future__ import annotations

from mind.sources.onchain_streams import OnchainEvent
from mind.sources.provider_base import Provider
from mind.sources.rest_client import get as http_get
from mind.sources.source_types import SourceKind


class _OnchainProviderBase(Provider):
    kind = SourceKind.ONCHAIN
    rate_limit_rps = 0.5


class EtherscanProvider(_OnchainProviderBase):
    name = "etherscan"
    api_key_env = "ETHERSCAN_API_KEY"
    api_key_required = True

    def poll(self) -> list[OnchainEvent]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get(
                f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={self._api_key()}",
                timeout_s=3.0,
            )
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            d = r.json() or {}
            block_hex = str(d.get("result", "0x0"))
            try:
                block = int(block_hex, 16)
            except ValueError:
                block = 0
            self._mark_ok(1)
            return [OnchainEvent(chain="ethereum", kind="block", payload={"number": block})]
        except Exception as e:
            self._mark_err(repr(e))
            return []


class SolanaRPCProvider(_OnchainProviderBase):
    name = "solana_rpc"
    api_key_env = "SOLANA_RPC_URL"
    api_key_required = False    # falls back to public mainnet RPC
    rate_limit_rps = 2.0

    def _endpoint(self) -> str:
        return self._api_key() or "https://api.mainnet-beta.solana.com"

    def poll(self) -> list[OnchainEvent]:
        if not self._bucket.acquire():
            return []
        # Public RPC requires POST; the stdlib rest_client only handles GET,
        # so we keep this as a soft-ping for now (returns a synthetic event).
        # Real block subscription is added via the chat interface later.
        self._mark_ok(1)
        return [OnchainEvent(chain="solana", kind="endpoint_ping",
                             payload={"endpoint": self._endpoint()})]


class BitqueryProvider(_OnchainProviderBase):
    name = "bitquery"
    api_key_env = "BITQUERY_API_KEY"
    api_key_required = True

    def poll(self) -> list[OnchainEvent]:
        # Bitquery is GraphQL only; stays enabled-but-silent until a query is
        # configured via chat ("run bitquery Q on ethereum daily swaps").
        self._bucket.acquire()
        self._mark_ok(0)
        return []


class GlassnodeProvider(_OnchainProviderBase):
    name = "glassnode"
    api_key_env = "GLASSNODE_API_KEY"
    api_key_required = True

    def poll(self) -> list[OnchainEvent]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get(
                f"https://api.glassnode.com/v1/metrics/addresses/active_count?a=BTC&api_key={self._api_key()}",
                timeout_s=4.0,
            )
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            series = r.json() or []
            latest = series[-1] if series else {}
            self._mark_ok(1)
            return [OnchainEvent(chain="bitcoin", kind="metric",
                                 payload={"active_addresses": latest})]
        except Exception as e:
            self._mark_err(repr(e))
            return []


class DuneProvider(_OnchainProviderBase):
    name = "dune"
    api_key_env = "DUNE_API_KEY"
    api_key_required = True

    def poll(self) -> list[OnchainEvent]:
        self._bucket.acquire()
        # Dune needs a saved-query ID; stays enabled-but-silent until chat
        # intent "run dune query <id>" configures a poll.
        self._mark_ok(0)
        return []


class EthplorerProvider(_OnchainProviderBase):
    name = "ethplorer"
    api_key_required = False
    rate_limit_rps = 0.1

    def poll(self) -> list[OnchainEvent]:
        if not self._bucket.acquire():
            return []
        try:
            r = http_get("https://api.ethplorer.io/getTop?apiKey=freekey&criteria=trade&limit=10",
                         timeout_s=4.0)
            if r.status != 200:
                self._mark_err(f"http={r.status}")
                return []
            tokens = (r.json() or {}).get("tokens", []) or []
            self._mark_ok(len(tokens))
            return [OnchainEvent(chain="ethereum", kind="top_traded",
                                 payload={"tokens": tokens})]
        except Exception as e:
            self._mark_err(repr(e))
            return []
