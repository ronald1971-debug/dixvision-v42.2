"""Memecoin launch + on-chain pool snapshot contracts.

These value objects feed the cockpit's memecoin tier (LaunchFirehose,
DevDumpWatchdog, BundleDetector, …). They are *inputs* to the
intelligence engine, not canonical bus events — the four canonical
cross-engine events live in ``core/contracts/events.py``.

INV-08 (typed cross-domain): consumers depend on these dataclasses,
not on the raw WS frames.
INV-15 (replay determinism): both types are frozen + slotted so they
are immutable and hashable; replay produces byte-identical state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LaunchEvent:
    """One memecoin launch announcement (e.g. a new Pump.fun mint).

    Attributes:
        ts_ns: Monotonic timestamp in nanoseconds (TimeAuthority).
        chain: ``"solana"``, ``"base"``, ``"ethereum"``, …
        venue: Origin tag (``"PUMPFUN"``, ``"RAYDIUM"``, …).
        mint: Token mint / contract address.
        symbol: Ticker (``"DOGE"`` etc.). Empty if the venue did not
            publish one in the launch frame.
        name: Human-readable token name. Empty if missing.
        creator: Wallet that minted the token (Solana base58 / EVM hex).
            Empty if the venue did not publish it.
        market_cap_usd: Initial market cap in USD if the venue
            published one; ``0.0`` otherwise.
        liquidity_usd: Initial liquidity in USD if known.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    chain: str
    venue: str
    mint: str
    symbol: str = ""
    name: str = ""
    creator: str = ""
    market_cap_usd: float = 0.0
    liquidity_usd: float = 0.0
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    """One AMM pool snapshot (Raydium / Uniswap / Jupiter / …).

    Attributes:
        ts_ns: Monotonic timestamp in nanoseconds (TimeAuthority).
        chain: ``"solana"``, ``"ethereum"``, ``"base"``, …
        venue: Origin tag (``"RAYDIUM"``, ``"UNISWAP_V3"``, …).
        pool_id: Pool address / ID on its chain.
        base_mint: Base token mint / contract.
        quote_mint: Quote token mint / contract.
        base_symbol: Ticker if known. Empty otherwise.
        quote_symbol: Ticker if known. Empty otherwise.
        price: Current ratio quote-per-base (``0.0`` if unknown).
        liquidity_usd: TVL in USD (``0.0`` if unknown).
        volume_24h_usd: 24h trade volume USD (``0.0`` if unknown).
        meta: Free-form structural metadata.
    """

    ts_ns: int
    chain: str
    venue: str
    pool_id: str
    base_mint: str
    quote_mint: str
    base_symbol: str = ""
    quote_symbol: str = ""
    price: float = 0.0
    liquidity_usd: float = 0.0
    volume_24h_usd: float = 0.0
    meta: Mapping[str, str] = field(default_factory=dict)


__all__ = ["LaunchEvent", "PoolSnapshot"]
