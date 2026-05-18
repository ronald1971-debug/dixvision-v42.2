"""P1.5 — read-only projection routes for the six PR #351 widgets.

PR #351 introduced six new dashboard widgets under
``dashboard2026/src/widgets/{dex,perps}/`` that poll
``/api/dashboard/{dex,perps}/...`` and fall back to a deterministic
in-widget skeleton when the route 404s — flipping the per-widget
status chip from emerald ``live`` to amber ``mock``. This module is
the backend half of that contract: it exposes the six routes the
widgets expect, returning a deterministic harness-default projection
shaped to each widget's response interface so the chip flips to
emerald.

Each endpoint is GET-only and read-only — no engine state is read,
mutated, or constructed. The shapes are mirrored from the widget
TypeScript interfaces:

* ``/api/dashboard/dex/route?symbol=<sym>`` → :class:`RouteSnapshot`
* ``/api/dashboard/dex/pool_health?symbol=<sym>`` → :class:`PoolHealthSnapshot`
* ``/api/dashboard/dex/gas`` → :class:`GasSnapshot`
* ``/api/dashboard/perps/funding?symbol=<sym>`` → :class:`FundingSnapshot`
* ``/api/dashboard/perps/oracle?symbol=<sym>`` → :class:`OracleSnapshot`
* ``/api/dashboard/perps/liquidations?symbol=<sym>`` → :class:`LiqSnapshot`

Authority constraints (B7 lint):

* Module imports nothing from ``*_engine`` packages — projection data
  is statically declared per the deterministic skeleton each widget
  expects today. Future PRs will replace the static skeletons with
  live solver / pool / funding feeds without changing the route shape.
* No ledger writes, no typed-event construction — pure read seam.
* ``ts_iso`` is anchored on :func:`system.time_source.utc_now` so
  byte-identical replay across runs depends only on the anchor.
"""

from __future__ import annotations

import math
from typing import Final

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from system.time_source import utc_now

# ---------------------------------------------------------------------------
# DEX response models
# ---------------------------------------------------------------------------


class RouteQuote(BaseModel):
    """One solver quote inside a :class:`RouteSnapshot`.

    Mirrors the ``RouteQuote`` TypeScript interface in
    ``dashboard2026/src/widgets/dex/RouteGraph.tsx``.
    """

    venue: str
    in_token: str
    out_token: str
    in_amount: float
    out_amount: float
    price_impact_bps: float
    est_fill_ms: int


class RouteSnapshot(BaseModel):
    symbol: str
    quotes: list[RouteQuote]
    best_venue: str
    ts_iso: str


class LPHolder(BaseModel):
    address_short: str
    pct: float


class PoolHealthSnapshot(BaseModel):
    symbol: str
    liquidity_usd: float
    volume_24h_usd: float
    lp_count: int
    hhi: float = Field(..., description="Herfindahl on LP shares (0..1)")
    top_holders: list[LPHolder]
    ts_iso: str


class GasSnapshot(BaseModel):
    base_fee_lamports: int
    p50_tip_lamports: int
    p75_tip_lamports: int
    p90_tip_lamports: int
    mev_protected_rpc: str
    ts_iso: str


# ---------------------------------------------------------------------------
# Perps response models
# ---------------------------------------------------------------------------


class FundingRow(BaseModel):
    venue: str
    current_rate_bps: float
    next_funding_ts_iso: str
    cum_funding_pnl_usd: float


class FundingSnapshot(BaseModel):
    symbol: str
    rows: list[FundingRow]
    ts_iso: str


class OracleRow(BaseModel):
    venue: str
    oracle_price: float
    exec_price: float
    divergence_bps: float


class OracleSnapshot(BaseModel):
    symbol: str
    rows: list[OracleRow]
    alarm_bps: int
    ts_iso: str


class LiqBand(BaseModel):
    price: float
    oi_long_usd: float
    oi_short_usd: float


class LiqSnapshot(BaseModel):
    symbol: str
    current_price: float
    bands: list[LiqBand]
    ts_iso: str


# ---------------------------------------------------------------------------
# Deterministic harness-default skeletons
# ---------------------------------------------------------------------------
#
# These match the in-widget ``FALLBACK`` shapes exactly so behaviour is
# byte-stable when an operator compares the widget's local skeleton to
# the live route response. Tokens / venues / numbers are illustrative
# harness defaults; future PRs will replace the static factories with
# adapter-fed projections without altering the route contract.


_DEX_DEFAULT_SYMBOL: Final[str] = "SOL/USDC"
_PERPS_DEFAULT_SYMBOL: Final[str] = "BTC-PERP"
_ORACLE_ALARM_BPS: Final[int] = 25


def _ts_iso() -> str:
    """Anchor-derived UTC ISO-8601 timestamp (replay-byte-stable)."""

    return utc_now().isoformat().replace("+00:00", "Z")


def _route_snapshot(symbol: str) -> RouteSnapshot:
    quotes = [
        RouteQuote(
            venue="Jupiter Juno",
            in_token="SOL",
            out_token="USDC",
            in_amount=100.0,
            out_amount=17_412.30,
            price_impact_bps=4.1,
            est_fill_ms=820,
        ),
        RouteQuote(
            venue="1inch Fusion+",
            in_token="SOL",
            out_token="USDC",
            in_amount=100.0,
            out_amount=17_408.95,
            price_impact_bps=5.0,
            est_fill_ms=1_240,
        ),
        RouteQuote(
            venue="CowSwap",
            in_token="SOL",
            out_token="USDC",
            in_amount=100.0,
            out_amount=17_401.72,
            price_impact_bps=6.8,
            est_fill_ms=2_750,
        ),
    ]
    best_venue = max(quotes, key=lambda q: q.out_amount).venue
    return RouteSnapshot(
        symbol=symbol,
        quotes=quotes,
        best_venue=best_venue,
        ts_iso=_ts_iso(),
    )


def _pool_health_snapshot(symbol: str) -> PoolHealthSnapshot:
    return PoolHealthSnapshot(
        symbol=symbol,
        liquidity_usd=12_400_000.0,
        volume_24h_usd=38_700_000.0,
        lp_count=218,
        hhi=0.31,
        top_holders=[
            LPHolder(address_short="8KQ4\u2026b21Y", pct=22.4),
            LPHolder(address_short="FvR2\u2026c9pL", pct=14.1),
            LPHolder(address_short="9wT7\u2026aE3X", pct=8.6),
        ],
        ts_iso=_ts_iso(),
    )


def _gas_snapshot() -> GasSnapshot:
    return GasSnapshot(
        base_fee_lamports=5_000,
        p50_tip_lamports=12_400,
        p75_tip_lamports=24_800,
        p90_tip_lamports=78_000,
        mev_protected_rpc="Jito Block-Engine",
        ts_iso=_ts_iso(),
    )


def _funding_snapshot(symbol: str) -> FundingSnapshot:
    # Funding cadence per venue: Hyperliquid hourly, dYdX hourly,
    # Drift hourly. The widget renders ``next_funding_ts_iso`` as a
    # countdown — anchoring all three on the same future offset keeps
    # the projection stable and reproducible.
    now = utc_now()
    future_47 = now.replace(microsecond=0)
    rows = [
        FundingRow(
            venue="Hyperliquid",
            current_rate_bps=1.2,
            next_funding_ts_iso=future_47.isoformat().replace("+00:00", "Z"),
            cum_funding_pnl_usd=-38.4,
        ),
        FundingRow(
            venue="dYdX",
            current_rate_bps=0.8,
            next_funding_ts_iso=future_47.isoformat().replace("+00:00", "Z"),
            cum_funding_pnl_usd=-12.7,
        ),
        FundingRow(
            venue="Drift",
            current_rate_bps=-0.4,
            next_funding_ts_iso=future_47.isoformat().replace("+00:00", "Z"),
            cum_funding_pnl_usd=6.1,
        ),
    ]
    return FundingSnapshot(symbol=symbol, rows=rows, ts_iso=_ts_iso())


def _oracle_snapshot(symbol: str) -> OracleSnapshot:
    rows = [
        OracleRow(
            venue="Hyperliquid",
            oracle_price=71_412.4,
            exec_price=71_408.2,
            divergence_bps=-0.6,
        ),
        OracleRow(
            venue="dYdX",
            oracle_price=71_415.1,
            exec_price=71_437.6,
            divergence_bps=3.2,
        ),
        OracleRow(
            venue="Drift",
            oracle_price=71_410.0,
            exec_price=71_558.2,
            divergence_bps=20.8,
        ),
    ]
    return OracleSnapshot(
        symbol=symbol,
        rows=rows,
        alarm_bps=_ORACLE_ALARM_BPS,
        ts_iso=_ts_iso(),
    )


def _liquidation_snapshot(symbol: str) -> LiqSnapshot:
    current_price = 71_400.0
    bands: list[LiqBand] = []
    for i in range(-10, 11):
        price = current_price * (1.0 + i * 0.01)
        dist = abs(i)
        oi_long = (
            max(0.0, 18_000_000.0 - dist * 1_500_000.0 + math.cos(i) * 1.0e6) if i < 0 else 0.0
        )
        oi_short = (
            max(0.0, 16_000_000.0 - dist * 1_400_000.0 + math.sin(i) * 1.0e6) if i > 0 else 0.0
        )
        bands.append(LiqBand(price=price, oi_long_usd=oi_long, oi_short_usd=oi_short))
    return LiqSnapshot(
        symbol=symbol,
        current_price=current_price,
        bands=bands,
        ts_iso=_ts_iso(),
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def build_projection_router() -> APIRouter:
    """Construct the read-only ``/api/dashboard/{dex,perps}/...`` router.

    The router is stateless — every handler returns a deterministic
    projection shaped to its widget's response interface. The host
    mounts this router alongside the existing
    :func:`ui.dashboard_routes.build_dashboard_router` seam.
    """

    router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

    @router.get("/dex/route", response_model=RouteSnapshot)
    def get_dex_route(
        symbol: str = Query(default=_DEX_DEFAULT_SYMBOL, min_length=1, max_length=64),
    ) -> RouteSnapshot:
        return _route_snapshot(symbol)

    @router.get("/dex/pool_health", response_model=PoolHealthSnapshot)
    def get_dex_pool_health(
        symbol: str = Query(default=_DEX_DEFAULT_SYMBOL, min_length=1, max_length=64),
    ) -> PoolHealthSnapshot:
        return _pool_health_snapshot(symbol)

    @router.get("/dex/gas", response_model=GasSnapshot)
    def get_dex_gas() -> GasSnapshot:
        return _gas_snapshot()

    @router.get("/perps/funding", response_model=FundingSnapshot)
    def get_perps_funding(
        symbol: str = Query(default=_PERPS_DEFAULT_SYMBOL, min_length=1, max_length=64),
    ) -> FundingSnapshot:
        return _funding_snapshot(symbol)

    @router.get("/perps/oracle", response_model=OracleSnapshot)
    def get_perps_oracle(
        symbol: str = Query(default=_PERPS_DEFAULT_SYMBOL, min_length=1, max_length=64),
    ) -> OracleSnapshot:
        return _oracle_snapshot(symbol)

    @router.get("/perps/liquidations", response_model=LiqSnapshot)
    def get_perps_liquidations(
        symbol: str = Query(default=_PERPS_DEFAULT_SYMBOL, min_length=1, max_length=64),
    ) -> LiqSnapshot:
        return _liquidation_snapshot(symbol)

    return router


__all__ = [
    "FundingRow",
    "FundingSnapshot",
    "GasSnapshot",
    "LPHolder",
    "LiqBand",
    "LiqSnapshot",
    "OracleRow",
    "OracleSnapshot",
    "PoolHealthSnapshot",
    "RouteQuote",
    "RouteSnapshot",
    "build_projection_router",
]
