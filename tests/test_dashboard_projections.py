"""P1.5 — projection-route contract tests for the six PR #351 widgets.

These tests pin the JSON shape every PR #351 widget expects from
``/api/dashboard/{dex,perps}/...`` so the per-widget status chip flips
from amber ``mock`` to emerald ``live``. Each test mounts only the
projection router (no engine wiring) so the suite is fast and
free of harness state.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui.dashboard_projection_routes import build_projection_router

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(build_projection_router())
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/dashboard/dex/route
# ---------------------------------------------------------------------------


def test_dex_route_default_symbol_shape(client: TestClient) -> None:
    resp = client.get("/api/dashboard/dex/route")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SOL/USDC"
    assert isinstance(body["quotes"], list)
    assert len(body["quotes"]) == 3
    venues = {q["venue"] for q in body["quotes"]}
    assert venues == {"Jupiter Juno", "1inch Fusion+", "CowSwap"}
    for quote in body["quotes"]:
        assert set(quote.keys()) == {
            "venue",
            "in_token",
            "out_token",
            "in_amount",
            "out_amount",
            "price_impact_bps",
            "est_fill_ms",
        }
        assert quote["in_token"] == "SOL"
        assert quote["out_token"] == "USDC"
    assert body["best_venue"] in venues
    assert _ISO_RE.match(body["ts_iso"])


def test_dex_route_best_venue_is_highest_out_amount(client: TestClient) -> None:
    body = client.get("/api/dashboard/dex/route").json()
    best = max(body["quotes"], key=lambda q: q["out_amount"])
    assert body["best_venue"] == best["venue"]


def test_dex_route_custom_symbol_passthrough(client: TestClient) -> None:
    body = client.get("/api/dashboard/dex/route?symbol=ETH%2FUSDC").json()
    assert body["symbol"] == "ETH/USDC"


# ---------------------------------------------------------------------------
# /api/dashboard/dex/pool_health
# ---------------------------------------------------------------------------


def test_dex_pool_health_shape(client: TestClient) -> None:
    resp = client.get("/api/dashboard/dex/pool_health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "symbol",
        "liquidity_usd",
        "volume_24h_usd",
        "lp_count",
        "hhi",
        "top_holders",
        "ts_iso",
    }
    assert body["symbol"] == "SOL/USDC"
    assert body["liquidity_usd"] > 0
    assert body["volume_24h_usd"] > 0
    assert body["lp_count"] >= 1
    assert 0.0 <= body["hhi"] <= 1.0
    assert isinstance(body["top_holders"], list)
    assert all(set(h.keys()) == {"address_short", "pct"} for h in body["top_holders"])
    assert _ISO_RE.match(body["ts_iso"])


def test_dex_pool_health_custom_symbol(client: TestClient) -> None:
    body = client.get("/api/dashboard/dex/pool_health?symbol=ETH%2FUSDC").json()
    assert body["symbol"] == "ETH/USDC"


# ---------------------------------------------------------------------------
# /api/dashboard/dex/gas
# ---------------------------------------------------------------------------


def test_dex_gas_shape(client: TestClient) -> None:
    resp = client.get("/api/dashboard/dex/gas")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "base_fee_lamports",
        "p50_tip_lamports",
        "p75_tip_lamports",
        "p90_tip_lamports",
        "mev_protected_rpc",
        "ts_iso",
    }
    # Percentile monotonicity — widget renders these as p50/p75/p90.
    assert body["p50_tip_lamports"] <= body["p75_tip_lamports"]
    assert body["p75_tip_lamports"] <= body["p90_tip_lamports"]
    assert body["base_fee_lamports"] >= 0
    assert body["mev_protected_rpc"]
    assert _ISO_RE.match(body["ts_iso"])


# ---------------------------------------------------------------------------
# /api/dashboard/perps/funding
# ---------------------------------------------------------------------------


def test_perps_funding_shape(client: TestClient) -> None:
    resp = client.get("/api/dashboard/perps/funding")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTC-PERP"
    assert isinstance(body["rows"], list)
    assert len(body["rows"]) == 3
    venues = {r["venue"] for r in body["rows"]}
    assert venues == {"Hyperliquid", "dYdX", "Drift"}
    for row in body["rows"]:
        assert set(row.keys()) == {
            "venue",
            "current_rate_bps",
            "next_funding_ts_iso",
            "cum_funding_pnl_usd",
        }
        assert _ISO_RE.match(row["next_funding_ts_iso"])
    assert _ISO_RE.match(body["ts_iso"])


def test_perps_funding_custom_symbol(client: TestClient) -> None:
    body = client.get("/api/dashboard/perps/funding?symbol=ETH-PERP").json()
    assert body["symbol"] == "ETH-PERP"


# ---------------------------------------------------------------------------
# /api/dashboard/perps/oracle
# ---------------------------------------------------------------------------


def test_perps_oracle_shape(client: TestClient) -> None:
    resp = client.get("/api/dashboard/perps/oracle")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTC-PERP"
    assert isinstance(body["rows"], list)
    assert len(body["rows"]) == 3
    assert body["alarm_bps"] == 25
    for row in body["rows"]:
        assert set(row.keys()) == {
            "venue",
            "oracle_price",
            "exec_price",
            "divergence_bps",
        }
    assert _ISO_RE.match(body["ts_iso"])


def test_perps_oracle_custom_symbol(client: TestClient) -> None:
    body = client.get("/api/dashboard/perps/oracle?symbol=SOL-PERP").json()
    assert body["symbol"] == "SOL-PERP"


# ---------------------------------------------------------------------------
# /api/dashboard/perps/liquidations
# ---------------------------------------------------------------------------


def test_perps_liquidations_shape(client: TestClient) -> None:
    resp = client.get("/api/dashboard/perps/liquidations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTC-PERP"
    assert body["current_price"] > 0
    assert isinstance(body["bands"], list)
    # 21 bands at i ∈ [-10, 10] inclusive (widget renders these
    # symmetrically around ``current_price``).
    assert len(body["bands"]) == 21
    for band in body["bands"]:
        assert set(band.keys()) == {"price", "oi_long_usd", "oi_short_usd"}
        assert band["oi_long_usd"] >= 0
        assert band["oi_short_usd"] >= 0
    assert _ISO_RE.match(body["ts_iso"])


def test_perps_liquidations_long_short_segregation(client: TestClient) -> None:
    """Bands below current price carry long OI; bands above carry short OI."""

    body = client.get("/api/dashboard/perps/liquidations").json()
    current = body["current_price"]
    for band in body["bands"]:
        if band["price"] < current:
            assert band["oi_short_usd"] == 0.0
        elif band["price"] > current:
            assert band["oi_long_usd"] == 0.0


def test_perps_liquidations_custom_symbol(client: TestClient) -> None:
    body = client.get("/api/dashboard/perps/liquidations?symbol=ETH-PERP").json()
    assert body["symbol"] == "ETH-PERP"


# ---------------------------------------------------------------------------
# HarnessRouteRegistrar inventory pin
# ---------------------------------------------------------------------------


def test_dashboard_inventory_includes_projection_routes() -> None:
    """The new projection routes are listed in the canonical inventory.

    Pins the (METHOD, PATH) tuples HarnessRouteRegistrar audits at boot
    so a future drift on the projection-router prefix is caught by the
    fail-closed audit rather than at runtime by a widget 404.
    """

    from ui.harness.route_registrar import _DASHBOARD_ROUTES

    expected: set[tuple[str, str]] = {
        ("GET", "/api/dashboard/dex/route"),
        ("GET", "/api/dashboard/dex/pool_health"),
        ("GET", "/api/dashboard/dex/gas"),
        ("GET", "/api/dashboard/perps/funding"),
        ("GET", "/api/dashboard/perps/oracle"),
        ("GET", "/api/dashboard/perps/liquidations"),
    }
    assert expected.issubset(_DASHBOARD_ROUTES)
