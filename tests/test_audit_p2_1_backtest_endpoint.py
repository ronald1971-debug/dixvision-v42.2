"""AUDIT-P2.1 — regression for ``POST /api/testing/backtest``.

Pins:
* the endpoint exists and accepts the form payload the dashboard
  ``Backtester`` widget sends;
* the response carries a stable ``seed`` keyed off the request fields,
  so the same payload returns byte-identical equity / drawdown / trades
  every time;
* the response shape matches the TypeScript ``BacktestRunResponse`` the
  widget consumes;
* malformed payloads produce a 400 (Pydantic validation), not a 500.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv(
        "DIXVISION_LEDGER_PATH", str(tmp_path / "governance.db")
    )
    monkeypatch.setenv(
        "DIXVISION_INTENT_HMAC_KEY", "test-hmac-key-audit-p2-1"
    )
    if "ui.server" in list(os.sys.modules):
        del os.sys.modules["ui.server"]
    from ui.server import app

    return TestClient(app)


def _payload() -> dict[str, object]:
    return {
        "strategy": "ema_cross_20_50",
        "symbol": "BTC/USDT",
        "start_iso": "2025-01-01T00:00:00Z",
        "end_iso": "2025-04-01T00:00:00Z",
        "fill_model": "next_tick",
        "slippage_bps": 8.0,
    }


def test_backtest_endpoint_returns_canonical_shape(client: TestClient) -> None:
    res = client.post("/api/testing/backtest", json=_payload())
    assert res.status_code == 200, res.text
    body = res.json()

    assert isinstance(body["seed"], str) and len(body["seed"]) >= 8
    assert body["request"]["strategy"] == "ema_cross_20_50"
    assert body["request"]["symbol"] == "BTC/USDT"

    assert isinstance(body["equity"], list) and len(body["equity"]) == 241
    assert isinstance(body["drawdown"], list) and len(body["drawdown"]) == 241
    assert isinstance(body["trades"], list) and len(body["trades"]) > 0

    metrics = body["metrics"]
    for key in (
        "final_equity_pct",
        "cagr",
        "sharpe",
        "sortino",
        "max_dd_pct",
        "win_rate",
        "avg_trade_pct",
        "longest_loss_streak",
        "n_trades",
    ):
        assert key in metrics, f"missing metric: {key}"

    assert "deterministic-internal" in body["notes"]
    assert "no-execution-authority" in body["notes"]


def test_backtest_endpoint_is_replay_deterministic(client: TestClient) -> None:
    a = client.post("/api/testing/backtest", json=_payload()).json()
    b = client.post("/api/testing/backtest", json=_payload()).json()
    assert a["seed"] == b["seed"]
    assert a["equity"] == b["equity"]
    assert a["drawdown"] == b["drawdown"]
    assert a["trades"] == b["trades"]
    assert a["metrics"] == b["metrics"]


def test_backtest_endpoint_seed_changes_on_param_change(
    client: TestClient,
) -> None:
    a = client.post("/api/testing/backtest", json=_payload()).json()
    payload = _payload()
    payload["slippage_bps"] = 12.5
    b = client.post("/api/testing/backtest", json=payload).json()
    assert a["seed"] != b["seed"], (
        "Seed must depend on slippage_bps so two operators with different "
        "params get different reports"
    )


def test_backtest_endpoint_rejects_bad_iso_with_400(
    client: TestClient,
) -> None:
    payload = _payload()
    payload["start_iso"] = "not-an-iso-date"
    res = client.post("/api/testing/backtest", json=payload)
    assert res.status_code == 400, res.text


def test_backtest_endpoint_rejects_negative_slippage_with_422(
    client: TestClient,
) -> None:
    payload = _payload()
    payload["slippage_bps"] = -1.0
    res = client.post("/api/testing/backtest", json=payload)
    assert res.status_code == 422, res.text


def test_seed_repr_matches_javascript_string_for_whole_floats() -> None:
    """``_seed_repr`` must collapse whole-number floats to their integer
    form so the FNV-1a seed agrees with the legacy browser fallback in
    ``dashboard2026/src/widgets/testing/Backtester.tsx``.

    JS: ``String(8.0) === "8"`` — Python: ``str(8.0) == "8.0"``. Without
    normalisation an integer ``slippage_bps`` produced two different
    seeds on the same payload (server vs. browser fallback), breaking
    the bit-for-bit replay claim.
    """

    from system_engine.backtest_ingest.internal.deterministic import (
        _seed_repr,
    )

    assert _seed_repr(8.0) == "8"
    assert _seed_repr(0.0) == "0"
    assert _seed_repr(-3.0) == "-3"
    assert _seed_repr(8) == "8"
    assert _seed_repr(8.5) == "8.5"
    assert _seed_repr("ema_cross_20_50") == "ema_cross_20_50"
    assert _seed_repr(True) == "true"
    assert _seed_repr(False) == "false"


def test_backtest_seed_stable_when_int_passed_as_float() -> None:
    """The whole-number float ``slippage_bps=8.0`` must hash to the same
    seed as the int ``slippage_bps=8`` (after Pydantic coercion both
    arrive as ``float``). This pins the JS-parity normalisation in
    ``_seed_repr`` against accidental regression.
    """

    from system_engine.backtest_ingest.internal.deterministic import (
        _fnv1a,
    )

    parts_int: tuple[str | int | float, ...] = (
        "ema_cross_20_50",
        "BTC/USDT",
        "2025-01-01T00:00:00Z",
        "2025-04-01T00:00:00Z",
        "next_tick",
        8,
    )
    parts_float: tuple[str | int | float, ...] = (
        "ema_cross_20_50",
        "BTC/USDT",
        "2025-01-01T00:00:00Z",
        "2025-04-01T00:00:00Z",
        "next_tick",
        8.0,
    )
    assert _fnv1a(parts_int) == _fnv1a(parts_float)
